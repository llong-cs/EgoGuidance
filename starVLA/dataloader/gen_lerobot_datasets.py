# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
#
# Generation co-training dataset:
#   On top of the standard StarVLA LeRobot sample (image / lang / action / state),
#   additionally decode the *future* frame(s) reached after executing the action
#   chunk and attach them as ``gen_target_frames`` / ``gen_target_image``. These
#   frames are the generation-expert targets (see model/modules/gen_model/).
#
# The action-only pipeline (``lerobot_datasets``) does not depend on this file.

import logging
from contextlib import contextmanager
from pathlib import Path

from PIL import Image

from starVLA.dataloader.gen_frame_transform import resolve_gen_frame_aug
from starVLA.dataloader.gen_target_sampling import gen_target_indices
from starVLA.dataloader.gr00t_lerobot.data_config import apply_action_horizon, apply_video_keys
from starVLA.dataloader.gr00t_lerobot.datasets import (
    TRANSITION_PERMUTATION,
    LeRobotMixtureDataset,
    LeRobotSingleDataset,
    normalize_sampling_strategy,
)
from starVLA.dataloader.gr00t_lerobot.registry import (
    DATASET_NAMED_MIXTURES,
    ROBOT_TYPE_CONFIG_MAP,
    EmbodimentTag,
)
from starVLA.dataloader.view_tiling import (
    ANCHOR_ROLE,
    assemble_role_observation,
    configure_robot_view_contract,
    normalize_obs_resize_mode,
    normalize_view_role,
    order_roles,
    resize_cover_crop,
    resolve_image_size,
    sample_dropped_roles,
    tile_roles,
)

logger = logging.getLogger(__name__)


def collate_fn(batch):
    """Identity collate: keep the list-of-dict sample format StarVLA expects."""
    return batch


class LeRobotGenSingleDataset(LeRobotSingleDataset):
    """LeRobot single dataset that also emits the future-frame generation target.

    Injection strategy (works for both direct ``__getitem__`` and the
    ``LeRobotMixtureDataset`` path, which calls ``get_step_data`` /
    ``_pack_sample`` directly and bypasses ``__getitem__``):

      * ``get_step_data`` stashes ``(trajectory_id, base_index)``.
      * ``_pack_sample`` decodes each view's ``[current, *future]`` window in one pass and
        attaches the observation and the shared generation keys.

    Config keys read from ``data_cfg`` (cfg.datasets.vla_data):
      * ``gen_future_offset`` (int): frames ahead of the horizon end point, derived by align_dataset_configs.
      * ``gen_video_key``     (role|'tile'): what the gen head predicts, INDEPENDENT of view_tiling.
                                     A canonical role -> that role's future frame(s);
                                     'tile' -> the 2x2 role future canvas. Default:
                                     central_global (the anchor). Raw camera keys are rejected.
      * ``obs_resize_mode``   ('fixed'|'native'): 'fixed' (default) cover+center-crops obs AND
                                     gen-target frames to obs_image_size; 'native' keeps the
                                     decoded frame (bounded downstream by Qwen max_pixels / the
                                     gen encoder).
      * ``obs_image_size``    ([W, H]): the ONE fixed-mode resize target for obs AND gen-target
                                     frames. Default [224, 224].
      * ``view_drop_prob``    (float): training-only per-view drop probability over the non-anchor
                                     roles ({auxiliary_global, primary_effector, secondary_effector});
                                     central_global and a specific gen-target role are never dropped.
                                     Default 0 (no drop). One drop-set is shared by obs + gen target.
      * ``gen_stride_frames`` / ``num_target_frames``: the multi-frame target grid (``T``
                                       frames, frame k at ``round(k * gen_stride_frames)``),
                                       derived by ``align_dataset_configs`` from
                                       ``framework.gen_model.gen_horizon_seconds`` /
                                       ``num_future_frames``. ``T = 1`` -> one frame at the horizon.
      * ``dataset_fps``       (float): the stream's recorded frames-per-second, used by
                                       ``align_dataset_configs`` to convert the horizon (seconds)
                                       to frames. REQUIRED for generation runs.

    Multi-view (unified view-role system; roles come from DataConfig.video_view_roles):
      * ``view_tiling``       ('off'|'tile'): governs the OBSERVATION only. 'off' -> present,
                                       non-dropped views as N images in canonical role order,
                                       each preceded by its description (sample["view_texts"]).
                                       'tile' -> ONE 2x2 role canvas (missing/dropped cells black)
                                       + a layout paragraph (sample["view_texts"][0]).
      * ``tile_cell_size``    ([W, H]): 2x2 cell size (canvas = 2W x 2H). Default = obs_image_size
                                       (fixed) else [256, 256].

    Output keys attached to every sample:
      * ``view_texts``        (list[str]): aligned 1:1 with ``image`` (per-view description for off;
                                       the single layout paragraph for tile) — interleaved into the prompt.
      * ``gen_target_frames`` (list[PIL.Image]): the T target frames (T>=1).
                                       This is the unified target both the
                                       regression head and the DiT head consume.
      * ``gen_target_image``  (PIL.Image): the LAST target frame (single-frame
                                       regression path).
    """

    def _required_future_offset(self) -> int:
        """Include the generation endpoint in strict future-tail validation."""
        cfg = self.data_cfg if self.data_cfg is not None else {}
        gen_future_offset = int(cfg.get("gen_future_offset", 8))
        if gen_future_offset < 0:
            raise ValueError(f"gen_future_offset must be non-negative for {self.dataset_name}, got {gen_future_offset}.")
        return max(super()._required_future_offset(), gen_future_offset)

    # Injected by _make_gen_single: True once the DataConfig has proven that its video modality
    # read has no observable effect, so it can be skipped outright (see _without_base_video_keys).
    # Conservative class-level default: never skip.
    skip_base_video_read: bool = False

    @contextmanager
    def _without_base_video_keys(self):
        """Hide the video keys from the base dataset for the duration of ONE super() call.

        This class rebuilds every observation frame from its own same-source ``[current,
        *future]`` window decode, which makes the base class's video work pure overhead:
        ``get_step_data`` decodes each view's current frame a SECOND time, and ``_pack_sample``
        turns those frames into an observation list that is overwritten a few lines later.
        Emptying ``modality_keys["video"]`` turns both loops into no-ops without touching the
        base class or duplicating its action/state/language packing.

        Engaged only when ``skip_base_video_read``: a DataConfig that declares an effective video
        modality transform keeps the read so that transform still receives real frames (its
        output is ignored regardless — see ``_warn_inert_video_modality_transforms``).
        """
        if not self.skip_base_video_read:
            yield
            return
        video_keys = self.modality_keys["video"]
        self.modality_keys["video"] = []
        try:
            yield
        finally:
            self.modality_keys["video"] = video_keys

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        # Stash (traj, base) so _pack_sample can fetch the whole [current, *future] window per
        # view in ONE decode via get_video_at_indices. Observation (frame 0) and generation
        # target (frames 1:) are then split from the SAME decode + SAME base transform, so they
        # are strictly same-source (no transformed-obs-vs-raw-future divergence).
        # That window IS the observation source, so the base video read decodes nothing anyone
        # reads; skip it when the DataConfig proves it inert.
        self._gen_last_traj = trajectory_id
        self._gen_last_base = base_index
        with self._without_base_video_keys():
            return super().get_step_data(trajectory_id, base_index)

    def _gen_target_indices(self, future_offset: int) -> list[int]:
        """Absolute frame indices of the generation target.

        T = 1 -> [base + future_offset]; T > 1 -> T frames evenly spaced in time up to it.

        Delegates the temporal math to the shared ``gen_target_sampling`` helper so
        the pure-video dataset (``video_datasets``) samples an identically-shaped
        target from the SAME knobs. ``gen_stride_frames`` / ``num_target_frames`` are
        derived per stream by ``align_dataset_configs``.
        """
        cfg = self.data_cfg if self.data_cfg is not None else {}
        gen_stride_frames = float(cfg.get("gen_stride_frames", 0.0))
        n_frames = int(cfg.get("num_target_frames", 1))
        return gen_target_indices(self._gen_last_base, future_offset, gen_stride_frames, n_frames)

    def _resolve_gen_video_key(self, cfg) -> str:
        """``gen_video_key`` -> a canonical role, or ``'tile'``. Default: the anchor role.

        The gen target is chosen by ROLE (or the 2x2 tile), decoupled from which raw cameras
        a dataset ships AND from the VLM's ``view_tiling``. Raw camera keys are rejected.
        """
        val = cfg.get("gen_video_key", None)
        if val is None:
            return ANCHOR_ROLE
        raw_text = str(val).strip()
        text = raw_text.lower()
        if text == "tile":
            return "tile"
        try:
            return normalize_view_role(text)
        except ValueError as role_error:
            # A raw camera key is accepted when the DataConfig maps that ACTIVE camera
            # to exactly one role; it is translated to that role.
            roles = getattr(self, "view_roles", {}) or {}
            matched_key = next(
                (key for key in roles if str(key).lower() == text),
                None,
            )
            if matched_key is None:
                raise ValueError(
                    f"{self.dataset_name}: gen_video_key={raw_text!r} is neither "
                    "a canonical view role, 'tile', nor a mapped physical camera."
                ) from role_error
            if matched_key not in self.modality_keys["video"]:
                raise ValueError(
                    f"{self.dataset_name}: camera gen_video_key={raw_text!r} maps "
                    f"to role {roles[matched_key]!r}, but that physical camera is "
                    f"not active (active={list(self.modality_keys['video'])})."
                ) from role_error
            role = normalize_view_role(roles[matched_key])
            logger.warning(
                "%s: translating camera gen_video_key=%r to role=%r. " "Set the role in the config instead.",
                self.dataset_name,
                raw_text,
                role,
            )
            return role

    def _role_to_key(self) -> dict:
        """``role -> active video_key`` for the roles this dataset actually ships (anchor required)."""
        roles = getattr(self, "view_roles", {}) or {}
        mapping = {roles[k]: k for k in self.modality_keys["video"] if k in roles}
        if ANCHOR_ROLE not in mapping:
            raise ValueError(
                f"{self.dataset_name}: view-role obs requires the anchor role {ANCHOR_ROLE!r}; resolved "
                f"{mapping} from video_keys={list(self.modality_keys['video'])} (check DataConfig.video_view_roles)."
            )
        return mapping

    def _pack_sample(self, data: dict) -> dict:
        # The base helper yields action/state/lang without trying to assemble an
        # observation. Every frame is rebuilt role-aware from the same-source window below.
        sample = super()._pack_nonvisual_sample(data)
        cfg = self.data_cfg if self.data_cfg is not None else {}
        future_offset = int(cfg.get("gen_future_offset", 8))
        obs_resize_mode = normalize_obs_resize_mode(cfg.get("obs_resize_mode", None))
        # fixed -> cover+crop every image to obs_image_size; native -> keep raw (encoder / Qwen
        # max_pixels bound it downstream). Tiling uses tile_cell_size for its 2x2 cells.
        obs_size = resolve_image_size(cfg.get("obs_image_size"), (224, 224)) if obs_resize_mode == "fixed" else None
        cell_size = resolve_image_size(cfg.get("tile_cell_size"), obs_size or (256, 256))
        gen_key = self._resolve_gen_video_key(cfg)

        role_to_key = self._role_to_key()
        present = order_roles(role_to_key.keys())
        # ONE drop-set, computed BEFORE resize/fork so the observation AND the gen target
        # agree. central_global never drops; a specific gen-target role is also protected.
        protect = () if gen_key == "tile" else (gen_key,)
        dropped = sample_dropped_roles(present, cfg.get("view_drop_prob", 0.0), protect=protect)
        kept = [r for r in present if r not in dropped]
        if gen_key != "tile" and gen_key not in role_to_key:
            raise ValueError(
                f"{self.dataset_name}: gen_video_key role {gen_key!r} is unavailable "
                f"(dataset roles: {list(role_to_key)}). Use an available role or 'tile'."
            )

        # ── SAME-SOURCE fetch + optional shared augmentation ──
        # Decode each kept role's window [current, *future] in ONE pass, then split via the shared
        # GenFrameAug (identity when augmentation is disabled): frame 0 -> observation, frames 1: ->
        # gen target.
        # obs (→VLM) and the gen target (→target encoder) are thus strictly same-source (same decode,
        # same geometry); any photometric augmentation is applied identically per window and the
        # obs/target split follows gen_target_clean. A role whose future is not needed (an obs-only
        # view under a specific-role gen target) decodes just its current frame.
        future_indices = self._gen_target_indices(future_offset)  # T strict-future absolute indices
        future_roles = set(kept) if gen_key == "tile" else {gen_key}
        aug = getattr(self, "_gen_frame_aug", None)
        obs_cur, tgt_future = {}, {}
        for role in kept:
            window = [self._gen_last_base] + (list(future_indices) if role in future_roles else [])
            frames = [
                Image.fromarray(f) for f in self.get_video_at_indices(self._gen_last_traj, role_to_key[role], window)
            ]
            if aug is not None:
                obs_cur[role], tgt_future[role] = aug.process(frames)
            else:
                obs_cur[role], tgt_future[role] = frames[0], list(frames[1:])

        # ── VLM observation (governed by view_tiling) ──
        # Use the exact same physical-camera -> role -> resize/tile -> prompt path
        # as the ordinary action loader. Generation differs only by attaching the
        # future target below.
        observation_frames = {role_to_key[role]: obs_cur[role] for role in kept}
        sample["image"], sample["view_texts"] = assemble_role_observation(
            observation_frames,
            self.view_roles,
            self.view_descriptions,
            cfg,
            context=self.dataset_name,
        )

        # ── gen target (governed by gen_video_key, INDEPENDENT of view_tiling) ──
        # Future frames come from the SAME per-role windows as the observation above.
        if gen_key == "tile":
            # Every kept role is a future_role under a tiled target, so each carries the full T frames.
            assert all(
                len(tgt_future[r]) == len(future_indices) for r in kept
            ), "tiled target: per-role future length mismatch"
            frames = [tile_roles({r: tgt_future[r][k] for r in kept}, cell_size) for k in range(len(future_indices))]
            sample["gen_target_frames"] = frames
            sample["gen_target_image"] = frames[-1]
        else:
            tgt = tgt_future[gen_key]  # future frames (clean, or jittered per gen_target_clean)
            frames = [resize_cover_crop(f, obs_size) if obs_size is not None else f for f in tgt]
            sample["gen_target_frames"] = frames
            sample["gen_target_image"] = frames[-1]
        return sample


def _warn_inert_video_modality_transforms(transforms, data_name: str, robot_type: str) -> bool:
    """Warn that video *modality* transforms are IGNORED by the generation path; report inertness.

    gen_lerobot sources BOTH the observation and the gen target from a single raw
    same-source window decode (get_video_at_indices), so any effective video modality transform
    (VideoCrop / VideoResize / VideoColorJitter / ...) declared in the DataConfig no longer
    affects the gen obs/target — it only runs on the discarded modality read (wasted compute).
    Steer generation augmentation to datasets.<block>.gen_augment (gen_target_clean-aware),
    which applies to obs and target consistently.

    Returns True only when the video modality read is provably inert — no effective transform
    observes it — which lets LeRobotGenSingleDataset skip that read (and its duplicate decode)
    entirely. Detection is best-effort and must never block construction, so anything it cannot
    inspect returns False and keeps the read.
    """
    try:
        from starVLA.dataloader.gr00t_lerobot.transform.video import VideoToNumpy, VideoToTensor, VideoTransform

        items = getattr(transforms, "transforms", None)
        if items is None:
            return False  # unrecognised transform container: assume something observes the read
        effective = [
            type(t).__name__
            for t in items
            if isinstance(t, VideoTransform) and not isinstance(t, (VideoToTensor, VideoToNumpy))
        ]
        if effective:
            logger.warning(
                f"[gen] DataConfig for {data_name}/{robot_type} declares video modality transforms "
                f"{effective}, but gen_lerobot sources obs AND the gen target from a single raw same-source "
                "window decode — these transforms are IGNORED for the generation path (they only run on the "
                "discarded modality read). Put generation augmentation in datasets.<block>.gen_augment "
                "(with gen_target_clean) so it applies to obs and target consistently."
            )
            return False
        return True
    except Exception:  # detection is best-effort; must never block dataset construction
        return False


def _make_gen_single(
    data_root_dir: Path,
    data_name: str,
    robot_type: str,
    delete_pause_frame: bool = False,
    data_cfg: dict | None = None,
    mode: str = "train",
) -> LeRobotGenSingleDataset:
    """Build a single LeRobotGenSingleDataset (mirror of make_LeRobotSingleDataset)."""
    unsupported_tile_keys = [
        key for key in ("tile_layout", "tile_view_keys") if data_cfg is not None and data_cfg.get(key, None) is not None
    ]
    if unsupported_tile_keys:
        raise ValueError(
            f"Unsupported tiling keys {unsupported_tile_keys}: the 2x2 tile layout is defined by view "
            "roles (DataConfig.video_view_roles). Remove these keys and use view_tiling / tile_cell_size."
        )

    data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
    # action_horizon is the single source of truth for the chunk length: when the
    # training entry propagated it into data_cfg, derive action_indices from it.
    data_config = apply_action_horizon(data_config, data_cfg)
    # Optional per-stream obs view selection (e.g. all three DROID cameras) without a subclass.
    data_config = apply_video_keys(data_config, data_cfg)
    modality_config = data_config.modality_config()
    transforms = data_config.transform()
    base_video_read_inert = _warn_inert_video_modality_transforms(transforms, data_name, robot_type)
    dataset_path = data_root_dir / data_name
    embodiment_tag = getattr(data_config, "embodiment_tag", None)
    if embodiment_tag is None:
        embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_backend = data_cfg.get("video_backend", "torchvision_av") if data_cfg else "torchvision_av"
    dataset = LeRobotGenSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend=video_backend,
        delete_pause_frame=delete_pause_frame,
        data_cfg=data_cfg,
    )
    dataset = configure_robot_view_contract(
        dataset,
        data_config,
        data_cfg,
        context=f"{data_name}/{robot_type}",
    )
    # The base video read only fed an observation this class rebuilds from its own same-source
    # window, so drop it once the DataConfig has proven nothing else observes it.
    dataset.skip_base_video_read = base_video_read_inert
    # Optional shared photometric augmentation (train only; disabled by default -> None ->
    # identity split). Same-source geometry is preserved regardless; only obs vs target
    # photometric aug follows gen_target_clean. See gen_frame_transform.GenFrameAug.
    dataset._gen_frame_aug = resolve_gen_frame_aug(data_cfg, mode)
    return dataset


def get_gen_vla_dataset(
    data_cfg: dict,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    seed: int = 42,
    **kwargs,
) -> LeRobotMixtureDataset:
    """Build a mixture of LeRobotGenSingleDataset (mirror of get_vla_dataset).

    The mixture class is the unmodified ``LeRobotMixtureDataset``; it delegates
    to each single's ``_pack_sample``, which we override to attach the future
    frame. So no change to the mixture is required.
    """
    data_root_dir = Path(data_cfg.data_root_dir)
    data_mix = data_cfg.data_mix
    delete_pause_frame = data_cfg.get("delete_pause_frame", False)

    if data_mix not in DATASET_NAMED_MIXTURES:
        raise KeyError(
            f"Unknown data_mix={data_mix!r}; registered mixtures: {sorted(DATASET_NAMED_MIXTURES)}. "
            "Mixtures are declared in examples/<Benchmark>/train_files/data_registry/data_config.py."
        )
    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    logger.info(f"[gen-dataloader] Using mixture '{data_mix}': {[(d, w, r) for d, w, r in mixture_spec]}")
    sampling_strategy = normalize_sampling_strategy(data_cfg.get("sampling_strategy", None))
    if sampling_strategy == TRANSITION_PERMUTATION:
        seed = int(data_cfg.get("seed", seed))

    included, filtered = set(), []
    for d_name, d_weight, robot_type in mixture_spec:
        key = (d_name, robot_type)
        if key in included:
            if sampling_strategy == TRANSITION_PERMUTATION:
                raise ValueError(
                    "transition_permutation refuses a duplicate mixture leaf: "
                    f"data_mix={data_mix!r}, dataset={key!r}."
                )
            continue
        included.add(key)
        filtered.append((d_name, d_weight, robot_type))

    dataset_mixture = [
        (
            _make_gen_single(
                data_root_dir, d_name, robot_type, delete_pause_frame=delete_pause_frame, data_cfg=data_cfg, mode=mode
            ),
            d_weight,
        )
        for d_name, d_weight, robot_type in filtered
    ]

    return LeRobotMixtureDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=seed,
        data_cfg=data_cfg,
        **kwargs,
    )


if __name__ == "__main__":
    # Smoke test (run from the repository root), e.g.
    #   python starVLA/dataloader/gen_lerobot_datasets.py --config_yaml configs/finetune/libero.yaml
    import argparse

    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from starVLA.dataloader import align_dataset_configs
    from starVLA.model.framework.share_tools import apply_config_compat, load_config_with_bases

    parser = argparse.ArgumentParser(description="Smoke-test the LeRobot action + future-frame dataloader.")
    parser.add_argument("--config_yaml", type=str, required=True)
    parser.add_argument("--num", type=int, default=4)
    args = parser.parse_args()

    # Resolve `_base_` includes and derive the per-stream generation grid exactly as training does.
    cfg = align_dataset_configs(apply_config_compat(load_config_with_bases(args.config_yaml)))
    vla_cfg = cfg.datasets.vla_data
    dataset = get_gen_vla_dataset(data_cfg=vla_cfg)

    loader = DataLoader(dataset, batch_size=2, num_workers=1, collate_fn=collate_fn)
    for i, batch in enumerate(tqdm(loader)):
        if i >= args.num:
            break
        s = batch[0]
        print({k: type(v).__name__ if k not in ("action", "state") else v.shape for k, v in s.items()})
        print("gen_target_frames:", len(s["gen_target_frames"]), "last size:", s["gen_target_image"].size)
