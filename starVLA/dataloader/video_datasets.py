# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Pure-video (action-less) dataset for generation-loss co-training.

Motivation
----------
Robot data with actions is scarce and hard to scale; raw video is abundant.  The
StarVLA-PI generation loss is computed from images ONLY (current frame + future
frame(s)); it never touches the action.  So we can pour cheap video into the gen
objective while the (scarce) robot data still drives BOTH the action loss and the
gen loss.

Unified sample contract
-----------------------
This dataset emits the SAME "shared" gen keys the robot generation dataset
(``gen_lerobot_datasets``) produces, so the two sources feed ONE gen-loss head:

    image             : List[PIL.Image]  (the observation frame(s) at a random start)
    lang              : str              (a text description of the clip)
    gen_target_frames : List[PIL.Image]  (the T future frame(s))
    gen_target_image  : PIL.Image        (the last future frame; single-frame compat)

It carries **no** ``action`` / ``state`` key, so a batch of these samples is
consumable only by ``model.forward(examples, loss_keys={"gen_loss"})`` — the
gen-only path — never by the action loss.

Video source
------------
An ``index.jsonl`` with one record per line, in one of two formats (an index must not
mix them). Frame-index records describe a clip file, e.g. a pre-cut Ego4D clip:

    {"video": "000001.mp4", "total_frames": 982, "fps": 30.0, "width": 1440,
     "height": 1080, "source_video": "...", "start_time": 1069.3,
     "end_time": 1099.3, "step_description": "preheat pots",
     "step_category": "...", "goal_category": "...", "split": "train"}

Timestamp-window records describe a legal interval of a (longer) source video; the
observation and the future target must both stay inside it:

    {"video": "/data/source.mp4", "valid_start_s": 1069.3,
     "valid_end_s": 1074.8, "sampling_weight_s": 5.1,
     "step_description": "grasp a bolt", "split": "train"}

``start_ts`` / ``end_ts`` are accepted as aliases of ``valid_start_s`` / ``valid_end_s``.
Relative ``video`` paths are resolved against ``data_root_dir``.

Each record is one "trajectory".  Mirroring the robot dataset, every ``__getitem__``
draws a random record + a random legal start. Frame-index records are sampled
uniformly by frame index. Timestamp-window records are drawn in proportion to ``sampling_weight_s``
and decoded by timestamp, keeping both the observation and future target inside
``[valid_start_s, valid_end_s]``. The temporal offsets come from the SHARED
``gen_target_sampling`` helper the robot gen dataset uses, so both sources emit an
identically-shaped target (same ``T``) — required because the gen head's
``num_gen_queries`` is resolved once from these knobs.

Config (``cfg.datasets.video_data``)
------------------------------------
    dataset_py        : "video_datasets"
    losses            : [gen_loss]                (consumed by the cotrain trainer)
    index_jsonl       : path to the index.jsonl
    data_root_dir     : dir holding the mp4 files (default: dirname(index_jsonl))
    per_device_batch_size : int
    split             : "train" | null            (filter records by the split field)
    text_key          : "step_description"        (which record field -> lang; "" -> no text)
    obs_resize_mode   : "fixed" | "native"        (fixed: cover+crop obs AND gen-target to obs_image_size;
                                                   native: keep decoded size, bounded by Qwen max_pixels)
    obs_image_size    : [224, 224]                (the ONE fixed-mode resize target for obs + gen-target)
    gen_future_offset : (derived)                  (frames ahead; from framework.gen_model.gen_horizon_seconds)
    gen_stride_frames : (derived)                  (multi-frame grid; from framework.gen_model.num_future_frames)
    num_target_frames : (derived)                  (T future frames; the same on every stream)
    dataset_fps       : (required for gen)          (this stream's recorded fps; converts seconds to frames)
    view_tiling       : "off" | "tile"             (tile mode: composite views into one canvas)
    view_role         : "central_global"           (this single view's canonical role)
    view_description  : "..."                        (natural-language phrase for the prompt)
    gen_video_key     : role | "tile"              (gen target; default = view_role)
    tile_cell_size    : null | [W, H]              (2x2 cell size; canvas = 2W x 2H)
    video_backend     : "pyav"
    video_backend_kwargs : {}                       (backend constructor options; e.g. Decord num_threads)
"""

import json
import logging
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import Dataset

from starVLA.dataloader.gen_frame_transform import resolve_gen_frame_aug
from starVLA.dataloader.gen_target_sampling import gen_target_indices
from starVLA.dataloader.gr00t_lerobot.datasets import safe_hash
from starVLA.dataloader.gr00t_lerobot.video import get_frames_by_indices, get_frames_by_timestamps
from starVLA.dataloader.shared_epoch import SharedEpoch
from starVLA.dataloader.view_tiling import (
    ANCHOR_ROLE,
    build_view_prompt_contract,
    layout_text,
    normalize_obs_resize_mode,
    normalize_view_prompt_mode,
    normalize_view_role,
    normalize_view_tiling_mode,
    resize_cover_crop,
    resolve_image_size,
    tile_roles,
)

logger = logging.getLogger(__name__)


def collate_fn(batch):
    """Identity collate: keep the list-of-dict sample format StarVLA expects."""
    return batch


def _cfg_get(data_cfg, key, default=None):
    if data_cfg is None:
        return default
    if hasattr(data_cfg, "get"):
        return data_cfg.get(key, default)
    return getattr(data_cfg, key, default)


class VideoClipDataset(Dataset):
    """Action-less video dataset yielding the shared generation-target keys.

    Sampling mirrors ``LeRobotMixtureDataset.sample_step``: a deterministic-per
    -(epoch, index) RNG draws a random clip and a random start frame, so shuffling
    is internal (the DataLoader is built without a shuffling sampler, exactly like
    the robot path).
    """

    def __init__(self, data_cfg, seed: int = 42):
        self.data_cfg = data_cfg
        self.seed = int(_cfg_get(data_cfg, "seed", seed))
        self.epoch = 0
        self._shared_epoch = SharedEpoch(0)
        self._persistent_worker_epoch = bool(_cfg_get(data_cfg, "persistent_workers", True))
        # Last successfully-built sample (per worker process). Fallback when a
        # __getitem__ exhausts its retries mid-run, so a transient bad clip can't
        # kill the DataLoader worker — a worker death raises in the main process,
        # crashing that rank, after which the other ranks hang forever at the next
        # NCCL collective (GPU util -> 0).
        self._last_good = None

        index_jsonl = _cfg_get(data_cfg, "index_jsonl", None)
        if index_jsonl is None:
            raise ValueError("datasets.video_data.index_jsonl is required for dataset_py='video_datasets'.")
        self.index_jsonl = Path(index_jsonl)
        data_root_dir = _cfg_get(data_cfg, "data_root_dir", None)
        self.data_root_dir = Path(data_root_dir) if data_root_dir else self.index_jsonl.parent

        self.split = _cfg_get(data_cfg, "split", "train")
        # Optional shared photometric augmentation (train split only; None -> identity split).
        # Same as gen_lerobot: obs is jittered when enabled; the target follows gen_target_clean.
        self._gen_frame_aug = resolve_gen_frame_aug(data_cfg, "train" if self.split in (None, "train") else "val")
        self.text_key = _cfg_get(data_cfg, "text_key", "step_description")
        self.view_prompt_mode = normalize_view_prompt_mode(_cfg_get(data_cfg, "view_prompt_mode", None))
        if self.view_prompt_mode != "role":
            raise ValueError(
                "video_datasets emits semantic view_texts and requires " "datasets.shared.view_prompt_mode=role."
            )
        # obs_image_size is the ONE fixed-mode resize target for obs AND gen-target frames
        # (default 224x224), reached by aspect-preserving cover+center-crop. ``native`` mode
        # ignores it and keeps the decoded frame (bounded downstream by Qwen max_pixels /
        # the gen encoder). Same policy as the robot streams.
        self.obs_resize_mode = normalize_obs_resize_mode(_cfg_get(data_cfg, "obs_resize_mode", None))
        self.obs_image_size = _cfg_get(data_cfg, "obs_image_size", None)
        self.view_tiling = normalize_view_tiling_mode(_cfg_get(data_cfg, "view_tiling", None))
        self.tile_cell_size = _cfg_get(data_cfg, "tile_cell_size", None)
        # Pure-video is single-view: its one view IS the anchor (central_global). A prompt
        # description follows the dataset (default a generic ego phrase). gen_video_key may
        # only be this role or 'tile' (a 2x2 with only the anchor cell filled).
        self.view_role = normalize_view_role(_cfg_get(data_cfg, "view_role", ANCHOR_ROLE))
        self.view_description = str(
            _cfg_get(data_cfg, "view_description", None) or "primary global view from a moving human head camera"
        )
        prompt_contract = build_view_prompt_contract(
            {self.view_role: self.view_description},
            self.view_tiling,
        )
        existing_prompt_contract = _cfg_get(
            data_cfg,
            "resolved_view_prompt",
            None,
        )
        if existing_prompt_contract is not None:
            existing_prompt_contract = (
                OmegaConf.to_container(existing_prompt_contract, resolve=True)
                if OmegaConf.is_config(existing_prompt_contract)
                else existing_prompt_contract
            )
            if existing_prompt_contract != prompt_contract:
                raise ValueError(
                    "video_datasets resolved a view prompt contract that "
                    "disagrees with the stream config: "
                    f"existing={existing_prompt_contract}, "
                    f"new={prompt_contract}."
                )
        else:
            data_cfg["resolved_view_prompt"] = prompt_contract
        self.gen_video_key = _cfg_get(data_cfg, "gen_video_key", None)
        self.gen_future_offset = int(_cfg_get(data_cfg, "gen_future_offset", 8))
        # gen_future_offset / gen_stride_frames / num_target_frames are derived per stream by
        # align_dataset_configs from the horizon in seconds and this stream's dataset_fps
        # (0.0 / 1 -> one future frame at the horizon).
        self.gen_stride_frames = float(_cfg_get(data_cfg, "gen_stride_frames", 0.0))
        self.num_target_frames = int(_cfg_get(data_cfg, "num_target_frames", 1))
        self.dataset_fps = float(_cfg_get(data_cfg, "dataset_fps", 0.0))
        self.target_frame_offsets = gen_target_indices(
            0,
            self.gen_future_offset,
            self.gen_stride_frames,
            self.num_target_frames,
        )
        self.max_target_frame_offset = max(self.target_frame_offsets)
        self.video_backend = _cfg_get(data_cfg, "video_backend", "pyav")
        backend_kwargs = _cfg_get(data_cfg, "video_backend_kwargs", {})
        if OmegaConf.is_config(backend_kwargs):
            backend_kwargs = OmegaConf.to_container(backend_kwargs, resolve=True)
        if backend_kwargs is None:
            backend_kwargs = {}
        if not isinstance(backend_kwargs, dict):
            raise TypeError("datasets.video_data.video_backend_kwargs must be a mapping")
        self.video_backend_kwargs = dict(backend_kwargs)
        # Epoch length. Sampling is random per index, so this is just how many samples
        # constitute one "epoch" — NOT which clips are reachable (all are). It exists to
        # BOUND DataLoader-worker lifetime: PyAV's VP9 decode leaks ~1MB per open that
        # no close()/malloc_trim reclaims, so with persistent_workers=False a smaller
        # epoch recycles the workers periodically, reclaiming the leaked host RAM. None
        # (default) -> one epoch == the full clip list (no early recycling).
        self.samples_per_epoch = _cfg_get(data_cfg, "samples_per_epoch", None)
        # Optional cap on the number of reachable clips (episodes). Set by the
        # co-training trainer from the vla episode_fraction (resolve_video_episode_cap);
        # None -> use every clip. Truncates self.records below,
        # making excluded clips physically unreachable regardless of samples_per_epoch.
        self.max_episodes = _cfg_get(data_cfg, "max_episodes", None)
        # Optional fractional pool budget in (0, 1]: keep round(fraction * N) clips.
        # This is the per-stream, ratio-based twin of the robot streams' episode_fraction
        # (independent of the batch sampling_ratio). It shares max_episodes' fixed-seed
        # prefix logic below, so the same fraction always samples the same clips and
        # smaller fractions are nested subsets of larger ones. When both are set the
        # fraction wins. None -> use every clip.
        self.episode_fraction = _cfg_get(data_cfg, "episode_fraction", None)

        self.records = self._load_index()
        if not self.records:
            raise ValueError(
                f"No usable clips in {self.index_jsonl} (split={self.split!r}). " "Check the index path / split filter."
            )
        self._apply_episode_cap()
        self._prepare_record_sampling()
        logger.info(
            f"[video_datasets] Loaded {len(self.records)} records from {self.index_jsonl} "
            f"(split={self.split}, text_key={self.text_key}, gen_future_offset={self.gen_future_offset}, "
            f"num_target_frames={self.num_target_frames}, gen_stride_frames={self.gen_stride_frames}, "
            f"sampling_mode={'timestamp-weighted' if self.timestamp_sampling else 'frame-uniform'})"
        )

        # Warm a fallback sample ONCE, here in the main process, so every DataLoader
        # worker inherits it (fork copies this object; spawn pickles it). Without this,
        # a worker that fails to decode its very FIRST sample has no _last_good to fall
        # back on and would raise -> kill the worker -> crash its rank -> the other
        # ranks deadlock at the next collective. Workers recycle every epoch
        # (persistent_workers=False), so each fresh cohort would otherwise re-hit that
        # first-sample window; inheriting a warm fallback closes it. Guarded: an unlucky
        # first clip just leaves it None (we then rely on the first successful getitem).
        try:
            self._last_good = self._build_sample(np.random.default_rng(self.seed))
        except Exception as exc:  # decode errors vary by backend
            logger.warning(
                f"[video_datasets] startup fallback decode failed ({exc}); "
                "relying on the first successful __getitem__ instead."
            )

    def _load_index(self) -> list[dict]:
        records = []
        incomplete = 0
        timestamp_records = 0
        frame_records = 0
        target_offsets_s = self._target_offsets_seconds(require=False)
        with open(self.index_jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if self.split is not None and rec.get("split", None) not in (None, self.split):
                    continue
                # Accept start_ts/end_ts as aliases for valid_start_s/valid_end_s so a
                # seconds-windowed index (legal interval given as start_ts/end_ts of a
                # longer source video) is read via the timestamp path. The aliases are only
                # used when the canonical keys are absent.
                if "valid_start_s" not in rec and "start_ts" in rec and "end_ts" in rec:
                    rec["valid_start_s"] = rec["start_ts"]
                    rec["valid_end_s"] = rec["end_ts"]
                has_start = "valid_start_s" in rec
                has_end = "valid_end_s" in rec
                if has_start != has_end:
                    record_id = rec.get("clip_id", rec.get("video"))
                    raise ValueError(f"Timestamp record must define both valid_start_s and valid_end_s: {record_id}")
                if has_start:
                    if target_offsets_s is None:
                        raise ValueError(
                            "datasets.video_data.dataset_fps must be positive for timestamp-window records."
                        )
                    start_s = float(rec["valid_start_s"])
                    end_s = float(rec["valid_end_s"])
                    sampleable_s = end_s - start_s - target_offsets_s[-1]
                    if not np.isfinite(start_s) or not np.isfinite(end_s) or sampleable_s <= 0:
                        incomplete += 1
                        continue
                    weight_s = float(rec.get("sampling_weight_s", sampleable_s))
                    if not np.isfinite(weight_s) or weight_s <= 0:
                        raise ValueError(f"sampling_weight_s must be finite and positive, got {weight_s!r}")
                    if weight_s > sampleable_s + 1e-5:
                        raise ValueError(
                            f"sampling_weight_s={weight_s:.6f} exceeds the legal sampleable interval "
                            f"{sampleable_s:.6f}s for {rec.get('clip_id', rec.get('video'))}"
                        )
                    rec["_sampling_weight_s"] = weight_s
                    timestamp_records += 1
                else:
                    if int(rec.get("total_frames", 0)) <= self.max_target_frame_offset:
                        incomplete += 1
                        continue
                    frame_records += 1
                records.append(rec)
        if timestamp_records and frame_records:
            raise ValueError("An index_jsonl cannot mix timestamp-window records with frame-index records.")
        self.timestamp_sampling = timestamp_records > 0
        if incomplete:
            logger.warning(
                f"[video_datasets] Filtered {incomplete} records that cannot cover "
                f"the complete strict-future horizon without leaving their legal interval."
            )
        return records

    def _target_offsets_seconds(self, require: bool = True) -> list[float] | None:
        """Convert the aligned video-stream frame offsets to physical seconds."""
        if self.dataset_fps <= 0:
            if require:
                raise ValueError("datasets.video_data.dataset_fps must be positive for timestamp decoding.")
            return None
        return [frame_offset / self.dataset_fps for frame_offset in self.target_frame_offsets]

    def _prepare_record_sampling(self) -> None:
        """Precompute an O(log N) duration-weighted record sampler."""
        if not self.timestamp_sampling:
            self._record_sampling_cdf = None
            self._record_sampling_total = None
            return
        weights = np.asarray([rec["_sampling_weight_s"] for rec in self.records], dtype=np.float64)
        self._record_sampling_cdf = np.cumsum(weights)
        self._record_sampling_total = float(self._record_sampling_cdf[-1])
        if not np.isfinite(self._record_sampling_total) or self._record_sampling_total <= 0:
            raise ValueError("Timestamp record sampling weights must have a finite positive sum.")

    def set_epoch(self, epoch: int):
        """Change the sampling hash so a new epoch draws different clips/frames."""
        self.epoch = int(epoch)
        self._shared_epoch.set(epoch)

    def _apply_episode_cap(self) -> None:
        """Deterministically keep a subset of clips (episodes).

        Two mutually-compatible knobs feed a single fixed-seed prefix subset:
        ``episode_fraction`` in (0, 1] keeps ``round(fraction * N)`` clips, and the
        integer ``max_episodes`` keeps at most that many. When both are set the
        fraction wins. A fixed-seed shuffle (independent of the ratio) picks the kept
        clips and drops the rest, so excluded clips are UNREACHABLE (``_build_sample``
        only ever indexes into ``self.records``; ``samples_per_epoch`` /
        ``worker_generation_batches_per_rank`` change only ``__len__``, not the
        reachable set) and smaller fractions are nested subsets of larger ones. No-op
        when neither knob is set.
        """
        num_clips = len(self.records)
        if self.episode_fraction is not None:
            fraction = float(self.episode_fraction)
            if not (0.0 < fraction <= 1.0):
                raise ValueError(
                    f"datasets.video_data.episode_fraction must be in (0, 1], got {self.episode_fraction!r}"
                )
            keep_count = max(1, round(fraction * num_clips))
            budget_desc = f"episode_fraction={fraction}"
        elif self.max_episodes is not None:
            keep_count = min(max(1, int(self.max_episodes)), num_clips)
            budget_desc = f"max_episodes={self.max_episodes}"
        else:
            return
        keep_count = min(keep_count, num_clips)
        if keep_count >= num_clips:
            return
        rng = np.random.default_rng(safe_hash((self.seed, "video_episode_cap")))
        keep = sorted(rng.permutation(num_clips)[:keep_count].tolist())
        self.records = [self.records[i] for i in keep]
        logger.info(
            f"[episode-fraction] video: keep {keep_count}/{num_clips} clips "
            f"({budget_desc}, seed={self.seed}); excluded clips are unreachable."
        )

    def __len__(self) -> int:
        if self.samples_per_epoch:
            return int(self.samples_per_epoch)
        return len(self.records)

    def _pick_base(self, rng, total: int) -> int:
        """Random start frame with room for the complete strict-future window."""
        return int(rng.integers(0, total - self.max_target_frame_offset))

    def _pick_record_index(self, rng) -> int:
        if self._record_sampling_cdf is None:
            return int(rng.integers(len(self.records)))
        draw = float(rng.random()) * self._record_sampling_total
        return min(int(np.searchsorted(self._record_sampling_cdf, draw, side="right")), len(self.records) - 1)

    def _pick_base_timestamp(self, rng, rec: dict, max_target_offset_s: float) -> float:
        start_s = float(rec["valid_start_s"])
        latest_start_s = float(rec["valid_end_s"]) - max_target_offset_s
        return start_s + float(rng.random()) * (latest_start_s - start_s)

    def _decode(self, video_path: Path, indices: list[int]) -> np.ndarray:
        return get_frames_by_indices(
            str(video_path),
            indices,
            video_backend=self.video_backend,
            video_backend_kwargs=self.video_backend_kwargs,
        )

    def _decode_timestamps(self, video_path: Path, timestamps_s: list[float]) -> np.ndarray:
        return get_frames_by_timestamps(
            str(video_path),
            timestamps_s,
            video_backend=self.video_backend,
            video_backend_kwargs=self.video_backend_kwargs,
        )

    def _resolve_gen_video_key(self) -> str:
        """``gen_video_key`` -> a canonical role or ``'tile'``. Default: this stream's view_role."""
        if self.gen_video_key is None:
            return self.view_role
        text = str(self.gen_video_key).strip().lower()
        return "tile" if text == "tile" else normalize_view_role(text)

    def _build_sample(self, rng) -> dict:
        rec = self.records[self._pick_record_index(rng)]
        video_path = self.data_root_dir / rec["video"]
        if self.timestamp_sampling:
            target_offsets_s = self._target_offsets_seconds()
            base_s = self._pick_base_timestamp(rng, rec, target_offsets_s[-1])
            timestamps_s = [base_s, *(base_s + offset_s for offset_s in target_offsets_s)]
            frames = self._decode_timestamps(video_path, timestamps_s)
        else:
            total = int(rec["total_frames"])
            base = self._pick_base(rng, total)
            target_idx = [base + frame_offset for frame_offset in self.target_frame_offsets]
            frames = self._decode(video_path, [base, *target_idx])

        # Pure-video is single-view; its one view IS self.view_role (the anchor). No drop.
        obs_size = resolve_image_size(self.obs_image_size, (224, 224)) if self.obs_resize_mode == "fixed" else None
        cell_size = resolve_image_size(self.tile_cell_size, obs_size or (256, 256))
        gen_key = self._resolve_gen_video_key()

        current = Image.fromarray(frames[0])
        future = [Image.fromarray(f) for f in frames[1:]]
        # Optional shared augmentation (train only; None -> identity). obs is jittered when enabled;
        # the target follows gen_target_clean.
        aug = getattr(self, "_gen_frame_aug", None)
        if aug is not None:
            obs_current, future = aug.process([current, *future])
        else:
            obs_current = current

        # ── VLM observation (governed by view_tiling) ──
        # view_texts is aligned 1:1 with the one obs image (off: this view's description;
        # tile: the layout paragraph for the composited canvas).
        if self.view_tiling == "tile":
            obs = tile_roles({self.view_role: obs_current}, cell_size)
            view_texts = [layout_text({self.view_role: self.view_description})]
        else:
            obs = resize_cover_crop(obs_current, obs_size) if obs_size is not None else obs_current
            view_texts = [self.view_description]

        # ── gen target (governed by gen_video_key, INDEPENDENT of view_tiling) ──
        if gen_key == "tile":
            target_frames = [tile_roles({self.view_role: f}, cell_size) for f in future]
        else:
            if gen_key != self.view_role:
                raise ValueError(
                    f"video_datasets is single-view (role {self.view_role!r}); gen_video_key={gen_key!r} "
                    "is unavailable — use that role or 'tile'."
                )
            target_frames = [resize_cover_crop(f, obs_size) if obs_size is not None else f for f in future]

        lang = str(rec.get(self.text_key, "") or "") if self.text_key else ""

        sample = {
            "image": [obs],
            "lang": lang,
            "view_texts": view_texts,
            "gen_target_frames": target_frames,
            "gen_target_image": target_frames[-1],
        }
        return sample

    def __getitem__(self, index: int) -> dict:
        # Retry a few times on decode failures (corrupt/short clip): reseed and
        # resample a different clip, mirroring the robot mixture's retry loop.
        max_retries = 10
        last_exc = None
        epoch = self._shared_epoch.get() if self._persistent_worker_epoch else self.epoch
        for attempt in range(max_retries):
            seed = safe_hash((epoch, index, attempt, self.seed))
            rng = np.random.default_rng(seed)
            try:
                sample = self._build_sample(rng)
                self._last_good = sample  # cache for the transient-failure fallback below
                return sample
            except Exception as exc:  # decode errors vary by backend
                last_exc = exc
                logger.warning(f"[video_datasets] decode failed (index={index}, attempt={attempt}): {exc}")
        # Retries exhausted. Do NOT raise if this worker has ever produced a good
        # sample: a raise kills the worker -> crashes this rank -> the other ranks
        # deadlock at the next collective (GPU util -> 0). A rare duplicate frame is
        # harmless to the (image-only) gen loss, so fall back to the last good sample.
        # Only raise when NOTHING has ever decoded — that means a systemic problem
        # (bad index_jsonl / data_root_dir / split) that should fail loudly at startup
        # rather than silently training on one repeated frame.
        if self._last_good is not None:
            logger.error(
                f"[video_datasets] index={index} failed all {max_retries} retries; reusing the "
                "last good sample to keep the DataLoader worker (and this rank) alive."
            )
            return self._last_good
        raise RuntimeError(
            f"[video_datasets] failed to build ANY sample (index={index}, {max_retries} retries) with "
            "no prior good sample to fall back on — check index_jsonl / data_root_dir / split."
        ) from last_exc


def get_video_dataset(data_cfg, seed: int = 42) -> VideoClipDataset:
    """Build a :class:`VideoClipDataset` from ``cfg.datasets.video_data``."""
    return VideoClipDataset(data_cfg, seed=seed)


if __name__ == "__main__":
    import argparse

    from torch.utils.data import DataLoader
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, required=True)
    parser.add_argument("--num", type=int, default=4)
    args = parser.parse_args()

    # Resolve `_base_` includes and derive the per-stream generation grid exactly as
    # training does, e.g. --config_yaml configs/pretrain/cotrain_openaoe.yaml.
    from starVLA.dataloader import align_dataset_configs
    from starVLA.model.framework.share_tools import apply_config_compat, load_config_with_bases

    cfg = align_dataset_configs(apply_config_compat(load_config_with_bases(args.config_yaml)))
    dataset = get_video_dataset(cfg.datasets.video_data)
    loader = DataLoader(dataset, batch_size=2, num_workers=1, collate_fn=collate_fn)
    for i, batch in enumerate(tqdm(loader)):
        if i >= args.num:
            break
        s = batch[0]
        print({k: type(v).__name__ if k != "lang" else v for k, v in s.items()})
        print(
            "image:",
            s["image"][0].size,
            "gen_target_frames:",
            len(s["gen_target_frames"]),
            "last:",
            s["gen_target_image"].size,
        )
