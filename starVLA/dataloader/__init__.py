"""Dataloader entry points: config alignment across streams and DataLoader construction.

* :func:`align_dataset_configs` propagates ``datasets.shared`` and derives each stream's
  future-frame offsets from the physical prediction horizon.
* :func:`resolve_stream_batch_sizes` splits a per-device batch budget across streams.
* :func:`build_dataloader` builds one stream (``lerobot_datasets``, ``gen_lerobot_datasets``
  or ``video_datasets``).
"""

import itertools
import json
import math
from numbers import Integral
from pathlib import Path

import numpy as np
import torch.distributed as dist
from accelerate.logging import get_logger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from starVLA.dataloader.preprocessed_batch import QwenPreprocessingCollator, unpack_preprocessed_batch

logger = get_logger(__name__)


# Reserved (non-dataset-block) keys under ``datasets:`` — skipped when iterating
# the per-dataset blocks for alignment/ratio resolution.
_DATASETS_RESERVED = ("shared", "total_per_device_batch_size")


def align_dataset_configs(cfg):
    """Align the data streams of one run.

    * ``framework.action_model.action_horizon`` -> every stream's ``action_horizon``
      (single source of truth for the action chunk length).
    * ``datasets.shared.<key>`` -> every stream (``shared`` wins). ``dataset_fps`` is
      deliberately per stream: streams legitimately run at different frame rates.
    * Generation runs (``framework.gen_model.enable``): the future frame ``o_{t+Delta}`` is
      aligned with the end of the action chunk, ``a_{t+H}``: the horizon is
      ``tau = H / f_robot`` seconds (``H`` = action_horizon, ``f_robot`` =
      ``datasets.vla_data.dataset_fps``; e.g. 8 / 15 Hz = 0.533 s on DROID, 8 / 20 Hz = 0.4 s
      on LIBERO) unless ``framework.gen_model.gen_horizon_seconds`` sets it explicitly. It is
      converted to each stream's frame rate ``f`` as ``floor(tau * f + 1/2)`` frames; with
      ``T = framework.gen_model.num_future_frames`` targets, frame ``k`` lies at ``k * tau / T``
      seconds. Every stream therefore predicts the same physical instants and emits the
      same ``T``. The derived per-stream keys ``gen_future_offset`` / ``gen_stride_frames`` /
      ``num_target_frames`` are written here and must not be set by hand.
    """
    from omegaconf import DictConfig

    datasets = cfg.get("datasets", None) if hasattr(cfg, "get") else getattr(cfg, "datasets", None)
    if datasets is None:
        return cfg

    # ── action_horizon: single source of truth for the action chunk length ──
    action_horizon = OmegaConf.select(cfg, "framework.action_model.action_horizon", default=None)
    if action_horizon is not None:
        # Propagate into EVERY data-stream block, not just vla_data: any stream that
        # loads an action modality (e.g. a lerobot-backed gen-only video stream) must
        # resolve H action slots from the SAME single source of truth
        # (apply_action_horizon), so it can't silently keep its DataConfig's placeholder
        # chunk length. Streams without an action modality (pure-video) just carry an
        # unused key. Runs before the datasets.shared block below, so it also applies to
        # plain single-source action-only configs.
        applied = []
        for name, block in datasets.items():
            if name in _DATASETS_RESERVED or not isinstance(block, DictConfig):
                continue
            block["action_horizon"] = int(action_horizon)
            applied.append(name)
        logger.info(f"[align] framework.action_model.action_horizon={int(action_horizon)} -> {applied}")

    # ── datasets.shared.<key> -> every stream (shared WINS). dataset_fps is EXCLUDED:
    #    it is a per-stream knob (streams run at different fps), declared per dataset and
    #    required for gen runs (see the generation-target block below). ──
    shared = datasets.get("shared", None)
    if shared:
        if "dataset_fps" in shared:
            logger.warning(
                "[align] datasets.shared.dataset_fps is IGNORED — dataset_fps is a PER-STREAM knob "
                "(streams run at different fps). Declare datasets.<stream>.dataset_fps instead."
            )
        for name, block in datasets.items():
            if name in _DATASETS_RESERVED or not isinstance(block, DictConfig):
                continue
            for key in shared:
                if key == "dataset_fps":
                    continue  # per-stream frame rate; never copied onto stream blocks
                # Surface the footgun: a per-block value (e.g. a CLI/experiment override of
                # datasets.vla_data.obs_image_size) that differs from the shared value is about
                # to be overwritten — shared is the single source of truth for aligned keys.
                if key in block and block[key] != shared[key]:
                    logger.warning(
                        f"[align] datasets.{name}.{key}={block[key]!r} is overridden by "
                        f"datasets.shared.{key}={shared[key]!r} (shared wins for aligned params). "
                        f"To change it, set datasets.shared.{key} instead of datasets.{name}.{key}."
                    )
                block[key] = shared[key]  # shared is the single source of truth
        propagated = {k: shared[k] for k in shared if k != "dataset_fps"}
        logger.info(f"[align] datasets.shared -> all streams: {propagated}")

    # ── obs / gen-target preprocessing policy: ONE model-level choice across streams ──
    # These keys decide HOW every stream is observed and WHAT the shared gen expert is
    # asked to generate; they must be identical on every stream or the expert learns a
    # split task (2x2 canvas on one stream, single image on another, or the same target at
    # different resolutions). Unlike dataset_fps (legitimately per-stream) they are unified
    # here — shared wins, else all explicit declarers must agree — before any per-stream
    # normalization / native budget check below.
    _unify_observation_policy(datasets)

    for legacy_key in (
        "framework.gen_model.gen_horizon",
        "datasets.shared.gen_stride",
        "datasets.shared.temporal_reference_fps",
    ):
        if OmegaConf.select(cfg, legacy_key, default=None) is not None:
            raise ValueError(
                f"{legacy_key} is no longer supported: the prediction horizon is set in seconds with "
                "framework.gen_model.gen_horizon_seconds and the number of target frames with "
                "framework.gen_model.num_future_frames."
            )

    # ── generation target: physical horizon tau -> per-stream frame offsets ──
    if bool(OmegaConf.select(cfg, "framework.gen_model.enable", default=False)):
        from starVLA.dataloader.gen_target_sampling import resolve_future_frames

        num_frames = int(OmegaConf.select(cfg, "framework.gen_model.num_future_frames", default=1))
        gen_streams = [
            name for name, block in datasets.items() if name not in _DATASETS_RESERVED and isinstance(block, DictConfig)
        ]
        if "vla_data" not in gen_streams:
            raise ValueError(
                "framework.gen_model.enable is set but there is no datasets.vla_data — "
                "the generation head resolves its target shape from that anchor block."
            )
        horizon_seconds = OmegaConf.select(cfg, "framework.gen_model.gen_horizon_seconds", default=None)
        if horizon_seconds is None:
            # Align o_{t+Delta} with a_{t+H}: the end of the action chunk on the robot clock.
            if action_horizon is None:
                raise ValueError(
                    "framework.action_model.action_horizon is required to align the generation target "
                    "with the end of the action chunk (or set framework.gen_model.gen_horizon_seconds)."
                )
            horizon_seconds = int(action_horizon) / _require_dataset_fps(datasets, "vla_data")
        horizon_seconds = float(horizon_seconds)
        resolved = {}
        for name in gen_streams:
            block = datasets[name]
            grid = resolve_future_frames(horizon_seconds, _require_dataset_fps(datasets, name), num_frames)
            for key in ("gen_future_offset", "gen_stride_frames", "num_target_frames"):
                if key in block and block[key] != grid[key]:
                    logger.warning(
                        f"[align] datasets.{name}.{key}={block[key]!r} is derived from "
                        "framework.gen_model.gen_horizon_seconds / num_future_frames and is overwritten."
                    )
                block[key] = grid[key]
            resolved[name] = {"fps": float(block["dataset_fps"]), "offsets": grid["offsets"]}
        logger.info(
            f"[align] generation target: {num_frames} future frame(s) within {horizon_seconds:g} s -> {resolved}"
        )

    _validate_observation_preprocessing(cfg, datasets)
    return cfg


def _require_dataset_fps(datasets, name) -> float:
    """Return ``datasets.<name>.dataset_fps`` as a float, or raise if it is unset.

    The generation-target alignment depends on the recorded fps. This is enforced
    rather than defaulted so a missing value cannot silently mis-time targets.
    """
    block = datasets.get(name, None)
    fps = block.get("dataset_fps", None) if (block is not None and hasattr(block, "get")) else None
    if fps is None:
        raise ValueError(
            f"datasets.{name}.dataset_fps must be set: the generation horizon "
            "(framework.gen_model.gen_horizon_seconds) is converted to frames with each stream's "
            f"recorded frame rate. Set datasets.{name}.dataset_fps."
        )
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"datasets.{name}.dataset_fps must be finite and positive, got {fps!r}")
    return fps


# Observation / generation-target preprocessing keys that are a MODEL-LEVEL choice, not a
# per-stream one: the shared gen expert must see one observation policy and generate one
# kind of target on every stream. `view_tiling` governs the OBSERVATION layout (a 2x2 role
# canvas vs N images) and `gen_video_key` the GENERATION target (a 2x2 canvas vs a single
# role) — decoupled in the loader, so BOTH must be unified for the expert to face one task.
_UNIFIED_OBS_KEYS = ("obs_image_size", "obs_resize_mode", "view_tiling", "tile_cell_size", "gen_video_key")


def _normalize_obs_key(key, value):
    """Canonicalize a unified-obs value for cross-stream equality (list vs tuple, None ->
    the loader default) so ``[256, 256]`` == ``(256, 256)`` and ``fixed`` == unset."""
    from starVLA.dataloader.view_tiling import normalize_obs_resize_mode, normalize_view_tiling_mode, resolve_image_size

    if key == "obs_resize_mode":
        return normalize_obs_resize_mode(value)  # None -> "fixed"
    if key == "view_tiling":
        return normalize_view_tiling_mode(value)  # None/False -> "off"
    if key in ("obs_image_size", "tile_cell_size"):
        return resolve_image_size(value, None)  # None -> None; [W,H]/(W,H) -> (W, H)
    if key == "gen_video_key":
        return str(value).strip().lower()
    return value


def _unify_observation_policy(datasets):
    """Force the model-level observation / gen-target policy to be ONE choice across streams.

    Mirrors the ``datasets.shared`` propagation loop, but for keys that MUST be identical on
    every stream (unlike ``dataset_fps``). Single source of truth, per key:

      1. ``datasets.shared.<key>`` present -> it is the unified value (the shared loop above
         already wrote it onto every block), so skip.
      2. else derive from the streams that EXPLICITLY declare the key: they must all agree,
         and the agreed value fills the streams that left it unset.
      3. nobody declares it -> leave untouched (every stream then takes the same loader
         default, so they are trivially consistent).
      4. two or more explicit declarers disagree -> raise (fail-loud), pointing at
         ``datasets.shared`` — never silently let one stream win.

    A ``default`` value is NOT treated as a declaration (only explicit per-block keys count),
    so "vla unset (implicit 224) + video explicit 256" is a single declarer that fills vla
    with 256 rather than a 224-vs-256 conflict. Single-stream configs (<= 1 block) are a
    no-op.
    """
    from omegaconf import DictConfig

    blocks = {
        name: block
        for name, block in datasets.items()
        if name not in _DATASETS_RESERVED and isinstance(block, DictConfig)
    }
    if len(blocks) <= 1:
        return  # single stream: no cross-stream consistency to enforce

    shared = datasets.get("shared", None)
    for key in _UNIFIED_OBS_KEYS:
        if shared is not None and key in shared:
            continue  # shared is the single source of truth; already propagated to every block

        declared = {name: block[key] for name, block in blocks.items() if key in block and block[key] is not None}
        normalized = {name: _normalize_obs_key(key, value) for name, value in declared.items()}
        distinct = set(normalized.values())
        if len(distinct) > 1:
            raise ValueError(
                f"datasets.*.{key} disagrees across streams: {normalized}. This is a MODEL-LEVEL "
                f"observation/generation policy (the shared gen expert must see one observation "
                f"layout and generate one kind of target on every stream) and must be identical. "
                f"Set datasets.shared.{key} (single source of truth) instead of conflicting per-stream values."
            )
        if not distinct:
            continue  # nobody declared it -> loader defaults are consistent

        unified_raw = next(iter(declared.values()))  # keep the author's original spelling (list/str)
        unified_norm = next(iter(distinct))
        for block in blocks.values():
            if _normalize_obs_key(key, block.get(key, None)) != unified_norm:
                block[key] = unified_raw
        logger.info(f"[align] unified observation policy {key}={unified_raw!r} across {list(blocks)}")


def _validate_observation_preprocessing(cfg, datasets):
    """Validate the unified observation policy after cross-stream propagation.

    Native inputs may still yield different observation-token counts because their
    source dimensions differ.  What remains strict here is each stream's normalized
    mode/layout plus the processor-side pixel budget required by native mode.
    """
    from omegaconf import DictConfig

    from starVLA.dataloader.view_tiling import normalize_obs_resize_mode, normalize_view_tiling_mode, resolve_image_size

    native_streams = []
    for name, block in datasets.items():
        if name in _DATASETS_RESERVED or not isinstance(block, DictConfig):
            continue

        mode = normalize_obs_resize_mode(block.get("obs_resize_mode", None))
        block["obs_resize_mode"] = mode
        if mode == "native":
            native_streams.append(name)

        try:
            view_tiling = normalize_view_tiling_mode(block.get("view_tiling", None))
        except ValueError as exc:
            raise ValueError(f"datasets.{name}.{exc}") from exc
        block["view_tiling"] = view_tiling
        if view_tiling == "tile":
            # Validate the 2x2 role-tile cell geometry so a bad config fails at load,
            # not deep in the loader (canvas = 2*cell; cell defaults handled downstream).
            resolve_image_size(block.get("tile_cell_size", None))

    if not native_streams:
        return

    max_pixels = OmegaConf.select(cfg, "framework.qwenvl.max_pixels", default=None)
    if isinstance(max_pixels, bool) or not isinstance(max_pixels, Integral) or int(max_pixels) <= 0:
        raise ValueError(
            f"obs_resize_mode='native' for streams {native_streams} requires a positive integer "
            "framework.qwenvl.max_pixels so Qwen observation memory remains bounded; "
            f"got {max_pixels!r}."
        )
    base_vlm = str(OmegaConf.select(cfg, "framework.qwenvl.base_vlm", default=""))
    if "qwen3.5" not in base_vlm.lower():
        raise ValueError(
            f"obs_resize_mode='native' is currently budgeted only by the Qwen3.5 processor, "
            f"but framework.qwenvl.base_vlm={base_vlm!r} for streams {native_streams}."
        )
    logger.info(
        f"[align] native observations enabled for {native_streams}; "
        f"Qwen3.5 max_pixels={int(max_pixels)}. Observation token counts may differ across streams."
    )


def resolve_stream_batch_sizes(cfg, stream_names):
    """Split a total per-device batch budget across streams by their sampling ratio.

    Each dataset block may declare ``sampling_ratio`` (default 1.0). When
    ``datasets.total_per_device_batch_size`` is set, the full budget is apportioned
    with Hamilton's largest-remainder method, then any zero-seat stream is repaired
    to the required minimum of one. The result is written back into the config so
    :func:`build_dataloader` picks it up. Unlike independent ``round`` calls, this
    guarantees that the resolved stream batches sum exactly to the configured total.

    When ``total_per_device_batch_size`` is absent, each stream keeps its explicit
    ``per_device_batch_size`` (the ratio is then only informational).
    """
    stream_names = list(stream_names)
    if not stream_names:
        raise ValueError("resolve_stream_batch_sizes requires at least one stream")
    if len(set(stream_names)) != len(stream_names):
        raise ValueError(f"stream_names must be unique, got {stream_names!r}")

    datasets = cfg.datasets
    total = datasets.get("total_per_device_batch_size", None)
    ratios = {}
    for name in stream_names:
        try:
            ratio = float(datasets[name].get("sampling_ratio", 1.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"datasets.{name}.sampling_ratio must be a finite positive number") from exc
        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError(f"datasets.{name}.sampling_ratio must be a finite positive number, got {ratio!r}")
        ratios[name] = ratio

    if total is not None:
        if isinstance(total, bool) or not isinstance(total, Integral) or int(total) < len(stream_names):
            raise ValueError(
                "datasets.total_per_device_batch_size must be an integer at least as large as "
                f"the number of streams ({len(stream_names)}), got {total!r}"
            )
        total = int(total)
        denominator = sum(ratios.values())
        quotas = {name: total * ratios[name] / denominator for name in stream_names}
        sizes = {name: math.floor(quotas[name]) for name in stream_names}
        seats_left = total - sum(sizes.values())
        order = sorted(
            range(len(stream_names)),
            key=lambda index: (-(quotas[stream_names[index]] - sizes[stream_names[index]]), index),
        )
        for index in order[:seats_left]:
            sizes[stream_names[index]] += 1

        # A very small positive ratio can receive zero seats under unconstrained
        # Hamilton apportionment. Keep every declared stream active by transferring
        # one seat from the currently most over-allocated eligible donor.
        for name in stream_names:
            if sizes[name] > 0:
                continue
            donors = [candidate for candidate in stream_names if sizes[candidate] > 1]
            if not donors:
                raise RuntimeError(
                    "cannot satisfy the one-sample-per-stream minimum despite a valid total: "
                    f"resolved={sizes}, total={total}"
                )
            donor = max(
                donors,
                key=lambda candidate: (
                    sizes[candidate] - quotas[candidate],
                    sizes[candidate],
                    -stream_names.index(candidate),
                ),
            )
            sizes[donor] -= 1
            sizes[name] = 1

        if sum(sizes.values()) != total:
            raise RuntimeError("internal stream-batch apportionment error: " f"resolved={sizes}, total={total}")
        for name, batch_size in sizes.items():
            datasets[name].per_device_batch_size = batch_size
    else:
        sizes = {}
        for name in stream_names:
            value = datasets[name].per_device_batch_size
            if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
                raise ValueError(f"datasets.{name}.per_device_batch_size must be a positive integer, got {value!r}")
            sizes[name] = int(value)

    resolved_total = sum(sizes.values())
    resolved_fractions = {name: sizes[name] / resolved_total for name in stream_names}
    logger.info(
        f"[ratio] per-stream per_device_batch_size "
        f"(total_per_device_batch_size={total}, sampling_ratio={ratios}): "
        f"sizes={sizes}, resolved_fractions={resolved_fractions}"
    )
    return sizes


def resolve_video_episode_cap(cfg, num_vla_used_episodes):
    """Derive the pure-video episode cap from the vla ``episode_fraction`` (co-train).

    When ``datasets.vla_data.episode_fraction`` is below 1, the robot stream trains on a
    restricted episode subset (e.g. a quick smoke run); the video stream's usable clip
    count is then scaled to match by the co-training batch split ratio:

        cap = clamp(round(num_vla_used * r_video / r_vla), 1, N_video)

    where ``r_vla`` / ``r_video`` are the streams' ``sampling_ratio``. The cap is
    written to ``datasets.video_data.max_episodes`` so :func:`build_dataloader`'s
    ``video_datasets`` branch truncates its clip list to it (the upper clamp to the
    actual clip count ``N_video`` happens inside ``VideoClipDataset``).

    No-op (returns None) when ``episode_fraction`` is unset or 1 (the full robot data):
    the video stream then uses every clip of its index (or its own ``episode_fraction``).
    """
    fraction = cfg.datasets.vla_data.get("episode_fraction", None)
    if fraction is None or float(fraction) >= 1.0:
        return None  # full robot data -> do not restrict the video stream

    # An explicit video episode_fraction sizes the video pool independently (its own
    # per-stream data budget), so do NOT let the robot fraction derive a competing cap.
    if cfg.datasets.video_data.get("episode_fraction", None) is not None:
        logger.info(
            "[episode-fraction] video stream sets its own episode_fraction; " "skipping the vla-derived episode cap."
        )
        return None

    r_vla = float(cfg.datasets.vla_data.get("sampling_ratio", 1.0)) or 1.0  # guard /0
    r_video = float(cfg.datasets.video_data.get("sampling_ratio", 1.0))
    cap = max(1, round(num_vla_used_episodes * r_video / r_vla))
    cfg.datasets.video_data.max_episodes = cap
    logger.info(
        f"[episode-fraction] video episode cap = {cap} "
        f"(num_vla_used={num_vla_used_episodes}, r_vla={r_vla}, r_video={r_video}, "
        f"episode_fraction={fraction})"
    )
    return cap


def _loader_kwargs(block_cfg):
    """DataLoader worker settings for a dataset block (config-driven).

    Reads num_workers / prefetch_factor / persistent_workers /
    dataloader_timeout_seconds from the block cfg (defaults 16 / 4 / True / 0).
    A positive timeout makes a stuck worker or file-system read fail loudly instead
    of leaving every distributed rank blocked forever. A co-training config can
    dial these per stream — e.g. the pure-video stream may set
    persistent_workers=False (+ a bounded worker generation) so its workers
    recycle and reclaim PyAV's per-open VP9 leak, and lower num_workers so that
    two loaders per rank fit the CPU budget.
    num_workers==0 forbids persistent_workers/prefetch_factor and requires a zero
    timeout (PyTorch), so guard that.
    """
    get = block_cfg.get if hasattr(block_cfg, "get") else (lambda k, d=None: getattr(block_cfg, k, d))
    num_workers = int(get("num_workers", 16))
    timeout_seconds = int(get("dataloader_timeout_seconds", 0))
    if timeout_seconds < 0:
        raise ValueError("dataloader_timeout_seconds must be a non-negative integer, " f"got {timeout_seconds}")
    if num_workers <= 0:
        if timeout_seconds != 0:
            raise ValueError("dataloader_timeout_seconds must be 0 when num_workers=0; " f"got {timeout_seconds}")
        return {"num_workers": 0}
    return {
        "num_workers": num_workers,
        "persistent_workers": bool(get("persistent_workers", True)),
        "prefetch_factor": int(get("prefetch_factor", 4)),
        "timeout": timeout_seconds,
    }


def _val_loader_kwargs(block_cfg):
    """DataLoader worker settings for the hold-out validation loader.

    The val loader is used only every ``eval_interval`` steps for a handful of
    batches, so a second persistent 16-worker pool per rank would waste CPU/RAM.
    Default to a small, non-persistent pool (``val_num_workers``, default 2).
    """
    get = block_cfg.get if hasattr(block_cfg, "get") else (lambda k, d=None: getattr(block_cfg, k, d))
    num_workers = int(get("val_num_workers", 2))
    timeout_seconds = int(
        get(
            "val_dataloader_timeout_seconds",
            get("dataloader_timeout_seconds", 0),
        )
    )
    if timeout_seconds < 0:
        raise ValueError("val_dataloader_timeout_seconds must be a non-negative integer, " f"got {timeout_seconds}")
    if num_workers <= 0:
        if timeout_seconds != 0:
            raise ValueError(
                "val_dataloader_timeout_seconds must be 0 when val_num_workers=0; " f"got {timeout_seconds}"
            )
        return {"num_workers": 0}
    return {
        "num_workers": num_workers,
        "persistent_workers": False,
        "prefetch_factor": 2,
        "timeout": timeout_seconds,
    }


def _resolve_collate_fn(base_collate, block_cfg, batch_preprocessor, *, is_val: bool):
    """Optionally attach exact Qwen preprocessing to the DataLoader worker.

    The wrapper retains the original list-of-dict examples for action labels and
    generation targets, and adds only the CPU processor tensors. By default the
    model-agnostic identity collate is used.
    """

    get = block_cfg.get if hasattr(block_cfg, "get") else (lambda k, d=None: getattr(block_cfg, k, d))
    if not bool(get("worker_preprocess_qwen", False)):
        return base_collate
    if batch_preprocessor is None:
        raise ValueError("worker_preprocess_qwen=true requires a framework-provided Qwen batch preprocessor.")
    workers_key = "val_num_workers" if is_val else "num_workers"
    default_workers = 2 if is_val else 16
    if int(get(workers_key, default_workers)) <= 0:
        raise ValueError(f"worker_preprocess_qwen=true requires {workers_key} > 0.")
    logger.info(
        f"[worker-preprocess] Qwen resize/tokenization enabled ({workers_key}={int(get(workers_key, default_workers))})"
    )
    return QwenPreprocessingCollator(base_collate=base_collate, preprocessor=batch_preprocessor)


def _resolve_video_worker_generation_samples(block_cfg, world_size=None):
    """Resolve a rank-local worker lifetime into the map-dataset length.

    ``worker_generation_batches_per_rank`` counts batches consumed from each
    prepared rank-local video loader before ``persistent_workers=False`` recycles
    its workers. The underlying dataset exists before Accelerate shards its batch
    sampler, so its global length must be ``batches * world_size * batch_size``.
    Exact divisibility also prevents Accelerate from padding the final shard.

    This helper is deliberately pure: the authoritative config keeps only the
    user-facing unit, while the derived sample count is applied to the dataset
    instance. Configs that set only ``samples_per_epoch`` use it unchanged.
    Defining both forms is rejected because they describe the same physical
    boundary with different units.
    """
    get = block_cfg.get if hasattr(block_cfg, "get") else (lambda k, d=None: getattr(block_cfg, k, d))
    generation_batches = get("worker_generation_batches_per_rank", None)
    configured_samples = get("samples_per_epoch", None)
    double_buffer = bool(get("double_buffer_worker_generations", False))

    if generation_batches is None:
        if double_buffer:
            raise ValueError("double_buffer_worker_generations=true requires worker_generation_batches_per_rank.")
        return configured_samples
    if configured_samples is not None:
        raise ValueError(
            "datasets.video_data must set only one of samples_per_epoch or " "worker_generation_batches_per_rank"
        )

    if isinstance(generation_batches, bool) or not isinstance(generation_batches, Integral) or generation_batches <= 0:
        raise ValueError("worker_generation_batches_per_rank must be a positive integer")
    generation_batches = int(generation_batches)
    if double_buffer:
        prewarm_batches = get("worker_generation_prewarm_batches", 8)
        if (
            isinstance(prewarm_batches, bool)
            or not isinstance(prewarm_batches, Integral)
            or int(prewarm_batches) <= 0
            or int(prewarm_batches) >= generation_batches
        ):
            raise ValueError(
                "worker_generation_prewarm_batches must be a positive integer smaller than "
                f"worker_generation_batches_per_rank={generation_batches}; got {prewarm_batches!r}."
            )

    if bool(get("persistent_workers", True)):
        raise ValueError("worker_generation_batches_per_rank requires persistent_workers=false")
    if int(get("num_workers", 16)) <= 0:
        raise ValueError("worker_generation_batches_per_rank requires num_workers > 0")

    batch_size = get("per_device_batch_size", None)
    if batch_size is None or int(batch_size) <= 0:
        raise ValueError("datasets.video_data.per_device_batch_size must be a positive integer")
    batch_size = int(batch_size)

    if world_size is None:
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    world_size = int(world_size)
    if world_size <= 0:
        raise ValueError("world_size must be a positive integer")

    return generation_batches * world_size * batch_size


def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                if isinstance(stats["action"][k], np.ndarray):
                    stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                if isinstance(stats["num_trajectories"], np.ndarray):
                    stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                if isinstance(stats["num_transitions"], np.ndarray):
                    stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    logger.info(f"Saved dataset statistics file at path {out_path}")


def _resolve_data_config_for_split(block_cfg, split):
    """Build a stream config for one split without mutating its training contract.

    ``validation_overrides`` is intentionally applied only to validation. This is
    useful for controlled ablations whose training time grid differs while their
    official-val ground truth must remain identical.
    """
    if split != "val":
        return block_cfg
    overrides = block_cfg.get("validation_overrides", {})
    # Validation must be deterministic + comparable across evals: disable the training-only
    # random view-drop augmentation on the val split (mode="val" is otherwise deterministic).
    return OmegaConf.merge(block_cfg, overrides, {"split": "val", "view_drop_prob": 0.0})


def build_dataloader(
    cfg,
    dataset_py="lerobot_datasets",
    split="train",
    block=None,
    world_size=None,
    batch_preprocessor=None,
):
    """Build the DataLoader for one data stream.

    ``dataset_py`` selects the backend:

    * ``lerobot_datasets``     — robot LeRobot data (image / lang / action / optional state);
    * ``gen_lerobot_datasets`` — the same, plus the future-frame generation target;
    * ``video_datasets``       — action-free video clips from an ``index.jsonl`` (generation only).

    ``block`` selects which ``cfg.datasets.<block>`` the two LeRobot backends read, so a
    LeRobot backend can serve either the primary ``vla_data`` stream or a secondary one
    (default ``vla_data``). The ``video_datasets`` backend always reads ``datasets.video_data``.
    """
    is_val = split == "val"
    if world_size is None:
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    if isinstance(world_size, bool) or not isinstance(world_size, Integral) or int(world_size) <= 0:
        raise ValueError(f"world_size must be a positive integer, got {world_size!r}")
    world_size = int(world_size)

    if dataset_py == "lerobot_datasets":
        from starVLA.dataloader.lerobot_datasets import collate_fn, get_vla_dataset

        resolved_block = block or "vla_data"
        vla_dataset_cfg = cfg.datasets[resolved_block]
        # val loader: deterministic sampling (mode="val") over the hold-out episode
        # split (split="val"); a merged copy avoids mutating the shared train cfg.
        data_cfg = _resolve_data_config_for_split(vla_dataset_cfg, split)

        vla_dataset = get_vla_dataset(
            data_cfg=data_cfg,
            mode="val" if is_val else "train",
            balance_dataset_weights=vla_dataset_cfg.get("balance_dataset_weights", False),
            balance_trajectory_weights=vla_dataset_cfg.get("balance_trajectory_weights", False),
            sampling_epoch_multiple=(1 if is_val else int(vla_dataset_cfg.per_device_batch_size) * world_size),
        )

        vla_dataloader = DataLoader(
            vla_dataset,
            batch_size=vla_dataset_cfg.per_device_batch_size,
            collate_fn=_resolve_collate_fn(collate_fn, vla_dataset_cfg, batch_preprocessor, is_val=is_val),
            pin_memory=True,
            **(_val_loader_kwargs(vla_dataset_cfg) if is_val else _loader_kwargs(vla_dataset_cfg)),
        )
        # Stats are full-dataset (train and val share them); only the train build writes.
        # Non-primary blocks (e.g. a lerobot-backed video_data stream) write a
        # block-suffixed file so they never clobber the canonical vla_data stats.
        if not is_val and (not dist.is_initialized() or dist.get_rank() == 0):
            output_dir = Path(cfg.output_dir)
            stats_name = (
                "dataset_statistics.json"
                if resolved_block == "vla_data"
                else f"dataset_statistics.{resolved_block}.json"
            )
            vla_dataset.save_dataset_statistics(output_dir / stats_name)
        return vla_dataloader
    elif dataset_py == "gen_lerobot_datasets":
        # Same as lerobot_datasets, but each sample additionally carries a
        # `gen_target_image` (the future frame) for the generation head.
        from starVLA.dataloader.gen_lerobot_datasets import collate_fn, get_gen_vla_dataset

        resolved_block = block or "vla_data"
        vla_dataset_cfg = cfg.datasets[resolved_block]
        data_cfg = _resolve_data_config_for_split(vla_dataset_cfg, split)
        vla_dataset = get_gen_vla_dataset(
            data_cfg=data_cfg,
            mode="val" if is_val else "train",
            balance_dataset_weights=vla_dataset_cfg.get("balance_dataset_weights", False),
            balance_trajectory_weights=vla_dataset_cfg.get("balance_trajectory_weights", False),
            sampling_epoch_multiple=(1 if is_val else int(vla_dataset_cfg.per_device_batch_size) * world_size),
        )
        vla_dataloader = DataLoader(
            vla_dataset,
            batch_size=vla_dataset_cfg.per_device_batch_size,
            collate_fn=_resolve_collate_fn(collate_fn, vla_dataset_cfg, batch_preprocessor, is_val=is_val),
            pin_memory=True,
            **(_val_loader_kwargs(vla_dataset_cfg) if is_val else _loader_kwargs(vla_dataset_cfg)),
        )
        # Non-primary blocks (e.g. a lerobot-backed video_data gen stream) write a
        # block-suffixed stats file so they never clobber the canonical vla_data stats.
        if not is_val and (not dist.is_initialized() or dist.get_rank() == 0):
            output_dir = Path(cfg.output_dir)
            stats_name = (
                "dataset_statistics.json"
                if resolved_block == "vla_data"
                else f"dataset_statistics.{resolved_block}.json"
            )
            vla_dataset.save_dataset_statistics(output_dir / stats_name)
        return vla_dataloader
    elif dataset_py == "video_datasets":
        # Pure-video (action-less) generation data: each sample carries the shared
        # gen keys (image / lang / gen_target_frames) but NO action/state, so it
        # feeds ONLY the gen loss (model.forward(..., loss_keys={"gen_loss"})).
        # No action normalization stats to save (there is no action here).
        from starVLA.dataloader.video_datasets import collate_fn, get_video_dataset

        video_dataset_cfg = cfg.datasets.video_data
        samples_per_generation = _resolve_video_worker_generation_samples(
            video_dataset_cfg,
            world_size=world_size,
        )
        video_dataset = get_video_dataset(video_dataset_cfg, seed=cfg.get("seed", 42))
        if video_dataset_cfg.get("worker_generation_batches_per_rank", None) is not None:
            video_dataset.samples_per_epoch = samples_per_generation
            logger.info(
                "[video-worker-generation] "
                f"{video_dataset_cfg.worker_generation_batches_per_rank} batches/rank x "
                f"{world_size} ranks x "
                f"{video_dataset_cfg.per_device_batch_size} samples/batch = "
                f"{samples_per_generation} samples"
            )
        video_train_dataloader = DataLoader(
            video_dataset,
            batch_size=video_dataset_cfg.per_device_batch_size,
            collate_fn=_resolve_collate_fn(collate_fn, video_dataset_cfg, batch_preprocessor, is_val=False),
            pin_memory=True,
            **_loader_kwargs(video_dataset_cfg),
        )
        return video_train_dataloader
    else:
        raise ValueError(
            f"unsupported dataset_py={dataset_py!r}; expected one of "
            "'lerobot_datasets', 'gen_lerobot_datasets', 'video_datasets'."
        )
