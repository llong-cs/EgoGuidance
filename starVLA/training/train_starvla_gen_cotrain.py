# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [starVLA community] in [2025].

"""Multi-stream co-training: robot data (action + generation) with action-free video (generation only).

Every data source is a ``datasets.<block>`` with its own loss binding, for example:

    datasets.vla_data    (robot, e.g. DROID)              losses: [action_loss, gen_loss]
    datasets.vla_data_b  (optional second robot dataset)  losses: [action_loss, gen_loss]
    datasets.video_data  (egocentric video, e.g. Open-AoE) losses: [gen_loss]

Robot and video streams share the same generation loss (computed from images only), so cheap
video supervises the generation expert while robot data also drives the action expert. All
streams are stepped together (``trainer.cotrain_mode: joint``): each training step combines one
mini-batch per stream, the mini-batches are forwarded separately (or, optionally, in one fused
VLM forward with ``trainer.fuse_joint_vlm_forward: true``), each computes only its bound losses,
and their gradients accumulate into a single optimizer step.

Notes:
  * Within one stream, loss aggregation is the ``sum(*_loss * loss_scale)`` rule of
    ``train_starvla.py``. Each loss is averaged within its own mini-batch and the streams'
    losses are summed, so the per-device ``sampling_ratio`` changes how many samples of
    each source are seen per step, not the relative weight of the objectives.
  * Action validation runs on the primary robot stream only (video has no actions).
  * ``trainer.save_full_state: true`` additionally writes resumable
    ``checkpoints/steps_<N>/`` directories (model, optimizer, scheduler, RNG).

Launch with ``bash scripts/train.sh <config.yaml> [--dotted.key value ...]``.
"""

# Standard Library
import argparse
import json
import os
import time
from pathlib import Path

# Third-Party Libraries
import numpy as np
import torch
import torch.distributed as dist
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local Modules
from starVLA.dataloader import (
    align_dataset_configs,
    build_dataloader,
    resolve_stream_batch_sizes,
    resolve_video_episode_cap,
)
from starVLA.dataloader.preprocessed_batch import batch_size as get_batch_size
from starVLA.dataloader.preprocessed_batch import unpack_preprocessed_batch
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat, load_config_with_bases
from starVLA.training.distributed import distributed_barrier, release_distributed
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.tracking import MetricsTracker
from starVLA.training.trainer_utils.trainer_tools import (
    DoubleBufferedWorkerIterator,
    TrainerUtils,
    build_accelerator,
    mean_action_mse,
    normalize_dotlist_args,
    resolve_output_dir,
    setup_optimizer_and_scheduler,
)
from starVLA.training.trainer_utils.training_state import (
    COMPLETE_MARKER,
    TRAINER_STATE_FILE,
    resolve_progress_interval,
)

# Optional Ascend NPU support: maps CUDA calls to NPU when torch_npu is installed.
try:
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
except ImportError:
    pass

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = get_logger(__name__)


def setup_directories(cfg) -> Path:
    """Create ``<run_root_dir>/<run_id>`` and its ``checkpoints/`` subdirectory."""
    output_dir = Path(resolve_output_dir(cfg))

    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

    return output_dir


_DATASETS_RESERVED = ("shared", "total_per_device_batch_size")


def discover_stream_blocks(cfg) -> list:
    """Ordered co-training stream block names present in the config.

    The primary ``vla_data`` first, then every other non-reserved ``datasets.<block>``
    that declares a ``dataset_py`` (e.g. ``video_data`` or a second robot stream such as
    ``vla_data_b``), in config order.
    """
    blocks = [
        b
        for b in cfg.datasets
        if b not in _DATASETS_RESERVED and hasattr(cfg.datasets[b], "get") and cfg.datasets[b].get("dataset_py", None)
    ]
    if "vla_data" not in blocks:
        raise ValueError(
            "gen co-training requires a datasets.vla_data (primary) block with a dataset_py "
            f"(found stream blocks: {blocks})."
        )
    return ["vla_data"] + [b for b in blocks if b != "vla_data"]


def _stream_prefix(name: str) -> str:
    """Log-metric prefix for a stream: "" for the primary ``vla_data``, "<x>_" for the others
    (e.g. video_data -> "video_", vla_data_b -> "vla_b_")."""
    return "" if name == "vla_data" else f"{name.replace('_data', '')}_"


def _stream_exposure_name(block, name: str) -> str:
    """Stable human-facing source label for exposure audit records."""
    label = str(block.get("exposure_name", name.removesuffix("_data"))).strip().lower()
    if not label or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in label):
        raise ValueError(
            f"datasets.{name}.exposure_name must contain only lowercase letters, digits, and underscores; "
            f"got {label!r}"
        )
    return label


def _stream_loss_keys(cfg, name: str) -> set:
    """Loss binding for a stream from ``datasets.<name>.losses`` (defaults: the primary
    trains action_loss+gen_loss, a secondary trains gen_loss)."""
    default = ["action_loss", "gen_loss"] if name == "vla_data" else ["gen_loss"]
    return set(cfg.datasets[name].get("losses", default))


def prepare_data(cfg, accelerator, output_dir, batch_preprocessor=None) -> "tuple[list, DataLoader | None, list]":
    """Prepare the N co-training streams (primary ``vla_data`` + secondaries) and the
    optional hold-out validation loaders.

    Returns ``(streams, val_dataloader, loss_val_streams)``. The primary robot val loader
    remains separate for deterministic ``val/action_mse`` evaluation. The loss-val list
    contains split-capable secondary streams evaluated on their raw bound losses."""
    block_names = discover_stream_blocks(cfg)
    streams = []
    primary_num_episodes = None
    for name in block_names:
        block = cfg.datasets[name]
        # When episode_fraction restricts the robot stream, cap the video stream's usable
        # episodes to match (no-op otherwise). Runs after the primary stream is built and
        # before the video stream.
        if name == "video_data" and primary_num_episodes is not None:
            resolve_video_episode_cap(cfg, primary_num_episodes)
        src = block.get("data_mix", None) or block.get("index_jsonl", "?")
        logger.info(f"Creating '{name}' stream (dataset_py={block.dataset_py}, source=`{src}`)")
        loader = build_dataloader(
            cfg=cfg,
            dataset_py=block.dataset_py,
            block=name,
            batch_preprocessor=batch_preprocessor,
        )
        if name == "vla_data":
            primary_num_episodes = loader.dataset.num_episodes
        streams.append(
            {
                "name": name,
                "dataloader": loader,
                "loss_keys": _stream_loss_keys(cfg, name),
                "prefix": _stream_prefix(name),
                "exposure_name": _stream_exposure_name(block, name),
                "is_primary": name == "vla_data",
                "iter": None,
                "epoch_count": 0,
            }
        )

    # Action validation loader for the primary robot stream. Uses either the dataset's
    # official split or the deterministic episode hold-out. Not accelerator.prepare'd, so
    # every rank sees the same validation batches.
    val_dataloader = None
    if (
        cfg.datasets.vla_data.get("use_official_splits", False)
        or cfg.datasets.vla_data.get("val_episode_fraction", None) is not None
    ):
        logger.info("Creating VLA validation loader (split='val')")
        val_dataloader = build_dataloader(
            cfg=cfg,
            dataset_py=cfg.datasets.vla_data.dataset_py,
            split="val",
            block="vla_data",
            batch_preprocessor=batch_preprocessor,
        )

    # Independent secondary validation streams. They MUST NOT share the primary
    # val iterator: advancing it for loss evaluation would change future batches
    # used by robot action MSE. Only LeRobot backends implement split="val".
    loss_val_streams = []
    split_capable_backends = {"lerobot_datasets", "gen_lerobot_datasets"}
    for stream in streams:
        if stream["is_primary"]:
            continue
        name = stream["name"]
        block = cfg.datasets[name]
        has_validation_split = bool(
            block.get("use_official_splits", False) or block.get("val_episode_fraction", None) is not None
        )
        if not has_validation_split:
            continue
        if block.dataset_py not in split_capable_backends:
            raise ValueError(
                f"datasets.{name} requests validation, but dataset_py={block.dataset_py!r} "
                "does not implement split='val'. Supported backends: "
                f"{sorted(split_capable_backends)}"
            )
        logger.info(f"Creating '{name}' loss-validation loader (split='val', " f"losses={sorted(stream['loss_keys'])})")
        loss_val_streams.append(
            {
                "name": name,
                "dataloader": build_dataloader(
                    cfg=cfg,
                    dataset_py=block.dataset_py,
                    split="val",
                    block=name,
                    batch_preprocessor=batch_preprocessor,
                ),
                "loss_keys": stream["loss_keys"],
                "prefix": stream["prefix"],
                "iter": None,
                "epoch_count": 0,
            }
        )

    accelerator.dataloader_config.dispatch_batches = False
    distributed_barrier()
    return streams, val_dataloader, loss_val_streams


def resolve_schedule(cfg) -> None:
    """Resolve the horizon and every cadence knob into ``trainer.resolved_*`` steps.

    As in ``train_starvla.py``, each cadence accepts either an absolute step count or a
    ratio of the horizon (exactly one of the two). Runtime consumers read only the
    resolved values, so a config written in ratios keeps its schedule shape when the
    step budget changes.

    | absolute            | ratio                  |
    |---------------------|------------------------|
    | num_warmup_steps    | warmup_ratio           |
    | logging_frequency   | logging_interval_ratio |
    | eval_interval       | eval_interval_ratio    |
    | save_interval       | save_interval_ratio    |

    ``trainer.num_sampling_epochs`` is rejected: with several streams of different
    length a sampling epoch is undefined.
    """
    if cfg.trainer.get("num_sampling_epochs", None) is not None:
        raise ValueError(
            "trainer.num_sampling_epochs is not supported by gen co-training: with two "
            "streams of different length a sampling epoch is undefined. Set "
            "trainer.max_train_steps (optionally with the *_ratio cadence keys) instead."
        )

    total_steps = cfg.trainer.get("max_train_steps", None)
    if total_steps is None:
        raise ValueError("trainer.max_train_steps must be set for gen co-training.")
    if isinstance(total_steps, bool) or not isinstance(total_steps, int) or int(total_steps) < 1:
        raise ValueError(f"trainer.max_train_steps must be a positive integer, got {total_steps!r}.")
    total_steps = int(total_steps)
    cfg.trainer.resolved_max_train_steps = total_steps

    schedule_fields = (
        ("num_warmup_steps", "warmup_ratio", True),
        ("logging_frequency", "logging_interval_ratio", False),
        ("eval_interval", "eval_interval_ratio", False),
        ("save_interval", "save_interval_ratio", False),
    )
    for absolute_field, ratio_field, allow_zero_absolute in schedule_fields:
        setattr(
            cfg.trainer,
            f"resolved_{absolute_field}",
            resolve_progress_interval(
                absolute_field=absolute_field,
                ratio_field=ratio_field,
                absolute_value=cfg.trainer.get(absolute_field, None),
                ratio_value=cfg.trainer.get(ratio_field, None),
                total_steps=total_steps,
                allow_zero_absolute=allow_zero_absolute,
            ),
        )

    logger.info(
        "[gen-cotrain] resolved schedule: max_train_steps=%d warmup=%d logging=%d eval=%d save=%d",
        total_steps,
        cfg.trainer.resolved_num_warmup_steps,
        cfg.trainer.resolved_logging_frequency,
        cfg.trainer.resolved_eval_interval,
        cfg.trainer.resolved_save_interval,
    )


def validate_cotrain_options(cfg) -> None:
    """Reject options that this entry does not implement."""
    mode = cfg.trainer.get("cotrain_mode", "joint")
    if mode != "joint":
        raise ValueError(f"trainer.cotrain_mode must be 'joint' (all streams every step), got {mode!r}")
    if cfg.trainer.get("transfer_policy", None) not in (None, ""):
        raise ValueError(
            "trainer.transfer_policy is implemented by starVLA/training/train_starvla.py only; "
            "use trainer.pretrained_checkpoint (optionally with trainer.reload_modules) here."
        )


class VLAGenCotrainer(TrainerUtils):
    def __init__(
        self,
        cfg,
        model,
        streams,
        optimizer,
        lr_scheduler,
        accelerator,
        val_dataloader=None,
        loss_val_streams=None,
    ):
        self.config = cfg
        self.model = model
        # Ordered stream descriptors: primary vla_data first, then the other streams.
        self.streams = streams
        self.primary = next(s for s in streams if s["is_primary"])
        # Alias to the primary (robot) loader.
        self.vla_train_dataloader = self.primary["dataloader"]
        self.val_dataloader = val_dataloader
        self.loss_val_streams = loss_val_streams or []
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.total_train_steps = int(cfg.trainer.resolved_max_train_steps)
        # Encode all streams' VLM inputs in one forward and split the hidden states back per
        # stream (requires worker-preprocessed Qwen inputs on every stream).
        self._fuse_joint_vlm_forward = bool(cfg.trainer.get("fuse_joint_vlm_forward", False))
        if self._fuse_joint_vlm_forward:
            if len(streams) < 2:
                raise ValueError("trainer.fuse_joint_vlm_forward requires at least two active streams.")
            if not bool(getattr(model, "supports_fused_cotrain_forward", False)):
                raise ValueError(f"trainer.fuse_joint_vlm_forward is unsupported by framework {type(model).__name__}.")
            missing_preprocess = [
                stream["name"]
                for stream in streams
                if not bool(cfg.datasets[stream["name"]].get("worker_preprocess_qwen", False))
            ]
            if missing_preprocess:
                raise ValueError(
                    "trainer.fuse_joint_vlm_forward requires worker-preprocessed Qwen inputs for every stream; "
                    f"missing on {missing_preprocess}."
                )

        exposure_names = [stream["exposure_name"] for stream in self.streams]
        if len(set(exposure_names)) != len(exposure_names):
            raise ValueError(f"datasets.*.exposure_name values must be unique, got {exposure_names!r}")
        self._stream_micro_batches = {
            stream["name"]: int(self.config.datasets[stream["name"]].per_device_batch_size) for stream in self.streams
        }
        self._planned_ratios = {
            stream["name"]: float(self.config.datasets[stream["name"]].get("sampling_ratio", 1.0))
            for stream in self.streams
        }
        planned_total = sum(self._planned_ratios.values())
        self._planned_fractions = {name: ratio / planned_total for name, ratio in self._planned_ratios.items()}
        self._pending_stream_samples = {stream["name"]: 0 for stream in self.streams}
        self._cumulative_stream_samples = {stream["name"]: 0 for stream in self.streams}
        self._exposure_log_path = Path(self.config.output_dir) / "stream_exposure.jsonl"

        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()

        # Optional full-state checkpointing. Each save always writes the model-only file
        # steps_<N>_pytorch_model.pt; with save_full_state it also writes a steps_<N>/
        # directory with optimizer, scheduler and RNG state, which trainer.is_resume restores.
        # Unlike train_starvla.py's require_exact_resume, the data cursor is not restored:
        # each stream restarts its current epoch (with the saved epoch counter).
        self._save_full_state = bool(cfg.trainer.get("save_full_state", False))

        logger.info(
            "[gen-cotrain] loss binding: " + ", ".join(f"{s['name']}={sorted(s['loss_keys'])}" for s in self.streams)
        )
        if self._fuse_joint_vlm_forward:
            logger.info("[gen-cotrain] fused joint Qwen forward enabled")

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        self._save_initial_configs()

        if hasattr(self.config.trainer, "pretrained_checkpoint") and self.config.trainer.pretrained_checkpoint:
            pretrained_checkpoint = self.config.trainer.pretrained_checkpoint
            reload_modules = (
                self.config.trainer.reload_modules if hasattr(self.config.trainer, "reload_modules") else None
            )
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)

        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)

        # Prepare model + optimizer + every stream loader (in stream order) via Accelerate.
        prepared = self.setup_distributed_training(
            self.accelerator,
            self.model,
            self.optimizer,
            *[s["dataloader"] for s in self.streams],
        )
        self.model, self.optimizer = prepared[0], prepared[1]
        for stream, dl in zip(self.streams, prepared[2:], strict=True):
            stream["dataloader"] = dl
        self.vla_train_dataloader = self.primary["dataloader"]

        if self._save_full_state:
            # The scheduler is not accelerator.prepare()'d here, so save_state only
            # captures it if it is registered explicitly.
            self.accelerator.register_for_checkpointing(self.lr_scheduler)

        self._init_tracking()
        self._init_checkpointing()
        self._init_exposure_logging()

    def _init_exposure_logging(self):
        """Validate resume alignment, then open the exposure audit on rank zero.

        A preempted attempt may have logged steps beyond its latest complete
        checkpoint. Resume therefore keeps exactly the checkpointed prefix before
        appending; otherwise duplicate optimizer-step rows would make cumulative
        exposure ambiguous. Every rank validates the same shared file before rank
        zero performs the atomic prefix rewrite.
        """
        self._exposure_log_handle = None
        records = []
        if self._exposure_log_path.exists():
            with self._exposure_log_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid exposure record at {self._exposure_log_path}:{line_number}: {exc}"
                        ) from exc
                    expected_step = len(records) + 1
                    if record.get("schema_version") != "starvla.stream_exposure.v1":
                        raise ValueError(
                            f"Unsupported exposure schema at {self._exposure_log_path}:{line_number}: "
                            f"{record.get('schema_version')!r}"
                        )
                    if record.get("optimizer_step") != expected_step:
                        raise ValueError(
                            f"Exposure optimizer steps must be contiguous from 1; "
                            f"{self._exposure_log_path}:{line_number} has "
                            f"{record.get('optimizer_step')!r}, expected {expected_step}."
                        )
                    records.append(record)

        if self.completed_steps == 0:
            if records:
                raise FileExistsError(
                    f"Refusing to append a fresh run to non-empty exposure log {self._exposure_log_path}. "
                    "Use trainer.is_resume=true with a matching full-state checkpoint or choose a new run_id."
                )
        else:
            if len(records) < self.completed_steps:
                raise ValueError(
                    f"Exposure log {self._exposure_log_path} has {len(records)} complete step(s), "
                    f"but the resume checkpoint is at step {self.completed_steps}."
                )
            checkpoint_record = records[self.completed_steps - 1]
            labels = {stream["name"]: stream["exposure_name"] for stream in self.streams}
            logged_cumulative = checkpoint_record.get("cumulative_samples", {})
            expected_cumulative = {labels[name]: int(count) for name, count in self._cumulative_stream_samples.items()}
            if logged_cumulative != expected_cumulative:
                raise ValueError(
                    "Exposure log/checkpoint cumulative samples disagree at resume step "
                    f"{self.completed_steps}: log={logged_cumulative}, checkpoint={expected_cumulative}."
                )

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            if len(records) > self.completed_steps:
                retained = records[: self.completed_steps]
                temporary_path = self._exposure_log_path.with_name(
                    f".{self._exposure_log_path.name}.resume-{os.getpid()}.tmp"
                )
                with temporary_path.open("w", encoding="utf-8") as handle:
                    for record in retained:
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_path, self._exposure_log_path)
                logger.info(
                    f"[exposure] truncated {self._exposure_log_path} from {len(records)} "
                    f"to checkpoint step {self.completed_steps} before resume"
                )
            self._exposure_log_handle = open(self._exposure_log_path, "a", encoding="utf-8", buffering=1)
        self.accelerator.wait_for_everyone()

    def _save_initial_configs(self):
        """Save full config and accessed-config snapshot at the very start."""
        if not self.accelerator.is_main_process:
            return

        output_dir = Path(self.config.output_dir)
        if isinstance(self.config, AccessTrackedConfig):
            full_cfg = self.config.unwrap()
        else:
            full_cfg = self.config
        OmegaConf.save(full_cfg, output_dir / "config.full.yaml", resolve=True)
        logger.info(f"\U0001f4dd Full config saved at {output_dir / 'config.full.yaml'}")

        if isinstance(self.config, AccessTrackedConfig):
            self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
            logger.info(f"\U0001f4ca Accessed config snapshot saved at {output_dir / 'config.yaml'}")

    def _calculate_total_batch_size(self):
        """Validate the active micro-batch contract and return effective global batch."""
        configured_total = self.config.datasets.get("total_per_device_batch_size", None)
        configured_total = int(configured_total) if configured_total is not None else None

        total_micro = sum(self._stream_micro_batches.values())
        if configured_total is not None and total_micro != configured_total:
            raise ValueError(
                "co-training stream batches must sum exactly to "
                "datasets.total_per_device_batch_size; "
                f"resolved={self._stream_micro_batches} (sum={total_micro}), configured={configured_total}."
            )

        self.total_micro_batch_size = int(total_micro)
        effective_global_batch = (
            self.total_micro_batch_size
            * int(self.accelerator.num_processes)
            * int(self.accelerator.gradient_accumulation_steps)
        )
        requested_global_batch = self.config.trainer.get("global_batch_size", None)
        if requested_global_batch is not None and int(requested_global_batch) != effective_global_batch:
            raise ValueError(
                "co-training effective global batch does not match trainer.global_batch_size: "
                f"total_micro={self.total_micro_batch_size}, world_size={self.accelerator.num_processes}, "
                f"gradient_accumulation_steps={self.accelerator.gradient_accumulation_steps}, "
                f"effective={effective_global_batch}, requested={requested_global_batch}."
            )
        return effective_global_batch

    def _init_tracking(self):
        """Create the rank-0 metrics tracker (W&B or metrics.jsonl); never fails the run."""
        self.tracker = MetricsTracker(
            self.config,
            self.config.output_dir,
            group="vla-gen-cotrain",
            is_main_process=self.accelerator.is_main_process,
        )
        # Rendezvous after rank-0 tracker init so a slow W&B start cannot let the other
        # ranks reach the first collective alone and hit the NCCL watchdog.
        self.accelerator.wait_for_everyone()

    def _init_checkpointing(self):
        """Initialize the checkpoint directory and resume if asked to."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if not bool(getattr(self.config.trainer, "is_resume", False)):
            return

        resume_from = getattr(self.config.trainer, "resume_from_checkpoint", None)
        if not resume_from:
            raise ValueError(
                "trainer.is_resume=true requires trainer.resume_from_checkpoint to point at a "
                "checkpoints/steps_<N>/ full-state directory (written with trainer.save_full_state=true)."
            )
        self._load_checkpoint(resume_from)

    def _load_checkpoint(self, checkpoint_path):
        """Restore model/optimizer/scheduler/RNG and the step counter."""
        marker = os.path.join(checkpoint_path, COMPLETE_MARKER)
        if not os.path.isfile(marker):
            raise FileNotFoundError(
                f"{checkpoint_path} has no {COMPLETE_MARKER}. A directory without the marker is an "
                "interrupted save and must be inspected explicitly rather than silently resumed. "
                "Model-only files (steps_<N>_pytorch_model.pt) cannot be resumed from at all -- "
                "they carry no optimizer, scheduler or RNG state."
            )
        self.accelerator.load_state(checkpoint_path)

        with open(os.path.join(checkpoint_path, TRAINER_STATE_FILE)) as f:
            state = json.load(f)
        self.completed_steps = int(state["completed_steps"])
        # Restore per-stream epoch counters so the reshuffle seed keeps advancing
        # instead of replaying the epoch that was in flight when the run died.
        for stream in self.streams:
            epoch = state.get("stream_epochs", {}).get(stream["name"])
            if epoch is not None:
                stream["epoch_count"] = int(epoch)
        saved_exposure = state.get("cumulative_stream_samples", {})
        unknown_exposure_streams = set(saved_exposure).difference(self._cumulative_stream_samples)
        if unknown_exposure_streams:
            raise ValueError(
                "checkpoint exposure state contains streams absent from the current config: "
                f"{sorted(unknown_exposure_streams)}"
            )
        for name in self._cumulative_stream_samples:
            self._cumulative_stream_samples[name] = int(saved_exposure.get(name, 0))
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path} at step {self.completed_steps}")

    def _save_checkpoint(self):
        """Write ``steps_<N>_pytorch_model.pt`` and, with save_full_state, ``steps_<N>/``.

        The model-only file is what evaluation loads; the full-state directory holds the
        optimizer, scheduler and RNG state that a resume restores.
        """
        save_format = getattr(self.config.trainer, "save_format", "pt")
        if save_format not in {"pt", "safetensors"}:
            raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")

        checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
        full_state_dir = checkpoint_path if self._save_full_state else None

        # Make the exposure prefix durable before publishing checkpoint state that
        # claims the same completed_steps. This keeps resume validation one-way:
        # a complete checkpoint can never be newer than its audit log.
        if self.accelerator.is_main_process and self._exposure_log_handle is not None:
            self._exposure_log_handle.flush()
            os.fsync(self._exposure_log_handle.fileno())
        self.accelerator.wait_for_everyone()

        if full_state_dir is not None:
            # Every rank checks BEFORE rank 0 creates the directory. Letting only rank 0
            # raise would strand the others in the barrier below and turn a clear failure
            # into a hang; checking on a shared filesystem makes all ranks raise together.
            if os.path.exists(full_state_dir):
                raise FileExistsError(
                    f"Refusing to overwrite {full_state_dir}. A directory without "
                    f"{COMPLETE_MARKER} is an interrupted save and must be inspected or removed "
                    "explicitly; a completed one is immutable."
                )
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                os.makedirs(full_state_dir)
            self.accelerator.wait_for_everyone()
            # Collective under DeepSpeed: every rank must enter.
            self.accelerator.save_state(full_state_dir)

        # DeepSpeed state consolidation can also be collective, so every rank enters
        # get_state_dict even though only rank 0 writes the file.
        state_dict = self.accelerator.get_state_dict(self.model)
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, checkpoint_path + "_model.safetensors")
            else:
                torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")

            if full_state_dir is not None:
                trainer_state = {
                    "completed_steps": self.completed_steps,
                    "total_train_steps": self.total_train_steps,
                    "stream_epochs": {s["name"]: int(s["epoch_count"]) for s in self.streams},
                    "cumulative_stream_samples": dict(self._cumulative_stream_samples),
                }
                with open(os.path.join(full_state_dir, TRAINER_STATE_FILE), "w") as f:
                    json.dump(trainer_state, f, indent=2, sort_keys=True)
                    f.flush()
                    os.fsync(f.fileno())
                # The completion marker is written LAST and is the only proof that the
                # directory is whole; a crash mid-save leaves it absent.
                with open(os.path.join(full_state_dir, COMPLETE_MARKER), "w") as f:
                    json.dump({"completed_steps": self.completed_steps}, f)
                    f.flush()
                    os.fsync(f.fileno())

            summary_data = {"steps": self.completed_steps}
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")
            if full_state_dir is not None:
                self.accelerator.print(f"✅ Full-state checkpoint saved at {full_state_dir}")

            if isinstance(self.config, AccessTrackedConfig):
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)

        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """Send metrics to the tracker and the log every ``logging_frequency`` steps (rank 0)."""
        if (
            self.completed_steps % self.config.trainer.resolved_logging_frequency == 0
            and self.accelerator.is_main_process
        ):
            last_lrs = self.lr_scheduler.get_last_lr()
            for i, group in enumerate(self.optimizer.param_groups):
                group_name = group.get("name", str(i))
                metrics[f"learning_rate/{group_name}"] = last_lrs[i] if i < len(last_lrs) else last_lrs[-1]
            metrics["epoch"] = round(self.completed_steps / len(self.primary["dataloader"]), 2)
            self.tracker.log(metrics, step=self.completed_steps)
            logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    def _create_data_iterators(self):
        """Open every stream's training iterator and the robot validation iterator."""
        self.val_iter = None
        for s in self.loss_val_streams:
            s["iter"] = None
        for s in self.streams:
            s["iter"] = self._create_stream_iterator(s)
        if self.val_dataloader is not None:
            self.val_iter = iter(self.val_dataloader)

    def _create_stream_iterator(self, stream):
        block = self.config.datasets[stream["name"]]
        if bool(block.get("double_buffer_worker_generations", False)):
            prewarm_batches = int(block.get("worker_generation_prewarm_batches", 8))
            logger.info(
                f"[worker-double-buffer] stream={stream['name']} generation_batches={len(stream['dataloader'])} "
                f"prewarm_batches={prewarm_batches}"
            )
            return DoubleBufferedWorkerIterator(
                stream["dataloader"],
                epoch_counter=stream["epoch_count"],
                prewarm_batches=prewarm_batches,
                reset_fn=TrainerUtils._reset_dataloader,
            )
        return iter(stream["dataloader"])

    @staticmethod
    def _close_stream_iterator(stream):
        iterator = stream.get("iter")
        close = getattr(iterator, "close", None)
        if callable(close):
            close()
        stream["iter"] = None

    def _get_next_val_batch(self):
        """Get next hold-out validation batch (robot stream only; auto-reset)."""
        try:
            batch_vla = next(self.val_iter)
        except StopIteration:
            if not hasattr(self, "val_epoch_count"):
                self.val_epoch_count = 0
            self.val_iter, self.val_epoch_count = TrainerUtils._reset_dataloader(
                self.val_dataloader, self.val_epoch_count
            )
            batch_vla = next(self.val_iter)

        return batch_vla

    def _get_next_loss_val_batch(self, stream):
        """Get the next batch from an independent secondary validation stream."""
        if stream["iter"] is None:
            stream["iter"] = iter(stream["dataloader"])
        try:
            return next(stream["iter"])
        except StopIteration:
            stream["iter"], stream["epoch_count"] = TrainerUtils._reset_dataloader(
                stream["dataloader"],
                stream["epoch_count"],
            )
            return next(stream["iter"])

    def _get_next_stream_batch(self, stream):
        """Next batch from ``stream``, auto-resetting its iterator on exhaustion (per-stream
        epoch counter)."""
        if isinstance(stream["iter"], DoubleBufferedWorkerIterator):
            batch = next(stream["iter"])
            stream["epoch_count"] = stream["iter"].epoch
            return batch
        try:
            return next(stream["iter"])
        except StopIteration:
            stream["iter"], stream["epoch_count"] = TrainerUtils._reset_dataloader(
                stream["dataloader"], stream["epoch_count"]
            )
            return next(stream["iter"])

    def _next_train_substeps(self):
        """Return ``[(stream_descriptor, batch), ...]`` with one batch per stream (primary first)."""
        return [(stream, self._get_next_stream_batch(stream)) for stream in self.streams]

    def train(self):
        """Execute training loop."""
        self._log_training_config()
        self._create_data_iterators()
        progress_bar = tqdm(
            total=self.total_train_steps,
            initial=self.completed_steps,
            disable=not self.accelerator.is_local_main_process,
        )

        while self.completed_steps < self.total_train_steps:
            t_start_data = time.perf_counter()
            substeps = self._next_train_substeps()
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(substeps)
            t_end_model = time.perf_counter()

            # DeepSpeed decides accumulation boundaries; _train_step reports whether
            # an optimizer step actually happened. Fall back to sync_gradients.
            optimizer_stepped = step_metrics.pop("_optimizer_step", self.accelerator.sync_gradients)
            if optimizer_stepped:
                progress_bar.update(1)
                self.completed_steps += 1
                step_metrics = self._record_optimizer_exposure(step_metrics)

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    {
                        "data_times": f"{t_end_data - t_start_data:.3f}",
                        "model_times": f"{t_end_model - t_start_model:.3f}",
                    }
                )

            should_eval = optimizer_stepped and self.completed_steps % self.config.trainer.resolved_eval_interval == 0
            if should_eval:
                step_metrics = self.eval_action_model(step_metrics)
                if self.loss_val_streams:
                    step_metrics = self.eval_stream_losses(step_metrics)

            step_metrics["timing/data"] = t_end_data - t_start_data
            step_metrics["timing/model"] = t_end_model - t_start_model
            if optimizer_stepped:
                self._log_metrics(step_metrics)

            if (
                optimizer_stepped
                and self.completed_steps % self.config.trainer.resolved_save_interval == 0
                and self.completed_steps > 0
            ):
                self._save_checkpoint()
                distributed_barrier()

            if self.completed_steps >= self.total_train_steps:
                break

        self._finalize_training()

    def eval_action_model(self, step_metrics: dict | None = None) -> dict:
        """Attach the hold-out action MSE of the primary robot stream to ``step_metrics``.

        Averages the per-batch MSE over ``trainer.val_num_batches`` batches of the
        (un-sharded, deterministic) robot validation set; without a validation loader no
        metric is emitted. ``predict_action`` runs on every rank so the ranks stay in
        lockstep (and collective-safe); only the main process records the score.
        """
        if self.val_dataloader is None:
            return step_metrics
        num_batches = int(self.config.trainer.get("val_num_batches", 8))

        action_mses = []
        for _ in range(num_batches):
            examples = self._get_next_val_batch()
            output_dict = self.accelerator.unwrap_model(self.model).predict_action(examples=examples)
            if self.accelerator.is_main_process:
                raw_examples, _ = unpack_preprocessed_batch(examples)
                actions = np.array([example["action"] for example in raw_examples])
                action_mses.append(mean_action_mse(output_dict["normalized_actions"], actions))

        if self.accelerator.is_main_process:
            step_metrics["val/action_mse"] = float(np.mean(action_mses))

        distributed_barrier()
        return step_metrics

    def eval_stream_losses(self, step_metrics: dict | None = None) -> dict:
        """Evaluate raw losses on independent secondary validation streams.

        Each rank consumes the same unsharded batches and executes the same wrapped-model
        forwards. Only the main process records metrics. Forking a fixed RNG makes
        stochastic flow-matching losses comparable without perturbing training RNG.
        """
        if step_metrics is None:
            step_metrics = {}
        if not self.loss_val_streams:
            return step_metrics

        accelerator_device = self.accelerator.device
        rng_devices = []
        if accelerator_device.type == "cuda":
            rng_devices = [
                accelerator_device.index if accelerator_device.index is not None else torch.cuda.current_device()
            ]

        default_seed = int(self.config.seed) if hasattr(self.config, "seed") else 42
        val_seed = int(self.config.trainer.get("val_seed", default_seed))
        was_training = self.model.training
        metric_values = {}
        self.model.eval()
        try:
            for stream_index, stream in enumerate(self.loss_val_streams):
                block = self.config.datasets[stream["name"]]
                num_batches = int(
                    block.get(
                        "val_num_batches",
                        self.config.trainer.get("val_num_batches", 8),
                    )
                )
                if num_batches < 1:
                    raise ValueError(f"datasets.{stream['name']}.val_num_batches must be >= 1, " f"got {num_batches}")

                expected_keys = {key for key in stream["loss_keys"] if key.endswith("_loss")}
                if not expected_keys:
                    raise ValueError(
                        f"Validation stream {stream['name']!r} has no '*_loss' keys: " f"{sorted(stream['loss_keys'])}"
                    )

                totals = {key: 0.0 for key in expected_keys}
                sample_counts = {key: 0 for key in expected_keys}
                with torch.random.fork_rng(devices=rng_devices):
                    stream_seed = val_seed + stream_index
                    torch.manual_seed(stream_seed)
                    if rng_devices:
                        torch.cuda.manual_seed(stream_seed)

                    for _ in range(num_batches):
                        examples = self._get_next_loss_val_batch(stream)
                        batch_size = get_batch_size(examples)
                        if batch_size < 1:
                            raise RuntimeError(f"Validation stream {stream['name']!r} returned an empty batch")
                        with torch.inference_mode(), self.accelerator.autocast():
                            output_dict = self.model.forward(
                                examples,
                                loss_keys=stream["loss_keys"],
                            )

                        missing = expected_keys.difference(output_dict)
                        if missing:
                            raise RuntimeError(
                                f"Validation stream {stream['name']!r} did not return "
                                f"requested losses {sorted(missing)}; got {sorted(output_dict)}"
                            )
                        for key in expected_keys:
                            value = output_dict[key]
                            if not isinstance(value, torch.Tensor):
                                raise TypeError(
                                    f"Validation loss {key!r} from stream "
                                    f"{stream['name']!r} must be a tensor, got "
                                    f"{type(value).__name__}"
                                )
                            if self.accelerator.is_main_process:
                                totals[key] += float(value.detach().float().item()) * batch_size
                                sample_counts[key] += batch_size

                if self.accelerator.is_main_process:
                    stream_label = stream["name"].replace("_data", "")
                    for key in expected_keys:
                        metric_values[f"val/{stream_label}/{key}"] = totals[key] / sample_counts[key]
        finally:
            if was_training:
                self.model.train()
            else:
                self.model.eval()

        if self.accelerator.is_main_process:
            step_metrics.update(metric_values)
        distributed_barrier()
        return step_metrics

    def _log_training_config(self):
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info("  Co-training mode = joint (all streams every step)")
            logger.info(f"  Total optimization steps = {self.total_train_steps}")
            for s in self.streams:
                logger.info(
                    f"  {s['name']} ({s['exposure_name']}): planned_ratio={self._planned_ratios[s['name']]}, "
                    f"resolved_micro={self._stream_micro_batches[s['name']]}, "
                    f"resolved_fraction={self._stream_micro_batches[s['name']] / self.total_micro_batch_size:.6f}"
                )
            logger.info(f"  Gradient accumulation steps = {self.accelerator.gradient_accumulation_steps}")
            logger.info(f"  Total micro batch size (all active streams) = {self.total_micro_batch_size}")
            logger.info(f"  Actual global batch size = {self.total_batch_size}")

    def _stream_loss(self, output_dict):
        """Loss of one stream's mini-batch: ``sum(loss_scale[k] * loss_k)`` over its bound losses.

        Each loss is already averaged within its own mini-batch, and the streams' losses are
        summed (their gradients accumulate into one optimizer step), so the sampling ratio
        changes how many samples of each source are seen per step but not the relative
        weight of the objectives.
        """
        loss_scale = self.config.trainer.get("loss_scale", {}) if hasattr(self.config.trainer, "get") else {}
        return sum(
            v * float(loss_scale.get(k, 1.0))
            for k, v in output_dict.items()
            if isinstance(v, torch.Tensor) and k.endswith("_loss")
        )

    @staticmethod
    def _loss_items(output_dict, prefix=""):
        return {
            f"{prefix}{k}": v.item()
            for k, v in output_dict.items()
            if isinstance(v, torch.Tensor) and k.endswith("_loss")
        }

    def _record_optimizer_exposure(self, step_metrics):
        """Commit pending micro-step counts and emit one optimizer-step audit record."""
        step_samples = dict(self._pending_stream_samples)
        actual_global_batch = sum(step_samples.values())
        if actual_global_batch != self.total_batch_size:
            raise RuntimeError(
                "realized stream exposure does not match the effective global batch: "
                f"samples={step_samples} (sum={actual_global_batch}), expected={self.total_batch_size}."
            )

        for name, count in step_samples.items():
            self._cumulative_stream_samples[name] += int(count)
            self._pending_stream_samples[name] = 0

        cumulative_total = sum(self._cumulative_stream_samples.values())
        labels = {stream["name"]: stream["exposure_name"] for stream in self.streams}

        def by_label(values):
            return {labels[name]: values[name] for name in labels}

        step_fractions = {name: count / actual_global_batch for name, count in step_samples.items()}
        cumulative_fractions = {
            name: count / cumulative_total for name, count in self._cumulative_stream_samples.items()
        }
        record = {
            "schema_version": "starvla.stream_exposure.v1",
            "optimizer_step": int(self.completed_steps),
            "planned_ratio": by_label(self._planned_ratios),
            "planned_fraction": by_label(self._planned_fractions),
            "resolved_micro_batch": by_label(self._stream_micro_batches),
            "gradient_accumulation_steps": int(self.accelerator.gradient_accumulation_steps),
            "actual_global_batch_size": int(actual_global_batch),
            "samples_this_optimizer_step": by_label(step_samples),
            "cumulative_samples": by_label(self._cumulative_stream_samples),
            "realized_fraction_this_optimizer_step": by_label(step_fractions),
            "realized_fraction_cumulative": by_label(cumulative_fractions),
        }
        if self._exposure_log_handle is not None:
            self._exposure_log_handle.write(json.dumps(record, sort_keys=True) + "\n")

        step_metrics["exposure/actual_global_batch_size"] = actual_global_batch
        step_metrics["exposure/gradient_accumulation_steps"] = int(self.accelerator.gradient_accumulation_steps)
        for stream in self.streams:
            name = stream["name"]
            label = stream["exposure_name"]
            step_metrics[f"exposure/{label}/planned_ratio"] = self._planned_ratios[name]
            step_metrics[f"exposure/{label}/planned_fraction"] = self._planned_fractions[name]
            step_metrics[f"exposure/{label}/resolved_micro_batch"] = self._stream_micro_batches[name]
            step_metrics[f"exposure/{label}/samples_this_optimizer_step"] = step_samples[name]
            step_metrics[f"exposure/{label}/cumulative_samples"] = self._cumulative_stream_samples[name]
            step_metrics[f"exposure/{label}/realized_fraction_this_optimizer_step"] = step_fractions[name]
            step_metrics[f"exposure/{label}/realized_fraction_cumulative"] = cumulative_fractions[name]
        return step_metrics

    def _train_step(self, substeps):
        """Single co-training step over one or more source sub-steps.

        Each sub-step is ``(stream_descriptor, batch)``. Runtime batch sizes must match
        their resolved config, and joint stream means are weighted by their actual
        share of this micro-step before backward. Gradients accumulate across streams,
        then one optimizer/LR step is attempted.
        """
        log_dict = {}
        total_loss = 0.0

        resolved_substeps = []
        for stream, batch in substeps:
            batch_size = get_batch_size(batch)
            expected_batch_size = self._stream_micro_batches[stream["name"]]
            if batch_size != expected_batch_size:
                raise RuntimeError(
                    f"stream {stream['name']!r} returned runtime batch {batch_size}, "
                    f"but its resolved per-device batch is {expected_batch_size}; "
                    "partial batches would change both loss weights and effective global batch."
                )
            resolved_substeps.append((stream, batch, batch_size))

        runtime_total = sum(batch_size for _, _, batch_size in resolved_substeps)
        if runtime_total != self.total_micro_batch_size:
            raise RuntimeError(
                "active runtime stream batches must equal the resolved total micro batch; "
                f"runtime={runtime_total}, resolved={self.total_micro_batch_size}."
            )

        for stream, _, batch_size in resolved_substeps:
            self._pending_stream_samples[stream["name"]] += batch_size * self.accelerator.num_processes

        def fused_forward_and_loss():
            outputs = self.model.forward(
                cotrain_batches=[(batch, stream["loss_keys"]) for stream, batch, _ in resolved_substeps]
            )
            if not isinstance(outputs, (tuple, list)) or len(outputs) != len(resolved_substeps):
                raise RuntimeError(
                    "Fused cotrain framework must return one loss dictionary per stream; "
                    f"got {type(outputs).__name__} with length "
                    f"{len(outputs) if isinstance(outputs, (tuple, list)) else 'n/a'}."
                )
            combined_loss = None
            for (stream, _, _), output in zip(resolved_substeps, outputs, strict=True):
                stream_loss = self._stream_loss(output)
                combined_loss = stream_loss if combined_loss is None else combined_loss + stream_loss
                log_dict.update(self._loss_items(output, prefix=stream["prefix"]))
            return combined_loss

        # ---- DeepSpeed (ZeRO stage 2/3) path: drive accumulation via the engine. ----
        if hasattr(self.model, "is_gradient_accumulation_boundary") and hasattr(self.model, "backward"):
            if self._fuse_joint_vlm_forward:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = fused_forward_and_loss()
                self.model.backward(loss)
                optimizer_stepped = bool(self.model.is_gradient_accumulation_boundary())
                self.model.step()
                if optimizer_stepped:
                    self.lr_scheduler.step()

                log_dict["total_loss"] = float(loss.item())
                log_dict["_optimizer_step"] = optimizer_stepped
                return log_dict

            for stream, batch, _ in resolved_substeps:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = self.model.forward(batch, loss_keys=stream["loss_keys"])
                    loss = self._stream_loss(out)
                self.model.backward(loss)
                log_dict.update(self._loss_items(out, prefix=stream["prefix"]))
                total_loss += float(loss.item())

            optimizer_stepped = bool(self.model.is_gradient_accumulation_boundary())
            self.model.step()
            if optimizer_stepped:
                self.lr_scheduler.step()

            log_dict["total_loss"] = total_loss
            log_dict["_optimizer_step"] = optimizer_stepped
            return log_dict

        # ---- Non-DeepSpeed (DDP) path: accelerator.accumulate handles no_sync. ----
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()

            if self._fuse_joint_vlm_forward:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = fused_forward_and_loss()
                self.accelerator.backward(loss)
                total_loss += float(loss.item())
            else:
                for stream, batch, _ in resolved_substeps:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        out = self.model.forward(batch, loss_keys=stream["loss_keys"])
                        loss = self._stream_loss(out)
                    self.accelerator.backward(loss)
                    log_dict.update(self._loss_items(out, prefix=stream["prefix"]))
                    total_loss += float(loss.item())

            if self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            self.optimizer.step()
            if self.accelerator.sync_gradients:
                self.lr_scheduler.step()

            log_dict["total_loss"] = total_loss
            log_dict["_optimizer_step"] = self.accelerator.sync_gradients

        return log_dict

    def _finalize_training(self):
        """Training end processing."""
        for stream in self.streams:
            self._close_stream_iterator(stream)

        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, os.path.join(final_checkpoint, "model.safetensors"))
            elif save_format == "pt":
                torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")

        self.tracker.finish()

        if self._exposure_log_handle is not None:
            self._exposure_log_handle.close()
            self._exposure_log_handle = None

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    logger.info("VLA Gen Co-Training :: Warming Up")

    # Resolve the absolute-or-ratio schedule on the raw config, before anything reads it.
    validate_cotrain_options(cfg)
    resolve_schedule(cfg)
    stream_names = discover_stream_blocks(cfg)

    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    accelerator = build_accelerator(cfg)
    accelerator.print(accelerator.state)

    output_dir = setup_directories(cfg=cfg)
    # Build every rank from the same deterministic initialization. prepare_training()
    # later resets to seed + rank so runtime stochasticity remains rank-specific and
    # independent of how much RNG a particular model variant consumed while building.
    model_init_seed = cfg.seed if hasattr(cfg, "seed") else 3047
    set_seed(model_init_seed)
    vla = build_framework(cfg)
    worker_preprocess_enabled = any(
        bool(cfg.datasets[name].get("worker_preprocess_qwen", False)) for name in stream_names
    )
    batch_preprocessor = None
    if worker_preprocess_enabled:
        make_preprocessor = getattr(vla.qwen_vl_interface, "make_worker_preprocessor", None)
        if not callable(make_preprocessor):
            raise ValueError(
                "datasets.*.worker_preprocess_qwen=true, but the selected VLM does not provide "
                "make_worker_preprocessor()."
            )
        batch_preprocessor = make_preprocessor()
        logger.info("[worker-preprocess] using the selected VLM's canonical processor in DataLoader workers")
    streams, val_dataloader, loss_val_streams = prepare_data(
        cfg=cfg,
        accelerator=accelerator,
        output_dir=output_dir,
        batch_preprocessor=batch_preprocessor,
    )
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    trainer = VLAGenCotrainer(
        cfg=cfg,
        model=vla,
        streams=streams,
        val_dataloader=val_dataloader,
        loss_val_streams=loss_val_streams,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    trainer.prepare_training()
    try:
        trainer.train()
    except BaseException:
        # Skip the closing barrier: the peers are wherever the failure left them, so
        # synchronizing here would block until the NCCL watchdog fires.
        logger.exception("training failed; releasing the process group before exit")
        raise
    else:
        logger.info("... and that's all, folks!")
        distributed_barrier()
    finally:
        release_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Robot + video co-training. Extra `--dotted.key value` arguments override the config."
    )
    parser.add_argument("--config_yaml", type=str, required=True, help="Path to the training YAML config.")
    args, clipargs = parser.parse_known_args()

    cfg = load_config_with_bases(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    cfg = apply_config_compat(cfg)

    # Cross-dataset normalization on the raw OmegaConf (before wrap_config in main()):
    #   1) align_dataset_configs: copy datasets.shared.* into every stream so the
    #      observation / generation-target policy cannot diverge, and derive each stream's
    #      future-frame offsets from framework.gen_model.gen_horizon_seconds and its own
    #      dataset_fps.
    #   2) resolve_stream_batch_sizes: split datasets.total_per_device_batch_size across
    #      all streams by their sampling_ratio.
    cfg = align_dataset_configs(cfg)
    resolve_stream_batch_sizes(cfg, discover_stream_blocks(cfg))

    cfg.config_yaml = args.config_yaml

    main(cfg)
