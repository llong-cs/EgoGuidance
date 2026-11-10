# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""Single-stream VLA training entry (e.g. LIBERO fine-tuning).

Built directly on PyTorch + Accelerate + DeepSpeed, keeping the loop explicit. Launch it with
``bash scripts/train.sh <config.yaml> [--dotted.key value ...]``.

Checkpointing has two modes, selected by ``trainer.require_exact_resume``:

* ``false``: model-only files ``checkpoints/steps_<N>_pytorch_model.pt``; ``trainer.is_resume``
  reloads the latest one and fast-forwards the LR schedule.
* ``true``: full-state directories ``checkpoints/steps_<N>/`` (model, optimizer, scheduler, RNG
  and the exact data cursor) published atomically with a completion marker; ``trainer.is_resume``
  restores the exact next training sample. The data pipeline must prove index-deterministic
  sampling (see ``training_state.build_data_resume_contract``).
"""

# Standard Library
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import random
import re
import resource
import shutil
import subprocess
import time
from pathlib import Path
from typing import Tuple

# Third-Party Libraries
import numpy as np
import torch
import torch.distributed as dist
from accelerate.logging import get_logger
from accelerate.utils import broadcast_object_list, set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local Modules
from starVLA.dataloader import (
    align_dataset_configs,
    build_dataloader,
    resolve_stream_batch_sizes,
    unpack_preprocessed_batch,
)
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat, load_config_with_bases
from starVLA.training.distributed import distributed_barrier
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.tracking import MetricsTracker
from starVLA.training.trainer_utils.trainer_tools import (
    LAUNCHER_LOG_DIR,
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
    TrainingHorizon,
    build_data_resume_contract,
    build_trainer_state,
    config_fingerprint,
    describe_code_change,
    find_latest_complete_checkpoint,
    is_complete_checkpoint,
    is_resumable_checkpoint,
    load_complete_checkpoint_metadata,
    resolve_progress_interval,
    resolve_sampling_position,
    resolve_training_horizon,
    validate_resume_state,
)
from starVLA.training.trainer_utils.transfer_initialization import (
    add_dataset_statistics_identity,
    load_transfer_checkpoint,
    validate_transfer_checkpoint,
    validate_transfer_options,
    write_initialization_manifest,
)

# Optional Ascend NPU support: maps CUDA calls to NPU when torch_npu is installed.
try:
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
except ImportError:
    pass

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = get_logger(__name__)

# Written once into a fresh exact-resume run directory; a full resume requires it.
RUN_OWNER_FILE = "run.owner.json"


def _rank0_consensus(accelerator, payload):
    """Broadcast one rank-0 filesystem decision to every process."""

    if int(accelerator.num_processes) == 1:
        return payload
    objects = [payload if accelerator.is_main_process else None]
    broadcast_object_list(objects, from_process=0)
    return objects[0]


def _assert_rank_state_consensus(accelerator, local_state: dict, label: str) -> dict:
    """Fail on every rank before a collective if step accounting has diverged."""

    expected = _rank0_consensus(
        accelerator,
        local_state if accelerator.is_main_process else None,
    )
    if int(accelerator.num_processes) == 1:
        return expected
    mismatch = torch.tensor(
        [0 if local_state == expected else 1],
        dtype=torch.int64,
        device=accelerator.device,
    )
    mismatch_count = accelerator.reduce(mismatch, reduction="sum")
    if int(mismatch_count.item()) != 0:
        raise RuntimeError(
            f"{label} requires identical progress counters on all ranks; "
            f"rank-0={expected}, local={local_state}, "
            f"mismatched_ranks={int(mismatch_count.item())}."
        )
    return expected


def _raise_if_any_rank_failed(
    accelerator,
    local_failed,
    label: str,
    local_detail: str,
    exc_type=RuntimeError,
) -> None:
    """Make a hot-path health failure visible on every rank before proceeding.

    The normal path is one scalar reduction. Rank details are gathered only
    after at least one rank reports a failure, so no healthy rank can enter the
    next backward/optimizer collective while another raises locally.
    """

    if isinstance(local_failed, torch.Tensor):
        if local_failed.numel() != 1:
            raise ValueError(f"{label}: local_failed must be scalar")
        local_flag = local_failed.detach().to(
            device=accelerator.device,
            dtype=torch.int32,
        )
    else:
        local_flag = torch.tensor(
            int(bool(local_failed)),
            dtype=torch.int32,
            device=accelerator.device,
        )
    local_flag = local_flag.reshape(1)
    global_flag = local_flag.clone()
    if int(accelerator.num_processes) > 1:
        dist.all_reduce(global_flag, op=dist.ReduceOp.MAX)
    if int(global_flag.item()) == 0:
        return

    local_payload = str(local_detail) if int(local_flag.item()) else None
    if int(accelerator.num_processes) > 1:
        details = [None] * int(accelerator.num_processes)
        dist.all_gather_object(details, local_payload)
    else:
        details = [local_payload]
    failures = [f"rank {rank}: {detail}" for rank, detail in enumerate(details) if detail is not None]
    raise exc_type(f"{label} failed on one or more ranks: {' | '.join(failures)}")


def _all_rank_result_consensus(accelerator, local_result: dict, label: str) -> list[dict]:
    """Collect per-rank validation results and make every rank fail together."""

    if int(accelerator.num_processes) == 1:
        results = [local_result]
    else:
        results = [None] * int(accelerator.num_processes)
        dist.all_gather_object(results, local_result)
    errors = [f"rank {rank}: {result.get('error')}" for rank, result in enumerate(results) if result.get("error")]
    digests = {result.get("digest") for result in results if not result.get("error")}
    if None in digests:
        errors.append("one or more ranks returned no validation digest")
    if errors or len(digests) != 1:
        detail = " | ".join(errors) if errors else f"rank digests disagree: {sorted(digests)}"
        raise RuntimeError(f"{label} failed rank-consensus validation: {detail}")
    return results


def setup_directories(cfg, accelerator) -> Path:
    """Create ``<run_root_dir>/<run_id>``, with atomic ownership for exact-resume runs.

    The ``logs/`` subdirectory written by scripts/train.sh is ignored when checking that a
    fresh exact-resume run directory is empty.
    """

    output_dir = Path(resolve_output_dir(cfg))
    is_resume = bool(cfg.trainer.get("is_resume", False))
    exact_resume_enabled = bool(cfg.trainer.get("require_exact_resume", False))

    decision = None
    if accelerator.is_main_process:
        try:
            if not exact_resume_enabled:
                if cfg.trainer.get("resume_from_checkpoint", None):
                    raise ValueError(
                        "trainer.resume_from_checkpoint requires "
                        "trainer.require_exact_resume=true. Use "
                        "trainer.pretrained_checkpoint for model-only initialization."
                    )
                checkpoint_dir = output_dir / "checkpoints"
                full_state_directories = (
                    sorted(
                        path.name
                        for path in checkpoint_dir.iterdir()
                        if re.fullmatch(r"steps_\d+", path.name) and path.is_dir()
                    )
                    if is_resume and checkpoint_dir.is_dir()
                    else []
                )
                if full_state_directories:
                    raise RuntimeError(
                        "Model-only resume found full-state checkpoint directories "
                        f"{full_state_directories[-3:]} in {checkpoint_dir}. Set "
                        "trainer.require_exact_resume=true to resume from them."
                    )
                # Model-only mode reuses an existing run directory (resume or re-run).
                os.makedirs(output_dir, exist_ok=True)
            elif is_resume:
                if not output_dir.is_dir():
                    raise RuntimeError(
                        f"Full resume requires the original run directory, but {output_dir} does not exist."
                    )
                if not (output_dir / "config.full.yaml").is_file():
                    raise RuntimeError(
                        f"Full resume requires {output_dir / 'config.full.yaml'}; "
                        "refusing to treat an unowned directory as this run."
                    )
                if not (output_dir / RUN_OWNER_FILE).is_file():
                    raise RuntimeError(
                        f"Full resume requires the atomic ownership record " f"{output_dir / RUN_OWNER_FILE}."
                    )
                with open(output_dir / RUN_OWNER_FILE) as f:
                    owner = json.load(f)
                if owner.get("version") != 1 or owner.get("run_id") != str(cfg.run_id):
                    raise RuntimeError(f"Run ownership record does not match run_id={cfg.run_id!r}: {owner!r}.")
            else:
                existing = (
                    sorted(path.name for path in output_dir.iterdir() if path.name != LAUNCHER_LOG_DIR)
                    if output_dir.exists()
                    else []
                )
                if existing:
                    raise RuntimeError(
                        f"Fresh exact-resume run directory {output_dir} is not empty ({existing[:5]}). "
                        "Choose a new run_id, or set trainer.is_resume=true to continue this run."
                    )
                os.makedirs(output_dir, exist_ok=True)
                owner_path = output_dir / RUN_OWNER_FILE
                owner_fd = os.open(owner_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(owner_fd, "w") as f:
                    json.dump(
                        {
                            "version": 1,
                            "run_id": str(cfg.run_id),
                            "created_unix": time.time(),
                        },
                        f,
                        sort_keys=True,
                    )
                    f.flush()
                    os.fsync(f.fileno())
            decision = {"error": None}
        except Exception as exc:
            decision = {"error": f"{type(exc).__name__}: {exc}"}
    decision = _rank0_consensus(accelerator, decision)
    if decision["error"]:
        raise RuntimeError(decision["error"])

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)
    accelerator.wait_for_everyone()

    return output_dir


def _checkpoint_payload_manifest(
    checkpoint_path: str,
    portable_name: str,
    *,
    world_size: int,
    backend: str,
) -> tuple[dict, list, dict]:
    """Inventory every payload file before atomically publishing completeness."""

    payload_files = {}
    for root, _, filenames in os.walk(checkpoint_path):
        for filename in filenames:
            if filename == COMPLETE_MARKER or filename.startswith(f".{COMPLETE_MARKER}.tmp"):
                continue
            full_path = os.path.join(root, filename)
            if not os.path.isfile(full_path):
                continue
            relative_path = os.path.relpath(full_path, checkpoint_path)
            size = os.path.getsize(full_path)
            if size <= 0:
                raise RuntimeError(f"Checkpoint payload {relative_path!r} is empty; refusing to publish completeness.")
            payload_files[relative_path] = size

    required = {portable_name, TRAINER_STATE_FILE}
    missing = sorted(required.difference(payload_files))
    if missing:
        raise RuntimeError(f"Checkpoint is missing required payloads before publication: {missing}.")
    accelerate_payload_files = sorted(set(payload_files).difference(required))
    if not accelerate_payload_files:
        raise RuntimeError(
            "Accelerate save_state produced no model/optimizer/RNG/custom scheduler payload; "
            "refusing to publish a model-only checkpoint as full state."
        )

    rng_files = [f"random_states_{rank}.pkl" for rank in range(world_size)]
    scheduler_files = ["custom_checkpoint_0.pkl"]
    missing_control = sorted(set(rng_files + scheduler_files).difference(accelerate_payload_files))
    if missing_control:
        raise RuntimeError(
            "Accelerate full state is missing required per-rank RNG or custom scheduler " f"payloads: {missing_control}."
        )

    if backend == "DEEPSPEED":
        if "latest" not in accelerate_payload_files:
            raise RuntimeError("DeepSpeed checkpoint has no root `latest` tag file.")
        with open(os.path.join(checkpoint_path, "latest")) as f:
            deepspeed_tag = f.read().strip()
        if deepspeed_tag != "pytorch_model":
            raise RuntimeError(
                f"DeepSpeed checkpoint tag is {deepspeed_tag!r}; expected Accelerate tag 'pytorch_model'."
            )
        model_state_files = sorted(
            path
            for path in accelerate_payload_files
            if path.startswith("pytorch_model/") and path.endswith("model_states.pt")
        )
        optimizer_state_files = sorted(
            path
            for path in accelerate_payload_files
            if path.startswith("pytorch_model/") and path.endswith("optim_states.pt")
        )
        if not model_state_files or len(optimizer_state_files) < world_size:
            raise RuntimeError(
                "DeepSpeed checkpoint is incomplete: expected model state plus at least one "
                f"optimizer shard per rank; model={model_state_files}, "
                f"optimizer_shards={len(optimizer_state_files)}, world_size={world_size}."
            )
    elif backend in {"MULTI_GPU", "NO"}:
        model_state_files = [
            path for path in ("model.safetensors", "pytorch_model.bin") if path in accelerate_payload_files
        ]
        optimizer_state_files = [
            path for path in accelerate_payload_files if re.fullmatch(r"optimizer(?:_\d+)?\.bin", path)
        ]
        if not model_state_files or not optimizer_state_files:
            raise RuntimeError(
                f"Accelerate {backend} checkpoint is missing model/optimizer payload: "
                f"model={model_state_files}, optimizer={optimizer_state_files}."
            )
    else:
        raise RuntimeError(
            f"Full-state checkpoint layout validation is not implemented for backend {backend!r}; "
            "refusing to claim exact resume."
        )

    state_layout = {
        "backend": backend,
        "world_size": world_size,
        "rng_files": rng_files,
        "scheduler_files": scheduler_files,
        "model_state_files": model_state_files,
        "optimizer_state_files": optimizer_state_files,
    }
    control_files = [TRAINER_STATE_FILE, *rng_files, *scheduler_files]
    if backend == "DEEPSPEED":
        state_layout["deepspeed_tag_file"] = "latest"
        state_layout["deepspeed_tag"] = deepspeed_tag
        control_files.append("latest")
    control_sha256 = {}
    for relative_path in control_files:
        hasher = hashlib.sha256()
        with open(os.path.join(checkpoint_path, relative_path), "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(chunk)
        control_sha256[relative_path] = hasher.hexdigest()
    state_layout["control_sha256"] = control_sha256
    return (
        dict(sorted(payload_files.items())),
        accelerate_payload_files,
        state_layout,
    )


def _atomic_write_json(path: str, payload: dict) -> None:
    """Publish a small JSON record atomically after flushing its bytes."""

    tmp_path = os.path.join(os.path.dirname(path), f".{os.path.basename(path)}.tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


@contextlib.contextmanager
def _preserve_host_rng_state():
    """Keep a resume-only worker spawn from advancing model-side host RNG."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_device = torch.cuda.current_device() if torch.cuda.is_available() else None
    cuda_state = torch.cuda.get_rng_state(cuda_device) if cuda_device is not None else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device=cuda_device)


def _inspect_data_resume_contract(cfg, dataloader) -> dict:
    """Prove whether this loader supports index-only mid-epoch reconstruction."""

    mixture = dataloader.dataset
    transform_names_by_dataset = []
    subdatasets = []
    for dataset in list(getattr(mixture, "datasets", [])):
        transforms = getattr(getattr(dataset, "transforms", None), "transforms", None)
        if transforms is None:
            transform_names = ["<uninspectable-transform-pipeline>"]
        else:
            transform_names = [type(transform).__name__ for transform in transforms]
        transform_names_by_dataset.append(transform_names)
        subdatasets.append(
            {
                "dataset_name": str(getattr(dataset, "dataset_name", type(dataset).__name__)),
                "transform_types": transform_names,
                "video_keys": _jsonable(getattr(dataset, "modality_keys", {}).get("video", [])),
                "action_keys": _jsonable(getattr(dataset, "modality_keys", {}).get("action", [])),
                "video_override_provenance": _jsonable(getattr(dataset, "video_override_provenance", None)),
            }
        )

    gen_cfg = cfg.framework.gen_model
    contract = build_data_resume_contract(
        dataset_type=type(mixture).__name__,
        mode=str(getattr(mixture, "mode", "")),
        sample_error_policy=str(getattr(mixture, "sample_error_policy", "resample")),
        sampler_type=type(dataloader.sampler).__name__,
        drop_last=bool(getattr(dataloader, "drop_last", False)),
        dataset_py=str(cfg.datasets.vla_data.dataset_py),
        generation_enabled=bool(gen_cfg.get("enable", False)),
        transform_names_by_dataset=transform_names_by_dataset,
    )
    contract["subdatasets"] = subdatasets
    contract["dataset_seed"] = int(getattr(mixture, "seed", -1))
    contract["sampling"] = _jsonable(getattr(mixture, "sampling_provenance", None))
    return contract


def build_worker_batch_preprocessor(cfg, vla):
    """Build the selected Qwen processor wrapper only when the stream opts in."""

    if not bool(cfg.datasets.vla_data.get("worker_preprocess_qwen", False)):
        return None
    make_preprocessor = getattr(vla.qwen_vl_interface, "make_worker_preprocessor", None)
    if not callable(make_preprocessor):
        raise ValueError(
            "datasets.vla_data.worker_preprocess_qwen=true, but the selected VLM "
            "does not provide make_worker_preprocessor()."
        )
    logger.info("[worker-preprocess] using the selected VLM's canonical processor in " "DataLoader workers")
    return make_preprocessor()


def prepare_data(cfg, accelerator, output_dir, batch_preprocessor=None) -> "Tuple[DataLoader, DataLoader | None]":
    """Prepare VLA training data (+ optional validation loader)."""
    vla_train_dataloader = None
    val_dataloader = None
    result = None
    try:
        logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
        vla_train_dataloader = build_dataloader(
            cfg=cfg,
            dataset_py=cfg.datasets.vla_data.dataset_py,
            world_size=int(accelerator.num_processes),
            batch_preprocessor=batch_preprocessor,
        )

        # Validation uses either the dataset's official split or our deterministic hold-out.
        # Not accelerator.prepare'd (kept un-sharded) so every rank iterates the same
        # deterministic (mode="val") val batches — the rank-0 score is then global.
        if (
            cfg.datasets.vla_data.get("use_official_splits", False)
            or cfg.datasets.vla_data.get("val_episode_fraction", None) is not None
        ):
            logger.info("Creating VLA validation loader (split='val')")
            val_dataloader = build_dataloader(
                cfg=cfg,
                dataset_py=cfg.datasets.vla_data.dataset_py,
                split="val",
                # Validation is deliberately not accelerator.prepare'd: every rank
                # iterates the same complete deterministic loader.
                world_size=1,
                batch_preprocessor=batch_preprocessor,
            )
        contract = {
            "train_length": len(vla_train_dataloader.dataset),
            "val_length": len(val_dataloader.dataset) if val_dataloader is not None else None,
        }
        result = {
            "error": None,
            "digest": hashlib.sha256(json.dumps(contract, sort_keys=True).encode("utf-8")).hexdigest(),
        }
    except Exception as exc:
        result = {
            "error": f"{type(exc).__name__}: {exc}",
            "digest": None,
        }
    _all_rank_result_consensus(accelerator, result, "DataLoader construction")

    accelerator.dataloader_config.dispatch_batches = False
    distributed_barrier()
    return vla_train_dataloader, val_dataloader


def _resolved_split_contract(cfg) -> dict:
    """Return the data-selection fields that must remain equal across resume."""

    block = cfg.datasets.vla_data
    return {
        "data_mix": block.get("data_mix", None),
        "split": block.get("split", "train"),
        "use_official_splits": bool(block.get("use_official_splits", False)),
        "val_episode_fraction": block.get("val_episode_fraction", None),
        "episode_fraction": block.get("episode_fraction", None),
        "split_seed": block.get("seed", 42),
        "training_seed": cfg.get("seed", 42),
        "tail_policy": block.get("tail_policy", None),
        "sampling_strategy": block.get("sampling_strategy", "weighted_with_replacement"),
    }


def _git_output(repo_root: Path, *args) -> bytes:
    command = ["git", *args]
    try:
        return subprocess.run(
            command,
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        ).stdout
    except OSError as exc:
        raise RuntimeError(
            f"Cannot execute Git provenance command in {repo_root}: " f"command={command!r}, error={exc}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"").decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"Git provenance command failed in {repo_root}: "
            f"command={command!r}, returncode={exc.returncode}, stderr={stderr!r}"
        ) from exc


def _code_provenance() -> dict:
    """Record the Git commit plus a hash of all uncommitted changes, for reference only.

    Returns ``None`` fields when the code is not a Git checkout or Git is unavailable.
    """

    repo_root = Path(__file__).resolve().parents[2]
    try:
        return _git_code_provenance(repo_root)
    except (RuntimeError, OSError) as exc:
        logger.warning(f"Code provenance unavailable (not recorded): {exc}")
        return {"git_commit": None, "git_dirty": None, "git_dirty_sha256": None}


def _git_code_provenance(repo_root: Path) -> dict:
    commit = _git_output(repo_root, "rev-parse", "HEAD").decode().strip()
    status = _git_output(repo_root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    diff = _git_output(repo_root, "diff", "--binary", "HEAD", "--")
    untracked = _git_output(repo_root, "ls-files", "--others", "--exclude-standard", "-z")

    dirty_hasher = hashlib.sha256()
    dirty_hasher.update(status)
    dirty_hasher.update(diff)
    for raw_path in filter(None, untracked.split(b"\0")):
        relative_path = raw_path.decode("utf-8", errors="surrogateescape")
        file_path = repo_root / relative_path
        dirty_hasher.update(raw_path)
        if file_path.is_file():
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    dirty_hasher.update(chunk)
    return {
        "git_commit": commit,
        "git_dirty": bool(status),
        "git_dirty_sha256": dirty_hasher.hexdigest(),
    }


def _runtime_provenance() -> dict:
    versions = {}
    for package in ("torch", "accelerate", "deepspeed", "transformers", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _dataset_provenance(dataloader) -> tuple[dict, dict]:
    """Hash selected episode IDs/lengths and registered camera/data metadata."""

    mixture = dataloader.dataset
    subdatasets = list(getattr(mixture, "datasets", [mixture]))
    manifest_entries = []
    summaries = []
    for dataset in subdatasets:
        raw_trajectory_ids = list(getattr(dataset, "trajectory_ids", []))
        trajectory_ids = _jsonable(raw_trajectory_ids)
        trajectory_lengths = _jsonable(getattr(dataset, "trajectory_lengths", []))
        sampleable_lengths = _jsonable(getattr(dataset, "sampleable_trajectory_lengths", []))
        info_meta = _jsonable(getattr(dataset, "lerobot_info_meta", {}))
        modality_keys = _jsonable(getattr(dataset, "modality_keys", {}))
        metadata_by_id = getattr(dataset, "trajectory_ids_to_metadata", None)
        selected_episode_metadata = None
        if metadata_by_id is not None:
            selected_episode_metadata = [
                {
                    "trajectory_id": _jsonable(trajectory_id),
                    "metadata": _jsonable(metadata_by_id[trajectory_id]),
                }
                for trajectory_id in raw_trajectory_ids
            ]
        entry = {
            "dataset_name": str(getattr(dataset, "dataset_name", type(dataset).__name__)),
            "dataset_path": str(getattr(dataset, "dataset_path", "")),
            "lerobot_version": str(getattr(dataset, "_lerobot_version", "")),
            "v3_video_index_schema": _jsonable(getattr(dataset, "_v3_video_index_schema", None)),
            "trajectory_ids": trajectory_ids,
            "trajectory_lengths": trajectory_lengths,
            "sampleable_trajectory_lengths": sampleable_lengths,
            "modality_keys": modality_keys,
            "lerobot_info_meta": info_meta,
            "selected_episode_metadata": selected_episode_metadata,
            "video_override_provenance": _jsonable(getattr(dataset, "video_override_provenance", None)),
        }
        manifest_entries.append(entry)
        summaries.append(
            {
                "dataset_name": entry["dataset_name"],
                "dataset_path": entry["dataset_path"],
                "num_episodes": len(trajectory_ids),
                "num_sampleable_steps": int(sum(sampleable_lengths)),
                "modality_keys": modality_keys,
                "video_override_provenance": entry["video_override_provenance"],
            }
        )

    mixture_video_provenance = _jsonable(getattr(mixture, "video_override_provenance", None))
    mixture_sampling_provenance = _jsonable(getattr(mixture, "sampling_provenance", None))
    manifest = {
        "mixture_type": type(mixture).__name__,
        "dataset_length": len(mixture),
        "video_override_provenance": mixture_video_provenance,
        "sampling_provenance": mixture_sampling_provenance,
        "entries": manifest_entries,
    }
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    summary = {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "video_override_provenance": mixture_video_provenance,
        "sampling_provenance": mixture_sampling_provenance,
        "subdatasets": summaries,
    }
    return summary, manifest


def resolve_and_record_training_horizon(cfg, dataloader, accelerator) -> tuple[TrainingHorizon, dict, dict, str]:
    """Resolve the user-facing duration after the actual train dataset exists."""

    dataset_length = len(dataloader.dataset)
    effective_global_batch_size = (
        int(cfg.datasets.vla_data.per_device_batch_size)
        * int(accelerator.num_processes)
        * int(accelerator.gradient_accumulation_steps)
    )
    if cfg.trainer.get("num_sampling_epochs", None) is not None and int(accelerator.gradient_accumulation_steps) != 1:
        raise ValueError(
            "trainer.num_sampling_epochs currently requires gradient_accumulation_steps=1. "
            "With accumulation, optimizer boundaries can cross DataLoader epoch boundaries; "
            "refusing to mislabel that physical duration."
        )
    horizon = resolve_training_horizon(
        max_train_steps=cfg.trainer.get("max_train_steps", None),
        num_sampling_epochs=cfg.trainer.get("num_sampling_epochs", None),
        dataset_length=dataset_length,
        effective_global_batch_size=effective_global_batch_size,
        world_size=accelerator.num_processes,
    )

    # Persist every derived value in config.full.yaml/config.yaml. The user inputs stay
    # untouched; all runtime consumers read the resolved_* values.
    cfg.trainer.resolved_horizon_source = horizon.source
    cfg.trainer.resolved_max_train_steps = horizon.max_train_steps
    cfg.trainer.resolved_dataset_length = horizon.dataset_length
    cfg.trainer.resolved_effective_global_batch_size = horizon.effective_global_batch_size
    cfg.trainer.resolved_steps_per_sampling_epoch = horizon.steps_per_sampling_epoch
    cfg.trainer.resolved_requested_samples = horizon.requested_samples
    cfg.trainer.resolved_planned_samples = horizon.planned_samples
    cfg.trainer.resolved_padding_samples = horizon.padding_samples
    cfg.trainer.resolved_world_size = horizon.world_size

    schedule_fields = (
        ("num_warmup_steps", "warmup_ratio", True),
        ("logging_frequency", "logging_interval_ratio", False),
        ("eval_interval", "eval_interval_ratio", False),
        ("save_interval", "save_interval_ratio", False),
    )
    for absolute_field, ratio_field, allow_zero_absolute in schedule_fields:
        resolved_value = resolve_progress_interval(
            absolute_field=absolute_field,
            ratio_field=ratio_field,
            absolute_value=cfg.trainer.get(absolute_field, None),
            ratio_value=cfg.trainer.get(ratio_field, None),
            total_steps=horizon.max_train_steps,
            allow_zero_absolute=allow_zero_absolute,
        )
        setattr(cfg.trainer, f"resolved_{absolute_field}", resolved_value)

    code_provenance_result = None
    if accelerator.is_main_process:
        try:
            code_provenance_result = {
                "value": _code_provenance(),
                "error": None,
            }
        except Exception as exc:
            code_provenance_result = {
                "value": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
    code_provenance_result = _rank0_consensus(
        accelerator,
        code_provenance_result,
    )
    if code_provenance_result["error"]:
        raise RuntimeError("Rank-0 code provenance collection failed: " f"{code_provenance_result['error']}")
    code_provenance = code_provenance_result["value"]
    runtime_provenance = _runtime_provenance()
    dataset_provenance, dataset_manifest = _dataset_provenance(dataloader)
    data_resume_contract = _inspect_data_resume_contract(cfg, dataloader)
    exact_resume_required = bool(cfg.trainer.get("require_exact_resume", False))
    if exact_resume_required and not data_resume_contract["exact"]:
        raise RuntimeError(
            "This run requires exact full-state resume, but the data pipeline "
            "cannot prove deterministic epoch+index reconstruction: "
            f"{data_resume_contract['reason']}. Exact resume needs action-only training with "
            "dataset_py=lerobot_datasets, gen_model.enable=false, a SequentialSampler, "
            "sample_error_policy=raise, and only deterministic action transforms."
        )
    cfg.trainer.resolved_exact_resume_required = exact_resume_required
    cfg.trainer.resolved_data_resume_strategy = (
        "prepared_batch_sampler_skip_v1" if exact_resume_required and data_resume_contract["exact"] else None
    )
    provenance = {
        "code": code_provenance,
        "runtime": runtime_provenance,
        "dataset": dataset_provenance,
        "data_resume_contract": data_resume_contract,
    }
    _assert_rank_state_consensus(
        accelerator,
        {
            "code": code_provenance,
            "runtime": runtime_provenance,
            "dataset_manifest_sha256": dataset_provenance["sha256"],
            "data_resume_contract": data_resume_contract,
        },
        "Code/dataset provenance",
    )
    cfg.trainer.resolved_code_commit = code_provenance["git_commit"]
    cfg.trainer.resolved_code_dirty_sha256 = code_provenance["git_dirty_sha256"]
    cfg.trainer.resolved_dataset_manifest_sha256 = dataset_provenance["sha256"]
    cfg.trainer.resolved_video_override_provenance_sha256 = (
        dataset_provenance.get("video_override_provenance") or {}
    ).get("sha256")
    sampling_provenance = dataset_provenance.get("sampling_provenance") or {}
    cfg.trainer.resolved_sampling_strategy = sampling_provenance.get("strategy", "weighted_with_replacement")
    cfg.trainer.resolved_transition_population_size = sampling_provenance.get("population_size")
    cfg.trainer.resolved_transition_epoch_size = sampling_provenance.get("epoch_size")
    cfg.trainer.resolved_transition_dropped_tail = sampling_provenance.get("dropped_tail_per_epoch")

    full_cfg = cfg.unwrap() if isinstance(cfg, AccessTrackedConfig) else cfg
    plain_cfg = OmegaConf.to_container(full_cfg, resolve=True)
    fingerprint = config_fingerprint(plain_cfg)
    cfg.trainer.config_fingerprint = fingerprint
    split_contract = _resolved_split_contract(cfg)

    manifest_result = None
    if accelerator.is_main_process:
        try:
            manifest_path = Path(cfg.output_dir) / "dataset_manifest.json"
            if exact_resume_required and bool(cfg.trainer.get("is_resume", False)):
                with open(manifest_path) as f:
                    stored_manifest = json.load(f)
                stored_encoded = json.dumps(
                    stored_manifest,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode("utf-8")
                stored_digest = hashlib.sha256(stored_encoded).hexdigest()
                if stored_digest != dataset_provenance["sha256"]:
                    raise RuntimeError(
                        f"Stored dataset manifest hash {stored_digest} disagrees with "
                        f"current {dataset_provenance['sha256']}."
                    )
            else:
                _atomic_write_json(str(manifest_path), dataset_manifest)
            provenance_path = Path(cfg.output_dir) / "dataset_provenance.json"
            if exact_resume_required and bool(cfg.trainer.get("is_resume", False)):
                with open(provenance_path) as f:
                    stored_provenance = json.load(f)
                if stored_provenance != dataset_provenance:
                    raise RuntimeError(
                        "Stored dataset_provenance.json disagrees with the current " "all-rank dataset/video provenance."
                    )
            else:
                _atomic_write_json(str(provenance_path), dataset_provenance)
            manifest_result = {"error": None}
        except Exception as exc:
            manifest_result = {"error": f"{type(exc).__name__}: {exc}"}
    manifest_result = _rank0_consensus(accelerator, manifest_result)
    if manifest_result["error"]:
        raise RuntimeError(f"Dataset manifest publication/validation failed: {manifest_result['error']}")

    if accelerator.is_main_process:
        logger.info(
            "[training-horizon] "
            f"source={horizon.source}, dataset_length={horizon.dataset_length}, "
            f"effective_global_batch={horizon.effective_global_batch_size}, "
            f"steps_per_sampling_epoch={horizon.steps_per_sampling_epoch}, "
            f"num_sampling_epochs={horizon.num_sampling_epochs}, "
            f"resolved_max_train_steps={horizon.max_train_steps}, "
            f"requested_samples={horizon.requested_samples}, "
            f"planned_samples={horizon.planned_samples}, padding_samples={horizon.padding_samples}, "
            f"warmup_steps={cfg.trainer.resolved_num_warmup_steps}, "
            f"logging_interval={cfg.trainer.resolved_logging_frequency}, "
            f"eval_interval={cfg.trainer.resolved_eval_interval}, "
            f"save_interval={cfg.trainer.resolved_save_interval}, "
            f"config_fingerprint={fingerprint}, code={code_provenance}, "
            f"runtime={runtime_provenance}, "
            f"dataset_manifest_sha256={dataset_provenance['sha256']}"
            f", data_resume_contract={data_resume_contract}"
        )
    return horizon, split_contract, provenance, fingerprint


class VLATrainer(TrainerUtils):
    def __init__(
        self,
        cfg,
        model,
        vla_train_dataloader,
        optimizer,
        lr_scheduler,
        accelerator,
        horizon,
        split_contract,
        provenance,
        fingerprint,
        val_dataloader=None,
    ):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        self.horizon = horizon
        self.split_contract = split_contract
        self.provenance = provenance
        self.config_fingerprint = fingerprint
        self._exact_resume_enabled = bool(self.config.trainer.get("require_exact_resume", False))

        self.completed_steps = 0
        self.samples_seen = 0
        self.train_microbatches_consumed = 0
        self._resume_state = None
        self._is_full_resume = False
        self._last_saved_step = None
        self.total_batch_size = self._calculate_total_batch_size()
        grad_accum = int(self.accelerator.gradient_accumulation_steps)
        self.microbatches_per_sampling_epoch = (
            self.horizon.dataset_length * grad_accum + self.horizon.effective_global_batch_size - 1
        ) // self.horizon.effective_global_batch_size
        # trainer.telemetry.enabled adds windowed throughput / timing / memory metrics and
        # enforces the optional required_loss_keys / forbidden_loss_keys /
        # expected_images_per_sample checks at every logging step.
        telemetry_cfg = self.config.trainer.get("telemetry", {})
        self._telemetry_enabled = bool(telemetry_cfg.get("enabled", False) if hasattr(telemetry_cfg, "get") else False)
        self._telemetry_records = []
        self._telemetry_sample_ids = []
        self._telemetry_dataset_counts = {}
        self._telemetry_violations = []
        self._telemetry_window_started_at = time.perf_counter()
        self._telemetry_process_cpu_started_at = time.process_time()

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        if (
            self.horizon.source == "num_sampling_epochs"
            and self.val_dataloader is None
            and int(self.config.trainer.resolved_eval_interval) <= self.horizon.max_train_steps
        ):
            raise ValueError(
                "trainer.num_sampling_epochs requires an independent validation split "
                "(datasets.vla_data.val_episode_fraction or use_official_splits), or "
                "eval_interval beyond the resolved horizon."
            )

        if self._exact_resume_enabled:
            # Resolve/validate exact resume before writing config snapshots. A
            # mismatched resume attempt must not overwrite the original audit.
            self._init_checkpointing()
            if not self._is_full_resume:
                self._save_initial_configs()
        else:
            # Model-only checkpointing: write the config snapshots first.
            self._save_initial_configs()
            self._init_checkpointing()
            self._adjust_lr_scheduler_for_resume()

        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)

        self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
            self.accelerator,
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
        )
        if self._exact_resume_enabled:
            prepared_batches_per_epoch = len(self.vla_train_dataloader)
            _assert_rank_state_consensus(
                self.accelerator,
                {
                    "prepared_batches_per_epoch": prepared_batches_per_epoch,
                    "resolved_microbatches_per_epoch": self.microbatches_per_sampling_epoch,
                },
                "Prepared DataLoader epoch length",
            )
            if prepared_batches_per_epoch != self.microbatches_per_sampling_epoch:
                raise RuntimeError(
                    "Prepared DataLoader length disagrees with the exact-resume cursor "
                    "contract: "
                    f"len(prepared_loader)={prepared_batches_per_epoch}, "
                    f"resolved={self.microbatches_per_sampling_epoch}. Refusing to save "
                    "an offset that cannot be replayed by skip_first_batches."
                )

            # The scheduler is intentionally not accelerator.prepare()'d, so
            # register it explicitly for exact save_state/load_state.
            self.accelerator.register_for_checkpointing(self.lr_scheduler)
        self._init_tracking()

        # Restore RNG last, after model/DeepSpeed/tracker initialization. Otherwise
        # resume-only setup work could consume restored RNG before the next
        # DataLoader workers are spawned.
        if self._is_full_resume:
            self._load_checkpoint(self.resume_from_checkpoint)

    def _calculate_total_batch_size(self):
        """Calculate global batch size."""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_tracking(self):
        """Create the rank-0 metrics tracker (W&B or metrics.jsonl); never fails the run."""
        self.tracker = MetricsTracker(
            self.config,
            self.config.output_dir,
            group="vla-train",
            is_main_process=self.accelerator.is_main_process,
        )
        # Rendezvous after rank-0 tracker init so a slow W&B start cannot let the other
        # ranks reach the first collective alone and hit the NCCL watchdog.
        self.accelerator.wait_for_everyone()

    def _save_initial_configs(self):
        """Save full config and training script at the very start of training."""
        result = None
        if self.accelerator.is_main_process:
            try:
                output_dir = Path(self.config.output_dir)

                # 1. Save config.full.yaml — the complete merged config (all parameters)
                if isinstance(self.config, AccessTrackedConfig):
                    full_cfg = self.config.unwrap()
                else:
                    full_cfg = self.config
                full_yaml_path = output_dir / "config.full.yaml"
                OmegaConf.save(full_cfg, full_yaml_path, resolve=True)
                logger.info(f"📝 Full config saved at {full_yaml_path}")

                # 2. Save config.yaml — accessed-only snapshot (will be updated at checkpoints)
                if isinstance(self.config, AccessTrackedConfig):
                    self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
                    logger.info(f"📊 Accessed config snapshot saved at {output_dir / 'config.yaml'}")
                result = {"error": None}
            except Exception as exc:
                result = {"error": f"{type(exc).__name__}: {exc}"}
        result = _rank0_consensus(self.accelerator, result)
        if result["error"]:
            raise RuntimeError(f"Initial config publication failed: {result['error']}")

    def _publish_initialization_manifest(self, manifest: dict) -> None:
        """Write initialization_manifest.json (transfer record + this run's dataset statistics hash)."""

        result = None
        if self.accelerator.is_main_process:
            try:
                manifest = add_dataset_statistics_identity(
                    manifest,
                    Path(self.config.output_dir) / "dataset_statistics.json",
                )
                write_initialization_manifest(
                    Path(self.config.output_dir) / "initialization_manifest.json",
                    manifest,
                )
                result = {"error": None}
            except Exception as exc:
                result = {"error": f"{type(exc).__name__}: {exc}"}
        result = _rank0_consensus(self.accelerator, result)
        if result["error"]:
            raise RuntimeError(f"Initialization manifest publication failed: {result['error']}")

    def _load_weights_only_initialization(self, checkpoint_path: str) -> None:
        """Initialize from ``trainer.pretrained_checkpoint``.

        Without ``trainer.transfer_policy`` the checkpoint is loaded non-strictly (optionally only
        the modules listed in ``trainer.reload_modules``). With a transfer policy the strict
        loader in ``transfer_initialization`` is used and ``initialization_manifest.json`` records
        exactly which tensors were loaded, reset and dropped.
        """

        transfer_policy = self.config.trainer.get("transfer_policy", None)
        reload_modules = self.config.trainer.get("reload_modules", None)
        retain_gen_model = self.config.trainer.get("transfer_retain_gen_model", False)
        normalized_policy = validate_transfer_options(
            transfer_policy,
            reload_modules,
            retain_gen_model=retain_gen_model,
        )
        if normalized_policy is None:
            self.model = self.load_pretrained_backbones(
                self.model,
                checkpoint_path,
                reload_modules=reload_modules,
            )
            return

        source = None
        try:
            source = validate_transfer_checkpoint(checkpoint_path)
            check_result = {"error": None, "digest": json.dumps(source, sort_keys=True)}
        except Exception as exc:
            check_result = {"error": f"{type(exc).__name__}: {exc}", "digest": None}
        _all_rank_result_consensus(self.accelerator, check_result, "Transfer checkpoint check")

        manifest = None
        try:
            manifest = load_transfer_checkpoint(
                self.model,
                checkpoint_path,
                normalized_policy,
                source=source,
                retain_gen_model=retain_gen_model,
            )
            digest = hashlib.sha256(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            local_result = {"error": None, "digest": digest}
        except Exception as exc:
            local_result = {"error": f"{type(exc).__name__}: {exc}", "digest": None}
        _all_rank_result_consensus(self.accelerator, local_result, "Strict weights-only transfer")
        self._publish_initialization_manifest(manifest)

    def _init_checkpointing(self):
        """Select either weights-only initialization or a complete full resume."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        explicit_resume = getattr(self.config.trainer, "resume_from_checkpoint", None)
        is_resume = bool(getattr(self.config.trainer, "is_resume", False))
        transfer_policy = self.config.trainer.get("transfer_policy", None)
        reload_modules = self.config.trainer.get("reload_modules", None)
        retain_gen_model = self.config.trainer.get("transfer_retain_gen_model", False)
        normalized_policy = validate_transfer_options(
            transfer_policy,
            reload_modules,
            retain_gen_model=retain_gen_model,
        )
        self.resume_from_checkpoint = pretrained_checkpoint

        if normalized_policy is not None and not pretrained_checkpoint:
            raise ValueError(
                "trainer.transfer_policy requires trainer.pretrained_checkpoint as initialization provenance."
            )

        if not self._exact_resume_enabled:
            if explicit_resume:
                raise ValueError(
                    "trainer.resume_from_checkpoint is reserved for exact full-state "
                    "resume. Set trainer.require_exact_resume=true, or pass a model "
                    "file through trainer.pretrained_checkpoint."
                )

            if is_resume:
                full_state_directories = sorted(
                    name
                    for name in os.listdir(self.checkpoint_dir)
                    if re.fullmatch(r"steps_\d+", name) and os.path.isdir(os.path.join(self.checkpoint_dir, name))
                )
                if full_state_directories:
                    raise RuntimeError(
                        "Model-only resume found full-state checkpoint directories "
                        f"{full_state_directories[-3:]} in {self.checkpoint_dir}. Set "
                        "trainer.require_exact_resume=true to validate and restore them."
                    )
                resume_from_checkpoint, completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
                if resume_from_checkpoint:
                    self.resume_from_checkpoint = resume_from_checkpoint
                    self.completed_steps = completed_steps
                    self.model = self.load_pretrained_backbones(
                        self.model,
                        resume_from_checkpoint,
                        reload_modules=None,
                    )
                    logger.info(
                        "Resuming model-only training from checkpoint: "
                        f"{resume_from_checkpoint}, steps: {completed_steps}"
                    )
                    return

                logger.warning(
                    f"No model-only checkpoint found in {self.checkpoint_dir}. " "Starting training from scratch."
                )

            if pretrained_checkpoint:
                self._load_weights_only_initialization(pretrained_checkpoint)
                self.resume_from_checkpoint = pretrained_checkpoint
                logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: 0")
            else:
                logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0
            self.samples_seen = 0
            self.train_microbatches_consumed = 0
            return

        if is_resume:
            decision = None
            if self.accelerator.is_main_process:
                try:
                    warning = None
                    if explicit_resume:
                        resume_from_checkpoint = str(explicit_resume)
                        resume_state, _ = load_complete_checkpoint_metadata(resume_from_checkpoint)
                        if not is_resumable_checkpoint(resume_from_checkpoint):
                            raise RuntimeError(
                                f"Explicit checkpoint {resume_from_checkpoint!r} is complete but does not "
                                "prove the exact next training sample. Use a full-state checkpoint with "
                                "a deterministic epoch+index contract, or use its portable model as "
                                "weights-only initialization."
                            )
                        validate_resume_state(
                            resume_state,
                            horizon=self.horizon,
                            split=self.split_contract,
                            provenance=self.provenance,
                            fingerprint=self.config_fingerprint,
                        )
                    else:
                        latest_complete = find_latest_complete_checkpoint(self.checkpoint_dir)
                        candidates = []
                        for name in os.listdir(self.checkpoint_dir):
                            match = re.fullmatch(r"steps_(\d+)", name)
                            path = os.path.join(self.checkpoint_dir, name)
                            if match and is_resumable_checkpoint(path):
                                candidates.append((int(match.group(1)), path))
                        candidates.sort(reverse=True)

                        resume_from_checkpoint = None
                        resume_state = None
                        rejected = []
                        for _, candidate in candidates:
                            try:
                                candidate_state, _ = load_complete_checkpoint_metadata(candidate)
                                validate_resume_state(
                                    candidate_state,
                                    horizon=self.horizon,
                                    split=self.split_contract,
                                    provenance=self.provenance,
                                    fingerprint=self.config_fingerprint,
                                )
                            except (OSError, ValueError, RuntimeError) as exc:
                                rejected.append(f"{candidate}: {exc}")
                                continue
                            resume_from_checkpoint = candidate
                            resume_state = candidate_state
                            break

                        if resume_from_checkpoint is None:
                            model_only = [
                                name
                                for name in os.listdir(self.checkpoint_dir)
                                if re.fullmatch(r"steps_\d+_(?:pytorch_model\.pt|model\.safetensors)", name)
                            ]
                            details = []
                            if latest_complete:
                                details.append(
                                    f"Latest complete checkpoint {latest_complete!r} is not compatible "
                                    "and exact-resumable."
                                )
                            if rejected:
                                details.append("Rejected exact candidates: " + " | ".join(rejected[:3]))
                            if model_only:
                                details.append(
                                    f"Model-only files {sorted(model_only)[-3:]} are valid only as "
                                    "trainer.pretrained_checkpoint, not as full resume state."
                                )
                            suffix = " " + " ".join(details) if details else ""
                            raise RuntimeError(
                                "trainer.is_resume=true but no compatible, completed, exact-resumable "
                                f"checkpoint exists in {self.checkpoint_dir}.{suffix}"
                            )

                        if latest_complete and latest_complete != resume_from_checkpoint:
                            latest_match = re.fullmatch(r"steps_(\d+)", os.path.basename(latest_complete))
                            selected_match = re.fullmatch(r"steps_(\d+)", os.path.basename(resume_from_checkpoint))
                            if (
                                latest_match
                                and selected_match
                                and int(latest_match.group(1)) > int(selected_match.group(1))
                            ):
                                warning = (
                                    f"Skipping newer complete checkpoint {latest_complete} because it is "
                                    "not exact-compatible; resuming from the latest proven position "
                                    f"{resume_from_checkpoint}."
                                )

                    code_warning = describe_code_change(resume_state, self.provenance)
                    if code_warning:
                        warning = f"{warning} {code_warning}" if warning else code_warning
                    decision = {
                        "error": None,
                        "path": resume_from_checkpoint,
                        "state": resume_state,
                        "warning": warning,
                    }
                except Exception as exc:
                    decision = {
                        "error": f"{type(exc).__name__}: {exc}",
                        "path": None,
                        "state": None,
                        "warning": None,
                    }
            decision = _rank0_consensus(self.accelerator, decision)
            if decision["error"]:
                raise RuntimeError(decision["error"])

            resume_from_checkpoint = decision["path"]
            resume_state = decision["state"]
            local_validation = None
            try:
                local_state, local_marker = load_complete_checkpoint_metadata(resume_from_checkpoint)
                validate_resume_state(
                    local_state,
                    horizon=self.horizon,
                    split=self.split_contract,
                    provenance=self.provenance,
                    fingerprint=self.config_fingerprint,
                )
                if local_state != resume_state:
                    raise RuntimeError("Locally visible trainer_state differs from the rank-0 selected state.")
                metadata_bytes = json.dumps(
                    {"state": local_state, "marker": local_marker},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                local_validation = {
                    "error": None,
                    "digest": hashlib.sha256(metadata_bytes).hexdigest(),
                }
            except Exception as exc:
                local_validation = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "digest": None,
                }
            _all_rank_result_consensus(
                self.accelerator,
                local_validation,
                "Full-resume payload preflight",
            )
            if decision["warning"]:
                logger.warning(decision["warning"])
            self._resume_state = resume_state
            self._is_full_resume = True
            self.resume_from_checkpoint = resume_from_checkpoint
            self.completed_steps = int(resume_state["completed_steps"])
            self.samples_seen = int(resume_state["samples_seen"])
            self.train_microbatches_consumed = int(resume_state["train_microbatches_consumed"])
            self._last_saved_step = self.completed_steps
            logger.info(
                f"Validated full resume checkpoint {resume_from_checkpoint}: "
                f"completed_steps={self.completed_steps}, samples_seen={self.samples_seen}, "
                f"sampling_epoch={resume_state['sampling_epoch']}"
            )
            if pretrained_checkpoint:
                logger.info(
                    "Full resume restores the saved model/optimizer/scheduler/RNG state; "
                    "trainer.pretrained_checkpoint is not loaded again."
                )
            return

        if explicit_resume:
            raise ValueError(
                "trainer.resume_from_checkpoint is set but trainer.is_resume is false. "
                "Set trainer.is_resume=true for a full-state resume, or use "
                "trainer.pretrained_checkpoint for weights-only initialization."
            )

        if pretrained_checkpoint:
            self._load_weights_only_initialization(pretrained_checkpoint)
            self.completed_steps = 0
            self.samples_seen = 0
            self.train_microbatches_consumed = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(
                f"Loaded weights-only pretrained checkpoint: {pretrained_checkpoint}; "
                "optimizer/scheduler/RNG start fresh at step 0."
            )
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0
            self.samples_seen = 0
            self.train_microbatches_consumed = 0

    def _adjust_lr_scheduler_for_resume(self):
        """Fast-forward the LR scheduler after a model-only resume."""
        if self.completed_steps > 0:
            logger.info(f"Adjusting LR scheduler for model-only resume from step {self.completed_steps}")
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            logger.info(
                "LR scheduler adjusted to step " f"{self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}"
            )

    def _load_checkpoint(self, checkpoint_path):
        """Collectively restore prepared model/optimizer/scheduler/RNG state."""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.wait_for_everyone()
        self._last_saved_step = self.completed_steps
        self.accelerator.print(
            f"Resumed full training state from {checkpoint_path}: "
            f"step={self.completed_steps}, samples_seen={self.samples_seen}"
        )

    def _save_checkpoint(self):
        """Save a model-only file or a full-state directory (trainer.require_exact_resume)."""
        if not self._exact_resume_enabled:
            return self._save_model_only_checkpoint()
        return self._save_exact_checkpoint()

    def _save_model_only_checkpoint(self):
        """Save ``checkpoints/steps_<N>_pytorch_model.pt`` (or ``_model.safetensors``)."""
        save_format = getattr(self.config.trainer, "save_format", "pt")
        if save_format not in {"pt", "safetensors"}:
            raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")

        # DeepSpeed state consolidation can be collective, so every rank enters
        # even though only rank 0 writes the model file.
        state_dict = self.accelerator.get_state_dict(self.model)
        self.accelerator.wait_for_everyone()
        publication = None
        if self.accelerator.is_main_process:
            try:
                checkpoint_prefix = os.path.join(
                    self.checkpoint_dir,
                    f"steps_{self.completed_steps}",
                )
                if save_format == "safetensors":
                    from safetensors.torch import save_file

                    checkpoint_path = checkpoint_prefix + "_model.safetensors"
                    save_file(state_dict, checkpoint_path)
                else:
                    checkpoint_path = checkpoint_prefix + "_pytorch_model.pt"
                    torch.save(state_dict, checkpoint_path)

                self._prune_old_checkpoints()
                with open(
                    os.path.join(self.config.output_dir, "summary.jsonl"),
                    "a",
                ) as f:
                    f.write(json.dumps({"steps": self.completed_steps}) + "\n")

                if isinstance(self.config, AccessTrackedConfig):
                    self.config.save_accessed_config(
                        Path(self.config.output_dir) / "config.yaml",
                        use_original_values=False,
                    )
                publication = {
                    "error": None,
                    "checkpoint_path": checkpoint_path,
                }
            except Exception as exc:
                publication = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "checkpoint_path": None,
                }
        publication = _rank0_consensus(self.accelerator, publication)
        if publication["error"]:
            raise RuntimeError(f"Checkpoint publication failed: {publication['error']}")

        self.accelerator.print(f"✅ Model-only checkpoint saved at {publication['checkpoint_path']}")
        self.accelerator.wait_for_everyone()
        self._last_saved_step = self.completed_steps

    def _save_exact_checkpoint(self):
        """Collectively save full state, then publish a completion marker last."""
        save_format = getattr(self.config.trainer, "save_format", "pt")
        if save_format not in {"pt", "safetensors"}:
            raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
        progress = _assert_rank_state_consensus(
            self.accelerator,
            {
                "completed_steps": int(self.completed_steps),
                "samples_seen": int(self.samples_seen),
                "train_microbatches_consumed": int(self.train_microbatches_consumed),
            },
            "Full-state checkpoint",
        )
        checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{progress['completed_steps']}")

        preflight = None
        if self.accelerator.is_main_process:
            try:
                if os.path.exists(checkpoint_path):
                    raise FileExistsError(
                        f"Refusing to overwrite checkpoint path {checkpoint_path}. "
                        "A directory without checkpoint.complete is an interrupted save and must be "
                        "inspected/removed explicitly; a completed directory is immutable."
                    )
                os.makedirs(checkpoint_path)
                preflight = {"error": None}
            except Exception as exc:
                preflight = {"error": f"{type(exc).__name__}: {exc}"}
        preflight = _rank0_consensus(self.accelerator, preflight)
        if preflight["error"]:
            raise RuntimeError(preflight["error"])
        self.accelerator.wait_for_everyone()

        # save_state is collective under DeepSpeed and captures the prepared
        # model/optimizer plus RNG. The scheduler is included because it was
        # explicitly register_for_checkpointing()'d in prepare_training().
        self.accelerator.save_state(checkpoint_path)

        # DeepSpeed state consolidation may also be collective. Every rank MUST
        # enter get_state_dict; only the filesystem write is rank-0-only.
        state_dict = self.accelerator.get_state_dict(self.model)
        trainer_state = build_trainer_state(
            completed_steps=self.completed_steps,
            samples_seen=self.samples_seen,
            train_microbatches_consumed=self.train_microbatches_consumed,
            horizon=self.horizon,
            gradient_accumulation_steps=self.accelerator.gradient_accumulation_steps,
            split=self.split_contract,
            provenance=self.provenance,
            fingerprint=self.config_fingerprint,
        )
        self.accelerator.wait_for_everyone()

        publication = None
        if self.accelerator.is_main_process:
            try:
                if save_format == "safetensors":
                    from safetensors.torch import save_file

                    portable_name = "portable_model.safetensors"
                    save_file(state_dict, os.path.join(checkpoint_path, portable_name))
                else:
                    portable_name = "portable_pytorch_model.pt"
                    torch.save(state_dict, os.path.join(checkpoint_path, portable_name))

                with open(os.path.join(checkpoint_path, TRAINER_STATE_FILE), "w") as f:
                    json.dump(trainer_state, f, indent=2, sort_keys=True)
                    f.flush()
                    os.fsync(f.fileno())

                backend = getattr(
                    self.accelerator.distributed_type,
                    "name",
                    str(self.accelerator.distributed_type).split(".")[-1],
                )
                payload_files, accelerate_payload_files, state_layout = _checkpoint_payload_manifest(
                    checkpoint_path,
                    portable_name,
                    world_size=int(self.accelerator.num_processes),
                    backend=backend,
                )
                marker = {
                    "version": trainer_state["version"],
                    "completed_steps": self.completed_steps,
                    "portable_model": portable_name,
                    "trainer_state": TRAINER_STATE_FILE,
                    "payload_files": payload_files,
                    "accelerate_payload_files": accelerate_payload_files,
                    "state_layout": state_layout,
                }
                _atomic_write_json(os.path.join(checkpoint_path, COMPLETE_MARKER), marker)
                load_complete_checkpoint_metadata(checkpoint_path)
                publication = {"error": None}
            except Exception as exc:
                publication = {"error": f"{type(exc).__name__}: {exc}"}
        publication = _rank0_consensus(self.accelerator, publication)
        if publication["error"]:
            raise RuntimeError(f"Checkpoint {checkpoint_path} was not published as complete: {publication['error']}")

        if self.accelerator.is_main_process:
            if isinstance(self.config, AccessTrackedConfig):
                try:
                    logger.info("📊 Saving accessed configuration...")
                    output_dir = Path(self.config.output_dir)
                    self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
                    logger.info("✅ Configuration files saved")
                except Exception as exc:
                    logger.warning(f"Checkpoint is complete, but accessed-config refresh failed: {exc}")

            summary_data = {
                "steps": self.completed_steps,
                "samples_seen": self.samples_seen,
                "train_microbatches_consumed": self.train_microbatches_consumed,
                "sampling_epoch": trainer_state["sampling_epoch"],
                "sampling_epoch_offset": trainer_state["sampling_epoch_offset"],
                "exact_data_position": trainer_state["exact_data_position"],
                "checkpoint": checkpoint_path,
            }
            try:
                with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                    f.write(json.dumps(summary_data) + "\n")
            except OSError as exc:
                logger.warning(f"Checkpoint is complete, but summary.jsonl update failed: {exc}")

            if not trainer_state["exact_data_position"]:
                logger.warning(
                    f"Checkpoint {checkpoint_path} is complete for weights/optimizer diagnostics but "
                    "cannot be used for exact full resume: "
                    f"{trainer_state['exact_data_position_reason']}. Exact restart requires "
                    "gradient_accumulation_steps=1, deterministic epoch+index data, and no "
                    "training-iterator consumption outside optimizer steps."
                )
            self.accelerator.print(f"✅ Full-state checkpoint saved at {checkpoint_path}")
            self._prune_old_checkpoints()

        self.accelerator.wait_for_everyone()
        self._last_saved_step = self.completed_steps

    def _prune_old_checkpoints(self):
        """Keep the newest ``trainer.save_total_limit`` checkpoints of the active mode.

        Exact mode only touches completed state directories and always keeps the newest
        exact-resumable one; model-only mode only touches model-only files.
        """
        save_total_limit = getattr(self.config.trainer, "save_total_limit", None)
        if not save_total_limit or save_total_limit <= 0:
            return

        if not self._exact_resume_enabled:
            pattern = re.compile(r"steps_(\d+)_(?:pytorch_model\.pt|model\.safetensors)$")
            checkpoints = [
                (
                    int(pattern.match(name).group(1)),
                    os.path.join(self.checkpoint_dir, name),
                )
                for name in os.listdir(self.checkpoint_dir)
                if pattern.match(name) and os.path.isfile(os.path.join(self.checkpoint_dir, name))
            ]
            checkpoints.sort(key=lambda item: item[0])
            for _, path in checkpoints[:-save_total_limit]:
                try:
                    os.remove(path)
                    self.accelerator.print(f"🗑️  Pruned old checkpoint: {path}")
                except OSError as exc:
                    logger.warning(f"Failed to prune checkpoint {path}: {exc}")
            return

        pattern = re.compile(r"steps_(\d+)$")
        checkpoints = [
            (int(pattern.match(name).group(1)), os.path.join(self.checkpoint_dir, name))
            for name in os.listdir(self.checkpoint_dir)
            if pattern.match(name) and is_complete_checkpoint(os.path.join(self.checkpoint_dir, name))
        ]
        checkpoints.sort(key=lambda item: item[0])
        keep_paths = {path for _, path in checkpoints[-save_total_limit:]}
        resumable = []
        for step, path in checkpoints:
            if not is_resumable_checkpoint(path):
                continue
            try:
                state, _ = load_complete_checkpoint_metadata(path)
                validate_resume_state(
                    state,
                    horizon=self.horizon,
                    split=self.split_contract,
                    provenance=self.provenance,
                    fingerprint=self.config_fingerprint,
                )
            except (OSError, ValueError):
                continue
            resumable.append((step, path))
        if resumable:
            # A newer diagnostic checkpoint can be complete but non-resumable.
            # Preserve the newest exact boundary in addition to the ordinary
            # retention window so automatic restart never loses its only anchor.
            keep_paths.add(resumable[-1][1])

        for _, path in checkpoints:
            if path in keep_paths:
                continue
            try:
                shutil.rmtree(path)
                self.accelerator.print(f"🗑️  Pruned old full-state checkpoint: {path}")
            except OSError as exc:
                logger.warning(f"Failed to prune checkpoint {path}: {exc}")

    def _record_step_telemetry(
        self,
        *,
        batch,
        metrics,
        data_time,
        model_time,
        eval_time,
        step_time,
    ) -> None:
        if not self._telemetry_enabled:
            return

        telemetry_cfg = self.config.trainer.get("telemetry", {})
        examples, _ = unpack_preprocessed_batch(batch)
        sample_ids = []
        dataset_counts = {}
        image_count = 0
        strict_expected = str(self.config.datasets.vla_data.get("sample_error_policy", "resample")) == "raise"
        for example_index, example in enumerate(examples):
            image_count += len(example.get("image", []))
            provenance = example.get("_sample_provenance")
            if provenance is None:
                if strict_expected:
                    self._telemetry_violations.append(
                        "Strict telemetry requires _sample_provenance on every sample; "
                        f"missing at local batch position {example_index}."
                    )
                continue
            try:
                replacement_draws = int(provenance.get("replacement_draws", -1))
                if replacement_draws != 0:
                    self._telemetry_violations.append(
                        "sample_error_policy='raise' was violated at local batch "
                        f"position {example_index}: provenance={provenance!r}"
                    )
                dataset_name = str(provenance["dataset_name"])
                sample_ids.append(
                    (
                        dataset_name,
                        int(provenance["episode_index"]),
                        int(provenance["base_index"]),
                    )
                )
                dataset_counts[dataset_name] = dataset_counts.get(dataset_name, 0) + 1
            except Exception as exc:
                self._telemetry_violations.append(
                    "Invalid sample provenance at local batch position "
                    f"{example_index}: {type(exc).__name__}: {exc}; "
                    f"provenance={provenance!r}"
                )

        for key in telemetry_cfg.get("required_loss_keys", []):
            value = metrics.get(str(key))
            if value is None or not np.isfinite(float(value)):
                self._telemetry_violations.append(f"required finite metric {key!r} is missing or invalid: {value!r}")
        for key in telemetry_cfg.get("forbidden_loss_keys", []):
            if str(key) in metrics:
                self._telemetry_violations.append(f"forbidden loss metric {key!r} was produced")
        expected_images_per_sample = telemetry_cfg.get(
            "expected_images_per_sample",
            None,
        )
        if expected_images_per_sample is not None:
            expected_images = len(examples) * int(expected_images_per_sample)
            if image_count != expected_images:
                self._telemetry_violations.append(
                    "observation view-count contract failed: "
                    f"images={image_count}, samples={len(examples)}, "
                    f"expected_images_per_sample={expected_images_per_sample}"
                )

        record = {
            "data_time": float(data_time),
            "model_time": float(model_time),
            "eval_time": float(eval_time),
            "step_time": float(step_time),
            "samples": len(examples),
            "images": int(image_count),
            "action_loss": metrics.get("action_loss"),
            "total_loss": metrics.get("total_loss"),
            "grad_norm": metrics.get("optimization/grad_norm"),
            "gradient_clipped": metrics.get("optimization/gradient_clipped", 0.0),
            "optimizer_overflow": metrics.get("optimization/overflow", 0.0),
        }
        self._telemetry_records.append(record)
        self._telemetry_sample_ids.extend(sample_ids)
        for dataset_name, count in dataset_counts.items():
            self._telemetry_dataset_counts[dataset_name] = self._telemetry_dataset_counts.get(dataset_name, 0) + count

    @staticmethod
    def _telemetry_distribution(metrics, prefix, values) -> None:
        values = [float(value) for value in values if value is not None and np.isfinite(value)]
        if not values:
            return
        array = np.asarray(values, dtype=np.float64)
        metrics[f"{prefix}/mean"] = float(np.mean(array))
        metrics[f"{prefix}/p50"] = float(np.percentile(array, 50))
        metrics[f"{prefix}/p95"] = float(np.percentile(array, 95))
        metrics[f"{prefix}/p99"] = float(np.percentile(array, 99))
        metrics[f"{prefix}/max"] = float(np.max(array))

    def _flush_step_telemetry(self) -> dict:
        if not self._telemetry_enabled or not self._telemetry_records:
            return {}

        now = time.perf_counter()
        process_now = time.process_time()
        cuda_payload = {
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
            "current_allocated_bytes": 0,
            "current_reserved_bytes": 0,
        }
        if torch.cuda.is_available():
            cuda_payload = {
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "current_allocated_bytes": int(torch.cuda.memory_allocated()),
                "current_reserved_bytes": int(torch.cuda.memory_reserved()),
            }

        local_payload = {
            "rank": int(dist.get_rank()) if dist.is_initialized() else 0,
            "records": self._telemetry_records,
            "sample_ids": self._telemetry_sample_ids,
            "dataset_counts": self._telemetry_dataset_counts,
            "violations": self._telemetry_violations,
            "window_wall": float(now - self._telemetry_window_started_at),
            "process_cpu": float(process_now - self._telemetry_process_cpu_started_at),
            # Linux ru_maxrss is KiB. The supported training environment is Linux.
            "process_maxrss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
            "cuda": cuda_payload,
        }
        if dist.is_initialized() and dist.get_world_size() > 1:
            payloads = [None] * dist.get_world_size()
            dist.all_gather_object(payloads, local_payload)
        else:
            payloads = [local_payload]

        window_lengths = {len(payload["records"]) for payload in payloads}
        if len(window_lengths) != 1:
            raise RuntimeError(
                "Telemetry detected rank step-count divergence: " f"{[len(payload['records']) for payload in payloads]}"
            )
        rank_sample_counts = [sum(record["samples"] for record in payload["records"]) for payload in payloads]
        steps_in_window = next(iter(window_lengths))
        expected_local_samples = steps_in_window * int(self.config.datasets.vla_data.per_device_batch_size)
        audit_errors = [
            f"rank {payload['rank']}: {violation}" for payload in payloads for violation in payload.get("violations", [])
        ]
        if any(count != expected_local_samples for count in rank_sample_counts):
            audit_errors.append(
                "Telemetry observed an unexpected valid-sample count: "
                f"per_rank={rank_sample_counts}, expected_each={expected_local_samples}"
            )
        if audit_errors:
            raise RuntimeError("Telemetry provenance audit failed on every rank: " + " | ".join(audit_errors))

        records = [record for payload in payloads for record in payload["records"]]
        sample_ids = [tuple(sample_id) for payload in payloads for sample_id in payload["sample_ids"]]
        dataset_counts = {}
        for payload in payloads:
            for name, count in payload["dataset_counts"].items():
                dataset_counts[name] = dataset_counts.get(name, 0) + int(count)

        max_wall = max(float(payload["window_wall"]) for payload in payloads)
        global_samples = sum(rank_sample_counts)
        global_images = sum(record["images"] for record in records)
        if (
            str(self.config.datasets.vla_data.get("sample_error_policy", "resample")) == "raise"
            and len(sample_ids) != global_samples
        ):
            raise RuntimeError(
                "Strict telemetry sample provenance is incomplete: "
                f"provenance={len(sample_ids)}, actual={global_samples}."
            )
        aggregated = {
            "throughput/samples_per_second": float(global_samples / max_wall),
            "throughput/images_per_second": float(global_images / max_wall),
            "throughput/window_global_samples": int(global_samples),
            "throughput/window_global_images": int(global_images),
            "timing/window_wall_max": float(max_wall),
            "timing/rank_window_skew_ratio": float(
                max_wall
                / max(
                    min(float(payload["window_wall"]) for payload in payloads),
                    1e-12,
                )
            ),
            "data/valid_samples_window": len(sample_ids),
            "data/unique_samples_window": len(set(sample_ids)),
            "data/unique_sample_ratio_window": float(len(set(sample_ids)) / max(len(sample_ids), 1)),
            "data/sample_substitutions_window": 0,
        }
        for dataset_name, count in sorted(dataset_counts.items()):
            aggregated[f"data/dataset_fraction/{dataset_name}"] = float(count / max(len(sample_ids), 1))

        for field, prefix in (
            ("step_time", "timing/step_seconds"),
            ("data_time", "timing/data_seconds"),
            ("model_time", "timing/model_seconds"),
            ("eval_time", "timing/eval_seconds"),
            ("action_loss", "window/action_loss"),
            ("total_loss", "window/total_loss"),
            ("grad_norm", "optimization/grad_norm"),
        ):
            self._telemetry_distribution(
                aggregated,
                prefix,
                [record[field] for record in records],
            )
        aggregated["optimization/gradient_clipped_fraction"] = float(
            np.mean([record["gradient_clipped"] for record in records])
        )
        aggregated["optimization/overflow_fraction"] = float(
            np.mean([record["optimizer_overflow"] for record in records])
        )

        rank_step_skews = []
        for offset in range(steps_in_window):
            times = [float(payload["records"][offset]["step_time"]) for payload in payloads]
            rank_step_skews.append(max(times) / max(min(times), 1e-12))
        self._telemetry_distribution(
            aggregated,
            "timing/rank_step_skew_ratio",
            rank_step_skews,
        )

        gib = float(1024**3)
        for field in (
            "peak_allocated_bytes",
            "peak_reserved_bytes",
            "current_allocated_bytes",
            "current_reserved_bytes",
        ):
            values = [float(payload["cuda"][field]) / gib for payload in payloads]
            aggregated[f"gpu/{field.removesuffix('_bytes')}_gib_min"] = min(values)
            aggregated[f"gpu/{field.removesuffix('_bytes')}_gib_mean"] = float(np.mean(values))
            aggregated[f"gpu/{field.removesuffix('_bytes')}_gib_max"] = max(values)
        rss_values = [float(payload["process_maxrss_mib"]) for payload in payloads]
        aggregated["host/process_maxrss_mib_mean"] = float(np.mean(rss_values))
        aggregated["host/process_maxrss_mib_max"] = max(rss_values)
        cpu_values = [
            100.0 * float(payload["process_cpu"]) / max(float(payload["window_wall"]), 1e-12) for payload in payloads
        ]
        aggregated["host/process_cpu_util_pct_mean"] = float(np.mean(cpu_values))
        aggregated["host/process_cpu_util_pct_max"] = max(cpu_values)

        self._telemetry_records = []
        self._telemetry_sample_ids = []
        self._telemetry_dataset_counts = {}
        self._telemetry_violations = []
        self._telemetry_window_started_at = time.perf_counter()
        self._telemetry_process_cpu_started_at = time.process_time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        return aggregated

    def _log_metrics(self, metrics, *, force=False):
        """Send metrics to the tracker and the log every ``logging_frequency`` steps."""
        should_log = force or (self.completed_steps % int(self.config.trainer.resolved_logging_frequency) == 0)
        if not should_log:
            return
        last_lrs = self.lr_scheduler.get_last_lr()
        for i, group in enumerate(self.optimizer.param_groups):
            group_name = group.get("name", str(i))
            metrics[f"learning_rate/{group_name}"] = last_lrs[i] if i < len(last_lrs) else last_lrs[-1]
        metrics["sampling_epoch"] = round(self.train_microbatches_consumed / self.microbatches_per_sampling_epoch, 4)
        metrics["samples_seen"] = self.samples_seen
        self.tracker.log(metrics, step=self.completed_steps)
        if self.accelerator.is_main_process:
            logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    def _create_data_iterators(self):
        """Create data iterators."""
        self.vla_epoch_count = int(self._resume_state["sampling_epoch"]) if self._resume_state else 0
        resume_offset = int(self._resume_state["sampling_epoch_offset"]) if self._resume_state else 0
        expected_epoch, expected_offset = resolve_sampling_position(
            self.train_microbatches_consumed,
            self.microbatches_per_sampling_epoch,
        )
        if (self.vla_epoch_count, resume_offset) != (expected_epoch, expected_offset):
            raise RuntimeError(
                "Resolved train cursor disagrees with checkpoint counters: "
                f"state={self.vla_epoch_count}/{resume_offset}, "
                f"expected={expected_epoch}/{expected_offset}."
            )
        if resume_offset:
            if resume_offset >= len(self.vla_train_dataloader):
                raise RuntimeError(
                    f"Resume offset {resume_offset} is outside prepared epoch length "
                    f"{len(self.vla_train_dataloader)}."
                )
            # Accelerate 1.5 constructs a fresh DataLoaderShard whose iteration
            # defaults to zero. Set the saved epoch on the wrapper (not merely the
            # canonical loader) before iterating, or __iter__ would silently reset
            # LeRobotMixtureDataset.epoch to zero. SkipBatchSampler discards batch
            # indices without yielding them to workers, so the prefix is not decoded.
            resume_loader = self.accelerator.skip_first_batches(
                self.vla_train_dataloader,
                resume_offset,
            )
            if not callable(getattr(resume_loader, "set_epoch", None)):
                raise RuntimeError(
                    "Accelerate skip_first_batches returned a loader without set_epoch; "
                    "cannot prove epoch+offset resume semantics."
                )
            resume_loader.set_epoch(self.vla_epoch_count)
            # An uninterrupted mid-epoch run keeps existing workers, whereas resume
            # creates a new iterator and consumes host torch RNG for worker base_seed.
            # DROID samples/transforms are index-deterministic, so worker RNG is not
            # semantic; restore the parent RNG so model stochasticity resumes exactly.
            with _preserve_host_rng_state():
                self.vla_iter = iter(resume_loader)
                # DataLoaderShard.__iter__ is a lazy generator: worker creation and
                # base_seed consumption happen on first next(), not iter(). Prime
                # exactly one real (non-skipped) batch inside the RNG guard and cache
                # it for the training loop.
                try:
                    self._resume_prefetched_batch = next(self.vla_iter)
                except StopIteration as exc:
                    raise RuntimeError(
                        "skip_first_batches produced an empty epoch remainder despite "
                        f"offset={resume_offset} < len={len(self.vla_train_dataloader)}."
                    ) from exc
            self._resume_epoch_loader = resume_loader
            logger.info(
                "Reconstructed exact train cursor with prepared batch-sampler skip: "
                f"sampling_epoch={self.vla_epoch_count}, rank_microbatch_offset={resume_offset}"
            )
        else:
            if callable(getattr(self.vla_train_dataloader, "set_epoch", None)):
                self.vla_train_dataloader.set_epoch(self.vla_epoch_count)
            else:
                sampler = getattr(self.vla_train_dataloader, "sampler", None)
                if callable(getattr(sampler, "set_epoch", None)):
                    sampler.set_epoch(self.vla_epoch_count)
                dataset = getattr(self.vla_train_dataloader, "dataset", None)
                if callable(getattr(dataset, "set_epoch", None)):
                    dataset.set_epoch(self.vla_epoch_count)
            # At initial start or an exact epoch boundary, the uninterrupted run
            # also creates the next iterator, so its normal worker-seed RNG advance
            # must be preserved rather than compensated.
            self.vla_iter = iter(self.vla_train_dataloader)
            self._resume_epoch_loader = None
            self._resume_prefetched_batch = None

    def _get_next_batch(self):
        """Get next batch (automatically handle data loop)."""
        if getattr(self, "_resume_prefetched_batch", None) is not None:
            batch_vla = self._resume_prefetched_batch
            self._resume_prefetched_batch = None
        else:
            try:
                batch_vla = next(self.vla_iter)
            except StopIteration:
                if not hasattr(self, "vla_epoch_count"):
                    self.vla_epoch_count = 0
                # A resume SkipBatchSampler is valid only for the saved epoch
                # remainder. Reset the canonical prepared loader so subsequent
                # epochs never repeat the saved offset.
                self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                    self.vla_train_dataloader, self.vla_epoch_count
                )
                self._resume_epoch_loader = None
                batch_vla = next(self.vla_iter)

        self.train_microbatches_consumed += 1
        return batch_vla

    def _get_next_val_batch(self):
        """Get next hold-out validation batch (auto-reset at val-epoch end)."""
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

    def train(self):
        """Execute training loop."""
        self._log_training_config()
        self._create_data_iterators()
        progress_bar = tqdm(
            total=self.horizon.max_train_steps,
            initial=self.completed_steps,
            disable=not self.accelerator.is_local_main_process,
        )
        if self._telemetry_enabled:
            # Exclude model/DeepSpeed construction from steady-state throughput,
            # but include first-batch worker startup in the first training window.
            self._telemetry_records = []
            self._telemetry_sample_ids = []
            self._telemetry_dataset_counts = {}
            self._telemetry_violations = []
            self._telemetry_window_started_at = time.perf_counter()
            self._telemetry_process_cpu_started_at = time.process_time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

        while self.completed_steps < self.horizon.max_train_steps:
            t_start_step = time.perf_counter()
            t_start_data = t_start_step
            batch_vla = None
            batch_error = None
            try:
                batch_vla = self._get_next_batch()
            except Exception as exc:
                batch_error = f"{type(exc).__name__}: {exc}"
            _raise_if_any_rank_failed(
                self.accelerator,
                batch_error is not None,
                "training batch load",
                batch_error or "",
            )
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            # DeepSpeed engine drives accumulation; _train_step reports whether an
            # optimizer step actually happened this micro-batch (fall back to sync_gradients).
            optimizer_stepped = step_metrics.pop("_optimizer_step", self.accelerator.sync_gradients)
            if optimizer_stepped:
                progress_bar.update(1)
                self.completed_steps += 1
                self.samples_seen += self.total_batch_size

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    {
                        "data_times": f"{t_end_data - t_start_data:.3f}",
                        "model_times": f"{t_end_model - t_start_model:.3f}",
                    }
                )

            eval_time = 0.0
            if optimizer_stepped and self.completed_steps % self.config.trainer.resolved_eval_interval == 0:
                t_start_eval = time.perf_counter()
                step_metrics = self.eval_action_model(step_metrics)
                eval_time = time.perf_counter() - t_start_eval

            step_metrics["timing/data"] = t_end_data - t_start_data
            step_metrics["timing/model"] = t_end_model - t_start_model
            step_time = time.perf_counter() - t_start_step
            self._record_step_telemetry(
                batch=batch_vla,
                metrics=step_metrics,
                data_time=t_end_data - t_start_data,
                model_time=t_end_model - t_start_model,
                eval_time=eval_time,
                step_time=step_time,
            )
            if optimizer_stepped:
                if self.completed_steps % int(self.config.trainer.resolved_logging_frequency) == 0:
                    step_metrics.update(self._flush_step_telemetry())
                self._log_metrics(step_metrics)

            sampling_epoch_boundary = (
                optimizer_stepped
                and self._exact_resume_enabled
                and self.horizon.source == "num_sampling_epochs"
                and self.completed_steps > 0
                and self.train_microbatches_consumed % self.microbatches_per_sampling_epoch == 0
            )
            if sampling_epoch_boundary and self._last_saved_step != self.completed_steps:
                # The independent validation iterator is deterministic and reset
                # for every evaluation, so this end-of-iteration save captures RNG
                # after all step-N work while the train iterator remains exactly at
                # the next sampling-epoch boundary.
                self._save_checkpoint()

            if (
                optimizer_stepped
                and self.completed_steps % self.config.trainer.resolved_save_interval == 0
                and self.completed_steps > 0
                and self._last_saved_step != self.completed_steps
            ):
                self._save_checkpoint()

            if self.completed_steps >= self.horizon.max_train_steps:
                break

        if self._telemetry_enabled and self._telemetry_records:
            trailing_metrics = {
                "telemetry/final_partial_window": 1,
                **self._flush_step_telemetry(),
            }
            self._log_metrics(trailing_metrics, force=True)
        self._finalize_training()

    def eval_action_model(self, step_metrics: dict | None = None) -> dict:
        """Attach deterministic hold-out action MSE to metrics.

        With a hold-out val loader configured, averages the per-batch MSE over
        ``trainer.val_num_batches`` batches from the (un-sharded, deterministic)
        validation set. Without a real validation loader, no validation metric is
        emitted. predict_action runs on every rank (keeps collectives / iterators
        in lockstep); only rank 0 reads the scores.
        """
        if self.val_dataloader is None:
            return step_metrics
        num_batches = int(self.config.trainer.get("val_num_batches", 8))

        # Skip action eval when the robot action expert is disabled (e.g. an ego-only
        # single-source run): there is no robot action to predict/score.
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        if not getattr(unwrapped_model, "action_enabled", True):
            return step_metrics

        default_seed = int(self.config.seed) if hasattr(self.config, "seed") else 42
        eval_seed = int(self.config.trainer.get("eval_seed", default_seed))
        local_result = {"error": None, "digest": "action-eval-v2"}
        action_mse = None
        was_training = bool(unwrapped_model.training)
        try:
            # Enter before creating the iterator: DataLoader worker base-seed
            # construction itself consumes parent RNG. Restore all model-side RNG
            # afterward so changing eval cadence cannot change future training.
            with _preserve_host_rng_state():
                random.seed(eval_seed)
                np.random.seed(eval_seed % (2**32))
                cpu_generator = torch.Generator(device="cpu")
                cpu_generator.manual_seed(eval_seed)
                torch.random.set_rng_state(cpu_generator.get_state())
                if torch.cuda.is_available():
                    cuda_device = torch.cuda.current_device()
                    cuda_generator = torch.Generator(device=torch.device("cuda", cuda_device))
                    cuda_generator.manual_seed(eval_seed)
                    torch.cuda.set_rng_state(
                        cuda_generator.get_state(),
                        device=cuda_device,
                    )

                # Always evaluate the same fixed prefix under the dedicated RNG stream.
                self.val_epoch_count = 0
                self.val_iter = iter(self.val_dataloader)

                unwrapped_model.eval()
                action_mses = []
                with torch.inference_mode():
                    for _ in range(num_batches):
                        eval_batch = self._get_next_val_batch()
                        examples, _ = unpack_preprocessed_batch(eval_batch)
                        actions = np.asarray([example["action"] for example in examples])
                        output_dict = unwrapped_model.predict_action(
                            examples=eval_batch,
                            use_ddim=True,
                            num_ddim_steps=20,
                        )
                        if self.accelerator.is_main_process:
                            normalized_actions = np.asarray(output_dict["normalized_actions"])
                            action_mses.append(mean_action_mse(normalized_actions, actions))
                        del eval_batch, examples
                if self.accelerator.is_main_process:
                    action_mse = float(np.mean(action_mses))
                    if not np.isfinite(action_mse):
                        raise FloatingPointError(f"validation action MSE is non-finite: {action_mse!r}")
        except Exception as exc:
            local_result = {
                "error": f"{type(exc).__name__}: {exc}",
                "digest": None,
            }
        finally:
            unwrapped_model.train(was_training)

        _all_rank_result_consensus(
            self.accelerator,
            local_result,
            "deterministic action validation",
        )
        if self.accelerator.is_main_process:
            step_metrics["val/action_mse"] = action_mse
        return step_metrics

    def _log_training_config(self):
        """Record training config."""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Horizon source = {self.horizon.source}")
            logger.info(f"  Total optimization steps = {self.horizon.max_train_steps}")
            logger.info(f"  Dataset length = {self.horizon.dataset_length}")
            logger.info(f"  Steps per sampling epoch = {self.horizon.steps_per_sampling_epoch}")
            logger.info(f"  Microbatches per sampling epoch = {self.microbatches_per_sampling_epoch}")
            logger.info(f"  Sampling epochs = {self.horizon.num_sampling_epochs}")
            logger.info(f"  Planned samples = {self.horizon.planned_samples}")
            logger.info(f"  Warmup steps = {self.config.trainer.resolved_num_warmup_steps}")
            logger.info(f"  Logging interval = {self.config.trainer.resolved_logging_frequency}")
            logger.info(f"  Evaluation interval = {self.config.trainer.resolved_eval_interval}")
            logger.info(f"  Checkpoint interval = {self.config.trainer.resolved_save_interval}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.accelerator.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def _train_step(self, batch_vla, batch_vlm=None):
        """Execute single training step."""
        loss_scale = self.config.trainer.get("loss_scale", {}) if hasattr(self.config.trainer, "get") else {}

        def _sum_losses(out):
            # Sum every "*_loss" entry, each scaled by trainer.loss_scale (default 1.0).
            return sum(
                v * float(loss_scale.get(k, 1.0))
                for k, v in out.items()
                if isinstance(v, torch.Tensor) and k.endswith("_loss")
            )

        def _forward_with_rank_health():
            output_dict = None
            total_loss = None
            local_error = None
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output_dict = self.model.forward(
                        batch_vla,
                        loss_keys=self._loss_keys,
                    )
                    total_loss = _sum_losses(output_dict)
                if not bool(torch.isfinite(total_loss.detach()).all()):
                    local_error = (
                        f"non-finite total loss at optimizer_step="
                        f"{self.completed_steps + 1}; "
                        f"loss_keys={sorted(output_dict)}"
                    )
            except Exception as exc:
                local_error = f"{type(exc).__name__}: {exc}"
            _raise_if_any_rank_failed(
                self.accelerator,
                local_error is not None,
                "training forward/loss",
                local_error or "",
                FloatingPointError,
            )
            return output_dict, total_loss

        # datasets.vla_data.losses restricts the forward pass to those objectives (e.g.
        # [action_loss] or [action_loss, gen_loss]); unset computes every enabled loss.
        if not hasattr(self, "_loss_keys"):
            _lk = self.config.datasets.vla_data.get("losses", None)
            self._loss_keys = set(_lk) if _lk is not None else None

        # DeepSpeed (ZeRO stage 2/3) forbids Accelerate's accumulate()/no_sync() under
        # gradient partitioning. When the prepared model is a DeepSpeed engine, drive
        # accumulation through the engine: backward() accumulates and step() only updates
        # on the engine's own boundary (gradient clipping handled via ds_config), as in
        # train_starvla_gen_cotrain.py.
        grad_norm = None
        optimizer_overflow = False
        if hasattr(self.model, "is_gradient_accumulation_boundary") and hasattr(self.model, "backward"):
            output_dict, total_loss = _forward_with_rank_health()
            self.model.backward(total_loss)
            optimizer_stepped = bool(self.model.is_gradient_accumulation_boundary())
            self.model.step()
            if optimizer_stepped:
                optimizer_overflow = bool(getattr(getattr(self.model, "optimizer", None), "overflow", False))
                health_errors = []
                if optimizer_overflow:
                    health_errors.append("DeepSpeed reported gradient overflow")
                get_global_grad_norm = getattr(self.model, "get_global_grad_norm", None)
                if callable(get_global_grad_norm):
                    try:
                        resolved_grad_norm = get_global_grad_norm()
                        if resolved_grad_norm is not None:
                            grad_norm = float(resolved_grad_norm)
                    except Exception as exc:
                        health_errors.append(f"gradient norm read failed: {type(exc).__name__}: {exc}")
                if grad_norm is not None and not np.isfinite(grad_norm):
                    health_errors.append(f"non-finite gradient norm {grad_norm!r}")
                _raise_if_any_rank_failed(
                    self.accelerator,
                    bool(health_errors),
                    "DeepSpeed optimizer health",
                    "; ".join(health_errors),
                    FloatingPointError,
                )
                self.lr_scheduler.step()
        else:
            # Non-DeepSpeed fallback: Accelerate's accumulate() (uses no_sync()).
            with self.accelerator.accumulate(self.model):
                self.optimizer.zero_grad()
                output_dict, total_loss = _forward_with_rank_health()
                self.accelerator.backward(total_loss)
                if self.config.trainer.gradient_clipping is not None:
                    resolved_grad_norm = self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.trainer.gradient_clipping,
                    )
                    if resolved_grad_norm is not None:
                        grad_norm = float(resolved_grad_norm)
                _raise_if_any_rank_failed(
                    self.accelerator,
                    grad_norm is not None and not np.isfinite(grad_norm),
                    "optimizer gradient health",
                    f"non-finite gradient norm {grad_norm!r}",
                    FloatingPointError,
                )
                self.optimizer.step()
                if self.accelerator.sync_gradients:
                    self.lr_scheduler.step()
            optimizer_stepped = self.accelerator.sync_gradients

        metrics = {k: v.item() for k, v in output_dict.items() if isinstance(v, torch.Tensor) and k.endswith("_loss")}
        metrics["total_loss"] = total_loss.item()
        metrics["optimization/overflow"] = float(optimizer_overflow)
        if grad_norm is not None:
            metrics["optimization/grad_norm"] = grad_norm
            clipping_threshold = self.config.trainer.get(
                "gradient_clipping",
                self.config.trainer.get("max_grad_norm", None),
            )
            metrics["optimization/gradient_clipped"] = float(
                clipping_threshold is not None and grad_norm > float(clipping_threshold)
            )
        metrics["_optimizer_step"] = optimizer_stepped
        return metrics

    def _finalize_training(self):
        """Save the final full state (exact mode) and ``final_model/``."""
        if self._exact_resume_enabled:
            save_decision = _rank0_consensus(
                self.accelerator,
                (
                    {"save_full_state": self._last_saved_step != self.completed_steps}
                    if self.accelerator.is_main_process
                    else None
                ),
            )
            if save_decision["save_full_state"]:
                self._save_checkpoint()

        # Under ZeRO, state consolidation may invoke collectives. All ranks enter;
        # only rank 0 writes the portable artifact.
        state_dict = self.accelerator.get_state_dict(self.model)
        final_result = None
        if self.accelerator.is_main_process:
            try:
                save_format = getattr(self.config.trainer, "save_format", "pt")
                final_checkpoint = os.path.join(self.config.output_dir, "final_model")
                os.makedirs(final_checkpoint, exist_ok=True)
                if save_format == "safetensors":
                    from safetensors.torch import save_file

                    final_path = os.path.join(final_checkpoint, "model.safetensors")
                    tmp_path = final_path + ".tmp"
                    save_file(state_dict, tmp_path)
                elif save_format == "pt":
                    final_path = os.path.join(final_checkpoint, "pytorch_model.pt")
                    tmp_path = final_path + ".tmp"
                    torch.save(state_dict, tmp_path)
                else:
                    raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
                os.replace(tmp_path, final_path)
                logger.info(f"Training complete. Final model saved at {final_checkpoint}")
                final_result = {"error": None}
            except Exception as exc:
                final_result = {"error": f"{type(exc).__name__}: {exc}"}
        final_result = _rank0_consensus(self.accelerator, final_result)
        if final_result["error"]:
            raise RuntimeError(f"Final portable-model publication failed: {final_result['error']}")

        self.tracker.finish()
        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    logger.info("VLA Training :: Warming Up")

    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    accelerator = build_accelerator(cfg)
    accelerator.print(accelerator.state)

    output_dir = setup_directories(cfg=cfg, accelerator=accelerator)
    # Build every rank from the same deterministic initialization. prepare_training()
    # later resets to seed + rank so runtime stochasticity remains rank-specific and
    # independent of how much RNG a particular model variant consumed while building.
    model_init_seed = cfg.seed if hasattr(cfg, "seed") else 3047
    set_seed(model_init_seed)
    vla = build_framework(cfg)
    batch_preprocessor = build_worker_batch_preprocessor(cfg, vla)
    vla_train_dataloader, val_dataloader = prepare_data(
        cfg=cfg,
        accelerator=accelerator,
        output_dir=output_dir,
        batch_preprocessor=batch_preprocessor,
    )
    horizon, split_contract, provenance, fingerprint = resolve_and_record_training_horizon(
        cfg, vla_train_dataloader, accelerator
    )
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        horizon=horizon,
        split_contract=split_contract,
        provenance=provenance,
        fingerprint=fingerprint,
    )

    trainer.prepare_training()
    trainer.train()

    logger.info("... and that's all, folks!")
    distributed_barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single-stream VLA training. Extra `--dotted.key value` arguments override the config."
    )
    parser.add_argument("--config_yaml", type=str, required=True, help="Path to the training YAML config.")
    args, clipargs = parser.parse_known_args()

    cfg = load_config_with_bases(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # Normalize older config key spellings into the current `version_id == "0.22"` schema
    # (idempotent).
    cfg = apply_config_compat(cfg)

    # Propagate datasets.shared.* into every stream and derive each stream's future-frame
    # offsets from framework.gen_model.gen_horizon_seconds (generation runs only).
    cfg = align_dataset_configs(cfg)

    # Resolve `datasets.total_per_device_batch_size` into the stream batch size (no-op when
    # the key is absent).
    resolve_stream_batch_sizes(cfg, ["vla_data"])

    cfg.config_yaml = args.config_yaml

    main(cfg)
