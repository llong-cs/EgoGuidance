"""Pure helpers for resolving and auditing long-running training state.

This module intentionally depends only on the Python standard library so its
semantic checks can be unit-tested without importing torch/accelerate.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Any, Mapping

TRAINER_STATE_VERSION = 3
COMPLETE_MARKER = "checkpoint.complete"
TRAINER_STATE_FILE = "trainer_state.json"
_CHECKPOINT_RE = re.compile(r"^steps_(\d+)$")
_DETERMINISTIC_ACTION_TRANSFORMS = frozenset(
    {
        "StateActionToTensor",
        "StateActionTransform",
    }
)
# Provenance recorded for reference only. Resuming with different code is allowed (the
# trainer logs a warning); every other provenance entry must match exactly.
_ADVISORY_PROVENANCE_KEYS = ("code",)
# Config fields that describe where/how a run is launched or logged rather than what it
# trains; they are excluded from the resume fingerprint.
_RUNTIME_ONLY_TOP_LEVEL_KEYS = (
    "output_dir",
    "run_root_dir",
    "run_id",
    "config_yaml",
    "wandb_project",
    "wandb_entity",
    "wandb_run_name",
)
_RUNTIME_ONLY_TRAINER_KEYS = (
    "is_resume",
    "resume_from_checkpoint",
    "config_fingerprint",
    "disable_tracking",
    "resolved_code_commit",
    "resolved_code_dirty_sha256",
)


@dataclass(frozen=True)
class TrainingHorizon:
    """Resolved optimizer-step horizon and its physical accounting."""

    source: str
    max_train_steps: int
    num_sampling_epochs: int | None
    dataset_length: int
    effective_global_batch_size: int
    steps_per_sampling_epoch: int
    requested_samples: int
    planned_samples: int
    padding_samples: int
    world_size: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_sampling_position(
    train_microbatches_consumed: Any,
    microbatches_per_sampling_epoch: Any,
) -> tuple[int, int]:
    """Resolve an absolute train-batch cursor into ``(epoch, batch offset)``."""

    if (
        isinstance(train_microbatches_consumed, bool)
        or not isinstance(train_microbatches_consumed, Integral)
        or int(train_microbatches_consumed) < 0
    ):
        raise ValueError(
            "train_microbatches_consumed must be a non-negative integer, got " f"{train_microbatches_consumed!r}."
        )
    microbatches_per_sampling_epoch = _positive_integer(
        microbatches_per_sampling_epoch,
        "microbatches per sampling epoch",
    )
    return divmod(int(train_microbatches_consumed), microbatches_per_sampling_epoch)


def build_data_resume_contract(
    *,
    dataset_type: str,
    mode: str,
    sample_error_policy: str,
    sampler_type: str,
    drop_last: bool,
    dataset_py: str,
    generation_enabled: bool,
    transform_names_by_dataset: list[list[str]],
) -> dict[str, Any]:
    """Audit the narrow map-style action-only contract supported by exact skip.

    Mid-epoch restart intentionally does not recreate already-consumed workers or
    decode the skipped prefix. It is therefore exact only when the sample is a
    pure function of ``(sampling_epoch, dataset_index)`` and worker RNG does not
    affect transforms or error recovery.
    """

    reasons = []
    if dataset_type != "LeRobotMixtureDataset":
        reasons.append(f"unsupported dataset type {dataset_type!r}")
    if mode != "train":
        reasons.append(f"expected train mode, got {mode!r}")
    if sample_error_policy != "raise":
        reasons.append(
            "sample_error_policy is not 'raise'; decode failures may substitute "
            "a different sample for the same dataset index"
        )
    if sampler_type != "SequentialSampler":
        reasons.append(f"sampler {sampler_type!r} is not index-stable")
    if drop_last:
        reasons.append("drop_last=true is not supported")
    if dataset_py != "lerobot_datasets":
        reasons.append(f"dataset_py {dataset_py!r} is not the action-only LeRobot loader")
    if generation_enabled:
        reasons.append("generation head is enabled; exact resume currently supports action-only training")

    unexpected_transforms = sorted(
        {name for names in transform_names_by_dataset for name in names if name not in _DETERMINISTIC_ACTION_TRANSFORMS}
    )
    if unexpected_transforms:
        reasons.append("transform determinism is not proven for " + ", ".join(unexpected_transforms))

    return {
        "version": 1,
        "exact": not reasons,
        "reason": "; ".join(reasons) if reasons else None,
        "position_key": ["sampling_epoch", "dataset_index"],
        "resume_skip": "accelerator.skip_first_batches",
        "skip_decodes_prefix": False,
        "dataset_type": str(dataset_type),
        "mode": str(mode),
        "sample_error_policy": str(sample_error_policy),
        "sampler_type": str(sampler_type),
        "drop_last": bool(drop_last),
        "dataset_py": str(dataset_py),
        "generation_enabled": bool(generation_enabled),
        "transform_names_by_dataset": copy.deepcopy(transform_names_by_dataset),
        "allowed_transform_types": sorted(_DETERMINISTIC_ACTION_TRANSFORMS),
    }


def _positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{field} must be a positive integer, got {value!r}.")
    return int(value)


def resolve_progress_interval(
    *,
    absolute_field: str,
    ratio_field: str,
    absolute_value: Any,
    ratio_value: Any,
    total_steps: Any,
    allow_zero_absolute: bool = False,
) -> int:
    """Resolve one absolute-or-relative training schedule value into steps."""

    has_absolute = absolute_value is not None
    has_ratio = ratio_value is not None
    if has_absolute == has_ratio:
        field_pair = f"trainer.{absolute_field} and trainer.{ratio_field}"
        if has_absolute:
            raise ValueError(f"{field_pair} are mutually exclusive; set exactly one.")
        raise ValueError(f"Training schedule is undefined: set exactly one of {field_pair}.")

    total_steps = _positive_integer(total_steps, "resolved training horizon")
    if has_ratio:
        if (
            isinstance(ratio_value, bool)
            or not isinstance(ratio_value, Real)
            or not math.isfinite(float(ratio_value))
            or not 0 < float(ratio_value) <= 1
        ):
            raise ValueError(f"trainer.{ratio_field} must be finite and in (0, 1], got {ratio_value!r}.")
        return max(1, round(float(ratio_value) * total_steps))

    minimum = 0 if allow_zero_absolute else 1
    if isinstance(absolute_value, bool) or not isinstance(absolute_value, Integral) or int(absolute_value) < minimum:
        qualifier = "a non-negative" if allow_zero_absolute else "a positive"
        raise ValueError(f"trainer.{absolute_field} must be {qualifier} integer, got {absolute_value!r}.")
    return int(absolute_value)


def resolve_training_horizon(
    *,
    max_train_steps: Any,
    num_sampling_epochs: Any,
    dataset_length: Any,
    effective_global_batch_size: Any,
    world_size: Any,
) -> TrainingHorizon:
    """Resolve exactly one user-facing horizon into optimizer steps.

    ``num_sampling_epochs`` means complete nominal passes through the map-style
    dataset. Each pass therefore has ``ceil(dataset_length / global_batch)``
    optimizer steps; multiplying that per-pass value (rather than rounding only
    once after multiplying samples) preserves the requested number of passes.
    """

    has_steps = max_train_steps is not None
    has_epochs = num_sampling_epochs is not None
    if has_steps == has_epochs:
        if has_steps:
            raise ValueError(
                "trainer.max_train_steps and trainer.num_sampling_epochs are mutually exclusive; "
                "set exactly one and set the other to null."
            )
        raise ValueError(
            "Training horizon is undefined: set exactly one of trainer.max_train_steps "
            "or trainer.num_sampling_epochs to a positive integer."
        )

    dataset_length = _positive_integer(dataset_length, "training dataset length")
    effective_global_batch_size = _positive_integer(effective_global_batch_size, "effective global batch size")
    world_size = _positive_integer(world_size, "world size")
    steps_per_epoch = math.ceil(dataset_length / effective_global_batch_size)

    if has_epochs:
        epochs = _positive_integer(num_sampling_epochs, "trainer.num_sampling_epochs")
        steps = steps_per_epoch * epochs
        requested_samples = dataset_length * epochs
        source = "num_sampling_epochs"
    else:
        epochs = None
        steps = _positive_integer(max_train_steps, "trainer.max_train_steps")
        requested_samples = steps * effective_global_batch_size
        source = "max_train_steps"

    planned_samples = steps * effective_global_batch_size
    return TrainingHorizon(
        source=source,
        max_train_steps=steps,
        num_sampling_epochs=epochs,
        dataset_length=dataset_length,
        effective_global_batch_size=effective_global_batch_size,
        steps_per_sampling_epoch=steps_per_epoch,
        requested_samples=requested_samples,
        planned_samples=planned_samples,
        padding_samples=max(0, planned_samples - requested_samples),
        world_size=world_size,
    )


def _without_runtime_only_fields(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the resolved training config without launch/resume/code-identity fields."""

    normalized = copy.deepcopy(dict(config))
    for key in _RUNTIME_ONLY_TOP_LEVEL_KEYS:
        normalized.pop(key, None)

    trainer = normalized.get("trainer")
    if isinstance(trainer, Mapping):
        trainer = dict(trainer)
        for key in _RUNTIME_ONLY_TRAINER_KEYS:
            trainer.pop(key, None)
        normalized["trainer"] = trainer
    return normalized


def config_fingerprint(config: Mapping[str, Any]) -> str:
    """Hash the resolved training config, excluding launch/resume controls and code identity."""

    payload = json.dumps(
        _without_runtime_only_fields(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_trainer_state(
    *,
    completed_steps: int,
    samples_seen: int,
    train_microbatches_consumed: int,
    horizon: TrainingHorizon,
    gradient_accumulation_steps: int,
    split: Mapping[str, Any],
    provenance: Mapping[str, Any],
    fingerprint: str,
) -> dict[str, Any]:
    """Build an auditable state record for one optimizer-boundary checkpoint."""

    completed_steps = int(completed_steps)
    samples_seen = int(samples_seen)
    train_microbatches_consumed = int(train_microbatches_consumed)
    gradient_accumulation_steps = _positive_integer(gradient_accumulation_steps, "gradient accumulation steps")
    if completed_steps < 0 or completed_steps > horizon.max_train_steps:
        raise ValueError(f"completed_steps={completed_steps} is outside [0, {horizon.max_train_steps}].")
    if samples_seen < 0:
        raise ValueError(f"samples_seen must be non-negative, got {samples_seen}.")
    if train_microbatches_consumed < 0:
        raise ValueError("train_microbatches_consumed must be non-negative, got " f"{train_microbatches_consumed}.")

    microbatches_per_sampling_epoch = math.ceil(
        horizon.dataset_length * gradient_accumulation_steps / horizon.effective_global_batch_size
    )
    sampling_epoch, sampling_epoch_offset = resolve_sampling_position(
        train_microbatches_consumed,
        microbatches_per_sampling_epoch,
    )
    data_resume_contract = provenance.get("data_resume_contract", {})
    data_order_exact = bool(isinstance(data_resume_contract, Mapping) and data_resume_contract.get("exact", False))
    exact_position = (
        gradient_accumulation_steps == 1 and train_microbatches_consumed == completed_steps and data_order_exact
    )
    if gradient_accumulation_steps != 1:
        exact_reason = (
            "gradient_accumulation_steps != 1: optimizer boundaries can cross a "
            "DataLoader epoch boundary and worker/iterator state is not checkpointed"
        )
    elif train_microbatches_consumed != completed_steps:
        exact_reason = "the training DataLoader was also consumed outside optimizer steps"
    elif not data_order_exact:
        exact_reason = (
            "the data pipeline does not prove deterministic epoch+index sampling: "
            f"{data_resume_contract.get('reason', 'missing data resume contract')}"
        )
    else:
        exact_reason = None

    return {
        "version": TRAINER_STATE_VERSION,
        "checkpoint_type": "full_training_state",
        "completed_steps": completed_steps,
        "samples_seen": samples_seen,
        "train_microbatches_consumed": train_microbatches_consumed,
        "sampling_epoch": sampling_epoch,
        "sampling_epoch_offset": sampling_epoch_offset,
        "sampling_epoch_offset_unit": "rank_microbatches",
        "exact_data_position": exact_position,
        "exact_data_position_reason": exact_reason,
        "resume_strategy": ("prepared_batch_sampler_skip_v1" if data_order_exact else None),
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "world_size": horizon.world_size,
        "effective_global_batch_size": horizon.effective_global_batch_size,
        "dataset_length": horizon.dataset_length,
        "steps_per_sampling_epoch": horizon.steps_per_sampling_epoch,
        "microbatches_per_sampling_epoch": microbatches_per_sampling_epoch,
        "resolved_max_train_steps": horizon.max_train_steps,
        "horizon_source": horizon.source,
        "num_sampling_epochs": horizon.num_sampling_epochs,
        "split": copy.deepcopy(dict(split)),
        "provenance": copy.deepcopy(dict(provenance)),
        "config_fingerprint": str(fingerprint),
    }


def _strict_provenance(provenance: Any) -> Any:
    if not isinstance(provenance, Mapping):
        return provenance
    return {key: value for key, value in provenance.items() if key not in _ADVISORY_PROVENANCE_KEYS}


def describe_code_change(state: Mapping[str, Any], provenance: Mapping[str, Any]) -> str | None:
    """Return a warning when a checkpoint was written by different code, else None."""

    saved = (state.get("provenance") or {}).get("code")
    current = provenance.get("code")
    if saved == current:
        return None
    return (
        "Resuming with code that differs from the checkpoint: "
        f"checkpoint={json.dumps(saved, sort_keys=True)}, current={json.dumps(current, sort_keys=True)}."
    )


def validate_resume_state(
    state: Mapping[str, Any],
    *,
    horizon: TrainingHorizon,
    split: Mapping[str, Any],
    provenance: Mapping[str, Any],
    fingerprint: str,
) -> None:
    """Reject any resume whose optimizer/data contract cannot be proven equal.

    Code identity (Git commit / uncommitted changes) is not part of the contract; use
    :func:`describe_code_change` to report it.
    """

    required = {
        "version",
        "checkpoint_type",
        "completed_steps",
        "samples_seen",
        "train_microbatches_consumed",
        "sampling_epoch",
        "sampling_epoch_offset",
        "microbatches_per_sampling_epoch",
        "exact_data_position",
        "resume_strategy",
        "gradient_accumulation_steps",
        "world_size",
        "effective_global_batch_size",
        "dataset_length",
        "resolved_max_train_steps",
        "split",
        "provenance",
        "config_fingerprint",
    }
    missing = sorted(required.difference(state))
    if missing:
        raise ValueError(f"Full-state checkpoint trainer_state is missing fields: {missing}.")
    if state["version"] != TRAINER_STATE_VERSION:
        raise ValueError(
            f"Unsupported trainer_state version {state['version']!r}; " f"expected {TRAINER_STATE_VERSION}."
        )
    if state["checkpoint_type"] != "full_training_state":
        raise ValueError("Resume requires checkpoint_type='full_training_state'; " f"got {state['checkpoint_type']!r}.")

    comparisons = {
        "world_size": horizon.world_size,
        "effective_global_batch_size": horizon.effective_global_batch_size,
        "dataset_length": horizon.dataset_length,
        "resolved_max_train_steps": horizon.max_train_steps,
        "split": dict(split),
        "provenance": _strict_provenance(dict(provenance)),
        "config_fingerprint": fingerprint,
    }
    stored = {key: state.get(key) for key in comparisons}
    stored["provenance"] = _strict_provenance(stored["provenance"])
    mismatches = {
        key: {"checkpoint": stored[key], "current": expected}
        for key, expected in comparisons.items()
        if stored[key] != expected
    }
    if mismatches:
        raise ValueError(
            "Resume contract mismatch; refusing to change training semantics: " + json.dumps(mismatches, sort_keys=True)
        )

    if not bool(state["exact_data_position"]):
        reason = state.get("exact_data_position_reason") or "unknown reason"
        raise ValueError(
            "This checkpoint contains model/optimizer/scheduler/RNG state but cannot "
            "resume the exact next sample: "
            f"{reason}. Use a checkpoint produced with gradient_accumulation_steps=1 "
            "and a proven deterministic epoch+index data contract, or use its portable "
            "model file as trainer.pretrained_checkpoint for an intentional weights-only start."
        )
    if state["resume_strategy"] != "prepared_batch_sampler_skip_v1":
        raise ValueError(
            "Exact resume requires resume_strategy='prepared_batch_sampler_skip_v1'; "
            f"got {state['resume_strategy']!r}."
        )

    completed_steps = int(state["completed_steps"])
    if completed_steps < 0 or completed_steps > horizon.max_train_steps:
        raise ValueError(
            f"Checkpoint completed_steps={completed_steps} is outside the resolved horizon "
            f"[0, {horizon.max_train_steps}]."
        )
    expected_samples = completed_steps * horizon.effective_global_batch_size
    if int(state["samples_seen"]) != expected_samples:
        raise ValueError(
            f"Checkpoint samples_seen={state['samples_seen']} disagrees with "
            f"completed_steps*effective_global_batch_size={expected_samples}."
        )
    train_microbatches_consumed = int(state["train_microbatches_consumed"])
    if train_microbatches_consumed != completed_steps:
        raise ValueError(
            "Exact resume requires train_microbatches_consumed == completed_steps when "
            "gradient_accumulation_steps=1; got "
            f"{train_microbatches_consumed} != {completed_steps}."
        )
    state_grad_accum = int(state.get("gradient_accumulation_steps", 0))
    if state_grad_accum != 1:
        raise ValueError("Exact DataLoader resume currently requires checkpoint " "gradient_accumulation_steps=1.")
    expected_microbatches_per_epoch = math.ceil(
        horizon.dataset_length * state_grad_accum / horizon.effective_global_batch_size
    )
    if int(state["microbatches_per_sampling_epoch"]) != expected_microbatches_per_epoch:
        raise ValueError(
            "Checkpoint microbatches_per_sampling_epoch disagrees with the resolved "
            f"data contract: stored={state['microbatches_per_sampling_epoch']}, "
            f"expected={expected_microbatches_per_epoch}."
        )
    expected_epoch, expected_offset = resolve_sampling_position(
        train_microbatches_consumed,
        expected_microbatches_per_epoch,
    )
    if int(state["sampling_epoch"]) != expected_epoch or int(state["sampling_epoch_offset"]) != expected_offset:
        raise ValueError(
            "Checkpoint sampling position disagrees with train_microbatches_consumed: "
            f"stored epoch/offset={state['sampling_epoch']}/{state['sampling_epoch_offset']}, "
            f"expected={expected_epoch}/{expected_offset}."
        )


def checkpoint_step(path: str) -> int | None:
    match = _CHECKPOINT_RE.match(os.path.basename(os.path.normpath(path)))
    return int(match.group(1)) if match else None


def load_complete_checkpoint_metadata(path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Parse and cross-check an atomically published checkpoint manifest."""

    step = checkpoint_step(path)
    if not os.path.isdir(path) or step is None:
        raise ValueError(f"Not a steps_<N> checkpoint directory: {path!r}.")

    marker_path = os.path.join(path, COMPLETE_MARKER)
    state_path = os.path.join(path, TRAINER_STATE_FILE)
    if not os.path.isfile(marker_path) or not os.path.isfile(state_path):
        raise ValueError(f"Checkpoint {path!r} has no complete marker/trainer state.")
    try:
        with open(marker_path) as f:
            marker = json.load(f)
        with open(state_path) as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Checkpoint {path!r} has unreadable completion metadata: {exc}") from exc

    if marker.get("version") != TRAINER_STATE_VERSION:
        raise ValueError(f"Checkpoint {path!r} has unsupported marker version {marker.get('version')!r}.")
    if state.get("version") != TRAINER_STATE_VERSION or state.get("checkpoint_type") != "full_training_state":
        raise ValueError(
            f"Checkpoint {path!r} has invalid trainer-state identity: "
            f"version={state.get('version')!r}, type={state.get('checkpoint_type')!r}."
        )
    marker_step = marker.get("completed_steps")
    state_step = state.get("completed_steps")
    if marker_step != step or state_step != step:
        raise ValueError(
            f"Checkpoint step mismatch for {path!r}: dirname={step}, "
            f"marker={marker_step!r}, trainer_state={state_step!r}."
        )

    portable_name = marker.get("portable_model")
    trainer_state_name = marker.get("trainer_state")
    payload_files = marker.get("payload_files")
    accelerate_payload_files = marker.get("accelerate_payload_files")
    state_layout = marker.get("state_layout")
    if not isinstance(portable_name, str) or not portable_name:
        raise ValueError(f"Checkpoint {path!r} marker has no portable_model.")
    if trainer_state_name != TRAINER_STATE_FILE:
        raise ValueError(
            f"Checkpoint {path!r} marker names trainer state {trainer_state_name!r}; "
            f"expected {TRAINER_STATE_FILE!r}."
        )
    if not isinstance(payload_files, dict) or not payload_files:
        raise ValueError(f"Checkpoint {path!r} marker has no payload manifest.")
    if not isinstance(accelerate_payload_files, list) or not accelerate_payload_files:
        raise ValueError(f"Checkpoint {path!r} marker has no Accelerate/DeepSpeed state payload.")
    if not isinstance(state_layout, dict):
        raise ValueError(f"Checkpoint {path!r} marker has no full-state layout contract.")
    if TRAINER_STATE_FILE not in payload_files or portable_name not in payload_files:
        raise ValueError(
            f"Checkpoint {path!r} payload manifest must contain {TRAINER_STATE_FILE!r} "
            f"and portable model {portable_name!r}."
        )

    root = os.path.realpath(path)
    for relative_path, expected_size in payload_files.items():
        if not isinstance(relative_path, str) or os.path.isabs(relative_path):
            raise ValueError(f"Checkpoint {path!r} has invalid payload path {relative_path!r}.")
        if isinstance(expected_size, bool) or not isinstance(expected_size, Integral) or expected_size <= 0:
            raise ValueError(
                f"Checkpoint {path!r} has invalid payload size for " f"{relative_path!r}: {expected_size!r}."
            )
        full_path = os.path.realpath(os.path.join(path, relative_path))
        if os.path.commonpath([root, full_path]) != root:
            raise ValueError(f"Checkpoint payload escapes its directory: {relative_path!r}.")
        if not os.path.isfile(full_path):
            raise ValueError(f"Checkpoint {path!r} is missing payload {relative_path!r}.")
        actual_size = os.path.getsize(full_path)
        if actual_size != expected_size:
            raise ValueError(
                f"Checkpoint payload size mismatch for {relative_path!r}: "
                f"expected={expected_size}, actual={actual_size}."
            )
    if len(set(accelerate_payload_files)) != len(accelerate_payload_files):
        raise ValueError(f"Checkpoint {path!r} repeats an Accelerate payload path.")
    for relative_path in accelerate_payload_files:
        if not isinstance(relative_path, str):
            raise ValueError(f"Checkpoint {path!r} has invalid Accelerate payload path " f"{relative_path!r}.")
        if relative_path not in payload_files:
            raise ValueError(f"Checkpoint {path!r} names unmanifested Accelerate payload {relative_path!r}.")
        if relative_path in {TRAINER_STATE_FILE, portable_name}:
            raise ValueError(
                f"Checkpoint {path!r} incorrectly classifies {relative_path!r} " "as an Accelerate payload."
            )

    layout_world_size = state_layout.get("world_size")
    if (
        isinstance(layout_world_size, bool)
        or not isinstance(layout_world_size, Integral)
        or int(layout_world_size) <= 0
        or int(layout_world_size) != int(state.get("world_size", 0))
    ):
        raise ValueError(
            f"Checkpoint {path!r} state-layout world size {layout_world_size!r} "
            f"disagrees with trainer state {state.get('world_size')!r}."
        )
    world_size = int(layout_world_size)
    backend = state_layout.get("backend")
    rng_files = state_layout.get("rng_files")
    scheduler_files = state_layout.get("scheduler_files")
    model_state_files = state_layout.get("model_state_files")
    optimizer_state_files = state_layout.get("optimizer_state_files")
    expected_rng = [f"random_states_{rank}.pkl" for rank in range(world_size)]
    if rng_files != expected_rng:
        raise ValueError(
            f"Checkpoint {path!r} must contain one ordered RNG state per rank: "
            f"stored={rng_files!r}, expected={expected_rng!r}."
        )
    if scheduler_files != ["custom_checkpoint_0.pkl"]:
        raise ValueError(
            f"Checkpoint {path!r} must contain the registered scheduler state "
            f"custom_checkpoint_0.pkl; got {scheduler_files!r}."
        )
    for role, role_files in (
        ("rng", rng_files),
        ("scheduler", scheduler_files),
        ("model", model_state_files),
        ("optimizer", optimizer_state_files),
    ):
        if not isinstance(role_files, list) or not role_files:
            raise ValueError(f"Checkpoint {path!r} has no {role} state files.")
        if not set(role_files).issubset(accelerate_payload_files):
            raise ValueError(
                f"Checkpoint {path!r} {role} state names files outside the " f"Accelerate payload: {role_files!r}."
            )

    if backend == "DEEPSPEED":
        tag_file = state_layout.get("deepspeed_tag_file")
        tag = state_layout.get("deepspeed_tag")
        if tag_file != "latest" or tag != "pytorch_model" or tag_file not in accelerate_payload_files:
            raise ValueError(
                f"Checkpoint {path!r} has invalid DeepSpeed tag contract: " f"tag_file={tag_file!r}, tag={tag!r}."
            )
        try:
            with open(os.path.join(path, tag_file)) as f:
                actual_tag = f.read().strip()
        except OSError as exc:
            raise ValueError(f"Checkpoint {path!r} cannot read DeepSpeed tag file: {exc}") from exc
        if actual_tag != tag:
            raise ValueError(f"Checkpoint {path!r} DeepSpeed tag file contains {actual_tag!r}; expected {tag!r}.")
        if not all(name.startswith(f"{tag}/") and name.endswith("model_states.pt") for name in model_state_files):
            raise ValueError(f"Checkpoint {path!r} has invalid DeepSpeed model-state paths.")
        if len(optimizer_state_files) < world_size or not all(
            name.startswith(f"{tag}/") and name.endswith("optim_states.pt") for name in optimizer_state_files
        ):
            raise ValueError(
                f"Checkpoint {path!r} has {len(optimizer_state_files)} DeepSpeed optimizer "
                f"shards for world_size={world_size}."
            )
    elif backend in {"MULTI_GPU", "NO"}:
        if not all(name in {"model.safetensors", "pytorch_model.bin"} for name in model_state_files):
            raise ValueError(f"Checkpoint {path!r} has invalid Accelerate model-state paths.")
        if not all(re.fullmatch(r"optimizer(?:_\d+)?\.bin", name) for name in optimizer_state_files):
            raise ValueError(f"Checkpoint {path!r} has invalid Accelerate optimizer-state paths.")
    else:
        raise ValueError(f"Checkpoint {path!r} has unsupported state backend {backend!r}.")

    control_sha256 = state_layout.get("control_sha256")
    required_control = {TRAINER_STATE_FILE, *rng_files, *scheduler_files}
    if backend == "DEEPSPEED":
        required_control.add("latest")
    if not isinstance(control_sha256, dict) or set(control_sha256) != required_control:
        stored_control = sorted(control_sha256) if isinstance(control_sha256, dict) else control_sha256
        raise ValueError(
            f"Checkpoint {path!r} control checksum manifest mismatch: "
            f"stored={stored_control!r}, expected={sorted(required_control)!r}."
        )
    for relative_path, expected_digest in control_sha256.items():
        if not isinstance(expected_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
            raise ValueError(f"Checkpoint {path!r} has invalid SHA-256 for {relative_path!r}: " f"{expected_digest!r}.")
        hasher = hashlib.sha256()
        with open(os.path.join(path, relative_path), "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(chunk)
        actual_digest = hasher.hexdigest()
        if actual_digest != expected_digest:
            raise ValueError(
                f"Checkpoint {path!r} control payload checksum mismatch for "
                f"{relative_path!r}: expected={expected_digest}, actual={actual_digest}."
            )
    return state, marker


def is_complete_checkpoint(path: str) -> bool:
    try:
        load_complete_checkpoint_metadata(path)
    except (OSError, ValueError):
        return False
    return True


def is_resumable_checkpoint(path: str) -> bool:
    try:
        state, _ = load_complete_checkpoint_metadata(path)
    except (OSError, ValueError):
        return False
    try:
        completed_steps = int(state["completed_steps"])
        samples_seen = int(state["samples_seen"])
        train_microbatches = int(state["train_microbatches_consumed"])
        global_batch = int(state["effective_global_batch_size"])
        microbatches_per_epoch = int(state["microbatches_per_sampling_epoch"])
        epoch = int(state["sampling_epoch"])
        offset = int(state["sampling_epoch_offset"])
        grad_accum = int(state["gradient_accumulation_steps"])
        resume_strategy = state["resume_strategy"]
    except (KeyError, TypeError, ValueError):
        return False
    if (
        not bool(state.get("exact_data_position", False))
        or completed_steps < 0
        or global_batch <= 0
        or samples_seen != completed_steps * global_batch
        or train_microbatches != completed_steps
        or grad_accum != 1
        or microbatches_per_epoch <= 0
        or resume_strategy != "prepared_batch_sampler_skip_v1"
    ):
        return False
    expected_epoch, expected_offset = resolve_sampling_position(
        train_microbatches,
        microbatches_per_epoch,
    )
    return epoch == expected_epoch and offset == expected_offset


def find_latest_complete_checkpoint(checkpoint_dir: str, *, resumable_only: bool = False) -> str | None:
    """Return the latest fully manifested checkpoint, optionally exact-resumable."""

    if not os.path.isdir(checkpoint_dir):
        return None
    candidates = [
        (checkpoint_step(os.path.join(checkpoint_dir, name)), os.path.join(checkpoint_dir, name))
        for name in os.listdir(checkpoint_dir)
    ]
    complete = [
        (step, path)
        for step, path in candidates
        if step is not None and (is_resumable_checkpoint(path) if resumable_only else is_complete_checkpoint(path))
    ]
    return max(complete, default=(None, None), key=lambda item: item[0])[1]
