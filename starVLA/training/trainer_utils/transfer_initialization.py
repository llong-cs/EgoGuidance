"""Strict weights-only initialization for downstream fine-tuning.

``trainer.transfer_policy`` selects which top-level branches of a pre-training checkpoint
(``trainer.pretrained_checkpoint``) are loaded; every other parameter keeps its seeded
scratch initialization:

    full         qwen_vl_interface + the complete action_model
    action_core  qwen_vl_interface + action_model, except the noisy-action input layer
                 (action_encoder.layer1) and the velocity output layer (action_decoder.layer2)
    vlm_only     qwen_vl_interface only; with ``trainer.transfer_retain_gen_model: true`` the
                 source gen_model is transferred as well

The source gen_model branch is otherwise discarded. Key sets and tensor shapes must match
exactly; any mismatch raises instead of silently skipping parameters.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

TRANSFER_POLICIES = frozenset({"full", "action_core", "vlm_only"})
_LOADABLE_BRANCHES = frozenset({"qwen_vl_interface", "action_model"})
_DROPPED_BRANCHES = frozenset({"gen_model"})
_KNOWN_BRANCHES = _LOADABLE_BRANCHES | _DROPPED_BRANCHES
_ACTION_CORE_RESET_PREFIXES = (
    "action_model.action_encoder.layer1",
    "action_model.action_decoder.layer2",
)
_RUN_CONFIG_FILES = ("config.yaml", "config.full.yaml")
_DATASET_STATISTICS_FILE = "dataset_statistics.json"


class TransferInitializationError(RuntimeError):
    """Raised when a transfer checkpoint does not exactly satisfy its policy."""


def validate_transfer_options(
    policy: str | None,
    reload_modules: str | None,
    *,
    retain_gen_model: bool = False,
) -> str | None:
    """Validate the mutually exclusive ``reload_modules`` and strict ``transfer_policy`` controls.

    Returns the normalized policy, or None when no strict policy is configured.
    """

    if not isinstance(retain_gen_model, bool):
        raise TransferInitializationError("trainer.transfer_retain_gen_model must be a boolean")
    if policy in (None, ""):
        if retain_gen_model:
            raise TransferInitializationError(
                "trainer.transfer_retain_gen_model requires trainer.transfer_policy=vlm_only"
            )
        return None
    normalized = str(policy).lower()
    if normalized not in TRANSFER_POLICIES:
        raise TransferInitializationError(
            f"unsupported trainer.transfer_policy {normalized!r}; expected one of {sorted(TRANSFER_POLICIES)}"
        )
    if reload_modules is not None:
        raise TransferInitializationError(
            "trainer.transfer_policy and trainer.reload_modules are mutually exclusive; "
            "the transfer policy fully defines which parameters are loaded."
        )
    if retain_gen_model and normalized != "vlm_only":
        raise TransferInitializationError(
            "trainer.transfer_retain_gen_model is supported only with trainer.transfer_policy=vlm_only"
        )
    return normalized


def resolve_checkpoint_run_dir(checkpoint_path: str | Path) -> Path:
    """Return the training run directory that produced ``checkpoint_path``.

    Supported layouts (as written by the training entries):

        <run_dir>/checkpoints/steps_<N>_pytorch_model.pt        (model-only checkpoint)
        <run_dir>/checkpoints/steps_<N>/portable_pytorch_model.pt  (inside a full-state checkpoint)
        <run_dir>/final_model/pytorch_model.pt                  (final model)

    The ``.safetensors`` variants are accepted as well.
    """

    path = Path(checkpoint_path)
    parent = path.parent
    if parent.name in ("checkpoints", "final_model"):
        return parent.parent
    if re.fullmatch(r"steps_\d+", parent.name) and parent.parent.name == "checkpoints":
        return parent.parent.parent
    raise TransferInitializationError(
        "cannot locate the run directory of trainer.pretrained_checkpoint; expected "
        "<run_dir>/checkpoints/<file>, <run_dir>/checkpoints/steps_<N>/<file> or "
        f"<run_dir>/final_model/<file>, got {path}"
    )


def validate_transfer_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    """Check that a transfer checkpoint and its run metadata exist; return a source record.

    The run directory must hold the training config (``config.yaml`` or ``config.full.yaml``)
    and ``dataset_statistics.json`` next to the checkpoint, so a checkpoint is never used
    without the run that produced it.
    """

    path = Path(checkpoint_path)
    if not path.is_file():
        raise TransferInitializationError(f"trainer.pretrained_checkpoint is not an existing file: {path}")
    run_dir = resolve_checkpoint_run_dir(path)
    config_files = [name for name in _RUN_CONFIG_FILES if (run_dir / name).is_file()]
    if not config_files:
        raise TransferInitializationError(
            f"checkpoint run directory {run_dir} has neither {' nor '.join(_RUN_CONFIG_FILES)}"
        )
    if not (run_dir / _DATASET_STATISTICS_FILE).is_file():
        raise TransferInitializationError(f"checkpoint run directory {run_dir} has no {_DATASET_STATISTICS_FILE}")
    return {
        "path": str(path),
        "run_dir": str(run_dir),
        "config_files": config_files,
        "size_bytes": int(path.stat().st_size),
    }


def _branch(key: str) -> str:
    return key.split(".", 1)[0]


def _keys_below(keys: set[str], prefix: str) -> set[str]:
    dotted_prefix = f"{prefix}."
    return {key for key in keys if key.startswith(dotted_prefix)}


def _format_keys(keys: set[str], *, limit: int = 8) -> str:
    ordered = sorted(keys)
    suffix = "" if len(ordered) <= limit else f" ... (+{len(ordered) - limit} more)"
    return f"{ordered[:limit]}{suffix}"


def _require_equal_keys(label: str, source_keys: set[str], target_keys: set[str]) -> None:
    missing = target_keys - source_keys
    unexpected = source_keys - target_keys
    if missing or unexpected:
        raise TransferInitializationError(
            f"{label} key mismatch: missing={_format_keys(missing)}, unexpected={_format_keys(unexpected)}"
        )


def _tensor_metadata(tensor: torch.Tensor) -> dict[str, Any]:
    cpu_tensor = tensor.detach().cpu().contiguous()
    byte_view = cpu_tensor.view(torch.uint8).reshape(-1).numpy()
    return {
        "dtype": str(cpu_tensor.dtype),
        "shape": list(cpu_tensor.shape),
        "sha256": hashlib.sha256(memoryview(byte_view)).hexdigest(),
    }


def _branch_counts(keys: set[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key in keys:
        branch = _branch(key)
        counts[branch] = counts.get(branch, 0) + 1
    return dict(sorted(counts.items()))


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file

            checkpoint = load_file(str(path))
        else:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except Exception as exc:
        raise TransferInitializationError(f"failed to load transfer checkpoint {path}: {exc}") from exc

    if not isinstance(checkpoint, Mapping) or not checkpoint:
        raise TransferInitializationError(f"transfer checkpoint must be a non-empty state dict: {path}")
    invalid = [key for key, value in checkpoint.items() if not isinstance(key, str) or not torch.is_tensor(value)]
    if invalid:
        raise TransferInitializationError(
            f"transfer checkpoint contains non-tensor/non-string state entries: {invalid[:8]}"
        )
    return dict(checkpoint)


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 of one required regular file."""

    path = Path(path)
    if not path.is_file():
        raise TransferInitializationError(f"required file does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_dataset_statistics_identity(manifest: Mapping[str, Any], path: str | Path) -> dict[str, Any]:
    """Return a copy of ``manifest`` that records the SHA-256 of the run's normalization statistics."""

    path = Path(path)
    if "dataset_statistics" in manifest:
        raise TransferInitializationError("initialization manifest already contains a dataset_statistics entry")
    augmented = dict(manifest)
    augmented["dataset_statistics"] = {"path": str(path), "sha256": file_sha256(path)}
    return augmented


def load_transfer_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    policy: str,
    *,
    source: Mapping[str, Any] | None = None,
    retain_gen_model: bool = False,
) -> dict[str, Any]:
    """Apply one strict transfer policy to ``model`` and return its audit manifest.

    ``full`` and ``action_core`` targets must be pure action models (no gen_model). A
    ``vlm_only`` target may build a generation branch; it stays at its seeded scratch
    initialization unless ``retain_gen_model=True``, which transfers the source gen_model
    while still resetting the complete action expert.
    """

    policy = validate_transfer_options(policy, None, retain_gen_model=retain_gen_model)
    checkpoint_path = Path(checkpoint_path)
    checkpoint = _load_state_dict(checkpoint_path)
    target_state = model.state_dict()
    source_keys = set(checkpoint)
    target_keys = set(target_state)

    source_unknown = {key for key in source_keys if _branch(key) not in _KNOWN_BRANCHES}
    if source_unknown:
        raise TransferInitializationError(
            f"transfer checkpoint contains unsupported top-level branches: {_format_keys(source_unknown)}"
        )
    target_unknown = {key for key in target_keys if _branch(key) not in _KNOWN_BRANCHES}
    target_gen = _keys_below(target_keys, "gen_model")
    if target_unknown:
        raise TransferInitializationError(
            f"transfer target contains unsupported top-level branches: {_format_keys(target_unknown)}"
        )
    if target_gen and policy != "vlm_only":
        raise TransferInitializationError(
            f"{policy} transfer does not define generation-branch initialization; "
            "use vlm_only to keep a seeded scratch gen_model or disable generation"
        )

    source_qwen = _keys_below(source_keys, "qwen_vl_interface")
    source_action = _keys_below(source_keys, "action_model")
    target_qwen = _keys_below(target_keys, "qwen_vl_interface")
    target_action = _keys_below(target_keys, "action_model")
    source_gen = _keys_below(source_keys, "gen_model")
    if not source_qwen or not source_action or not target_qwen or not target_action:
        raise TransferInitializationError(
            "transfer requires non-empty qwen_vl_interface and action_model branches in both source and target"
        )

    _require_equal_keys("qwen_vl_interface", source_qwen, target_qwen)
    reset_keys: set[str]
    if policy == "full":
        _require_equal_keys("action_model", source_action, target_action)
        load_keys = source_qwen | source_action
        reset_keys = set()
    elif policy == "action_core":
        source_reset: set[str] = set()
        reset_keys = set()
        for prefix in _ACTION_CORE_RESET_PREFIXES:
            source_matches = _keys_below(source_action, prefix)
            target_matches = _keys_below(target_action, prefix)
            if not source_matches or not target_matches:
                raise TransferInitializationError(
                    f"action_core reset prefix must exist in source and target: {prefix!r}"
                )
            _require_equal_keys(f"action_core reset prefix {prefix}", source_matches, target_matches)
            source_reset.update(source_matches)
            reset_keys.update(target_matches)
        _require_equal_keys("action_model core", source_action - source_reset, target_action - reset_keys)
        load_keys = source_qwen | (source_action - source_reset)
    else:
        if retain_gen_model:
            if not target_gen:
                raise TransferInitializationError(
                    "trainer.transfer_retain_gen_model=true requires a non-empty target gen_model branch"
                )
            _require_equal_keys("gen_model", source_gen, target_gen)
            load_keys = source_qwen | source_gen
            reset_keys = target_action
        else:
            load_keys = source_qwen
            reset_keys = target_action | target_gen

    expected_loaded_target_keys = target_keys - reset_keys
    _require_equal_keys(f"{policy} selected tensors", load_keys, expected_loaded_target_keys)
    shape_mismatches = {
        key: {"checkpoint": list(checkpoint[key].shape), "target": list(target_state[key].shape)}
        for key in load_keys
        if checkpoint[key].shape != target_state[key].shape
    }
    if shape_mismatches:
        preview = dict(list(sorted(shape_mismatches.items()))[:8])
        raise TransferInitializationError(f"{policy} tensor shape mismatch: {preview}")

    selected_state = {key: checkpoint[key] for key in sorted(load_keys)}
    incompatible = model.load_state_dict(selected_state, strict=False)
    actual_missing = set(incompatible.missing_keys)
    actual_unexpected = set(incompatible.unexpected_keys)
    if actual_missing != reset_keys or actual_unexpected:
        raise TransferInitializationError(
            f"{policy} load result violated the policy: expected_missing={_format_keys(reset_keys)}, "
            f"actual_missing={_format_keys(actual_missing)}, unexpected={_format_keys(actual_unexpected)}"
        )

    loaded_tensors = {key: _tensor_metadata(target_state[key]) for key in sorted(load_keys)}
    loaded_digest = hashlib.sha256(
        json.dumps(loaded_tensors, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    dropped_keys = source_keys - load_keys
    reset_tensors = {key: _tensor_metadata(target_state[key]) for key in sorted(reset_keys)}
    reset_digest = hashlib.sha256(
        json.dumps(reset_tensors, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    loadable_branches = set(_LOADABLE_BRANCHES)
    always_dropped_branches = set(_DROPPED_BRANCHES)
    if retain_gen_model:
        loadable_branches.add("gen_model")
        always_dropped_branches.remove("gen_model")
    contract = {
        "loadable_branches": sorted(loadable_branches),
        "always_dropped_branches": sorted(always_dropped_branches),
        "scratch_target_branches": (
            sorted({"action_model", "gen_model"} if target_gen else {"action_model"}) if policy == "vlm_only" else []
        ),
        "action_core_reset_prefixes": list(_ACTION_CORE_RESET_PREFIXES) if policy == "action_core" else [],
    }
    if retain_gen_model:
        contract["transfer_retain_gen_model"] = True
        contract["scratch_target_branches"] = ["action_model"]

    return {
        "version": 2,
        "mode": "weights_only_transfer",
        "policy": policy,
        "source": dict(source) if source is not None else {"path": str(checkpoint_path)},
        "contract": contract,
        "counts": {
            "checkpoint_keys": len(source_keys),
            "target_keys": len(target_keys),
            "loaded_keys": len(load_keys),
            "reset_keys": len(reset_keys),
            "dropped_keys": len(dropped_keys),
        },
        "branch_counts": {
            "loaded": _branch_counts(load_keys),
            "reset": _branch_counts(reset_keys),
            "dropped": _branch_counts(dropped_keys),
        },
        "loaded_tensors_sha256": loaded_digest,
        "reset_tensors_sha256": reset_digest,
        "reset_tensors": reset_tensors,
        "loaded_keys": sorted(load_keys),
        "reset_keys": sorted(reset_keys),
        "dropped_keys": sorted(dropped_keys),
    }


def write_initialization_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    """Atomically publish an initialization manifest; refuse to replace a different one."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def require_existing_match() -> None:
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TransferInitializationError(f"existing initialization manifest is unreadable: {path}") from exc
        if existing != dict(manifest):
            raise TransferInitializationError(f"refusing to overwrite a different initialization manifest: {path}")

    if path.exists():
        require_existing_match()
        return
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            require_existing_match()
    finally:
        if temporary.exists():
            temporary.unlink()
