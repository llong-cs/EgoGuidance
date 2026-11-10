# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
In this file, we define 3 types of datasets:
1. LeRobotSingleDataset: a single dataset for a given embodiment tag
2. LeRobotMixtureDataset: a mixture of datasets for a given list of embodiment tags
3. CachedLeRobotSingleDataset: a single dataset for a given embodiment tag,
                                with caching for the video frames

See ``starVLA/dataloader/lerobot_datasets.py`` for how these datasets are built from a config.
"""

import copy
import gc
import hashlib
import io
import json
import os
import pickle
import random
import tempfile
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import ClassVar, Sequence

import numpy as np
import pandas as pd
import torch.distributed as dist
from PIL import Image
from pydantic import BaseModel, Field, ValidationError
from torch.utils.data import Dataset
from tqdm import tqdm

from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.episode_filter import episode_rejection_reason, parse_episode_filter
from starVLA.dataloader.gr00t_lerobot.schema import (
    DatasetMetadata,
    DatasetStatisticalValues,
    LeRobotModalityMetadata,
    LeRobotStateActionMetadata,
)
from starVLA.dataloader.gr00t_lerobot.transform import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionTransform
from starVLA.dataloader.gr00t_lerobot.video import get_all_frames, get_frames_by_timestamps
from starVLA.dataloader.shared_epoch import SharedEpoch
from starVLA.dataloader.video_overrides import (
    VideoOverrideError,
    canonical_metadata_digest,
    canonical_provenance_digest,
    file_identity,
    load_manifest,
    normalize_logical_path,
    normalize_sample_error_policy,
    parse_file_identity,
    parse_nonnegative_integer,
    parse_nonnegative_timestamp,
    resolve_entry,
    validate_manifest_leaf_contract,
)
from starVLA.dataloader.view_tiling import assemble_role_observation
from starVLA.training.distributed import distributed_barrier

# LeRobot v2.0 dataset file names
LE_ROBOT_MODALITY_FILENAME = "meta/modality.json"
LE_ROBOT_EPISODE_FILENAME = "meta/episodes.jsonl"
LE_ROBOT_TASKS_FILENAME = "meta/tasks.jsonl"
LE_ROBOT_INFO_FILENAME = "meta/info.json"
LE_ROBOT_STATS_FILENAME = "meta/stats_gr00t.json"
LE_ROBOT_DATA_FILENAME = "data/*/*.parquet"
LE_ROBOT_STEPS_FILENAME = "meta/steps.pkl"
LE_ROBOT_STATS_FORMAT_VERSION = 2
SHARED_CACHE_VISIBILITY_TIMEOUT_S = 30.0
SHARED_CACHE_VISIBILITY_POLL_S = 0.25
EPSILON = 5e-4

#  LeRobot v3.0 dataset file names
LE_ROBOT3_TASKS_FILENAME = "meta/tasks.parquet"
LE_ROBOT3_EPISODE_FILENAME = "meta/episodes/*/*.parquet"


def calculate_dataset_statistics(parquet_paths: list[Path]) -> dict:
    """Calculate the dataset statistics of all columns for a list of parquet files."""
    # Dataset statistics
    all_low_dim_data_list = []
    # Collect all the data
    # parquet_paths = parquet_paths[:3]
    for parquet_path in tqdm(
        sorted(list(parquet_paths)),
        desc="Collecting all parquet files...",
    ):
        # Load the parquet file
        parquet_data = pd.read_parquet(parquet_path)
        parquet_data = parquet_data
        all_low_dim_data_list.append(parquet_data)

    if not all_low_dim_data_list:
        raise FileNotFoundError(
            f"No parquet files found under the provided paths: {[str(p) for p in parquet_paths[:3]]}..."
            f" — make sure the dataset has been downloaded/converted before training."
        )

    all_low_dim_data = pd.concat(all_low_dim_data_list, axis=0)
    # Compute dataset statistics
    dataset_statistics = {}
    for le_modality in tqdm(all_low_dim_data.columns, desc="Processing modalities"):
        print(le_modality)
        if "task_info" in le_modality:
            continue
        print(f"Computing statistics for {le_modality}...")
        try:
            np_data = np.vstack([np.asarray(x, dtype=np.float32) for x in all_low_dim_data[le_modality]])
        except Exception as e:
            print(f"Warning: Failed to process modality {le_modality} due to error: {e}")
            continue

        dataset_statistics[le_modality] = {
            "mean": np.mean(np_data, axis=0).tolist(),
            "std": np.std(np_data, axis=0).tolist(),
            "min": np.min(np_data, axis=0).tolist(),
            "max": np.max(np_data, axis=0).tolist(),
            "q01": np.quantile(np_data, 0.01, axis=0).tolist(),
            "q99": np.quantile(np_data, 0.99, axis=0).tolist(),
        }
    return dataset_statistics


def _normalize_action_mode(mode: str) -> str:
    """Normalize action mode names to {abs, delta, rel}."""
    mode = str(mode).lower()
    if mode in {"absolute", "raw"}:
        mode = "abs"
    if mode not in {"abs", "delta", "rel"}:
        mode = "abs"
    return mode


def _normalize_action_mode_apply_keys(
    action_mode_apply_keys: Sequence[str] | None,
    fallback_keys: Sequence[str] | None = None,
) -> list[str]:
    source_keys = action_mode_apply_keys if action_mode_apply_keys else (fallback_keys or [])
    normalized = []
    for key in source_keys:
        key = str(key)
        if not key.startswith("action."):
            key = f"action.{key}"
        normalized.append(key)
    return normalized


def _normalize_action_mode_state_map(action_mode_state_map: dict[str, str] | None) -> dict[str, str]:
    normalized = {}
    for action_key, state_key in (action_mode_state_map or {}).items():
        action_key = str(action_key)
        state_key = str(state_key)
        if not action_key.startswith("action."):
            action_key = f"action.{action_key}"
        if not state_key.startswith("state."):
            state_key = f"state.{state_key}"
        normalized[action_key] = state_key
    return normalized


def _build_stats_cache_config(
    action_mode: str,
) -> dict:
    return {
        "mode": action_mode,
    }


def _invalidate_legacy_stats_cache(stats_path: Path, reason: str) -> None:
    if not stats_path.exists():
        return
    print(f"Removing stale dataset statistics cache at {stats_path}: {reason}")
    try:
        stats_path.unlink()
    except OSError as exc:
        print(f"  (could not remove stale cache on read-only fs: {exc})")


def _load_stats_cache(
    stats_path: Path,
    expected_config: dict,
    *,
    invalidate_legacy: bool,
) -> dict | None:
    if not stats_path.exists():
        return None

    try:
        with open(stats_path, "r") as f:
            payload = json.load(f)
    except Exception as exc:
        if invalidate_legacy:
            _invalidate_legacy_stats_cache(stats_path, f"failed to load JSON ({exc})")
        return None

    if not isinstance(payload, dict):
        if invalidate_legacy:
            _invalidate_legacy_stats_cache(stats_path, "unexpected top-level format")
        return None

    format_version = payload.get("__format_version")
    cache_config = payload.get("__cache_config")
    statistics = payload.get("statistics")
    if format_version != LE_ROBOT_STATS_FORMAT_VERSION or cache_config is None or statistics is None:
        if invalidate_legacy:
            _invalidate_legacy_stats_cache(stats_path, "legacy statistics format detected")
        return None

    if cache_config != expected_config:
        if invalidate_legacy:
            _invalidate_legacy_stats_cache(stats_path, "statistics config mismatch, rebuilding cache")
        return None

    return statistics


@contextmanager
def _atomic_cache_writer(cache_path: Path, mode: str):
    """Write one shared cache through a writer-unique temporary file."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode=mode,
            dir=cache_path.parent,
            prefix=f".{cache_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as cache_file:
            tmp_path = Path(cache_file.name)
            yield cache_file
        os.replace(tmp_path, cache_path)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def _load_shared_cache_with_retry(
    load_once,
    *,
    timeout_s: float = SHARED_CACHE_VISIBILITY_TIMEOUT_S,
    poll_s: float = SHARED_CACHE_VISIBILITY_POLL_S,
):
    """Wait briefly for another node or concurrent job to publish a cache."""
    deadline = time.monotonic() + max(timeout_s, 0.0)
    while True:
        cached = load_once()
        if cached is not None:
            return cached
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            return None
        time.sleep(min(max(poll_s, 0.001), remaining_s))


def _save_stats_cache(stats_path: Path, cache_config: dict, statistics: dict) -> None:
    payload = {
        "__format_version": LE_ROBOT_STATS_FORMAT_VERSION,
        "__cache_config": cache_config,
        "statistics": statistics,
    }
    with _atomic_cache_writer(stats_path, "w") as f:
        json.dump(payload, f, indent=4)


def _starvla_cache_base() -> Path:
    """Writable base dir for per-dataset meta caches (stats + steps index) used when
    the dataset dir itself is read-only (e.g. a read-only dataset mount).

    ``$STARVLA_STATS_CACHE_DIR`` if set (RECOMMENDED for multi-node — point it at a
    SHARED writable path so all nodes hit the same cache), else the node-local
    ``~/.cache/starvla/stats_gr00t``.
    """
    cache_dir = os.environ.get("STARVLA_STATS_CACHE_DIR")
    return Path(cache_dir) if cache_dir else (Path.home() / ".cache" / "starvla" / "stats_gr00t")


def _fallback_stats_path(stats_path: Path) -> Path:
    """Writable mirror of a read-only dataset's stats_gr00t.json, keyed deterministically
    by the canonical stats path so every rank/run resolves the SAME fallback file."""
    key = hashlib.sha1(str(stats_path).encode()).hexdigest()[:16]
    dataset_name = stats_path.parent.parent.name  # <dataset_dir>/meta/stats_gr00t.json
    return _starvla_cache_base() / f"{dataset_name}.{key}.json"


def _fallback_cache_path(canonical_path: Path) -> Path:
    """Writable mirror of any per-dataset ``meta/`` cache file (e.g. steps_data_index.pkl)
    when the dataset dir is read-only. Keyed deterministically by the canonical path."""
    key = hashlib.sha1(str(canonical_path).encode()).hexdigest()[:16]
    dataset_name = canonical_path.parent.parent.name
    return _starvla_cache_base() / f"{dataset_name}.{key}.{canonical_path.name}"


def _cache_write_targets(canonical_path: Path, fallback_path: Path) -> tuple[Path, ...]:
    """Return cache write targets without probing an explicitly read-only dataset.

    An explicit STARVLA_STATS_CACHE_DIR is a contract that the canonical dataset
    tree must not be touched. Without it, try the canonical path first, then the
    fallback.
    """
    if "STARVLA_STATS_CACHE_DIR" in os.environ:
        return (fallback_path,)
    return canonical_path, fallback_path


def _save_steps_cache(cache_path: Path, cache_data: dict) -> None:
    """Atomically persist a steps-index payload."""
    with _atomic_cache_writer(cache_path, "wb") as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)


def _save_stats_cache_safe(stats_path: Path, cache_config: dict, statistics: dict) -> None:
    """Save stats to the configured cache target without failing the run.

    With an explicit STARVLA_STATS_CACHE_DIR, write only to its deterministic
    fallback path. Otherwise try the canonical path first, then the fallback.
    """
    fallback_path = _fallback_stats_path(stats_path)
    for target in _cache_write_targets(stats_path, fallback_path):
        try:
            _save_stats_cache(target, cache_config, statistics)
            return
        except OSError as exc:
            print(f"[stats-cache] could not write {target} ({exc}); trying next location")
    print(
        "[stats-cache] WARNING: no writable location for dataset statistics cache. "
        "Stats will be RECOMPUTED every run. Set STARVLA_STATS_CACHE_DIR to a writable "
        "(shared, for multi-node) path to cache them."
    )


def _load_stats_cache_any(stats_path: Path, expected_config: dict, *, invalidate_legacy: bool) -> dict | None:
    """Load canonical first, then fallback, without mutating an explicit read-only source."""
    may_invalidate_canonical = invalidate_legacy and "STARVLA_STATS_CACHE_DIR" not in os.environ
    stats = _load_stats_cache(stats_path, expected_config, invalidate_legacy=may_invalidate_canonical)
    if stats is not None:
        return stats
    return _load_stats_cache(_fallback_stats_path(stats_path), expected_config, invalidate_legacy=False)


def _compute_statistics_for_mode(
    parquet_paths: list[Path],
    dataset_name: str,
    action_mode: str,
    lerobot_modality_meta: "LeRobotModalityMetadata",
    action_keys_full: list[str],
    state_keys_full: list[str],
    action_indices: list[int] | None,
    state_indices: list[int] | None,
    action_mode_apply_keys: list[str] | None,
    action_mode_state_map: dict[str, str] | None,
) -> dict:
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[RANK 0] Calculating dataset statistics for {dataset_name} (mode={action_mode})")

    base_stats = calculate_dataset_statistics(parquet_paths)

    if action_mode == "abs":
        return base_stats

    if action_indices is None or state_indices is None:
        raise ValueError(
            "Both action and state modalities are required to compute " f"{action_mode} action mode statistics."
        )

    if action_mode == "delta":
        return calculate_delta_action_statistics(
            parquet_paths=parquet_paths,
            lerobot_modality_meta=lerobot_modality_meta,
            action_keys_full=action_keys_full,
            state_keys_full=state_keys_full,
            action_indices=action_indices,
            state_indices=state_indices,
            action_mode_apply_keys=action_mode_apply_keys,
            action_mode_state_map=action_mode_state_map,
            base_stats=base_stats,
        )
    if action_mode == "rel":
        return calculate_rel_action_statistics(
            parquet_paths=parquet_paths,
            lerobot_modality_meta=lerobot_modality_meta,
            action_keys_full=action_keys_full,
            state_keys_full=state_keys_full,
            action_indices=action_indices,
            state_indices=state_indices,
            action_mode_apply_keys=action_mode_apply_keys,
            action_mode_state_map=action_mode_state_map,
            base_stats=base_stats,
        )
    raise ValueError(f"Unsupported action mode for statistics: {action_mode}")


def _load_or_compute_statistics(
    stats_path: Path,
    stats_cache_config: dict,
    parquet_paths: list[Path],
    dataset_name: str,
    action_mode: str,
    lerobot_modality_meta: "LeRobotModalityMetadata",
    action_keys_full: list[str],
    state_keys_full: list[str],
    action_indices: list[int] | None,
    state_indices: list[int] | None,
    action_mode_apply_keys: list[str] | None,
    action_mode_state_map: dict[str, str] | None,
) -> dict:
    le_statistics = _load_stats_cache_any(
        stats_path,
        stats_cache_config,
        invalidate_legacy=True,
    )
    if le_statistics is not None:
        return le_statistics

    le_statistics = _compute_statistics_for_mode(
        parquet_paths=parquet_paths,
        dataset_name=dataset_name,
        action_mode=action_mode,
        lerobot_modality_meta=lerobot_modality_meta,
        action_keys_full=action_keys_full,
        state_keys_full=state_keys_full,
        action_indices=action_indices,
        state_indices=state_indices,
        action_mode_apply_keys=action_mode_apply_keys,
        action_mode_state_map=action_mode_state_map,
    )
    # Canonical path first; fall back to a writable cache dir for read-only mounts.
    _save_stats_cache_safe(stats_path, stats_cache_config, le_statistics)
    return le_statistics


def _get_action_col_slices(
    lerobot_modality_meta: "LeRobotModalityMetadata",
    action_keys_full: list[str],
    state_keys_full: list[str],
    action_mode_apply_keys: list[str] | None = None,
    action_mode_state_map: dict[str, str] | None = None,
) -> dict[str, list[tuple[tuple[int, int], str, tuple[int, int], str, str]]]:
    apply_keys = _normalize_action_mode_apply_keys(action_mode_apply_keys, action_keys_full)
    action_mode_state_map = _normalize_action_mode_state_map(action_mode_state_map)

    action_meta = lerobot_modality_meta.action
    state_meta = lerobot_modality_meta.state

    # Build per-column mapping: action column -> list of (action_slice, state_column, state_slice)
    action_col_slices: dict[str, list[tuple[tuple[int, int], str, tuple[int, int]]]] = {}
    for action_key in apply_keys:
        if not action_key.startswith("action."):
            raise ValueError(f"Invalid action key {action_key}. Expected prefix 'action.'.")
        state_key = action_mode_state_map.get(action_key, action_key.replace("action.", "state.", 1))
        if state_key not in state_keys_full:
            raise ValueError(
                f"State key {state_key} not found for action key {action_key}. "
                f"Add it to action_mode_state_map or remove {action_key} from action_mode_apply_keys."
            )

        action_subkey = action_key.replace("action.", "", 1)
        state_subkey = state_key.replace("state.", "", 1)
        if action_subkey not in action_meta or state_subkey not in state_meta:
            raise ValueError(f"Action/state key missing in metadata: {action_key} -> {state_key}")

        action_cfg = action_meta[action_subkey]
        state_cfg = state_meta[state_subkey]
        action_col = action_cfg.original_key or action_subkey
        state_col = state_cfg.original_key or state_subkey
        action_slice = (action_cfg.start, action_cfg.end)
        state_slice = (state_cfg.start, state_cfg.end)
        action_padding = "first_last" if action_cfg.absolute else "zero"
        state_padding = "first_last" if state_cfg.absolute else "zero"
        action_col_slices.setdefault(action_col, []).append(
            (action_slice, state_col, state_slice, action_padding, state_padding)
        )

    return action_col_slices


def calculate_delta_action_statistics(
    parquet_paths: list[Path],
    lerobot_modality_meta: "LeRobotModalityMetadata",
    action_keys_full: list[str],
    state_keys_full: list[str],
    action_indices: list[int],
    state_indices: list[int],
    action_mode_apply_keys: list[str] | None = None,
    action_mode_state_map: dict[str, str] | None = None,
    base_stats: dict | None = None,
) -> dict:
    """
    Calculate action statistics using delta mode.

    Rule:
      - For t>0: a_t - a_{t-1}
      - For t=0: a_0 - s_0

    Mapping rule (only two cases):
      1) Use explicit action_mode_state_map if provided.
      2) Otherwise, replace 'action.' with 'state.' directly.
    """
    if base_stats is None:
        base_stats = calculate_dataset_statistics(parquet_paths)

    action_col_slices = _get_action_col_slices(
        lerobot_modality_meta, action_keys_full, state_keys_full, action_mode_apply_keys, action_mode_state_map
    )
    if not action_col_slices:
        raise ValueError("No action columns found in the dataset.")

    def _get_chunk(array: np.ndarray, step_indices: np.ndarray, padding_strategy: str) -> np.ndarray:
        max_length = array.shape[0]
        front_padding = step_indices < 0
        end_padding = step_indices >= max_length
        padding_positions = np.logical_or(front_padding, end_padding)
        output = np.zeros((len(step_indices), array.shape[1]), dtype=array.dtype)
        if (~padding_positions).any():
            output[~padding_positions] = array[step_indices[~padding_positions]]
        if padding_positions.any():
            if padding_strategy == "first_last":
                output[front_padding] = array[0]
                output[end_padding] = array[-1]
            elif padding_strategy == "zero":
                output[padding_positions] = 0
            else:
                raise ValueError(f"Invalid padding strategy: {padding_strategy}")
        return output

    accum: dict[str, list[np.ndarray]] = {col: [] for col in action_col_slices.keys()}
    for parquet_path in tqdm(sorted(list(parquet_paths)), desc="Collecting delta action stats"):
        data = pd.read_parquet(parquet_path)
        trajectory_length = len(data)
        for action_col, slice_list in action_col_slices.items():
            if action_col not in data.columns:
                raise ValueError(f"{action_col} not found in parquet columns.")
            action_matrix = np.stack(data[action_col])
            action_padding_ref = slice_list[0][3]
            prepared_slices = []
            for a_slice, state_col, s_slice, _action_padding, state_padding in slice_list:
                if state_col not in data.columns:
                    raise ValueError(f"{state_col} not found in parquet columns.")
                state_matrix = np.stack(data[state_col])
                state_part_full = state_matrix[:, s_slice[0] : s_slice[1]]
                prepared_slices.append((a_slice, state_part_full, state_padding))
            for base_index in range(trajectory_length):
                action_steps = np.array(action_indices) + base_index
                action_chunk_full = _get_chunk(action_matrix, action_steps, action_padding_ref)

                for a_slice, state_part_full, state_padding in prepared_slices:
                    action_part_chunk = action_chunk_full[:, a_slice[0] : a_slice[1]]
                    state_chunk = _get_chunk(state_part_full, np.array(state_indices) + base_index, state_padding)
                    if action_part_chunk.shape[1] != state_chunk.shape[1]:
                        raise ValueError(f"Action/state dim mismatch for {action_col}:{a_slice}")

                    out = action_part_chunk.copy()
                    if len(out) > 1:
                        out[1:] = action_part_chunk[1:] - action_part_chunk[:-1]
                    out[0] = action_part_chunk[0] - state_chunk[0]
                    action_chunk_full[:, a_slice[0] : a_slice[1]] = out

                accum[action_col].append(action_chunk_full)

    delta_stats = copy.deepcopy(base_stats)
    for action_col, series_list in accum.items():
        if not series_list:
            continue
        all_values = np.concatenate(series_list, axis=0).astype(np.float32)
        delta_stats[action_col] = {
            "mean": np.mean(all_values, axis=0).tolist(),
            "std": np.std(all_values, axis=0).tolist(),
            "min": np.min(all_values, axis=0).tolist(),
            "max": np.max(all_values, axis=0).tolist(),
            "q01": np.quantile(all_values, 0.01, axis=0).tolist(),
            "q99": np.quantile(all_values, 0.99, axis=0).tolist(),
        }
    return delta_stats


def calculate_rel_action_statistics(
    parquet_paths: list[Path],
    lerobot_modality_meta: "LeRobotModalityMetadata",
    action_keys_full: list[str],
    state_keys_full: list[str],
    action_indices: list[int],
    state_indices: list[int],
    action_mode_apply_keys: list[str] | None = None,
    action_mode_state_map: dict[str, str] | None = None,
    base_stats: dict | None = None,
) -> dict:
    """
    Calculate action statistics using rel mode.

    Rule:
      - For all t: a_t - s_0

    Mapping rule (only two cases):
      1) Use explicit action_mode_state_map if provided.
      2) Otherwise, replace 'action.' with 'state.' directly.
    """
    if base_stats is None:
        base_stats = calculate_dataset_statistics(parquet_paths)

    action_col_slices = _get_action_col_slices(
        lerobot_modality_meta, action_keys_full, state_keys_full, action_mode_apply_keys, action_mode_state_map
    )
    if not action_col_slices:
        raise ValueError("No action columns found in the dataset.")

    def _get_chunk(array: np.ndarray, step_indices: np.ndarray, padding_strategy: str) -> np.ndarray:
        max_length = array.shape[0]
        front_padding = step_indices < 0
        end_padding = step_indices >= max_length
        padding_positions = np.logical_or(front_padding, end_padding)
        output = np.zeros((len(step_indices), array.shape[1]), dtype=array.dtype)
        if (~padding_positions).any():
            output[~padding_positions] = array[step_indices[~padding_positions]]
        if padding_positions.any():
            if padding_strategy == "first_last":
                output[front_padding] = array[0]
                output[end_padding] = array[-1]
            elif padding_strategy == "zero":
                output[padding_positions] = 0
            else:
                raise ValueError(f"Invalid padding strategy: {padding_strategy}")
        return output

    accum: dict[str, list[np.ndarray]] = {col: [] for col in action_col_slices.keys()}
    for parquet_path in tqdm(sorted(list(parquet_paths)), desc="Collecting rel action stats"):
        data = pd.read_parquet(parquet_path)
        trajectory_length = len(data)
        for action_col, slice_list in action_col_slices.items():
            if action_col not in data.columns:
                raise ValueError(f"{action_col} not found in parquet columns.")
            action_matrix = np.stack(data[action_col])
            action_padding_ref = slice_list[0][3]
            prepared_slices = []
            for a_slice, state_col, s_slice, _action_padding, state_padding in slice_list:
                if state_col not in data.columns:
                    raise ValueError(f"{state_col} not found in parquet columns.")
                state_matrix = np.stack(data[state_col])
                state_part_full = state_matrix[:, s_slice[0] : s_slice[1]]
                prepared_slices.append((a_slice, state_part_full, state_padding))
            for base_index in range(trajectory_length):
                action_steps = np.array(action_indices) + base_index
                action_chunk_full = _get_chunk(action_matrix, action_steps, action_padding_ref)

                for a_slice, state_part_full, state_padding in prepared_slices:
                    action_part_chunk = action_chunk_full[:, a_slice[0] : a_slice[1]]
                    state_chunk = _get_chunk(state_part_full, np.array(state_indices) + base_index, state_padding)
                    if action_part_chunk.shape[1] != state_chunk.shape[1]:
                        raise ValueError(f"Action/state dim mismatch for {action_col}:{a_slice}")

                    out = action_part_chunk - state_chunk[0]
                    action_chunk_full[:, a_slice[0] : a_slice[1]] = out

                accum[action_col].append(action_chunk_full)

    rel_stats = copy.deepcopy(base_stats)
    for action_col, series_list in accum.items():
        if not series_list:
            continue
        all_values = np.concatenate(series_list, axis=0).astype(np.float32)
        rel_stats[action_col] = {
            "mean": np.mean(all_values, axis=0).tolist(),
            "std": np.std(all_values, axis=0).tolist(),
            "min": np.min(all_values, axis=0).tolist(),
            "max": np.max(all_values, axis=0).tolist(),
            "q01": np.quantile(all_values, 0.01, axis=0).tolist(),
            "q99": np.quantile(all_values, 0.99, axis=0).tolist(),
        }
    return rel_stats


class ModalityConfig(BaseModel):
    """Configuration for a modality."""

    delta_indices: list[int]
    """Delta indices to sample relative to the current index. The returned data will correspond to the original data at a sampled base index + delta indices."""
    modality_keys: list[str]
    """The keys to load for the modality in the dataset."""


class LeRobotSingleDataset(Dataset):
    """
    Base dataset class for LeRobot that supports sharding.
    """

    # Injected by the loader factory after resolving the active DataConfig cameras.
    # Keeping loud empty defaults makes direct construction fail at sample packing
    # instead of silently reverting to implicit physical-camera order.
    view_roles: ClassVar[dict] = {}
    view_descriptions: ClassVar[dict] = {}

    def __init__(
        self,
        dataset_path: Path | str,
        modality_configs: dict[str, ModalityConfig],
        embodiment_tag: str | EmbodimentTag,
        video_backend: str = "decord",
        video_backend_kwargs: dict | None = None,
        transforms: ComposedModalityTransform | None = None,
        delete_pause_frame: bool = False,
        data_cfg=None,
        **kwargs,
    ):
        """
        Initialize the dataset.

        Args:
            dataset_path (Path | str): The path to the dataset.
            modality_configs (dict[str, ModalityConfig]): The configuration for each modality. The keys are the modality names, and the values are the modality configurations.
                See `ModalityConfig` for more details.
            video_backend (str): Backend for video reading.
            video_backend_kwargs (dict): Keyword arguments for the video backend when initializing the video reader.
            transforms (ComposedModalityTransform): The transforms to apply to the dataset.
            embodiment_tag (EmbodimentTag): Overload the embodiment tag for the dataset. e.g. define it as "new_embodiment"
            delete_pause_frame (bool): Whether to drop pause frames when indexing steps.
            data_cfg: The stream config block (``cfg.datasets.<block>``).
        """
        # first check if the path directory exists
        self.data_cfg = data_cfg
        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")
        # LeRobot on-disk format version ("v2.0" or "v3.0").
        self._lerobot_version = self.data_cfg.get("lerobot_version", "v2.0")

        self._action_mode = None
        self._action_mode_state_map = {}
        self._action_mode_apply_keys = None

        self.delete_pause_frame = delete_pause_frame

        self.modality_configs = modality_configs
        self.video_backend = video_backend
        self.video_backend_kwargs = video_backend_kwargs if video_backend_kwargs is not None else {}
        self.transforms = transforms if transforms is not None else ComposedModalityTransform(transforms=[])

        self._dataset_path = Path(dataset_path)
        self._dataset_name = self._dataset_path.name
        if isinstance(embodiment_tag, EmbodimentTag):
            self.tag = embodiment_tag.value
        else:
            self.tag = embodiment_tag

        self._init_action_mode()
        self._metadata = self._get_metadata(EmbodimentTag(self.tag))

        # LeRobot-specific config
        self._lerobot_modality_meta = self._get_lerobot_modality_meta()
        self._lerobot_info_meta = self._get_lerobot_info_meta()
        self._data_path_pattern = self._get_data_path_pattern()
        self._video_path_pattern = self._get_video_path_pattern()
        self._chunk_size = self._get_chunk_size()
        self._tasks = self._get_tasks()
        self.curr_traj_data = None
        self.curr_traj_id = None

        self._trajectory_ids, self._trajectory_lengths = self._get_trajectories()
        self._video_override_provenance = {}
        self._video_overrides = self._init_video_overrides()
        self._modality_keys = self._get_modality_keys()
        self._delta_indices = self._get_delta_indices()
        self._all_steps = self._get_all_steps()
        # Optionally restrict training to a deterministic fraction of episodes.
        # Runs AFTER _get_all_steps() so the (full) steps cache is already on disk
        # (episode_fraction never pollutes the shared meta/steps_data_index.pkl) and
        # BEFORE the mixture reads trajectory_ids/lengths, so it inherits the subset.
        self._apply_episode_fraction()
        # Keep the raw trajectory lengths for data bounds, but optionally restrict the
        # base indices the direct and mixture samplers may draw from. This runs after
        # episode selection so every sampling array remains position-aligned.
        self._apply_tail_policy()
        self.set_transforms_metadata(self.metadata)
        self.set_epoch(0)

        if int(os.environ.get("RANK", "0")) == 0:
            print(f"Initialized dataset {self.dataset_name} with {embodiment_tag}")

        # Check if the dataset is valid
        self._check_integrity()

    @property
    def dataset_path(self) -> Path:
        """The path to the dataset that contains the METADATA_FILENAME file."""
        return self._dataset_path

    @property
    def video_override_provenance(self) -> dict:
        """Resolved manifest/object identity used for immutable video repairs."""
        return copy.deepcopy(self._video_override_provenance)

    @property
    def metadata(self) -> DatasetMetadata:
        """The metadata for the dataset, loaded from metadata.json in the dataset directory"""
        return self._metadata

    @property
    def trajectory_ids(self) -> np.ndarray:
        """The trajectory IDs in the dataset, stored as a 1D numpy array of strings."""
        return self._trajectory_ids

    @property
    def trajectory_lengths(self) -> np.ndarray:
        """The trajectory lengths in the dataset, stored as a 1D numpy array of integers.
        The order of the lengths is the same as the order of the trajectory IDs.
        """
        return self._trajectory_lengths

    @property
    def sampleable_trajectory_lengths(self) -> np.ndarray:
        """Number of valid base indices for each trajectory.

        Unlike :attr:`trajectory_lengths`, these lengths describe the prefix from
        which a sampler may draw. Under ``tail_policy=drop`` the prefix excludes
        bases whose required future target would leave the episode.
        """
        return self._sampleable_trajectory_lengths

    @property
    def all_steps(self) -> list[tuple[int, int]]:
        """The trajectory IDs and base indices for all steps in the dataset.
        Example:
            self.trajectory_ids: [0, 1, 2]
            self.trajectory_lengths: [3, 2, 4]
            return: [
                ("traj_0", 0), ("traj_0", 1), ("traj_0", 2),
                ("traj_1", 0), ("traj_1", 1),
                ("traj_2", 0), ("traj_2", 1), ("traj_2", 2), ("traj_2", 3)
            ]
        """
        return self._all_steps

    @property
    def modality_keys(self) -> dict:
        """The modality keys for the dataset. The keys are the modality names, and the values are the keys for each modality.

        Example: {
            "video": ["video.image_side_0", "video.image_side_1"],
            "state": ["state.eef_position", "state.eef_rotation"],
            "action": ["action.eef_position", "action.eef_rotation"],
            "language": ["language.human.task"],
            "timestamp": ["timestamp"],
            "reward": ["reward"],
        }
        """
        return self._modality_keys

    @property
    def delta_indices(self) -> dict[str, np.ndarray]:
        """The delta indices for the dataset. The keys are the modality.key, and the values are the delta indices for each modality.key."""
        return self._delta_indices

    @property
    def dataset_name(self) -> str:
        """The name of the dataset."""
        return self._dataset_name

    @property
    def lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        """The metadata for the LeRobot dataset."""
        return self._lerobot_modality_meta

    @property
    def lerobot_info_meta(self) -> dict:
        """The metadata for the LeRobot dataset."""
        return self._lerobot_info_meta

    @property
    def data_path_pattern(self) -> str:
        """The path pattern for the LeRobot dataset."""
        return self._data_path_pattern

    @property
    def video_path_pattern(self) -> str:
        """The path pattern for the LeRobot dataset."""
        return self._video_path_pattern

    @property
    def chunk_size(self) -> int:
        """The chunk size for the LeRobot dataset."""
        return self._chunk_size

    @property
    def tasks(self) -> pd.DataFrame:
        """The tasks for the dataset."""
        return self._tasks

    def _get_metadata(self, embodiment_tag: EmbodimentTag) -> DatasetMetadata:
        """Get the metadata for the dataset.

        Returns:
            dict: The metadata for the dataset.
        """

        # 1. Modality metadata (dataset's own meta/modality.json, or a data_cfg override)
        modality_meta_path = self._resolve_modality_meta_path()
        # 1.1. State and action modalities
        simplified_modality_meta: dict[str, dict] = {}
        with open(modality_meta_path, "r") as f:
            le_modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))
        for modality in ["state", "action"]:
            simplified_modality_meta[modality] = {}
            le_state_action_meta: dict[str, LeRobotStateActionMetadata] = getattr(le_modality_meta, modality)
            for subkey in le_state_action_meta:
                state_action_dtype = np.dtype(le_state_action_meta[subkey].dtype)
                if np.issubdtype(state_action_dtype, np.floating):
                    continuous = True
                else:
                    continuous = False
                simplified_modality_meta[modality][subkey] = {
                    "absolute": le_state_action_meta[subkey].absolute,
                    "rotation_type": le_state_action_meta[subkey].rotation_type,
                    "shape": [le_state_action_meta[subkey].end - le_state_action_meta[subkey].start],
                    "continuous": continuous,
                }

        # 1.2. Video modalities
        le_info_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        assert le_info_path.exists(), f"Please provide a {LE_ROBOT_INFO_FILENAME} file in {self.dataset_path}"
        with open(le_info_path, "r") as f:
            le_info = json.load(f)
        simplified_modality_meta["video"] = {}
        for new_key in le_modality_meta.video:
            original_key = le_modality_meta.video[new_key].original_key
            if original_key is None:
                original_key = new_key
            le_video_meta = le_info["features"][original_key]
            # Height/width across LeRobot versions: v2.0 has names=["height","width","channel"];
            # v3.0 (e.g. DROID) drops "names" but carries info.video.{height,width}. Fall back to
            # the [height, width, channel] shape convention if neither is present.
            names = le_video_meta.get("names") or []
            info = le_video_meta.get("info", {}) or {}
            if "height" in names and "width" in names:
                height = le_video_meta["shape"][names.index("height")]
                width = le_video_meta["shape"][names.index("width")]
            elif "video.height" in info and "video.width" in info:
                height, width = info["video.height"], info["video.width"]
            else:
                height, width = le_video_meta["shape"][0], le_video_meta["shape"][1]
            # NOTE(FH): different lerobot dataset versions have different keys for the number of channels and fps
            try:
                channels = le_video_meta["shape"][le_video_meta["names"].index("channel")]
                fps = le_video_meta["video_info"]["video.fps"]
            except (ValueError, KeyError):
                try:
                    channels = le_video_meta["info"]["video.channels"]
                    fps = le_video_meta["info"]["video.fps"]
                except (ValueError, KeyError):
                    # Fallback for image-only datasets (e.g. VLA-Arena) that lack video_info
                    channels = 3
                    fps = le_info.get("fps", 30)
            simplified_modality_meta["video"][new_key] = {
                "resolution": [width, height],
                "channels": channels,
                "fps": fps,
            }

        # 2. Dataset statistics
        def is_main():
            return (not dist.is_initialized()) or dist.get_rank() == 0

        action_mode = _normalize_action_mode(self.data_cfg.get("action_mode", "abs") if self.data_cfg else "abs")

        stats_path = self.dataset_path / LE_ROBOT_STATS_FILENAME
        action_cfg = self.modality_configs.get("action")
        state_cfg = self.modality_configs.get("state")
        action_keys_full = list(action_cfg.modality_keys) if action_cfg else []
        state_keys_full = list(state_cfg.modality_keys) if state_cfg else []
        action_indices = list(action_cfg.delta_indices) if action_cfg else None
        state_indices = list(state_cfg.delta_indices) if state_cfg else None

        apply_keys = _normalize_action_mode_apply_keys(
            self.data_cfg.get("action_mode_apply_keys", None) if self.data_cfg else None,
            action_keys_full,
        )
        normalized_state_map = _normalize_action_mode_state_map(
            self.data_cfg.get("action_mode_state_map", {}) if self.data_cfg else {}
        )
        stats_cache_config = _build_stats_cache_config(
            action_mode=action_mode,
        )
        parquet_files = list(self.dataset_path.glob(LE_ROBOT_DATA_FILENAME))
        parquet_files_filtered = [pf for pf in parquet_files if "episode_033675.parquet" not in pf.name]

        if is_main():
            le_statistics = _load_or_compute_statistics(
                stats_path,
                stats_cache_config=stats_cache_config,
                parquet_paths=parquet_files_filtered,
                dataset_name=self.dataset_name,
                action_mode=action_mode,
                lerobot_modality_meta=le_modality_meta,
                action_keys_full=action_keys_full,
                state_keys_full=state_keys_full,
                action_indices=action_indices,
                state_indices=state_indices,
                action_mode_apply_keys=apply_keys,
                action_mode_state_map=normalized_state_map,
            )
        else:
            le_statistics = None

        if dist.is_initialized():
            distributed_barrier()

        if le_statistics is None:
            le_statistics = _load_shared_cache_with_retry(
                lambda: _load_stats_cache_any(
                    stats_path,
                    stats_cache_config,
                    invalidate_legacy=False,
                )
            )
            if le_statistics is None:
                raise RuntimeError(
                    f"Dataset statistics cache is missing or invalid after sync: {stats_path} "
                    f"(and fallback {_fallback_stats_path(stats_path)}). For a read-only dataset "
                    f"mount in a multi-node run, set STARVLA_STATS_CACHE_DIR to a SHARED writable path."
                )

        for stat in le_statistics.values():
            DatasetStatisticalValues.model_validate(stat)

        dataset_statistics = {}
        for our_modality in ["state", "action"]:
            dataset_statistics[our_modality] = {}
            for subkey in simplified_modality_meta[our_modality]:
                dataset_statistics[our_modality][subkey] = {}
                state_action_meta = le_modality_meta.get_key_meta(f"{our_modality}.{subkey}")
                assert isinstance(state_action_meta, LeRobotStateActionMetadata)
                le_modality = state_action_meta.original_key
                for stat_name in le_statistics[le_modality]:
                    indices = np.arange(
                        state_action_meta.start,
                        state_action_meta.end,
                    )
                    stat = np.array(le_statistics[le_modality][stat_name])
                    dataset_statistics[our_modality][subkey][stat_name] = stat[indices].tolist()

        # 3. Full dataset metadata
        metadata = DatasetMetadata(
            statistics=dataset_statistics,  # type: ignore
            modalities=simplified_modality_meta,  # type: ignore
            embodiment_tag=embodiment_tag,
        )

        return metadata

    def _get_trajectories(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the trajectories in the dataset."""
        # Get trajectory lengths, IDs, and whitelist from dataset metadata
        # v2.0
        if self._lerobot_version == "v2.0":
            if parse_episode_filter(self.data_cfg) is not None:
                raise ValueError("episode_filter reads LeRobot v3.0 episode metadata; it is not supported for v2.0.")
            file_path = self.dataset_path / LE_ROBOT_EPISODE_FILENAME
            with open(file_path, "r") as f:
                episode_metadata = [json.loads(line) for line in f]
            trajectory_ids = []
            trajectory_lengths = []
            for episode in episode_metadata:
                trajectory_ids.append(episode["episode_index"])
                trajectory_lengths.append(episode["length"])
            return np.array(trajectory_ids), np.array(trajectory_lengths)
        # v3.0
        elif self._lerobot_version == "v3.0":
            file_paths = sorted(list((self.dataset_path).glob(LE_ROBOT3_EPISODE_FILENAME)))
            trajectory_ids = []
            trajectory_lengths = []
            self.trajectory_ids_to_metadata = {}
            self._v3_video_index_schema = None
            required_video_keys = self._required_v3_video_original_keys()
            episode_filter = parse_episode_filter(self.data_cfg)
            action_original_keys = self._action_original_keys() if episode_filter is not None else []
            rejected = defaultdict(int)
            for file_path in file_paths:
                episodes_data = pd.read_parquet(file_path)
                if episodes_data.empty:
                    continue
                episode_columns = set(map(str, episodes_data.columns))
                chunk_keys = {
                    str(col)[len("videos/") : -len("/chunk_index")]
                    for col in episodes_data.columns
                    if str(col).startswith("videos/") and str(col).endswith("/chunk_index")
                }
                file_keys = {
                    str(col)[len("videos/") : -len("/file_index")]
                    for col in episodes_data.columns
                    if str(col).startswith("videos/") and str(col).endswith("/file_index")
                }
                file_schema = "modern" if chunk_keys or file_keys else "legacy"
                if self._v3_video_index_schema is None:
                    self._v3_video_index_schema = file_schema
                elif self._v3_video_index_schema != file_schema:
                    first_episode = episodes_data.iloc[0]
                    missing_fields = (
                        "all per-camera chunk/file indices" if file_schema == "legacy" else "in a previous metadata file"
                    )
                    raise ValueError(
                        f"LeRobot v3 video-index schema is mixed under {self.dataset_path}: "
                        f"previous files were {self._v3_video_index_schema}, but {file_path} is {file_schema}. "
                        f"episode={first_episode['episode_index']}, cameras={sorted(required_video_keys)}, "
                        f"data_shard=({first_episode['data/chunk_index']}, {first_episode['data/file_index']}), "
                        f"missing_fields={missing_fields}. "
                        "All episode metadata files must consistently use per-camera indices or the legacy schema."
                    )

                timestamp_cols = [
                    c
                    for c in episodes_data.columns
                    if str(c).startswith("videos/") and str(c).endswith("/from_timestamp")
                ]
                for index, episode in episodes_data.iterrows():
                    episode_id = self._parse_v3_video_shard_index(
                        episode["episode_index"],
                        "episode_index",
                        f"source={file_path}",
                    )
                    if episode_id in self.trajectory_ids_to_metadata:
                        raise ValueError(
                            f"Duplicate LeRobot v3 episode_index={episode_id} under "
                            f"{self.dataset_path}; source={file_path}."
                        )
                    episode_length = self._parse_v3_video_shard_index(
                        episode["length"],
                        "length",
                        f"episode={episode_id}, source={file_path}",
                    )
                    if episode_length == 0:
                        raise ValueError(
                            f"LeRobot v3 episode length must be positive: "
                            f"episode={episode_id}, length={episode['length']!r}, source={file_path}."
                        )
                    data_chunk_index = self._parse_v3_video_shard_index(
                        episode["data/chunk_index"],
                        "data/chunk_index",
                        f"episode={episode_id}, source={file_path}",
                    )
                    data_file_index = self._parse_v3_video_shard_index(
                        episode["data/file_index"],
                        "data/file_index",
                        f"episode={episode_id}, source={file_path}",
                    )
                    if episode_filter is not None:
                        reason = episode_rejection_reason(episode, episode_columns, action_original_keys, episode_filter)
                        if reason is not None:
                            rejected[reason] += 1
                            continue
                    trajectory_ids.append(episode_id)
                    trajectory_lengths.append(episode_length)

                    from_timestamps = {}
                    for col in timestamp_cols:
                        value = episode[col]
                        if pd.isna(value):
                            continue
                        # videos/{video_key}/from_timestamp -> {video_key}
                        video_key = str(col)[len("videos/") : -len("/from_timestamp")]
                        from_timestamps[video_key] = self._parse_v3_video_timestamp(
                            value,
                            str(col),
                            f"episode={episode_id}, source={file_path}",
                        )

                    video_file_indices = {}
                    for video_key in sorted(required_video_keys if file_schema == "modern" else set()):
                        chunk_col = f"videos/{video_key}/chunk_index"
                        file_col = f"videos/{video_key}/file_index"
                        timestamp_col = f"videos/{video_key}/from_timestamp"
                        missing_fields = [
                            field
                            for field in (chunk_col, file_col, timestamp_col)
                            if field not in episodes_data.columns or pd.isna(episode.get(field, None))
                        ]
                        if missing_fields:
                            raise ValueError(
                                f"LeRobot v3 per-camera video provenance is incomplete: "
                                f"episode={episode_id}, camera={video_key}, "
                                f"data_shard=({data_chunk_index}, {data_file_index}), "
                                f"missing_fields={missing_fields}, source={file_path}. "
                                "Modern metadata cannot fall back to the data shard or timestamp zero."
                            )
                        chunk_value = episode[chunk_col]
                        file_value = episode[file_col]
                        context = (
                            f"episode={episode_id}, camera={video_key}, "
                            f"data_shard=({data_chunk_index}, {data_file_index}), "
                            f"source={file_path}"
                        )
                        video_chunk_index = self._parse_v3_video_shard_index(chunk_value, chunk_col, context)
                        video_file_index = self._parse_v3_video_shard_index(file_value, file_col, context)
                        from_timestamps[video_key] = self._parse_v3_video_timestamp(
                            episode[timestamp_col],
                            timestamp_col,
                            context,
                        )
                        video_file_indices[video_key] = {
                            "chunk_index": video_chunk_index,
                            "file_index": video_file_index,
                        }
                    episode_meta = {
                        "episode_index": episode_id,
                        "length": episode_length,
                        "data/chunk_index": data_chunk_index,
                        "data/file_index": data_file_index,
                        "data/file_from_index": index,
                        "videos/from_timestamps": from_timestamps,
                    }
                    if file_schema == "modern":
                        episode_meta["videos/file_indices"] = video_file_indices
                    self.trajectory_ids_to_metadata[episode_id] = episode_meta

            if self._v3_video_index_schema is None:
                raise ValueError(f"No LeRobot v3 episode metadata found under {self.dataset_path}.")
            if episode_filter is not None:
                total = len(trajectory_ids) + sum(rejected.values())
                if int(os.environ.get("RANK", "0")) == 0:
                    print(
                        f"[episode_filter] {self.dataset_path}: kept {len(trajectory_ids)}/{total} episodes "
                        f"({sum(trajectory_lengths)} frames); dropped {dict(rejected)}"
                    )
                if not trajectory_ids:
                    raise ValueError(f"episode_filter dropped every episode under {self.dataset_path}.")
            return np.array(trajectory_ids), np.array(trajectory_lengths)

    def _action_original_keys(self) -> list[str]:
        """Parquet keys of the action features in use (for metadata-based episode curation)."""
        action_config = self.modality_configs.get("action")
        keys = []
        for key in list(action_config.modality_keys) if action_config is not None else []:
            subkey = key.removeprefix("action.")
            if subkey not in self.lerobot_modality_meta.action:
                raise ValueError(
                    f"Configured action key {key!r} is missing from modality metadata; "
                    f"available={sorted(self.lerobot_modality_meta.action)}."
                )
            keys.append(self.lerobot_modality_meta.action[subkey].original_key or subkey)
        return keys

    @staticmethod
    def _parse_v3_video_shard_index(value, field: str, context: str) -> int:
        """Parse an exact non-negative v3 shard index with provenance-rich errors."""
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"Invalid LeRobot v3 video index {field}={value!r}; {context}.")
        if isinstance(value, (int, np.integer)):
            if value < 0:
                raise ValueError(f"Invalid LeRobot v3 video index {field}={value!r}; {context}.")
            return int(value)
        try:
            return parse_nonnegative_integer(value, field=field, context=context)
        except VideoOverrideError as exc:
            raise ValueError(f"Invalid LeRobot v3 video index {field}={value!r}; {context}.") from exc

    @staticmethod
    def _parse_v3_video_timestamp(value, field: str, context: str) -> float:
        """Parse a finite, non-negative per-camera start timestamp."""
        if isinstance(value, np.bool_):
            raise ValueError(f"Invalid LeRobot v3 video timestamp {field}={value!r}; {context}.")
        try:
            return parse_nonnegative_timestamp(value, field=field, context=context)
        except VideoOverrideError as exc:
            raise ValueError(f"Invalid LeRobot v3 video timestamp {field}={value!r}; {context}.") from exc

    def _init_video_overrides(self) -> dict[str, Path]:
        """Resolve exact, content-addressed replacements before episode splitting."""

        cfg = self.data_cfg if self.data_cfg is not None else {}
        configured = cfg.get("video_override_manifests", None)
        if configured is None:
            self._video_override_provenance = {"status": "unconfigured"}
            return {}
        if not hasattr(configured, "get") or not hasattr(configured, "keys"):
            raise VideoOverrideError(
                "video_override_manifests must map dataset leaf names to manifest paths; " f"got {configured!r}"
            )
        if self.dataset_name not in configured:
            raise VideoOverrideError(
                "video_override_manifests is present but does not explicitly name "
                f"dataset leaf {self.dataset_name!r}; configured={sorted(map(str, configured.keys()))}"
            )
        manifest_value = configured[self.dataset_name]
        if manifest_value is None:
            self._video_override_provenance = {"status": "explicit-none"}
            return {}
        if manifest_value == "":
            raise VideoOverrideError(
                f"video_override_manifests[{self.dataset_name!r}] must be an absolute "
                "manifest path or null, not an empty string"
            )
        if self._lerobot_version != "v3.0":
            raise VideoOverrideError(
                f"video overrides require LeRobot v3.0, got "
                f"dataset={self.dataset_name}, version={self._lerobot_version}"
            )

        manifest_path = Path(str(manifest_value))
        if not manifest_path.is_absolute():
            raise VideoOverrideError(
                f"video override manifest path must be absolute for dataset={self.dataset_name}: " f"{manifest_value!r}"
            )
        manifest, manifest_identity = load_manifest(manifest_path)
        if manifest.get("dataset_name") != self.dataset_name:
            raise VideoOverrideError(
                f"video override manifest dataset_name={manifest.get('dataset_name')!r} "
                f"does not match leaf {self.dataset_name!r}"
            )
        if manifest.get("lerobot_version") != self._lerobot_version:
            raise VideoOverrideError(
                f"video override manifest version mismatch for {self.dataset_name}: "
                f"manifest={manifest.get('lerobot_version')!r}, runtime={self._lerobot_version!r}"
            )
        if manifest.get("video_index_schema") != self._v3_video_index_schema:
            raise VideoOverrideError(
                f"video override manifest schema mismatch for {self.dataset_name}: "
                f"manifest={manifest.get('video_index_schema')!r}, "
                f"runtime={self._v3_video_index_schema!r}"
            )

        info_contract = manifest.get("dataset_info")
        if not isinstance(info_contract, dict):
            raise VideoOverrideError("video override manifest requires a dataset_info identity")
        info_relative = normalize_logical_path(info_contract.get("logical_path"))
        if info_relative != LE_ROBOT_INFO_FILENAME:
            raise VideoOverrideError(
                f"dataset_info.logical_path must be {LE_ROBOT_INFO_FILENAME!r}, got {info_relative!r}"
            )
        expected_info = parse_file_identity(info_contract, field="dataset_info")
        actual_info = file_identity(self.dataset_path / info_relative, label="dataset_info")
        if actual_info != expected_info:
            raise VideoOverrideError(
                f"dataset_info identity mismatch for {self.dataset_name}: "
                f"actual={actual_info}, expected={expected_info}"
            )

        entries = manifest.get("entries")
        if not isinstance(entries, list) or not entries:
            raise VideoOverrideError("video override manifest entries must be a non-empty list")

        available_cameras = {metadata.original_key or key for key, metadata in self.lerobot_modality_meta.video.items()}
        resolved: dict[str, Path] = {}
        provenance_entries = []
        for entry_index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise VideoOverrideError(f"entries[{entry_index}] must be an object")
            camera = entry.get("camera")
            if camera not in available_cameras:
                raise VideoOverrideError(
                    f"entries[{entry_index}].camera={camera!r} is not registered; "
                    f"available={sorted(available_cameras)}"
                )
            chunk_index = self._parse_v3_video_shard_index(
                entry.get("chunk_index"),
                f"entries[{entry_index}].chunk_index",
                f"dataset={self.dataset_name}",
            )
            file_index = self._parse_v3_video_shard_index(
                entry.get("file_index"),
                f"entries[{entry_index}].file_index",
                f"dataset={self.dataset_name}",
            )
            logical_path = normalize_logical_path(entry.get("logical_path"))
            expected_logical_path = self.video_path_pattern.format(
                video_key=camera,
                chunk_index=chunk_index,
                file_index=file_index,
            )
            if logical_path != expected_logical_path:
                raise VideoOverrideError(
                    f"entries[{entry_index}] logical path disagrees with camera/shard metadata: "
                    f"stored={logical_path!r}, expected={expected_logical_path!r}"
                )
            if logical_path in resolved:
                raise VideoOverrideError(f"duplicate video override entry for {logical_path!r}")

            records = []
            for episode_id in sorted(self.trajectory_ids_to_metadata):
                episode_meta = self.trajectory_ids_to_metadata[episode_id]
                camera_index = episode_meta.get("videos/file_indices", {}).get(camera)
                if camera_index != {"chunk_index": chunk_index, "file_index": file_index}:
                    continue
                from_timestamp = episode_meta.get("videos/from_timestamps", {}).get(camera)
                if from_timestamp is None:
                    raise VideoOverrideError(
                        f"{logical_path}: validated camera timestamp is missing at manifest "
                        f"resolution time; episode={episode_id}, camera={camera}"
                    )
                records.append(
                    {
                        "episode_index": int(episode_id),
                        "episode_length": int(episode_meta["length"]),
                        "data_chunk_index": int(episode_meta["data/chunk_index"]),
                        "data_file_index": int(episode_meta["data/file_index"]),
                        "camera_chunk_index": int(camera_index["chunk_index"]),
                        "camera_file_index": int(camera_index["file_index"]),
                        "from_timestamp": float(from_timestamp),
                    }
                )
            if not records:
                raise VideoOverrideError(f"{logical_path}: no episode metadata points to the declared camera shard")
            episode_ids = [record["episode_index"] for record in records]
            declared_episode_ids = entry.get("episode_indices")
            if declared_episode_ids != episode_ids:
                raise VideoOverrideError(
                    f"{logical_path}: episode provenance mismatch: "
                    f"manifest={declared_episode_ids!r}, runtime={episode_ids!r}"
                )
            metadata_sha256 = canonical_metadata_digest(records)
            if entry.get("metadata_sha256") != metadata_sha256:
                raise VideoOverrideError(
                    f"{logical_path}: camera/timestamp metadata digest mismatch: "
                    f"manifest={entry.get('metadata_sha256')!r}, runtime={metadata_sha256}"
                )

            selected = resolve_entry(dataset_path=self.dataset_path, entry=entry)
            resolved[logical_path] = selected.selected_path
            provenance_entries.append(
                {
                    "logical_path": logical_path,
                    "camera": camera,
                    "chunk_index": chunk_index,
                    "file_index": file_index,
                    "episode_indices": episode_ids,
                    "metadata_sha256": metadata_sha256,
                    "selected_source": selected.selected_source,
                    "selected_path": str(selected.selected_path),
                    "expected_good": {
                        "size": selected.expected_good.size,
                        "sha256": selected.expected_good.sha256,
                    },
                }
            )
            if int(os.environ.get("RANK", "0")) == 0:
                print(
                    "[video-override] "
                    f"dataset={self.dataset_name} logical_path={logical_path} "
                    f"source={selected.selected_source} selected={selected.selected_path} "
                    f"sha256={selected.expected_good.sha256}"
                )
        self._video_override_provenance = {
            "status": "active",
            "manifest_path": str(manifest_path),
            "manifest_size": manifest_identity.size,
            "manifest_sha256": manifest_identity.sha256,
            "entries": provenance_entries,
        }
        return resolved

    def _resolve_video_original_key(self, key: str) -> str:
        """Resolve ``video.foo`` or ``foo`` to the dataset's physical camera key."""
        subkey = key.removeprefix("video.")
        if subkey not in self.lerobot_modality_meta.video:
            raise ValueError(
                f"Configured video key {key!r} is missing from modality metadata; "
                f"available={sorted(self.lerobot_modality_meta.video)}."
            )
        key_meta = self.lerobot_modality_meta.video[subkey]
        return key_meta.original_key or subkey

    def _required_v3_video_original_keys(self) -> set[str]:
        """Original physical cameras whose v3 MP4 provenance must be present.

        ``gen_video_key`` is a semantic role (for example ``central_global``) or
        the virtual target ``tile``. Neither is a metadata camera key. Generation
        validates that role against the active physical views in its own loader;
        provenance validation must only enumerate the cameras enabled by the
        already-resolved modality config.
        """
        video_config = self.modality_configs.get("video")
        configured = list(video_config.modality_keys) if video_config is not None else []

        original_keys = set()
        for key in configured:
            original_keys.add(self._resolve_video_original_key(key))
        return original_keys

    def _get_all_steps(self) -> list[tuple[int, int]]:
        """Get the trajectory IDs and base indices for all steps in the dataset.

        Returns:
            list[tuple[str, int]]: A list of (trajectory_id, base_index) tuples.
        """

        def is_main():
            return (not dist.is_initialized()) or dist.get_rank() == 0

        config_key = self._get_steps_config_key()
        steps_filename = "steps_data_index.pkl"
        steps_path = self.dataset_path / "meta" / steps_filename
        # Writable mirror for read-only dataset mounts (same policy as the stats cache).
        fallback_steps_path = _fallback_cache_path(steps_path)

        def _try_load_steps(p: Path):
            if not p.exists():
                return None
            try:
                with open(p, "rb") as f:
                    return pickle.load(f)["steps"]
            except Exception as e:
                # include EOFError / PickleError / KeyError
                print(f"[RANK {os.environ.get('RANK', 'NA')}] Failed to load cached steps at {p} ({e}), will rebuild.")
                return None

        def _try_load_steps_any():
            cached = _try_load_steps(steps_path)
            if cached is None:
                cached = _try_load_steps(fallback_steps_path)
            return cached

        # Rank 0 is authoritative so every distributed rank executes the same
        # barrier even when a newly published shared cache is not yet visible on
        # every node.
        cached_steps = None
        built_steps = None
        if is_main():
            cached_steps = _try_load_steps_any()
            if cached_steps is None:
                built_steps = self._get_all_steps_single_process()

                cache_data = {
                    "config_key": config_key,
                    "steps": built_steps,
                    "num_trajectories": len(self.trajectory_ids),
                    "total_steps": len(built_steps),
                    "computed_timestamp": pd.Timestamp.now().isoformat(),
                    "delete_pause_frame": self.delete_pause_frame,
                }

                # An explicit fallback directory is also an instruction never to probe
                # the canonical dataset mount for writability.
                for target in _cache_write_targets(steps_path, fallback_steps_path):
                    try:
                        _save_steps_cache(target, cache_data)
                        print(f"[RANK 0] Cached steps saved to {target}")
                        break
                    except OSError as e:
                        print(f"[steps-cache] could not write {target} ({e}); trying next location")

        if dist.is_initialized():
            distributed_barrier()

        if cached_steps is not None:
            return cached_steps
        if built_steps is not None:
            return built_steps

        cached_steps = _load_shared_cache_with_retry(_try_load_steps_any)
        if cached_steps is not None:
            return cached_steps
        raise RuntimeError(
            f"Steps index cache missing after sync: {steps_path} (and fallback {fallback_steps_path}). "
            f"For a read-only dataset mount in a multi-node run, set STARVLA_STATS_CACHE_DIR to a SHARED writable path."
        )

    def _get_steps_config_key(self) -> str:
        """Generate a configuration key for steps caching."""
        config_dict = {
            "delete_pause_frame": self.delete_pause_frame,
            "dataset_name": self.dataset_name,
        }
        episode_filter = parse_episode_filter(self.data_cfg)
        if episode_filter is not None:
            # A curated episode set must not reuse the steps index of the full dataset.
            config_dict["episode_filter"] = sorted(episode_filter.items())
        # Create a hash of the configuration
        config_str = str(sorted(config_dict.items()))
        return hashlib.md5(config_str.encode()).hexdigest()[:12]  #

    def _get_all_steps_single_process(self) -> list[tuple[int, int]]:
        """Original single-process implementation as fallback."""
        all_steps: list[tuple[int, int]] = []
        skipped_trajectories = 0
        processed_trajectories = 0

        # Check if language modality is configured
        has_language_modality = "language" in self.modality_keys and len(self.modality_keys["language"]) > 0
        # Pre-filtering empty-language trajectories reads EVERY episode's parquet, which is
        # O(episodes) I/O — prohibitive for large v3.0 datasets (e.g. DROID: 57k episodes ->
        # ~10h). `prevalidate_language: false` skips this pre-check (safe when language is
        # always present, e.g. a task_index -> tasks.parquet mapping); trajectory_length
        # comes from metadata (the zip below), not from reading the data.
        prevalidate_language = self.data_cfg.get("prevalidate_language", True) if self.data_cfg else True
        for trajectory_id, trajectory_length in tqdm(
            zip(self.trajectory_ids, self.trajectory_lengths, strict=False),
            total=len(self.trajectory_ids),
            desc="Getting All Step",
        ):
            try:
                # Check if trajectory has valid language instruction (if configured + enabled)
                if has_language_modality and prevalidate_language:
                    if self._lerobot_version == "v2.0":
                        data = self.get_trajectory_data(trajectory_id)
                    elif self._lerobot_version == "v3.0":
                        data = self.get_trajectory_data_lerobot_v3(trajectory_id)
                    self.curr_traj_data = data  # Set current trajectory data for get_language to work

                    language_instruction = self.get_language(trajectory_id, self.modality_keys["language"][0], 0)
                    if not language_instruction or language_instruction[0] == "":
                        print(f"Skipping trajectory {trajectory_id} due to empty language instruction")
                        skipped_trajectories += 1
                        continue

            except Exception as e:
                print(f"Skipping trajectory {trajectory_id} due to read error: {e}")
                skipped_trajectories += 1
                continue

            processed_trajectories += 1

            for base_index in range(trajectory_length):
                all_steps.append((trajectory_id, base_index))

        # Print summary statistics
        print(
            f"Single-process summary: Processed {processed_trajectories} trajectories, skipped {skipped_trajectories} empty trajectories"
        )
        print(f"Total steps: {len(all_steps)} from {len(self.trajectory_ids)} trajectories")

        return all_steps

    def _get_position_and_gripper_values(self, data: pd.DataFrame) -> tuple[list, list]:
        """Get position and gripper values based on available columns in the dataset."""
        # Get action keys from modality_keys
        action_keys = self.modality_keys.get("action", [])

        # Extract position data
        delta_position_values = None
        position_candidates = ["delta_eef_position"]
        coordinate_candidates = ["x", "y", "z"]

        # First try combined position fields
        for pos_key in position_candidates:
            full_key = f"action.{pos_key}"
            if full_key in action_keys:
                try:
                    # Get the lerobot key for this modality
                    le_action_cfg = self.lerobot_modality_meta.action
                    subkey = pos_key
                    if subkey in le_action_cfg:
                        le_key = le_action_cfg[subkey].original_key or subkey
                        if le_key in data.columns:
                            data_array = np.stack(data[le_key])
                            le_indices = np.arange(le_action_cfg[subkey].start, le_action_cfg[subkey].end)
                            filtered_data = data_array[:, le_indices]
                            delta_position_values = filtered_data.tolist()
                            break
                except Exception:
                    continue

        # If combined fields not found, try individual x,y,z coordinates
        if delta_position_values is None:
            x_data, y_data, z_data = None, None, None
            for coord in coordinate_candidates:
                full_key = f"action.{coord}"
                if full_key in action_keys:
                    try:
                        le_action_cfg = self.lerobot_modality_meta.action
                        if coord in le_action_cfg:
                            le_key = le_action_cfg[coord].original_key or coord
                            if le_key in data.columns:
                                data_array = np.stack(data[le_key])
                                le_indices = np.arange(le_action_cfg[coord].start, le_action_cfg[coord].end)
                                coord_data = data_array[:, le_indices].flatten()
                                if coord == "x":
                                    x_data = coord_data
                                elif coord == "y":
                                    y_data = coord_data
                                elif coord == "z":
                                    z_data = coord_data
                    except Exception:
                        continue

            if x_data is not None and y_data is not None and z_data is not None:
                delta_position_values = np.column_stack((x_data, y_data, z_data)).tolist()

        if delta_position_values is None:
            # Fallback to the old hardcoded approach if metadata approach fails
            if "action.delta_eef_position" in data.columns:
                delta_position_values = data["action.delta_eef_position"].to_numpy().tolist()
            elif all(col in data.columns for col in ["action.x", "action.y", "action.z"]):
                x_vals = data["action.x"].to_numpy()
                y_vals = data["action.y"].to_numpy()
                z_vals = data["action.z"].to_numpy()
                delta_position_values = np.column_stack((x_vals, y_vals, z_vals)).tolist()
            else:
                raise ValueError(f"No suitable position columns found. Available columns: {data.columns.tolist()}")

        # Extract gripper data
        gripper_values = None
        gripper_candidates = ["gripper_close", "gripper"]

        for grip_key in gripper_candidates:
            full_key = f"action.{grip_key}"
            if full_key in action_keys:
                try:
                    le_action_cfg = self.lerobot_modality_meta.action
                    if grip_key in le_action_cfg:
                        le_key = le_action_cfg[grip_key].original_key or grip_key
                        if le_key in data.columns:
                            data_array = np.stack(data[le_key])
                            le_indices = np.arange(le_action_cfg[grip_key].start, le_action_cfg[grip_key].end)
                            gripper_data = data_array[:, le_indices].flatten()
                            gripper_values = gripper_data.tolist()
                            break
                except Exception:
                    continue

        if gripper_values is None:
            # Fallback to the old hardcoded approach if metadata approach fails
            if "action.gripper_close" in data.columns:
                gripper_values = data["action.gripper_close"].to_numpy().tolist()
            elif "action.gripper" in data.columns:
                gripper_values = data["action.gripper"].to_numpy().tolist()
            else:
                raise ValueError(f"No suitable gripper columns found. Available columns: {data.columns.tolist()}")

        return delta_position_values, gripper_values

    def _get_modality_keys(self) -> dict:
        """Get the modality keys for the dataset.
        The keys are the modality names, and the values are the keys for each modality.
        See property `modality_keys` for the expected format.
        """
        modality_keys = defaultdict(list)
        for modality, config in self.modality_configs.items():
            modality_keys[modality] = config.modality_keys
        return modality_keys

    def _get_delta_indices(self) -> dict[str, np.ndarray]:
        """Restructure the delta indices to use modality.key as keys instead of just the modalities."""
        delta_indices: dict[str, np.ndarray] = {}
        for config in self.modality_configs.values():
            for key in config.modality_keys:
                delta_indices[key] = np.array(config.delta_indices)
        return delta_indices

    def _init_action_mode(self) -> None:
        if self.data_cfg is None:
            self._action_mode = "abs"
            return

        action_mode = self.data_cfg.get("action_mode", "abs")
        if action_mode is None:
            action_mode = "abs"
        action_mode = _normalize_action_mode(action_mode)
        if action_mode not in {"abs", "delta", "rel"}:
            raise ValueError(f"Invalid action_mode: {action_mode}. Expected one of: abs, delta, rel.")
        self._action_mode = action_mode

        apply_keys = _normalize_action_mode_apply_keys(self.data_cfg.get("action_mode_apply_keys", None))
        if apply_keys:
            self._action_mode_apply_keys = apply_keys

        self._action_mode_state_map = _normalize_action_mode_state_map(
            self.data_cfg.get("action_mode_state_map", {}) or {}
        )

    def _infer_state_key_for_action(self, action_key: str) -> str | None:
        if action_key in self._action_mode_state_map:
            return self._action_mode_state_map[action_key]

        if not action_key.startswith("action."):
            return None
        base = action_key.replace("action.", "", 1)
        if f"state.{base}" in self.modality_keys.get("state", []):
            return f"state.{base}"
        return None

    def _apply_action_mode(self, data: dict) -> dict:
        if self._action_mode in (None, "abs"):
            return data

        action_keys = self._action_mode_apply_keys or self.modality_keys.get("action", [])
        for action_key in action_keys:
            if action_key not in data:
                print(f"[WARNING] Action key {action_key} not found in data")
                continue
            state_key = self._infer_state_key_for_action(action_key)

            # for safety, check if the state key is valid
            if state_key is None or state_key not in data:
                continue

            action_values = np.asarray(data[action_key])
            state_values = np.asarray(data[state_key])
            if action_values.ndim != 2 or state_values.ndim != 2:
                raise ValueError(
                    f"Expected 2D arrays for action/state, got {action_key}: {action_values.shape}, {state_key}: {state_values.shape}"
                )
            if action_values.shape[1] != state_values.shape[1]:
                raise ValueError(
                    f"Action/state dim mismatch for {action_key} vs {state_key}: {action_values.shape} vs {state_values.shape}"
                )

            state0 = state_values[0]
            if self._action_mode == "delta":
                out = action_values.copy()
                if len(out) > 1:
                    out[1:] = action_values[1:] - action_values[:-1]
                out[0] = action_values[0] - state0
            elif self._action_mode == "rel":
                out = action_values - state0
            else:
                out = action_values

            data[action_key] = out

        return data

    def _resolve_modality_meta_path(self) -> Path:
        """Path to the gr00t ``modality.json``.

        Prefers the dataset's own ``meta/modality.json``; falls back to
        ``data_cfg.modality_json_path`` for RAW LeRobot datasets that ship without one
        (e.g. DROID v3.0 on a read-only mount — the mapping lives in the repo instead).
        Datasets that ship meta/modality.json always use their own file."""
        p = self.dataset_path / LE_ROBOT_MODALITY_FILENAME
        if p.exists():
            return p
        cfg = getattr(self, "data_cfg", None)
        override = cfg.get("modality_json_path", None) if (cfg is not None and hasattr(cfg, "get")) else None
        if override:
            op = Path(override)
            if op.exists():
                return op
            raise FileNotFoundError(
                f"datasets.<block>.modality_json_path={override!r} does not exist "
                f"(and no {LE_ROBOT_MODALITY_FILENAME} in {self.dataset_path})."
            )
        raise FileNotFoundError(
            f"Please provide a {LE_ROBOT_MODALITY_FILENAME} file in {self.dataset_path}, or set "
            f"datasets.<block>.modality_json_path to a modality.json for this raw LeRobot dataset."
        )

    def _get_lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        """Get the metadata for the LeRobot dataset."""
        modality_meta_path = self._resolve_modality_meta_path()
        with open(modality_meta_path, "r") as f:
            modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))
        return modality_meta

    def _get_lerobot_info_meta(self) -> dict:
        """Get the metadata for the LeRobot dataset."""
        info_meta_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        with open(info_meta_path, "r") as f:
            info_meta = json.load(f)
        return info_meta

    def _get_data_path_pattern(self) -> str:
        """Get the data path pattern for the LeRobot dataset."""
        return self.lerobot_info_meta["data_path"]

    def _get_video_path_pattern(self) -> str:
        """Get the video path pattern for the LeRobot dataset."""
        return self.lerobot_info_meta["video_path"]

    def _get_chunk_size(self) -> int:
        """Get the chunk size for the LeRobot dataset."""
        return self.lerobot_info_meta["chunks_size"]

    def _get_tasks(self) -> pd.DataFrame:
        """Get the tasks for the dataset."""
        if self._lerobot_version == "v2.0":
            tasks_path = self.dataset_path / LE_ROBOT_TASKS_FILENAME
            with open(tasks_path, "r") as f:
                tasks = [json.loads(line) for line in f]
            df = pd.DataFrame(tasks)
            return df.set_index("task_index")

        elif self._lerobot_version == "v3.0":
            tasks_path = self.dataset_path / LE_ROBOT3_TASKS_FILENAME
            df = pd.read_parquet(tasks_path)
            df = df.reset_index()  # convert index to a column, typically named 'index'
            df = df.rename(columns={"index": "task"})  # rename 'index' column to 'task'
            df = df[["task_index", "task"]]  # reorder columns
            return df

    def _check_integrity(self):
        """Use the config to check if the keys are valid and detect silent data corruption."""
        ERROR_MSG_HEADER = f"Error occurred in initializing dataset {self.dataset_name}:\n"

        for modality_config in self.modality_configs.values():
            for key in modality_config.modality_keys:
                if key == "lapa_action" or key == "dream_actions":
                    continue  # no need for any metadata for lapa actions because it comes normalized
                # Check if the key is valid
                try:
                    self.lerobot_modality_meta.get_key_meta(key)
                except Exception as e:
                    raise ValueError(ERROR_MSG_HEADER + f"Unable to find key {key} in modality metadata:\n{e}") from e

    def _get_official_split_positions(self, split: str) -> np.ndarray:
        """Resolve an official ``start:end`` episode-ID split to array positions.

        LeRobot stores split ranges in ``meta/info.json`` as episode IDs, while
        the sampler indexes the positional ``_trajectory_ids`` array. Resolve
        through an explicit ID-to-position map so reordered metadata cannot
        silently select the wrong episodes.
        """
        splits = self.lerobot_info_meta.get("splits")
        if not isinstance(splits, dict):
            raise ValueError(f"use_official_splits=true requires meta/info.json['splits'] " f"for {self.dataset_name}")
        if split not in splits:
            available = ", ".join(sorted(str(name) for name in splits)) or "<none>"
            raise ValueError(
                f"Official split {split!r} is not defined for {self.dataset_name}; " f"available splits: {available}"
            )

        spec = splits[split]
        if not isinstance(spec, str):
            raise ValueError(
                f"Official split {split!r} for {self.dataset_name} must be a " f"'start:end' string, got {spec!r}"
            )
        bounds = [part.strip() for part in spec.split(":")]
        if len(bounds) != 2 or any(not part.isdigit() for part in bounds):
            raise ValueError(
                f"Official split {split!r} for {self.dataset_name} must be a "
                f"non-negative 'start:end' range, got {spec!r}"
            )
        start, end = (int(part) for part in bounds)
        if start >= end:
            raise ValueError(
                f"Official split {split!r} for {self.dataset_name} must be non-empty " f"with start < end, got {spec!r}"
            )

        trajectory_ids = self._trajectory_ids.tolist()
        if len(set(trajectory_ids)) != len(trajectory_ids):
            raise ValueError(f"Duplicate episode IDs found in {self.dataset_name}")
        id_to_position = {episode_id: position for position, episode_id in enumerate(trajectory_ids)}
        missing = [episode_id for episode_id in range(start, end) if episode_id not in id_to_position]
        if missing:
            preview = ", ".join(str(episode_id) for episode_id in missing[:5])
            suffix = "..." if len(missing) > 5 else ""
            raise ValueError(
                f"Official split {split!r} for {self.dataset_name} references "
                f"{len(missing)} missing episode ID(s): {preview}{suffix}"
            )
        return np.asarray([id_to_position[episode_id] for episode_id in range(start, end)], dtype=np.int64)

    def _select_episode_indices(self) -> "np.ndarray | None":
        """Deterministically choose which episode *positions* this dataset keeps.

        Reads from ``data_cfg`` (all optional):
          * ``val_episode_fraction`` (float in [0, 1) or None): fraction of this
            sub-dataset's episodes carved out as a hold-out VALIDATION set. The
            carve is fixed and independent of ``episode_fraction``.
          * ``use_official_splits`` (bool, default False): resolve ``split`` from
            ``meta/info.json["splits"]`` instead of making a random hold-out.
            When enabled, this takes precedence over ``val_episode_fraction``.
          * ``split`` ("train" | "val" | "test", default "train"): which side
            to keep. ``test`` requires ``use_official_splits=true``.
          * ``episode_fraction`` (float in (0, 1] or None): further restrict the
            TRAIN pool to a fraction of its episodes (ignored for val/test).
          * ``seed`` (int, default 42).

        Returns positional indices (into the full ``_trajectory_ids`` array) to
        keep, or ``None`` when nothing restricts the dataset (caller then keeps
        every episode).

        Official ranges are interpreted as episode IDs and mapped back to array
        positions. Otherwise, a SINGLE fixed-seed ``permutation`` (salted by the
        rank-invariant dataset name) drives BOTH carves, so train and val are exact
        disjoint complements on every rank / resume.
        """
        cfg = self.data_cfg if self.data_cfg is not None else {}
        val_fraction = cfg.get("val_episode_fraction", None)
        fraction = cfg.get("episode_fraction", None)
        use_official_splits = bool(cfg.get("use_official_splits", False))
        if not use_official_splits and val_fraction is None and fraction is None:
            return None

        if fraction is not None:
            fraction = float(fraction)
            if not (0.0 < fraction <= 1.0):
                raise ValueError(f"episode_fraction must be in (0, 1], got {fraction} for {self.dataset_name}")

        split = str(cfg.get("split", "train"))
        num_episodes = len(self._trajectory_ids)
        seed = int(cfg.get("seed", 42))
        rng = np.random.default_rng(safe_hash((seed, self.dataset_name)))
        perm = rng.permutation(num_episodes)

        if use_official_splits:
            keep = self._get_official_split_positions(split)
            if split == "train" and fraction is not None:
                official_perm = rng.permutation(len(keep))
                keep_count = max(1, round(fraction * len(keep)))
                if keep_count < len(keep):
                    keep = keep[official_perm[:keep_count]]
            if val_fraction is not None and int(os.environ.get("RANK", "0")) == 0:
                print(
                    f"[official-split] {self.dataset_name}: using meta/info.json "
                    f"split={split!r}; val_episode_fraction={val_fraction!r} is ignored."
                )
            return np.sort(keep)

        # 1) Optional hold-out validation carve (fixed; independent of episode_fraction).
        n_val = 0
        if val_fraction is not None:
            val_fraction = float(val_fraction)
            if not (0.0 <= val_fraction < 1.0):
                raise ValueError(f"val_episode_fraction must be in [0, 1), got {val_fraction} for {self.dataset_name}")
            if val_fraction > 0.0 and num_episodes >= 2:
                n_val = min(max(1, round(val_fraction * num_episodes)), num_episodes - 1)
            elif val_fraction > 0.0 and int(os.environ.get("RANK", "0")) == 0:
                print(
                    f"[val-split] {self.dataset_name}: only {num_episodes} episode(s); "
                    f"cannot carve a hold-out val set — train keeps all."
                )

        val_ids = perm[:n_val]
        train_pool = perm[n_val:]

        if split == "val":
            return np.sort(val_ids)

        # split == "train": optionally sub-sample the (post-carve) train pool.
        keep = train_pool
        if fraction is not None:
            keep_count = max(1, round(fraction * len(train_pool)))
            if keep_count < len(train_pool):
                keep = train_pool[:keep_count]

        # Full-dataset no-op: no val carve AND train not reduced -> skip the slice entirely.
        if n_val == 0 and len(keep) == num_episodes:
            return None
        return np.sort(keep)

    def _apply_episode_fraction(self) -> None:
        """Restrict this dataset to the episodes chosen by ``_select_episode_indices``.

        Supports ``episode_fraction`` sub-sampling, the ``val_episode_fraction``
        hold-out, and official metadata splits. The
        excluded episodes become PHYSICALLY UNREACHABLE — removed from
        ``trajectory_ids`` / ``trajectory_lengths`` / ``all_steps``, the three
        arrays every sampling path (``sample_step`` / ``get_video`` /
        ``get_trajectory_index``) consults. ``trajectory_ids`` and
        ``trajectory_lengths`` are index-aligned, so a
        SINGLE ``keep`` mask indexes both.

        Note: normalization statistics (see ``_get_metadata``) are intentionally left at
        their full-dataset values, so runs at different fractions / splits share one
        input scale and stay comparable; only the sampled episode set is restricted.
        """
        keep = self._select_episode_indices()
        if keep is None:
            return

        num_episodes = len(self._trajectory_ids)
        self._trajectory_ids = self._trajectory_ids[keep]
        self._trajectory_lengths = self._trajectory_lengths[keep]
        kept_ids = set(self._trajectory_ids.tolist())
        self._all_steps = [(t, b) for (t, b) in self._all_steps if t in kept_ids]

        if int(os.environ.get("RANK", "0")) == 0:
            cfg = self.data_cfg if self.data_cfg is not None else {}
            split = str(cfg.get("split", "train"))
            print(
                f"[episode-select] {self.dataset_name} split={split}: keep "
                f"{len(keep)}/{num_episodes} episodes; {len(self._all_steps)} steps. "
                f"Excluded episodes are unreachable; statistics still use the full dataset."
            )

    def _required_future_offset(self) -> int:
        """Largest non-negative modality offset required by a sample.

        Negative history offsets intentionally do not affect future tail dropping;
        the existing front-padding behavior remains unchanged.
        """
        max_offset = 0
        for indices in self.delta_indices.values():
            if len(indices) > 0:
                max_offset = max(max_offset, int(np.max(indices)))
        return max_offset

    def _apply_tail_policy(self) -> None:
        """Build the per-episode sampleable prefix lengths.

        ``tail_policy=pad`` (default) samples every base index; out-of-range future
        rows/frames are then padded by the readers. ``tail_policy=drop`` removes every
        base index whose required future offset (actions and generation target) would
        leave the episode, so each sample is backed by real future data.
        """
        cfg = self.data_cfg if self.data_cfg is not None else {}
        policy = str(cfg.get("tail_policy", "pad")).strip().lower()
        if policy not in {"pad", "drop"}:
            raise ValueError(f"Invalid tail_policy={policy!r} for {self.dataset_name}; expected one of: 'pad', 'drop'.")

        if policy == "pad":
            # Mixture sampling uses the physical episode length; downstream readers
            # clamp/pad out-of-range indices.
            self._sampleable_trajectory_lengths = self._trajectory_lengths.copy()
            return

        required_future_offset = self._required_future_offset()
        sample_limits = {
            trajectory_id: max(0, int(trajectory_length) - required_future_offset)
            for trajectory_id, trajectory_length in zip(self.trajectory_ids, self.trajectory_lengths, strict=False)
        }

        old_num_steps = len(self._all_steps)
        filtered_steps: list[tuple[int, int]] = []
        next_expected_base: dict[int, int] = defaultdict(int)
        for trajectory_id, base_index in self._all_steps:
            if trajectory_id not in sample_limits:
                raise ValueError(f"Steps index for {self.dataset_name} references unknown trajectory {trajectory_id!r}.")
            base_index = int(base_index)
            if base_index < 0:
                raise ValueError(
                    f"Steps index for {self.dataset_name} contains negative base index {base_index} "
                    f"for trajectory {trajectory_id!r}."
                )
            if base_index >= sample_limits[trajectory_id]:
                continue

            expected_base = next_expected_base[trajectory_id]
            if base_index != expected_base:
                raise ValueError(
                    f"Sampleable steps for {self.dataset_name} trajectory {trajectory_id!r} must form "
                    f"a contiguous prefix starting at 0; expected {expected_base}, got {base_index}."
                )
            filtered_steps.append((trajectory_id, base_index))
            next_expected_base[trajectory_id] = expected_base + 1

        self._all_steps = filtered_steps
        self._sampleable_trajectory_lengths = np.asarray(
            [next_expected_base.get(trajectory_id, 0) for trajectory_id in self.trajectory_ids],
            dtype=np.int64,
        )
        if int(self._sampleable_trajectory_lengths.sum()) != len(self._all_steps):
            raise RuntimeError(
                f"Internal tail-drop index mismatch for {self.dataset_name}: "
                f"sum(sampleable_trajectory_lengths)={int(self._sampleable_trajectory_lengths.sum())}, "
                f"len(all_steps)={len(self._all_steps)}."
            )

        if int(os.environ.get("RANK", "0")) == 0:
            reachable_episodes = int(np.count_nonzero(self._sampleable_trajectory_lengths))
            print(
                f"[tail-policy] {self.dataset_name}: policy=drop, "
                f"required_future_offset={required_future_offset}, steps={old_num_steps}->{len(self._all_steps)}, "
                f"reachable_episodes={reachable_episodes}/{len(self.trajectory_ids)}."
            )

    def set_transforms_metadata(self, metadata: DatasetMetadata):
        """Set the metadata for the transforms. This is useful for transforms that need to know the metadata, such as the normalization values."""
        self.transforms.set_metadata(metadata)

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset.

        Args:
            epoch (int): The epoch to set.
        """
        self.epoch = epoch

    def sample_base_index(self, trajectory_index: int, rng: np.random.Generator) -> int:
        """Map one uniform sample slot to its physical frame index."""
        sampleable_length = int(self.sampleable_trajectory_lengths[trajectory_index])
        if sampleable_length <= 0:
            trajectory_id = self.trajectory_ids[trajectory_index]
            raise RuntimeError(f"Cannot sample unsampleable trajectory {trajectory_id!r} from {self.dataset_name}.")
        return int(rng.integers(sampleable_length))

    def __len__(self) -> int:
        """Get the total number of data points in the dataset.

        Returns:
            int: the total number of data points in the dataset.
        """
        return len(self.all_steps)

    def __str__(self) -> str:
        """Get the description of the dataset."""
        return f"{self.dataset_name} ({len(self)} steps)"

    def __getitem__(self, index: int) -> dict:
        """Get the data for a single step in a trajectory.

        Args:
            index (int): The index of the step to get.

        Returns:
            dict: The data for the step.
        """
        trajectory_id, base_index = self.all_steps[index]
        raw_data = self.get_step_data(trajectory_id, base_index)
        data = self.transforms(raw_data)
        return self._pack_sample(data)

    def _pack_sample(self, data: dict) -> dict:
        """Pack transformed modality data into training sample format."""
        sample = self._pack_nonvisual_sample(data)
        cfg = self.data_cfg if self.data_cfg is not None else {}
        frames_by_video_key = {}
        for video_key in self.modality_keys["video"]:
            frames_by_video_key[video_key] = Image.fromarray(data[video_key][0])
        step_images, view_texts = assemble_role_observation(
            frames_by_video_key,
            self.view_roles,
            self.view_descriptions,
            cfg,
            context=self.dataset_name,
        )
        sample["image"] = step_images
        sample["view_texts"] = view_texts
        return sample

    def _pack_nonvisual_sample(self, data: dict) -> dict:
        """Pack language, action, and optional state without an observation."""
        language = data[self.modality_keys["language"][0]][0]
        action_parts = []
        for action_key in self.modality_keys["action"]:
            action_parts.append(data[action_key])
        action = np.concatenate(action_parts, axis=1).astype(np.float16)

        sample = {
            "action": action,
            "lang": language,
            "robot_tag": self.tag,
        }

        if self.data_cfg is not None and self.data_cfg.get("include_state", False) not in ["False", False]:
            state = []
            for state_key in self.modality_keys.get("state", []):
                state.append(data[state_key])
            if not state:
                import warnings

                warnings.warn(
                    "include_state=True but no state modality keys found in modality_configs "
                    "(state modality may be disabled in the dataset's DataConfig). "
                    "Skipping state packing.",
                    stacklevel=2,
                )
            else:
                state = np.concatenate(state, axis=1).astype(np.float16)
                sample["state"] = state

        return sample

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        """Get the RAW data for a single step in a trajectory. No transforms are applied.

        Args:
            trajectory_id (int): The name of the trajectory.
            base_index (int): The base step index in the trajectory.

        Returns:
            dict: The RAW data for the step.

        Example return:
            {
                "video": {
                    "video.image_side_0": [B, T, H, W, C],
                    "video.image_side_1": [B, T, H, W, C],
                },
                "state": {
                    "state.eef_position": [B, T, state_dim],
                    "state.eef_rotation": [B, T, state_dim],
                },
                "action": {
                    "action.eef_position": [B, T, action_dim],
                    "action.eef_rotation": [B, T, action_dim],
                },
            }
        """
        data = {}
        # Load (and cache) the trajectory table once; every modality reader below uses it.
        self.curr_traj_data = self.get_trajectory_data(trajectory_id)
        for modality in self.modality_keys:
            # Get the data corresponding to each key in the modality
            for key in self.modality_keys[modality]:
                data[key] = self.get_data_by_modality(trajectory_id, modality, key, base_index)
        data = self._apply_action_mode(data)
        return data

    def get_trajectory_data(self, trajectory_id: int) -> pd.DataFrame:
        """Get the data for a trajectory."""
        if self._lerobot_version == "v2.0":

            if self.curr_traj_id == trajectory_id and self.curr_traj_data is not None:
                return self.curr_traj_data
            else:
                chunk_index = self.get_episode_chunk(trajectory_id)
                parquet_path = self.dataset_path / self.data_path_pattern.format(
                    episode_chunk=chunk_index, episode_index=trajectory_id
                )
                assert parquet_path.exists(), f"Parquet file not found at {parquet_path}"
                data = pd.read_parquet(parquet_path)
                self.curr_traj_id = trajectory_id
                self.curr_traj_data = data
                return data
        elif self._lerobot_version == "v3.0":
            return self.get_trajectory_data_lerobot_v3(trajectory_id)

    def get_trajectory_data_lerobot_v3(self, trajectory_id: int) -> pd.DataFrame:
        """Get the data for a trajectory from lerobot v3."""
        if self.curr_traj_id == trajectory_id and self.curr_traj_data is not None:
            return self.curr_traj_data
        else:
            episode_meta = self.trajectory_ids_to_metadata[trajectory_id]
            chunk_index = episode_meta["data/chunk_index"]
            file_index = self.get_episode_file_index(trajectory_id)

            parquet_path = self.dataset_path / self.data_path_pattern.format(
                chunk_index=chunk_index, file_index=file_index
            )
            assert parquet_path.exists(), f"Parquet file not found at {parquet_path}"
            file_data = pd.read_parquet(parquet_path)

            episode_data = file_data.loc[file_data["episode_index"] == trajectory_id].copy()
            del file_data
            self.curr_traj_id = trajectory_id
            self.curr_traj_data = episode_data
            return episode_data

    def get_trajectory_index(self, trajectory_id: int) -> int:
        """Get the index of the trajectory in the dataset by the trajectory ID.
        This is useful when you need to get the trajectory length or sampling weight corresponding to the trajectory ID.

        Args:
            trajectory_id (str): The ID of the trajectory.

        Returns:
            int: The index of the trajectory in the dataset.
        """
        trajectory_indices = np.where(self.trajectory_ids == trajectory_id)[0]
        if len(trajectory_indices) != 1:
            raise ValueError(f"Error finding trajectory index for {trajectory_id}, found {trajectory_indices=}")
        return trajectory_indices[0]

    def get_episode_chunk(self, ep_index: int) -> int:
        """Get the chunk index for an episode index."""
        return ep_index // self.chunk_size

    def get_episode_file_index(self, ep_index: int) -> int:
        """Get the file index for an episode index."""
        episode_meta = self.trajectory_ids_to_metadata[ep_index]
        return episode_meta["data/file_index"]

    def get_episode_file_from_index(self, ep_index: int) -> int:
        """Get the file from index for an episode index."""
        episode_meta = self.trajectory_ids_to_metadata[ep_index]
        return episode_meta["data/file_from_index"]

    def retrieve_data_and_pad(
        self,
        array: np.ndarray,
        step_indices: np.ndarray,
        max_length: int,
        padding_strategy: str = "first_last",
    ) -> np.ndarray:
        """Retrieve the data from the dataset and pad it if necessary.
        Args:
            array (np.ndarray): The array to retrieve the data from.
            step_indices (np.ndarray): The step indices to retrieve the data for.
            max_length (int): The maximum length of the data.
            padding_strategy (str): The padding strategy, either "first" or "last".
        """
        # Get the padding indices
        front_padding_indices = step_indices < 0
        end_padding_indices = step_indices >= max_length
        padding_positions = np.logical_or(front_padding_indices, end_padding_indices)
        # Retrieve the data with the non-padding indices
        # If there exists some padding, Given T step_indices, the shape of the retrieved data will be (T', ...) where T' < T
        raw_data = array[step_indices[~padding_positions]]
        assert isinstance(raw_data, np.ndarray), f"{type(raw_data)=}"
        # This is the shape of the output, (T, ...)
        if raw_data.ndim == 1:
            expected_shape = (len(step_indices),)
        else:
            expected_shape = (len(step_indices), *array.shape[1:])

        # Pad the data
        output = np.zeros(expected_shape)
        # Assign the non-padded data
        output[~padding_positions] = raw_data
        # If there exists some padding, pad the data
        if padding_positions.any():
            if padding_strategy == "first_last":
                # Use first / last step data to pad
                front_padding_data = array[0]
                end_padding_data = array[-1]
                output[front_padding_indices] = front_padding_data
                output[end_padding_indices] = end_padding_data
            elif padding_strategy == "zero":
                # Use zero padding
                output[padding_positions] = 0
            else:
                raise ValueError(f"Invalid padding strategy: {padding_strategy}")
        return output

    def get_video_path(self, trajectory_id: int, key: str) -> Path:
        chunk_index = self.get_episode_chunk(trajectory_id)
        original_key = self._resolve_video_original_key(key)
        if self._lerobot_version == "v2.0":
            video_filename = self.video_path_pattern.format(
                episode_chunk=chunk_index, episode_index=trajectory_id, video_key=original_key
            )
        elif self._lerobot_version == "v3.0":
            episode_meta = self.trajectory_ids_to_metadata[trajectory_id]
            schema = getattr(self, "_v3_video_index_schema", None)
            if schema == "modern":
                video_file_indices = episode_meta.get("videos/file_indices", {})
                if original_key not in video_file_indices:
                    raise ValueError(
                        f"LeRobot v3 per-camera video index missing after initialization validation: "
                        f"episode={trajectory_id}, camera={original_key}, "
                        f"data_shard=({episode_meta['data/chunk_index']}, {episode_meta['data/file_index']}), "
                        f"available_cameras={sorted(video_file_indices)}."
                    )
                video_chunk_index = video_file_indices[original_key]["chunk_index"]
                video_file_index = video_file_indices[original_key]["file_index"]
            elif schema == "legacy":
                video_chunk_index = episode_meta["data/chunk_index"]
                video_file_index = episode_meta["data/file_index"]
            else:
                raise ValueError(
                    f"LeRobot v3 video-index schema is unresolved for dataset={self.dataset_path}, "
                    f"episode={trajectory_id}, camera={original_key}; refusing an implicit data-shard fallback."
                )
            video_filename = self.video_path_pattern.format(
                video_key=original_key,
                chunk_index=video_chunk_index,
                file_index=video_file_index,
            )
        return self._video_overrides.get(video_filename, self.dataset_path / video_filename)

    def get_video(
        self,
        trajectory_id: int,
        key: str,
        base_index: int,
    ) -> np.ndarray:
        """Get the video frames for a trajectory by a base index.

        Thin wrapper: resolves ``self.delta_indices[key] + base_index`` and defers to
        :meth:`get_video_at_indices`. Kept for the standard modality read path.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (str): The ID of the trajectory.
            key (str): The key of the video.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The video frames for the trajectory and frame indices. Shape: (T, H, W, C)
        """
        step_indices = self.delta_indices[key] + base_index
        return self.get_video_at_indices(trajectory_id, key, step_indices)

    def get_video_at_indices(
        self,
        trajectory_id: int,
        key: str,
        step_indices,
    ) -> np.ndarray:
        """Decode video frames at EXPLICIT (absolute) trajectory frame indices in ONE pass.

        ``step_indices`` is a list / np.ndarray of absolute frame indices. Out-of-range
        values are clamped to ``[0, traj_len - 1]`` (edge-repeat padding — identical to
        :meth:`get_video`). Returns ``(len(step_indices), H, W, C)`` uint8 RGB.

        This lets a caller fetch a whole ``[current, *future]`` window in a single decode
        so that observation and generation-target frames stay strictly same-source (same
        decode + same downstream transform), rather than decoding obs and future separately.
        """
        step_indices = np.asarray(step_indices)
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Ensure the indices are within the valid range
        # This is equivalent to padding the video with extra frames at the beginning and end
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, self.trajectory_lengths[trajectory_index] - 1)
        assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
        # Get the sub-key
        key = key.replace("video.", "")

        # Image-only LeRobot datasets (e.g. VLA-Arena) may store frames directly
        # in parquet columns and have total_videos == 0 (no mp4 files). In this
        # case, load frames from the original image column and pad by step indices.
        original_key = self.lerobot_modality_meta.video[key].original_key
        if original_key is None:
            original_key = key
        if self.curr_traj_data is not None and original_key in self.curr_traj_data.columns:
            image_entries = self.curr_traj_data[original_key].tolist()

            def _decode_image_entry(entry):
                if isinstance(entry, np.ndarray):
                    return entry
                if isinstance(entry, Image.Image):
                    return np.array(entry)
                if isinstance(entry, dict):
                    img_bytes = entry.get("bytes", None)
                    img_path = entry.get("path", None)

                    if img_bytes is not None:
                        return np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))

                    if img_path is not None:
                        path_obj = Path(img_path)
                        if not path_obj.is_absolute():
                            path_obj = self.dataset_path / path_obj
                        return np.array(Image.open(path_obj).convert("RGB"))

                raise TypeError(f"Unsupported image entry type: {type(entry)}")

            frames = []
            for idx in step_indices:
                safe_idx = int(min(max(idx, 0), len(image_entries) - 1))
                frames.append(_decode_image_entry(image_entries[safe_idx]))

            return np.stack(frames)

        video_path = self.get_video_path(trajectory_id, key)
        # Get the action/state timestamps for each frame in the video
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert "timestamp" in self.curr_traj_data.columns, f"No timestamp found in {trajectory_id=}"
        timestamp: np.ndarray = self.curr_traj_data["timestamp"].to_numpy()
        # Get the corresponding video timestamps from the step indices
        video_timestamp = timestamp[step_indices]
        if self._lerobot_version == "v3.0":
            episode_meta = self.trajectory_ids_to_metadata.get(trajectory_id, {})
            from_timestamps = episode_meta.get("videos/from_timestamps", {})
            original_video_key = self._resolve_video_original_key(key)
            if self._v3_video_index_schema == "modern":
                if original_video_key not in from_timestamps:
                    video_indices = episode_meta.get("videos/file_indices", {}).get(
                        original_video_key,
                        {},
                    )
                    raise ValueError(
                        "LeRobot v3 per-camera timestamp missing after initialization "
                        f"validation: episode={trajectory_id}, camera={original_video_key}, "
                        f"video_shard=({video_indices.get('chunk_index')}, "
                        f"{video_indices.get('file_index')}), "
                        f"data_shard=({episode_meta.get('data/chunk_index')}, "
                        f"{episode_meta.get('data/file_index')})."
                    )
                from_timestamp = self._parse_v3_video_timestamp(
                    from_timestamps[original_video_key],
                    f"videos/{original_video_key}/from_timestamp",
                    f"episode={trajectory_id}",
                )
            else:
                # The legacy v3 metadata schema may omit per-camera offsets; it uses an
                # episode-local zero offset.
                from_timestamp = float(from_timestamps.get(original_video_key, 0.0))
            video_timestamp = video_timestamp + from_timestamp

        return get_frames_by_timestamps(
            video_path.as_posix(),
            video_timestamp,
            video_backend=self.video_backend,
            video_backend_kwargs=self.video_backend_kwargs,
        )

    def get_state_or_action(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        base_index: int,
    ) -> np.ndarray:
        """Get the state or action data for a trajectory by a base index.
        If the step indices are out of range, pad with the data:
            if the data is stored in absolute format, pad with the first or last step data;
            otherwise, pad with zero.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            modality (str): The modality of the data.
            key (str): The key of the data.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The data for the trajectory and step indices.
        """
        # Get the step indices
        step_indices = self.delta_indices[key] + base_index
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]
        assert key.startswith(modality + "."), f"{key} must start with {modality + '.'}, got {key}"
        # Get the sub-key, e.g. state.joint_angles -> joint_angles
        key = key.replace(modality + ".", "")
        # Get the lerobot key
        le_state_or_action_cfg = getattr(self.lerobot_modality_meta, modality)
        le_key = le_state_or_action_cfg[key].original_key
        if le_key is None:
            le_key = key
        # Get the data array, shape: (T, D)
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert le_key in self.curr_traj_data.columns, f"No {le_key} found in {trajectory_id=}"
        data_array: np.ndarray = np.stack(self.curr_traj_data[le_key])  # type: ignore
        if data_array.ndim == 1:
            # A LeRobot shape-[1] feature may be stored as a SCALAR per row (e.g. DROID's
            # gripper_position) -> np.stack gives (T,). Reshape to (T, 1) so the [start:end]
            # slice and downstream concat see a consistent 2-D array. No-op for the usual
            # array-valued columns (already 2-D).
            data_array = data_array[:, None]
        assert data_array.ndim == 2, f"Expected 2D array, got key {le_key} is{data_array.shape} array"
        le_indices = np.arange(
            le_state_or_action_cfg[key].start,
            le_state_or_action_cfg[key].end,
        )
        data_array = data_array[:, le_indices]
        # Get the state or action configuration
        state_or_action_cfg = getattr(self.metadata.modalities, modality)[key]

        # Pad the data
        return self.retrieve_data_and_pad(
            array=data_array,
            step_indices=step_indices,
            max_length=max_length,
            padding_strategy="first_last" if state_or_action_cfg.absolute else "zero",
        )

    def get_language(
        self,
        trajectory_id: int,
        key: str,
        base_index: int,
    ) -> list[str]:
        """Get the language annotation data for a trajectory by step indices.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            key (str): The key of the annotation.
            base_index (int): The base index of the trajectory.

        Returns:
            list[str]: The annotation data for the trajectory and step indices. If no matching data is found, return empty strings.
        """
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        # Get the step indices
        step_indices = self.delta_indices[key] + base_index
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]
        # Get the end times corresponding to the closest indices
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, max_length - 1)
        # Get the annotations
        assert key.startswith("annotation."), f"Language key must start with 'annotation.', got {key}"
        subkey = key.replace("annotation.", "")
        annotation_meta = self.lerobot_modality_meta.annotation
        assert annotation_meta is not None, f"Annotation metadata is None for {subkey}"
        assert (
            subkey in annotation_meta
        ), f"Annotation key {subkey} not found in metadata, available annotation keys: {annotation_meta.keys()}"
        subkey_meta = annotation_meta[subkey]
        original_key = subkey_meta.original_key
        if original_key is None:
            original_key = key
        # Two supported annotation layouts, detected per value by dtype so datasets
        # using either one work unchanged:
        #   (1) integer `task_index` column -> look up the string in tasks.jsonl
        #       (self.tasks); the standard gr00t layout.
        #   (2) inline string column -> the annotation text is stored directly in
        #       the column (e.g. LeRobot datasets that keep the task string inline).
        # Resolve each step in a single pass (rather than splitting into two lists and
        # returning only one), so the result always aligns 1:1 with step_indices in
        # order even if a column ever mixed the two layouts.
        texts: list[str] = []
        for i in range(len(step_indices)):
            value = self.curr_traj_data[original_key].iloc[step_indices[i]]
            # numpy/pandas scalar -> python scalar (numpy.str_ -> str, int64 -> int)
            if hasattr(value, "item"):
                value = value.item()
            if isinstance(value, str):
                texts.append(value)  # inline string: annotation text stored directly
            else:
                texts.append(self.tasks.loc[int(value)]["task"])  # integer task_index -> tasks.jsonl
        return texts

    def get_data_by_modality(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        base_index: int,
    ):
        """Get the data corresponding to the modality for a trajectory by a base index.
        This method will call the corresponding helper method based on the modality.
        See the helper methods for more details.
        NOTE: For the language modality, the data is padded with empty strings if no matching data is found.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            modality (str): The modality of the data.
            key (str): The key of the data.
            base_index (int): The base index of the trajectory.
        """
        if modality == "video":
            return self.get_video(trajectory_id, key, base_index)
        elif modality == "state" or modality == "action":
            return self.get_state_or_action(trajectory_id, modality, key, base_index)
        elif modality == "language":
            return self.get_language(trajectory_id, key, base_index)
        else:
            raise ValueError(f"Invalid modality: {modality}")

    def _save_dataset_statistics_(self, save_path: Path | str, format: str = "json") -> None:
        """
        Save dataset statistics to specified path in the required format.
        Only includes statistics for keys that are actually used in the dataset.
        Gripper-related keys will be placed at the end.

        Args:
            save_path (Path | str): Path to save the statistics file
            format (str): Save format, currently only supports "json"
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Build the data structure to save
        statistics_data = {}

        # Get used modality keys
        used_action_keys, used_state_keys = get_used_modality_keys(self.modality_keys)

        # Organize statistics by tag
        tag = self.tag
        tag_stats = {}

        # Process action statistics (only for used keys)
        if hasattr(self.metadata.statistics, "action") and self.metadata.statistics.action:
            action_stats = self.metadata.statistics.action

            # Filter to only include used action keys and reorder: non-gripper first, gripper last
            non_gripper_keys = []
            gripper_keys = []

            for key in action_stats.keys():
                if key in used_action_keys:
                    if "gripper" in key.lower():
                        gripper_keys.append(key)
                    else:
                        non_gripper_keys.append(key)

            # Reorder: non-gripper first, gripper last
            reordered_keys = non_gripper_keys + gripper_keys

            filtered_action_stats = {}
            for key in reordered_keys:
                filtered_action_stats[key] = action_stats[key]

            if filtered_action_stats:
                # Combine statistics from filtered action sub-keys
                combined_action_stats = combine_modality_stats(filtered_action_stats)

                # mask=False for dimensions whose normalization mode is "binary"
                _action_norm_modes = _extract_action_normalization_modes(self.transforms)
                mask = generate_action_mask_for_used_keys(
                    self.metadata.modalities.action,
                    filtered_action_stats.keys(),
                    normalization_modes=_action_norm_modes,
                )
                combined_action_stats["mask"] = mask

                tag_stats["action"] = combined_action_stats

        # Process state statistics (only for used keys)
        if hasattr(self.metadata.statistics, "state") and self.metadata.statistics.state:
            state_stats = self.metadata.statistics.state

            # Filter to only include used state keys, optionally reorder gripper to end
            non_gripper_keys = []
            gripper_keys = []

            for key in state_stats.keys():
                if key in used_state_keys:
                    if "gripper" in key.lower():
                        gripper_keys.append(key)
                    else:
                        non_gripper_keys.append(key)

            # Reorder: non-gripper first, gripper last
            reordered_keys = non_gripper_keys + gripper_keys

            filtered_state_stats = {}
            for key in reordered_keys:
                filtered_state_stats[key] = state_stats[key]

            if filtered_state_stats:
                combined_state_stats = combine_modality_stats(filtered_state_stats)
                tag_stats["state"] = combined_state_stats

        # Add dataset counts
        tag_stats["num_transitions"] = len(self)
        tag_stats["num_trajectories"] = len(self.trajectory_ids)

        statistics_data[tag] = tag_stats

        # Save as JSON file
        if format.lower() == "json":
            if not str(save_path).endswith(".json"):
                save_path = save_path.with_suffix(".json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(statistics_data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}. Currently only 'json' is supported.")

        print(f"Single dataset statistics saved to: {save_path}")
        print(f"Used action keys (reordered): {list(used_action_keys)}")
        print(f"Used state keys (reordered): {list(used_state_keys)}")


class CachedLeRobotSingleDataset(LeRobotSingleDataset):
    def __init__(self, img_resize: tuple[int, int] | None = None, *args, **kwargs):
        """
        This class caches the video frames for each trajectory and key.
        It is recommended to use this class if the video frames need to be accessed multiple times.

        Args:
            resize_img (tuple[int, int], optional): The size to resize the video frames to reduce memory usage.
        """
        # Convert img_resize to tuple if it is not already
        if img_resize is not None and not isinstance(img_resize, tuple):
            img_resize = tuple(img_resize)
            assert len(img_resize) == 2, f"Expected tuple of length 2, got {img_resize}"
        self.img_resize = img_resize

        # Initialize img_resize attribute first to ensure it exists
        super().__init__(*args, **kwargs)
        cached_frames: dict[str, np.ndarray] = {}

        for key in self.modality_keys["video"]:
            all_frames = []
            key = key.replace("video.", "")
            for trajectory_id in tqdm(
                self.trajectory_ids,
                total=len(self.trajectory_ids),
                desc=f"Caching {key} frames",
            ):
                video_path = self.get_video_path(trajectory_id, key)
                frames = get_all_frames(
                    video_path.as_posix(),
                    video_backend=self.video_backend,
                    video_backend_kwargs=self.video_backend_kwargs,
                    resize_size=img_resize,
                )
                assert frames.ndim == 4, f"Expected 4D array, got {frames.shape} array"
                assert frames.shape[3] == 3, f"Expected 3 channels, got {frames.shape[3]} channels"
                all_frames.append(frames)
            cached_frames[key] = np.concatenate(all_frames, axis=0)
            print(f"{key}: {cached_frames[key].shape}")
        self.cached_frames = cached_frames
        self.start_indices = np.cumsum(self.trajectory_lengths) - self.trajectory_lengths

    def get_video(self, trajectory_id: int, key: str, base_index: int) -> np.ndarray:
        step_indices = self.delta_indices[key] + base_index
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Ensure the indices are within the valid range
        # This is equivalent to padding the video with extra frames at the beginning and end
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, self.trajectory_lengths[trajectory_index] - 1)
        assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
        # Get the sub-key
        key = key.replace("video.", "")
        # Calculate the absolute indices
        absolute_indices = self.start_indices[trajectory_index] + step_indices
        return self.cached_frames[key][absolute_indices]

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        """Get the RAW data for a single step. No transforms are applied.

        Args:
            trajectory_id (str): The ID of the trajectory.
            base_index (int): The base index of the step.

        Returns:
            dict: The data for the step.
        """
        data = {}
        self.curr_traj_data = self.get_trajectory_data(trajectory_id)
        # Get the data for all modalities
        for modality in self.modality_keys:
            # Get the data corresponding to each key in the modality
            for key in self.modality_keys[modality]:
                data[key] = self.get_data_by_modality(trajectory_id, modality, key, base_index)
        return data

    def set_transforms_metadata(self, metadata: DatasetMetadata):
        """Set the metadata for the transforms. This is useful for transforms that need to know the metadata, such as the normalization values."""
        if self.img_resize is not None:
            all_video_keys = [key for key in self.modality_keys["video"]]
            for key in metadata.modalities.video:
                if key in all_video_keys:
                    metadata.modalities.video[key].resolution = self.img_resize
        super().set_transforms_metadata(metadata)


def safe_hash(input_tuple):
    # keep 128 bits of the hash
    tuple_string = repr(input_tuple).encode("utf-8")
    sha256 = hashlib.sha256()
    sha256.update(tuple_string)

    seed = int(sha256.hexdigest(), 16)

    return seed & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF


WEIGHTED_WITH_REPLACEMENT = "weighted_with_replacement"
TRANSITION_PERMUTATION = "transition_permutation"
TRANSITION_PERMUTATION_VERSION = 1


def normalize_sampling_strategy(value) -> str:
    """Normalize the mixture sampling strategy without silently changing semantics."""

    strategy = WEIGHTED_WITH_REPLACEMENT if value is None else str(value).strip().lower()
    supported = {WEIGHTED_WITH_REPLACEMENT, TRANSITION_PERMUTATION}
    if strategy not in supported:
        raise ValueError(f"Unsupported sampling_strategy={value!r}; expected one of {sorted(supported)}.")
    return strategy


class MixtureSpecElement(BaseModel):
    dataset_path: list[Path] | Path = Field(..., description="The path to the dataset.")
    dataset_weight: float = Field(..., description="The weight of the dataset in the mixture.")
    distribute_weights: bool = Field(
        default=False,
        description="Whether to distribute the weights of the dataset across all the paths. If True, the weights will be evenly distributed across all the paths.",
    )


# Helper functions for dataset statistics


def combine_modality_stats(modality_stats: dict) -> dict:
    """
    Combine statistics from all sub-keys under a modality.

    Args:
        modality_stats (dict): Statistics for a modality, containing multiple sub-keys.
                               Each sub-key contains DatasetStatisticalValues object.

    Returns:
        dict: Combined statistics
    """
    combined_stats = {"mean": [], "std": [], "max": [], "min": [], "q01": [], "q99": []}

    # Combine statistics in sub-key order
    for subkey in modality_stats.keys():
        subkey_stats = modality_stats[subkey]  # This is a DatasetStatisticalValues object

        # Convert DatasetStatisticalValues to dict-like access
        for stat_name in ["mean", "std", "max", "min", "q01", "q99"]:
            stat_value = getattr(subkey_stats, stat_name)
            if isinstance(stat_value, (list, tuple)):
                combined_stats[stat_name].extend(stat_value)
            else:
                # Handle NDArray case - convert to list
                if hasattr(stat_value, "tolist"):
                    combined_stats[stat_name].extend(stat_value.tolist())
                else:
                    combined_stats[stat_name].append(float(stat_value))

    return combined_stats


def _extract_action_normalization_modes(transforms) -> dict:
    """Extract normalization modes for action keys from a ComposedModalityTransform.

    Returns:
        dict: {subkey_without_action_prefix -> normalization_mode_str}
    """
    modes = {}
    for t in transforms.transforms:
        if isinstance(t, StateActionTransform):
            for key, mode in t.normalization_modes.items():
                if key.startswith("action."):
                    subkey = key[len("action.") :]
                    modes[subkey] = mode
    return modes


def generate_action_mask_for_used_keys(
    action_modalities: dict,
    used_action_keys_ordered,
    normalization_modes: dict | None = None,
) -> list[bool]:
    """Generate per-dimension mask for action statistics.

    A dimension gets ``mask=False`` only when its normalization mode is ``"binary"``.
    This tells the inference code to skip continuous de-normalization for that dimension.
    All other modes (q99, mean_std, min_max ...) produce ``mask=True``.

    Args:
        action_modalities (dict): Configuration information for action modalities.
        used_action_keys_ordered: Iterable of actually used action keys (no "action." prefix).
        normalization_modes (dict | None): Mapping {subkey -> mode} (no "action." prefix).
            If ``None``, all dimensions default to ``mask=True``.

    Returns:
        list[bool]: Per-dimension mask values.
    """
    mask = []

    for subkey in used_action_keys_ordered:
        if subkey in action_modalities:
            subkey_config = action_modalities[subkey]

            # Get dimension count from shape
            if hasattr(subkey_config, "shape") and len(subkey_config.shape) > 0:
                dim_count = subkey_config.shape[0]
            else:
                dim_count = 1

            # mask=False only when the normalization mode is explicitly "binary"
            is_binary = normalization_modes.get(subkey) == "binary" if normalization_modes is not None else False

            for _ in range(dim_count):
                mask.append(not is_binary)

    return mask


def get_used_modality_keys(modality_keys: dict) -> tuple[list, list]:
    """Extract used action and state keys from modality configuration."""
    used_action_keys = []
    used_state_keys = []

    # Extract action keys (remove "action." prefix)
    for action_key in modality_keys.get("action", []):
        if action_key.startswith("action."):
            clean_key = action_key.replace("action.", "")
            used_action_keys.append(clean_key)

    # Extract state keys (remove "state." prefix)
    for state_key in modality_keys.get("state", []):
        if state_key.startswith("state."):
            clean_key = state_key.replace("state.", "")
            used_state_keys.append(clean_key)

    return used_action_keys, used_state_keys


class LeRobotMixtureDataset(Dataset):
    """
    A mixture of multiple datasets. This class samples a single dataset based on the dataset weights and then calls the `__getitem__` method of the sampled dataset.
    It is recommended to modify the single dataset class instead of this class.
    """

    def __init__(
        self,
        data_mixture: Sequence[tuple[LeRobotSingleDataset, float]],
        mode: str,
        balance_dataset_weights: bool = True,
        balance_trajectory_weights: bool = True,
        seed: int = 42,
        sampling_epoch_multiple: int = 1,
        metadata_config: dict | None = None,
        **kwargs,
    ):
        """
        Initialize the mixture dataset.

        Args:
            data_mixture (list[tuple[LeRobotSingleDataset, float]]): Datasets and their corresponding weights.
            mode (str): If "train", __getitem__ will return different samples every epoch; if "val" or "test", __getitem__ will return the same sample every epoch.
            balance_dataset_weights (bool): If True, the weight of dataset will be multiplied by the total trajectory length of each dataset.
            balance_trajectory_weights (bool): If True, sample trajectories within a dataset weighted by their length; otherwise, use equal weighting.
            seed (int): Random seed for sampling.
            sampling_epoch_multiple (int): Train epoch length alignment. Permutation
                sampling uses this to avoid distributed tail padding duplicates.
            metadata_config (dict): Statistics-merging options; defaults to
                ``{"percentile_mixing_method": "min_max"}``.
        """
        if metadata_config is None:
            metadata_config = {"percentile_mixing_method": "min_max"}
        data_mixture = list(data_mixture)
        self.data_cfg = kwargs["data_cfg"] if "data_cfg" in kwargs else None
        configured_sampling_strategy = (
            self.data_cfg.get("sampling_strategy", WEIGHTED_WITH_REPLACEMENT)
            if self.data_cfg is not None
            else WEIGHTED_WITH_REPLACEMENT
        )
        self.sampling_strategy = normalize_sampling_strategy(configured_sampling_strategy)
        # A transition-permutation epoch owns a large, immutable order array.
        # Persistent worker copies cannot see the replacement array built in the
        # trainer process, so this rare boundary still respawns the pool to inherit
        # the exact canonical order. Weighted sampling needs only SharedEpoch and
        # keeps its workers throughout.
        self.persistent_epoch_requires_respawn = self.sampling_strategy == TRANSITION_PERMUTATION
        if (
            isinstance(sampling_epoch_multiple, bool)
            or not isinstance(sampling_epoch_multiple, (int, np.integer))
            or int(sampling_epoch_multiple) <= 0
        ):
            raise ValueError("sampling_epoch_multiple must be a positive integer, got " f"{sampling_epoch_multiple!r}.")
        self.sampling_epoch_multiple = int(sampling_epoch_multiple)

        datasets: list[LeRobotSingleDataset] = []
        dataset_sampling_weights: list[float] = []
        declared_leaf_names = [dataset.dataset_name for dataset, _ in data_mixture]
        for dataset, weight in data_mixture:
            # Check if dataset is valid and has data
            if len(dataset) == 0:
                if self.sampling_strategy == TRANSITION_PERMUTATION:
                    raise ValueError(
                        "transition_permutation refuses to skip an empty mixture leaf: "
                        f"dataset={dataset.dataset_name!r}."
                    )
                print(f"Warning: Skipping empty dataset {dataset.dataset_name}")
                continue
            datasets.append(dataset)
            dataset_sampling_weights.append(weight)

        if len(datasets) == 0:
            raise ValueError("No valid datasets found in the mixture. All datasets are empty.")

        self.datasets = datasets
        self.balance_dataset_weights = balance_dataset_weights
        self.balance_trajectory_weights = balance_trajectory_weights
        self.seed = seed
        self.mode = mode
        self._shared_epoch = SharedEpoch(0)
        self._persistent_worker_epoch = bool(
            self.data_cfg.get("persistent_workers", True) if self.data_cfg is not None else True
        )
        configured_error_policy = (
            self.data_cfg.get("sample_error_policy", "resample") if self.data_cfg is not None else "resample"
        )
        self.sample_error_policy = normalize_sample_error_policy(configured_error_policy)
        configured_manifests = self.data_cfg.get("video_override_manifests", None) if self.data_cfg is not None else None
        self._video_override_manifest_contract = validate_manifest_leaf_contract(
            configured_manifests,
            declared_leaf_names,
        )
        active_override = bool(
            self._video_override_manifest_contract
            and any(value is not None for value in self._video_override_manifest_contract.values())
        )
        if active_override and self.sample_error_policy != "raise":
            raise VideoOverrideError(
                "An active video override requires sample_error_policy='raise'; "
                f"got {self.sample_error_policy!r}. Silent replacement sampling would "
                "invalidate the repaired dataset provenance."
            )
        if self.sample_error_policy == "raise":
            self._preflight_required_video_files()

        provenance_payload = {
            "contract": "unconfigured" if self._video_override_manifest_contract is None else "explicit-exact-leaf-map",
            "sample_error_policy": self.sample_error_policy,
            "leaves": [
                {
                    "dataset_name": dataset.dataset_name,
                    "dataset_path": str(dataset.dataset_path),
                    "override": dataset.video_override_provenance,
                }
                for dataset in sorted(self.datasets, key=lambda item: item.dataset_name)
            ],
        }
        self._video_override_provenance = {
            **provenance_payload,
            "sha256": canonical_provenance_digest(provenance_payload),
        }

        # Set properties for sampling

        # 1. Dataset lengths
        self._dataset_lengths = np.array([len(dataset) for dataset in self.datasets])
        print(f"Dataset lengths: {self._dataset_lengths}")
        self._getitem_count = 0
        # 2. Dataset sampling weights
        self._declared_dataset_sampling_weights = np.asarray(dataset_sampling_weights, dtype=np.float64)
        self._validate_transition_permutation_contract()
        self._dataset_sampling_weights = self._declared_dataset_sampling_weights.copy()

        if self.balance_dataset_weights:
            self._dataset_sampling_weights *= self._dataset_lengths

        # Check for zero or negative weights before normalization
        if np.any(self._dataset_sampling_weights <= 0):
            print(f"Warning: Found zero or negative sampling weights: {self._dataset_sampling_weights}")
            # Set minimum weight to prevent division issues
            self._dataset_sampling_weights = np.maximum(self._dataset_sampling_weights, 1e-8)

        # Normalize weights
        weights_sum = self._dataset_sampling_weights.sum()
        if weights_sum == 0 or np.isnan(weights_sum):
            print(f"Error: Invalid weights sum: {weights_sum}")
            # Fallback to equal weights
            self._dataset_sampling_weights = np.ones(len(self.datasets)) / len(self.datasets)
            print("Fallback to equal weights")
        else:
            self._dataset_sampling_weights /= weights_sum

        # 3. Trajectory sampling weights
        self._trajectory_sampling_weights: list[np.ndarray] = []
        for i, dataset in enumerate(self.datasets):
            sampleable_lengths = np.asarray(dataset.sampleable_trajectory_lengths, dtype=np.float64)
            if np.any(sampleable_lengths < 0) or np.any(~np.isfinite(sampleable_lengths)):
                raise ValueError(f"Dataset {i} has invalid sampleable trajectory lengths: {sampleable_lengths}")

            if self.balance_trajectory_weights:
                trajectory_sampling_weights = sampleable_lengths.copy()
            else:
                # Equal episode weighting, but structurally unsampleable episodes must
                # retain exactly zero probability instead of being revived by epsilon.
                trajectory_sampling_weights = (sampleable_lengths > 0).astype(np.float64)

            weights_sum = trajectory_sampling_weights.sum()
            if weights_sum <= 0 or not np.isfinite(weights_sum):
                raise ValueError(
                    f"Dataset {i} ({dataset.dataset_name}) has no sampleable trajectories; "
                    f"sampleable lengths={sampleable_lengths}."
                )
            trajectory_sampling_weights /= weights_sum
            self._trajectory_sampling_weights.append(trajectory_sampling_weights)

        # 4. Primary dataset indices
        self._primary_dataset_indices = np.array(dataset_sampling_weights) == 1.0
        if not np.any(self._primary_dataset_indices):
            print(f"Warning: No dataset with weight 1.0 found. Original weights: {dataset_sampling_weights}")
            # Fallback: use the dataset(s) with maximum weight as primary
            max_weight = max(dataset_sampling_weights)
            self._primary_dataset_indices = np.array(dataset_sampling_weights) == max_weight
            print(f"Using datasets with maximum weight {max_weight} as primary: {self._primary_dataset_indices}")

        if not np.any(self._primary_dataset_indices):
            # This should never happen, but just in case
            print("Error: Still no primary dataset found. Using first dataset as primary.")
            self._primary_dataset_indices = np.zeros(len(self.datasets), dtype=bool)
            self._primary_dataset_indices[0] = True

        self._initialize_transition_permutation()

        # Set the epoch and sample the first epoch
        self.set_epoch(0)

        self.update_metadata(metadata_config)

    @property
    def dataset_lengths(self) -> np.ndarray:
        """The lengths of each dataset."""
        return self._dataset_lengths

    @property
    def dataset_sampling_weights(self) -> np.ndarray:
        """The sampling weights for each dataset."""
        return self._dataset_sampling_weights

    @property
    def trajectory_sampling_weights(self) -> list[np.ndarray]:
        """The sampling weights for each trajectory in each dataset."""
        return self._trajectory_sampling_weights

    @property
    def primary_dataset_indices(self) -> np.ndarray:
        """The indices of the primary datasets."""
        return self._primary_dataset_indices

    @property
    def video_override_provenance(self) -> dict:
        """Canonical mixture-level video provenance for run/checkpoint records."""
        return copy.deepcopy(self._video_override_provenance)

    @property
    def sampling_provenance(self) -> dict:
        """Return the versioned index-to-transition contract for audit/resume."""

        payload = {
            "strategy": self.sampling_strategy,
            "seed": int(self.seed),
        }
        if self.sampling_strategy == TRANSITION_PERMUTATION:
            payload.update(
                {
                    "version": TRANSITION_PERMUTATION_VERSION,
                    "population_size": int(self._transition_population_size),
                    "epoch_size": int(self._transition_epoch_size),
                    "sampling_epoch_multiple": int(self.sampling_epoch_multiple),
                    "dropped_tail_per_epoch": int(self._transition_dropped_tail),
                    "tail_schedule": "fixed_random_order_rotating_windows",
                    "guaranteed_disjoint_tail_epochs": (
                        int(self._transition_population_size // self._transition_dropped_tail)
                        if self._transition_dropped_tail
                        else None
                    ),
                    "padding_duplicates": 0,
                    "leaf_counts": [
                        {
                            "dataset_name": str(dataset.dataset_name),
                            "num_transitions": int(length),
                        }
                        for dataset, length in zip(self.datasets, self._dataset_lengths, strict=True)
                    ],
                }
            )
        return payload

    def _validate_transition_permutation_contract(self) -> None:
        """Refuse configurations whose weighting cannot mean one visit per transition."""

        if self.sampling_strategy != TRANSITION_PERMUTATION:
            return
        reasons = []
        if self.sample_error_policy != "raise":
            reasons.append("sample_error_policy must be 'raise' so a failed item cannot be replaced")
        if not self.balance_dataset_weights:
            reasons.append("balance_dataset_weights must be true to preserve the natural transition distribution")
        if not self.balance_trajectory_weights:
            reasons.append("balance_trajectory_weights must be true to preserve the natural transition distribution")
        weights = self._declared_dataset_sampling_weights
        if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
            reasons.append(f"all declared mixture weights must be finite and positive, got {weights.tolist()}")
        elif not np.allclose(weights, weights[0], rtol=0.0, atol=0.0):
            reasons.append(
                "all declared mixture weights must be identical; weighted exposure and "
                f"one visit per transition are different contracts, got {weights.tolist()}"
            )
        identities = [
            (str(getattr(dataset, "dataset_path", "")), str(dataset.dataset_name)) for dataset in self.datasets
        ]
        if len(set(identities)) != len(identities):
            reasons.append(f"mixture contains duplicate dataset leaves: {identities!r}")
        for dataset in self.datasets:
            sampleable_count = int(np.asarray(dataset.sampleable_trajectory_lengths, dtype=np.int64).sum())
            indexed_count = len(dataset.all_steps)
            if sampleable_count != indexed_count:
                reasons.append(
                    "sampleable trajectory lengths disagree with the explicit transition "
                    f"index for dataset={dataset.dataset_name!r}: "
                    f"sampleable={sampleable_count}, all_steps={indexed_count}"
                )
        if reasons:
            raise ValueError(
                "sampling_strategy='transition_permutation' has an incompatible "
                "mixture contract: " + "; ".join(reasons)
            )

    def _initialize_transition_permutation(self) -> None:
        """Build compact leaf offsets and the deterministic rotating-tail schedule."""

        self._transition_leaf_ends = None
        self._transition_population_size = None
        self._transition_epoch_size = None
        self._transition_dropped_tail = None
        self._transition_tail_order = None
        self._transition_order = None
        self._transition_order_epoch = None
        if self.sampling_strategy != TRANSITION_PERMUTATION:
            return

        self._transition_leaf_ends = np.cumsum(self._dataset_lengths, dtype=np.int64)
        population_size = int(self._transition_leaf_ends[-1])
        epoch_multiple = self.sampling_epoch_multiple if self.mode == "train" else 1
        epoch_size = population_size - (population_size % epoch_multiple)
        if epoch_size <= 0:
            raise ValueError(
                "transition_permutation has fewer transitions than one aligned train "
                f"microbatch: population_size={population_size}, "
                f"sampling_epoch_multiple={epoch_multiple}."
            )
        self._transition_population_size = population_size
        self._transition_epoch_size = epoch_size
        self._transition_dropped_tail = population_size - epoch_size

        dtype = np.uint32 if population_size <= np.iinfo(np.uint32).max else np.uint64
        tail_order = np.arange(population_size, dtype=dtype)
        tail_rng = np.random.default_rng(
            safe_hash(
                (
                    "transition-permutation-tail-v1",
                    int(self.seed),
                    population_size,
                )
            )
        )
        tail_rng.shuffle(tail_order)
        tail_order.flags.writeable = False
        self._transition_tail_order = tail_order

    def _build_transition_order(self, epoch: int) -> np.ndarray:
        """Return one deterministic, unique, full-microbatch-aligned epoch."""

        population_size = int(self._transition_population_size)
        dropped_tail = int(self._transition_dropped_tail)
        dtype = self._transition_tail_order.dtype
        order = np.arange(population_size, dtype=dtype)

        if dropped_tail:
            # Rotate a fixed random coverage order so the tiny omitted tail is
            # auditable and evenly distributed across epochs; consecutive epochs omit
            # disjoint windows as long as epochs * dropped_tail <= population_size.
            start = (int(epoch) * dropped_tail) % population_size
            stop = start + dropped_tail
            if stop <= population_size:
                omitted = self._transition_tail_order[start:stop]
            else:
                omitted = np.concatenate(
                    (
                        self._transition_tail_order[start:],
                        self._transition_tail_order[: stop - population_size],
                    )
                )
            keep = np.ones(population_size, dtype=np.bool_)
            keep[np.asarray(omitted, dtype=np.intp)] = False
            order = order[keep]

        if len(order) != int(self._transition_epoch_size):
            raise RuntimeError(
                "Internal transition permutation length mismatch: "
                f"resolved={len(order)}, expected={self._transition_epoch_size}."
            )
        epoch_rng = np.random.default_rng(
            safe_hash(
                (
                    "transition-permutation-order-v1",
                    int(self.seed),
                    int(epoch),
                    population_size,
                )
            )
        )
        epoch_rng.shuffle(order)
        order.flags.writeable = False
        return order

    def _preflight_required_video_files(self) -> None:
        """Resolve every unique required shard once before strict training starts."""
        for dataset in self.datasets:
            if int(dataset.lerobot_info_meta.get("total_videos", 0)) == 0:
                continue
            unique_paths = {
                dataset.get_video_path(trajectory_id, key)
                for trajectory_id in dataset.trajectory_ids
                for key in dataset.modality_keys["video"]
            }
            missing_paths = sorted(
                (path for path in unique_paths if not path.is_file()),
                key=str,
            )
            if missing_paths:
                preview = [str(path) for path in missing_paths[:20]]
                raise FileNotFoundError(
                    "Required video preflight failed under sample_error_policy='raise': "
                    f"dataset={dataset.dataset_name}, missing={len(missing_paths)}/"
                    f"{len(unique_paths)}, first_paths={preview}"
                )
            if int(os.environ.get("RANK", "0")) == 0:
                print(
                    "[video-preflight] "
                    f"dataset={dataset.dataset_name} unique_required_shards={len(unique_paths)} "
                    "status=ok"
                )

    @property
    def num_episodes(self) -> int:
        """Total episodes actually reachable across all sub-datasets.

        Reflects any ``episode_fraction`` restriction applied at sub-dataset build
        time (each sub-dataset's ``trajectory_ids`` is already the kept subset), so
        this is the true "episodes used" count — the basis for the co-training video
        episode cap. NOT ``len(self)`` (that is the per-epoch STEP count).
        """
        return int(sum(np.count_nonzero(d.sampleable_trajectory_lengths) for d in self.datasets))

    def __str__(self) -> str:
        dataset_descriptions = []
        for dataset, weight in zip(self.datasets, self.dataset_sampling_weights, strict=False):
            dataset_description = {
                "Dataset": str(dataset),
                "Sampling weight": float(weight),
            }
            dataset_descriptions.append(dataset_description)
        return json.dumps({"Mixture dataset": dataset_descriptions}, indent=2)

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset.

        Args:
            epoch (int): The epoch to set.
        """
        if isinstance(epoch, bool) or not isinstance(epoch, (int, np.integer)) or int(epoch) < 0:
            raise ValueError(f"epoch must be a non-negative integer, got {epoch!r}.")
        self._shared_epoch.set(epoch)
        self._set_local_epoch(epoch)

    def _set_local_epoch(self, epoch: int) -> None:
        """Update this process's sampling caches without republishing the epoch."""

        effective_epoch = int(epoch) if self.mode == "train" else 0
        self.epoch = int(epoch)
        if self.sampling_strategy == TRANSITION_PERMUTATION and self._transition_order_epoch != effective_epoch:
            self._transition_order = self._build_transition_order(effective_epoch)
            self._transition_order_epoch = effective_epoch
        # self.sampled_steps = self.sample_epoch()

    def _sync_persistent_worker_epoch(self) -> None:
        if not self._persistent_worker_epoch:
            return
        published_epoch = self._shared_epoch.get()
        if published_epoch != self.epoch:
            self._set_local_epoch(published_epoch)

    def sample_step(self, index: int) -> tuple[LeRobotSingleDataset, int, int]:
        """Sample a single step from the dataset."""
        # return self.sampled_steps[index]

        self._sync_persistent_worker_epoch()

        if self.sampling_strategy == TRANSITION_PERMUTATION:
            if (
                isinstance(index, bool)
                or not isinstance(index, (int, np.integer))
                or int(index) < 0
                or int(index) >= int(self._transition_epoch_size)
            ):
                raise IndexError(
                    f"transition permutation index {index!r} is outside " f"[0, {self._transition_epoch_size})."
                )
            flat_index = int(self._transition_order[int(index)])
            dataset_index = int(np.searchsorted(self._transition_leaf_ends, flat_index, side="right"))
            leaf_start = 0 if dataset_index == 0 else int(self._transition_leaf_ends[dataset_index - 1])
            local_index = flat_index - leaf_start
            dataset = self.datasets[dataset_index]
            trajectory_id, base_index = dataset.all_steps[local_index]
            return dataset, trajectory_id, base_index

        # Set seed
        seed = index if self.mode != "train" else safe_hash((self.epoch, index, self.seed))
        rng = np.random.default_rng(seed)

        # Sample dataset
        dataset_index = rng.choice(len(self.datasets), p=self.dataset_sampling_weights)
        dataset = self.datasets[dataset_index]

        # Sample trajectory
        trajectory_index = rng.choice(len(dataset.trajectory_ids), p=self.trajectory_sampling_weights[dataset_index])
        trajectory_id = dataset.trajectory_ids[trajectory_index]

        # Sample step
        base_index = dataset.sample_base_index(trajectory_index, rng)
        return dataset, trajectory_id, base_index

    def __getitem__(self, index: int) -> dict:
        """Get the data for a single trajectory and start index.

        Args:
            index (int): The index of the trajectory to get.

        Returns:
            dict: The data for the trajectory and start index.
        """
        self._getitem_count += 1
        # DataLoader workers fork after the model is built, so each inherits the
        # whole VLM + generation-head object graph via copy-on-write (including any
        # frozen target encoder). A full gc.collect() every 100 items
        # then has to walk that inherited graph and stalls the worker for tens to
        # hundreds of seconds; all workers hit the boundary in lockstep, starving
        # the rank and tripping the NCCL watchdog on the other ranks. Freeze the
        # inherited graph into the permanent generation once so the periodic
        # collection only scans this worker's own runtime garbage.
        if self._getitem_count == 1:
            gc.freeze()
        if self._getitem_count % 100 == 0:
            gc.collect()

        if self.sample_error_policy == "raise":
            dataset = None
            trajectory_id = None
            step = None
            try:
                dataset, trajectory_id, step = self.sample_step(index)
                raw_data = dataset.get_step_data(trajectory_id, step)
                data = dataset.transforms(raw_data)
                sample = dataset._pack_sample(data)
                sample["_sample_provenance"] = {
                    "requested_index": int(index),
                    "sampling_epoch": int(self.epoch),
                    "dataset_name": str(dataset.dataset_name),
                    "episode_index": int(trajectory_id),
                    "base_index": int(step),
                    "sampling_strategy": self.sampling_strategy,
                    "canonical_transition_index": (
                        int(self._transition_order[int(index)])
                        if self.sampling_strategy == TRANSITION_PERMUTATION
                        else None
                    ),
                    "permutation_position": int(index) if self.sampling_strategy == TRANSITION_PERMUTATION else None,
                    "sample_error_policy": "raise",
                    "replacement_draws": 0,
                }
                return sample
            except Exception as exc:
                dataset_name = getattr(dataset, "dataset_name", None)
                raise RuntimeError(
                    "Dataloader sample failed with sample_error_policy='raise'; "
                    f"index={index}, dataset={dataset_name}, episode={trajectory_id}, "
                    f"step={step}, cause={type(exc).__name__}: {exc}. "
                    "No replacement sample was drawn."
                ) from exc

        max_retries = 10
        last_exception = None

        for attempt in range(max_retries):
            try:
                sample_tries = 0
                max_sample_tries = 200
                while True:
                    sample_tries += 1
                    if sample_tries > max_sample_tries:
                        raise RuntimeError(
                            f"Unable to sample a valid item after {max_sample_tries} attempts. "
                            f"dataset={self.datasets[0].dataset_name if len(self.datasets)>0 else 'unknown'}"
                        )

                    dataset, trajectory_id, step = self.sample_step(index)
                    # If dataset has no physical videos (e.g., image frames in parquet
                    # for VLA-Arena), do not gate sampling on mp4 existence.
                    total_videos = int(dataset.lerobot_info_meta.get("total_videos", 0))
                    if total_videos == 0:
                        break

                    key = dataset.modality_keys["video"][0].replace("video.", "")
                    video_path = dataset.get_video_path(trajectory_id, key)
                    if os.path.exists(video_path):
                        break
                    index = random.randint(0, len(self) - 1)

                raw_data = dataset.get_step_data(trajectory_id, step)
                data = dataset.transforms(raw_data)
                sample = dataset._pack_sample(data)

                return sample

            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    # Log the error but continue trying
                    print(f"Attempt {attempt + 1}/{max_retries} failed for index {index}: {e}")
                    print("Retrying with new sample...")
                    # For retry, we can use a slightly different index to get a new sample
                    # This helps avoid getting stuck on the same problematic sample
                    index = random.randint(0, len(self) - 1)
                else:
                    # All retries exhausted
                    print(f"All {max_retries} attempts failed for index {index}")
                    print(f"Last error: {last_exception}")
                    raise

    def __len__(self) -> int:
        """Get the length of a single epoch in the mixture.

        Returns:
            int: The length of a single epoch in the mixture.
        """
        if self.sampling_strategy == TRANSITION_PERMUTATION:
            return int(self._transition_epoch_size)

        # Check for potential issues
        if len(self.datasets) == 0:
            return 0

        # Check if any dataset lengths are 0 or NaN
        if np.any(self.dataset_lengths == 0) or np.any(np.isnan(self.dataset_lengths)):
            print(f"Warning: Found zero or NaN dataset lengths: {self.dataset_lengths}")
            # Filter out zero/NaN length datasets
            valid_indices = (self.dataset_lengths > 0) & (~np.isnan(self.dataset_lengths))
            if not np.any(valid_indices):
                print("Error: All datasets have zero or NaN length")
                return 0
        else:
            valid_indices = np.ones(len(self.datasets), dtype=bool)

        # Check if any sampling weights are 0 or NaN
        if np.any(self.dataset_sampling_weights == 0) or np.any(np.isnan(self.dataset_sampling_weights)):
            print(f"Warning: Found zero or NaN sampling weights: {self.dataset_sampling_weights}")
            # Use only valid weights
            valid_weights = (self.dataset_sampling_weights > 0) & (~np.isnan(self.dataset_sampling_weights))
            valid_indices = valid_indices & valid_weights
            if not np.any(valid_indices):
                print("Error: All sampling weights are zero or NaN")
                return 0

        # Check primary dataset indices
        primary_and_valid = self.primary_dataset_indices & valid_indices
        if not np.any(primary_and_valid):
            print(
                f"Warning: No valid primary datasets found. Primary indices: {self.primary_dataset_indices}, Valid indices: {valid_indices}"
            )
            # Fallback: use the largest valid dataset
            if np.any(valid_indices):
                max_length = self.dataset_lengths[valid_indices].max()
                print(f"Fallback: Using maximum dataset length: {max_length}")
                return int(max_length)
            else:
                return 0

        # Calculate the ratio and get max
        ratios = (self.dataset_lengths / self.dataset_sampling_weights)[primary_and_valid]

        # Check for NaN or inf in ratios
        if np.any(np.isnan(ratios)) or np.any(np.isinf(ratios)):
            print(f"Warning: Found NaN or inf in ratios: {ratios}")
            print(f"Dataset lengths: {self.dataset_lengths[primary_and_valid]}")
            print(f"Sampling weights: {self.dataset_sampling_weights[primary_and_valid]}")
            # Filter out invalid ratios
            valid_ratios = ratios[~np.isnan(ratios) & ~np.isinf(ratios)]
            if len(valid_ratios) == 0:
                print("Error: All ratios are NaN or inf")
                return 0
            max_ratio = valid_ratios.max()
        else:
            max_ratio = ratios.max()

        result = int(max_ratio)
        if result == 0:
            print("Warning: Dataset mixture length is 0")
        return result

    @staticmethod
    def compute_overall_statistics(
        per_task_stats: list[dict[str, dict[str, list[float] | np.ndarray]]],
        dataset_sampling_weights: list[float] | np.ndarray,
        percentile_mixing_method: str = "weighted_average",
    ) -> dict[str, dict[str, list[float]]]:
        """
        Computes overall statistics from per-task statistics using dataset sample weights.

        Args:
            per_task_stats: List of per-task statistics.
            Example format of one element in the per-task statistics list:
                {
                    "state.gripper": {
                        "min": [...],
                        "max": [...],
                        "mean": [...],
                        "std": [...],
                        "q01": [...],
                        "q99": [...],
                    },
                    ...
                }
            dataset_sampling_weights: List of sample weights for each task.
            percentile_mixing_method: The method to mix the percentiles, either "weighted_average" or "weighted_std".

        Returns:
            A dict of overall statistics per modality.
        """
        # Normalize the sample weights to sum to 1
        dataset_sampling_weights = np.array(dataset_sampling_weights)
        normalized_weights = dataset_sampling_weights / dataset_sampling_weights.sum()

        # Initialize overall statistics dict
        overall_stats: dict[str, dict[str, list[float]]] = {}

        # Get the list of modality keys
        modality_keys = per_task_stats[0].keys()

        for modality in modality_keys:
            # Number of dimensions (assuming consistent across tasks)
            num_dims = len(per_task_stats[0][modality]["mean"])

            # Initialize accumulators for means and variances
            weighted_means = np.zeros(num_dims)
            weighted_squares = np.zeros(num_dims)

            # Collect min, max, q01, q99 from all tasks
            min_list = []
            max_list = []
            q01_list = []
            q99_list = []

            for task_idx, task_stats in enumerate(per_task_stats):
                w_i = normalized_weights[task_idx]
                stats = task_stats[modality]
                means = np.array(stats["mean"])
                stds = np.array(stats["std"])

                # Update weighted sums for mean and variance
                weighted_means += w_i * means
                weighted_squares += w_i * (stds**2 + means**2)

                # Collect min, max, q01, q99
                min_list.append(stats["min"])
                max_list.append(stats["max"])
                q01_list.append(stats["q01"])
                q99_list.append(stats["q99"])

            # Compute overall mean
            overall_mean = weighted_means.tolist()

            # Compute overall variance and std deviation
            overall_variance = weighted_squares - weighted_means**2
            overall_std = np.sqrt(overall_variance).tolist()

            # Compute overall min and max per dimension
            overall_min = np.min(np.array(min_list), axis=0).tolist()
            overall_max = np.max(np.array(max_list), axis=0).tolist()

            # Compute overall q01 and q99 per dimension
            # Use weighted average of per-task quantiles
            q01_array = np.array(q01_list)
            q99_array = np.array(q99_list)
            if percentile_mixing_method == "weighted_average":
                weighted_q01 = np.average(q01_array, axis=0, weights=normalized_weights).tolist()
                weighted_q99 = np.average(q99_array, axis=0, weights=normalized_weights).tolist()
                # std_q01 = np.std(q01_array, axis=0).tolist()
                # std_q99 = np.std(q99_array, axis=0).tolist()
                # print(modality)
                # print(f"{std_q01=}, {std_q99=}")
                # print(f"{weighted_q01=}, {weighted_q99=}")
            elif percentile_mixing_method == "min_max":
                weighted_q01 = np.min(q01_array, axis=0).tolist()
                weighted_q99 = np.max(q99_array, axis=0).tolist()
            else:
                raise ValueError(f"Invalid percentile mixing method: {percentile_mixing_method}")

            # Store the overall statistics for the modality
            overall_stats[modality] = {
                "min": overall_min,
                "max": overall_max,
                "mean": overall_mean,
                "std": overall_std,
                "q01": weighted_q01,
                "q99": weighted_q99,
            }

        return overall_stats

    @staticmethod
    def merge_metadata(
        metadatas: list[DatasetMetadata],
        dataset_sampling_weights: list[float],
        percentile_mixing_method: str,
    ) -> DatasetMetadata:
        """Merge multiple metadata into one."""
        # Convert to dicts
        metadata_dicts = [metadata.model_dump(mode="json") for metadata in metadatas]
        # Create a new metadata dict
        merged_metadata = {}

        # Check all metadata have the same embodiment tag
        assert all(
            metadata.embodiment_tag == metadatas[0].embodiment_tag for metadata in metadatas
        ), "All metadata must have the same embodiment tag"
        merged_metadata["embodiment_tag"] = metadatas[0].embodiment_tag

        # Merge the dataset statistics
        dataset_statistics = {}
        dataset_statistics["state"] = LeRobotMixtureDataset.compute_overall_statistics(
            per_task_stats=[m["statistics"]["state"] for m in metadata_dicts],
            dataset_sampling_weights=dataset_sampling_weights,
            percentile_mixing_method=percentile_mixing_method,
        )
        dataset_statistics["action"] = LeRobotMixtureDataset.compute_overall_statistics(
            per_task_stats=[m["statistics"]["action"] for m in metadata_dicts],
            dataset_sampling_weights=dataset_sampling_weights,
            percentile_mixing_method=percentile_mixing_method,
        )
        merged_metadata["statistics"] = dataset_statistics

        # Merge the modality configs
        modality_configs = defaultdict(set)
        for metadata in metadata_dicts:
            for modality, configs in metadata["modalities"].items():
                modality_configs[modality].add(json.dumps(configs))
        merged_metadata["modalities"] = {}
        for modality, configs in modality_configs.items():
            # Check that all modality configs correspond to the same tag matches
            assert len(configs) == 1, f"Multiple modality configs for modality {modality}: {list(configs)}"
            merged_metadata["modalities"][modality] = json.loads(configs.pop())

        return DatasetMetadata.model_validate(merged_metadata)

    def update_metadata(self, metadata_config: dict, cached_statistics_path: Path | str | None = None) -> None:
        """
        Merge multiple metadatas into one and set the transforms with the merged metadata.

        Args:
            metadata_config (dict): Configuration for the metadata.
                "percentile_mixing_method": The method to mix the percentiles, either "weighted_average" or "min_max".
                    weighted_average: Use the weighted average of the percentiles using the weight used in sampling the datasets.
                    min_max: Use the min of the 1st percentile and max of the 99th percentile.
        """
        # If cached path is provided, try to load and apply
        if cached_statistics_path is not None:
            try:
                cached_stats = self.load_merged_statistics(cached_statistics_path)
                self.apply_cached_statistics(cached_stats)
                return
            except (FileNotFoundError, KeyError, ValidationError) as e:
                print(f"Failed to load cached statistics: {e}")
                print("Falling back to computing statistics from scratch...")

        self.tag = EmbodimentTag.NEW_EMBODIMENT.value
        self.merged_metadata: dict[str, DatasetMetadata] = {}
        # Group metadata by tag
        all_metadatas: dict[str, list[DatasetMetadata]] = {}
        for dataset in self.datasets:
            if dataset.tag not in all_metadatas:
                all_metadatas[dataset.tag] = []
            all_metadatas[dataset.tag].append(dataset.metadata)
        for tag, metadatas in all_metadatas.items():
            self.merged_metadata[tag] = self.merge_metadata(
                metadatas=metadatas,
                dataset_sampling_weights=self.dataset_sampling_weights.tolist(),
                percentile_mixing_method=metadata_config["percentile_mixing_method"],
            )
        for dataset in self.datasets:
            dataset.set_transforms_metadata(self.merged_metadata[dataset.tag])

    def save_dataset_statistics(self, save_path: Path | str, format: str = "json") -> None:
        """
        Save merged dataset statistics to specified path in the required format.
        Only includes statistics for keys that are actually used in the datasets.
        Gripper-related keys will be placed at the end.

        Args:
            save_path (Path | str): Path to save the statistics file
            format (str): Save format, currently only supports "json"
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Build the data structure to save
        statistics_data = {}

        # Collect actually used keys from all datasets
        all_used_action_keys = []
        all_used_state_keys = []

        for dataset in self.datasets:
            used_action_keys, used_state_keys = get_used_modality_keys(dataset.modality_keys)
            for used_action_key in used_action_keys:
                if used_action_key not in all_used_action_keys:
                    all_used_action_keys.append(used_action_key)
            for used_state_key in used_state_keys:
                if used_state_key not in all_used_state_keys:
                    all_used_state_keys.append(used_state_key)

        # Organize statistics by tag
        for tag, merged_metadata in self.merged_metadata.items():
            tag_stats = {}

            # Process action statistics
            if hasattr(merged_metadata.statistics, "action") and merged_metadata.statistics.action:
                action_stats = merged_metadata.statistics.action

                # Filter and reorder keys - iterate in all_used_action_keys order
                non_gripper_keys = []
                gripper_keys = []

                for key in all_used_action_keys:
                    if key in action_stats:
                        non_gripper_keys.append(key)

                reordered_keys = non_gripper_keys + gripper_keys

                filtered_action_stats = {}
                for key in reordered_keys:
                    filtered_action_stats[key] = action_stats[key]

                if filtered_action_stats:
                    combined_action_stats = combine_modality_stats(filtered_action_stats)

                    # Collect action normalization modes from datasets of this tag.
                    # "binary" takes precedence: if any dataset marks a key as binary, use binary.
                    _action_norm_modes: dict = {}
                    for _ds in self.datasets:
                        if _ds.tag == tag:
                            for _k, _m in _extract_action_normalization_modes(_ds.transforms).items():
                                if _k not in _action_norm_modes or _m == "binary":
                                    _action_norm_modes[_k] = _m
                    mask = generate_action_mask_for_used_keys(
                        merged_metadata.modalities.action,
                        filtered_action_stats.keys(),
                        normalization_modes=_action_norm_modes,
                    )
                    combined_action_stats["mask"] = mask

                    tag_stats["action"] = combined_action_stats

            # Process state statistics
            if hasattr(merged_metadata.statistics, "state") and merged_metadata.statistics.state:
                state_stats = merged_metadata.statistics.state

                # Filter and reorder keys - iterate in all_used_state_keys order
                # Filter and reorder keys - iterate in all_used_state_keys order
                non_gripper_keys = []
                gripper_keys = []

                for key in all_used_state_keys:
                    if key in state_stats:
                        non_gripper_keys.append(key)

                reordered_keys = non_gripper_keys + gripper_keys

                filtered_state_stats = {}
                for key in reordered_keys:
                    filtered_state_stats[key] = state_stats[key]

                if filtered_state_stats:
                    combined_state_stats = combine_modality_stats(filtered_state_stats)
                    tag_stats["state"] = combined_state_stats

            # Add dataset counts
            tag_stats.update(self._get_dataset_counts(tag))

            statistics_data[tag] = tag_stats

        # Save file
        if format.lower() == "json":
            if not str(save_path).endswith(".json"):
                save_path = save_path.with_suffix(".json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(statistics_data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}. Currently only 'json' is supported.")

        print(f"Merged dataset statistics saved to: {save_path}")
        print(f"Used action keys (reordered): {list(all_used_action_keys)}")
        print(f"Used state keys (reordered): {list(all_used_state_keys)}")

    def _combine_modality_stats(self, modality_stats: dict) -> dict:
        """Backward compatibility wrapper."""
        return combine_modality_stats(modality_stats)

    def _generate_action_mask_for_used_keys(self, action_modalities: dict, used_action_keys_ordered) -> list[bool]:
        """Backward compatibility wrapper."""
        return generate_action_mask_for_used_keys(action_modalities, used_action_keys_ordered)

    def _get_dataset_counts(self, tag: str) -> dict:
        """
        Get dataset count information for specified tag.

        Args:
            tag (str): embodiment tag

        Returns:
            dict: Dictionary containing num_transitions and num_trajectories
        """
        num_transitions = 0
        num_trajectories = 0

        # Count dataset information belonging to this tag
        for dataset in self.datasets:
            if dataset.tag == tag:
                num_transitions += len(dataset)
                num_trajectories += len(dataset.trajectory_ids)

        return {"num_transitions": num_transitions, "num_trajectories": num_trajectories}

    @classmethod
    def load_merged_statistics(cls, load_path: Path | str) -> dict:
        """
        Load merged dataset statistics from file.

        Args:
            load_path (Path | str): Path to the statistics file

        Returns:
            dict: Dictionary containing merged statistics
        """
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"Statistics file not found: {load_path}")

        if load_path.suffix.lower() == ".json":
            with open(load_path, "r", encoding="utf-8") as f:
                return json.load(f)
        elif load_path.suffix.lower() == ".pkl":
            import pickle

            with open(load_path, "rb") as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {load_path.suffix}")

    def apply_cached_statistics(self, cached_statistics: dict) -> None:
        """
        Apply cached statistics to avoid recomputation.

        Args:
            cached_statistics (dict): Statistics loaded from file
        """
        # Validate that cached statistics match current datasets
        if "metadata" in cached_statistics:
            cached_dataset_names = set(cached_statistics["metadata"]["dataset_names"])
            current_dataset_names = set(dataset.dataset_name for dataset in self.datasets)

            if cached_dataset_names != current_dataset_names:
                print("Warning: Cached statistics dataset names don't match current datasets.")
                print(f"Cached: {cached_dataset_names}")
                print(f"Current: {current_dataset_names}")
                return

        # Apply cached statistics
        self.merged_metadata = {}
        for tag, stats_data in cached_statistics.items():
            if tag == "metadata":  # Skip metadata field
                continue

            # Convert back to DatasetMetadata format
            metadata_dict = {"embodiment_tag": tag, "statistics": {"action": {}, "state": {}}, "modalities": {}}

            # Convert action statistics back
            if "action" in stats_data:
                action_data = stats_data["action"]
                # This is simplified - you may need to split back to sub-keys
                metadata_dict["statistics"]["action"] = action_data

            # Convert state statistics back
            if "state" in stats_data:
                state_data = stats_data["state"]
                metadata_dict["statistics"]["state"] = state_data

            self.merged_metadata[tag] = DatasetMetadata.model_validate(metadata_dict)

        # Update transforms metadata for each dataset
        for dataset in self.datasets:
            if dataset.tag in self.merged_metadata:
                dataset.set_transforms_metadata(self.merged_metadata[dataset.tag])

        print(f"Applied cached statistics for {len(self.merged_metadata)} embodiment tags.")
