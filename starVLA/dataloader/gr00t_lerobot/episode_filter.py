"""Episode-level curation of LeRobot v3.0 datasets from their episode metadata.

The robot data of the report is curated from DROID by keeping successful episodes and
dropping those without a valid instruction, without valid action annotations, or with
no motion. LeRobot v3.0 stores, per episode, the task strings (``tasks``) and per-feature
statistics (``stats/<feature>/{min,max,...}``) in ``meta/episodes/*/*.parquet``, so the
curation needs no pass over the frame data:

    datasets.vla_data.episode_filter:
      require_instruction: true   # at least one non-empty task string
      require_motion: true        # finite action statistics and at least one action
                                  # dimension that changes during the episode
      motion_tolerance: 1.0e-6    # max - min below this counts as "does not change"

The successful / failed split is chosen by the data mixture (``droid_success``).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

_FILTER_KEYS = {"require_instruction", "require_motion", "motion_tolerance"}

NO_INSTRUCTION = "no_instruction"
INVALID_ACTION = "invalid_action"
NO_MOTION = "no_motion"


def parse_episode_filter(data_cfg) -> dict | None:
    """Return the normalized ``episode_filter`` spec of a stream block, or ``None`` if disabled."""
    raw = data_cfg.get("episode_filter", None) if data_cfg is not None and hasattr(data_cfg, "get") else None
    if not raw:
        return None
    keys = set(raw.keys())
    unknown = keys - _FILTER_KEYS
    if unknown:
        raise ValueError(f"Unknown episode_filter keys {sorted(unknown)}; expected a subset of {sorted(_FILTER_KEYS)}.")
    spec = {
        "require_instruction": bool(raw.get("require_instruction", False)),
        "require_motion": bool(raw.get("require_motion", False)),
        "motion_tolerance": float(raw.get("motion_tolerance", 1e-6)),
    }
    if spec["motion_tolerance"] < 0:
        raise ValueError(f"episode_filter.motion_tolerance must be >= 0, got {spec['motion_tolerance']}.")
    if not (spec["require_instruction"] or spec["require_motion"]):
        return None
    return spec


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _stat(episode: Mapping, columns: set, key: str, stat: str) -> np.ndarray:
    column = f"stats/{key}/{stat}"
    if column not in columns:
        raise ValueError(
            f"episode_filter.require_motion needs the per-episode statistic {column!r} in the LeRobot v3.0 "
            "episode metadata, but it is missing. Disable require_motion for this dataset."
        )
    return np.asarray(_as_list(episode[column]), dtype=np.float64).reshape(-1)


def episode_rejection_reason(
    episode: Mapping, columns: Iterable[str], action_keys: Iterable[str], spec: dict
) -> str | None:
    """Return why an episode is dropped (``no_instruction`` / ``invalid_action`` / ``no_motion``), or ``None``.

    ``episode`` is one row of the LeRobot v3.0 episode metadata, ``columns`` the metadata
    columns and ``action_keys`` the original (parquet) keys of the action features in use.
    """
    columns = set(columns)
    if spec.get("require_instruction"):
        if "tasks" not in columns:
            raise ValueError(
                "episode_filter.require_instruction needs the 'tasks' column of the LeRobot v3.0 episode "
                "metadata, but it is missing. Disable require_instruction for this dataset."
            )
        tasks = [str(task).strip() for task in _as_list(episode["tasks"]) if task is not None]
        if not any(task and task.lower() not in ("nan", "none") for task in tasks):
            return NO_INSTRUCTION

    if spec.get("require_motion"):
        action_keys = list(action_keys)
        if not action_keys:
            raise ValueError("episode_filter.require_motion needs at least one action feature.")
        tolerance = float(spec.get("motion_tolerance", 1e-6))
        moving = False
        for key in action_keys:
            low, high = _stat(episode, columns, key, "min"), _stat(episode, columns, key, "max")
            if low.size == 0 or low.shape != high.shape or not (np.isfinite(low).all() and np.isfinite(high).all()):
                return INVALID_ACTION
            moving = moving or bool(np.any(high - low > tolerance))
        if not moving:
            return NO_MOTION
    return None
