# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#


"""DataConfig helpers and the base robot-type registry.

A *DataConfig* describes how one robot dataset is read: its camera / state / action /
language keys, the delta indices sampled around each base frame, the semantic view role
and description of every camera, and the normalization transform. Benchmark-specific
DataConfigs and mixtures live next to their example under
``examples/<Benchmark>/train_files/data_registry/data_config.py`` and are merged into the
global registries by :mod:`starVLA.dataloader.gr00t_lerobot.registry` (see the LIBERO,
DROID examples). ``ROBOT_TYPE_CONFIG_MAP`` below is the (empty) base map
those registries extend.
"""

import copy
import itertools
import logging
from abc import ABC, abstractmethod

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ModalityTransform

logger = logging.getLogger(__name__)


def apply_action_horizon(data_config, data_cfg):
    """Resolve the per-stream action index grid from the global chunk length.

    ``action_horizon`` remains the single model-shape contract. A dataset block may
    additionally provide either an exact ``action_delta_indices`` list or the compact
    integer form ``action_index_start`` + ``action_index_stride``. The exact list
    is what cross-stream FPS alignment uses because a ratio such as 30/20 cannot be
    represented by one integer stride.

    Without an explicit grid, plain ``range(N)`` classvars are resized to
    ``range(action_horizon)``; custom non-range classvars are left untouched. The
    override is applied on a shallow copy, so the shared singleton in
    ``ROBOT_TYPE_CONFIG_MAP`` is never mutated.
    """
    horizon = data_cfg.get("action_horizon") if data_cfg else None
    if horizon is None:
        return data_config
    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError(f"action_horizon must be positive, got {horizon}")

    explicit_indices = data_cfg.get("action_delta_indices", None)
    has_compact_grid = "action_index_start" in data_cfg or "action_index_stride" in data_cfg
    if explicit_indices is not None and has_compact_grid:
        raise ValueError("Set either action_delta_indices or action_index_start/action_index_stride, not both.")

    current = list(getattr(data_config, "action_indices", []) or [])
    is_plain_range = bool(current) and current == list(range(len(current)))
    if explicit_indices is not None:
        resolved = [int(index) for index in explicit_indices]
        source = "action_delta_indices"
    elif has_compact_grid:
        start = int(data_cfg.get("action_index_start", 0))
        stride = int(data_cfg.get("action_index_stride", 1))
        if start < 0:
            raise ValueError(f"action_index_start must be non-negative, got {start}")
        if stride <= 0:
            raise ValueError(f"action_index_stride must be positive, got {stride}")
        resolved = [start + i * stride for i in range(horizon)]
        source = f"start={start},stride={stride}"
    elif is_plain_range:
        resolved = list(range(horizon))
        source = "native"
    else:
        if current:
            logger.warning(
                "[action_horizon] %s.action_indices=%s is not a plain range(N); cannot "
                "derive it from action_horizon=%d - keeping the config's own indices.",
                type(data_config).__name__,
                current,
                horizon,
            )
        return data_config

    if len(resolved) != horizon:
        raise ValueError(f"action_delta_indices must contain action_horizon={horizon} entries, got {resolved}")
    if any(index < 0 for index in resolved):
        raise ValueError(f"action delta indices must be non-negative, got {resolved}")
    if any(right <= left for left, right in itertools.pairwise(resolved)):
        raise ValueError(f"action delta indices must be strictly increasing, got {resolved}")

    if current != resolved:
        data_config = copy.copy(data_config)
        data_config.action_indices = resolved
    logger.info(
        "[action-grid] %s horizon=%d source=%s -> %s",
        type(data_config).__name__,
        horizon,
        source,
        resolved,
    )
    return data_config


def apply_video_keys(data_config, data_cfg):
    """Override a DataConfig's observation ``video_keys`` from ``datasets.<block>.video_keys``.

    Lets a stream select WHICH camera views (and in what order) feed the VLM without
    subclassing the DataConfig — e.g. DROID records 3 cameras but defaults to 2; a
    multi-view tiling run selects all three by setting ``datasets.vla_data.video_keys``.
    The override is applied on a shallow copy so the shared singleton in
    ``ROBOT_TYPE_CONFIG_MAP`` is never mutated (mirrors :func:`apply_action_horizon`).

    The requested keys must be a subset of the config's ``available_video_keys`` when
    that classvar is declared (so a typo fails loudly instead of decoding a missing
    camera); configs without it accept any keys. No-op when ``video_keys`` is unset.
    """
    keys = data_cfg.get("video_keys") if data_cfg else None
    if keys is None:
        return data_config
    keys = [str(k) for k in keys]
    if not keys:
        raise ValueError("datasets.<block>.video_keys, when set, must be a non-empty list of camera keys.")

    available = list(getattr(data_config, "available_video_keys", []) or [])
    if available:
        unknown = [k for k in keys if k not in available]
        if unknown:
            raise ValueError(
                f"datasets.<block>.video_keys {unknown} not in {type(data_config).__name__}."
                f"available_video_keys {available}."
            )

    if list(getattr(data_config, "video_keys", []) or []) != keys:
        data_config = copy.copy(data_config)
        data_config.video_keys = keys
        logger.info("[video-keys] %s obs views -> %s", type(data_config).__name__, keys)
    return data_config


def resolve_view_roles(data_config, video_keys=None):
    """Resolve the ACTIVE video_keys -> (role, description) maps for a DataConfig.

    Reads the optional ``video_view_roles`` / ``video_view_descriptions`` classvars
    (keyed by video_key, so they survive :func:`apply_video_keys` reordering) and returns
    two dicts keyed by the active video_keys: ``{video_key: role}`` and
    ``{video_key: description}``.

    Validates (fail-fast with dataset context): every active key has a canonical role,
    every active key has an explicit non-empty description, roles are unique across the
    active views, and the anchor role ``central_global`` is present — the view-role
    context system relies on all four.
    """
    from starVLA.dataloader.view_tiling import ANCHOR_ROLE, normalize_view_role

    keys = list(video_keys if video_keys is not None else (getattr(data_config, "video_keys", []) or []))
    role_map = dict(getattr(data_config, "video_view_roles", {}) or {})
    desc_map = dict(getattr(data_config, "video_view_descriptions", {}) or {})
    name = type(data_config).__name__
    if not role_map:
        raise ValueError(f"{name} has no `video_view_roles`; the view-role system needs a video_key->role map.")
    if not desc_map:
        raise ValueError(
            f"{name} has no `video_view_descriptions`; the role-aware observation contract "
            "requires explicit text for every active camera."
        )

    roles, descriptions = {}, {}
    for key in keys:
        if key not in role_map:
            raise ValueError(
                f"{name}.video_view_roles has no role for active video_key {key!r} (have {list(role_map)})."
            )
        role = normalize_view_role(role_map[key])
        roles[key] = role
        if key not in desc_map or not isinstance(desc_map[key], str) or not desc_map[key].strip():
            raise ValueError(
                f"{name}.video_view_descriptions has no non-empty description for " f"active video_key {key!r}."
            )
        descriptions[key] = desc_map[key].strip()

    assigned = list(roles.values())
    duplicated = sorted({r for r in assigned if assigned.count(r) > 1})
    if duplicated:
        raise ValueError(f"{name}: view roles must be unique across active views, but {duplicated} repeat ({roles}).")
    if ANCHOR_ROLE not in assigned:
        raise ValueError(f"{name}: anchor role {ANCHOR_ROLE!r} must be present among active views (roles={roles}).")
    return roles, descriptions


class BaseDataConfig(ABC):
    """Interface of a DataConfig.

    Besides the two methods below, the loaders read these attributes: ``embodiment_tag``,
    ``video_keys`` / ``state_keys`` / ``action_keys`` / ``language_keys``,
    ``observation_indices`` / ``action_indices`` / ``state_indices``,
    ``video_view_roles`` / ``video_view_descriptions`` (one canonical role and one prompt
    description per camera; see :func:`resolve_view_roles`) and, optionally,
    ``available_video_keys`` (the cameras a ``datasets.<block>.video_keys`` override may select).
    """

    @abstractmethod
    def modality_config(self) -> dict[str, ModalityConfig]:
        pass

    @abstractmethod
    def transform(self) -> ModalityTransform:
        pass


# Base registry: robot_type -> DataConfig instance. Benchmark registries under
# examples/*/train_files/data_registry/ add their entries at import time.
ROBOT_TYPE_CONFIG_MAP: dict = {}
