# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Policy server wrapper.

Encapsulates a `baseframework` instance plus a :class:`PolicyNormProcessor`
that reuses the *training-time* :class:`ComposedModalityTransform` for action
un-normalization (no hand-rolled math). The websocket server returns
already-unnormalized actions.

Client-side responsibilities that REMAIN on the client:
  - environment-specific adapters (image_history, gripper sticky, action
    ensembling)
  - chunk-cache scheduling (`step % chunk_size == 0` triggers a new infer)

Exposed API:
  - ``metadata`` (dict, sent at handshake): ``action_chunk_size``,
    ``available_unnorm_keys``, ``action_keys``, ``state_keys``, and the
    checkpointed observation prompt contract.
  - ``predict_action(examples, unnorm_key=None, **kwargs)`` returns
    ``{"actions": np.ndarray[B, T, action_dim]}``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from deployment.model_server.policy_norm_processor import PolicyNormProcessor
from starVLA.dataloader.view_tiling import (
    normalize_view_prompt_mode,
    normalize_view_tiling_mode,
)
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import read_mode_config


class PolicyServerWrapper:
    """Wraps a `baseframework` for use as a websocket-server policy."""

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        use_bf16: bool = False,
        unnorm_key: Optional[str] = None,
    ) -> None:
        self._ckpt_path = str(ckpt_path)

        logging.info("PolicyServerWrapper: loading framework from %s", self._ckpt_path)
        framework = baseframework.from_pretrained(self._ckpt_path)
        if use_bf16:
            framework = framework.to(torch.bfloat16)
        framework = framework.to(device).eval()
        self._framework = framework

        # Co-located metadata.
        model_cfg, _ = read_mode_config(self._ckpt_path)
        self._model_cfg = model_cfg
        datasets_cfg = model_cfg.get("datasets", {})
        shared_cfg = datasets_cfg.get("shared", {})
        vla_cfg = datasets_cfg.get("vla_data", {})
        self._view_prompt_mode = normalize_view_prompt_mode(
            vla_cfg.get(
                "view_prompt_mode",
                shared_cfg.get("view_prompt_mode", None),
            )
        )
        self._view_tiling = normalize_view_tiling_mode(vla_cfg.get("view_tiling", shared_cfg.get("view_tiling", None)))
        resolved_view_prompt = vla_cfg.get("resolved_view_prompt", None)
        if resolved_view_prompt is None or not hasattr(resolved_view_prompt, "get"):
            raise ValueError(
                "PolicyServerWrapper: the role-aware runtime requires a valid "
                "datasets.vla_data.resolved_view_prompt mapping in the checkpoint's "
                "config.yaml, but this checkpoint does not contain one."
            )
        if int(resolved_view_prompt.get("version", 0)) != 1:
            raise ValueError(
                "PolicyServerWrapper: unsupported resolved_view_prompt "
                f"version {resolved_view_prompt.get('version', None)!r}."
            )
        if resolved_view_prompt.get("mode", None) != "role":
            raise ValueError(
                "PolicyServerWrapper: resolved_view_prompt.mode must be "
                f"'role', got {resolved_view_prompt.get('mode', None)!r}."
            )
        prompt_tiling = normalize_view_tiling_mode(resolved_view_prompt.get("view_tiling", None))
        if prompt_tiling != self._view_tiling:
            raise ValueError(
                "PolicyServerWrapper: resolved prompt tiling does not match "
                f"the checkpoint dataset config: prompt={prompt_tiling!r}, "
                f"dataset={self._view_tiling!r}."
            )
        self._view_prompt_texts = list(resolved_view_prompt.get("texts", []))
        self._view_prompt_roles = list(resolved_view_prompt.get("image_roles", []))
        invalid_texts = [
            index
            for index, prompt in enumerate(self._view_prompt_texts)
            if not isinstance(prompt, str) or not prompt.strip()
        ]
        if not self._view_prompt_texts or len(self._view_prompt_texts) != len(self._view_prompt_roles) or invalid_texts:
            raise ValueError(
                "PolicyServerWrapper: resolved_view_prompt must contain one "
                "non-empty text per image role; "
                f"texts={self._view_prompt_texts!r}, "
                f"image_roles={self._view_prompt_roles!r}."
            )

        # Number of actions returned per inference call (= action_horizon).
        action_model_cfg = model_cfg["framework"]["action_model"]

        if "action_horizon" in action_model_cfg:
            self._action_chunk_size = int(action_model_cfg["action_horizon"])
        elif "future_action_window_size" in action_model_cfg:
            self._action_chunk_size = int(action_model_cfg["future_action_window_size"]) + 1
        else:
            raise ValueError(
                f"PolicyServerWrapper: no action_horizon or future_action_window_size found in model config for {self._ckpt_path}"
            )
        # Cache of PolicyNormProcessor instances per unnorm_key.
        # For single-dataset ckpts unnorm_key is auto-selected; for multi-dataset
        # ckpts clients must pass unnorm_key per request.
        self._default_unnorm_key = unnorm_key
        self._norm_processors: Dict[str, PolicyNormProcessor] = {}

        # Peek at available keys without building a full processor.
        _, _ns = read_mode_config(self._ckpt_path)
        self._available_unnorm_keys: List[str] = list(_ns.keys())

        # Eagerly build when unambiguous; defer for multi-key / no explicit key.
        if unnorm_key is not None or len(self._available_unnorm_keys) == 1:
            default_proc = self._get_processor(unnorm_key)
            self._default_unnorm_key = default_proc.unnorm_key
            logging.info(
                "PolicyServerWrapper ready: action_chunk_size=%d, default_unnorm_key=%s, "
                "available_unnorm_keys=%s, action_keys=%s, state_keys=%s",
                self._action_chunk_size,
                default_proc.unnorm_key,
                default_proc.available_unnorm_keys,
                default_proc.action_keys,
                default_proc.state_keys,
            )
        else:
            logging.info(
                "PolicyServerWrapper ready (multi-key): action_chunk_size=%d, "
                "available_unnorm_keys=%s — clients must pass unnorm_key per request.",
                self._action_chunk_size,
                self._available_unnorm_keys,
            )

    def _get_processor(self, unnorm_key: Optional[str]) -> PolicyNormProcessor:
        cache_key = unnorm_key if unnorm_key is not None else "__default__"
        if cache_key not in self._norm_processors:
            self._norm_processors[cache_key] = PolicyNormProcessor(self._ckpt_path, unnorm_key=unnorm_key)
        return self._norm_processors[cache_key]

    @property
    def metadata(self) -> Dict[str, Any]:
        """Model-invariant metadata; sent to client at websocket handshake."""
        base = {
            "env": "starvla_policy_server",
            "ckpt_path": self._ckpt_path,
            "action_chunk_size": self._action_chunk_size,
            "available_unnorm_keys": self._available_unnorm_keys,
            "default_unnorm_key": self._default_unnorm_key,
            "view_prompt_mode": self._view_prompt_mode,
            "requires_view_texts": True,
            "view_tiling": self._view_tiling,
            "view_prompt_texts": self._view_prompt_texts,
            "view_prompt_roles": self._view_prompt_roles,
        }
        # Enrich with per-embodiment keys when a default processor already exists.
        if self._default_unnorm_key is not None:
            proc = self._get_processor(self._default_unnorm_key)
            base["action_keys"] = proc.action_keys
            base["state_keys"] = proc.state_keys
        return base

    def predict_action(
        self,
        examples: List[dict],
        unnorm_key: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """Run the framework, then un-normalize via training-time transforms.

        Args:
            examples: list of dicts (each with ``image`` / ``lang`` / optional ``state``).
            unnorm_key: dataset key for un-normalization stats. ``None`` -->
                use the wrapper's default (auto-picked at startup).
            **kwargs: forwarded to the framework's ``predict_action``
                (``do_sample``, ``use_ddim``, ``num_ddim_steps``, ...).

        Returns:
            ``{"actions": np.ndarray[B, T, D]}`` -- un-normalized.
        """
        effective_key = unnorm_key if unnorm_key is not None else self._default_unnorm_key
        if effective_key is None:
            if len(self._available_unnorm_keys) == 1:
                effective_key = self._available_unnorm_keys[0]
            else:
                raise ValueError(
                    f"predict_action: unnorm_key not specified and no default set. "
                    f"Pass one of {self._available_unnorm_keys}."
                )
        proc = self._get_processor(effective_key)

        out = self._framework.predict_action(examples=examples, **kwargs)
        normalized = np.asarray(out["normalized_actions"])  # (B, T, D)

        unnorm = np.stack(
            [proc.unapply_actions(normalized[b]) for b in range(normalized.shape[0])],
            axis=0,
        )
        return {"actions": unnorm}
