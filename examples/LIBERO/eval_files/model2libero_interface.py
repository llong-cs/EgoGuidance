# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""LIBERO-side policy client (thin WebSocket client).

The policy server (``deployment/model_server/server_policy.py``) loads the checkpoint,
applies the checkpointed observation contract (image resizing, role prompts) and returns
**already un-normalized** action chunks. It advertises ``action_chunk_size`` and the
role-prompt contract in its handshake metadata. This client therefore only:

  - checks that the checkpoint's view roles match LIBERO's ``[front, wrist]`` cameras,
  - attaches the checkpointed per-view role prompts (``view_texts``) to every request,
  - requests a new chunk every ``action_chunk_size`` steps and executes it open-loop
    (no temporal ensembling).
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

# LIBERO renders [agentview (front), robot0_eye_in_hand (wrist)]; the training data config
# (examples/LIBERO/train_files/data_registry) maps them to these canonical view roles.
EXPECTED_VIEW_ROLES = ["central_global", "primary_effector"]
ACTION_DIM = 7  # delta xyz (3) + delta axis-angle (3) + gripper open (1)


class ModelClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 10093, unnorm_key: Optional[str] = None) -> None:
        # Connects (retrying until the server is up) and receives the handshake metadata.
        self.client = WebsocketClientPolicy(host, port)
        meta = self.client.get_server_metadata()
        self.server_metadata = meta
        self.action_chunk_size = int(meta["action_chunk_size"])

        if meta.get("view_prompt_mode", None) != "role" or meta.get("requires_view_texts", None) is not True:
            raise ValueError(
                "The LIBERO client requires the role-aware observation contract, but the server did not advertise it "
                f"(view_prompt_mode={meta.get('view_prompt_mode')!r}, "
                f"requires_view_texts={meta.get('requires_view_texts')!r})."
            )
        if meta.get("view_tiling", "off") == "tile":
            raise ValueError(
                "The LIBERO client sends separate front/wrist images, but this checkpoint expects a "
                "pre-composited view mosaic (view_tiling=tile)."
            )
        view_texts = meta.get("view_prompt_texts", None)
        if not view_texts:
            raise ValueError("The server did not publish view_prompt_texts (checkpointed resolved_view_prompt).")
        view_roles = list(meta.get("view_prompt_roles", None) or [])
        if view_roles != EXPECTED_VIEW_ROLES:
            raise ValueError(
                f"The LIBERO client sends [front, wrist] images and therefore requires checkpoint view roles "
                f"{EXPECTED_VIEW_ROLES!r}, got {view_roles!r}."
            )
        self.view_texts: List[str] = list(view_texts)
        self.unnorm_key = unnorm_key

        self.task_description: Optional[str] = None
        self.action_chunk: Optional[np.ndarray] = None  # (T, ACTION_DIM), un-normalized
        logging.info(
            "Connected to policy server %s:%s (action_chunk_size=%d, unnorm_key=%s, view_texts=%s)",
            host,
            port,
            self.action_chunk_size,
            unnorm_key,
            self.view_texts,
        )

    def reset(self, task_description: str) -> None:
        """Start a new episode: drop the cached action chunk."""
        self.task_description = task_description
        self.action_chunk = None

    def step(self, images: List[np.ndarray], instruction: str, step: int) -> np.ndarray:
        """Return the un-normalized action for policy step ``step`` of the current episode.

        Args:
            images: ``[front, wrist]`` uint8 HWC images at the simulator's native resolution.
                Resizing follows the checkpoint's observation contract on the server.
            instruction: task language instruction.
            step: policy step counter within the episode (0, 1, 2, ...). A new chunk is
                requested whenever ``step % action_chunk_size == 0``.

        Returns:
            ``np.ndarray`` of shape ``(7,)``: delta xyz, delta axis-angle, gripper
            (1 = open, 0 = closed).
        """
        if instruction != self.task_description:
            self.reset(instruction)
        if len(images) != len(self.view_texts):
            raise ValueError(
                f"Image count does not match the checkpointed role-prompt contract: images={len(images)}, "
                f"view_texts={len(self.view_texts)}."
            )

        if step % self.action_chunk_size == 0 or self.action_chunk is None:
            example = {"image": list(images), "lang": instruction, "view_texts": list(self.view_texts)}
            response = self.client.predict_action({"examples": [example], "unnorm_key": self.unnorm_key})
            if not response.get("ok", False):
                raise RuntimeError(f"Policy server returned an error: {response.get('error', response)}")
            actions = np.asarray(response["data"]["actions"])  # (B, T, D), un-normalized server-side
            if actions.ndim != 3 or actions.shape[-1] != ACTION_DIM:
                raise ValueError(f"Expected actions of shape (1, T, {ACTION_DIM}), got {actions.shape}.")
            self.action_chunk = actions[0]

        return self.action_chunk[step % self.action_chunk_size]

    def close(self) -> None:
        self.client.close()
