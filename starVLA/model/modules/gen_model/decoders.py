# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Decoder axis of the generation head.

The decoder maps the backbone's query states to the target feature space.  The
``regression`` mode uses a small MLP projector (``decoder.type: mlp``) and applies a
pointwise loss to its prediction; the ``diffusion`` mode has no separate decoder (the
diffusion expert's own output projection predicts the velocity).

Freeze/unfreeze: every decoder exposes ``set_freeze(freeze)`` (``freeze=None`` ->
the decoder's ``default_freeze``).
"""

import torch.nn as nn

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


class _Decoder(nn.Module):
    """Common decoder interface + freeze semantics."""

    default_freeze = False

    def set_freeze(self, freeze):
        """freeze=None -> the decoder's default; else force (un)freeze the module."""
        f = self.default_freeze if freeze is None else bool(freeze)
        self.requires_grad_(not f)
        if f:
            self.eval()
        logger.info(f"[gen_model] decoder '{type(self).__name__}': {'FROZEN' if f else 'trainable'}")


class MLPDecoder(_Decoder):
    """num_layers-deep MLP: input_dim -> hidden -> ... -> output_dim (GELU, final LayerNorm).

    The ``regression`` mode's decoder.
    """

    def __init__(self, input_dim, output_dim, hidden_dim=None, num_layers=3):
        super().__init__()
        hidden_dim = hidden_dim or max(input_dim, output_dim)
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # no activation on the final projection
                layers.append(nn.GELU(approximate="tanh"))
        self.net = nn.Sequential(*layers)
        self.out_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        return self.out_norm(self.net(x))
