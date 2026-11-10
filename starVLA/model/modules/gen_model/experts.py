# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Backbone axis (the ``expert`` variant) of the generation head.

Two dedicated-parameter "generation experts", both layer-wise cross-attention
DiTs over the SAME per-layer VLM hidden states (``vl_embs_list``) the action
expert consumes — so the VLM is never fed generation tokens and stays untouched
even when frozen:

  * :class:`GenerationExpert` — DETERMINISTIC (timestep pinned to 0, no
    diffusion).  Turns learnable query tokens into contextualised query states
    for the ``regression`` decoder.

  * :class:`DiffusionExpert` — GENERATIVE (the ``diffusion`` mode).  The exact
    twin of ``LayerwiseFlowmatchingActionHead``, only the noised input is a
    visual-latent target instead of an action chunk: noised target tokens go in
    as the DiT ``hidden_states``, the DiT cross-attends layer-by-layer to
    ``vl_embs_list``, and flow matching regresses the velocity.
"""

import torch
import torch.nn as nn

from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT, TimestepEncoder
from starVLA.model.modules.gen_model.losses import build_flow_matching
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# Fallback DiT shape (same defaults as the action expert's DiTConfig); the
# framework populates the real shape via populate_gen_expert_dit_cfg before the
# expert is built, so these only cover keys left unset.
_GEN_DIT_DEFAULTS = {
    "num_layers": 36,
    "input_embedding_dim": 2048,
    "attention_head_dim": 64,
    "num_attention_heads": 32,
}


def _dit_kwargs(gen_cfg):
    dit_cfg = gen_cfg.expert_dit_cfg
    for k, v in _GEN_DIT_DEFAULTS.items():
        if dit_cfg.get(k, None) is None:
            dit_cfg[k] = v
    return {k: v for k, v in dit_cfg.items()}


class GenerationExpert(nn.Module):
    """Layer-wise cross-attention DiT processing learnable generation queries.

    Forward:
        vl_embs_list: list of (B, L, H) layer-wise VLM hidden states.
        encoder_attention_mask: optional (B, L) bool mask over VLM tokens.
    Returns:
        (B, num_gen_queries, hidden) contextualised query states, fed to the
        decoder exactly like the VLM-produced query states in the "vlm" backbone.
    """

    def __init__(self, global_config, num_gen_queries: int, frame_conditioned: bool = False):
        super().__init__()
        gen_cfg = global_config.framework.gen_model
        dit_kwargs = _dit_kwargs(gen_cfg)

        self.hidden_dim = int(dit_kwargs["num_attention_heads"]) * int(dit_kwargs["attention_head_dim"])
        self.num_gen_queries = int(num_gen_queries)
        self.model = DiT(**dit_kwargs)

        self.gen_query_tokens = nn.Embedding(self.num_gen_queries, self.hidden_dim)
        nn.init.normal_(self.gen_query_tokens.weight, mean=0.0, std=0.02)

        add_pos = bool(gen_cfg.get("expert_add_pos_embed", True))
        if add_pos:
            self.position_embedding = nn.Embedding(self.num_gen_queries, self.hidden_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        else:
            self.position_embedding = None

        # frame_mode='conditioned': ONE shared query group produces DIFFERENT future frames'
        # representations, selected by a per-sample frame index injected via adaLN (added to
        # the DiT's zero timestep embedding as an additive temb bias). Only built when
        # conditioned, so the parallel path has no extra parameters.
        self.frame_conditioned = bool(frame_conditioned)
        self.frame_embedder = TimestepEncoder(embedding_dim=self.hidden_dim) if self.frame_conditioned else None

        logger.info(
            f"[GenerationExpert] num_layers={dit_kwargs['num_layers']} "
            f"hidden={self.hidden_dim} queries={self.num_gen_queries} add_pos={add_pos} "
            f"frame_conditioned={self.frame_conditioned}"
        )

    def forward(self, vl_embs_list: list, encoder_attention_mask=None, frame_index=None, return_all_hidden_states=False):
        B = vl_embs_list[0].shape[0]
        device = vl_embs_list[0].device

        query = self.gen_query_tokens.weight.unsqueeze(0).expand(B, -1, -1)
        if self.position_embedding is not None:
            pos_ids = torch.arange(self.num_gen_queries, dtype=torch.long, device=device)
            query = query + self.position_embedding(pos_ids).unsqueeze(0)

        # Deterministic regression: a constant (zero) timestep just conditions the DiT's
        # ada-norm; there is no diffusion process here. When frame_conditioned, a per-sample
        # frame-index embedding is added on top (temb_bias) so the SAME queries decode
        # different future frames' representations.
        timestep = torch.zeros(B, dtype=torch.long, device=device)
        temb_bias = None
        if self.frame_embedder is not None and frame_index is not None:
            temb_bias = self.frame_embedder(frame_index)

        return self.model(
            hidden_states=query,
            encoder_hidden_states=vl_embs_list,
            timestep=timestep,
            encoder_attention_mask=encoder_attention_mask,
            return_pre_output=True,
            return_all_hidden_states=return_all_hidden_states,
            temb_bias=temb_bias,
        )  # (B, num_gen_queries, hidden_dim), or (final, all_hidden_states) if return_all_hidden_states


class DiffusionExpert(nn.Module):
    """Layer-wise cross-attention *diffusion* DiT — the ``diffusion`` mode.

    Structural twin of ``LayerwiseFlowmatchingActionHead``: the noised target
    tokens are the DiT input; the DiT cross-attends layer-by-layer to
    ``vl_embs_list``; flow matching regresses the velocity ``v = x0 - noise``.
    The DiT's own output projection (``output_dim = target_dim``) emits the
    predicted velocity, so no separate decoder MLP is needed.
    """

    def __init__(self, global_config, target, frame_conditioned: bool = False):
        super().__init__()
        gen_cfg = global_config.framework.gen_model
        dit_kwargs = _dit_kwargs(gen_cfg)

        self.hidden_dim = int(dit_kwargs["num_attention_heads"]) * int(dit_kwargs["attention_head_dim"])
        self.target_dim = int(target.feature_dim())

        # noised target tokens (target_dim) -> DiT width; DiT proj_out -> velocity (target_dim).
        self.latent_in = nn.Linear(self.target_dim, self.hidden_dim)
        self.model = DiT(**{**dit_kwargs, "output_dim": self.target_dim})

        # Learnable per-token position embedding table; the target token count
        # (e.g. a VAE latent's T*H*W) must not exceed max_seq_len. Default 8192 is
        # generous for image/video latents; forward() asserts to fail loudly.
        self.max_seq_len = int(gen_cfg.get("max_seq_len", 8192))
        add_pos = bool(gen_cfg.get("expert_add_pos_embed", True))
        if add_pos:
            self.position_embedding = nn.Embedding(self.max_seq_len, self.hidden_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        else:
            self.position_embedding = None

        # frame_mode='conditioned': denoise a SINGLE randomly selected future frame's
        # features per step, conditioned on its frame index via adaLN (added to the
        # flow-matching timestep embedding as a temb bias). Only built when conditioned, so
        # the parallel diffusion path has no extra parameters.
        self.frame_conditioned = bool(frame_conditioned)
        self.frame_embedder = TimestepEncoder(embedding_dim=self.hidden_dim) if self.frame_conditioned else None

        # Timestep distribution is target-dependent (RAE recipe for repr, WAN recipe for vae).
        self.fm = build_flow_matching(
            gen_cfg, target, num_timestep_buckets=int(gen_cfg.get("num_timestep_buckets", 1000))
        )
        logger.info(
            f"[DiffusionExpert] num_layers={dit_kwargs['num_layers']} "
            f"hidden={self.hidden_dim} target_dim={self.target_dim} add_pos={add_pos} "
            f"frame_conditioned={self.frame_conditioned}"
        )

    def forward(
        self,
        vl_embs_list: list,
        target: torch.Tensor,
        encoder_attention_mask=None,
        t_override=None,
        frame_index=None,
        return_all_hidden_states=False,
    ):
        """vl_embs_list: list of (B, L, H); target: (B, N_t, D_t) -> scalar loss.

        ``t_override`` (B,) data-fraction timestep forces the SAME noise level the
        action expert uses (the ``isolation=off`` joint path); otherwise this
        expert samples its own ``t``. ``frame_index`` (B,) is the conditioned-mode per-sample
        frame position, injected via adaLN (temb bias) alongside the flow-matching timestep
        so ONE frame's features are denoised per step, selected by its index. With
        ``return_all_hidden_states`` the DiT's per-layer internal hidden states are also
        returned so the action expert can cross-attend to them (returns
        ``(loss, all_hidden_states)``)."""
        target = target.float()
        if t_override is not None:
            x_t, velocity, t_disc = self.fm.add_noise_with_t(target, t_override)
        else:
            x_t, velocity, t_disc = self.fm.add_noise(target)

        temb_bias = None
        if self.frame_embedder is not None and frame_index is not None:
            temb_bias = self.frame_embedder(frame_index)

        h = self._embed_latent(x_t)
        out = self.model(
            hidden_states=h,
            encoder_hidden_states=vl_embs_list,
            timestep=t_disc,
            encoder_attention_mask=encoder_attention_mask,
            return_all_hidden_states=return_all_hidden_states,
            temb_bias=temb_bias,
        )
        if return_all_hidden_states:
            pred, all_hidden = out  # (B, N_t, D_t), list of (B, N_t, hidden)
            return self.fm.loss(pred.float(), velocity, t_disc), all_hidden
        return self.fm.loss(out.float(), velocity, t_disc)

    def _embed_latent(self, x_t: torch.Tensor) -> torch.Tensor:
        """Project a noised target latent to DiT width and add position embedding."""
        h = self.latent_in(x_t)
        if self.position_embedding is not None:
            n = h.shape[1]
            assert n <= self.max_seq_len, (
                f"target token count {n} exceeds expert max_seq_len={self.max_seq_len}; "
                "raise framework.gen_model.max_seq_len or reduce the target resolution/frames."
            )
            pos_ids = torch.arange(n, dtype=torch.long, device=h.device)
            h = h + self.position_embedding(pos_ids).unsqueeze(0)
        return h

    @torch.no_grad()
    def denoise_step(self, vl_embs_list: list, x_t: torch.Tensor, t_disc: torch.Tensor, encoder_attention_mask=None):
        """One joint-inference denoising step (``isolation=off`` diffusion path).

        Given the current noised latent ``x_t`` (B, N_t, D_t) at discretized
        timestep ``t_disc`` (B,), return ``(velocity_pred, all_hidden_states)`` —
        the flow-matching velocity for the Euler step plus the DiT's per-layer
        internal hidden states the action expert cross-attends to.
        """
        h = self._embed_latent(x_t)
        pred, all_hidden = self.model(
            hidden_states=h,
            encoder_hidden_states=vl_embs_list,
            timestep=t_disc,
            encoder_attention_mask=encoder_attention_mask,
            return_all_hidden_states=True,
        )
        return pred, all_hidden
