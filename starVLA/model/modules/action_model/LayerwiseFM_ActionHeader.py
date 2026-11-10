# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Jinhui YE/ HKUST] in [2026].
# Modification: [rm and add some connect adapter to match with starVLA, e.g., "rm "].
"""Layer-wise cross-attention flow-matching action head (adapted from GR00T N1.5)."""

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta

from starVLA.model.modules.action_model.flow_matching_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)
from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, output_dim=2048):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


class ActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size=1024):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError("Expected `timesteps` to have shape (B,) so we can replicate across T.")

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.layer1(actions)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then layer2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.layer2(x))

        # 5) Finally W3 => (B, T, w)
        x = self.layer3(x)
        return x


DiTConfig = {
    "num_layers": 36,
    "input_embedding_dim": 2048,
    "attention_head_dim": 64,
    "num_attention_heads": 32,
}  # fallback DiT shape; the framework populates the real shape before construction


class LayerwiseFlowmatchingActionHead(nn.Module):
    """
    Layer-wise cross-attention DiT action head.

    NOTE on configuration boundary:
        This module is intentionally decoupled from any specific VLM backbone.
        It ONLY reads from ``global_config.framework.action_model`` (and its
        ``diffusion_model_cfg`` sub-tree).  The framework (``Qwen_PI``) is
        responsible for populating ``diffusion_model_cfg`` with the correct DiT
        shape **before** calling ``get_action_model``
        (see ``share_tools.populate_layerwise_dit_cfg``):

            diffusion_model_cfg.num_layers           = <DiT depth>
            diffusion_model_cfg.input_embedding_dim  = <DiT internal hidden>
            diffusion_model_cfg.cross_attention_dim  = <DiT internal hidden>
            diffusion_model_cfg.num_attention_heads  = input_embedding_dim // attention_head_dim

        The head never reads ``framework.qwenvl.*`` directly.
    """

    def __init__(self, global_config, **kwargs):
        super().__init__()
        action_config = global_config.framework.action_model
        diffusion_model_cfg = action_config.diffusion_model_cfg

        # ------------------------------------------------------------------
        # Pure consumer: trust diffusion_model_cfg.  Apply DiTConfig only as
        # a fallback for keys the framework forgot to set.  This keeps the
        # head free of any VLM-specific knowledge.
        # ------------------------------------------------------------------
        for k, v in DiTConfig.items():
            if diffusion_model_cfg.get(k, None) is None:
                diffusion_model_cfg[k] = v

        # Build a plain dict view and drop framework-side hints that are NOT
        # DiT constructor kwargs (e.g. `action_dit_hidden_dim`).  This keeps the
        # head agnostic of any framework convention while still being safe
        # against accidentally-leaked hint keys.
        _DIT_NON_KWARGS = {"action_dit_hidden_dim"}
        diffusion_model_cfg_kwargs = {k: v for k, v in diffusion_model_cfg.items() if k not in _DIT_NON_KWARGS}

        self.input_embedding_dim = diffusion_model_cfg_kwargs["input_embedding_dim"]
        self.model = DiT(**diffusion_model_cfg_kwargs)
        self.dit_out_hidden_size = self.input_embedding_dim
        self.action_dim = action_config.action_dim
        # `action_horizon` is the canonical chunk length (share_tools.apply_config_compat
        # derives `future_action_window_size` from it; this head never reads that key).
        self.action_horizon = int(action_config.action_horizon)
        self.num_inference_timesteps = action_config.num_inference_timesteps

        self.state_encoder = (
            MLP(
                input_dim=action_config.state_dim,
                output_dim=self.input_embedding_dim,
            )
            if action_config.state_dim
            else None
        )

        self.action_encoder = ActionEncoder(
            action_dim=action_config.action_dim,
            hidden_size=self.input_embedding_dim,
        )
        self.action_decoder = MLP(
            input_dim=self.input_embedding_dim,
            hidden_dim=1024,
            output_dim=self.action_dim,
        )
        self.future_tokens = nn.Embedding(action_config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        if action_config.add_pos_embed:
            self.position_embedding = nn.Embedding(action_config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(action_config.noise_beta_alpha, action_config.noise_beta_beta)
        self.num_timestep_buckets = action_config.num_timestep_buckets
        self.config = action_config

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def forward(
        self,
        vl_embs_list: list,
        actions: torch.Tensor,
        state: torch.Tensor = None,
        encoder_attention_mask=None,
        t_override=None,
    ):
        """
        vl_embs: list of torch.Tensor, each shape (B, seq_length, feature_dim)
        actions: shape (B, action_horizon, D_action)
        encoder_attention_mask: optional (B, seq_length) bool/int
        t_override: optional (B,) data-fraction timestep. When given, skip
            ``sample_time`` and noise at this exact ``t`` — used by the diffusion
            ``isolation=off`` joint path so the action and generation experts
            share one timestep.
        """
        device = actions.device
        B = vl_embs_list[0].shape[0]
        # Embed noised action trajectory.
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        if t_override is not None:
            t = t_override.to(device=actions.device, dtype=actions.dtype)
        else:
            t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized)

        # Embed state
        state_features = self.state_encoder(state) if state is not None else None

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # state and action embedding along sequence dimension.
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(B, -1, -1)
        sa_embs = (
            torch.cat((state_features, future_tokens, action_features), dim=1)
            if state_features is not None
            else torch.cat((future_tokens, action_features), dim=1)
        )

        # Layer-wise DiT forward. DiT handles cross/self-attention interleaving.
        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs_list,
            timestep=t_discretized,
            encoder_attention_mask=encoder_attention_mask,
            return_pre_output=True,
        )

        # Decode only the action-token positions.
        pred = self.action_decoder(model_output)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        loss = ((pred_actions - velocity) ** 2).mean()
        return loss

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs_list: list,
        state: torch.Tensor = None,
        encoder_attention_mask=None,
    ) -> torch.Tensor:
        # Set initial actions as the sampled noise.
        batch_size = vl_embs_list[0].shape[0]
        device = vl_embs_list[0].device
        actions = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=vl_embs_list[0].dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        state_features = self.state_encoder(state) if state is not None else None

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)
            t_discretized_int = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized_int, device=device, dtype=torch.long
            )

            # Embed current action trajectory with timestep
            action_features = self.action_encoder(actions, timesteps_tensor)

            # Maybe add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
            sa_embs = (
                torch.cat((state_features, future_tokens, action_features), dim=1)
                if state_features is not None
                else torch.cat((future_tokens, action_features), dim=1)
            )

            # Layer-wise DiT forward. DiT handles cross/self-attention interleaving.
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs_list,
                timestep=timesteps_tensor,
                encoder_attention_mask=encoder_attention_mask,
                return_pre_output=True,
            )
            # Decode only the action-token positions.
            pred = self.action_decoder(model_output)
            pred_velocity = pred[:, -self.action_horizon :]

            # Euler integration
            actions = actions + dt * pred_velocity
        return actions

    def action_denoise_step(
        self,
        vl_embs_list: list,
        actions: torch.Tensor,
        t_bucket: int,
        state_features: torch.Tensor = None,
        encoder_attention_mask=None,
    ) -> torch.Tensor:
        """One flow-matching velocity prediction, cross-attending to ``vl_embs_list``
        (optionally the ``isolation=off`` augmented encoder = VLM context + gen states)
        with ``encoder_attention_mask``.

        Mirrors the ``predict_action`` loop body but takes an explicit discretized
        timestep and honours the encoder mask; used by the diffusion joint inference
        loop so the action and generation experts step on a shared clock.

        Args:
            actions: (B, action_horizon, action_dim) current latent.
            t_bucket: discretized timestep (int).
            state_features: precomputed state features (or None).
        Returns:
            (B, action_horizon, action_dim) predicted velocity.
        """
        B = actions.shape[0]
        device = actions.device
        timesteps = torch.full((B,), int(t_bucket), device=device, dtype=torch.long)
        action_features = self.action_encoder(actions, timesteps)
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(B, -1, -1)
        sa_embs = (
            torch.cat((state_features, future_tokens, action_features), dim=1)
            if state_features is not None
            else torch.cat((future_tokens, action_features), dim=1)
        )
        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs_list,
            timestep=timesteps,
            encoder_attention_mask=encoder_attention_mask,
            return_pre_output=True,
        )
        pred = self.action_decoder(model_output)
        return pred[:, -self.action_horizon :]

    def predict_action_realtime(
        self,
        vl_embs_list: list,
        state: torch.Tensor = None,
        prev_action_chunk: torch.Tensor = None,
        inference_delay: int = 1,
        mode: str = "pigdm",
        suffix_length: int | None = None,
        prefix_attention_schedule: str = "exp",
        max_guidance_weight: float = 10.0,
        vl_embs_per_step: list | None = None,
        encoder_attention_mask=None,
    ) -> torch.Tensor:
        """RTC-aware flow-matching inference.

        Conditions sampling on the previous chunk's un-executed actions as a
        known prefix, so chunks splice smoothly under execution latency.

        Args:
            vl_embs_list: layer-wise VL embeddings, list of (B, seq, D).
            state: optional state tensor (B, 1, state_dim).
            prev_action_chunk: (B, action_horizon, action_dim) normalised
                actions from the previous prediction.
            inference_delay: number of leading timesteps to fix as prefix (d).
            mode: "pigdm" (default, VJP-guided; works without retraining,
                ~2x cost per step) or "simulated_delay" (naive
                replace-and-denoise with per-position timesteps; requires
                model trained with simulated-delay loss).
            suffix_length: ΠGDM only. Number of trailing timesteps with
                weight 0. Defaults to ``inference_delay``.
            prefix_attention_schedule: ΠGDM only. Weight schedule for the
                transition zone — "exp", "linear", "ones", or "zeros".
            max_guidance_weight: ΠGDM only. Cap on the guidance weight.
            vl_embs_per_step: optional list (length num_inference_timesteps) of
                per-step layer-wise encoders — used by the diffusion ``isolation=off``
                joint path, where the action expert cross-attends to a DIFFERENT
                augmented encoder ([VLM context + gen states] at that denoise step)
                each step. When None, ``vl_embs_list`` is used at every step.
            encoder_attention_mask: optional (B, S) mask applied at every cross-attn
                (needed when the encoder is the augmented [context + gen] sequence).

        Returns:
            actions: (B, action_horizon, action_dim) predicted actions.
        """
        if prev_action_chunk is None or inference_delay <= 0:
            return self.predict_action(vl_embs_list, state, encoder_attention_mask=encoder_attention_mask)

        import math

        batch_size = vl_embs_list[0].shape[0]
        device = vl_embs_list[0].device
        dtype = vl_embs_list[0].dtype

        def _enc(step):
            """Per-step encoder (augmented [context + gen states] under isolation=off)."""
            return vl_embs_per_step[step] if vl_embs_per_step is not None else vl_embs_list

        def _prefix_weights(start: int, end: int, total: int, schedule: str) -> torch.Tensor:
            """ΠGDM prefix attention weight schedule across the action horizon."""
            start = min(start, end)
            positions = torch.arange(total, device=device, dtype=torch.float32)
            if schedule == "ones":
                w = torch.ones(total, device=device)
            elif schedule == "zeros":
                w = (positions < start).float()
            elif schedule in ("linear", "exp"):
                w = ((start - 1 - positions) / (end - start + 1) + 1).clamp(0, 1)
                if schedule == "exp":
                    w = w * torch.expm1(w) / (math.e - 1)
            else:
                raise ValueError(f"Invalid schedule: {schedule}")
            return torch.where(positions >= end, torch.zeros_like(w), w)

        def _forward_velocity(actions, timesteps, state_features, vl=None, mask=None):
            """Single DiT forward pass returning predicted velocity."""
            B = actions.shape[0]
            action_features = self.action_encoder(actions, timesteps)
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(B, -1, -1)
            sa_embs = (
                torch.cat((state_features, future_tokens, action_features), dim=1)
                if state_features is not None
                else torch.cat((future_tokens, action_features), dim=1)
            )
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl if vl is not None else vl_embs_list,
                timestep=timesteps,
                encoder_attention_mask=mask,
                return_pre_output=True,
            )
            pred = self.action_decoder(model_output)
            return pred[:, -self.action_horizon :]

        T_prev = prev_action_chunk.shape[1]
        if T_prev < self.action_horizon:
            pad = torch.zeros(
                batch_size,
                self.action_horizon - T_prev,
                prev_action_chunk.shape[2],
                device=device,
                dtype=prev_action_chunk.dtype,
            )
            prev_action_chunk = torch.cat([prev_action_chunk, pad], dim=1)

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        actions = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=dtype,
            device=device,
        )

        state_features = self.state_encoder(state) if state is not None else None

        if mode == "pigdm":
            # ── ΠGDM: pseudo-inverse guided diffusion ──
            if suffix_length is None:
                suffix_length = inference_delay
            prefix_attention_end = self.action_horizon - suffix_length
            weights = _prefix_weights(
                inference_delay,
                prefix_attention_end,
                self.action_horizon,
                prefix_attention_schedule,
            )
            prev = prev_action_chunk.to(dtype)

            for t_step in range(num_steps):
                t_cont = t_step / float(num_steps)
                t_bucket = int(t_cont * self.num_timestep_buckets)
                timesteps_tensor = torch.full(
                    (batch_size,),
                    t_bucket,
                    device=device,
                    dtype=torch.long,
                )

                with torch.enable_grad():
                    x_t = actions.detach().requires_grad_(True)
                    v_t = _forward_velocity(
                        x_t,
                        timesteps_tensor,
                        state_features,
                        vl=_enc(t_step),
                        mask=encoder_attention_mask,
                    )
                    x_1_hat = x_t + v_t * (1.0 - t_cont)
                    error = (prev - x_1_hat) * weights[None, :, None]
                    pinv_correction = torch.autograd.grad(
                        x_1_hat,
                        x_t,
                        grad_outputs=error,
                    )[0]

                # Guidance weight schedule from ΠGDM paper
                t = t_cont
                inv_r2 = (t**2 + (1 - t) ** 2) / ((1 - t) ** 2 + 1e-8)
                c = (1 - t) / (t + 1e-8)
                gw = min(c * inv_r2, max_guidance_weight)

                v_corrected = v_t.detach() + gw * pinv_correction.detach()
                actions = actions.detach() + dt * v_corrected

            return actions

        # ── simulated_delay mode ──
        with torch.no_grad():
            prefix_mask = torch.zeros(
                1,
                self.action_horizon,
                1,
                device=device,
                dtype=torch.bool,
            )
            prefix_mask[:, :inference_delay, :] = True

            for t_step in range(num_steps):
                t_cont = t_step / float(num_steps)
                actions = torch.where(prefix_mask, prev_action_chunk.to(dtype), actions)

                t_bucket = int(t_cont * self.num_timestep_buckets)
                t_bucket_prefix = int(1.0 * self.num_timestep_buckets) - 1
                timesteps_tensor = torch.full(
                    (batch_size, self.action_horizon),
                    t_bucket,
                    device=device,
                    dtype=torch.long,
                )
                timesteps_tensor[:, :inference_delay] = t_bucket_prefix

                # Inline ActionEncoder with per-position timesteps
                a_emb = self.action_encoder.layer1(actions)
                tau_emb = self.action_encoder.pos_encoding(timesteps_tensor).to(dtype=a_emb.dtype)
                x_enc = torch.cat([a_emb, tau_emb], dim=-1)
                x_enc = swish(self.action_encoder.layer2(x_enc))
                action_features = self.action_encoder.layer3(x_enc)

                if self.config.add_pos_embed:
                    pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                    pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                    action_features = action_features + pos_embs

                future_tokens = self.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
                sa_embs = (
                    torch.cat((state_features, future_tokens, action_features), dim=1)
                    if state_features is not None
                    else torch.cat((future_tokens, action_features), dim=1)
                )

                temb_tensor = torch.full(
                    (batch_size,),
                    t_bucket,
                    device=device,
                    dtype=torch.long,
                )
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=_enc(t_step),
                    timestep=temb_tensor,
                    encoder_attention_mask=encoder_attention_mask,
                    return_pre_output=True,
                )

                pred = self.action_decoder(model_output)
                pred_velocity = pred[:, -self.action_horizon :]
                actions = actions + dt * pred_velocity
                actions = torch.where(prefix_mask, prev_action_chunk.to(dtype), actions)

            return actions

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


def get_action_model(config=None):
    """
    Factory: build the action expert from the global framework config.

    Args:
        config: Global config (expects a ``config.framework.action_model`` namespace).

    Returns:
        LayerwiseFlowmatchingActionHead: the initialized action head.
    """
    return LayerwiseFlowmatchingActionHead(global_config=config)
