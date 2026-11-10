# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""``diffusion`` mode: the generation expert IS a denoising DiT.

The exact twin of the action expert, only generating visual features: the noised
target tokens go in as the DiT input, the DiT cross-attends layer-by-layer to
the VLM hidden states (``vl_embs_list``), and flow matching regresses the
velocity.  There is no separate condition producer and no learnable VLM queries
— the expert both consumes the VLM context and denoises.

  * backbone : ``expert`` only (enforced).
  * decoder  : none (the expert is the decoder).
  * target   : ``wan_vae`` (VAE latent) or a representation target (``auto`` /
    ``siglip`` / ``dino`` / ``vjepa`` / ``lingbot`` / ``pixel``).
  * loss     : ``flow_matching`` (enforced).
"""

import torch

from starVLA.model.modules.gen_model import register_gen_head
from starVLA.model.modules.gen_model.base import BaseGenHead
from starVLA.model.modules.gen_model.experts import DiffusionExpert
from starVLA.model.modules.gen_model.targets import build_target, grid_to_tokens
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


@register_gen_head("diffusion")
class DiffusionGenHead(BaseGenHead):
    requires_vl_embs = True  # always consumes the layer-wise VLM hidden states

    def __init__(self, global_config, vlm_interface=None, **kwargs):
        super().__init__(global_config, vlm_interface=vlm_interface)
        if self.backbone != "expert":
            logger.warning(f"diffusion mode forces backbone='expert' (got {self.backbone!r}); overriding.")
            self.backbone = "expert"
        loss_type = str(self.gen_cfg.loss.get("type", "flow_matching"))
        if loss_type != "flow_matching":
            raise ValueError(f"diffusion mode requires loss.type='flow_matching', got {loss_type!r}.")

        self.target = build_target(global_config, vlm_interface, self.input_dim)
        logger.info(
            f"[DiffusionGenHead] target={self.gen_cfg.target.get('type')} " f"target_dim={self.target.feature_dim()}"
        )

    def build(self, vlm_interface):
        # The expert is a layer-wise cross-attention DiT (same shape as the
        # action expert); no VLM queries are registered.
        self._resolve_frame_conditioning()  # T + whether frame_mode='conditioned' is active
        self._populate_expert_dit_cfg()
        self.expert = DiffusionExpert(self._global_config, self.target, frame_conditioned=self._conditioned)
        self._cached_loss = None  # stashed by isolation_context (joint isolation=off)
        self._apply_freezes()

    def _encode_target(self, target_frames):
        target = self.target.encode(target_frames)
        if self.target.kind == "vae":
            target = grid_to_tokens(target)
        return target

    def forward(self, *, query_states=None, vl_embs_list=None, target_frames, encoder_attention_mask=None):
        # In the joint isolation=off path the loss was already computed (at the
        # shared timestep) by isolation_context; reuse it to avoid a second pass.
        cached = self._cached_loss
        if cached is not None:
            self._cached_loss = None
            return cached
        # frame_mode='conditioned': denoise ONE randomly chosen future frame's features per
        # step, conditioned on its frame index (adaLN). isolation=off is rejected at build
        # time for this mode, so it never collides with the cached joint-loss path.
        if self._conditioned:
            device = vl_embs_list[0].device
            selected, frame_index = self._sample_conditioned_frames(target_frames, self._num_frames, device)
            target = self._encode_target(selected)
            with torch.autocast("cuda", dtype=torch.float32):
                return self.expert(
                    vl_embs_list, target, encoder_attention_mask=encoder_attention_mask, frame_index=frame_index
                )
        # Encode the target OUTSIDE autocast (frozen, no_grad). The expert is
        # token-based, so a VAE latent grid is flattened to tokens.
        target = self._encode_target(target_frames)
        with torch.autocast("cuda", dtype=torch.float32):
            return self.expert(vl_embs_list, target, encoder_attention_mask=encoder_attention_mask)

    # ── isolation=off (joint) support ──
    def isolation_context(
        self,
        *,
        query_states=None,
        vl_embs_list=None,
        encoder_attention_mask=None,
        t_override=None,
        target_frames=None,
    ):
        """Run the diffusion expert ONCE at the SHARED timestep ``t_override``:
        stash the flow-matching loss for :meth:`forward`, and return the expert's
        per-layer hidden states so the action expert can cross-attend to the
        (partially noised) future-frame representation — joint denoising with a
        shared timestep, one-directional (the gen expert never sees action tokens)."""
        target = self._encode_target(target_frames)
        # repeated_diffusion_steps>1: the framework repeats the VLM context and draws
        # reps*B shared timesteps so gen & action share the SAME per-row timestep. The
        # target is encoded ONCE (batch B) then repeated to match, so each copy is noised
        # at its own shared t. reps==1 (or vl_embs already at B) -> no-op.
        b_enc, b_tgt = vl_embs_list[0].shape[0], target.shape[0]
        if b_enc != b_tgt:
            assert b_enc % b_tgt == 0, f"gen encoder batch {b_enc} is not a multiple of target batch {b_tgt}."
            target = target.repeat(b_enc // b_tgt, *([1] * (target.dim() - 1)))
        with torch.autocast("cuda", dtype=torch.float32):
            loss, all_hidden = self.expert(
                vl_embs_list,
                target,
                encoder_attention_mask=encoder_attention_mask,
                t_override=t_override,
                return_all_hidden_states=True,
            )
        self._cached_loss = loss
        return self._align_gen_layers(all_hidden, len(vl_embs_list))

    def init_gen_latent(self, batch_size, device, dtype=torch.float32):
        """Initial pure-noise generation latent ``(B, N_t, D_t)`` for the joint inference loop."""
        d = int(self.target.feature_dim())
        n_tok = int(self.target.count_tokens())
        return torch.randn(batch_size, n_tok, d, device=device, dtype=dtype)

    def gen_denoise_step(self, vl_embs_list, x_gen, t_cont, encoder_attention_mask=None):
        """One joint-inference gen denoising step -> (velocity_pred, per-layer G_list)."""
        B = x_gen.shape[0]
        buckets = self.expert.fm.num_timestep_buckets
        t_disc = torch.full((B,), int(t_cont * buckets), device=x_gen.device, dtype=torch.long)
        pred, all_hidden = self.expert.denoise_step(
            vl_embs_list, x_gen, t_disc, encoder_attention_mask=encoder_attention_mask
        )
        return pred, self._align_gen_layers(all_hidden, len(vl_embs_list))

    @torch.no_grad()
    def denoise_gen_trajectory(self, vl_embs_list, encoder_attention_mask, num_steps, device):
        """Run the gen expert's Euler denoising for ``num_steps`` and return the list
        of per-layer G_lists (one entry per step).

        The generation expert never attends to the action tokens, so its trajectory
        is INDEPENDENT of the action denoising — precomputing it lets both the plain
        joint loop and the RTC (execution-latency-aware) loop reuse the same, shared
        timestep grid the action expert steps on.
        """
        B = vl_embs_list[0].shape[0]
        dt = 1.0 / num_steps
        x = self.init_gen_latent(B, device, dtype=torch.float32)
        traj = []
        for step in range(num_steps):
            t_cont = step / float(num_steps)
            v, G = self.gen_denoise_step(vl_embs_list, x, t_cont, encoder_attention_mask=encoder_attention_mask)
            traj.append(G)
            x = x + dt * v.float()
        return traj
