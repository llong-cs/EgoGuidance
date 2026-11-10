# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""``regression`` mode: learnable tokens -> backbone -> small MLP -> regress target.

A few FFN layers map the backbone's query hidden states to a frozen visual
encoder's representation of the future frame(s), regressed directly (no diffusion).

  * backbone : ``vlm`` (query states from the VLM) | ``expert`` (a dedicated
    cross-attention DiT produces them from ``vl_embs_list``).
  * decoder  : ``mlp`` (a small FFN projector).
  * target   : any encoder in ``targets.py`` (``auto`` / ``siglip`` / ``dino`` /
    ``vjepa`` / ``lingbot`` / ``pixel`` / ``wan_vae``; a VAE latent grid is
    flattened to tokens).
  * loss     : ``mse`` | ``cosine`` | ``cosine_magnitude`` | ``smooth_l1``.
"""

import torch

from starVLA.model.modules.gen_model import register_gen_head
from starVLA.model.modules.gen_model.base import BaseGenHead
from starVLA.model.modules.gen_model.decoders import MLPDecoder
from starVLA.model.modules.gen_model.experts import GenerationExpert
from starVLA.model.modules.gen_model.losses import get_pointwise_loss
from starVLA.model.modules.gen_model.targets import align_lengths, build_target, grid_to_tokens
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


@register_gen_head("regression")
class RegressionGenHead(BaseGenHead):
    def __init__(self, global_config, vlm_interface=None, **kwargs):
        super().__init__(global_config, vlm_interface=vlm_interface)
        self.pool = str(self.gen_cfg.target.get("pool", "none"))
        self.loss_type = str(self.gen_cfg.loss.get("type", "mse"))
        decoder_type = str(self.gen_cfg.decoder.get("type", "mlp")).lower()
        if decoder_type != "mlp":
            raise ValueError(f"regression mode supports decoder.type='mlp' only, got {decoder_type!r}.")

        self.target = build_target(global_config, vlm_interface, self.input_dim)

        # ── Optional target-feature standardization (gen_model.normalize_target) ──
        # Target encoders emit features at very different scales (VLM hidden states vs
        # unit-variance VAE latents vs ViT-g features), so a plain MSE at a fixed
        # loss_scale applies very different gradient pressure per target. When enabled,
        # the target is standardized with per-target (mean, std) so gen_loss lands in a
        # comparable range across targets. The stats default to each target class's
        # calibrated constants (feat_mean / feat_std) and can be overridden per target via
        # ``targets[i].target.feat_mean`` / ``feat_std``. Only applied for the magnitude
        # losses (mse / smooth_l1); cosine is scale-invariant. cosine_magnitude consumes
        # RAW features and uses feat_std only as the norm-error scale. Uncalibrated targets
        # keep (0, 1) -> exact no-op. Affects the TRAINING target only.
        self.normalize_target = bool(self.gen_cfg.get("normalize_target", False)) and self.loss_type in (
            "mse",
            "smooth_l1",
        )
        tcfg = self.gen_cfg.target
        self._feat_mean = float(tcfg.get("feat_mean", getattr(self.target, "feat_mean", 0.0)))
        self._feat_std = float(tcfg.get("feat_std", getattr(self.target, "feat_std", 1.0)))
        if self._feat_std <= 0:
            raise ValueError(f"gen_model.targets[i].target.feat_std must be > 0, got {self._feat_std}")
        self.loss_fn = get_pointwise_loss(
            self.loss_type,
            feat_std=self._feat_std,
            loss_cfg=self.gen_cfg.loss,
        )

        self.decoder = MLPDecoder(
            input_dim=self.input_dim,
            output_dim=self.target.feature_dim(),
            num_layers=int(self.gen_cfg.decoder.get("num_layers", 3)),
        )
        logger.info(
            f"[RegressionGenHead] backbone={self.backbone} target={self.gen_cfg.target.get('type')} "
            f"target_dim={self.target.feature_dim()} loss={self.loss_type} pool={self.pool} "
            f"normalize_target={self.normalize_target} (mean={self._feat_mean}, std={self._feat_std})"
        )

    def _standardize_target(self, target):
        """Standardize target features (see __init__). No-op when disabled or at the (0, 1) default."""
        if not self.normalize_target:
            return target
        return (target - self._feat_mean) / self._feat_std

    def _resolve_num_queries(self):
        """Number of learnable queries / expert tokens.

        Explicit ``num_gen_queries`` wins. Otherwise (``auto``): parallel mode uses the
        target's full token count (per_frame * T); conditioned mode uses ONE frame's worth
        of tokens (per_frame == count_tokens // T), since a single random frame is generated
        per step.
        """
        if self.num_gen_queries is not None:
            return self.num_gen_queries
        ct = int(self.target.count_tokens())
        if self._conditioned:
            T = int(self._num_frames)
            if ct % T != 0:
                raise ValueError(
                    f"frame_mode='conditioned' needs a per-frame-decomposable target: "
                    f"target.type={self.gen_cfg.target.get('type')!r} emits {ct} tokens for T={T} "
                    f"(not divisible). Use a per-frame target (auto/siglip/dino/lingbot/pixel)."
                )
            return ct // T
        return ct

    def build(self, vlm_interface):
        # 0) resolve conditioned-frame state: T and whether frame_mode='conditioned' is active.
        self._resolve_frame_conditioning()
        # 1) resolve num_gen_queries (auto -> target token count, or per-frame when conditioned).
        n = self._resolve_num_queries()
        self._write_num_queries(n)
        # 2) wire the backbone.
        if self.backbone == "expert":
            self._populate_expert_dit_cfg()
            self.expert = GenerationExpert(self._global_config, n, frame_conditioned=self._conditioned)
            self.requires_vl_embs = True
        else:
            vlm_interface.setup_gen_queries(n)
            self.requires_vlm_queries = True
        self._apply_freezes()

    def _condition(self, query_states, vl_embs_list, encoder_attention_mask, frame_index=None):
        if self.backbone == "expert":
            cached = self._consume_cached_condition()  # set by isolation_context (isolation=off)
            if cached is not None:
                return cached
            return self.expert(vl_embs_list, encoder_attention_mask=encoder_attention_mask, frame_index=frame_index)
        return query_states

    def forward(self, *, query_states=None, vl_embs_list=None, target_frames, encoder_attention_mask=None):
        if self._conditioned:
            return self._forward_conditioned(vl_embs_list, target_frames, encoder_attention_mask)
        # Encode the target OUTSIDE autocast (frozen encoder, no_grad, own dtype).
        # A VAE target returns a latent GRID; flatten it to tokens (repr targets
        # already emit tokens) so the deterministic MLP can regress it.
        tgt = self.target.encode(target_frames)
        if self.target.kind == "vae":
            tgt = grid_to_tokens(tgt)
        target = align_lengths(tgt.float(), self.num_gen_queries, self.pool)
        target = self._standardize_target(target)
        dtype = torch.bfloat16 if self.requires_vlm_queries else torch.float32
        with torch.autocast("cuda", dtype=dtype):
            condition = self._condition(query_states, vl_embs_list, encoder_attention_mask)
            pred = self.decoder(condition).float()  # (B, N_q, D_t)

        if pred.shape[1] != target.shape[1]:
            raise ValueError(
                f"query/target token mismatch: pred {pred.shape[1]} vs target {target.shape[1]}. "
                f"Set gen_model.targets[i].num_gen_queries to the target token count (or 'auto'), "
                f"or set gen_model.targets[i].target.pool='target_to_query'."
            )
        return self.loss_fn(pred, target)

    def _forward_conditioned(self, vl_embs_list, target_frames, encoder_attention_mask):
        """frame_mode='conditioned': per sample regress the representation of ONE randomly
        chosen future frame, using the SAME per-frame queries conditioned on that frame's
        index (adaLN)."""
        device = vl_embs_list[0].device
        selected, frame_index = self._sample_conditioned_frames(target_frames, self._num_frames, device)
        tgt = self.target.encode(selected)
        if self.target.kind == "vae":
            tgt = grid_to_tokens(tgt)
        target = align_lengths(tgt.float(), self.num_gen_queries, self.pool)  # (B, per_frame, D_t)
        target = self._standardize_target(target)
        with torch.autocast("cuda", dtype=torch.float32):
            condition = self._condition(None, vl_embs_list, encoder_attention_mask, frame_index=frame_index)
            pred = self.decoder(condition).float()
        if pred.shape[1] != target.shape[1]:
            raise ValueError(
                f"query/target token mismatch (conditioned): pred {pred.shape[1]} vs target {target.shape[1]}."
            )
        return self.loss_fn(pred, target)
