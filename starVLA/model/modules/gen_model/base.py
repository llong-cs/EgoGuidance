# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Base contract every generation-mode head implements.

The framework (e.g. ``Qwen_PI``) only ever talks to this small surface, so it
stays mode-agnostic:

    head = get_gen_model(config, vlm_interface)   # picks the mode
    head.build(vlm_interface)                     # resolve queries / build expert / register VLM queries
    ...
    loss = head(query_states=..., vl_embs_list=..., target_frames=...,
                encoder_attention_mask=...)

Capability flags tell the framework how to feed the head:
    requires_vlm_queries -> append learnable gen queries to the VLM input and
                            pass the sliced ``query_states``.
    requires_vl_embs     -> the head consumes the layer-wise ``vl_embs_list``
                            (expert backbone / diffusion).

``build()`` centralises the mode-specific wiring (auto ``num_gen_queries``
resolution, generation-expert DiT-shape population + construction, VLM query
registration).
"""

import torch
import torch.nn as nn

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


def set_frozen(module, freeze: bool, name: str = ""):
    """Freeze (requires_grad_(False) + eval) or unfreeze an nn.Module in place."""
    if module is None:
        return
    module.requires_grad_(not freeze)
    if freeze:
        module.eval()
    logger.info(f"[gen_model] component '{name}': {'FROZEN' if freeze else 'trainable'}")


class BaseGenHead(nn.Module):
    #: framework reads these to decide how to feed the head at forward time.
    requires_vlm_queries: bool = False
    requires_vl_embs: bool = False

    def __init__(self, global_config, vlm_interface=None):
        super().__init__()
        self._global_config = global_config
        self.gen_cfg = global_config.framework.gen_model
        # VLM hidden size (framework-populated before the head is built); the
        # decoder / projector / expert all run at this width.
        self.input_dim = int(self.gen_cfg.get("input_dim"))
        self.backbone = str(self.gen_cfg.get("backbone", "vlm")).lower()
        self.num_gen_queries = self._parse_nq(self.gen_cfg.get("num_gen_queries", "auto"))
        # frame_mode (video gen): 'parallel' (default — all T future frames' tokens are
        # generated at once by T*per_frame queries) | 'conditioned' (ONE shared per-frame
        # query group; per sample a random frame index t is drawn and injected via adaLN so
        # the head decodes just that frame). 'conditioned' is only wired for the expert
        # backbone (regression/diffusion); the factory validates the combination. Subclass
        # build() sets _conditioned/_num_frames.
        self.frame_mode = str(self.gen_cfg.get("frame_mode", "parallel")).lower()
        self._conditioned = False
        self._num_frames = 1
        # set by build() for the expert backbone
        self.expert = None
        # scratch slot: isolation_context() stashes the expert output here so the
        # subsequent loss forward reuses it instead of re-running the expert.
        self._cached_condition = None

    # ── helpers shared by the mode heads ──
    @staticmethod
    def _parse_nq(nq):
        if isinstance(nq, str) and nq.lower() == "auto":
            return None
        return int(nq)

    def _write_num_queries(self, n):
        """Persist the resolved query count onto self + config (for the snapshot)."""
        self.num_gen_queries = int(n)
        self._global_config.framework.gen_model.num_gen_queries = int(n)

    def _resolve_frame_conditioning(self):
        """Resolve conditioned-frame state once at build(): store ``_num_frames`` (T) and whether
        ``frame_mode='conditioned'`` is ACTIVE (needs T>1; T==1 degrades to parallel).

        Called by the expert-backbone heads (regression / diffusion) at the start of build();
        returns ``_conditioned`` so build() can size queries / the expert accordingly.
        """
        from starVLA.model.modules.gen_model.targets import resolve_num_target_frames

        self._num_frames = int(resolve_num_target_frames(self._global_config))
        self._conditioned = self.frame_mode == "conditioned" and self._num_frames > 1
        return self._conditioned

    def _sample_conditioned_frames(self, target_frames, num_frames, device):
        """Conditioned-frame sampling: per sample pick ONE of the ``T`` future frames at random.

        Returns ``(selected_frames, frame_index)`` where ``selected_frames`` is a
        ``List[List[PIL]]`` of one frame each (B, 1) fed to the target encoder, and
        ``frame_index`` is a ``(B,)`` float tensor of the normalized frame position
        ``(t+1)/T`` in ``(0, 1]`` fed to the gen expert's frame embedder (adaLN). Each
        sample draws its own ``t`` independently (like a per-sample diffusion timestep),
        so no cross-rank synchronization is needed.
        """
        b = len(target_frames)
        t_idx = torch.randint(0, int(num_frames), (b,)).tolist()
        selected = []
        for sample_idx, t in enumerate(t_idx):
            frames = target_frames[sample_idx]
            frames = list(frames) if isinstance(frames, (list, tuple)) else [frames]
            selected.append([frames[t]])
        frame_index = torch.tensor([(t + 1) / float(num_frames) for t in t_idx], device=device, dtype=torch.float32)
        return selected, frame_index

    def _apply_freezes(self):
        """Apply per-component freeze/unfreeze, called at the end of build().

        Every gen-owned submodule can be frozen/unfrozen from config:
          * ``target.freeze``   (default True — target encoders/VAEs are frozen)
          * ``decoder.freeze``  (default = the decoder's own ``default_freeze``;
            the MLP decoder is trainable by default)
          * ``expert_freeze``   (default False — the generation/diffusion expert)
        Frozen params are naturally excluded from the optimizer (the trainer's
        LR-group builder filters ``requires_grad``).
        """
        if getattr(self, "target", None) is not None:
            set_frozen(self.target, bool(self.gen_cfg.target.get("freeze", True)), "target")
        dec = getattr(self, "decoder", None)
        if dec is not None:
            df = self.gen_cfg.decoder.get("freeze", None)  # None -> decoder default
            if hasattr(dec, "set_freeze"):
                dec.set_freeze(df)
            elif df is not None:
                set_frozen(dec, bool(df), "decoder")
        if self.expert is not None:
            set_frozen(self.expert, bool(self.gen_cfg.get("expert_freeze", False)), "expert")

    def _populate_expert_dit_cfg(self):
        """Populate ``gen_model.expert_dit_cfg`` from the (framework-written) VLM shape.

        Mirrors ``populate_layerwise_dit_cfg`` for the action expert: the
        generation expert is a layer-wise cross-attention DiT with num_vl_layers
        layers at the VLM hidden width.
        """
        from starVLA.model.framework.share_tools import populate_gen_expert_dit_cfg

        q = self._global_config.framework.qwenvl
        populate_gen_expert_dit_cfg(
            self._global_config,
            dit_hidden_dim=int(q.vl_hidden_dim),
            num_dit_layers=int(q.num_vl_layers),
        )

    # ── contract ──
    def build(self, vlm_interface):
        """Mode-specific setup. Called once by the framework after construction."""
        raise NotImplementedError

    def forward(self, *, query_states=None, vl_embs_list=None, target_frames, encoder_attention_mask=None):
        """Return the scalar generation loss.

        The framework passes whatever it has; the head uses only what it needs
        (per its capability flags).
        """
        raise NotImplementedError

    # ── isolation=off support (action expert cross-attends to gen content) ──
    def isolation_context(
        self,
        *,
        query_states=None,
        vl_embs_list=None,
        encoder_attention_mask=None,
        t_override=None,
        target_frames=None,
    ):
        """Per-layer gen states the action expert cross-attends to under ``isolation=off``.

        Returns a list of ``(B, N, H)`` tensors, one per action-DiT layer, or
        ``None`` when this head has no *separate* gen stream to expose — e.g. the
        ``vlm`` backbone, where gen content already lives inside ``vl_embs_list``
        (isolation there is handled by masking, not injection).

        Default (this method): the DETERMINISTIC ``GenerationExpert`` path of the
        ``regression`` head — run the expert ONCE, cache its
        final output for the subsequent loss (consumed via
        ``_consume_cached_condition``), and return the per-layer states. The
        ``diffusion`` head overrides this (its expert needs the noised target and a
        shared timestep). The framework calls this BEFORE the action expert so the
        states can be concatenated into the action expert's cross-attention encoder.
        """
        if self.backbone != "expert" or self.expert is None:
            return None
        with torch.autocast("cuda", dtype=torch.float32):
            final, all_hidden = self.expert(
                vl_embs_list, encoder_attention_mask=encoder_attention_mask, return_all_hidden_states=True
            )
        self._cached_condition = final
        return self._align_gen_layers(all_hidden, len(vl_embs_list))

    def _consume_cached_condition(self):
        """Return and clear the expert output stashed by ``isolation_context``."""
        c = self._cached_condition
        self._cached_condition = None
        return c

    @staticmethod
    def _align_gen_layers(all_hidden_states, num_layers):
        """Align gen-DiT per-layer states to the action DiT's layers.

        ``all_hidden_states`` = ``[input, after_block_0, …, after_block_{L-1}]``
        (length L+1). We keep the POST-block states (drop the raw query/latent
        input), so action layer ``idx`` cross-attends to the gen stream's
        *contextualised* representation at matching depth — mirroring the ``vlm``
        backbone, where the action sees each query's per-VLM-layer (already
        contextualised) hidden state.
        """
        g = list(all_hidden_states[1:])
        assert len(g) == num_layers, (
            f"gen expert exposes {len(g)} layers but the action expert has {num_layers}; "
            "the two DiTs must share depth (populate_gen_expert_dit_cfg mirrors the action DiT)."
        )
        return g
