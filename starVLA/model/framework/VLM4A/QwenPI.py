# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by Jinhui YE / HKUST University] in [2025].
"""
QwenPI framework: a Qwen3.5 VLM extended into a Mixture-of-Transformers.

* Action expert: a layer-wise cross-attention DiT trained with flow matching on action
  chunks; block ``i`` cross-attends to the hidden states of LLM layer ``i``. The
  flow-matching head is adapted from GR00T N1.5; the layer-wise MoT design follows pi_0.
* Generation expert (optional, ``framework.gen_model``): predicts features of a future
  frame produced by a frozen visual encoder, trained with regression or flow matching.
  It is dropped at inference when isolated from the action expert.
"""

from dataclasses import dataclass, field
from numbers import Integral
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.dataloader.preprocessed_batch import unpack_preprocessed_batch
from starVLA.dataloader.view_tiling import (
    normalize_obs_resize_mode,
    normalize_view_prompt_mode,
    normalize_view_tiling_mode,
    resolve_image_size,
)
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import (
    merge_framework_config,
    populate_layerwise_dit_cfg,
    resolve_action_isolation,
)
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import LayerwiseFlowmatchingActionHead, get_action_model
from starVLA.model.modules.gen_model import get_gen_model
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)


def _gen_target_frames(examples: List[dict]) -> List[list]:
    """Per-sample list of future target frames (``gen_target_frames``, else ``[gen_target_image]``)."""
    return [ex["gen_target_frames"] if "gen_target_frames" in ex else [ex["gen_target_image"]] for ex in examples]


# ──────────────────────────────────────────────────────────────────────
#  Default Config for QwenPI
#  - Documents every framework-level parameter with type + description
#  - YAML values override these defaults; extra YAML keys are preserved
# ──────────────────────────────────────────────────────────────────────
@dataclass
class QwenPIDefaultConfig:
    """QwenPI framework default parameters.

    Layer-wise cross-DiT flow-matching action prediction conditioned on
    multi-layer VLM hidden states.  All fields can be overridden by the
    corresponding key in the YAML ``framework:`` section.
    """

    # --- Registry identifier (must match @FRAMEWORK_REGISTRY.register) ---
    name: str = "QwenPI"

    # Freeze the WHOLE VLM (visual + LLM). Default False -> the LLM backbone trains
    # end-to-end (the visual encoder is governed separately by freeze_visual_encoder
    # below). When the generation head uses the ``vlm`` backbone, freezing still keeps
    # the learnable query embeddings trainable. Prefer this over trainer.freeze_modules,
    # which cannot exempt the query embeddings.
    freeze_vlm: bool = False

    # Keep the visual encoder (vision tower + spatial-merge merger) frozen at ALL
    # times, regardless of freeze_vlm, so unfreezing the VLM fine-tunes ONLY the LLM
    # backbone. Default True. This also keeps the ``auto`` generation target (the VLM's
    # own visual encoding of the future frame) a STABLE target rather than a moving one.
    # Set False to also fine-tune the vision tower (only meaningful with freeze_vlm=false).
    freeze_visual_encoder: bool = True

    # === VLM backbone (Qwen3.5) ===
    qwenvl: dict = field(
        default_factory=lambda: {
            # Path to the base VLM checkpoint (local path or HF hub id). The path must
            # contain "Qwen3.5" (see starVLA/model/modules/vlm/__init__.py).
            "base_vlm": "playground/Pretrained_models/Qwen3.5-2B",
            # The Qwen3.5 wrapper always uses "sdpa"; other values are ignored with a warning.
            "attn_implementation": "sdpa",
            # Qwen dynamic-resize budget. Required when any dataset stream uses
            # obs_resize_mode=native.
            "max_pixels": None,
            # VLM hidden dimension (overwritten at runtime from the model config)
            "vl_hidden_dim": 2048,
            # Number of LLM layers (overwritten at runtime from the model config)
            "num_vl_layers": 24,
        }
    )

    # === Action expert (layer-wise flow matching / cross-DiT) ===
    action_model: dict = field(
        default_factory=lambda: {
            # Build the action expert. Set False for a generation-only model.
            "enable": True,
            # Action head architecture type
            "action_model_type": "LayerwiseFM",
            # Dimensionality of each action vector (e.g., 7 for 6-DoF + gripper)
            "action_dim": 7,
            # State dimension (proprioception input; 0 -> no state encoder)
            "state_dim": 7,
            # Chunk length (number of action steps the head predicts). The single source
            # of truth; apply_config_compat derives future_action_window_size from it.
            "action_horizon": 16,
            # Number of (timestep, noise) draws per sample for the flow-matching loss
            "repeated_diffusion_steps": 2,
            # Inference denoising (Euler) steps
            "num_inference_timesteps": 4,
            "add_pos_embed": True,
            "max_seq_len": 1024,
            "num_target_vision_tokens": 32,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            # DiT architecture settings — shape fields (num_layers,
            # input_embedding_dim, cross_attention_dim, num_attention_heads)
            # are auto-populated by populate_layerwise_dit_cfg at runtime.
            "diffusion_model_cfg": {
                "dropout": 0.2,
                "final_dropout": True,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "positional_embeddings": None,
                "attention_head_dim": 64,
            },
        }
    )

    # === Generation expert (optional; default disabled) ===
    # See starVLA/model/modules/gen_model/__init__.py for the full contract.
    #   mode      : regression | diffusion
    #   backbone  : vlm (learnable queries through the VLM) | expert (dedicated gen expert)
    #   targets   : list of {target, decoder, loss, num_gen_queries, weight}
    #     target  : auto | siglip | dino | vjepa | lingbot | pixel | wan_vae
    #     decoder : mlp (regression only)
    #     loss    : mse | cosine | cosine_magnitude | smooth_l1 (regression) | flow_matching (diffusion)
    gen_model: dict = field(
        default_factory=lambda: {
            "enable": False,
            # None -> normalize_gen_config sets "regression".
            "mode": None,
            "backbone": "vlm",
            # Isolation switch — controls whether the ACTION expert can see the
            # generation content ("auto" | true | false):
            #   auto  -> vlm backbone -> coupled; expert backbone (incl. diffusion) -> isolated.
            #   true  -> ISOLATE: action cross-attends to VLM context ONLY; gen
            #            loss influences action only via VLM-param updates.
            #   false -> COUPLE: action additionally cross-attends to the gen
            #            content (vlm: the learnable queries already in the VLM
            #            sequence; expert: the gen expert's per-layer states —
            #            for diffusion, a shared-timestep joint denoise). gen
            #            never sees action tokens (one-directional).
            "isolation": "auto",
            # Visual prediction horizon tau in seconds and the number of future frames
            # predicted within it (frame k at k * tau / T). None aligns the target with
            # the end of the action chunk, tau = action_horizon / datasets.vla_data.dataset_fps.
            # Converted to each data stream's frame rate by dataloader.align_dataset_configs.
            "gen_horizon_seconds": None,
            "num_future_frames": 1,
            # Generation targets — a LIST of per-target specs, each {target, decoder, loss,
            # num_gen_queries, weight}. A single target is a 1-element list. None ->
            # normalize_gen_config synthesises one default (auto target). Per-target
            # defaults live in gen_model/__init__._GEN_TARGET_DEFAULTS.
            # See gen_model/multi.py::MultiGenHead.
            "targets": None,
            # VLM hidden size; auto-populated by the framework at runtime.
            "input_dim": 2048,
            # --- expert backbone / diffusion mode: generation-expert DiT shape ---
            # Shape fields (num_layers, input_embedding_dim, cross_attention_dim,
            # num_attention_heads) are auto-populated at build() via
            # populate_gen_expert_dit_cfg, mirroring the action expert.
            "expert_add_pos_embed": True,
            "expert_freeze": False,  # freeze/unfreeze the generation/diffusion expert
            "max_seq_len": 8192,  # diffusion expert token-position embedding cap
            "num_timestep_buckets": 1000,  # diffusion expert flow-matching schedule
            "expert_dit_cfg": {
                "dropout": 0.2,
                "final_dropout": True,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "positional_embeddings": None,
                "attention_head_dim": 64,
            },
        }
    )


@FRAMEWORK_REGISTRY.register("QwenFM")
@FRAMEWORK_REGISTRY.register("QwenPI")
class Qwen_PI(baseframework):
    """
    Vision-language-action model: Qwen3.5 backbone + layer-wise flow-matching action
    expert, with an optional generation expert (see the module docstring).

    Focus: predict future continuous actions conditioned on images + instruction.
    """

    supports_fused_cotrain_forward = True

    def _validate_native_observation_budget(self) -> None:
        """Fail before model loading if native observations have no enforced budget."""
        datasets = getattr(self.config, "datasets", None)
        if datasets is None:
            return
        native_streams = []
        for name, block in datasets.items():
            if name in ("shared", "total_per_device_batch_size") or not hasattr(block, "get"):
                continue
            if normalize_obs_resize_mode(block.get("obs_resize_mode", None)) == "native":
                native_streams.append(name)
        if not native_streams:
            return

        qwenvl_cfg = self.config.framework.qwenvl
        max_pixels = qwenvl_cfg.get("max_pixels", None)
        if isinstance(max_pixels, bool) or not isinstance(max_pixels, Integral) or max_pixels <= 0:
            raise ValueError(
                f"obs_resize_mode='native' for streams {native_streams} requires a positive integer "
                f"framework.qwenvl.max_pixels, got {max_pixels!r}."
            )
        base_vlm = str(qwenvl_cfg.get("base_vlm", ""))
        if "qwen3.5" not in base_vlm.lower():
            raise ValueError(
                "obs_resize_mode='native' currently has a processor-side max_pixels "
                f"implementation only for Qwen3.5, got framework.qwenvl.base_vlm={base_vlm!r}."
            )

    def _configured_view_prompt_mode(self) -> str:
        """Resolve (and validate) the model-wide observation prompt mode from the dataset config."""
        datasets_cfg = getattr(self.config, "datasets", None)
        shared_cfg = datasets_cfg.get("shared", {}) if datasets_cfg is not None and hasattr(datasets_cfg, "get") else {}
        vla_cfg = datasets_cfg.get("vla_data", {}) if datasets_cfg is not None and hasattr(datasets_cfg, "get") else {}
        configured_mode = vla_cfg.get("view_prompt_mode", None) if hasattr(vla_cfg, "get") else None
        if configured_mode is None and hasattr(shared_cfg, "get"):
            configured_mode = shared_cfg.get("view_prompt_mode", None)
        return normalize_view_prompt_mode(configured_mode)

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """

        super().__init__()
        # Merge framework defaults with YAML config (YAML wins on conflicts)
        self.config = merge_framework_config(QwenPIDefaultConfig, config)
        self._validate_native_observation_budget()
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        self._configured_view_prompt_mode()
        if not bool(getattr(self.qwen_vl_interface, "supports_view_role_prompts", False)):
            raise ValueError(
                "The role-aware observation contract requires a VLM interface that "
                "consumes and strictly aligns view_texts with images; "
                f"{type(self.qwen_vl_interface).__name__} does not declare "
                "supports_view_role_prompts=True."
            )

        # Read the actual hidden size and layer count from the loaded VLM.
        # `output_hidden_states=True` returns (num_hidden_layers + 1) tensors;
        # we keep the last num_hidden_layers of them, so DiT depth must match.
        # Qwen3.5 stores num_hidden_layers under text_config; the getattr fallback
        # also handles configs that keep it at the top level.
        vlm_hf_cfg = self.qwen_vl_interface.model.config
        text_cfg = getattr(vlm_hf_cfg, "text_config", vlm_hf_cfg)
        num_vl_layers = int(text_cfg.num_hidden_layers)
        llm_hidden_size = int(vlm_hf_cfg.hidden_size)
        self.config.framework.qwenvl.vl_hidden_dim = llm_hidden_size
        self.config.framework.qwenvl.num_vl_layers = num_vl_layers

        # The action DiT runs at the LLM hidden size (no compression); the head itself
        # does not look at qwenvl.*. `action_horizon` is the single source of truth for
        # the chunk length (available even when the action expert is disabled).
        self.action_horizon = int(self.config.framework.action_model.action_horizon)

        # ── Action expert (default enabled; action_model.enable=false builds a
        #    generation-only model without allocating an unused DiT). ──
        self.action_enabled = bool(self.config.framework.action_model.get("enable", True))
        self.action_model: LayerwiseFlowmatchingActionHead = None
        if self.action_enabled:
            populate_layerwise_dit_cfg(
                self.config,
                dit_hidden_dim=llm_hidden_size,
                num_dit_layers=num_vl_layers,
            )
            self.action_model = get_action_model(config=self.config)

        # ── Optional generation expert (default disabled) ──
        gen_cfg = getattr(self.config.framework, "gen_model", None)
        self.gen_enabled = bool(gen_cfg.get("enable", False)) if gen_cfg is not None else False
        self.gen_model = None
        # Isolation state (see gen_model.isolation). _isolate_action=True -> the
        # action expert cannot see gen content. Defaults keep the framework
        # gen-agnostic when disabled.
        self._isolate_action = False
        self._gen_mode = None
        if self.gen_enabled:
            # Tell the head the VLM hidden size (its decoder / expert run at this
            # width). qwenvl.vl_hidden_dim / num_vl_layers were written above, so the
            # head's build() can populate its own expert DiT shape from them.
            self.config.framework.gen_model.input_dim = llm_hidden_size
            # Build the mode-dispatched head, then let it wire itself up:
            # resolve num_gen_queries, populate + construct the generation expert,
            # and/or register learnable query tokens on the VLM — whatever the
            # chosen (mode, backbone) needs. The framework stays mode-agnostic.
            self.gen_model = get_gen_model(config=self.config, vlm_interface=self.qwen_vl_interface)
            self.gen_model.build(self.qwen_vl_interface)

            # Resolve the isolation switch now that the head (and its backbone) is
            # built. resolve_action_isolation is the single source of truth (shared with
            # from_pretrained); pass the BUILT head's backbone since diffusion mode
            # forces 'expert' regardless of the YAML value.
            self._gen_mode = str(self.config.framework.gen_model.get("mode", "regression")).lower()
            raw_iso = self.config.framework.gen_model.get("isolation", "auto")
            backbone = str(self.gen_model.backbone).lower()
            self._isolate_action = resolve_action_isolation(self.config.framework.gen_model, backbone=backbone)
            logger.info(
                f"[QwenPI] gen isolation = {'ON (action cannot see gen content)' if self._isolate_action else 'OFF (action sees gen content)'} "
                f"[mode={self._gen_mode}, backbone={backbone}, isolation={raw_iso!r}]"
            )

        # ── VLM freezing (two orthogonal YAML switches) ──
        # framework.freeze_vlm (default False): freeze the WHOLE VLM (visual + LLM).
        #   Only when the head routes learnable query tokens THROUGH the VLM
        #   (requires_vlm_queries) are the query embeddings re-enabled — which
        #   trainer.freeze_modules cannot express. With an expert/diffusion backbone
        #   there are no VLM-side queries; the expert lives in self.gen_model and stays
        #   trainable even while the VLM is frozen.
        # framework.freeze_visual_encoder (default True): keep the visual encoder
        #   (vision tower + merger) frozen REGARDLESS of freeze_vlm, so unfreezing the
        #   VLM fine-tunes ONLY the LLM backbone. This also keeps the `auto` gen target
        #   (the VLM's own visual encoding of the future frame) a STABLE target.
        if bool(getattr(self.config.framework, "freeze_vlm", False)):
            self.qwen_vl_interface.requires_grad_(False)
            if self.gen_enabled and self.gen_model.requires_vlm_queries:
                self.qwen_vl_interface.set_gen_query_trainable()
        if bool(getattr(self.config.framework, "freeze_visual_encoder", True)):
            if hasattr(self.qwen_vl_interface, "freeze_visual_encoder"):
                self.qwen_vl_interface.freeze_visual_encoder()
            else:
                logger.warning(
                    "framework.freeze_visual_encoder=True but the VLM wrapper "
                    f"{type(self.qwen_vl_interface).__name__} has no freeze_visual_encoder(); "
                    "the visual encoder will train. Add the method or set freeze_visual_encoder=false."
                )

    def _resolve_view_texts(self, examples: List[dict], *, phase: str):
        """Require role descriptions and enforce the checkpointed inference contract."""
        if not examples:
            raise ValueError(f"{phase}: examples must be a non-empty list")

        datasets_cfg = getattr(self.config, "datasets", None)
        vla_cfg = datasets_cfg.get("vla_data", {}) if datasets_cfg is not None and hasattr(datasets_cfg, "get") else {}
        self._configured_view_prompt_mode()

        view_texts = [example.get("view_texts") for example in examples]
        missing = [index for index, texts in enumerate(view_texts) if texts is None]
        if missing:
            raise ValueError(
                f"{phase}: every sample must provide role-aware view_texts aligned "
                f"1:1 with its image list; missing samples={missing}. Images-only "
                "prompts are not supported."
            )

        for index, (example, texts) in enumerate(zip(examples, view_texts, strict=True)):
            images = example.get("image")
            if not isinstance(images, (list, tuple)) or not images:
                raise ValueError(
                    f"{phase}: sample {index} must provide a non-empty image list, " f"got {type(images).__name__}."
                )
            if not isinstance(texts, (list, tuple)):
                raise ValueError(f"{phase}: view_texts[{index}] must be a list/tuple, got " f"{type(texts).__name__}.")
            if len(texts) != len(images):
                raise ValueError(
                    f"{phase}: view_texts[{index}] must align 1:1 with image[{index}]: "
                    f"texts={len(texts)}, images={len(images)}."
                )
            invalid = [
                text_index for text_index, text in enumerate(texts) if not isinstance(text, str) or not text.strip()
            ]
            if invalid:
                raise ValueError(
                    f"{phase}: view_texts[{index}] contains non-string/empty entries " f"at indices {invalid}."
                )

        if phase.startswith("predict_action"):
            contract = vla_cfg.get("resolved_view_prompt", None) if hasattr(vla_cfg, "get") else None
            if contract is None or not hasattr(contract, "get"):
                raise ValueError(
                    f"{phase}: role-aware inference requires the checkpointed "
                    "datasets.vla_data.resolved_view_prompt contract, but it is "
                    f"missing or malformed ({contract!r}). The training dataloader "
                    "writes it into the run config; camera roles are never guessed."
                )
            if int(contract.get("version", 0)) != 1:
                raise ValueError(
                    f"{phase}: unsupported resolved_view_prompt version " f"{contract.get('version', None)!r}."
                )
            if contract.get("mode", None) != "role":
                raise ValueError(
                    f"{phase}: resolved_view_prompt.mode must be 'role', got " f"{contract.get('mode', None)!r}."
                )
            expected_tiling = normalize_view_tiling_mode(vla_cfg.get("view_tiling", None))
            contract_tiling = normalize_view_tiling_mode(contract.get("view_tiling", None))
            if contract_tiling != expected_tiling:
                raise ValueError(
                    f"{phase}: checkpointed prompt tiling {contract_tiling!r} "
                    f"does not match datasets.vla_data.view_tiling="
                    f"{expected_tiling!r}."
                )
            expected_texts = list(contract.get("texts", []))
            expected_roles = list(contract.get("image_roles", []))
            if not expected_texts or len(expected_texts) != len(expected_roles):
                raise ValueError(
                    f"{phase}: resolved_view_prompt must contain one non-empty "
                    "text per image role; "
                    f"texts={expected_texts!r}, image_roles={expected_roles!r}."
                )
            mismatched = [index for index, texts in enumerate(view_texts) if list(texts) != expected_texts]
            if mismatched:
                raise ValueError(
                    f"{phase}: view_texts do not match the checkpointed "
                    f"inference contract for samples {mismatched}; expected "
                    f"{expected_texts!r}."
                )
        return view_texts

    def _encode_vl_hidden_states(
        self,
        batch_images: List,
        instructions: List[str],
        *,
        view_texts: List,
        append_gen: bool = True,
        qwen_inputs=None,
    ) -> tuple:
        """Run QwenVL and return (layer-wise hidden states, attention_mask, query_states).

        With the "vlm" generation backbone, the learnable query tokens are
        appended to the VLM input, so ``query_states`` is the final-layer hidden
        states at the query positions.  ``append_gen=False`` suppresses the append
        (used at inference under ``isolation=on``, so the action path sees exactly
        the baseline context — identical architecture to a no-gen baseline).  With
        the "expert" backbone (or when generation is disabled) the VLM input is
        left untouched and ``query_states`` is None — the generation expert
        produces its own query states from ``vl_embs_list``.

        ``view_texts``: required per-sample view-role descriptions interleaved before
        each image.
        """
        if qwen_inputs is None:
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                images=batch_images, instructions=instructions, view_texts=view_texts
            )
        else:
            # append_gen_queries replaces mapping entries in place. Copy the
            # worker-produced mapping so the raw batch remains an immutable audit
            # record and can still be compared against main-process preprocessing.
            qwen_inputs = self.qwen_vl_interface.move_qwenvl_inputs(dict(qwen_inputs))
        gen_via_vlm = self.gen_enabled and self.gen_model.requires_vlm_queries and append_gen
        if gen_via_vlm:
            qwen_inputs = self.qwen_vl_interface.append_gen_queries(qwen_inputs)
        attention_mask = qwen_inputs.get("attention_mask", None)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # The action DiT has one cross-attention block per kept LLM layer.
            if self.action_model is not None:
                expected_layers = len(self.action_model.model.transformer_blocks)
                vl_embs_list = list(qwenvl_outputs.hidden_states[-expected_layers:])
            else:
                # Generation-only model (action_model.enable=false). The vlm-backbone gen
                # head reads query_states, not vl_embs_list, so an empty list is safe. An
                # expert-backbone gen head DOES consume vl_embs_list; reject that
                # unsupported combination loudly rather than feed it an empty context.
                if self.gen_enabled and getattr(self.gen_model, "requires_vl_embs", False):
                    raise RuntimeError(
                        "expert-backbone generation needs an action expert to size "
                        "vl_embs_list, but none is built. Use framework.gen_model.backbone=vlm "
                        "for a pure-generation (action-expert-free) run."
                    )
                vl_embs_list = []
            query_states = (
                self.qwen_vl_interface.extract_query_states(qwenvl_outputs.hidden_states[-1]) if gen_via_vlm else None
            )
        return vl_embs_list, attention_mask, query_states

    # ── isolation encoder helpers (no new parameters; attention wiring only) ──
    def _augment_action_encoder(self, vl_embs_list, base_mask, G_list, reps):
        """isolation=off (expert backbone): concatenate the generation expert's
        per-layer states ``G_list`` into the action expert's cross-attention
        encoder and extend the mask with N ones (the gen tokens are always valid).
        Repeats by ``reps`` to match the action head's ``repeated_diffusion_steps``.
        Returns ``(aug_vl_list, aug_mask)``.

        ``G_list`` comes in one of two batch shapes, both yielding rows in
        ``h.repeat(reps)`` (copy-major ``[ex_0..ex_{B-1}, ex_0..ex_{B-1}, …]``) order:
          * batch == ``B`` — a DETERMINISTIC gen expert (timestep-independent, i.e.
            regression): the SAME gen states are reused by every action-
            noise copy, so concat then repeat by ``reps``.
          * batch == ``reps*B`` — the DIFFUSION joint path: each copy already has its
            OWN gen states (the future frame denoised at that copy's shared timestep),
            so repeat the VLM context by ``reps`` and concat per row (no extra repeat).
        """
        N = G_list[0].shape[1]
        b_vl, b_g = vl_embs_list[0].shape[0], G_list[0].shape[0]
        assert b_g in (b_vl, reps * b_vl), (
            f"gen states batch {b_g} must equal the VLM batch {b_vl} (deterministic gen) "
            f"or reps*batch {reps * b_vl} (diffusion joint)."
        )
        per_copy_gen = reps != 1 and b_g == reps * b_vl  # each action-noise copy has its own gen states
        aug_vl = []
        for h, g in zip(vl_embs_list, G_list, strict=False):
            g = g.to(dtype=h.dtype, device=h.device)  # G is fp32; vl_embs is bf16
            if per_copy_gen:
                aug_vl.append(torch.cat([h.repeat(reps, 1, 1), g], dim=1))
            else:
                aug_vl.append(torch.cat([h, g], dim=1).repeat(reps, 1, 1))
        if base_mask is not None:
            ones = torch.ones(base_mask.shape[0], N, dtype=base_mask.dtype, device=base_mask.device)
            aug_mask = torch.cat([base_mask, ones], dim=1).repeat(reps, 1).to(torch.bool)
        else:
            aug_mask = None
        return aug_vl, aug_mask

    def _mask_gen_block(self, base_mask, reps):
        """isolation=on (vlm backbone): zero the trailing ``[boi, gen…, eoi]`` block
        (num_gen_queries + 2 columns) in the action expert's encoder mask, so the
        action expert does not attend to the learnable generation queries even
        though they remain in the VLM sequence (needed for the gen loss).
        """
        n_block = int(self.qwen_vl_interface.num_gen_queries) + 2
        m = base_mask.clone()
        m[:, -n_block:] = 0
        return m.repeat(reps, 1).to(torch.bool)

    def _forward_fused_stream_heads(
        self,
        examples,
        loss_keys,
        vl_embs_list,
        backbone_attention_mask,
        query_states,
    ):
        """Run one stream's heads from a shared Qwen encoding."""

        want_action = self.action_enabled and (loss_keys is None or "action_loss" in loss_keys)
        want_gen = self.gen_enabled and (loss_keys is None or "gen_loss" in loss_keys)
        if not want_action and not want_gen:
            raise RuntimeError(f"Fused cotrain stream requested no available loss from {sorted(loss_keys or [])}.")

        gen_encoder_attention_mask = (
            backbone_attention_mask.to(dtype=torch.bool) if backbone_attention_mask is not None else None
        )
        gen_target_images = _gen_target_frames(examples) if self.gen_enabled else None

        couple = self.gen_enabled and not self._isolate_action
        gen_via_vlm = self.gen_enabled and self.gen_model.requires_vlm_queries
        is_diffusion = self.gen_enabled and self._gen_mode == "diffusion"
        couple_expert = couple and not gen_via_vlm

        losses = {}
        if want_action:
            actions = [example["action"] for example in examples]
            state = [example["state"] for example in examples] if "state" in examples[0] else None
            losses["action_loss"] = self._action_expert_loss(
                actions=actions,
                state=state,
                base_hidden=vl_embs_list[-1],
                vl_embs_list=vl_embs_list,
                backbone_attention_mask=backbone_attention_mask,
                query_states=query_states,
                gen_target_images=gen_target_images,
                gen_encoder_attention_mask=gen_encoder_attention_mask,
                couple_expert=couple_expert,
                is_diffusion=is_diffusion,
                gen_via_vlm=gen_via_vlm,
            )

        if want_gen:
            losses["gen_loss"] = self.gen_model(
                query_states=query_states,
                vl_embs_list=vl_embs_list,
                target_frames=gen_target_images,
                encoder_attention_mask=gen_encoder_attention_mask,
            )
        return losses

    def _forward_fused_cotrain(self, cotrain_batches):
        """Encode preprocessed streams in one Qwen forward, then split their heads."""

        if not cotrain_batches:
            raise ValueError("cotrain_batches must contain at least one (batch, loss_keys) pair.")

        prepared = []
        qwen_batches = []
        combined_examples = []
        for index, item in enumerate(cotrain_batches):
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise TypeError(
                    "Each fused cotrain entry must be a (batch, loss_keys) pair; "
                    f"entry {index} is {type(item).__name__}."
                )
            batch, stream_loss_keys = item
            stream_examples, qwen_inputs = unpack_preprocessed_batch(batch)
            if qwen_inputs is None:
                raise ValueError(
                    "Fused cotrain requires worker-preprocessed Qwen inputs for every stream; "
                    "set datasets.<stream>.worker_preprocess_qwen=true."
                )
            if not isinstance(stream_examples, list) or not stream_examples:
                raise ValueError(f"Fused cotrain stream {index} must contain a non-empty example list.")
            qwen_batch_size = int(qwen_inputs["input_ids"].shape[0])
            if qwen_batch_size != len(stream_examples):
                raise ValueError(
                    f"Fused cotrain stream {index} has {len(stream_examples)} raw examples but "
                    f"Qwen batch {qwen_batch_size}."
                )
            self._resolve_view_texts(stream_examples, phase=f"fused cotrain stream {index}")
            start = len(combined_examples)
            combined_examples.extend(stream_examples)
            prepared.append((stream_examples, stream_loss_keys, start, len(combined_examples)))
            qwen_batches.append(qwen_inputs)

        merged_inputs, sequence_lengths = self.qwen_vl_interface.merge_qwenvl_inputs(qwen_batches)
        vl_embs_list, backbone_attention_mask, query_states = self._encode_vl_hidden_states(
            [example["image"] for example in combined_examples],
            [example["lang"] for example in combined_examples],
            view_texts=[example["view_texts"] for example in combined_examples],
            qwen_inputs=merged_inputs,
        )

        merged_sequence_length = max(sequence_lengths)
        if backbone_attention_mask is not None:
            encoded_sequence_length = int(backbone_attention_mask.shape[1])
        elif vl_embs_list:
            encoded_sequence_length = int(vl_embs_list[-1].shape[1])
        else:
            raise RuntimeError("Fused Qwen encoding returned neither hidden states nor an attention mask.")
        appended_length = encoded_sequence_length - merged_sequence_length
        if appended_length < 0:
            raise RuntimeError(
                f"Encoded Qwen sequence length {encoded_sequence_length} is shorter than merged input "
                f"length {merged_sequence_length}."
            )

        outputs = []
        for (stream_examples, stream_loss_keys, start, end), sequence_length in zip(
            prepared, sequence_lengths, strict=True
        ):
            restored_length = sequence_length + appended_length
            stream_vl_embs = [hidden[start:end, -restored_length:, :] for hidden in vl_embs_list]
            stream_mask = (
                backbone_attention_mask[start:end, -restored_length:] if backbone_attention_mask is not None else None
            )
            stream_queries = query_states[start:end] if query_states is not None else None
            outputs.append(
                self._forward_fused_stream_heads(
                    stream_examples,
                    stream_loss_keys,
                    stream_vl_embs,
                    stream_mask,
                    stream_queries,
                )
            )
        return outputs

    def forward(
        self,
        examples: Optional[List[dict]] = None,
        loss_keys=None,
        cotrain_batches=None,
        **kwargs,
    ) -> Tuple:
        """
        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image] (multi-view, role-ordered)
                - view_texts: List[str] aligned 1:1 with ``image`` (view-role descriptions)
                - lang: str instruction
                - action: np.ndarray or list shaped [T, action_dim] (action batches)
                - state: optional np.ndarray shaped [1, state_dim]
                - gen_target_frames (List[PIL.Image]) or gen_target_image (PIL.Image)
                  when the generation expert is enabled
              or a worker-preprocessed batch (see ``starVLA.dataloader.preprocessed_batch``).
            loss_keys: optional iterable of loss names to compute for THIS batch
                (per-stream loss binding, ``datasets.<stream>.losses``).
                - None (default): ``action_loss`` always, plus ``gen_loss`` when the
                  generation expert is enabled.
                - a set without ``"action_loss"`` (e.g. ``{"gen_loss"}``): an action-free
                  video batch — take the GEN-ONLY path (``_forward_generation_only``).
                - otherwise only the requested losses are returned.
            cotrain_batches: list of ``(preprocessed batch, loss_keys)`` pairs encoded in
                ONE fused VLM forward (``trainer.fuse_joint_vlm_forward``); returns one loss
                dict per stream. Cannot be combined with ``examples`` / ``loss_keys``.
        Returns:
            dict of scalar losses (subset of {action_loss, gen_loss}).
        """
        if cotrain_batches is not None:
            if examples is not None or loss_keys is not None:
                raise ValueError("cotrain_batches cannot be combined with examples or loss_keys.")
            return self._forward_fused_cotrain(cotrain_batches)

        examples, qwen_inputs = unpack_preprocessed_batch(examples)

        # Route by requested losses: want_action -> the action expert consumes
        # example["action"]; want_gen -> the generation expert.
        want_action = self.action_enabled and (loss_keys is None or "action_loss" in loss_keys)
        want_gen = self.gen_enabled and (loss_keys is None or "gen_loss" in loss_keys)

        # No action requested -> pure generation (action-free video) batch. The action
        # expert (which needs example["action"] / ["state"]) is skipped entirely.
        if not want_action:
            return self._forward_generation_only(examples, qwen_inputs=qwen_inputs)

        batch_images = [example["image"] for example in examples]  # [B, [PIL]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        view_texts = self._resolve_view_texts(examples, phase="training forward")
        actions = [example["action"] for example in examples]  # label [B, len, action_dim]

        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]

        # Step 1: encode through QwenVL
        vl_embs_list, backbone_attention_mask, query_states = self._encode_vl_hidden_states(
            batch_images,
            instructions,
            view_texts=view_texts,
            qwen_inputs=qwen_inputs,
        )
        base_hidden = vl_embs_list[-1]

        # Capture the pristine encoder mask for the generation expert BEFORE the
        # action block repeats/reassigns `backbone_attention_mask`. In the "expert"
        # backbone the gen expert cross-attends to the (un-repeated) vl_embs_list,
        # so it needs the matching (un-repeated) mask.
        gen_encoder_attention_mask = (
            backbone_attention_mask.to(dtype=torch.bool) if backbone_attention_mask is not None else None
        )

        # Step 2: generation target frames. Gathered up front because the diffusion
        # joint (isolation=off) path runs the gen expert BEFORE the action expert.
        gen_target_images = _gen_target_frames(examples) if self.gen_enabled else None

        # ── Isolation wiring (no new params; only changes what the action sees). ──
        couple = self.gen_enabled and not self._isolate_action  # action should see gen content
        gen_via_vlm = self.gen_enabled and self.gen_model.requires_vlm_queries
        is_diffusion = self.gen_enabled and self._gen_mode == "diffusion"
        couple_expert = couple and not gen_via_vlm  # inject gen states into the action encoder

        # Step 3: action expert (reps / shared timestep / coupled gen states are wired
        # in _action_expert_loss).
        losses = {
            "action_loss": self._action_expert_loss(
                actions=actions,
                state=state,
                base_hidden=base_hidden,
                vl_embs_list=vl_embs_list,
                backbone_attention_mask=backbone_attention_mask,
                query_states=query_states,
                gen_target_images=gen_target_images,
                gen_encoder_attention_mask=gen_encoder_attention_mask,
                couple_expert=couple_expert,
                is_diffusion=is_diffusion,
                gen_via_vlm=gen_via_vlm,
            )
        }

        # Step 4: generation expert (optional). The head reuses any output cached by
        # isolation_context inside _action_expert_loss, so the gen expert runs once.
        if want_gen:
            gen_loss = self.gen_model(
                query_states=query_states,
                vl_embs_list=vl_embs_list,
                target_frames=gen_target_images,
                encoder_attention_mask=gen_encoder_attention_mask,
            )
            losses["gen_loss"] = gen_loss

        # Per-stream loss binding: when an explicit loss set was requested, return
        # only those keys (loss_keys=None keeps the full dict).
        if loss_keys is not None:
            losses = {k: v for k, v in losses.items() if k in loss_keys}

        return losses

    def _action_expert_loss(
        self,
        *,
        actions,
        state,
        base_hidden,
        vl_embs_list,
        backbone_attention_mask,
        query_states,
        gen_target_images,
        gen_encoder_attention_mask,
        couple_expert: bool,
        is_diffusion: bool,
        gen_via_vlm: bool,
    ):
        """Flow-matching loss of the action expert, including the isolation/coupling
        wiring (``reps`` copies, shared diffusion timestep, coupled gen states)."""
        expert = self.action_model
        horizon = self.action_horizon
        # `repeated_diffusion_steps`: number of (timestep, noise) draws per sample.
        reps = max(1, int(expert.config.get("repeated_diffusion_steps", 2)))
        shared_t = None
        G_list = None

        # When coupling an expert-backbone gen head, run the gen expert ONCE to obtain
        # its per-layer states (G_list); the head caches its output so the gen-loss
        # forward reuses it (no double compute).
        if couple_expert:
            if is_diffusion:
                # Diffusion joint (isolation=off): gen & action share ONE timestep per row.
                B = base_hidden.shape[0]
                shared_t = expert.sample_time(reps * B, device=base_hidden.device, dtype=torch.float32)
                gen_vl_embs = [h.repeat(reps, 1, 1) for h in vl_embs_list]
                gen_mask = gen_encoder_attention_mask.repeat(reps, 1) if gen_encoder_attention_mask is not None else None
            else:
                # Deterministic gen expert (regression): timestep-independent.
                gen_vl_embs = vl_embs_list
                gen_mask = gen_encoder_attention_mask
            G_list = self.gen_model.isolation_context(
                query_states=query_states,
                vl_embs_list=gen_vl_embs,
                encoder_attention_mask=gen_mask,
                t_override=shared_t,
                target_frames=gen_target_images,
            )
            if G_list is None:
                raise NotImplementedError(
                    f"isolation=off (coupled) unsupported for this gen head "
                    f"(mode={self._gen_mode}, backbone={self.gen_model.backbone}): "
                    "it exposes no per-layer generation states for the action expert."
                )

        with torch.autocast("cuda", dtype=torch.float32):
            # Label alignment: take the last `horizon` steps of the action chunk.
            acts = torch.tensor(np.array(actions), device=base_hidden.device, dtype=base_hidden.dtype)
            actions_target = acts[:, -horizon:, :]  # (B, horizon, action_dim)
            actions_target_repeated = actions_target.repeat(reps, 1, 1)

            state_repeated = None
            if state is not None:
                st = torch.tensor(np.array(state), device=base_hidden.device, dtype=base_hidden.dtype)
                state_repeated = st.repeat(reps, 1, 1)

            # Build the action expert's per-layer encoder + mask for this isolation mode.
            if couple_expert:
                vl_embs_for_action, action_mask = self._augment_action_encoder(
                    vl_embs_list, backbone_attention_mask, G_list, reps
                )
            else:
                vl_embs_for_action = [h.repeat(reps, 1, 1) for h in vl_embs_list]
                if gen_via_vlm and self._isolate_action:
                    action_mask = self._mask_gen_block(backbone_attention_mask, reps)
                elif backbone_attention_mask is not None:
                    action_mask = backbone_attention_mask.repeat(reps, 1).to(dtype=torch.bool)
                else:
                    action_mask = None

            return expert(
                vl_embs_for_action,
                actions_target_repeated,
                state_repeated,
                encoder_attention_mask=action_mask,
                t_override=(shared_t if (couple_expert and is_diffusion) else None),
            )

    def _forward_generation_only(self, examples: List[dict], qwen_inputs=None) -> dict:
        """Generation-loss-only forward for action-free (pure-video) batches.

        Mirrors the generation branch of :meth:`forward` but skips the action
        expert entirely: the batch carries ``image`` + ``view_texts`` + ``lang`` +
        ``gen_target_frames`` (see ``video_datasets``) but NO ``action`` / ``state``.
        This is how action-free egocentric video supervises the generation expert
        alongside robot data.  Requires ``gen_model.enable=true``.
        """
        if not self.gen_enabled:
            raise RuntimeError(
                "forward(loss_keys without 'action_loss') requires framework.gen_model.enable=true; "
                "a pure-video batch has no action to compute an action_loss from."
            )

        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        view_texts = self._resolve_view_texts(
            examples,
            phase="generation-only training forward",
        )

        # Same VLM encode as the action path (append_gen queries when the gen head
        # routes them through the VLM). The gen head predicts the future frame(s)
        # conditioned on the current frame + instruction — no action involved.
        vl_embs_list, backbone_attention_mask, query_states = self._encode_vl_hidden_states(
            batch_images,
            instructions,
            view_texts=view_texts,
            qwen_inputs=qwen_inputs,
        )
        gen_encoder_attention_mask = (
            backbone_attention_mask.to(dtype=torch.bool) if backbone_attention_mask is not None else None
        )

        gen_loss = self.gen_model(
            query_states=query_states,
            vl_embs_list=vl_embs_list,
            target_frames=_gen_target_frames(examples),
            encoder_attention_mask=gen_encoder_attention_mask,
        )
        return {"gen_loss": gen_loss}

    def _prepare_inference_images(self, examples: List[dict]):
        """Apply the checkpoint's observation contract for both inference APIs.

        ``fixed`` cover-resizes each image to ``obs_image_size``. ``native`` leaves
        decoded images untouched so the Qwen processor performs its bounded dynamic
        resize. Tiled checkpoints accept one already-composited image per sample;
        accepting raw multi-camera lists here would silently change the number and
        spatial semantics of visual tokens relative to training.
        """
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        data_cfg = self.config.datasets.vla_data
        obs_resize_mode = normalize_obs_resize_mode(data_cfg.get("obs_resize_mode", None))
        view_tiling = normalize_view_tiling_mode(data_cfg.get("view_tiling", None))

        if view_tiling == "tile":
            # The role 2x2 obs canvas is (2*cell, 2*cell); cell defaults to obs_image_size
            # (fixed) else 256 — matching the dataloader (which nulls obs_size in native).
            obs_size = resolve_image_size(data_cfg.get("obs_image_size"), None) if obs_resize_mode == "fixed" else None
            cell = resolve_image_size(data_cfg.get("tile_cell_size"), obs_size or (256, 256))
            expected_size = (2 * cell[0], 2 * cell[1])
            for index, images in enumerate(batch_images):
                if not isinstance(images, (list, tuple)) or len(images) != 1:
                    count = len(images) if isinstance(images, (list, tuple)) else "non-list"
                    raise ValueError(
                        "A view_tiling='tile' checkpoint expects one pre-composited mosaic "
                        f"per sample at inference, but sample {index} contains {count} image(s). "
                        "Apply the training tile recipe before predict_action."
                    )
                if images[0].size != expected_size:
                    raise ValueError(
                        "A view_tiling='tile' checkpoint received a mosaic with the wrong geometry: "
                        f"sample {index} is {images[0].size}, expected {expected_size} (2x2 of cell={cell})."
                    )

        # Fixed, NON-tiled obs: cover+center-crop each image to obs_image_size (matches the
        # dataloader). Tiled checkpoints keep the pre-composited mosaic — its geometry was
        # already validated above.
        if obs_resize_mode == "fixed" and view_tiling == "off":
            target_size = resolve_image_size(data_cfg.get("obs_image_size", None), (224, 224))
            batch_images = resize_images(batch_images, target_size=target_size)
        return batch_images

    @torch.inference_mode()
    def predict_action(
        self,
        examples: Optional[List[dict]] = None,
        **kwargs,
    ) -> dict:
        """
        Inference: predict a normalized action chunk.

        Steps:
          1. Apply the checkpoint's observation contract (resize / tiling checks)
          2. Encode with QwenVL (layer-wise hidden states retained)
          3. Integrate the action expert's flow-matching ODE with
             ``num_inference_timesteps`` Euler steps (jointly with the generation
             expert only under ``isolation=false`` with diffusion)

        Args:
            examples: list of dicts with ``image`` / ``view_texts`` / ``lang`` and optional
                ``state``; ``view_texts`` must equal the checkpointed
                ``datasets.vla_data.resolved_view_prompt.texts``.

        Returns:
            dict:
                normalized_actions (np.ndarray): shape [B, T, action_dim], normalized actions.
        """
        if not self.action_enabled:
            raise RuntimeError(
                "predict_action requires the action expert (framework.action_model.enable=true); "
                "this model was built without it, so action inference is unavailable."
            )
        examples, qwen_inputs = unpack_preprocessed_batch(examples)
        if type(examples) is not list:
            examples = [examples]

        batch_images = self._prepare_inference_images(examples)  # [B, List[PIL]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        view_texts = self._resolve_view_texts(examples, phase="predict_action")

        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]

        # Isolation at inference: under isolation=on with the vlm backbone we do
        # NOT append the learnable queries, so the action path sees exactly the
        # baseline VLM context (identical architecture to a no-gen baseline).
        append_gen = not (self.gen_enabled and self.gen_model.requires_vlm_queries and self._isolate_action)

        # Step 1: encode through QwenVL
        vl_embs_list, backbone_attention_mask, _ = self._encode_vl_hidden_states(
            batch_images,
            instructions,
            append_gen=append_gen,
            view_texts=view_texts,
            qwen_inputs=qwen_inputs,
        )
        base_hidden = vl_embs_list[-1]
        if backbone_attention_mask is not None:
            backbone_attention_mask = backbone_attention_mask.to(dtype=torch.bool)

        state = (
            torch.from_numpy(np.array(state)).to(base_hidden.device, dtype=base_hidden.dtype)
            if state is not None
            else None
        )

        couple = self.gen_enabled and not self._isolate_action
        gen_via_vlm = self.gen_enabled and self.gen_model.requires_vlm_queries
        is_diffusion = self.gen_enabled and self._gen_mode == "diffusion"
        couple_expert = couple and not gen_via_vlm  # action must see the gen expert's states

        # Step 2: action expert
        with torch.autocast("cuda", dtype=torch.float32):
            if couple_expert and is_diffusion:
                # isolation=off (diffusion): joint Euler denoising of action + future frame.
                pred_actions = self._predict_action_joint(vl_embs_list, backbone_attention_mask, state)
            else:
                vl_for_action, action_mask = vl_embs_list, backbone_attention_mask
                if couple_expert:
                    # isolation=off (deterministic expert): inject the (fixed) gen states.
                    G_list = self.gen_model.isolation_context(
                        vl_embs_list=vl_embs_list, encoder_attention_mask=backbone_attention_mask
                    )
                    self.gen_model._consume_cached_condition()  # drop the cached loss/condition (unused at inference)
                    vl_for_action, action_mask = self._augment_action_encoder(
                        vl_embs_list, backbone_attention_mask, G_list, reps=1
                    )
                pred_actions = self.action_model.predict_action(
                    vl_for_action, state, encoder_attention_mask=action_mask
                )  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}

    @torch.inference_mode()
    def _predict_action_joint(self, vl_embs_list, base_mask, state):
        """Diffusion ``isolation=off`` inference: joint Euler denoising of the
        future-frame latent and the action chunk on a SHARED timestep grid.

        Each step: (1) the generation expert denoises the future frame (conditioned
        on the FIXED VLM context) and exposes its per-layer states; (2) the action
        expert cross-attends to ``[VLM context + those states]`` and predicts its
        velocity; (3) both latents are Euler-stepped. The gen expert never sees the
        action tokens (one-directional), so it is computed first each step. This
        mirrors the training coupling, keeping train/inference consistent.
        """
        am = self.action_model
        device = vl_embs_list[0].device
        num_steps = am.num_inference_timesteps
        dt = 1.0 / num_steps

        # The gen expert never sees the action, so its trajectory is independent —
        # precompute the per-step generation states once, then denoise the action
        # on the SAME timestep grid, cross-attending to [context + gen states].
        G_traj = self.gen_model.denoise_gen_trajectory(vl_embs_list, base_mask, num_steps, device)

        actions = torch.randn(
            vl_embs_list[0].shape[0],
            am.action_horizon,
            am.action_dim,
            dtype=vl_embs_list[0].dtype,
            device=device,
        )
        state_features = am.state_encoder(state) if state is not None else None
        for step in range(num_steps):
            aug_vl, aug_mask = self._augment_action_encoder(vl_embs_list, base_mask, G_traj[step], reps=1)
            t_bucket = int(step / float(num_steps) * am.num_timestep_buckets)
            v_act = am.action_denoise_step(aug_vl, actions, t_bucket, state_features, encoder_attention_mask=aug_mask)
            actions = actions + dt * v_act
        return actions

    # NOTE: torch.no_grad (NOT inference_mode) — the default RTC mode "pigdm" needs
    # autograd (torch.enable_grad + autograd.grad for its pseudo-inverse guidance),
    # which inference_mode forbids. no_grad lets that section re-enable grad locally.
    @torch.no_grad()
    def predict_action_realtime(
        self,
        examples: Optional[List[dict]] = None,
        prev_action_chunk_normalized: Optional[np.ndarray] = None,
        inference_delay: int = 1,
        **kwargs,
    ) -> dict:
        """RTC-aware inference: condition sampling on a known prefix.

        Fixes a known prefix of the action chunk to the previous prediction
        and resamples the rest, so chunks splice smoothly under execution
        latency.  Mode / schedule kwargs are forwarded to the action head's
        ``predict_action_realtime``.

        Args:
            examples: list of dict, same shape as ``predict_action``.
            prev_action_chunk_normalized: (B, T, action_dim) *normalized*
                continuous actions from the previous prediction.
            inference_delay: number of leading timesteps to keep as prefix.
            **kwargs: forwarded to ``self.action_model.predict_action_realtime``
                (e.g. ``mode``, ``suffix_length``, ``prefix_attention_schedule``,
                ``max_guidance_weight``).

        Returns:
            dict with key ``"normalized_actions"`` -> np.ndarray of shape
            ``(B, T, action_dim)``.
        """
        if prev_action_chunk_normalized is None or inference_delay <= 0:
            return self.predict_action(examples)

        examples, qwen_inputs = unpack_preprocessed_batch(examples)
        if type(examples) is not list:
            examples = [examples]

        batch_images = self._prepare_inference_images(examples)
        instructions = [example["lang"] for example in examples]
        view_texts = self._resolve_view_texts(
            examples,
            phase="predict_action_realtime",
        )
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        # Isolation: skip appending the learnable queries under isolation=on (vlm).
        append_gen = not (self.gen_enabled and self.gen_model.requires_vlm_queries and self._isolate_action)
        vl_embs_list, backbone_attention_mask, _ = self._encode_vl_hidden_states(
            batch_images,
            instructions,
            append_gen=append_gen,
            view_texts=view_texts,
            qwen_inputs=qwen_inputs,
        )
        base_hidden = vl_embs_list[-1]

        # isolation=off: the action expert must additionally see the gen states.
        couple_expert = self.gen_enabled and not self._isolate_action and not self.gen_model.requires_vlm_queries
        vl_embs_per_step = None
        rt_encoder_mask = None
        if couple_expert:
            if backbone_attention_mask is not None:
                backbone_attention_mask = backbone_attention_mask.to(dtype=torch.bool)
            # fp32 autocast so the gen expert (fp32) consumes the bf16 vl_embs cleanly.
            with torch.autocast("cuda", dtype=torch.float32):
                if self._gen_mode == "diffusion":
                    # Joint denoising: the gen trajectory is independent of the (RTC-guided)
                    # action, so precompute it on the shared grid and feed the action a
                    # per-step augmented encoder ([context + gen states] at that step).
                    num_steps = self.action_model.num_inference_timesteps
                    G_traj = self.gen_model.denoise_gen_trajectory(
                        vl_embs_list, backbone_attention_mask, num_steps, base_hidden.device
                    )
                    vl_embs_per_step = []
                    for G in G_traj:
                        avl, rt_encoder_mask = self._augment_action_encoder(
                            vl_embs_list, backbone_attention_mask, G, reps=1
                        )
                        vl_embs_per_step.append(avl)
                else:
                    # Deterministic gen expert: fixed states, same augmented encoder every step.
                    G_list = self.gen_model.isolation_context(
                        vl_embs_list=vl_embs_list, encoder_attention_mask=backbone_attention_mask
                    )
                    self.gen_model._consume_cached_condition()
                    vl_embs_list, rt_encoder_mask = self._augment_action_encoder(
                        vl_embs_list, backbone_attention_mask, G_list, reps=1
                    )

        state_t = (
            torch.from_numpy(np.array(state)).to(base_hidden.device, dtype=base_hidden.dtype)
            if state is not None
            else None
        )

        prev_chunk_t = torch.from_numpy(np.array(prev_action_chunk_normalized)).to(
            base_hidden.device, dtype=torch.float32
        )

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action_realtime(
                vl_embs_list,
                state_t,
                prev_action_chunk=prev_chunk_t,
                inference_delay=inference_delay,
                vl_embs_per_step=vl_embs_per_step,
                encoder_attention_mask=rt_encoder_mask,
                **kwargs,
            )

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}


if __name__ == "__main__":
    # Smoke test: build QwenPI from a training config, run one training forward (and a
    # generation-only forward when the generation expert is enabled) on fake role-aware
    # samples, then one predict_action call. Needs the pretrained weights referenced by
    # the config but no dataset. Run from the repo root:
    #   python starVLA/model/framework/VLM4A/QwenPI.py --config_yaml configs/pretrain/cotrain_openaoe.yaml
    import argparse

    from omegaconf import OmegaConf

    from starVLA.dataloader import align_dataset_configs
    from starVLA.dataloader.view_tiling import build_view_prompt_contract, publish_view_prompt_contract, tile_roles
    from starVLA.model.framework.share_tools import apply_config_compat, load_config_with_bases
    from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="configs/pretrain/cotrain_openaoe.yaml",
        help="Path to a training YAML (``_base_`` includes are resolved; extra --a.b value overrides are applied).",
    )
    args, clipargs = parser.parse_known_args()

    # Same config pipeline as the training entry points.
    cfg = load_config_with_bases(args.config_yaml)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(normalize_dotlist_args(clipargs)))
    cfg = apply_config_compat(cfg)
    cfg = align_dataset_configs(cfg)

    # The robot dataloader normally writes the role-prompt contract into the config;
    # publish an equivalent one for a two-camera robot (third-person + wrist).
    vla_cfg = cfg.datasets.vla_data
    role_descriptions = {
        "central_global": "primary global view from a fixed exterior camera",
        "primary_effector": "local effector view from a wrist camera",
    }
    view_tiling = normalize_view_tiling_mode(vla_cfg.get("view_tiling", None))
    publish_view_prompt_contract(
        vla_cfg, build_view_prompt_contract(role_descriptions, view_tiling), context="QwenPI smoke test"
    )
    view_texts = list(vla_cfg.resolved_view_prompt.texts)

    model = Qwen_PI(cfg)
    print(model)

    def _fake_image(size=(224, 224)):
        return Image.fromarray(np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8))

    # Observation images must match the checkpointed contract: one image per role
    # (view_tiling=off) or one pre-composited 2x2 mosaic (view_tiling=tile).
    if view_tiling == "tile":
        fixed = normalize_obs_resize_mode(vla_cfg.get("obs_resize_mode", None)) == "fixed"
        obs_size = resolve_image_size(vla_cfg.get("obs_image_size"), None) if fixed else None
        cell = resolve_image_size(vla_cfg.get("tile_cell_size"), obs_size or (256, 256))
        observation = [tile_roles({role: _fake_image() for role in role_descriptions}, cell)]
    else:
        observation = [_fake_image() for _ in view_texts]

    am_cfg = cfg.framework.action_model
    num_target_frames = int(vla_cfg.get("num_target_frames", 1))
    future_frames = [_fake_image() for _ in range(num_target_frames)]

    sample = {
        "image": observation,
        "view_texts": view_texts,
        "lang": "This is a fake instruction for testing.",
        "action": np.random.uniform(-1, 1, size=(int(am_cfg.action_horizon), int(am_cfg.action_dim))).astype(np.float32),
        "gen_target_frames": future_frames,
        "gen_target_image": future_frames[-1],
    }
    if vla_cfg.get("include_state", False) and int(am_cfg.get("state_dim", 0) or 0) > 0:
        sample["state"] = np.random.uniform(-1, 1, size=(1, int(am_cfg.state_dim))).astype(np.float32)
    sample2 = dict(sample, lang="Another fake instruction for testing.")
    batch = [sample, sample2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    vla_losses = vla_cfg.get("losses", None)
    forward_output = model(batch, loss_keys=set(vla_losses) if vla_losses is not None else None)
    print({k: float(v.item()) for k, v in forward_output.items()})

    if model.gen_enabled:
        # Action-free video sample (no action / state): generation loss only.
        video_batch = [{k: v for k, v in s.items() if k not in ("action", "state")} for s in batch]
        print(model(video_batch, loss_keys={"gen_loss"}))

    if model.action_enabled:
        inference_sample = {k: sample[k] for k in ("image", "view_texts", "lang", "state") if k in sample}
        predict_output = model.predict_action([inference_sample])
        print(f"Normalized actions: {predict_output['normalized_actions'].shape}")

    print("Finished")
