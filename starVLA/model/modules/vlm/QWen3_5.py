# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Shijie LIAN/ Huazhong University of Science & Technology] in [2026].
# Design and Merged by [Jinhui YE / HKUST University] in [2026].

import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Optional

import torch
import torch.nn as nn
from PIL import Image, ImageOps
from transformers import AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

from starVLA.training.trainer_utils import initialize_overwatch

try:
    from transformers import Qwen3_5ForConditionalGeneration
except ImportError as import_error:
    raise ImportError(
        "Qwen3.5 model class is unavailable. Please install transformers >= 5.2.0 or check your transformers version."
    ) from import_error

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 248056
VIDEO_TOKEN_INDEX = 248057
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

# Token-id range of discrete action tokens appended to the tokenizer (FAST-style action
# tokenizers). Only used to build VLM labels when ``solutions`` are supplied; the released
# recipes never pass solutions.
_ACTION_TOKEN_MIN = 248077
_ACTION_TOKEN_MAX = 248077 + 2047


_SEQUENCE_PAD_VALUES = {
    "input_ids": None,  # resolved from the selected tokenizer
    "attention_mask": 0,
    "mm_token_type_ids": 0,
    "token_type_ids": 0,
    "labels": IGNORE_INDEX,
}


def require_gated_delta_fast_path(qwenvl_config) -> None:
    """Fail early when a recipe requires Qwen3.5's fused linear-attention kernels."""

    if not bool(qwenvl_config.get("require_gated_delta_fast_path", False)):
        return
    modeling_qwen3_5 = importlib.import_module(Qwen3_5ForConditionalGeneration.__module__)
    if not bool(getattr(modeling_qwen3_5, "is_fast_path_available", False)):
        raise RuntimeError(
            "framework.qwenvl.require_gated_delta_fast_path=true, but Transformers "
            "reports that the Qwen3.5 gated-delta fast path is unavailable. Verify "
            "flash-linear-attention, causal-conv1d, and the Torch/Triton runtime pair."
        )


def merge_qwenvl_input_batches(batches: Sequence[Mapping[str, torch.Tensor]], pad_token_id: int):
    """Left-pad and concatenate independently processed Qwen multimodal batches.

    Qwen's processor flattens visual patches across the batch while keeping text-like
    tensors batch-major.  Fusing co-training streams therefore needs two distinct merge
    rules: left-pad sequence tensors to one common length, and concatenate visual patch
    tensors / grids along their leading dimension.  The returned ``sequence_lengths``
    lets the framework crop the added padding back off each stream's hidden states before
    its action or generation head runs.

    Unknown fields fail loudly.  In particular, precomputed ``position_ids`` cannot be
    padded safely because Qwen's multimodal RoPE contract must be recomputed by the model.
    """

    if not batches:
        raise ValueError("At least one Qwen input batch is required for fusion.")
    if pad_token_id is None:
        raise ValueError("Fused Qwen inputs require a tokenizer pad token id.")
    if any(not isinstance(batch, Mapping) for batch in batches):
        raise TypeError("Every fused Qwen input batch must be a mapping of tensors.")

    key_sets = [set(batch) for batch in batches]
    if any(keys != key_sets[0] for keys in key_sets[1:]):
        raise ValueError(f"Fused Qwen batches must expose identical keys, got {key_sets!r}.")
    if "input_ids" not in key_sets[0] or "attention_mask" not in key_sets[0]:
        raise ValueError("Fused Qwen batches require input_ids and attention_mask.")
    if "position_ids" in key_sets[0]:
        raise ValueError("Fused Qwen batches do not accept precomputed position_ids; let Qwen recompute mRoPE.")

    sequence_lengths = [int(batch["input_ids"].shape[1]) for batch in batches]
    batch_sizes = [int(batch["input_ids"].shape[0]) for batch in batches]
    max_length = max(sequence_lengths)
    merged = {}

    for key in batches[0]:
        values = [batch[key] for batch in batches]
        if any(not isinstance(value, torch.Tensor) for value in values):
            raise TypeError(f"Unsupported non-tensor Qwen field {key!r} in fused inputs.")
        if key in _SEQUENCE_PAD_VALUES:
            pad_value = pad_token_id if key == "input_ids" else _SEQUENCE_PAD_VALUES[key]
            padded = []
            for value, batch_size, sequence_length in zip(values, batch_sizes, sequence_lengths, strict=True):
                if value.ndim < 2 or tuple(value.shape[:2]) != (batch_size, sequence_length):
                    raise ValueError(
                        f"Fused Qwen sequence field {key!r} must start with [B, L]; "
                        f"got {tuple(value.shape)}, expected ({batch_size}, {sequence_length}, ...)."
                    )
                pad_width = max_length - sequence_length
                if pad_width:
                    prefix = torch.full(
                        (batch_size, pad_width, *value.shape[2:]),
                        pad_value,
                        dtype=value.dtype,
                        device=value.device,
                    )
                    value = torch.cat([prefix, value], dim=1)
                padded.append(value)
            merged[key] = torch.cat(padded, dim=0)
        else:
            try:
                merged[key] = torch.cat(values, dim=0)
            except RuntimeError as exc:
                raise ValueError(
                    f"Unsupported Qwen field {key!r} for fused concatenation: "
                    f"shapes={[tuple(value.shape) for value in values]}."
                ) from exc

    return merged, sequence_lengths


def _build_qwenvl_inputs_cpu(
    processor,
    images,
    instructions,
    solutions=None,
    view_texts=None,
    *,
    cot_prompt_enabled: bool = False,
    cot_prompt: str = "",
):
    """Build the canonical Qwen batch on CPU.

    Both the main-process path and the opt-in DataLoader-worker path call this
    function. Keeping message construction, smart resize, tokenization, padding,
    and label masking in one implementation makes worker offload a pure
    scheduling change rather than a second preprocessing recipe.
    """

    messages = []
    if len(images) != len(instructions):
        raise ValueError(
            "Images and instructions must have the same batch length: "
            f"images={len(images)}, instructions={len(instructions)}."
        )
    if solutions is not None and len(solutions) != len(images):
        raise ValueError(
            "Solutions must match the image batch length: " f"solutions={len(solutions)}, images={len(images)}."
        )
    if view_texts is None:
        raise ValueError(
            "view_texts is required: every image must be paired with a view-role description "
            "(images-only prompts are not supported)."
        )
    if len(view_texts) != len(images):
        raise ValueError(
            "view_texts must match the image batch length: " f"view_texts={len(view_texts)}, images={len(images)}."
        )

    for i, (imgs, instruction) in enumerate(zip(images, instructions, strict=True)):
        texts = view_texts[i]
        if not isinstance(texts, (list, tuple)):
            raise ValueError(f"view_texts[{i}] must be a list/tuple, got " f"{type(texts).__name__}.")
        if len(texts) != len(imgs):
            raise ValueError(
                f"view_texts[{i}] must align 1:1 with images[{i}]: " f"texts={len(texts)}, images={len(imgs)}."
            )
        invalid = [index for index, text in enumerate(texts) if not isinstance(text, str) or not text.strip()]
        if invalid:
            raise ValueError(f"view_texts[{i}] contains non-string/empty entries at " f"indices {invalid}.")
        content = []
        for text, img in zip(texts, imgs, strict=True):
            content.append({"type": "text", "text": text})
            content.append({"type": "image", "image": img})

        prompt = cot_prompt.replace("{instruction}", instruction) if cot_prompt_enabled else instruction
        content.append({"type": "text", "text": prompt})
        msg = [{"role": "user", "content": content}]

        if solutions is not None:
            msg.append({"role": "assistant", "content": [{"type": "text", "text": solutions[i]}]})
        messages.append(msg)

    batch_inputs = processor.apply_chat_template(
        messages, tokenize=True, padding=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )

    if solutions is not None:
        labels = batch_inputs["input_ids"].clone()
        for i in range(labels.size(0)):
            seq = labels[i]
            mask_seq = (seq >= _ACTION_TOKEN_MIN) & (seq <= _ACTION_TOKEN_MAX)
            nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
            if nonzero_indices.numel() > 0:
                seq[: nonzero_indices[0].item()] = IGNORE_INDEX
            else:
                seq[:] = IGNORE_INDEX
                logger.warning(
                    "No action token found in the label sequence; labels are fully masked. Check that "
                    "the tokenizer contains the discrete action tokens."
                )

        labels[labels == processor.tokenizer.pad_token_id] = IGNORE_INDEX
        batch_inputs["labels"] = labels

    return batch_inputs


@dataclass
class QwenWorkerPreprocessor:
    """Picklable processor-only callable used by DataLoader worker collators."""

    processor: object
    cot_prompt_enabled: bool = False
    cot_prompt: str = ""

    def __call__(self, examples):
        return _build_qwenvl_inputs_cpu(
            self.processor,
            images=[example["image"] for example in examples],
            instructions=[example["lang"] for example in examples],
            view_texts=[example["view_texts"] for example in examples],
            cot_prompt_enabled=self.cot_prompt_enabled,
            cot_prompt=self.cot_prompt,
        )


def _normalize_image_size(size) -> tuple[int, int]:
    """Return a validated ``(width, height)`` pair from a scalar or pair."""
    if isinstance(size, (str, int, float)):
        width = height = int(size)
    else:
        values = tuple(int(value) for value in size)
        if len(values) != 2:
            raise ValueError(f"image size must be a scalar or [W, H], got {size!r}")
        width, height = values
    if width <= 0 or height <= 0:
        raise ValueError(f"image dimensions must be positive, got {(width, height)}")
    return width, height


def _resize_to_cover_center_crop(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Preserve aspect ratio, then center-crop to an exact fixed target grid."""
    return ImageOps.fit(
        image.convert("RGB"),
        size,
        method=Image.Resampling.BICUBIC,
        centering=(0.5, 0.5),
    )


class _QWen3_5_VL_Interface(nn.Module):
    """
    This exists because of the diversity of VLMs, so we encapsulate the changes here.
    Lightweight wrapper around Qwen3.5-VL (Qwen3_5ForConditionalGeneration).

    Purpose:
        - Unify interface with other VLM backends (CausalLM-like usage).
        - Centralize preprocessing (tokenization + multimodal packing).
        - Provide consistent forward / generate signatures.

    """

    supports_view_role_prompts = True

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Initialize the Qwen3.5 wrapper from ``config.framework.qwenvl``.

        ``base_vlm`` is the local path (or HF hub id) of a Qwen3.5 checkpoint, e.g.
        ``playground/Pretrained_models/Qwen3.5-2B``.
        """
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", None)
        if not model_id:
            raise ValueError(
                "framework.qwenvl.base_vlm is required (path or HF id of a Qwen3.5 checkpoint, "
                "e.g. playground/Pretrained_models/Qwen3.5-2B)."
            )
        require_gated_delta_fast_path(qwenvl_config)

        # The Qwen3.5 backbone always runs with SDPA attention (every released recipe was
        # trained this way), so framework.qwenvl.attn_implementation is not honoured here.
        requested_attn = qwenvl_config.get("attn_implementation", "sdpa")
        attn_implementation = "sdpa"
        if requested_attn != attn_implementation:
            logger.warning(
                f"framework.qwenvl.attn_implementation={requested_attn!r} is ignored; the Qwen3.5 "
                "wrapper always uses 'sdpa'."
            )

        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
        )
        processor_kwargs = {}
        max_pixels = qwenvl_config.get("max_pixels", None)
        if max_pixels is not None:
            if isinstance(max_pixels, bool) or not isinstance(max_pixels, Integral) or max_pixels <= 0:
                raise ValueError(f"framework.qwenvl.max_pixels must be a positive integer, got {max_pixels!r}")
            processor_kwargs["max_pixels"] = int(max_pixels)
        processor = AutoProcessor.from_pretrained(model_id, **processor_kwargs)
        processor.tokenizer.padding_side = "left"

        self.model = model
        self.processor = processor
        self.config = config

        # Expose the LLM width at the top level (framework code reads config.hidden_size).
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

        # --- Learnable generation queries (gen_model backbone='vlm'; opt-in, default off) ---
        # Populated by setup_gen_queries(); when disabled, all gen-related code paths
        # below are no-ops. The queries are a
        # SEPARATE nn.Parameter (self.gen_query_embed), NOT rows appended to the VLM's
        # embedding table — so they never enter embed_tokens' optimizer group / weight
        # decay, and freezing them is a plain requires_grad_ toggle (no gradient hook).
        self._gen_queries_enabled = False
        self.num_gen_queries = 0
        # Per-group query counts (multi-target). Single-target keeps [num_gen_queries];
        # populated by setup_gen_queries(). The composite gen head slices the query
        # states per group using these sizes' running offsets.
        self._gen_group_sizes = None
        self.gen_query_embed = None  # nn.Parameter (block_len, hidden): [boi, gen_0..gen_{N-1}, eoi]
        self._gen_block_len = 0  # num_gen_queries + 2 (boi + gen + eoi)
        self._gen_placeholder_id = None  # text token id filling the appended block (embeddings overwritten)
        self._gen_inject_active = False  # one-shot flag: forward() injects gen_query_embed for this call

        # Checkpoints with discrete action tokens appended to the vocabulary.
        if "-Action" in model_id:
            self._ACTION_TOKEN_MIN = _ACTION_TOKEN_MIN
            self._ACTION_TOKEN_MAX = _ACTION_TOKEN_MAX

    def forward(
        self,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass delegating to underlying Qwen3.5-VL backbone.

        When append_gen_queries() flagged an injection, a scoped forward hook on
        embed_tokens overwrites the trailing gen-query block's embeddings with the
        standalone ``gen_query_embed`` Parameter (gradients flow into it; the VLM
        embedding table is untouched). The flag is one-shot (consumed here).
        """
        inject = bool(getattr(self, "_gen_inject_active", False))
        self._gen_inject_active = False
        if not inject:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return self.model(**kwargs)

        block_len = self._gen_block_len
        gen_embed = self.gen_query_embed

        def _inject_gen_block(module, inputs, output):
            output = output.clone()
            output[:, -block_len:, :] = gen_embed.to(dtype=output.dtype, device=output.device).unsqueeze(0)
            return output

        handle = self.model.get_input_embeddings().register_forward_hook(_inject_gen_block)
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return self.model(**kwargs)
        finally:
            handle.remove()

    def generate(
        self,
        **kwargs,
    ):
        """
        High-level generation interface (auto-regressive decoding), optionally vision-conditioned.

        Args:
            **kwargs: fully follow raw model.generate() signature.
        Returns:
            GenerateOutput | Model-dependent generation return.
        """
        with torch.autocast("cuda", dtype=torch.float16):
            generation_output = self.model.generate(
                **kwargs,
            )
        return generation_output

    def build_qwenvl_inputs(self, images, instructions, solutions=None, view_texts=None, **kwargs):
        """
        Build model inputs from raw data (images + instructions + optional solutions)
        using the Qwen3.5 chat template.

        ``view_texts``: required per-sample strings aligned 1:1 with each sample's
        ``images``. Each view's text is emitted right BEFORE its image so the model
        reads explicit ``<view_desc><image>`` pairs.
        """

        vla_cfg = self.config.datasets.vla_data
        cot_prompt_enabled = "CoT_prompt" in vla_cfg
        cot_prompt = vla_cfg.get("CoT_prompt", "") if cot_prompt_enabled else ""
        batch_inputs = _build_qwenvl_inputs_cpu(
            self.processor,
            images,
            instructions,
            solutions=solutions,
            view_texts=view_texts,
            cot_prompt_enabled=cot_prompt_enabled,
            cot_prompt=cot_prompt,
        )
        return self.move_qwenvl_inputs(batch_inputs)

    def make_worker_preprocessor(self) -> QwenWorkerPreprocessor:
        """Return a processor-only callable with the exact training prompt config."""

        vla_cfg = self.config.datasets.vla_data
        cot_prompt_enabled = "CoT_prompt" in vla_cfg
        cot_prompt = vla_cfg.get("CoT_prompt", "") if cot_prompt_enabled else ""
        return QwenWorkerPreprocessor(
            processor=self.processor,
            cot_prompt_enabled=cot_prompt_enabled,
            cot_prompt=cot_prompt,
        )

    def move_qwenvl_inputs(self, batch_inputs):
        """Move either a BatchFeature or worker-produced tensor dict to the VLM."""

        if hasattr(batch_inputs, "to"):
            return batch_inputs.to(self.model.device)
        return {
            key: value.to(self.model.device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch_inputs.items()
        }

    def merge_qwenvl_inputs(self, batches):
        """Merge worker-produced stream batches for one shared Qwen forward."""

        tokenizer = self.processor.tokenizer
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        return merge_qwenvl_input_batches(batches, pad_token_id)

    # ──────────────────────────────────────────────────────────────────
    #  Learnable generation queries (gen_model backbone='vlm'; opt-in)
    #  These methods are no-ops unless setup_gen_queries() was called.
    # ──────────────────────────────────────────────────────────────────
    def setup_gen_queries(self, num_queries):
        """Create the learnable generation-query block as a SEPARATE nn.Parameter.

        ``num_queries`` may be an int (single group) OR a list/tuple of ints
        (multi-target: one group per target). ALL groups live inside ONE
        ``[boi, group_0…, group_1…, …, eoi]`` block, so append/extract/mask logic is
        unchanged; ``num_gen_queries`` is the TOTAL gen-token count and
        ``_gen_group_sizes`` records the per-group split (the composite gen head slices
        the extracted query states by the groups' running offsets).

        This does NOT add tokens to the tokenizer or resize ``embed_tokens``: the block's
        embeddings are a standalone ``nn.Parameter`` injected at forward time (see
        :meth:`forward`). This keeps the VLM embedding table (and its optimizer group /
        weight decay / vocab) untouched, and makes freezing the queries a plain
        ``requires_grad_`` toggle.
        """
        if isinstance(num_queries, int):
            sizes = [int(num_queries)]
        else:
            sizes = [int(n) for n in num_queries]
        total = int(sum(sizes))

        self.num_gen_queries = total
        self._gen_group_sizes = sizes
        self._gen_block_len = total + 2  # [boi, gen..., eoi]

        hidden = int(self.model.config.hidden_size)
        self.gen_query_embed = nn.Parameter(torch.empty(self._gen_block_len, hidden))
        nn.init.normal_(self.gen_query_embed, mean=0.0, std=0.02)

        # Placeholder token id filling the appended positions: a plain TEXT token so mRoPE
        # numbers the block sequentially and mm_token_type marks it text. Its embedding is
        # ALWAYS overwritten by the forward hook, so only "non-image text" matters; fall
        # back to eos when the tokenizer defines no pad token.
        tok = self.processor.tokenizer
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        assert pad is not None, "tokenizer has neither pad nor eos token for the gen-query placeholder."
        self._gen_placeholder_id = int(pad)

        self._gen_queries_enabled = True
        self._gen_inject_active = False
        logger.info(
            f"[QWen3.5] enabled {total} learnable gen queries (block={self._gen_block_len}, groups={sizes}) "
            "as a standalone nn.Parameter (embed_tokens / vocab untouched)"
        )

    def set_gen_query_trainable(self):
        """Re-enable gradients on the gen-query Parameter (call AFTER any global VLM
        freeze)."""
        if self._gen_queries_enabled and self.gen_query_embed is not None:
            self.gen_query_embed.requires_grad_(True)

    def freeze_visual_encoder(self):
        """Freeze the visual encoder (vision tower + spatial-merge ``merger``).

        Called by the framework so that fine-tuning touches ONLY the LLM backbone
        while the visual encoder stays fixed. This also keeps the ``auto`` generation
        target (``encode_image_features`` = the vision tower + merger output) a STABLE
        regression/diffusion target instead of one that drifts as the VLM trains.

        ``requires_grad_(False)`` is the load-bearing part — the LR-group builder then
        drops these params from the optimizer, so the tower is never updated. ``.eval()``
        is best-effort determinism (the Qwen vision tower has no dropout, and a later
        ``model.train()`` may flip the mode back); the frozen weights alone already make
        the target stable.
        """
        visual = getattr(self.model, "visual", None)
        if visual is None:
            visual = getattr(getattr(self.model, "model", None), "visual", None)
        if visual is None:
            logger.warning("[QWen3.5-VL] freeze_visual_encoder: no `.visual` tower found; nothing frozen.")
            return
        visual.requires_grad_(False)
        visual.eval()
        logger.info("[QWen3.5-VL] visual encoder (vision tower + merger) FROZEN (requires_grad=False).")

    def append_gen_queries(self, batch_inputs):
        """Append the [boi, gen_0..gen_{N-1}, eoi] placeholder block at the causal end and
        flag :meth:`forward` to inject the learnable ``gen_query_embed`` at those positions.

        The block is filled with a plain text placeholder token id; its embeddings are
        overwritten by the forward hook. Queries sit at the causal end (they attend to all
        image+text tokens, while the left-padded image/text never see them). position_ids
        are recomputed by the model from input_ids, so the queries get continuous mRoPE
        positions.
        """
        if not self._gen_queries_enabled:
            return batch_inputs
        input_ids = batch_inputs["input_ids"]
        attn = batch_inputs["attention_mask"]
        B, L = input_ids.size(0), self._gen_block_len
        block = torch.full((B, L), self._gen_placeholder_id, dtype=input_ids.dtype, device=input_ids.device)
        batch_inputs["input_ids"] = torch.cat([input_ids, block], dim=1)
        batch_inputs["attention_mask"] = torch.cat([attn, torch.ones(B, L, dtype=attn.dtype, device=attn.device)], dim=1)
        # The gen block is TEXT: extend mm_token_type_ids so mRoPE keeps numbering
        # image/text exactly as without the block. Its absence would make the model fall
        # back to a cumulative position scheme for the WHOLE sequence (see modeling
        # get_rope_index) — warn loudly rather than silently changing positions.
        if "mm_token_type_ids" in batch_inputs:
            mm_ids = batch_inputs["mm_token_type_ids"]
            batch_inputs["mm_token_type_ids"] = torch.cat(
                [mm_ids, torch.zeros(B, L, dtype=mm_ids.dtype, device=mm_ids.device)], dim=1
            )
        else:
            logger.warning(
                "[QWen3.5-VL] append_gen_queries: inputs have no `mm_token_type_ids`; mRoPE may fall "
                "back to a cumulative position scheme. Ensure the processor emits it."
            )
        self._gen_inject_active = True  # forward() injects gen_query_embed for the next call
        return batch_inputs

    def extract_query_states(self, hidden_states):
        """Slice the N query hidden states out of a (B, L, H) last-layer tensor.

        The query block is the last (N+2) positions; the N gen tokens are between
        <boi> (at -(N+2)) and <eoi> (at -1).
        """
        n = self.num_gen_queries
        return hidden_states[:, -(n + 1) : -1, :]

    # ──────────────────────────────────────────────────────────────────
    #  Vision-tower target encoding (for the "auto" generation target)
    #  Encodes images through the SAME visual encoder the VLM uses to feed
    #  the LLM, so the generation head can regress the VLM's own visual
    #  representation of the future frame.
    # ──────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def encode_image_features(self, images, target_resolution=None):
        """Encode PIL images into per-image visual token features.

        Returns the visual tokens AS THEY ENTER THE LLM: the vision tower output
        AFTER the spatial-merge ``merger`` (dim == LLM hidden size). This is the
        exact representation a future frame would contribute as image tokens, so
        the generation head regresses the VLM's own visual encoding.

        When ``target_resolution`` is given, every image is first resized to
        cover the fixed scalar-square or ``[W, H]`` canvas and center-cropped,
        preserving geometry rather than stretching. The processor's adaptive
        smart-resize is then pinned (min_pixels == max_pixels == W*H) so the
        visual-token count is deterministic and identical across images —
        required because the number of learnable queries is fixed.

        Returns:
            list[Tensor]: length B, each (num_merged_tokens, llm_hidden_size).
        """
        proc = self.processor
        proc_kwargs = {}
        if target_resolution is not None:
            target_w, target_h = _normalize_image_size(target_resolution)
            images = [_resize_to_cover_center_crop(img, (target_w, target_h)) for img in images]
            target_pixels = target_w * target_h
            proc_kwargs = {"min_pixels": target_pixels, "max_pixels": target_pixels}

        visual = getattr(self.model, "visual", None) or getattr(self.model.model, "visual", None)

        # Encode ALL images in ONE processor call + ONE vision-tower forward, then
        # split the concatenated merged tokens back per image (a per-image loop would
        # serialize batch=1 forwards). The Qwen vision tower natively consumes
        # concatenated multi-image patches via per-image grid_thw rows, exactly as in
        # the normal VLM forward.
        images = list(images)
        inputs = proc.image_processor(images=images, return_tensors="pt", **proc_kwargs)
        pixel_values = inputs["pixel_values"].to(device=self.model.device, dtype=self.model.dtype)
        grid_thw = inputs["image_grid_thw"]  # CPU (num_images, 3), patch grid per image
        out = visual(pixel_values, grid_thw=grid_thw.to(self.model.device))
        pre_merge = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
        merged = visual.merger(pre_merge)  # (total_merged_tokens, out_hidden == llm_hidden)

        # merged tokens per image = prod(grid_thw_row) / spatial_merge_area. Derive
        # the merge area from the totals so we need no model-specific attribute, and
        # split on the CPU grid_thw tensor (no CUDA sync). Ordering matches the input
        # image order (the merger preserves it).
        patches_per_img = grid_thw.prod(dim=1)  # (num_images,)
        merge_area = max(1, int(patches_per_img.sum()) // int(merged.shape[0]))
        counts = (patches_per_img // merge_area).tolist()
        return list(torch.split(merged, counts, dim=0))

    @torch.no_grad()
    def count_image_tokens(self, target_resolution):
        """Number of visual tokens emitted at a scalar-square or ``[W, H]`` size."""
        from PIL import Image as _Image

        dummy = _Image.new("RGB", _normalize_image_size(target_resolution))
        return self.encode_image_features([dummy], target_resolution=target_resolution)[0].shape[0]


if __name__ == "__main__":
    # Smoke test: build the wrapper from a training config and run one forward pass on a
    # fake role-aware sample. Run from the repo root:
    #   python starVLA/model/modules/vlm/QWen3_5.py --config_yaml configs/pretrain/cotrain_openaoe.yaml
    import argparse

    from starVLA.model.framework.share_tools import load_config_with_bases

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="configs/pretrain/cotrain_openaoe.yaml",
        help="Path to a training YAML (``_base_`` includes are resolved).",
    )
    args = parser.parse_args()

    cfg = load_config_with_bases(args.config_yaml)
    qwen_vl = _QWen3_5_VL_Interface(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qwen_vl = qwen_vl.to(device)

    image = Image.new("RGB", (224, 224), color=(127, 127, 127))
    inputs = qwen_vl.build_qwenvl_inputs(
        images=[[image]],
        instructions=["pick up the cup"],
        view_texts=[["primary global view from a fixed exterior camera"]],
    )
    with torch.no_grad():
        outputs = qwen_vl(**inputs, output_hidden_states=True, return_dict=True)
    print(
        f"input_ids: {tuple(inputs['input_ids'].shape)}, hidden states: {len(outputs.hidden_states)} x "
        f"{tuple(outputs.hidden_states[-1].shape)}"
    )
