"""Opt-in worker-side model-input preprocessing without dropping raw samples."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable

_BATCH_MARKER = "starvla.qwen_preprocessed.v1"


@dataclass
class QwenPreprocessingCollator:
    """Run the exact Qwen processor after a dataset's ordinary collate step."""

    base_collate: Callable
    preprocessor: Callable

    def __call__(self, samples):
        examples = self.base_collate(samples)
        return {
            "batch_format": _BATCH_MARKER,
            "examples": examples,
            "qwen_inputs": dict(self.preprocessor(examples)),
        }


def unpack_preprocessed_batch(batch):
    """Return ``(raw examples, optional Qwen tensors)`` for either batch form."""

    if not isinstance(batch, Mapping) or batch.get("batch_format") != _BATCH_MARKER:
        return batch, None
    if "examples" not in batch or "qwen_inputs" not in batch:
        raise ValueError("A Qwen-preprocessed batch must contain examples and qwen_inputs.")
    examples = batch["examples"]
    qwen_inputs = batch["qwen_inputs"]
    if not isinstance(examples, list):
        raise TypeError(f"preprocessed batch examples must be a list, got {type(examples).__name__}.")
    if not isinstance(qwen_inputs, Mapping):
        raise TypeError(f"preprocessed qwen_inputs must be a mapping, got {type(qwen_inputs).__name__}.")
    return examples, qwen_inputs


def batch_size(batch) -> int:
    """Return the sample count without confusing wrapper keys for samples."""

    examples, _ = unpack_preprocessed_batch(batch)
    return len(examples)
