# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Temporal sampling of the future-frame generation target.

The future frame ``o_{t+Delta}`` is aligned with the end of the action chunk,
``a_{t+H}``, so the horizon is ``tau = H / f_robot`` seconds on the robot stream's clock
(8 / 15 Hz = 0.533 s on DROID, 8 / 20 Hz = 0.4 s on LIBERO; an explicit
``framework.gen_model.gen_horizon_seconds`` overrides it). ``tau`` is converted to each
stream's own frame rate ``f``:

    Delta = floor(tau * f + 1/2)            (frames ahead of the current observation)

e.g. with DROID as the robot stream (``tau = 8/15 s``) the target is 8 DROID frames and
16 frames of a 30 fps egocentric video ahead. With ``T = framework.gen_model.num_future_frames``
frames, frame ``k = 1..T`` lies at ``k * tau / T`` seconds, i.e. at

    floor(k * tau * f / T + 1/2)

frames ahead; the last one is always ``Delta``. The current observation frame is never
a target (strict future prediction).

``align_dataset_configs`` resolves these per stream and writes ``gen_future_offset``
(``Delta``), ``gen_stride_frames`` (``tau * f / T`` when ``T > 1``, else 0) and
``num_target_frames`` (``T``) onto every stream block; both the robot generation
dataset and the egocentric video dataset read them through :func:`gen_target_indices`,
so every stream emits an identically shaped target for the one generation head.
"""

import itertools
import math


def _round_half_up_nonnegative(value: float) -> int:
    """Round a non-negative frame position, resolving exact halves forward."""
    if value < 0:
        raise ValueError(f"frame position must be non-negative, got {value}")
    return math.floor(float(value) + 0.5 + 1e-9)


def resolve_future_frames(horizon_seconds: float, stream_fps: float, num_frames: int) -> dict:
    """Resolve the future-frame grid of one stream.

    Returns ``{"gen_future_offset": Delta, "gen_stride_frames": s, "num_target_frames": T,
    "offsets": [...]}`` where ``offsets`` are the frame offsets of the ``T`` targets.
    Raises if the stream's frame rate cannot represent ``T`` distinct future frames.
    """
    tau = float(horizon_seconds)
    fps = float(stream_fps)
    T = int(num_frames)
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError(f"framework.gen_model.gen_horizon_seconds must be positive, got {horizon_seconds!r}.")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"dataset_fps must be positive, got {stream_fps!r}.")
    if T < 1:
        raise ValueError(f"framework.gen_model.num_future_frames must be >= 1, got {num_frames!r}.")

    offset = max(1, _round_half_up_nonnegative(tau * fps))
    stride_frames = tau * fps / T if T > 1 else 0.0
    offsets = gen_target_indices(0, offset, stride_frames, T)
    if offsets[0] < 1 or any(b <= a for a, b in itertools.pairwise(offsets)) or offsets[-1] != offset:
        raise ValueError(
            f"A {fps:g} fps stream cannot represent {T} distinct future frames within {tau:g} s "
            f"(resolved offsets {offsets}); lower num_future_frames or raise gen_horizon_seconds."
        )
    return {
        "gen_future_offset": int(offset),
        "gen_stride_frames": float(stride_frames),
        "num_target_frames": T,
        "offsets": offsets,
    }


def gen_target_indices(base_index: int, gen_future_offset: int, gen_stride_frames: float, n_frames: int) -> list[int]:
    """Absolute frame indices of the generation target for a given ``base_index``.

    ``base_index`` is the CURRENT observation frame and is never returned.
    Non-positive ``gen_stride_frames`` -> ``[base + gen_future_offset]`` (one frame at the
    horizon). Positive stride -> ``base + round(k * gen_stride_frames)`` for ``k = 1..T``.
    """
    T = int(n_frames)
    if gen_stride_frames <= 0:
        return [base_index + int(gen_future_offset)]
    if T < 1:
        raise ValueError(f"n_frames must be positive for multi-frame targets, got {T}.")
    return [base_index + _round_half_up_nonnegative(k * float(gen_stride_frames)) for k in range(1, T + 1)]


if __name__ == "__main__":
    # Pre-training: tau = H / f_DROID = 8 / 15 s -> 8 DROID frames, 16 frames of a 30 fps video.
    assert resolve_future_frames(8 / 15, 15, 1)["offsets"] == [8]
    assert resolve_future_frames(8 / 15, 30, 1)["offsets"] == [16]
    # LIBERO fine-tuning: tau = 8 / 20 s = 0.4 s -> 8 frames.
    assert resolve_future_frames(8 / 20, 20, 1)["offsets"] == [8]
    # Four frames evenly spaced up to the end of the action chunk.
    assert resolve_future_frames(8 / 15, 15, 4)["offsets"] == [2, 4, 6, 8]
    assert resolve_future_frames(8 / 15, 30, 4)["offsets"] == [4, 8, 12, 16]
    # Off-grid instants are rounded on each stream's own clock.
    assert resolve_future_frames(0.4, 15, 4)["offsets"] == [2, 3, 5, 6]
    # The dataset-side helper reproduces the same grid from the derived keys.
    r = resolve_future_frames(8 / 15, 30, 4)
    assert gen_target_indices(100, r["gen_future_offset"], r["gen_stride_frames"], 4) == [104, 108, 112, 116]
    assert gen_target_indices(100, 8, 0.0, 1) == [108]
    # Too many frames for the stream's frame rate is rejected.
    try:
        resolve_future_frames(0.1, 15, 4)
    except ValueError:
        pass
    else:
        raise AssertionError("expected a ValueError for an unrepresentable grid")
    print("[OK] gen_target_sampling self-test passed")
