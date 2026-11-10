# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Shared, same-source-preserving frame augmentation for the generation datasets.

Both ``gen_lerobot_datasets`` and ``video_datasets`` decode a view's
``[current, *future]`` window in ONE pass (so observation and generation target are
strictly same-source) and then split it into observation / generation target.
This module centralises the OPTIONAL augmentation applied to that window so:

  * the split + augmentation policy is defined ONCE and shared by both datasets (no
    divergence between the two producers), and
  * observation vs generation-target augmentation is explicitly controllable.

``GenFrameAug.process(window)`` (``window = [current_pil, *future_pil]``, already decoded
and GEOMETRICALLY same-source) returns ``(obs_pil, target_pils)``:

  * ``obs_pil``     — the VLM observation's current frame; photometrically AUGMENTED when
    enabled (augmenting the model's input improves robustness).
  * ``target_pils`` — the future target frames; by DEFAULT (``gen_target_clean=false``) they
    receive the SAME jitter as obs, so obs and target stay photometrically unified; set
    ``gen_target_clean=true`` to keep the target CLEAN (regress an un-augmented target).

Only PHOTOMETRIC jitter (ColorJitter) is applied here; geometry is untouched (the
deterministic cover-crop / tiling downstream is identical for obs and target, so geometric
same-source is preserved regardless). A SINGLE set of jitter factors is sampled per window
via ``torchvision``'s ``ColorJitter.get_params`` (torch RNG — DataLoader-worker-seeded), so
every augmented frame of the window shares the same jitter. When augmentation is disabled the
transform is the IDENTITY split (``obs == current``, ``target == future``), i.e.
byte-identical to the no-augmentation path.
"""

import torchvision.transforms as T
import torchvision.transforms.functional as F


class GenFrameAug:
    """Photometric ColorJitter applied identically across a ``[current, *future]`` window.

    ``gen_target_clean`` (default False): the generation target receives the SAME jitter as the
    observation, so obs and target stay unified. True: only the observation is jittered; the
    target stays clean.
    """

    def __init__(
        self,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        hue: float = 0.0,
        gen_target_clean: bool = False,
    ):
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.hue = float(hue)
        # Fail loudly at construction (not mid-epoch inside a worker) — F.adjust_hue requires |hue|<=0.5.
        if not 0.0 <= self.hue <= 0.5:
            raise ValueError(f"gen_augment.hue must be in [0, 0.5] (torchvision adjust_hue bound), got {self.hue}.")
        self.gen_target_clean = bool(gen_target_clean)

    def _range(self, value: float, center: float = 1.0, clip_min: float | None = 0.0):
        """torchvision-style symmetric range around ``center`` (or [-v, v] for hue)."""
        if value <= 0:
            return None
        lo, hi = center - value, center + value
        if clip_min is not None:
            lo = max(clip_min, lo)
        return [lo, hi]

    def _sample_jitter(self):
        """Sample ONE set of ColorJitter factors (torch RNG) and return an ``img -> img`` fn.

        Mirrors ``torchvision.transforms.ColorJitter.forward`` but with FIXED factors so the
        same jitter is applied to every frame of the window (shared across obs + future)."""
        brightness = self._range(self.brightness)
        contrast = self._range(self.contrast)
        saturation = self._range(self.saturation)
        hue = self._range(self.hue, center=0.0, clip_min=None)
        fn_idx, b, c, s, h = T.ColorJitter.get_params(brightness, contrast, saturation, hue)

        def _apply(img):
            for i in fn_idx:
                if i == 0 and b is not None:
                    img = F.adjust_brightness(img, b)
                elif i == 1 and c is not None:
                    img = F.adjust_contrast(img, c)
                elif i == 2 and s is not None:
                    img = F.adjust_saturation(img, s)
                elif i == 3 and h is not None:
                    img = F.adjust_hue(img, h)
            return img

        return _apply

    def process(self, window: list):
        """window = [current, *future] (PIL) -> (obs, target_frames).

        obs is always jittered; the target follows ``gen_target_clean`` (jittered with the SAME
        sampled factors as obs by default, else kept clean)."""
        if not window:
            raise ValueError("GenFrameAug.process requires a non-empty [current, *future] window.")
        jitter = self._sample_jitter()
        current, future = window[0], list(window[1:])
        obs = jitter(current)
        if self.gen_target_clean:
            return obs, future
        return obs, [jitter(f) for f in future]


def resolve_gen_frame_aug(data_cfg, mode: str):
    """Build a :class:`GenFrameAug` from ``datasets.<block>.gen_augment`` (+ ``gen_target_clean``),
    or ``None`` when augmentation is disabled or ``mode != 'train'`` (no aug at eval/val).

    Config (all optional; disabled by default so this is behavior-preserving):
        datasets.shared.gen_augment: {enabled: false, brightness: .., contrast: .., saturation: .., hue: ..}
        datasets.shared.gen_target_clean: false  # (default) target shares the obs jitter; true keeps it clean
    """
    if data_cfg is None or str(mode) != "train":
        return None
    get = data_cfg.get if hasattr(data_cfg, "get") else (lambda k, d=None: getattr(data_cfg, k, d))
    aug = get("gen_augment", None)
    if not aug:
        return None
    aget = aug.get if hasattr(aug, "get") else (lambda k, d=None: getattr(aug, k, d))
    if not bool(aget("enabled", False)):
        return None
    return GenFrameAug(
        brightness=float(aget("brightness", 0.0)),
        contrast=float(aget("contrast", 0.0)),
        saturation=float(aget("saturation", 0.0)),
        hue=float(aget("hue", 0.0)),
        gen_target_clean=bool(get("gen_target_clean", False)),
    )
