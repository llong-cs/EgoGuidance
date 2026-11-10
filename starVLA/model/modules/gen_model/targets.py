# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Target axis of the generation head.

A *target* turns the future frame(s) (``List[List[PIL.Image]]`` — B samples,
each a list of T frames) into the tensor the generation objective is trained
against.  Two kinds:

  * ``kind == "repr"`` — per-token high-dimensional features ``(B, N_t, D_t)``.
    Sources: ``auto`` (the VLM's own vision tower — generation & perception
    share one space), ``siglip`` / ``dino`` / ``lingbot`` (a frozen external
    image encoder), ``vjepa`` (a frozen video encoder), ``pixel`` (raw
    patchified pixels).

  * ``kind == "vae"`` — a VAE latent grid ``(B, C, T, H, W)`` (``wan_vae``, the
    Wan2.2 3D video VAE); the heads flatten it to tokens ``(B, T*H*W, C)``.

Every target exposes ``feature_dim() -> D_t``, ``count_tokens() -> N_t`` (so the
framework can auto-resolve ``num_gen_queries``), and ``encode(target_frames)``.
Frozen encoders are registered as submodules so the trainer's ``.to(device)``
moves them; the ``auto`` target instead holds the VLM in a 1-element list to
avoid duplicating its params, and the VAE is list-wrapped to keep it out of the
state dict.

Per-target ``feat_mean`` / ``feat_std`` are feature statistics measured on
LIBERO + Ego4D future frames; they are used only when
``framework.gen_model.normalize_target`` is enabled (see ``regression.py``).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

TARGET_REGISTRY = {}


def register_target(*names):
    def _wrap(cls):
        for n in names:
            TARGET_REGISTRY[n] = cls
        return cls

    return _wrap


def build_target(global_config, vlm_interface, input_dim):
    """Instantiate the target encoder selected by ``framework.gen_model.target.type``."""
    tcfg = global_config.framework.gen_model.target
    ttype = str(tcfg.get("type", "auto")).lower()
    if ttype not in TARGET_REGISTRY:
        raise NotImplementedError(f"gen target `{ttype}` is not implemented. Available: {sorted(TARGET_REGISTRY)}")
    return TARGET_REGISTRY[ttype](global_config, vlm_interface=vlm_interface, input_dim=input_dim)


def resolve_num_target_frames(global_config) -> int:
    """Number of generation-target frames T (drives the gen head's query/token count).

    ``align_dataset_configs`` writes T onto every stream block (``num_target_frames``);
    the ``datasets.vla_data`` value is preferred here so the head always matches the
    dataloader. Outside the training entry points (e.g. a ``__main__`` smoke test that
    skips alignment) T falls back to ``framework.gen_model.num_future_frames``.
    """
    try:
        dcfg = global_config.datasets.vla_data
        get = dcfg.get if hasattr(dcfg, "get") else (lambda k, d=None: getattr(dcfg, k, d))
        stored = get("num_target_frames", None)
    except Exception:
        stored = None
    if stored is not None:
        return max(1, int(stored))
    try:
        gen_cfg = global_config.framework.gen_model
        configured = gen_cfg.get("num_future_frames", 1) if hasattr(gen_cfg, "get") else 1
    except Exception:
        configured = 1
    return max(1, int(configured))


def tubelet_padded_num_frames(n_frames: int, tubelet_size: int) -> int:
    """Round a frame count up to a whole multiple of ``tubelet_size``.

    V-JEPA2 tokenizes time with a stride-``tubelet_size`` Conv3d, so trailing
    frames that don't fill a tubelet are silently dropped (and a lone frame
    can't form a tubelet at all). We duplicate the last frame(s) up to the next
    multiple so the requested horizon is fully covered; the resulting temporal
    token count is ``padded // tubelet_size == ceil(n_frames / tubelet_size)``.
    """
    if n_frames < 1:
        raise ValueError(f"tubelet padding needs >=1 frame, got {n_frames}")
    if tubelet_size < 1:
        raise ValueError(f"tubelet_size must be >=1, got {tubelet_size}")
    return math.ceil(n_frames / tubelet_size) * tubelet_size


def grid_to_tokens(grid):
    """(B, C, T, H, W) VAE latent grid -> (B, T*H*W, C) token sequence.

    Both the regression MLP and the diffusion expert are token-based, so a VAE
    grid is flattened to tokens before use.
    """
    B, C = grid.shape[0], grid.shape[1]
    return grid.reshape(B, C, -1).transpose(1, 2)


def align_lengths(target, num_tokens, pool):
    """Reconcile the target token length to ``num_tokens`` if requested.

    ``pool='none'`` returns the target unchanged (caller must ensure lengths
    match); ``pool='target_to_query'`` average-pools the token axis down to
    ``num_tokens``.
    """
    if pool == "none":
        return target
    if pool == "target_to_query":
        t = target.transpose(1, 2)  # (B, D, N_t)
        t = F.adaptive_avg_pool1d(t, num_tokens)
        return t.transpose(1, 2)
    raise ValueError(f"Unknown pool mode: {pool}")


class BaseTarget(nn.Module):
    """Frozen target encoder producing ``(B, N_t, D_t)`` from future frames."""

    kind = "repr"  # "repr" | "vae"
    # Whether the target's tokens decompose per-frame (frame f occupies a contiguous slice,
    # count_tokens == per_frame * T). True for per-frame image encoders; False
    # for temporally-aggregating encoders (V-JEPA tubelets, 3D video VAEs) whose tokens MIX
    # frames. frame_mode='conditioned' (one frame per step) requires True (see _validate_frame_mode).
    per_frame_decomposable = True
    # Per-target feature standardization stats (see RegressionGenHead, normalize_target).
    # The regression head standardizes the encoded target as ``(feat - feat_mean) / feat_std``
    # so gen_loss magnitude is comparable across targets whose raw feature scales differ.
    # Measured over LIBERO+Ego4D future frames; each concrete target overrides these.
    # (0, 1) = exact no-op for an uncalibrated target. Config-overridable via
    # target.feat_mean / target.feat_std.
    feat_mean: float = 0.0
    feat_std: float = 1.0

    def train(self, mode: bool = True):
        """Target encoders are frozen: keep them in eval mode (no dropout / drop-path)
        even when the surrounding model is switched back to training, e.g. after validation."""
        return super().train(False)

    @staticmethod
    def _normalize_frames(target_images):
        """Accept List[PIL] (T=1) or List[List[PIL]] -> List[List[PIL]]."""
        out = []
        for t in target_images:
            out.append(list(t) if isinstance(t, (list, tuple)) else [t])
        return out

    def feature_dim(self) -> int:
        raise NotImplementedError

    @torch.no_grad()
    def count_tokens(self) -> int:
        raise NotImplementedError

    @torch.no_grad()
    def encode(self, target_frames) -> torch.Tensor:
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────
#  repr targets
# ──────────────────────────────────────────────────────────────────────
@register_target("auto")
class AutoTarget(BaseTarget):
    """Reuse the VLM backbone's OWN (frozen) vision tower as the target encoder.

    The target is exactly what the future frame would become as visual tokens
    feeding the LLM, so generation and perception share one visual-encoding
    space.  ``target_dim == LLM hidden size``; token count == whatever the VLM
    emits for a scalar-square or ``resolution: [W, H]`` image, times T.
    """

    kind = "repr"
    # calibrated on LIBERO+Ego4D
    feat_mean = -0.0028
    feat_std = 0.1414

    def __init__(self, global_config, vlm_interface=None, input_dim=None):
        super().__init__()
        assert vlm_interface is not None, "target.type='auto' requires vlm_interface"
        tcfg = global_config.framework.gen_model.target
        self._vlm_ref = [vlm_interface]  # list-wrap: not a submodule (no param duplication)
        resolution = tcfg.get("resolution", 224)
        if isinstance(resolution, (str, int, float)):
            self.target_resolution = int(resolution)
        else:
            self.target_resolution = tuple(int(value) for value in resolution)
            if len(self.target_resolution) != 2 or any(value <= 0 for value in self.target_resolution):
                raise ValueError(f"target.resolution must be a scalar or [W, H], got {resolution!r}")
        self._target_dim = int(input_dim)
        self.num_target_frames = resolve_num_target_frames(global_config)

    def feature_dim(self):
        return self._target_dim

    @torch.no_grad()
    def count_tokens(self):
        per_frame = self._vlm_ref[0].count_image_tokens(self.target_resolution)
        return int(per_frame) * int(self.num_target_frames)

    @torch.no_grad()
    def encode(self, target_frames):
        # Flatten (B samples x T frames) into ONE list and encode in a SINGLE batched
        # vision-tower pass (a per-image loop serializes B*T batch=1 forwards). Every
        # frame is pinned to the configured fixed size by encode_image_features
        # (min_pixels == max_pixels), so each yields the SAME token count N and the flat
        # (B*T, N, D) can be reshaped back to (B, T*N, D).
        samples = self._normalize_frames(target_images=target_frames)  # (B, T) PIL
        num_frames = len(samples[0])
        flat = [img for frames in samples for img in frames]  # B*T, sample-major
        per = self._vlm_ref[0].encode_image_features(
            flat, target_resolution=self.target_resolution
        )  # list of (N, D), length B*T
        f = torch.stack(per, dim=0)  # (B*T, N, D)
        n_tokens, dim = f.shape[1], f.shape[2]
        return f.reshape(len(samples), num_frames * n_tokens, dim)  # (B, T*N, D)


@register_target("siglip", "image")
class ImageEncoderTarget(BaseTarget):
    """Frozen external per-image encoder (SigLIP2 / DINOv2 / CLIP …) via ``AutoModel``.

    Each of the T frames is encoded independently and the per-frame tokens are
    concatenated -> ``N_t = T * tokens_per_frame``.
    """

    kind = "repr"
    # calibrated for SigLIP2 on LIBERO+Ego4D (DINO overrides below with its own)
    feat_mean = 0.0464
    feat_std = 1.1230

    def __init__(self, global_config, vlm_interface=None, input_dim=None):
        super().__init__()
        from transformers import AutoImageProcessor, AutoModel

        tcfg = global_config.framework.gen_model.target
        path = tcfg.get("path", None)
        assert path, "target.type in {siglip,dino,image} requires target.path=<encoder path/id>"
        full = AutoModel.from_pretrained(path, dtype=torch.bfloat16)
        self.encoder = getattr(full, "vision_model", full)
        self.image_processor = AutoImageProcessor.from_pretrained(path)
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self._target_dim = int(self.encoder.config.hidden_size)
        self.num_target_frames = resolve_num_target_frames(global_config)

    def feature_dim(self):
        return self._target_dim

    @torch.no_grad()
    def _encode_frames(self, frames):
        inputs = self.image_processor(images=frames, return_tensors="pt")
        ref = next(self.encoder.parameters())
        pixel_values = inputs["pixel_values"].to(device=ref.device, dtype=ref.dtype)
        out = self.encoder(pixel_values=pixel_values)
        f = out.last_hidden_state  # (T, N, D)
        return f.reshape(f.shape[0] * f.shape[1], f.shape[2])  # (T*N, D)

    @torch.no_grad()
    def count_tokens(self):
        from PIL import Image

        dummy = Image.new("RGB", (64, 64))
        per_frame = self._encode_frames([dummy]).shape[0]
        return int(per_frame) * int(self.num_target_frames)

    @torch.no_grad()
    def encode(self, target_frames):
        samples = self._normalize_frames(target_images=target_frames)
        return torch.stack([self._encode_frames(fr) for fr in samples], dim=0)


@register_target("dino")
class DinoTarget(ImageEncoderTarget):
    """Frozen DINOv2 (e.g. ``facebook/dinov2-giant``) per-image encoder — a named
    ``ImageEncoderTarget`` for DINOv2. ``last_hidden_state`` is ``[CLS] + patch``
    tokens; token 0 is the CLS token, KEPT by default (byte-identical to the
    generic image target). Set ``target.strip_cls: true`` to drop it and regress
    patch tokens only.

    Uses DINOv2's OWN ``AutoImageProcessor`` (its native resize + center-crop),
    exactly like the generic image target — no forced stretch / alignment with the
    other encoders. The ONLY thing this subclass adds over the base is the optional
    CLS-token strip. Loading / feature_dim / count_tokens are all inherited.
    """

    # calibrated for DINOv2-giant on LIBERO+Ego4D (overrides the SigLIP defaults above)
    feat_mean = -0.0184
    feat_std = 1.4617

    def __init__(self, global_config, vlm_interface=None, input_dim=None):
        super().__init__(global_config, vlm_interface=vlm_interface, input_dim=input_dim)
        self.strip_cls = bool(global_config.framework.gen_model.target.get("strip_cls", False))

    @torch.no_grad()
    def _encode_frames(self, frames):
        # Use DINOv2's own processor defaults (native resize + center-crop); this
        # subclass only optionally drops the CLS token. count_tokens() measures the
        # per-frame count via this same path, so num_gen_queries stays consistent.
        if not self.strip_cls:
            return super()._encode_frames(frames)
        inputs = self.image_processor(images=frames, return_tensors="pt")
        ref = next(self.encoder.parameters())
        pixel_values = inputs["pixel_values"].to(device=ref.device, dtype=ref.dtype)
        out = self.encoder(pixel_values=pixel_values)
        f = out.last_hidden_state  # (T, N, D); N = 1 CLS + patches
        f = f[:, 1:, :]  # drop the CLS token -> patch tokens only
        return f.reshape(f.shape[0] * f.shape[1], f.shape[2])  # (T * N', D)


@register_target("video")
class VideoEncoderTarget(BaseTarget):
    """Frozen video encoder (V-JEPA2). One repr for the whole T-frame clip.

    Its token count scales with the number of target frames T:
    ``ceil(T / tubelet_size)`` temporal tokens times the spatial patch grid.
    The clip is encoded at its actual length (frames duplicated only to fill a
    whole tubelet), NOT resampled to the encoder's nominal ``frames_per_clip``.
    """

    kind = "repr"
    per_frame_decomposable = False  # tubelets aggregate over time; tokens mix frames

    def __init__(self, global_config, vlm_interface=None, input_dim=None):
        super().__init__()
        from transformers import AutoModel, AutoVideoProcessor

        tcfg = global_config.framework.gen_model.target
        path = tcfg.get("path", None)
        assert path, "target.type in {vjepa,video} requires target.path=<encoder path/id>"
        self.encoder = AutoModel.from_pretrained(path, dtype=torch.bfloat16)
        self.video_processor = AutoVideoProcessor.from_pretrained(path)
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self._target_dim = int(self.encoder.config.hidden_size)
        self.tubelet_size = int(self.encoder.config.tubelet_size)
        patch_size = int(self.encoder.config.patch_size)
        crop_size = getattr(self.encoder.config, "crop_size", None)
        if crop_size is None:
            crop_size = getattr(self.encoder.config, "image_size", None)
        if crop_size is None:
            raise ValueError("V-JEPA config must define crop_size or image_size")
        crop_size = int(crop_size)
        if crop_size % patch_size != 0:
            raise ValueError(f"V-JEPA spatial grid not divisible: crop={crop_size}, patch={patch_size}")
        self._patch_grid = (crop_size // patch_size) ** 2
        # Adaptive token count: follow the ACTUAL number of target frames T (config-derived,
        # matching the dataloader and the other repr targets) rather than the encoder's nominal
        # frames_per_clip. The encoder handles a variable clip length natively; we only duplicate
        # the last frame(s) to fill a whole tubelet (see ``_encode_clip``), so the temporal token
        # count is ceil(T / tubelet_size).
        num_target_frames = resolve_num_target_frames(global_config)
        temporal_tokens = tubelet_padded_num_frames(num_target_frames, self.tubelet_size) // self.tubelet_size
        self._num_tokens = temporal_tokens * self._patch_grid

    def feature_dim(self):
        return self._target_dim

    @torch.no_grad()
    def count_tokens(self):
        return int(self._num_tokens)

    @torch.no_grad()
    def _encode_clip(self, frames):
        import numpy as np

        arr = np.stack([np.array(im.convert("RGB")) for im in frames])  # (T,H,W,C)
        clip = torch.from_numpy(arr).permute(0, 3, 1, 2)  # (T,C,H,W)
        # Encode the frames we were actually given (no resample to a fixed clip length);
        # duplicate the last frame(s) only to fill a whole tubelet so none are dropped by
        # the stride-``tubelet`` Conv3d. Yields ceil(T / tubelet) temporal tokens.
        padded = tubelet_padded_num_frames(clip.shape[0], self.tubelet_size)
        if padded > clip.shape[0]:
            clip = torch.cat([clip, clip[-1:].repeat(padded - clip.shape[0], 1, 1, 1)], dim=0)
        # Use V-JEPA2's OWN video processor defaults (native resize + center-crop) — no forced
        # stretch / alignment with the other encoders. Its native crop equals the model's
        # spatial ``crop_size``, i.e. the size ``_patch_grid`` / ``_num_tokens`` were computed
        # from in __init__, so the token count stays consistent. (Tubelet temporal padding above
        # is unrelated to the spatial processor and is kept.)
        proc = self.video_processor(clip, return_tensors="pt")
        ref = next(self.encoder.parameters())
        proc = {
            k: v.to(device=ref.device, dtype=ref.dtype if v.is_floating_point() else v.dtype) for k, v in proc.items()
        }
        emb = self.encoder.get_vision_features(**proc)  # (1, N_video, D)
        return emb[0]

    @torch.no_grad()
    def encode(self, target_frames):
        samples = self._normalize_frames(target_images=target_frames)
        return torch.stack([self._encode_clip(fr) for fr in samples], dim=0)


@register_target("vjepa")
class VJepaTarget(VideoEncoderTarget):
    """Frozen V-JEPA2 (e.g. ``vjepa2-vitg-fpc64-256``) video encoder — a named
    ``VideoEncoderTarget`` for V-JEPA2. One repr for the whole T-frame clip via
    ``get_vision_features`` (the paper uses ``vjepa2-vitg-fpc64-256``). Its token
    count is ``ceil(T / tubelet_size)`` temporal
    tokens times the spatial patch grid, adaptive to the requested horizon T.
    Loading / feature_dim / encode are inherited.
    """

    # calibrated for V-JEPA2 vitg-fpc64-256 on LIBERO+Ego4D
    feat_mean = -0.0334
    feat_std = 2.8600


@register_target("pixel")
class PixelTarget(BaseTarget):
    """Raw-pixel target: each frame is resized to ``resolution`` and patchified.

    Produces ``(B, T * (R/P)^2, 3*P*P)`` patch tokens in [-1, 1].  The simplest
    possible target (\"even raw pixels\"), useful as a sanity baseline.
    """

    kind = "repr"

    def __init__(self, global_config, vlm_interface=None, input_dim=None):
        super().__init__()
        tcfg = global_config.framework.gen_model.target
        self.resolution = int(tcfg.get("resolution", 224))
        self.patch = int(tcfg.get("patch_size", 16))
        assert self.resolution % self.patch == 0, "target.resolution must be divisible by patch_size"
        self.grid = self.resolution // self.patch
        self._target_dim = 3 * self.patch * self.patch
        self.num_target_frames = resolve_num_target_frames(global_config)

    def feature_dim(self):
        return self._target_dim

    @torch.no_grad()
    def count_tokens(self):
        return self.grid * self.grid * int(self.num_target_frames)

    @torch.no_grad()
    def _encode_frames(self, frames):
        import numpy as np

        toks = []
        for im in frames:
            arr = np.array(im.convert("RGB").resize((self.resolution, self.resolution)))
            t = torch.from_numpy(arr).float().div_(255.0).sub_(0.5).div_(0.5)  # (R,R,3) in [-1,1]
            t = t.permute(2, 0, 1)  # (3,R,R)
            # (3,R,R) -> (num_patches, 3*P*P)
            p = t.unfold(1, self.patch, self.patch).unfold(2, self.patch, self.patch)  # (3,g,g,P,P)
            p = p.permute(1, 2, 0, 3, 4).reshape(self.grid * self.grid, -1)
            toks.append(p)
        return torch.cat(toks, dim=0)  # (T*grid*grid, 3*P*P)

    @torch.no_grad()
    def encode(self, target_frames):
        samples = self._normalize_frames(target_images=target_frames)
        return torch.stack([self._encode_frames(fr) for fr in samples], dim=0)


@register_target("lingbot", "lingbot_vision")
class LingbotVisionTarget(BaseTarget):
    """Frozen LingBot-Vision (ViT-g/16) per-image encoder producing dense patch
    tokens. NOT a transformers ``AutoModel`` — loaded via the ``lingbot_vision``
    package (``load_pretrained_backbone`` + ``extract_patch_tokens``) from a local
    backbone-only ``model.pt``. ``kind='repr'``; ``feature_dim`` = the backbone's
    ``embed_dim``; each frame yields ``(resolution/patch)^2`` patch tokens (NO CLS),
    so ``num_gen_queries: auto`` works. Config: ``target.path`` (model dir with
    ``model.pt``), ``target.variant`` (default ``giant``), ``target.resolution``
    (default 512, snapped to the patch size).
    """

    kind = "repr"
    # calibrated for lingbot-vision ViT-giant on LIBERO+Ego4D
    feat_mean = -0.0076
    feat_std = 0.8641

    def __init__(self, global_config, vlm_interface=None, input_dim=None):
        super().__init__()
        from lingbot_vision import load_pretrained_backbone

        tcfg = global_config.framework.gen_model.target
        path = tcfg.get("path", None)
        assert path, "target.type='lingbot' requires target.path (lingbot-vision dir with model.pt)"
        variant = str(tcfg.get("variant", "giant"))
        backbone, embed_dim = load_pretrained_backbone(
            path,
            variant=variant,
            device="cpu",
            dtype=torch.bfloat16,
            checkpoint_file="model.pt",
            local_files_only=True,
            verbose=False,
        )
        backbone.requires_grad_(False)
        backbone.eval()
        self.backbone_net = backbone  # nn.Module submodule -> moved by model.to(device)
        self._target_dim = int(embed_dim)
        self.patch = int(getattr(backbone, "patch_size", 16))
        res = int(tcfg.get("resolution", 512))
        self.resolution = (res // self.patch) * self.patch  # snap down to a multiple of patch
        self.num_target_frames = resolve_num_target_frames(global_config)

    def feature_dim(self):
        return self._target_dim

    @torch.no_grad()
    def _prep(self, frames):
        """PIL frames -> normalized tensor (F, 3, R, R), mirroring lingbot_vision.load_image (square)."""
        import numpy as np
        from lingbot_vision.preprocess import IMAGENET_MEAN, IMAGENET_STD
        from PIL import Image

        arr = np.stack(
            [np.array(im.convert("RGB").resize((self.resolution, self.resolution), Image.BILINEAR)) for im in frames]
        )
        t = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(0, 3, 1, 2)  # (F, 3, R, R)
        return (t - IMAGENET_MEAN) / IMAGENET_STD

    @torch.no_grad()
    def _encode_frames(self, frames):
        from lingbot_vision import extract_patch_tokens

        ref = next(self.backbone_net.parameters())
        pt, _grid = extract_patch_tokens(self.backbone_net, self._prep(frames), str(ref.device), ref.dtype)
        return pt.reshape(pt.shape[0] * pt.shape[1], pt.shape[2])  # (F * N, D)

    @torch.no_grad()
    def count_tokens(self):
        from PIL import Image

        per_frame = self._encode_frames([Image.new("RGB", (self.resolution, self.resolution))]).shape[0]
        return int(per_frame) * int(self.num_target_frames)

    @torch.no_grad()
    def encode(self, target_frames):
        samples = self._normalize_frames(target_images=target_frames)
        return torch.stack([self._encode_frames(fr) for fr in samples], dim=0)


# ──────────────────────────────────────────────────────────────────────
#  VAE latent target (diffusers ``AutoencoderKLWan``). Needs diffusers-format
#  weights at target.path (+ optional target.vae_subfolder).
# ──────────────────────────────────────────────────────────────────────
class _VaeTargetBase(BaseTarget):
    """VAE encoder: frames -> native latent GRID ``(B, C, T, H, W)``.

    The heads flatten the grid to tokens via :func:`grid_to_tokens`.
    ``feature_dim`` = latent channels C; the concrete target resolves the token
    count from its VAE compression factors.
    """

    kind = "vae"

    def __init__(self, global_config, vlm_interface=None, input_dim=None):
        super().__init__()
        tcfg = global_config.framework.gen_model.target
        res = tcfg.get("resolution", 256)
        if isinstance(res, (int, float)):
            self.H = self.W = int(res)
        else:  # [H, W] as a list / tuple / OmegaConf ListConfig
            self.H, self.W = int(res[0]), int(res[1])
        self._latent_dim = None
        self._num_tokens = None

    def feature_dim(self):
        return int(self._latent_dim)

    def _apply(self, fn, recurse=True):
        """Move/cast the list-wrapped VAE together with the module.

        The VAE is held in a 1-element list (``self._vae``) so it is excluded from
        the head's ``state_dict`` (frozen, not a checkpointed param). That also
        hides it from ``nn.Module.to()`` (which only walks ``_modules``), leaving
        it on CPU while the rest of the model moves to CUDA — a device mismatch at
        encode/consume time. Re-apply the same transform to the VAE here so
        ``model.to(device)`` moves it too, without adding it to ``state_dict``.
        """
        super()._apply(fn, recurse)
        vae = getattr(self, "_vae", None)
        if vae:
            vae[0]._apply(fn)
        return self

    @torch.no_grad()
    def count_tokens(self):
        if self._num_tokens is None:
            raise ValueError(
                f"{type(self).__name__} did not resolve its token count; set "
                "framework.gen_model.targets[i].num_gen_queries explicitly."
            )
        return int(self._num_tokens)


@register_target("wan_vae")
class WanVaeTarget(_VaeTargetBase):
    """Wan2.2 3D video VAE via diffusers ``AutoencoderKLWan``.

    Needs diffusers-format weights (e.g. ``playground/Pretrained_models/Wan2.2-TI2V-5B-Diffusers``):
    set ``target.path`` to the repo root with ``target.vae_subfolder: vae``, or
    point ``target.path`` directly at the ``vae`` dir. Encoding + latent
    normalization follow the official ``WanPipeline`` scaling.
    """

    per_frame_decomposable = False  # 3D VAE compresses over time; a latent frame mixes RGB frames
    # calibrated for Wan2.2 VAE latents on LIBERO+Ego4D (already ~unit-var internally; std<1)
    feat_mean = -0.0190
    feat_std = 0.5649

    def __init__(self, global_config, vlm_interface=None, input_dim=None):
        super().__init__(global_config, vlm_interface, input_dim)
        from diffusers import AutoencoderKLWan
        from diffusers.video_processor import VideoProcessor

        tcfg = global_config.framework.gen_model.target
        path = tcfg.get("path", None)
        assert path, "target.type='wan_vae' requires target.path (diffusers-format Wan weights)"
        sub = tcfg.get("vae_subfolder", None)
        vae = AutoencoderKLWan.from_pretrained(path, torch_dtype=torch.bfloat16, **({"subfolder": sub} if sub else {}))
        vae.requires_grad_(False)
        vae.eval()
        self._vae = [vae]  # list-wrap: loaded frozen, not saved in the head state_dict
        self._latent_dim = int(vae.config.z_dim)
        # Spatial rounding factor for the VideoProcessor = the VAE's true spatial compression,
        # ``config.scale_factor_spatial`` (Wan2.2 = 16). ``2 ** len(temperal_downsample)`` = 8
        # undercounts by the VAE's internal ``patch_size=2`` stage; the gap only bites at a
        # resolution that is a multiple of 8 but not 16 (no-op at the 256 default). Mirrors the
        # official ``WanPipeline.__init__``; falls back to ``2 ** len(temperal_downsample)`` when
        # the config lacks the key.
        vae_scale = int(getattr(vae.config, "scale_factor_spatial", 2 ** len(vae.temperal_downsample)))
        self._vp = VideoProcessor(vae_scale_factor=vae_scale)
        num_target_frames = resolve_num_target_frames(global_config)
        temporal_scale = int(
            getattr(vae.config, "scale_factor_temporal", 2 ** sum(bool(v) for v in vae.temperal_downsample))
        )
        if self.H % vae_scale != 0 or self.W % vae_scale != 0:
            raise ValueError(f"Wan target resolution {(self.H, self.W)} must be divisible by {vae_scale}")
        latent_t = (num_target_frames - 1) // temporal_scale + 1
        self._num_tokens = latent_t * (self.H // vae_scale) * (self.W // vae_scale)

    @property
    def vae(self):
        return self._vae[0]

    @torch.no_grad()
    def encode(self, target_frames):
        samples = self._normalize_frames(target_images=target_frames)
        ref = next(self.vae.parameters())
        vids = [self._vp.preprocess_video(list(fr), height=self.H, width=self.W).squeeze(0) for fr in samples]
        video = torch.stack(vids, dim=0).to(device=ref.device, dtype=ref.dtype)  # (B,C,T,H,W)
        lat = self.vae.encode(video).latent_dist.mode()  # (B, z, T', H', W')
        mean = torch.tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1).to(lat)
        std = torch.tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1).to(lat)
        lat = (lat - mean) / std
        return lat.float()  # (B, C, T, H, W) native latent grid
