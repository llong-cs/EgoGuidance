# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Loss axis of the generation head.

Two families, kept apart on purpose:

  * **Pointwise losses** (``mse`` / ``cosine`` / ``cosine_magnitude`` /
    ``smooth_l1``) — compare a deterministic prediction against the target
    token-by-token (``regression`` mode).  Look them up with
    :func:`get_pointwise_loss`.

  * **Flow matching** (rectified flow) — NOT a standalone ``pred vs target``
    loss but a training objective that also owns the noising process; used by
    the ``diffusion`` mode's expert.  Implemented as :class:`FlowMatching`
    (``x_t = (1-t)*noise + t*x0``, ``v = x0 - noise``); the timestep distribution
    is chosen per target by :func:`build_flow_matching` (RAE recipe for high-dim
    reprs, WAN/fastwam recipe for VAE latents), defaulting back to the action
    head's shifted-Beta schedule only when ``loss.timestep_schedule=beta``.
"""

import math

import torch
import torch.nn.functional as F
from torch.distributions import Beta

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

LOSS_REGISTRY = {}


def register_loss(name):
    def _wrap(fn):
        LOSS_REGISTRY[name] = fn
        return fn

    return _wrap


@register_loss("mse")
def _mse(pred, target):
    return F.mse_loss(pred, target)


@register_loss("smooth_l1")
def _smooth_l1(pred, target):
    return F.smooth_l1_loss(pred, target)


@register_loss("cosine")
def _cosine(pred, target):
    return (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()


@register_loss("cosine_magnitude")
def _cosine_magnitude(
    pred,
    target,
    *,
    feat_std,
    cosine_weight=1.0,
    magnitude_weight=1.0,
    eps=1e-8,
):
    """Cosine direction loss plus feature-scale-normalised vector-norm MSE.

    Both terms operate on the raw feature vectors.  In particular, this loss
    does not centre or standardise ``target`` with the per-target feature
    statistics; ``feat_std`` is used only to make the vector-norm error
    dimensionless::

        scale = sqrt(D) * feat_std
        loss = cosine_weight * mean(1 - cosine(pred, target))
             + magnitude_weight * mse(norm(pred) / scale, norm(target) / scale)

    The magnitude term supplies a useful gradient when the target vector is
    zero, where the cosine term alone is constant with respect to ``pred``.
    """
    feature_dim = int(pred.shape[-1])
    if feature_dim <= 0:
        raise ValueError(f"cosine_magnitude requires a positive feature dimension, got {feature_dim}.")
    scale = math.sqrt(feature_dim) * feat_std
    cosine_loss = (1.0 - F.cosine_similarity(pred, target, dim=-1, eps=eps)).mean()
    pred_norm = torch.linalg.vector_norm(pred, dim=-1) / scale
    target_norm = torch.linalg.vector_norm(target, dim=-1) / scale
    magnitude_loss = F.mse_loss(pred_norm, target_norm)
    return cosine_weight * cosine_loss + magnitude_weight * magnitude_loss


def _finite_float(value, *, name, positive=False, nonnegative=False):
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"cosine_magnitude loss.{name} must be a finite number, got {value!r}.") from exc
    if not math.isfinite(value):
        raise ValueError(f"cosine_magnitude loss.{name} must be finite, got {value!r}.")
    if positive and value <= 0:
        raise ValueError(f"cosine_magnitude loss.{name} must be > 0, got {value!r}.")
    if nonnegative and value < 0:
        raise ValueError(f"cosine_magnitude loss.{name} must be >= 0, got {value!r}.")
    return value


def get_pointwise_loss(name, *, feat_std=None, loss_cfg=None):
    """Build a ``fn(pred, target) -> scalar`` pointwise loss by name.

    ``cosine_magnitude`` additionally consumes its per-target loss config and
    the effective ``feat_std`` (explicit target override first, calibrated
    target-class value otherwise).  Other pointwise losses are returned as-is and
    ignore these builder-only arguments.
    """
    if name not in LOSS_REGISTRY:
        raise NotImplementedError(
            f"pointwise loss `{name}` is not implemented. Available: {sorted(LOSS_REGISTRY)}. "
            f"(`flow_matching` is not a pointwise loss — see FlowMatching.)"
        )
    loss_fn = LOSS_REGISTRY[name]
    if name != "cosine_magnitude":
        return loss_fn

    cfg = loss_cfg or {}
    get = cfg.get if hasattr(cfg, "get") else (lambda key, default=None: getattr(cfg, key, default))
    normalization = str(get("magnitude_normalization", "target_feat_std")).lower()
    if normalization != "target_feat_std":
        raise ValueError(
            "cosine_magnitude loss.magnitude_normalization must be 'target_feat_std', " f"got {normalization!r}."
        )
    if feat_std is None:
        raise ValueError(
            "cosine_magnitude requires an effective_feat_std from the regression target; "
            "the loss is not supported by callers that do not bind target statistics."
        )
    feat_std = _finite_float(feat_std, name="effective_feat_std", positive=True)
    cosine_weight = _finite_float(get("cosine_weight", 1.0), name="cosine_weight", nonnegative=True)
    magnitude_weight = _finite_float(get("magnitude_weight", 1.0), name="magnitude_weight", nonnegative=True)
    if cosine_weight == 0 and magnitude_weight == 0:
        raise ValueError("cosine_magnitude requires cosine_weight or magnitude_weight to be > 0.")
    eps = _finite_float(get("eps", 1e-8), name="eps", positive=True)

    def _configured_cosine_magnitude(pred, target):
        return loss_fn(
            pred,
            target,
            feat_std=feat_std,
            cosine_weight=cosine_weight,
            magnitude_weight=magnitude_weight,
            eps=eps,
        )

    return _configured_cosine_magnitude


class FlowMatching:
    """Rectified-flow noising + velocity target, shared by the generative paths.

    The noising process is always rectified flow::

        x_t = (1 - t) * noise + t * x0        # t = DATA fraction (t=1 clean, t=0 noise)
        v   = x0 - noise                      # ground-truth velocity

    Only the *timestep distribution* is configurable, because the best noise
    schedule depends on what the diffuser predicts.  Build one per target via
    :func:`build_flow_matching`; the three schedules are:

      * ``schedule="beta"``            — the shifted-Beta schedule shared with
        ``LayerwiseFlowmatchingActionHead``:
        ``t = (noise_s - Beta(alpha, beta)) / noise_s``.
      * ``schedule="logit_normal"``    — RAE / SD3 recipe for high-dim reprs:
        sample the NOISE level ``sigma = sigmoid(N(ln_mu, ln_sigma))``, apply the
        resolution ``shift``, then ``t = 1 - sigma``.
      * ``schedule="shifted_uniform"`` — WAN / fastwam recipe for VAE latents:
        ``sigma = phi(U(0,1), shift)``, then ``t = 1 - sigma``.

    ``shift`` is the flow-matching resolution shift, applied in SIGMA (noise)
    space: ``phi(sigma, s) = s*sigma / (1 + (s-1)*sigma)``; ``s >= 1`` biases
    sampling toward higher noise.  ``loss_timestep_weight="gaussian_mid"``
    additionally re-weights the per-sample loss toward mid-noise (fastwam).

    NOTE on conventions: this code's ``t`` is the DATA fraction, the OPPOSITE of RAE /
    WAN / fastwam whose ``sigma`` is the NOISE fraction (``sigma = 1 - t``).  All
    ported recipes therefore sample/shift ``sigma`` and return ``1 - sigma``.
    """

    def __init__(
        self,
        beta_alpha=1.5,
        beta_beta=1.0,
        noise_s=0.999,
        num_timestep_buckets=1000,
        schedule="beta",
        ln_mu=0.0,
        ln_sigma=1.0,
        shift=1.0,
        loss_timestep_weight="none",
    ):
        self.beta_dist = Beta(beta_alpha, beta_beta)
        self.noise_s = float(noise_s)
        self.num_timestep_buckets = int(num_timestep_buckets)
        self.schedule = str(schedule).lower()
        self.ln_mu = float(ln_mu)
        self.ln_sigma = float(ln_sigma)
        self.shift = float(shift)
        self.loss_timestep_weight = str(loss_timestep_weight).lower()
        if self.schedule not in ("beta", "logit_normal", "shifted_uniform"):
            raise ValueError(f"FlowMatching schedule must be beta|logit_normal|shifted_uniform, got {schedule!r}.")

    def sample_time(self, batch_size, device, dtype):
        """Draw the data-fraction timestep ``t`` (t=1 clean, t=0 noise)."""
        if self.schedule == "beta":
            sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
            return (self.noise_s - sample) / self.noise_s
        # Sample the NOISE level ``sigma``, shift it, map back to the data fraction.
        if self.schedule == "logit_normal":
            z = torch.randn(batch_size, device=device, dtype=dtype)
            sigma = torch.sigmoid(z * self.ln_sigma + self.ln_mu)
        else:  # shifted_uniform
            sigma = torch.rand(batch_size, device=device, dtype=dtype)
        if self.shift != 1.0:
            s = self.shift
            sigma = s * sigma / (1.0 + (s - 1.0) * sigma)
        return 1.0 - sigma

    def add_noise(self, x0: torch.Tensor):
        """x0: clean target — tokens ``(B, N, D)`` or a latent grid ``(B, C, T, H, W)``.

        Returns ``(x_t, velocity, t_discretized)`` where ``t_discretized`` is a
        ``(B,)`` long tensor of bucket indices for the DiT's ada-norm timestep.
        Broadcasting is rank-agnostic so both token and grid targets work.
        """
        t = self.sample_time(x0.shape[0], device=x0.device, dtype=x0.dtype)  # (B,)
        return self.add_noise_with_t(x0, t)

    def add_noise_with_t(self, x0: torch.Tensor, t: torch.Tensor):
        """Same rectified-flow noising as :meth:`add_noise` but with an EXTERNALLY
        supplied data-fraction timestep ``t`` (shape ``(B,)`` in [0, 1], t=1 clean).

        Used by the diffusion ``isolation=off`` (joint) path: the generation
        expert must be noised at the SAME ``t`` the action expert is trained /
        denoised at, so the framework samples one ``t`` and hands it to both.
        This deliberately bypasses this schedule's own ``sample_time`` (and thus
        its shift / loss-weight tuning) — the joint variant shares the action
        head's timestep clock by design.
        """
        noise = torch.randn_like(x0)
        t = t.to(device=x0.device, dtype=x0.dtype)
        t_b = t.view((x0.shape[0],) + (1,) * (x0.dim() - 1))  # (B,1,...) broadcast
        x_t = (1.0 - t_b) * noise + t_b * x0
        velocity = x0 - noise
        t_discretized = (t * self.num_timestep_buckets).long()
        return x_t, velocity, t_discretized

    def loss(self, pred_velocity: torch.Tensor, velocity: torch.Tensor, t_discretized=None):
        """Velocity MSE.  With ``loss_timestep_weight='gaussian_mid'`` and the
        per-sample timesteps, re-weight toward mid-noise (fastwam); otherwise the
        unweighted global-mean MSE (RAE / WAN)."""
        if self.loss_timestep_weight == "gaussian_mid" and t_discretized is not None:
            dims = tuple(range(1, pred_velocity.dim()))
            mse = ((pred_velocity - velocity) ** 2).mean(dim=dims)  # (B,)
            w = self._gaussian_mid_weight(t_discretized.to(mse.dtype))  # (B,)
            return (mse * w).mean()
        return ((pred_velocity - velocity) ** 2).mean()

    def _gaussian_mid_weight(self, t):
        """fastwam bell weight centred at mid-noise, batch-normalised to mean 1."""
        mid = self.num_timestep_buckets / 2.0
        n = float(self.num_timestep_buckets)
        w = torch.exp(-2.0 * ((t - mid) / n) ** 2)
        return w / (w.mean() + 1e-6)


def _tokens_per_frame(target, fallback=None):
    """Single-frame token count of a repr target (``feature_dim * this`` = per-frame D).

    Returns 1 (or ``fallback``) when the target cannot report its token count.
    VAE targets use a fixed shift and never reach the repr dimension shift.
    """
    try:
        n = int(target.count_tokens())
    except Exception:
        n = None
    if n is None:
        return int(fallback) if fallback else 1
    frames = int(getattr(target, "num_target_frames", 1) or 1)
    return max(1, n // max(frames, 1))


def build_flow_matching(gen_cfg, target, *, tokens_hint=None, num_timestep_buckets=1000):
    """Build a :class:`FlowMatching` whose timestep distribution is chosen from
    the target kind — the single place the per-target schedule is decided.

    * ``kind == "repr"`` (auto / siglip / dino / vjepa / lingbot / pixel) -> **RAE recipe**:
      logit-normal(``ln_mu``, ``ln_sigma``) over the noise level + a dimension
      shift ``s = sqrt(feature_dim * tokens_per_frame / shift_base)`` (RAE's
      resolution shift), unweighted velocity MSE.
    * ``kind == "vae"`` (wan_vae) -> **WAN / fastwam recipe**: default
      shifted-uniform sampling with a fixed ``vae_shift`` + a mid-noise Gaussian
      loss weight (fastwam); set ``loss.timestep_schedule=logit_normal`` for the
      WAN/SD3 variant (logit-normal + fixed shift, unweighted).

    Config keys (all optional, under ``framework.gen_model.loss``):
      ``timestep_schedule``    : auto | beta | logit_normal | shifted_uniform
      ``ln_mu`` / ``ln_sigma`` : logit-normal params (default 0.0 / 1.0)
      ``shift``                : explicit sigma-space shift (>=1); skips auto-shift
      ``shift_base``           : RAE dimension reference for repr (default 4096)
      ``vae_shift``            : fixed shift for vae targets (default 5.0)
      ``loss_timestep_weight`` : none | gaussian_mid (default derived from schedule)
    """
    lcfg = gen_cfg.get("loss", {}) or {}
    get = lcfg.get if hasattr(lcfg, "get") else (lambda k, d=None: getattr(lcfg, k, d))
    kind = str(getattr(target, "kind", "repr")).lower()

    schedule = str(get("timestep_schedule", "auto")).lower()
    if schedule == "auto":
        schedule = "logit_normal" if kind == "repr" else "shifted_uniform"

    # ``shift`` lives in sigma (noise) space; the beta schedule ignores it.
    shift_override = get("shift", None)
    if schedule == "beta":
        shift = 1.0
    elif shift_override is not None:
        shift = float(shift_override)
    elif kind == "vae":
        shift = float(get("vae_shift", 5.0))
    else:  # repr -> RAE dimension shift
        d = int(target.feature_dim()) * _tokens_per_frame(target, fallback=tokens_hint)
        shift = max(1.0, math.sqrt(float(d) / float(get("shift_base", 4096))))

    default_weight = "gaussian_mid" if (kind == "vae" and schedule == "shifted_uniform") else "none"
    loss_timestep_weight = str(get("loss_timestep_weight", default_weight)).lower()

    fm = FlowMatching(
        beta_alpha=float(get("noise_beta_alpha", 1.5)),
        beta_beta=float(get("noise_beta_beta", 1.0)),
        noise_s=float(get("noise_s", 0.999)),
        num_timestep_buckets=int(num_timestep_buckets),
        schedule=schedule,
        ln_mu=float(get("ln_mu", 0.0)),
        ln_sigma=float(get("ln_sigma", 1.0)),
        shift=shift,
        loss_timestep_weight=loss_timestep_weight,
    )
    logger.info(
        f"[FlowMatching] target.kind={kind} schedule={schedule} shift={shift:.4f} "
        f"weight={loss_timestep_weight} buckets={num_timestep_buckets}"
    )
    return fm
