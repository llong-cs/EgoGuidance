# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Generation head factory.

A generation head trains a *generation loss* against features of the future
frame(s).  It is configured under ``framework.gen_model`` along these axes:

    mode      : regression | diffusion                (top-level dispatch)
    backbone  : vlm | expert                          (who produces the query states)
    decoder   : mlp                                   (regression only; diffusion has no decoder)
    target    : auto | siglip (image) | dino | vjepa (video) | lingbot | pixel | wan_vae
    loss      : mse | cosine | cosine_magnitude | smooth_l1 (regression) | flow_matching (diffusion)

The head is decoupled from the VLM backbone: it reads
``global_config.framework.gen_model`` and, at forward time, receives
``query_states`` and/or ``vl_embs_list`` plus the target frames (mirroring the
action_model contract).  The concrete heads live in ``regression.py`` /
``diffusion.py`` and register themselves by *mode*.

Targets: ``framework.gen_model.targets`` is the only way to define generation
targets — a LIST of per-target specs, each ``{target, decoder, loss,
num_gen_queries, weight}``. A single target is a 1-element list. Every config builds
through :class:`~starVLA.model.modules.gen_model.multi.MultiGenHead` (one child head +
one query group / expert per target, aggregated by per-target weights). Per-target
defaults live in :data:`_GEN_TARGET_DEFAULTS`.
"""

from omegaconf import OmegaConf

GEN_HEAD_REGISTRY = {}


def register_gen_head(name):
    def _wrap(cls):
        GEN_HEAD_REGISTRY[name] = cls
        return cls

    return _wrap


# Canonical per-target defaults. Every ``targets[i]`` entry is deep-merged OVER this
# template at normalize time, so each entry is complete before a child head reads it.
# (OmegaConf list merge REPLACES, so a dataclass-default targets list cannot fill user
# entries element-wise; the fill therefore lives here in code.)
_GEN_TARGET_DEFAULTS = {
    "num_gen_queries": "auto",
    "target": {
        "type": "auto",
        "path": None,
        "vae_subfolder": None,
        "resolution": 224,
        "pool": "none",
        "patch_size": 16,
        "freeze": True,
    },
    "decoder": {
        "type": "mlp",
        "num_layers": 3,
        "freeze": None,
    },
    "loss": {"type": "mse"},
}

# Per-target keys that are not accepted at the gen_model top level (they live inside each
# targets entry).
_SINGULAR_GEN_KEYS = ("target", "decoder", "loss", "num_gen_queries")


def _raw_node(node):
    """Underlying OmegaConf node, unwrapping an AccessTrackedConfig wrapper if present."""
    return node._cfg if hasattr(node, "_cfg") else node


def normalize_gen_config(config):
    """Normalise ``framework.gen_model`` into the canonical **targets-only** schema.

    Ends with a materialised ``targets`` LIST — each entry completed by a deep-merge over
    :data:`_GEN_TARGET_DEFAULTS` — and NO top-level ``target``/``decoder``/``loss``/
    ``num_gen_queries``. Idempotent. Called ONCE on the (parent) config in
    :func:`get_gen_model`; the per-child config that :class:`MultiGenHead` synthesises keeps
    its ``target``/``decoder``/``loss`` at the child's top level and is NOT re-normalised.

    Raises when a per-target key is set at the gen_model top level (it would otherwise be
    silently ignored), or when the unsupported flat ``head_type`` schema is used.
    """
    g = _raw_node(config.framework.gen_model)

    if g.get("head_type", None) is not None:
        raise ValueError(
            "framework.gen_model.head_type is not supported. Set framework.gen_model.mode "
            "(regression | diffusion) and describe each target in framework.gen_model.targets."
        )
    if g.get("mode", None) is None:
        g.mode = "regression"

    stray = [k for k in _SINGULAR_GEN_KEYS if g.get(k, None) is not None]
    if stray:
        raise ValueError(
            f"framework.gen_model top-level {stray} is not supported. Put it inside a targets "
            "entry, e.g. framework.gen_model.targets: [{target: {...}, decoder: {...}, "
            "loss: {...}, num_gen_queries: ...}]."
        )

    targets = g.get("targets", None)
    if targets:
        entries = targets if isinstance(targets, list) else OmegaConf.to_container(targets, resolve=True)
        if isinstance(entries, dict):
            # accessed-only snapshots can round-trip a list as an int-keyed map; coerce back.
            entries = [entries[k] for k in sorted(entries, key=lambda x: int(x))]
    else:
        entries = [{}]

    # Deep-merge each entry OVER the defaults so every entry is complete for the child head.
    materialized = []
    defaults = OmegaConf.create(_GEN_TARGET_DEFAULTS)
    for entry in entries:
        entry = dict(entry) if isinstance(entry, dict) else OmegaConf.to_container(entry, resolve=True)
        weight = entry.pop("weight", None)
        merged = OmegaConf.to_container(OmegaConf.merge(defaults, OmegaConf.create(entry)), resolve=True)
        if weight is not None:
            merged["weight"] = weight
        materialized.append(merged)

    g.targets = materialized
    for k in _SINGULAR_GEN_KEYS:
        if k in g:
            del g[k]


def _build_single_gen_head(config, vlm_interface=None):
    """Instantiate the SINGLE mode-dispatched head for ``config.framework.gen_model``.

    The per-target primitive, called once per child by :class:`MultiGenHead`. The config is
    a synthesised child whose ``target``/``decoder``/``loss`` live at the gen_model top level
    (materialised from a ``targets`` entry); it is NOT re-normalised here — ``normalize_gen_config``
    already ran once on the parent in :func:`get_gen_model`.
    """
    mode = str(config.framework.gen_model.get("mode", "regression")).lower()

    # frame_mode='conditioned' is only wired for the expert backbone, where the gen
    # expert's adaLN DiT can inject a per-frame index. Validate the combination centrally
    # with a clear error rather than silently ignoring the knob.
    _validate_frame_mode(config.framework.gen_model, mode)

    # Import side-effects register the concrete heads.
    if mode == "regression":
        from starVLA.model.modules.gen_model import regression
    elif mode == "diffusion":
        from starVLA.model.modules.gen_model import diffusion

    if mode not in GEN_HEAD_REGISTRY:
        raise NotImplementedError(f"gen mode `{mode}` is not implemented; expected 'regression' or 'diffusion'.")
    return GEN_HEAD_REGISTRY[mode](global_config=config, vlm_interface=vlm_interface)


def _validate_frame_mode(gen_cfg, mode):
    """Validate ``framework.gen_model.frame_mode`` against the mode/backbone/isolation.

    'parallel' (default) is always valid: all T future frames' tokens are predicted at once.
    'conditioned' (one shared per-frame query group + adaLN frame-index injection; each
    sample predicts one randomly chosen future frame) requires the expert backbone AND
    isolation (isolation != false) — under coupling the action expert would cross-attend a
    randomly sampled single frame each step.
    """
    frame_mode = str(gen_cfg.get("frame_mode", "parallel")).lower()
    if frame_mode == "parallel":
        return
    if frame_mode != "conditioned":
        raise ValueError(f"unknown framework.gen_model.frame_mode={frame_mode!r}; expected 'parallel' or 'conditioned'.")
    if mode not in ("regression", "diffusion"):
        raise NotImplementedError(
            f"frame_mode='conditioned' (per-frame video gen) supports mode in {{regression, diffusion}}; got {mode!r}."
        )
    backbone = "expert" if mode == "diffusion" else str(gen_cfg.get("backbone", "vlm")).lower()
    if backbone != "expert":
        raise NotImplementedError(
            "frame_mode='conditioned' requires backbone='expert' (the adaLN frame-index injection "
            f"lives in the gen expert); got backbone={backbone!r}."
        )
    from starVLA.model.framework.share_tools import resolve_action_isolation

    if not resolve_action_isolation(gen_cfg, backbone=backbone):
        raise ValueError(
            "frame_mode='conditioned' requires isolation (isolation != false): under coupling the "
            "action expert would cross-attend a randomly-sampled single frame each step. "
            "Set framework.gen_model.isolation to 'auto' or 'true'."
        )
    # The target must decompose per-frame: conditioned generates ONE frame per step, so a
    # temporally-aggregating target (V-JEPA tubelets, 3D video VAE) whose tokens mix multiple
    # frames is incoherent AND would run at a silently-wrong token count. Reject at build.
    from starVLA.model.modules.gen_model.targets import TARGET_REGISTRY

    ttype = str(gen_cfg.target.get("type", "auto")).lower()
    tcls = TARGET_REGISTRY.get(ttype)
    if tcls is not None and not getattr(tcls, "per_frame_decomposable", True):
        raise NotImplementedError(
            f"frame_mode='conditioned' generates ONE frame per step and is incompatible with the "
            f"temporally-aggregating target.type={ttype!r} (its tokens mix multiple frames). "
            "Use a per-frame target (auto/siglip/dino/lingbot/pixel)."
        )


def get_gen_model(config=None, vlm_interface=None):
    """Build a generation head from ``config.framework.gen_model``.

    `normalize_gen_config` materialises a ``targets`` LIST (single target = a 1-element
    list), and EVERY config builds through
    :class:`~starVLA.model.modules.gen_model.multi.MultiGenHead` (one child head + one query
    group / expert per target, aggregated by per-target weights). The per-target primitive is
    :func:`_build_single_gen_head` (one child config synthesised per entry).

    Args:
        config: global config exposing ``config.framework.gen_model``.
        vlm_interface: the VLM wrapper (forwarded to heads whose target/backbone
            reuse the backbone's own vision tower, e.g. target.type='auto').
    """
    normalize_gen_config(config)
    g = config.framework.gen_model
    targets = g.get("targets", None)
    # frame_mode='conditioned' needs ONE shared frame sample, so it is single-target only.
    if targets and len(targets) > 1 and str(g.get("frame_mode", "parallel")).lower() == "conditioned":
        raise NotImplementedError(
            "frame_mode='conditioned' currently supports a single generation target only. "
            "Multi-target conditioning needs one shared frame sample across all child heads."
        )
    from starVLA.model.modules.gen_model.multi import MultiGenHead

    return MultiGenHead(global_config=config, vlm_interface=vlm_interface)
