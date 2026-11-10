"""
Shared configuration / utility helpers for framework components:
- framework default-config merging and expert DiT-shape population
- checkpoint config / statistics loading
- generation isolation resolution
- ``_base_`` config composition and config-schema normalisation
"""

import json
import os
import re
from pathlib import Path

from omegaconf import OmegaConf

from starVLA.training.trainer_utils import initialize_overwatch

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


def dict_to_namespace(d):
    """
    Create an OmegaConf config from a plain dictionary.

    Args:
        d: Input dictionary.

    Returns:
        OmegaConf: DictConfig instance.
    """
    return OmegaConf.create(d)


def merge_framework_config(default_config_cls, cfg):
    """
    Merge a framework's default config (dataclass) with the incoming YAML config.

    Rules:
        - default_config_cls provides documented defaults for `cfg.framework`
        - YAML values (cfg.framework) override matching defaults
        - Extra YAML keys not in defaults are preserved (Config-as-API flexibility)
        - Missing YAML keys fall back to defaults (less YAML boilerplate)

    The merge only touches the `cfg.framework` sub-tree; datasets / trainer / etc.
    are left untouched.

    Args:
        default_config_cls: A dataclass **class** (not instance) whose fields() define
                            the default framework config with type hints and comments.
        cfg: The full OmegaConf config (must contain cfg.framework).

    Returns:
        cfg: The same config object with cfg.framework replaced by the merged result.
    """
    import dataclasses

    from omegaconf import DictConfig, OmegaConf

    # 1. Instantiate defaults and convert to OmegaConf
    defaults_instance = default_config_cls()
    defaults_dict = dataclasses.asdict(defaults_instance)
    defaults_omega = OmegaConf.create(defaults_dict)

    # 2. Extract the YAML framework section
    if hasattr(cfg, "framework"):
        # Unwrap AccessTrackedConfig if needed
        yaml_fw = cfg.framework
        if hasattr(yaml_fw, "_cfg"):
            yaml_fw = yaml_fw._cfg
        if not isinstance(yaml_fw, DictConfig):
            yaml_fw = OmegaConf.create(yaml_fw if isinstance(yaml_fw, dict) else {})
    else:
        yaml_fw = OmegaConf.create({})

    # 3. Merge: defaults first, YAML overrides (YAML wins on conflicts)
    merged_fw = OmegaConf.merge(defaults_omega, yaml_fw)

    # 4. Write back into the original cfg
    #    Handle both OmegaConf and AccessTrackedConfig transparently
    if hasattr(cfg, "_cfg") and isinstance(cfg._cfg, DictConfig):
        # AccessTrackedConfig caches child wrappers in _children dict.
        # After replacing the underlying DictConfig node, the old child wrapper
        # still points to the pre-merge node (stale data).  We must invalidate
        # the cache so the next attribute access creates a fresh wrapper around
        # the merged node.
        #
        # However, the old child's _local_accessed set records which keys were
        # already read (e.g. "name" from build_framework).  Deleting the child
        # would lose that tracking info, causing save_accessed_config to omit
        # those keys from config.yaml.  So we preserve and restore it.
        cfg._cfg.framework = merged_fw
        if hasattr(cfg, "_children") and "framework" in cfg._children:
            old_accessed = cfg._children["framework"]._local_accessed.copy()
            del cfg._children["framework"]  # invalidate stale cache
            new_child = cfg.framework  # re-create child around merged_fw
            new_child._local_accessed.update(old_accessed)  # restore tracking
    elif isinstance(cfg, DictConfig):
        cfg.framework = merged_fw
    else:
        # Fallback — try direct attribute setting
        try:
            cfg.framework = merged_fw
        except Exception:
            overwatch.warning("Could not write merged framework config back to cfg.")

    return cfg


def populate_layerwise_dit_cfg(cfg, *, dit_hidden_dim: int, num_dit_layers: int):
    """
    Populate ``framework.action_model.diffusion_model_cfg`` with the DiT shape
    fields required by ``LayerwiseFlowmatchingActionHead``.

    Why this helper exists:
        The action head is intentionally agnostic of the VLM backbone — it only
        consumes ``diffusion_model_cfg``.  The framework decides the DiT shape
        (depth + hidden) and writes it here BEFORE calling ``get_action_model``.

    Fields written (override any stale YAML values):
        - num_layers           = num_dit_layers
        - input_embedding_dim  = dit_hidden_dim
        - cross_attention_dim  = dit_hidden_dim   (encoder is pre-projected)
        - num_attention_heads  = dit_hidden_dim // attention_head_dim
                                 (uses existing attention_head_dim if set, else 64)

    Args:
        cfg: Full OmegaConf config.
        dit_hidden_dim: DiT internal hidden dim.
        num_dit_layers: Number of DiT cross-attention layers.

    Returns:
        The (mutated) diffusion_model_cfg node.
    """
    dit_cfg = cfg.framework.action_model.diffusion_model_cfg
    head_dim = dit_cfg.get("attention_head_dim", None) or 64
    dit_cfg.attention_head_dim = head_dim
    dit_cfg.num_layers = int(num_dit_layers)
    dit_cfg.input_embedding_dim = int(dit_hidden_dim)
    dit_cfg.cross_attention_dim = int(dit_hidden_dim)
    dit_cfg.num_attention_heads = int(dit_hidden_dim) // int(head_dim)
    return dit_cfg


def populate_gen_expert_dit_cfg(cfg, *, dit_hidden_dim: int, num_dit_layers: int):
    """
    Populate ``framework.gen_model.expert_dit_cfg`` with the DiT shape fields
    required by ``GenerationExpert`` (the MoT-style generation backbone).

    This is the generation-side twin of ``populate_layerwise_dit_cfg``: the
    generation expert is a layer-wise cross-attention DiT with the SAME
    architecture and shape conventions as the action expert, so it cross-attends
    to the same per-layer VLM hidden states.

    Fields written (override any stale YAML values):
        - num_layers           = num_dit_layers     (== number of VLM layers fed)
        - input_embedding_dim  = dit_hidden_dim
        - cross_attention_dim  = dit_hidden_dim      (encoder is pre-projected)
        - num_attention_heads  = dit_hidden_dim // attention_head_dim
                                 (uses existing attention_head_dim if set, else 64)

    Args:
        cfg: Full OmegaConf config (must contain framework.gen_model.expert_dit_cfg).
        dit_hidden_dim: DiT internal hidden dim (== VLM hidden size).
        num_dit_layers: Number of DiT cross-attention layers (== num VLM layers).

    Returns:
        The (mutated) expert_dit_cfg node.
    """
    dit_cfg = cfg.framework.gen_model.expert_dit_cfg
    head_dim = dit_cfg.get("attention_head_dim", None) or 64
    dit_cfg.attention_head_dim = head_dim
    dit_cfg.num_layers = int(num_dit_layers)
    dit_cfg.input_embedding_dim = int(dit_hidden_dim)
    dit_cfg.cross_attention_dim = int(dit_hidden_dim)
    dit_cfg.num_attention_heads = int(dit_hidden_dim) // int(head_dim)
    return dit_cfg


def _resolve_checkpoint_run_dir(checkpoint_pt: Path, config_name: str) -> Path:
    """Resolve the run directory for weights-only and full-state portable checkpoints.

    Supported layouts are intentionally explicit:

    - ``<run>/checkpoints/steps_N_pytorch_model.pt`` (weights-only)
    - ``<run>/checkpoints/steps_N/portable_pytorch_model.pt`` (full-state)
    - ``<run>/final_model/<file>`` (published final model)

    Merely finding a similarly named config in an arbitrary ancestor is not
    enough: accepting that would make a misplaced checkpoint silently inherit
    another run's configuration.
    """

    checkpoint_parent = checkpoint_pt.parent
    if checkpoint_parent.name == "checkpoints":
        run_dir = checkpoint_parent.parent
    elif checkpoint_parent.name == "final_model":
        run_dir = checkpoint_parent.parent
    elif re.fullmatch(r"steps_[1-9][0-9]*", checkpoint_parent.name) and checkpoint_parent.parent.name == "checkpoints":
        run_dir = checkpoint_parent.parent.parent
        marker_path = checkpoint_parent / "checkpoint.complete"
        if not marker_path.is_file():
            raise FileNotFoundError(
                f"Full-state checkpoint {checkpoint_parent} has no "
                "checkpoint.complete marker; refusing to consume a possibly "
                "interrupted portable model."
            )
        with open(marker_path) as marker_file:
            marker = json.load(marker_file)
        if marker.get("portable_model") != checkpoint_pt.name:
            raise ValueError(
                f"Checkpoint marker {marker_path} names portable model "
                f"{marker.get('portable_model')!r}, not {checkpoint_pt.name!r}."
            )
    else:
        raise ValueError(
            "Unsupported checkpoint layout. Expected "
            "<run>/checkpoints/<file> or "
            "<run>/checkpoints/steps_N/<portable-file> or "
            "<run>/final_model/<file>, got "
            f"{checkpoint_pt}."
        )

    missing = [
        str(run_dir / required_name)
        for required_name in (config_name, "dataset_statistics.json")
        if not (run_dir / required_name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Checkpoint {checkpoint_pt} resolved to run {run_dir}, but required " f"run metadata is missing: {missing}."
        )
    return run_dir


def read_mode_config(pretrained_checkpoint):
    """
    Load the training config (``config.yaml``) and dataset normalization statistics
    that belong to a saved checkpoint (see :func:`_resolve_checkpoint_run_dir` for the
    supported layouts).

    Args:
        pretrained_checkpoint: Path to a ``.pt`` / ``.safetensors`` checkpoint file.

    Returns:
        tuple:
            global_cfg (dict): the run's config, normalised by :func:`apply_config_compat`.
            norm_stats (dict): dataset statistics for action (un)normalization.

    Raises:
        FileNotFoundError: if the checkpoint or its run metadata is missing.
    """
    if os.path.isfile(pretrained_checkpoint):
        overwatch.info(f"Loading from local checkpoint path `{(checkpoint_pt := Path(pretrained_checkpoint))}`")

        if checkpoint_pt.suffix not in {".pt", ".safetensors"}:
            raise ValueError(f"Unsupported checkpoint suffix {checkpoint_pt.suffix!r}: " "expected .pt or .safetensors.")
        run_dir = _resolve_checkpoint_run_dir(checkpoint_pt, "config.yaml")

        config_yaml, dataset_statistics_json = run_dir / "config.yaml", run_dir / "dataset_statistics.json"
        try:
            ocfg = OmegaConf.load(str(config_yaml))
            # Normalise to the current config schema (idempotent).
            apply_config_compat(ocfg)
            global_cfg = OmegaConf.to_container(ocfg, resolve=True)
        except Exception as e:
            overwatch.error(f"Failed to load YAML config `{config_yaml}`: {e}")
            raise

        # Load Dataset Statistics for Action Denormalization
        with open(dataset_statistics_json, "r") as f:
            norm_stats = json.load(f)
    else:
        overwatch.error(f"Pretrained checkpoint `{pretrained_checkpoint}` does not exist.")
        raise FileNotFoundError(f"Pretrained checkpoint `{pretrained_checkpoint}` does not exist.")
    return global_cfg, norm_stats


def resolve_action_isolation(gen_cfg, backbone=None) -> bool:
    """Whether the action expert is ISOLATED from the generation branch (decoupled).

    Single source of truth for the isolation switch, shared by the framework
    (``QwenPI.__init__``, where it decides train-time attention wiring) and
    :meth:`baseframework.from_pretrained` (where it decides whether the gen branch
    is inference-irrelevant and can be dropped for a unified pure-VLA inference model):

        isolation == "auto"      -> decoupled iff the gen backbone is NOT the VLM
                                    (an ``expert`` / ``diffusion`` backbone defaults
                                    to isolated; a ``vlm`` backbone defaults to coupled).
        isolation == true/false  -> explicit.

    Returns ``True`` when the action expert CANNOT see gen content — i.e. the gen
    loss influences the action only through shared VLM/action parameter updates, so
    at inference the model is equivalent to its no-gen baseline.

    Args:
        gen_cfg: the ``framework.gen_model`` config node (or ``None`` when gen is off).
        backbone: optional resolved backbone override. The framework passes the BUILT
            head's ``backbone`` (some modes, e.g. diffusion, force ``expert`` regardless
            of the YAML value); callers without a built head (e.g. ``from_pretrained``,
            which only has the saved config) omit it, and the config's own ``backbone``
            key is used — with the same ``mode``-driven force applied here so both call
            sites agree even when the YAML ``backbone`` was left at its default.
    """
    if gen_cfg is None:
        return False
    raw = gen_cfg.get("isolation", "auto")
    if isinstance(raw, str) and raw.lower() == "auto":
        if backbone is not None:
            bk = str(backbone).lower()
        else:
            bk = str(gen_cfg.get("backbone", "vlm")).lower()
            # diffusion mode ALWAYS runs an expert backbone (enforced in
            # gen_model/diffusion.py, but only on the built instance — not written back
            # to the config), so a saved config whose YAML left backbone at its default
            # must still resolve to the expert (isolated) branch here.
            if str(gen_cfg.get("mode", "")).lower() == "diffusion":
                bk = "expert"
        return bk != "vlm"
    return bool(raw)


# =============================================================================
# Config composition and schema normalisation (current version_id "0.22").
#
# `apply_config_compat` keeps user-facing YAMLs short: it fills derived keys
# (e.g. `future_action_window_size` from `action_horizon`) so that every consumer
# sees one complete schema. It is *idempotent* — calling it multiple times yields
# the same result.
# =============================================================================

CONFIG_VERSION = "0.22"


def load_config_with_bases(config_yaml):
    """Load a training YAML, merging any ``_base_`` shared configs underneath it.

    A config may declare a ``_base_`` list of shared config paths (relative to the
    config file's own directory), e.g.::

        _base_: [robot_only.yaml]
        run_id: my_run

    Each base is loaded (bases may themselves declare ``_base_`` — resolution is
    recursive), merged in listed order, and the config's OWN keys are merged on top
    (so the config always wins over its bases; a later base wins over an earlier one).
    This lets the configs under ``configs/`` share one model / trainer recipe instead
    of duplicating it.

    A config with no ``_base_`` key behaves exactly like ``OmegaConf.load``. The
    ``_base_`` key itself is consumed here and never reaches the training config.
    """
    config_yaml = Path(config_yaml)
    cfg = OmegaConf.load(config_yaml)
    bases = cfg.pop("_base_", None) if "_base_" in cfg else None
    if not bases:
        return cfg
    if isinstance(bases, str):
        bases = [bases]
    merged = OmegaConf.create({})
    for rel in bases:
        base_path = config_yaml.parent / str(rel)
        merged = OmegaConf.merge(merged, load_config_with_bases(base_path))
    return OmegaConf.merge(merged, cfg)


def apply_config_compat(cfg, *, strict: bool = False):
    """
    Normalise a starVLA training config into the current `version_id == "0.22"` schema.

    Performed transformations (each applied only when needed):

      1.  `framework.action_model.action_horizon` ↔ `future_action_window_size`
          - `action_horizon` is canonical (preferred user-facing name).
          - `future_action_window_size = action_horizon - 1` is auto-filled for
            consumers that read that key.
          - If both are present and inconsistent, a warning is emitted and
            `action_horizon` wins.

      2.  `framework.action_model.diffusion_model_cfg.output_dim`
          - Auto-filled from `framework.action_model.hidden_size` when missing.

      3.  `framework.action_model.diffusion_model_cfg.cross_attention_dim`
          - Auto-filled from `framework.qwenvl.vl_hidden_dim` when missing
            (QwenPI overwrites the DiT shape at build time anyway).

      4.  `framework.action_model.action_hidden_dim`
          - Auto-filled from `hidden_size` when missing.

      5.  `framework.action_model.past_action_window_size`
          - Auto-filled to `0` when missing (QwenPI always runs with past=0).

      6.  `cfg.version_id` is stamped to `"0.22"`.

    Args:
        cfg: An OmegaConf DictConfig.
        strict: If True, raise on inconsistent values instead of warning.

    Returns:
        The same `cfg` object (mutated in place) for chaining convenience.
    """
    from omegaconf import OmegaConf

    if cfg is None:
        return cfg

    src_version = OmegaConf.select(cfg, "version_id", default=None)

    # ---- 1. action_horizon ↔ future_action_window_size ----
    am_path = "framework.action_model"
    am = OmegaConf.select(cfg, am_path, default=None)
    if am is not None:
        ah = OmegaConf.select(am, "action_horizon", default=None)
        fw = OmegaConf.select(am, "future_action_window_size", default=None)

        if ah is None and fw is not None:
            ah = int(fw) + 1
            OmegaConf.update(cfg, f"{am_path}.action_horizon", ah, force_add=True)
        elif ah is not None and fw is None:
            fw = int(ah) - 1
            OmegaConf.update(cfg, f"{am_path}.future_action_window_size", fw, force_add=True)
        elif ah is not None and fw is not None and int(ah) != int(fw) + 1:
            msg = (
                f"[apply_config_compat] inconsistent action_horizon={ah} vs "
                f"future_action_window_size={fw}; expected action_horizon == future + 1. "
                "Using action_horizon as canonical."
            )
            if strict:
                raise ValueError(msg)
            overwatch.warning(msg)
            OmegaConf.update(cfg, f"{am_path}.future_action_window_size", int(ah) - 1, force_add=True)

        # ---- 2 & 3. diffusion_model_cfg auto-fill ----
        dm_path = f"{am_path}.diffusion_model_cfg"
        dm = OmegaConf.select(cfg, dm_path, default=None)
        if dm is not None:
            hidden_size = OmegaConf.select(am, "hidden_size", default=None)
            if OmegaConf.select(dm, "output_dim", default=None) is None and hidden_size is not None:
                OmegaConf.update(cfg, f"{dm_path}.output_dim", int(hidden_size), force_add=True)

            if OmegaConf.select(dm, "cross_attention_dim", default=None) is None:
                vl_hidden = OmegaConf.select(cfg, "framework.qwenvl.vl_hidden_dim", default=None)
                if vl_hidden is not None:
                    OmegaConf.update(cfg, f"{dm_path}.cross_attention_dim", int(vl_hidden), force_add=True)
                # else: leave None — framework __init__ may auto-bind it

        # ---- 4. action_hidden_dim fallback ----
        if OmegaConf.select(am, "action_hidden_dim", default=None) is None:
            hidden_size = OmegaConf.select(am, "hidden_size", default=None)
            if hidden_size is not None:
                OmegaConf.update(cfg, f"{am_path}.action_hidden_dim", int(hidden_size), force_add=True)

        # ---- 5. past_action_window_size default ----
        if OmegaConf.select(am, "past_action_window_size", default=None) is None:
            OmegaConf.update(cfg, f"{am_path}.past_action_window_size", 0, force_add=True)

    # ---- 6. stamp version ----
    if src_version != CONFIG_VERSION:
        try:
            OmegaConf.update(cfg, "version_id", CONFIG_VERSION, force_add=True)
        except Exception:
            try:
                cfg.version_id = CONFIG_VERSION
            except Exception:
                pass
        overwatch.info(f"[apply_config_compat] normalised config from version_id={src_version!r} to {CONFIG_VERSION!r}")

    return cfg
