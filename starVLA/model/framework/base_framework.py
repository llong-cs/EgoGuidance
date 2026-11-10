"""
Base framework abstraction providing:
- Pretrained loading (config + normalization stats + weights)
- Action space utilities (dimension, stats, (un)normalization)
- Trainable module discovery helper
Note: No device placement or optimizer concerns handled here (delegated to trainer).
"""

import importlib
import pkgutil
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import PretrainedConfig, PreTrainedModel

from starVLA.model.framework.share_tools import dict_to_namespace, read_mode_config, resolve_action_isolation
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)
_FRAMEWORKS_IMPORTED = False


def _auto_import_framework_modules() -> None:
    global _FRAMEWORKS_IMPORTED
    if _FRAMEWORKS_IMPORTED:
        return

    _SKIP = {"__init__", "base_framework", "share_tools"}
    framework_dir = Path(__file__).resolve().parent

    # Scan top-level modules and one level of sub-packages (e.g. VLM4A/).
    for _, module_name, is_pkg in pkgutil.iter_modules([str(framework_dir)]):
        if module_name in _SKIP:
            continue
        if is_pkg:
            sub_dir = framework_dir / module_name
            for _, sub_name, _ in pkgutil.iter_modules([str(sub_dir)]):
                if sub_name.startswith("_"):
                    continue
                importlib.import_module(f"starVLA.model.framework.{module_name}.{sub_name}")
        else:
            importlib.import_module(f"starVLA.model.framework.{module_name}")

    _FRAMEWORKS_IMPORTED = True


def build_framework(cfg):  # The single entry point for building different model frameworks
    """
    Build a framework model from config.
    Args:
        cfg: Config object containing `cfg.framework.name`.
    Returns:
        nn.Module: Instantiated framework model.
    """
    if not hasattr(cfg, "framework") or not hasattr(cfg.framework, "name"):
        raise ValueError("Missing `cfg.framework.name`. The framework API now only accepts `framework.name`.")

    _auto_import_framework_modules()

    framework_id = cfg.framework.name
    if framework_id not in FRAMEWORK_REGISTRY._registry:
        available = sorted(FRAMEWORK_REGISTRY._registry.keys())
        raise NotImplementedError(f"Framework `{framework_id}` is not implemented. Available frameworks: {available}")

    model_class = FRAMEWORK_REGISTRY[framework_id]
    return model_class(cfg)


def _select_inference_state_dict(state_dict: Dict[str, "torch.Tensor"], model) -> Dict[str, "torch.Tensor"]:
    """Keep only the weights a unified pure-VLA *inference* model needs.

    A decoupled (isolated) generation model is inference-equivalent to its no-gen
    baseline: the gen branch — saved under the ``gen_model.`` prefix — never runs at
    inference (``predict_action`` skips every gen path when the action expert is
    isolated). So when :meth:`from_pretrained` rebuilds a decoupled checkpoint as a
    pure-VLA model (``gen_model.enable=False``), the checkpoint carries extra keys
    the model does not have. This filter:

      * drops every checkpoint key absent from the (pure-VLA) model — i.e. the whole
        ``gen_model.*`` subtree, so ALL decoupled gen forms load through ONE
        architecture and ONE inference code path, with no generation dependencies;
      * shape-truncates any tensor whose checkpoint copy is LARGER than the model's
        (keeping the leading slice), e.g. a token-embedding table that was resized during
        training; the leading rows are the baseline vocabulary.

    Raises if a required VLA weight is missing or a shape mismatch cannot be
    reconciled by leading-slice truncation.
    """
    model_sd = model.state_dict()
    model_keys = set(model_sd.keys())
    selected: Dict[str, "torch.Tensor"] = {}
    dropped: List[str] = []
    truncated: List[str] = []
    for k, v in state_dict.items():
        if k not in model_keys:
            dropped.append(k)
            continue
        mv = model_sd[k]
        if tuple(v.shape) != tuple(mv.shape):
            if v.dim() == mv.dim() and all(a >= b for a, b in zip(v.shape, mv.shape, strict=True)):
                v = v[tuple(slice(0, b) for b in mv.shape)].contiguous()
                truncated.append(k)
            else:
                raise RuntimeError(
                    f"Inference load: irreconcilable shape for '{k}': "
                    f"checkpoint {tuple(v.shape)} vs model {tuple(mv.shape)}."
                )
        selected[k] = v
    missing = model_keys - set(selected.keys())
    if missing:
        raise RuntimeError(
            f"Inference load: checkpoint is missing {len(missing)} required VLA weights, " f"e.g. {sorted(missing)[:8]}."
        )
    if dropped:
        logger.info(
            f"[from_pretrained] dropped {len(dropped)} generation-branch keys "
            f"(decoupled → unified pure-VLA inference)."
        )
    if truncated:
        logger.info(f"[from_pretrained] truncated resized embeddings to baseline vocab: {truncated}")
    return selected


class baseframework(PreTrainedModel):
    """
    Lightweight base class for higher-level VLA model assemblies.
    Subclasses are expected to:
      - Accept a structured config
      - Register components in __init__
      - Use provided helpers for action normalization handling
    """

    def __init__(self, hf_config: Optional[PretrainedConfig] = None) -> None:
        """
        Initialize base nn.Module. Subclasses add components.
        """

        super().__init__(hf_config if hf_config is not None else PretrainedConfig())

    # ------------------------------------------------------------------
    # Soft-constraint interface: subclasses should override these.
    # Default implementations raise NotImplementedError so that IDE
    # tooling (e.g. pylance, mypy) flags missing overrides, while
    # still allowing PreTrainedModel instantiation (no ABC).
    # ------------------------------------------------------------------

    def forward(self, examples: List[dict], **kwargs) -> dict:
        """Training forward pass.

        Args:
            examples: List[dict], each dict requires at least:
                - image: List[PIL.Image]
                - lang: str
                - action: np.ndarray shaped [T, action_dim]

        Returns:
            dict: Must contain ``"action_loss"`` (torch.Tensor scalar).
                  May contain extra keys for logging (e.g. ``"kl_loss"``).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement forward(examples) -> dict with 'action_loss' key."
        )

    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        """Inference: predict future actions from observations.

        Args:
            examples: Same schema as *forward* (minus ``action`` which is optional).
            **kwargs: Framework-specific inference options (e.g. ``use_ddim``).

        Returns:
            dict: Must contain ``"normalized_actions"`` (np.ndarray [B, T, action_dim]).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement predict_action(examples) -> dict with 'normalized_actions' key."
        )

    # ------------------------------------------------------------------
    # Unified loss interface for Trainer
    # ------------------------------------------------------------------

    def supports_training_tag(self, tag: str) -> bool:
        """Return whether this framework can consume batches for *tag*."""
        if tag == "vla":
            return type(self).forward is not baseframework.forward
        if tag == "vlm":
            return hasattr(self, "qwen_vl_interface") or type(self).forward_vlm is not baseframework.forward_vlm
        return False

    def compute_loss(self, tag: str, batch, loss_scale: Optional[dict] = None) -> Dict[str, torch.Tensor] | None:
        """Unified forward entry-point: route to the right forward by *tag*.

        The trainer calls ``model.compute_loss(tag, batch)`` for every
        ``(tag, batch)`` pair produced by :class:`DataLoaderManager`.
        The model internally dispatches:

        - ``"vla"`` → ``self.forward(batch)``
        - ``"vlm"`` → ``self.forward_vlm(batch)``

        Subclasses can override this to add more tags (e.g. ``"world"``).

        Args:
            tag: dataset type tag (``"vla"``, ``"vlm"``, …)
            batch: the batch produced by the corresponding DataLoader.
            loss_scale: ``{"vla": 1.0, "vlm": 0.1}`` per-tag loss multiplier.
                        Defaults to 1.0 for unspecified tags.

        Returns:
            dict[str, Tensor] | None: keyed losses (e.g. ``{"action_loss": ...}``).
                Returns ``None`` when this framework does not support the
                incoming dataloader tag so the trainer can ``continue``.
        """
        if not self.supports_training_tag(tag):
            return None

        scale = (loss_scale or {}).get(tag, 1.0)

        if tag == "vla":
            out = self.forward(batch)
        elif tag == "vlm":
            out = self.forward_vlm(batch)
        else:
            return None

        # Apply loss scale and filter to Tensor values only
        return {k: v * scale for k, v in out.items() if isinstance(v, torch.Tensor)}

    def forward_vlm(self, batch) -> Dict[str, torch.Tensor]:
        """VLM forward pass (default implementation).

        Delegates to ``self.qwen_vl_interface(**batch)`` which is present on
        every framework subclass that uses a Qwen VL backbone.

        Subclasses may override to add custom VLM logic.

        Args:
            batch: dict produced by the VLM dataloader.

        Returns:
            dict: Must contain ``"vlm_loss"`` (torch.Tensor scalar).
        """
        if not hasattr(self, "qwen_vl_interface"):
            raise NotImplementedError(
                f"{type(self).__name__} has no `qwen_vl_interface`. " "Override forward_vlm() to support VLM training."
            )
        out = self.qwen_vl_interface(**batch)
        return {"vlm_loss": out.loss}

    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: str,
        load_gen_branch: bool = False,
        **kwargs,
    ) -> None:
        """
        Restore a model instance from a saved checkpoint.

        Workflow:
            1. Resolve checkpoint path
            2. Load config + dataset normalization statistics
            3. UNIFY decoupled generation checkpoints for inference: when the run
               enabled a generation loss AND the action expert was isolated from it
               (``resolve_action_isolation`` — the ``expert``/``diffusion`` backbones,
               or ``isolation=true``), the gen branch is inference-irrelevant. We
               rebuild as a pure-VLA model (``gen_model.enable=False``) so that EVERY
               decoupled gen form (regression / diffusion, single / multi target,
               single / co-training) loads through ONE architecture, ONE inference
               code path, and needs NO generation dependencies (SigLIP / DINOv2 /
               V-JEPA / LingBot / Wan-VAE encoders). Coupled runs (``isolation=false``,
               or the ``vlm`` backbone under ``isolation=auto``) keep the gen branch —
               they genuinely use it at inference.
            4. Build model with the (possibly gen-disabled) config
            5. Load state_dict strictly. For a unified pure-VLA build, drop the saved
               ``gen_model.*`` weights first (see :func:`_select_inference_state_dict`).
            6. Attach normalization stats for later un-normalization

        Args:
            pretrained_checkpoint: Path to .pt file inside run/checkpoints directory.
            load_gen_branch: force-keep the generation branch even for a decoupled
                checkpoint (rebuild the full training model and strict-load every
                weight). Escape hatch for inspecting/finetuning the gen branch; the
                default (False) gives the unified pure-VLA inference model.
            **kwargs: Extra constructor overrides passed to subclass.

        Returns:
            baseframework: Instantiated model (left on CPU; caller decides device).

        Raises:
            RuntimeError: If state_dict key mismatch occurs under strict=True.
            FileNotFoundError: If underlying files are missing (surfaced earlier).
        """
        pretrained_checkpoint = Path(pretrained_checkpoint)
        model_config, norm_stats = read_mode_config(pretrained_checkpoint)

        config = dict_to_namespace(model_config)
        model_config = config
        model_config.trainer.pretrained_checkpoint = None

        # Unify decoupled inference: a decoupled gen model == its no-gen baseline at
        # inference, so build pure-VLA and drop the (inference-irrelevant) gen weights.
        gen_cfg = None
        framework_cfg = getattr(model_config, "framework", None)
        if framework_cfg is not None:
            gen_cfg = framework_cfg.get("gen_model", None) if hasattr(framework_cfg, "get") else None
        unify_pure_vla = (
            not load_gen_branch
            and gen_cfg is not None
            and bool(gen_cfg.get("enable", False))
            and resolve_action_isolation(gen_cfg)
        )
        if unify_pure_vla:
            logger.info(
                "[from_pretrained] decoupled generation checkpoint → building unified "
                "pure-VLA inference model (gen branch dropped)."
            )
            gen_cfg.enable = False

        FrameworkModel = build_framework(cfg=model_config)
        # Kept on the model for action un-normalization.
        FrameworkModel.norm_stats = norm_stats
        if pretrained_checkpoint.suffix == ".safetensors":
            from safetensors.torch import load_file

            model_state_dict = load_file(str(pretrained_checkpoint))
        else:
            model_state_dict = torch.load(pretrained_checkpoint, map_location="cpu")
        if unify_pure_vla:
            # Keep only the VLA weights the pure model expects (drops gen_model.*).
            model_state_dict = _select_inference_state_dict(model_state_dict, FrameworkModel)
        model_keys = set(FrameworkModel.state_dict().keys())
        checkpoint_keys = set(model_state_dict.keys())
        try:
            FrameworkModel.load_state_dict(model_state_dict, strict=True)
        except RuntimeError as e:
            # Report the key mismatch before re-raising (all keys must match).
            common_keys = model_keys.intersection(checkpoint_keys)
            missing_keys = model_keys - common_keys
            unexpected_keys = checkpoint_keys - common_keys
            if missing_keys:
                logger.warning(f"Missing keys in state_dict: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys in state_dict: {unexpected_keys}")

            raise e

        return FrameworkModel
