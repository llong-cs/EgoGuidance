"""Experiment tracking for the training entries.

Metrics go to Weights & Biases when it is installed, tracking is enabled and credentials are
available; otherwise they are appended to ``<output_dir>/metrics.jsonl``. Every logged step is
also printed to stdout by the trainers, so a failing tracker never stops training.

Controls:
    trainer.disable_tracking: true        -> never use W&B
    wandb_project / wandb_entity          -> W&B project / entity (top-level config keys)
    wandb_run_name                        -> W&B run name (default: run_id)
    WANDB_MODE=disabled / WANDB_DISABLED  -> never use W&B
    WANDB_MODE=offline                    -> W&B offline run, no API key needed
"""

import importlib.util
import json
import netrc
import os
from pathlib import Path
from urllib.parse import urlparse

from accelerate.logging import get_logger

logger = get_logger(__name__)

METRICS_JSONL = "metrics.jsonl"


def _wandb_has_credentials() -> bool:
    """Return True when a W&B API key is configured (env var or netrc), without prompting."""
    if os.environ.get("WANDB_API_KEY"):
        return True
    host = urlparse(os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai")).hostname or "api.wandb.ai"
    try:
        return netrc.netrc(os.environ.get("NETRC") or None).authenticators(host) is not None
    except (OSError, netrc.NetrcParseError):
        return False


def _wandb_unavailable_reason(cfg) -> str | None:
    """Return why W&B cannot be used for this run, or None when it can."""
    if bool(cfg.trainer.get("disable_tracking", False)):
        return "disabled by trainer.disable_tracking"
    if os.environ.get("WANDB_MODE", "").lower() == "disabled" or os.environ.get("WANDB_DISABLED", "").lower() in {
        "1",
        "true",
        "yes",
    }:
        return "disabled by WANDB_MODE / WANDB_DISABLED"
    if importlib.util.find_spec("wandb") is None:
        return "wandb is not installed"
    if os.environ.get("WANDB_MODE", "").lower() not in {"offline", "dryrun"} and not _wandb_has_credentials():
        return "no W&B API key found (run `wandb login`, set WANDB_API_KEY, or set WANDB_MODE=offline)"
    return None


def _to_builtin(value):
    """JSON fallback for numpy scalars / tensors with a single element."""
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    return str(value)


class MetricsTracker:
    """Rank-0 metrics sink. Construct it on every rank; only the main process records anything."""

    def __init__(self, cfg, output_dir, *, group: str, is_main_process: bool):
        self.backend = None
        self._wandb = None
        self._jsonl_path = None
        self._output_dir = Path(output_dir)
        if not is_main_process:
            return

        output_dir = self._output_dir
        reason = _wandb_unavailable_reason(cfg)
        if reason is None:
            try:
                import wandb

                wandb_dir = output_dir / "wandb"
                wandb_dir.mkdir(parents=True, exist_ok=True)
                wandb.init(
                    name=str(cfg.get("wandb_run_name", None) or cfg.run_id),
                    dir=str(wandb_dir),
                    project=cfg.get("wandb_project", None),
                    entity=cfg.get("wandb_entity", None),
                    group=group,
                )
                self._wandb = wandb
                self.backend = "wandb"
                logger.info(f"[tracking] logging metrics to W&B (local files under {wandb_dir})")
                return
            except Exception as exc:
                reason = f"wandb.init failed: {type(exc).__name__}: {exc}"

        self._use_jsonl(reason)

    def _use_jsonl(self, reason: str) -> None:
        self._wandb = None
        self._jsonl_path = self._output_dir / METRICS_JSONL
        self.backend = "jsonl"
        logger.info(f"[tracking] W&B not used ({reason}); appending metrics to {self._jsonl_path}")

    def log(self, metrics: dict, step: int) -> None:
        if self._wandb is not None:
            try:
                self._wandb.log(metrics, step=step)
                return
            except Exception as exc:
                self._use_jsonl(f"wandb.log failed: {type(exc).__name__}: {exc}")
        if self._jsonl_path is None:
            return
        try:
            with open(self._jsonl_path, "a") as handle:
                handle.write(json.dumps({"step": int(step), **metrics}, sort_keys=True, default=_to_builtin) + "\n")
        except OSError as exc:
            logger.warning(f"[tracking] cannot write {self._jsonl_path} ({exc}); metrics go to stdout only")
            self._jsonl_path = None

    def finish(self) -> None:
        if self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception:
                pass
            self._wandb = None
