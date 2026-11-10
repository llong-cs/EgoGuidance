#!/usr/bin/env python3
"""Print the launch metadata of a training config for scripts/train.sh.

    python scripts/resolve_runtime_config.py <config.yaml> [--dotted.key value ...]

Loads the config (with its ``_base_`` chain), applies the same ``--dotted.key value``
overrides the training entry will receive, and prints one tab-separated line:

    <train_entry> <TAB> <run_id> <TAB> <run_root_dir>

``train_entry`` defaults to ``starVLA/training/train_starvla.py`` and ``run_root_dir`` to
``playground/Checkpoints`` (the training entries use the same default); ``run_id`` is
required.
"""

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from starVLA.model.framework.share_tools import load_config_with_bases  # noqa: E402

DEFAULT_TRAIN_ENTRY = "starVLA/training/train_starvla.py"
# Keep in sync with starVLA.training.trainer_utils.trainer_tools.DEFAULT_RUN_ROOT_DIR
# (not imported here to avoid loading torch / accelerate in the launcher).
DEFAULT_RUN_ROOT_DIR = "playground/Checkpoints"


def _single_line(value, name: str) -> str:
    value = str(value)
    if not value or any(character in value for character in ("\t", "\n", "\r")):
        raise ValueError(f"{name} must be a non-empty single-line value, got {value!r}")
    return value


def _normalize_dotlist_args(args):
    """Same ``--key value`` / ``--key=value`` / ``--flag`` parsing as the training entries."""
    normalized = []
    index = 0
    while index < len(args):
        argument = args[index]
        if argument.startswith("--"):
            key = argument.lstrip("-")
            if "=" in key:
                normalized.append(key)
            elif index + 1 < len(args) and not args[index + 1].startswith("--"):
                normalized.append(f"{key}={args[index + 1]}")
                index += 1
            else:
                normalized.append(f"{key}=true")
        index += 1
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config_yaml")
    args, overrides = parser.parse_known_args()

    config_path = Path(args.config_yaml)
    if not config_path.is_file():
        raise FileNotFoundError(f"Training config does not exist: {config_path}")

    config = load_config_with_bases(config_path)
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(_normalize_dotlist_args(overrides)))

    run_id = config.get("run_id", None)
    if run_id in (None, ""):
        raise ValueError(f"{config_path} sets no top-level run_id; add one or pass --run_id <name>.")
    train_entry = config.get("train_entry", None) or DEFAULT_TRAIN_ENTRY
    run_root_dir = config.get("run_root_dir", None) or DEFAULT_RUN_ROOT_DIR

    print(
        "\t".join(
            (
                _single_line(train_entry, "train_entry"),
                _single_line(run_id, "run_id"),
                _single_line(run_root_dir, "run_root_dir"),
            )
        )
    )


if __name__ == "__main__":
    main()
