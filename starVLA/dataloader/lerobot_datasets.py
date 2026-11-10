# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Fangjing Wang/ SUST University] in [2025].
# Modification: [return raw data and suport multi-dataset mixture].
# Modified by [Jinhui YE/ HKUST University] in [2025].
# Modification: [suport topdowm processing, suport param from config].

import logging
from pathlib import Path

from starVLA.dataloader.gr00t_lerobot.data_config import apply_action_horizon, apply_video_keys
from starVLA.dataloader.gr00t_lerobot.datasets import (
    TRANSITION_PERMUTATION,
    LeRobotMixtureDataset,
    LeRobotSingleDataset,
    normalize_sampling_strategy,
)
from starVLA.dataloader.gr00t_lerobot.registry import (
    DATASET_NAMED_MIXTURES,
    ROBOT_TYPE_CONFIG_MAP,
    EmbodimentTag,
)
from starVLA.dataloader.view_tiling import configure_robot_view_contract

logger = logging.getLogger(__name__)


def collate_fn(batch):
    return batch


def make_LeRobotSingleDataset(
    data_root_dir: Path | str,
    data_name: str,
    robot_type: str,
    delete_pause_frame: bool = False,
    data_cfg: dict | None = None,
) -> LeRobotSingleDataset:
    """
    Make a LeRobotSingleDataset object.

    :param data_root_dir: The root directory of the dataset.
    :param data_name: The name of the dataset.
    :param robot_type: The robot type config to use.
    :param delete_pause_frame: Whether to drop pause frames when indexing steps.
    :param data_cfg: The stream config block (``cfg.datasets.<block>``).
    :return: A LeRobotSingleDataset object.
    """

    data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
    # action_horizon is the single source of truth for the chunk length: when the
    # training entry propagated it into data_cfg, derive action_indices from it.
    data_config = apply_action_horizon(data_config, data_cfg)
    # Optional per-stream obs view selection (same knob as the gen path) without a subclass.
    data_config = apply_video_keys(data_config, data_cfg)
    modality_config = data_config.modality_config()
    transforms = data_config.transform()
    dataset_path = data_root_dir / data_name
    embodiment_tag = getattr(data_config, "embodiment_tag", None)
    if embodiment_tag is None:
        print(
            f"Warning: DataConfig for robot_type={robot_type!r} has no embodiment_tag, using {EmbodimentTag.NEW_EMBODIMENT} as default"
        )
        embodiment_tag = EmbodimentTag.NEW_EMBODIMENT

    video_backend = data_cfg.get("video_backend", "decord") if data_cfg else "torchvision_av"

    # Opt-in factory hook: a DataConfig may define ``make_dataset(dataset_name=..., **ds_kwargs)``
    # to swap in a custom dataset class (e.g. with per-task filtering / chunk stride).
    # When absent, fall through to the default LeRobotSingleDataset construction below.
    if hasattr(data_config, "make_dataset"):
        dataset = data_config.make_dataset(
            dataset_path=dataset_path,
            modality_configs=modality_config,
            transforms=transforms,
            embodiment_tag=embodiment_tag,
            video_backend=video_backend,
            delete_pause_frame=delete_pause_frame,
            data_cfg=data_cfg,
            dataset_name=data_name,
        )
    else:
        dataset = LeRobotSingleDataset(
            dataset_path=dataset_path,
            modality_configs=modality_config,
            transforms=transforms,
            embodiment_tag=embodiment_tag,
            video_backend=video_backend,  # decord is more efficiency | torchvision_av for video.av1
            delete_pause_frame=delete_pause_frame,
            data_cfg=data_cfg,
        )

    return configure_robot_view_contract(
        dataset,
        data_config,
        data_cfg,
        context=f"{data_name}/{robot_type}",
    )


def get_vla_dataset(
    data_cfg: dict,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    seed: int = 42,
    **kwargs: dict,
) -> LeRobotMixtureDataset:
    """
    Get a LeRobotMixtureDataset object.
    """
    data_root_dir = data_cfg.data_root_dir
    data_mix = data_cfg.data_mix
    delete_pause_frame = data_cfg.get("delete_pause_frame", False)
    if data_mix not in DATASET_NAMED_MIXTURES:
        raise KeyError(
            f"Unknown data_mix={data_mix!r}; registered mixtures: {sorted(DATASET_NAMED_MIXTURES)}. "
            "Mixtures are declared in examples/<Benchmark>/train_files/data_registry/data_config.py."
        )
    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    logger.info(f"[dataloader] Using mixture '{data_mix}': {[(d, w, r) for d, w, r in mixture_spec]}")
    sampling_strategy = normalize_sampling_strategy(data_cfg.get("sampling_strategy", None))
    if sampling_strategy == TRANSITION_PERMUTATION:
        seed = int(data_cfg.get("seed", seed))
    included_datasets, filtered_mixture_spec = set(), []
    for d_name, d_weight, robot_type in mixture_spec:
        dataset_key = (d_name, robot_type)
        if dataset_key in included_datasets:
            if sampling_strategy == TRANSITION_PERMUTATION:
                raise ValueError(
                    "transition_permutation refuses a duplicate mixture leaf: "
                    f"data_mix={data_mix!r}, dataset={dataset_key!r}."
                )
            print(f"Skipping Duplicate Dataset: `{(d_name, d_weight, robot_type)}`")
            continue

        included_datasets.add(dataset_key)
        filtered_mixture_spec.append((d_name, d_weight, robot_type))

    dataset_mixture = []
    for d_name, d_weight, robot_type in filtered_mixture_spec:
        dataset_mixture.append(
            (
                make_LeRobotSingleDataset(
                    Path(data_root_dir), d_name, robot_type, delete_pause_frame=delete_pause_frame, data_cfg=data_cfg
                ),
                d_weight,
            )
        )

    return LeRobotMixtureDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=seed,
        data_cfg=data_cfg,
        **kwargs,
    )


if __name__ == "__main__":
    # Smoke test: build the action-only LeRobot loader from a training config and
    # iterate a few batches. Run from the repository root, e.g.
    #   python starVLA/dataloader/lerobot_datasets.py --config_yaml configs/finetune/libero.yaml
    import argparse

    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from starVLA.dataloader import align_dataset_configs
    from starVLA.model.framework.share_tools import apply_config_compat, load_config_with_bases

    parser = argparse.ArgumentParser(description="Smoke-test the LeRobot action dataloader.")
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="configs/finetune/libero.yaml",
        help="Training config whose datasets.vla_data block is loaded.",
    )
    parser.add_argument("--data_mix", type=str, default=None, help="Override datasets.vla_data.data_mix.")
    parser.add_argument("--data_root_dir", type=str, default=None, help="Override datasets.vla_data.data_root_dir.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="playground/Checkpoints/debug_lerobot_datasets",
        help="Where dataset_statistics.json is written.",
    )
    parser.add_argument("--num_batches", type=int, default=4, help="Number of batches to iterate.")
    args = parser.parse_args()

    if not Path(args.config_yaml).is_file():
        raise SystemExit(f"Config file not found: {args.config_yaml!r}. Pass --config_yaml <path to a training yaml>.")
    cfg = load_config_with_bases(args.config_yaml)
    if args.data_mix is not None:
        cfg.datasets.vla_data.data_mix = args.data_mix
    if args.data_root_dir is not None:
        cfg.datasets.vla_data.data_root_dir = args.data_root_dir
    data_root_dir = cfg.datasets.vla_data.get("data_root_dir", None)
    if not data_root_dir:
        raise SystemExit(
            "datasets.vla_data.data_root_dir is not set. Set it in the config or pass --data_root_dir "
            "(e.g. playground/Datasets/LEROBOT_LIBERO_DATA)."
        )
    if not Path(data_root_dir).is_dir():
        raise SystemExit(
            f"datasets.vla_data.data_root_dir={str(data_root_dir)!r} does not exist. Download the dataset "
            "first (for LIBERO: bash examples/LIBERO/data_preparation.sh) or pass --data_root_dir."
        )

    # Same normalization as the training entry points (datasets.shared -> stream blocks,
    # action_horizon -> action index grid).
    cfg = align_dataset_configs(apply_config_compat(cfg))
    vla_dataset_cfg = cfg.datasets.vla_data
    dataset = get_vla_dataset(
        data_cfg=vla_dataset_cfg,
        balance_dataset_weights=vla_dataset_cfg.get("balance_dataset_weights", False),
        balance_trajectory_weights=vla_dataset_cfg.get("balance_trajectory_weights", False),
    )

    train_dataloader = DataLoader(dataset, batch_size=2, num_workers=1, collate_fn=collate_fn)
    dataset.save_dataset_statistics(Path(args.output_dir) / "dataset_statistics.json")

    for i, batch in enumerate(tqdm(train_dataloader, desc="Processing batches", total=args.num_batches)):
        if i >= args.num_batches:
            break
        sample = batch[0]
        print({k: (v.shape if hasattr(v, "shape") else type(v).__name__) for k, v in sample.items()})
    print("[OK] lerobot_datasets smoke test passed")
