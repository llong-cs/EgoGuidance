"""DROID (LeRobot v3.0) — data config, embodiment tag, and mixtures.

Auto-discovered by ``starVLA.dataloader.gr00t_lerobot.registry.discover_and_merge``
(which scans ``examples/*/train_files/data_registry/data_config.py``).

Data: DROID in LeRobot **v3.0** format (multi-episode chunk/file parquet, ``tasks.parquet``,
``meta/episodes/``) with separate ``success/`` and ``failure/`` sub-datasets, e.g. the
``nvidia/Cosmos3-DROID`` release on Hugging Face. Point ``datasets.<block>.data_root_dir`` at
the directory that contains ``success/`` and set ``datasets.<block>.lerobot_version: v3.0``.

The raw LeRobot release ships WITHOUT a gr00t ``meta/modality.json``. The mapping therefore
lives in the repository as the sibling ``droid_modality.json`` and is selected with
``datasets.<block>.modality_json_path`` (the loader falls back to it when the dataset
directory has no modality.json).

Action space (same 7-dim layout as LIBERO's Δ-EEF + gripper):
    action.cartesian_velocity[0:6] + action.gripper_position[0:1] -> 7-dim.
State (only packed when ``include_state: true``):
    observation.state.cartesian_position[0:6] + observation.state.gripper_position[0:1] -> 7-dim.
Cameras: DROID records three (exterior_1, exterior_2, wrist); the default observation uses
    exterior_1 + wrist. The generation target is the central_global role (exterior_1); the
    wrist view moves with the end effector.
Language: task_index -> tasks.parquet (DROID stores up to three crowd-sourced instructions
    joined by ' | ').
"""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform


# ---------------------------------------------------------------------------
# DataConfig: raw PIL video + per-dimension min_max action/state normalization.
# ---------------------------------------------------------------------------
class DroidCosmos3DataConfig:
    embodiment_tag = EmbodimentTag.OXE_DROID
    # Observation views fed to the VLM. Select a different subset/order per stream with
    # ``datasets.<block>.video_keys`` (apply_video_keys); ``available_video_keys`` bounds
    # what that override may request.
    video_keys = ["video.exterior_1", "video.wrist"]
    available_video_keys = ("video.exterior_1", "video.exterior_2", "video.wrist")
    # Canonical view roles, mapped for ALL available cameras so any video_keys subset resolves.
    # exterior_1 is the primary task-context camera (the anchor role).
    video_view_roles = {
        "video.exterior_1": "central_global",
        "video.exterior_2": "auxiliary_global",
        "video.wrist": "primary_effector",
    }
    video_view_descriptions = {
        "video.exterior_1": "primary global view from a fixed exterior camera",
        "video.exterior_2": "auxiliary global view from a fixed exterior camera",
        "video.wrist": "local effector view from a wrist camera",
    }
    state_keys = ["state.cartesian_position", "state.gripper_position"]
    # Cartesian velocity (6) + gripper target (1) = 7-dim.
    action_keys = ["action.cartesian_velocity", "action.gripper_position"]
    language_keys = ["annotation.task"]
    observation_indices = [0]
    # Placeholder: resized by apply_action_horizon() to framework.action_model.action_horizon.
    action_indices = list(range(16))
    state_indices = [0]

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        # Per-dimension min_max from the dataset statistics (no DROID action/state dimension is
        # constant, so there is no divide-by-zero). cartesian_velocity is already ~[-1, 1];
        # the gripper position is in [0, 1].
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={
                        "action.cartesian_velocity": "min_max",
                        "action.gripper_position": "min_max",
                    },
                ),
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes={
                        "state.cartesian_position": "min_max",
                        "state.gripper_position": "min_max",
                    },
                ),
            ]
        )


ROBOT_TYPE_CONFIG_MAP = {
    "droid_cosmos3": DroidCosmos3DataConfig(),
}


# ---------------------------------------------------------------------------
# Mixtures. Each dataset name is a sub-directory of datasets.<block>.data_root_dir.
# ---------------------------------------------------------------------------
DATASET_NAMED_MIXTURES = {
    # The paper trains on successful episodes only.
    "droid_success": [
        ("success", 1.0, "droid_cosmos3"),
    ],
    "droid_failure": [
        ("failure", 1.0, "droid_cosmos3"),
    ],
    "droid_all": [
        ("success", 1.0, "droid_cosmos3"),
        ("failure", 1.0, "droid_cosmos3"),
    ],
}
