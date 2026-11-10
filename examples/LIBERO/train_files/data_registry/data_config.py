"""LIBERO benchmark — data config, embodiment tag, and mixtures.

Auto-discovered by ``starVLA.dataloader.gr00t_lerobot.registry.discover_and_merge``
(which scans ``examples/*/train_files/data_registry/data_config.py``).

Data: the four LIBERO suites converted to LeRobot v2.0 (``*_no_noops_1.0.0_lerobot``),
with ``examples/LIBERO/train_files/modality.json`` copied into each suite's ``meta/``
(see ``examples/LIBERO/data_preparation.sh``). Point ``datasets.<block>.data_root_dir``
at the directory that contains the four suites.

Action space: Δ-EEF xyz (3) + Δ-EEF roll/pitch/yaw (3) + gripper (1) -> 7-dim.
Cameras: front (``primary_image`` -> central_global) + wrist (``wrist_image`` -> primary_effector).
Language: task_index -> tasks.jsonl.
"""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform


# ---------------------------------------------------------------------------
# DataConfig
# ---------------------------------------------------------------------------
class Libero4in1DataConfig:
    embodiment_tag = EmbodimentTag.FRANKA
    video_keys = [
        "video.primary_image",
        "video.wrist_image",
    ]
    # Canonical view roles and the per-camera text that precedes each image in the prompt.
    video_view_roles = {
        "video.primary_image": "central_global",
        "video.wrist_image": "primary_effector",
    }
    video_view_descriptions = {
        "video.primary_image": "primary global view from a fixed front camera",
        "video.wrist_image": "local effector view from a wrist camera",
    }
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.roll",
        "state.pitch",
        "state.yaw",
        "state.pad",
        "state.gripper",
    ]
    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    # Resized by apply_action_horizon() to framework.action_model.action_horizon.
    action_indices = list(range(8))
    state_indices = [0]

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            # State is only packed into samples when datasets.<block>.include_state is true.
            "state": ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={
                        "action.x": "min_max",
                        "action.y": "min_max",
                        "action.z": "min_max",
                        "action.roll": "min_max",
                        "action.pitch": "min_max",
                        "action.yaw": "min_max",
                        "action.gripper": "binary",
                    },
                ),
            ]
        )


ROBOT_TYPE_CONFIG_MAP = {
    "libero_franka": Libero4in1DataConfig(),
}


# ---------------------------------------------------------------------------
# Mixtures. Each dataset name is a sub-directory of datasets.<block>.data_root_dir.
# ---------------------------------------------------------------------------
DATASET_NAMED_MIXTURES = {
    "libero_all": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "libero_goal": [
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
}
