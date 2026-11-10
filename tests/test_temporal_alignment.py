"""Generation-target timing (future frame aligned with the end of the action chunk) and DROID curation."""

import pytest

SAMPLING = "starVLA/dataloader/gen_target_sampling.py"
EPISODE_FILTER = "starVLA/dataloader/gr00t_lerobot/episode_filter.py"


DROID_TAU = 8 / 15  # o_{t+Delta} aligned with a_{t+H}: H = 8 actions of DROID at 15 Hz
LIBERO_TAU = 8 / 20  # the same rule on LIBERO (20 Hz) gives 0.4 s


@pytest.mark.parametrize(
    ("tau", "fps", "expected"),
    [(DROID_TAU, 15, [8]), (DROID_TAU, 30, [16]), (LIBERO_TAU, 20, [8])],  # DROID, egocentric video, LIBERO
)
def test_single_frame_offset_is_tau_times_fps_rounded(load_by_path, tau, fps, expected):
    sampling = load_by_path(SAMPLING)
    assert sampling.resolve_future_frames(tau, fps, 1)["offsets"] == expected


def test_multi_frame_grid_ends_at_horizon_on_every_stream(load_by_path):
    sampling = load_by_path(SAMPLING)
    grids = {fps: sampling.resolve_future_frames(DROID_TAU, fps, 4) for fps in (15, 30)}
    assert grids[15]["offsets"] == [2, 4, 6, 8]
    assert grids[30]["offsets"] == [4, 8, 12, 16]
    assert sampling.resolve_future_frames(LIBERO_TAU, 20, 4)["offsets"] == [2, 4, 6, 8]
    # Instants that fall between frames are rounded on each stream's own clock.
    assert sampling.resolve_future_frames(0.4, 15, 4)["offsets"] == [2, 3, 5, 6]
    for fps, grid in grids.items():
        assert grid["offsets"][-1] == grid["gen_future_offset"] == round(DROID_TAU * fps)
        assert grid["num_target_frames"] == 4
        # The dataset side rebuilds the same grid from the derived keys.
        indices = sampling.gen_target_indices(10, grid["gen_future_offset"], grid["gen_stride_frames"], 4)
        assert [i - 10 for i in indices] == grid["offsets"]


def test_unrepresentable_grid_is_rejected(load_by_path):
    sampling = load_by_path(SAMPLING)
    with pytest.raises(ValueError):
        sampling.resolve_future_frames(0.1, 15, 4)
    with pytest.raises(ValueError):
        sampling.resolve_future_frames(0.4, 15, 0)


def _episode(tasks, vel_min, vel_max, grip_min=(0.0,), grip_max=(0.0,)):
    return {
        "tasks": tasks,
        "stats/action.cartesian_velocity/min": list(vel_min),
        "stats/action.cartesian_velocity/max": list(vel_max),
        "stats/action.gripper_position/min": list(grip_min),
        "stats/action.gripper_position/max": list(grip_max),
    }


ACTION_KEYS = ["action.cartesian_velocity", "action.gripper_position"]
SPEC = {"require_instruction": True, "require_motion": True, "motion_tolerance": 1e-6}


@pytest.mark.parametrize(
    ("episode", "reason"),
    [
        (_episode(["pick up the cup"], [-0.2] * 6, [0.3] * 6), None),
        (_episode([""], [-0.2] * 6, [0.3] * 6), "no_instruction"),
        (_episode([], [-0.2] * 6, [0.3] * 6), "no_instruction"),
        (_episode(["wipe the table"], [0.0] * 6, [0.0] * 6), "no_motion"),
        (_episode(["open the drawer"], [0.0] * 6, [0.0] * 6, (0.0,), (1.0,)), None),  # gripper moves
        (_episode(["stack the cups"], [float("nan")] * 6, [0.1] * 6), "invalid_action"),
    ],
)
def test_droid_episode_curation(load_by_path, episode, reason):
    episode_filter = load_by_path(EPISODE_FILTER)
    assert episode_filter.episode_rejection_reason(episode, episode.keys(), ACTION_KEYS, SPEC) == reason


def test_episode_filter_requires_metadata_columns(load_by_path):
    episode_filter = load_by_path(EPISODE_FILTER)
    episode = {"tasks": ["pick"]}
    with pytest.raises(ValueError, match=r"stats/action\.cartesian_velocity/min"):
        episode_filter.episode_rejection_reason(episode, episode.keys(), ACTION_KEYS, SPEC)


def test_episode_filter_config_parsing(load_by_path):
    episode_filter = load_by_path(EPISODE_FILTER)
    assert episode_filter.parse_episode_filter({}) is None
    assert episode_filter.parse_episode_filter({"episode_filter": {"require_instruction": False}}) is None
    spec = episode_filter.parse_episode_filter({"episode_filter": {"require_motion": True}})
    assert spec == {"require_instruction": False, "require_motion": True, "motion_tolerance": 1e-6}
    with pytest.raises(ValueError, match="Unknown episode_filter keys"):
        episode_filter.parse_episode_filter({"episode_filter": {"min_hours": 1}})
