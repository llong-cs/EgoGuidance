"""Validation action-MSE metric: numerics of ``mean_action_mse`` and the logged metric name."""

import numpy as np
import pytest

TRAINING_ENTRIES = [
    "starVLA/training/train_starvla.py",
    "starVLA/training/train_starvla_gen_cotrain.py",
]


@pytest.fixture(scope="module")
def mean_action_mse():
    # trainer_tools imports torch / accelerate / transformers at module level.
    pytest.importorskip("torch")
    trainer_tools = pytest.importorskip("starVLA.training.trainer_utils.trainer_tools")
    return trainer_tools.mean_action_mse


def test_mean_action_mse_is_elementwise_squared_error(mean_action_mse):
    predicted = np.array([[1.0, 3.0], [2.0, -1.0]])
    target = np.array([[0.0, 1.0], [2.0, 1.0]])
    assert mean_action_mse(predicted, target) == pytest.approx(2.25)


def test_mean_action_mse_rejects_shape_mismatch(mean_action_mse):
    with pytest.raises(ValueError, match="shapes must match"):
        mean_action_mse(np.zeros((2, 3)), np.zeros((2, 4)))


def test_mean_action_mse_rejects_non_finite_result(mean_action_mse):
    with pytest.raises(FloatingPointError, match="non-finite"):
        mean_action_mse(np.array([np.nan]), np.array([0.0]))


@pytest.mark.parametrize("relative_path", TRAINING_ENTRIES)
def test_training_entries_log_namespaced_action_mse(repo_root, relative_path):
    # tools/analysis/extract_metrics.py reads this metric name from the training logs.
    source = (repo_root / relative_path).read_text()
    assert 'step_metrics["val/action_mse"]' in source
