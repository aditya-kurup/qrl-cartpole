import gymnasium as gym
import numpy as np
import pytest

from qrl_cartpole.shift import DynamicsShift, make_cartpole


def test_default_shift_reproduces_unmodified_cartpole():
    baseline = gym.make("CartPole-v1")
    shifted = make_cartpole(DynamicsShift())
    try:
        obs_a, _ = baseline.reset(seed=123)
        obs_b, _ = shifted.reset(seed=123)
        np.testing.assert_array_equal(obs_a, obs_b)
        for action in (0, 1, 1):
            next_a = baseline.step(action)
            next_b = shifted.step(action)
            np.testing.assert_array_equal(next_a[0], next_b[0])
            assert next_a[1:4] == next_b[1:4]
    finally:
        baseline.close()
        shifted.close()


@pytest.mark.parametrize(
    ("name", "multiplier"),
    [(name, value) for name, value in (
        ("gravity", 1.5), ("masscart", 1.5), ("masspole", 1.5),
        ("length", 1.5), ("force_mag", 1.5),
    )],
)
def test_each_physics_parameter_changes_next_state(name, multiplier):
    baseline = gym.make("CartPole-v1")
    shift = DynamicsShift(**{name: multiplier})
    changed = make_cartpole(shift)
    state = np.asarray([0.1, 0.2, 0.05, 0.1], dtype=np.float64)
    try:
        baseline.reset(seed=1)
        changed.reset(seed=1)
        baseline.unwrapped.state = state.copy()
        changed.unwrapped.state = state.copy()
        next_a = baseline.step(1)[0]
        next_b = changed.step(1)[0]
        assert np.max(np.abs(next_a - next_b)) > 1e-8
    finally:
        baseline.close()
        changed.close()


def test_observation_noise_is_seed_reproducible_and_velocity_masking_hides_velocity():
    shift = DynamicsShift(observation_noise_std=0.03, mask_velocity=True)
    first, second = make_cartpole(shift), make_cartpole(shift)
    try:
        obs_a, _ = first.reset(seed=18)
        obs_b, _ = second.reset(seed=18)
        np.testing.assert_array_equal(obs_a, obs_b)
        assert obs_a[1] == 0.0 and obs_a[3] == 0.0
    finally:
        first.close()
        second.close()
