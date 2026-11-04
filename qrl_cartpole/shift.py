"""Controlled CartPole dynamics and observation shifts."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class DynamicsShift:
    """Multipliers are relative to Gymnasium CartPole-v1 nominal settings."""

    gravity: float = 1.0
    masscart: float = 1.0
    masspole: float = 1.0
    length: float = 1.0
    force_mag: float = 1.0
    observation_noise_std: float = 0.0
    mask_velocity: bool = False

    def __post_init__(self) -> None:
        for name in ("gravity", "masscart", "masspole", "length", "force_mag"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} multiplier must be positive and finite")
        if not np.isfinite(self.observation_noise_std) or self.observation_noise_std < 0:
            raise ValueError("observation_noise_std must be finite and nonnegative")

    def to_dict(self) -> dict:
        return asdict(self)


class ObservationShift(gym.ObservationWrapper):
    """Optionally add reproducible Gaussian noise, then hide both velocities."""

    def __init__(self, env: gym.Env, noise_std: float, mask_velocity: bool):
        super().__init__(env)
        self.noise_std = noise_std
        self.mask_velocity = mask_velocity
        self._rng = np.random.default_rng(0)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        observation, info = self.env.reset(seed=seed, options=options)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        return self.observation(observation), info

    def observation(self, observation: np.ndarray) -> np.ndarray:
        result = np.asarray(observation, dtype=np.float32).copy()
        if self.noise_std:
            result += self._rng.normal(0.0, self.noise_std, size=result.shape).astype(np.float32)
        if self.mask_velocity:
            result[[1, 3]] = 0.0
        return result


def make_cartpole(
    shift: DynamicsShift | None = None,
    *,
    render_mode: str | None = None,
) -> gym.Env:
    """Create CartPole-v1 with per-instance parameters and derived terms updated."""
    settings = shift or DynamicsShift()
    env = gym.make("CartPole-v1", render_mode=render_mode)
    base = env.unwrapped
    base.gravity *= settings.gravity
    base.masscart *= settings.masscart
    base.masspole *= settings.masspole
    base.length *= settings.length
    base.force_mag *= settings.force_mag
    base.total_mass = base.masscart + base.masspole
    base.polemass_length = base.masspole * base.length
    if settings.observation_noise_std or settings.mask_velocity:
        env = ObservationShift(env, settings.observation_noise_std, settings.mask_velocity)
    return env


def sample_shift(rng: np.random.Generator, ranges: dict[str, tuple[float, float]]) -> DynamicsShift:
    """Sample configured dynamics multipliers uniformly in log space."""
    values = {}
    for name, (low, high) in ranges.items():
        if low <= 0 or high < low:
            raise ValueError(f"invalid positive range for {name}: {(low, high)}")
        values[name] = float(np.exp(rng.uniform(np.log(low), np.log(high))))
    return DynamicsShift(**values)
