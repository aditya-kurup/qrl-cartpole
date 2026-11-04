"""Train and compare quantum and classical REINFORCE policies on CartPole-v1."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch.distributions import Categorical

from .policies import ClassicalPolicy, VariationalQuantumPolicy


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def discounted_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    result = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        result.append(running)
    returns = torch.tensor(list(reversed(result)), dtype=torch.float32)
    if returns.numel() > 1:
        returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)
    return returns


def make_policy(name: str, quantum_layers: int, quantum_readout: str) -> torch.nn.Module:
    if name == "quantum":
        return VariationalQuantumPolicy(n_layers=quantum_layers, readout=quantum_readout)
    return ClassicalPolicy()


def evaluate(policy: torch.nn.Module, seed: int, episodes: int = 5) -> float:
    env = gym.make("CartPole-v1")
    scores = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        total = 0
        done = False
        while not done:
            with torch.no_grad():
                logits = policy(torch.as_tensor(observation, dtype=torch.float32))
                action = int(logits.argmax().item())
            observation, _, terminated, truncated, _ = env.step(action)
            total += 1
            done = terminated or truncated
        scores.append(total)
    env.close()
    return float(np.mean(scores))


def train_one(policy_name: str, seed: int, episodes: int, learning_rate: float,
              gamma: float, eval_interval: int, output_dir: Path,
              quantum_layers: int = 2, entropy_coef: float = 0.0,
              quantum_readout: str = "zx") -> dict:
    seed_everything(seed)
    policy = make_policy(policy_name, quantum_layers, quantum_readout)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    env = gym.make("CartPole-v1")
    episode_returns: list[float] = []
    evaluations: list[dict] = []

    print(f"[{policy_name} seed={seed}] Training REINFORCE for {episodes} episodes to compare policy learning.")
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        log_probs = []
        entropies = []
        rewards = []
        done = False
        while not done:
            logits = policy(torch.as_tensor(observation, dtype=torch.float32))
            distribution = Categorical(logits=logits)
            action = distribution.sample()
            observation, reward, terminated, truncated, _ = env.step(int(action.item()))
            log_probs.append(distribution.log_prob(action))
            entropies.append(distribution.entropy())
            rewards.append(float(reward))
            done = terminated or truncated

        returns = discounted_returns(rewards, gamma)
        log_prob_tensor = torch.stack(log_probs)
        entropy = torch.stack(entropies)
        loss = -(log_prob_tensor * returns).sum() - entropy_coef * entropy.sum()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)
        optimizer.step()
        episode_returns.append(sum(rewards))

        if (episode + 1) % eval_interval == 0 or episode == 0 or episode + 1 == episodes:
            score = evaluate(policy, seed * 10_000 + episode * 5, episodes=5)
            evaluations.append({"episode": episode + 1, "mean_return": score})
            print(f"  episode {episode + 1:4d}/{episodes}: train={episode_returns[-1]:6.1f}, eval(5)={score:6.1f}")
    env.close()

    result = {
        "policy": policy_name,
        "seed": seed,
        "episodes": episodes,
        "episode_returns": episode_returns,
        "evaluations": evaluations,
        "parameter_count": sum(parameter.numel() for parameter in policy.parameters()),
        "quantum_layers": quantum_layers,
        "entropy_coef": entropy_coef,
        "quantum_readout": quantum_readout,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{policy_name}_seed_{seed}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 21, 42])
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--quantum-layers", type=int, default=2)
    parser.add_argument("--quantum-readout", choices=("z", "zx"), default="zx")
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--policies", type=str, nargs="+", choices=("quantum", "classical"),
                        default=["quantum", "classical"])
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/final"))
    args = parser.parse_args()
    if args.episodes < 1 or args.eval_interval < 1 or args.quantum_layers < 1:
        parser.error("--episodes, --eval-interval, and --quantum-layers must be positive")

    results = []
    for seed in args.seeds:
        for policy_name in args.policies:
            results.append(train_one(policy_name, seed, args.episodes, args.learning_rate,
                                     args.gamma, args.eval_interval, args.out_dir,
                                     args.quantum_layers, args.entropy_coef, args.quantum_readout))
    (args.out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    from .plotting import make_figures
    make_figures(results, args.out_dir)


if __name__ == "__main__":
    main()
