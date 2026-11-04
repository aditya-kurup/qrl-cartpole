"""Preliminary, resumable CartPole comparison for presentation runs."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch.distributions import Categorical

from .policies import ClassicalPolicy, VariationalQuantumPolicy
from .shift import DynamicsShift, make_cartpole


POLICIES = ("quantum_full", "mlp_raw", "mlp_tanhpi", "quantum_noent", "quantum_frozen")
GRID = (0.5, 1.0, 2.0)
COLORS = {
    "quantum_full": "#0072B2",
    "mlp_raw": "#E69F00",
    "mlp_tanhpi": "#009E73",
    "quantum_noent": "#CC79A7",
    "quantum_frozen": "#D55E00",
}
_BASE_METADATA: dict | None = None


def make_policy(name: str) -> torch.nn.Module:
    if name == "quantum_full":
        return VariationalQuantumPolicy(n_layers=2, readout="zx")
    if name == "quantum_noent":
        return VariationalQuantumPolicy(n_layers=2, readout="zx", entanglement=False)
    if name == "quantum_frozen":
        return VariationalQuantumPolicy(n_layers=2, readout="zx", trainable_circuit=False)
    if name == "mlp_raw":
        return ClassicalPolicy()
    if name == "mlp_tanhpi":
        return ClassicalPolicy(tanhpi_input=True)
    raise ValueError(f"unknown policy: {name}")


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True, timeout=2).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _versions() -> dict:
    packages = ("gymnasium", "numpy", "torch", "matplotlib")
    return {name: importlib.metadata.version(name) for name in packages}


def _base_metadata() -> dict:
    global _BASE_METADATA
    if _BASE_METADATA is None:
        try:
            dirty = bool(subprocess.run(["git", "status", "--porcelain"], check=True,
                                        capture_output=True, text=True, timeout=2).stdout.strip())
        except (OSError, subprocess.SubprocessError):
            dirty = None
        _BASE_METADATA = {"git_commit": _git_commit(), "git_worktree_dirty": dirty,
                          "machine": platform.machine(), "platform": platform.platform(),
                          "python": sys.version.split()[0], "packages": _versions()}
    return _BASE_METADATA


def _atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def _atomic_torch(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def _model_action(model: torch.nn.Module, observation: np.ndarray) -> int:
    with torch.no_grad():
        logits = model(torch.as_tensor(observation, dtype=torch.float32))
        return int(logits.argmax().item())


def evaluate(model: torch.nn.Module, shift: DynamicsShift, seeds: list[int]) -> list[int]:
    env = make_cartpole(shift)
    scores = []
    try:
        for seed in seeds:
            observation, _ = env.reset(seed=seed)
            total = 0
            done = False
            while not done:
                observation, _, terminated, truncated, _ = env.step(_model_action(model, observation))
                total += 1
                done = terminated or truncated
            scores.append(total)
    finally:
        env.close()
    return scores


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def train_one(policy_name: str, seed: int, episodes: int, output_dir: str,
              eval_interval: int = 50, eval_episodes: int = 10,
              final_eval_episodes: int = 30) -> dict:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    result_path = root / f"run_{policy_name}_seed_{seed}.json"
    checkpoint_path = root / f"checkpoint_{policy_name}_seed_{seed}.pt"

    if result_path.exists():
        saved = json.loads(result_path.read_text())
        if saved.get("complete") and saved.get("episodes") == episodes:
            return saved

    _seed_all(seed)
    model = make_policy(policy_name)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=0.01)
    env = gym.make("CartPole-v1")
    episode_returns: list[float] = []
    episode_seconds: list[float] = []
    evaluations: list[dict] = []
    start_episode = 0
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("episodes_target") == episodes:
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            episode_returns = checkpoint["episode_returns"]
            episode_seconds = checkpoint["episode_seconds"]
            evaluations = checkpoint["evaluations"]
            start_episode = checkpoint["completed_episodes"]
            torch.set_rng_state(checkpoint["torch_rng"])

    began = time.perf_counter()
    try:
        for episode_idx in range(start_episode, episodes):
            episode_start = time.perf_counter()
            observation, _ = env.reset(seed=seed + episode_idx)
            log_probs, rewards = [], []
            done = False
            while not done:
                logits = model(torch.as_tensor(observation, dtype=torch.float32))
                distribution = Categorical(logits=logits)
                action = distribution.sample()
                observation, reward, terminated, truncated, _ = env.step(int(action.item()))
                log_probs.append(distribution.log_prob(action))
                rewards.append(float(reward))
                done = terminated or truncated

            returns = []
            running = 0.0
            for reward in reversed(rewards):
                running = reward + 0.99 * running
                returns.append(running)
            returns_tensor = torch.tensor(list(reversed(returns)), dtype=torch.float32)
            if returns_tensor.numel() > 1:
                returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std(unbiased=False) + 1e-8)
            loss = -(torch.stack(log_probs) * returns_tensor).sum()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            episode_returns.append(float(sum(rewards)))

            episode = episode_idx + 1
            if episode % eval_interval == 0:
                eval_seeds = [600_000 + (episode // eval_interval) * 100 + j for j in range(eval_episodes)]
                scores = evaluate(model, DynamicsShift(), eval_seeds)
                evaluations.append({"episode": episode, "scores": scores,
                                    "mean_return": float(np.mean(scores)),
                                    "std_return": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0})
            episode_seconds.append(time.perf_counter() - episode_start)
            partial = _run_record(policy_name, seed, episodes, episode_returns, episode_seconds,
                                  evaluations, model, complete=False)
            _atomic_json(result_path, partial)
            _atomic_torch(checkpoint_path, {
                "episodes_target": episodes,
                "completed_episodes": episode,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "episode_returns": episode_returns,
                "episode_seconds": episode_seconds,
                "evaluations": evaluations,
                "torch_rng": torch.get_rng_state(),
            })
            if episode % 25 == 0:
                print(f"[{policy_name} seed={seed}] {episode}/{episodes} episodes", flush=True)

        final_seeds = [700_000 + j for j in range(final_eval_episodes)]
        final_scores = evaluate(model, DynamicsShift(), final_seeds)
        record = _run_record(policy_name, seed, episodes, episode_returns, episode_seconds,
                             evaluations, model, complete=True)
        record["final_eval"] = {"seeds": final_seeds, "scores": final_scores,
                                "mean_return": float(np.mean(final_scores)),
                                "std_return": float(np.std(final_scores, ddof=1)) if len(final_scores) > 1 else 0.0}
        record["wall_seconds"] = time.perf_counter() - began
        _atomic_json(result_path, record)
        _atomic_torch(checkpoint_path, {"episodes_target": episodes,
                                        "completed_episodes": episodes,
                                        "model": model.state_dict(),
                                        "optimizer": optimizer.state_dict(),
                                        "episode_returns": episode_returns,
                                        "episode_seconds": episode_seconds,
                                        "evaluations": evaluations,
                                        "torch_rng": torch.get_rng_state()})
        return record
    finally:
        env.close()


def _run_record(policy_name: str, seed: int, episodes: int, episode_returns: list[float],
                episode_seconds: list[float], evaluations: list[dict], model: torch.nn.Module,
                complete: bool) -> dict:
    return {
        "policy": policy_name,
        "seed": seed,
        "episodes": episodes,
        "completed_episodes": len(episode_returns),
        "complete": complete,
        "episode_returns": episode_returns,
        "episode_seconds": episode_seconds,
        "seconds_per_training_episode": float(np.mean(episode_seconds)) if episode_seconds else None,
        "evaluations": evaluations,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **_base_metadata(),
            "config": {"learning_rate": 0.01, "gamma": 0.99, "gradient_clip_norm": 5.0,
                       "entropy_coefficient": 0.0, "eval_interval": 50,
                       "periodic_eval_episodes": 10, "final_eval_episodes": 30},
        },
    }


def evaluate_shifts(policy_name: str, seed: int, output_dir: str,
                    eval_episodes: int = 10) -> dict:
    root = Path(output_dir)
    result_path = root / f"shift_{policy_name}_seed_{seed}.json"
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        if saved.get("complete"):
            return saved
    run = json.loads((root / f"run_{policy_name}_seed_{seed}.json").read_text())
    model = make_policy(policy_name)
    checkpoint = torch.load(root / f"checkpoint_{policy_name}_seed_{seed}.pt",
                            map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    cells = {}
    if result_path.exists():
        partial = json.loads(result_path.read_text())
        if not partial.get("complete"):
            cells = partial.get("cells", {})
    for gi, gravity in enumerate(GRID):
        for li, length in enumerate(GRID):
            key = f"g{gravity:g}_l{length:g}"
            if key in cells:
                continue
            seeds = [800_000 + gi * 10_000 + li * 100 + i for i in range(eval_episodes)]
            scores = evaluate(model, DynamicsShift(gravity=gravity, length=length), seeds)
            cells[key] = {"gravity_multiplier": gravity, "length_multiplier": length,
                          "scores": scores, "mean_return": float(np.mean(scores))}
            partial = {"policy": policy_name, "seed": seed, "complete": False,
                       "grid": list(GRID), "eval_episodes": eval_episodes, "cells": cells,
                       "source_run": run["metadata"]}
            _atomic_json(result_path, partial)
    non_normal = [cell["mean_return"] for key, cell in cells.items() if key != "g1_l1"]
    result = {"policy": policy_name, "seed": seed, "complete": True,
              "grid": list(GRID), "eval_episodes": eval_episodes, "cells": cells,
              "robustness_score": float(np.mean(non_normal) / 500.0),
              "source_run": run["metadata"]}
    _atomic_json(result_path, result)
    return result


def _completed(root: Path, policy_names: list[str]) -> dict[str, list[dict]]:
    grouped = {name: [] for name in policy_names}
    for name in policy_names:
        for path in sorted(root.glob(f"run_{name}_seed_*.json")):
            row = json.loads(path.read_text())
            if row.get("complete"):
                grouped[name].append(row)
    return grouped


def _shift_completed(root: Path, policy_name: str) -> list[dict]:
    rows = []
    for path in sorted(root.glob(f"shift_{policy_name}_seed_*.json")):
        row = json.loads(path.read_text())
        if row.get("complete"):
            rows.append(row)
    return rows


def update_artifacts(root: Path, finished_policy_names: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 16, "axes.labelsize": 14,
                         "xtick.labelsize": 12, "ytick.labelsize": 12, "figure.dpi": 200})
    grouped = _completed(root, list(POLICIES))
    available = [name for name in POLICIES if grouped[name]]
    if not available:
        return

    fig, ax = plt.subplots(figsize=(11.5, 6.3))
    for name in available:
        runs = grouped[name]
        max_episode = min(len(row["episode_returns"]) for row in runs)
        x = np.asarray([item["episode"] for item in runs[0]["evaluations"]
                        if item["episode"] <= max_episode], dtype=float)
        if not len(x):
            continue
        curves = np.asarray([[next((e["mean_return"] for e in row["evaluations"]
                                    if e["episode"] == episode), np.nan) for episode in x]
                             for row in runs], dtype=float)
        mean = np.nanmean(curves, axis=0)
        std = np.nanstd(curves, axis=0, ddof=1) if len(runs) > 1 else np.zeros_like(mean)
        ax.plot(x, mean, color=COLORS[name], linewidth=2.7,
                label=f"{name.replace('_', ' ')} (n={len(runs)})")
        ax.fill_between(x, np.maximum(0, mean - std), np.minimum(500, mean + std),
                        color=COLORS[name], alpha=0.14, linewidth=0)
        final_points = [run["final_eval"]["mean_return"] for run in runs if "final_eval" in run]
        if final_points:
            ax.scatter([episodes_x := max_episode] * len(final_points), final_points,
                       color=COLORS[name], edgecolors="white", linewidths=0.8, s=45, zorder=4)
    ax.axhline(475, color="#666666", linestyle="--", linewidth=1.3, label="475-step reference")
    ax.set(xlim=(0, None), ylim=(0, 510), xlabel="Training episode",
           ylabel="Greedy CartPole return (10 episodes)")
    ax.set_title(f"PRELIMINARY: learning curves ({max(len(v) for v in grouped.values())} seeds/policy)")
    ax.grid(axis="y", alpha=0.24)
    ax.legend(frameon=False, ncol=2, fontsize=10)
    fig.text(0.5, 0.012, "Line = seed mean; shaded area = ±1 SD; dots = per-seed final 30-episode mean.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(root / "learning_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    heat_means: dict[str, np.ndarray] = {}
    for name in finished_policy_names:
        shifts = _shift_completed(root, name)
        if not shifts:
            continue
        matrix = np.zeros((len(GRID), len(GRID)), dtype=float)
        for gi, gravity in enumerate(GRID):
            for li, length in enumerate(GRID):
                key = f"g{gravity:g}_l{length:g}"
                per_seed = [row["cells"][key]["mean_return"] for row in shifts]
                matrix[gi, li] = np.mean(per_seed)
        heat_means[name] = matrix
        fig, ax = plt.subplots(figsize=(7.7, 6.1))
        im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=500, aspect="equal")
        ax.set_xticks(range(3), [f"{v:g}×" for v in GRID])
        ax.set_yticks(range(3), [f"{v:g}×" for v in GRID])
        ax.set(xlabel="Pole length multiplier", ylabel="Gravity multiplier")
        ax.set_title(f"PRELIMINARY: {name.replace('_', ' ')} (n={len(shifts)} seeds)")
        for gi in range(3):
            for li in range(3):
                ax.text(li, gi, f"{matrix[gi, li]:.0f}", ha="center", va="center",
                        color="white" if matrix[gi, li] < 280 else "black", fontsize=15, fontweight="bold")
        fig.colorbar(im, ax=ax, label="Mean return (10 episodes/seed)", shrink=0.84)
        fig.tight_layout()
        fig.savefig(root / f"heatmap_{name}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    if "quantum_full" in heat_means and "mlp_tanhpi" in heat_means:
        diff = heat_means["quantum_full"] - heat_means["mlp_tanhpi"]
        vmax = max(1.0, float(np.max(np.abs(diff))))
        fig, ax = plt.subplots(figsize=(7.7, 6.1))
        im = ax.imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
        ax.set_xticks(range(3), [f"{v:g}×" for v in GRID])
        ax.set_yticks(range(3), [f"{v:g}×" for v in GRID])
        ax.set(xlabel="Pole length multiplier", ylabel="Gravity multiplier")
        n = min(len(_shift_completed(root, "quantum_full")), len(_shift_completed(root, "mlp_tanhpi")))
        ax.set_title(f"PRELIMINARY: quantum − matched MLP (n={n} seeds/policy)")
        for gi in range(3):
            for li in range(3):
                ax.text(li, gi, f"{diff[gi, li]:+.0f}", ha="center", va="center", fontsize=15,
                        color="black", fontweight="bold")
        fig.colorbar(im, ax=ax, label="Return difference", shrink=0.84)
        fig.tight_layout()
        fig.savefig(root / "heatmap_diff_quantum_vs_mlp_tanhpi.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.7))
    labels, means = [], []
    for name in available:
        rows = _shift_completed(root, name)
        if not rows:
            continue
        vals = [row["robustness_score"] for row in rows]
        x = len(labels)
        ax.bar(x, float(np.mean(vals)), color=COLORS[name], alpha=0.78, width=0.66)
        offsets = np.linspace(-0.18, 0.18, len(vals)) if len(vals) > 1 else [0]
        ax.scatter(x + offsets, vals, color="#222222", edgecolors="white", linewidths=0.6,
                   s=48, zorder=3)
        labels.append(name.replace("_", " "))
        means.append(float(np.mean(vals)))
    if labels:
        ax.set_xticks(range(len(labels)), labels, rotation=18, ha="right")
        ax.set_ylim(0, 1.03)
        ax.set_ylabel("Mean non-normal return / 500")
        shift_counts = [len(_shift_completed(root, name)) for name in available if _shift_completed(root, name)]
        seed_label = (f"{shift_counts[0]} seeds per policy" if len(set(shift_counts)) == 1
                      else "seed counts shown by dots")
        ax.set_title(f"PRELIMINARY: shift robustness ({seed_label})")
        ax.grid(axis="y", alpha=0.24)
        fig.tight_layout()
        fig.savefig(root / "robustness_bar.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    _write_table(root, grouped)
    _write_summary(root, grouped)


def _write_table(root: Path, grouped: dict[str, list[dict]]) -> None:
    rows = []
    for name in POLICIES:
        runs = grouped[name]
        if not runs:
            continue
        final = [r["final_eval"]["mean_return"] for r in runs if "final_eval" in r]
        thresholds = []
        for run in runs:
            point = next((e["episode"] for e in run["evaluations"] if e["mean_return"] >= 475), None)
            if point is not None:
                thresholds.append(point)
        seconds = [r["seconds_per_training_episode"] for r in runs if r.get("seconds_per_training_episode") is not None]
        shift_rows = _shift_completed(root, name)
        robustness = [r["robustness_score"] for r in shift_rows]
        rows.append({"policy": name, "n_seeds": len(runs),
                     "final_greedy_eval_mean": float(np.mean(final)) if final else "",
                     "final_greedy_eval_sd_across_seeds": float(np.std(final, ddof=1)) if len(final) > 1 else "",
                     "first_episode_eval_reaching_475_mean": float(np.mean(thresholds)) if thresholds else "not reached",
                     "threshold_seed_count": len(thresholds),
                     "robustness_score_non_normal_cells": float(np.mean(robustness)) if robustness else "",
                     "seconds_per_training_episode": float(np.mean(seconds)) if seconds else ""})
    path = root / "results_table.csv"
    if rows:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


def _write_summary(root: Path, grouped: dict[str, list[dict]]) -> None:
    lines = ["# Preliminary CartPole Results", ""]
    complete = [(name, runs) for name, runs in grouped.items() if runs]
    total = sum(len(runs) for _, runs in complete)
    episode_counts = sorted({run["episodes"] for _, runs in complete for run in runs})
    lines.append(f"PRELIMINARY: {total} training runs, {len(complete)} policies, {', '.join(map(str, episode_counts)) or 'unknown'} episodes; REINFORCE with classical statevector simulation.")
    for name, runs in complete:
        final = [r["final_eval"]["mean_return"] for r in runs if "final_eval" in r]
        shifts = _shift_completed(root, name)
        robust = [r["robustness_score"] for r in shifts]
        final_text = (f"final {np.mean(final):.1f} ± {np.std(final, ddof=1) if len(final) > 1 else 0:.1f}"
                      if final else "final pending")
        shift_text = f"shift {np.mean(robust):.3f}/1" if robust else "shift pending"
        lines.append(f"{name}: {final_text}; {shift_text} (n={len(runs)}).")
    lines.append("Curves show seed mean ± 1 SD and seed dots; no significance tests. Few seeds, one task family, and no quantum hardware or advantage claim.")
    (root / "SUMMARY.md").write_text("\n".join(lines[:10]) + "\n")


def benchmark(output_dir: Path, episodes: int = 8) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    measured = {}
    for name in ("quantum_full", "mlp_raw"):
        path = output_dir / f"benchmark_{name}.json"
        started = time.perf_counter()
        row = train_one(name, 71_337, episodes, str(output_dir / "benchmark_runs"),
                        eval_interval=episodes + 1, final_eval_episodes=1, eval_episodes=1)
        seconds = [x for x in row["episode_seconds"] if x > 0]
        measured[name] = {"seconds_per_training_episode_mean": float(np.mean(seconds)),
                          "seconds_per_training_episode_median": float(np.median(seconds)),
                          "episode_count": len(seconds), "benchmark_wall_seconds": time.perf_counter() - started}
        _atomic_json(path, measured[name])
    result = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "measurements": measured}
    _atomic_json(output_dir / "runtime_benchmark.json", result)
    return result


def range_pilot(output_dir: Path, episodes: int = 100) -> dict:
    """Briefly train two representative policies and test the requested shift grid."""
    pilot_root = output_dir / "range_pilot"
    pilot_root.mkdir(parents=True, exist_ok=True)
    trained = {}
    with ProcessPoolExecutor(max_workers=min(2, os.cpu_count() or 1)) as pool:
        futures = {
            pool.submit(train_one, name, 19_731, episodes, str(pilot_root / name),
                        eval_interval=episodes + 1, eval_episodes=1,
                        final_eval_episodes=1): name
            for name in ("quantum_full", "mlp_tanhpi")
        }
        for future in as_completed(futures):
            trained[futures[future]] = future.result()
    pilot = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
             "training_episodes": episodes, "evaluation_episodes_per_cell": 10,
             "grid": list(GRID), "policies": {}}
    for name in ("quantum_full", "mlp_tanhpi"):
        model = make_policy(name)
        checkpoint = torch.load(pilot_root / name / f"checkpoint_{name}_seed_19731.pt",
                                map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        cells = {}
        for gi, gravity in enumerate(GRID):
            for li, length in enumerate(GRID):
                seeds = [930_000 + gi * 10_000 + li * 100 + i for i in range(10)]
                scores = evaluate(model, DynamicsShift(gravity=gravity, length=length), seeds)
                cells[f"g{gravity:g}_l{length:g}"] = {
                    "gravity_multiplier": gravity, "length_multiplier": length,
                    "scores": scores, "mean_return": float(np.mean(scores)),
                }
        pilot["policies"][name] = {"training_final_mean": trained[name]["final_eval"]["mean_return"],
                                    "cells": cells}
    all_ceiling = all(cell["mean_return"] >= 499.5
                      for policy in pilot["policies"].values() for cell in policy["cells"].values())
    pilot["requested_grid_all_ceiling"] = all_ceiling
    pilot["decision"] = "use expanded grid [0.25, 0.5, 1, 2, 4]" if all_ceiling else "use requested grid [0.5, 1, 2]"
    _atomic_json(output_dir / "range_pilot.json", pilot)
    return pilot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/fast"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 21, 42, 84, 126])
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--benchmark-episodes", type=int, default=8)
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--pilot-episodes", type=int, default=100)
    parser.add_argument("--skip-shifts", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.benchmark_only:
        report = benchmark(args.out_dir, args.benchmark_episodes)
        print(json.dumps(report, indent=2))
        return
    if args.pilot_only:
        report = range_pilot(args.out_dir, args.pilot_episodes)
        print(json.dumps(report, indent=2))
        return

    workers = os.cpu_count() or 1
    run_config = {"policies": list(POLICIES), "seeds": args.seeds, "episodes": args.episodes,
                  "eval_interval": 50, "eval_episodes": 10, "final_eval_episodes": 30,
                  "learning_rate": 0.01, "gamma": 0.99, "gradient_clip_norm": 5.0,
                  "entropy_coefficient": 0.0, "workers": workers,
                  "created_utc": datetime.now(timezone.utc).isoformat(), "git_commit": _git_commit()}
    config_path = args.out_dir / "run_config.json"
    _atomic_json(config_path, run_config)
    print(f"Using {workers} worker processes; each policy uses {len(args.seeds)} seed jobs.", flush=True)

    completed_names = []
    for name in POLICIES:
        print(f"\n=== Training {name} ===", flush=True)
        futures = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for seed in args.seeds:
                futures.append(pool.submit(train_one, name, seed, args.episodes, str(args.out_dir)))
            for future in as_completed(futures):
                row = future.result()
                print(f"Finished {name} seed={row['seed']} in {row.get('wall_seconds', 0):.1f}s; "
                      f"final={row.get('final_eval', {}).get('mean_return', float('nan')):.1f}", flush=True)
        completed_names.append(name)

        if not args.skip_shifts:
            print(f"Evaluating zero-shot shifts for {name}", flush=True)
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(evaluate_shifts, name, seed, str(args.out_dir)) for seed in args.seeds]
                for future in as_completed(futures):
                    row = future.result()
                    print(f"Finished shift grid {name} seed={row['seed']}; "
                          f"robustness={row['robustness_score']:.3f}", flush=True)
        update_artifacts(args.out_dir, completed_names)

    print(f"Results and figures updated in {args.out_dir}")


if __name__ == "__main__":
    main()
