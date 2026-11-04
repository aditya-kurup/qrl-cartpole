"""Presentation graphics for CartPole policy comparisons."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {"quantum": "#d1495b", "classical": "#00798c"}


def make_figures(results: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), gridspec_kw={"width_ratios": [1.7, 1]})
    for name in ("quantum", "classical"):
        runs = [row for row in results if row["policy"] == name]
        if not runs:
            continue
        curves = np.asarray([row["episode_returns"] for row in runs], dtype=float)
        window = min(15, curves.shape[1]) if curves.shape[1] >= 15 else 1
        smooth = np.asarray([np.convolve(curve, np.ones(window) / window, mode="valid") for curve in curves])
        x = np.arange(window, curves.shape[1] + 1)
        mean = smooth.mean(axis=0)
        spread = smooth.std(axis=0)
        axes[0].plot(x, mean, color=COLORS[name], lw=2.5, label=f"{name.title()} policy")
        axes[0].fill_between(x, mean - spread, mean + spread, color=COLORS[name], alpha=0.15, linewidth=0)
        final_scores = [row["evaluations"][-1]["mean_return"] for row in runs]
        axes[1].scatter([name.title()] * len(final_scores), final_scores, color=COLORS[name], s=58, zorder=3)
        axes[1].plot([name.title()], [np.mean(final_scores)], marker="D", color="#222222", markersize=8)
    axes[0].axhline(500, color="#777777", lw=1, ls="--", alpha=0.7)
    axes[0].set(title="Learning on CartPole-v1", xlabel="Training episode", ylabel="Episode return", ylim=(0, 520))
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].set(title="Final greedy evaluation", ylabel="Mean return (5 episodes)", ylim=(0, 520))
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("Quantum policy vs classical baseline", fontsize=15, fontweight="bold")
    fig.text(0.5, 0.01, "Lines show seed mean; shaded band is ±1 standard deviation. Dashed line = 500-step cap.",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(output_dir / "learning_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    draw_circuit(output_dir / "variational_circuit.png")


def draw_circuit(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.set_xlim(-0.7, 10.2)
    ax.set_ylim(-0.7, 3.7)
    ax.axis("off")
    labels = ["q0", "q1", "q2", "q3"]
    yvals = [3, 2, 1, 0]
    for label, y in zip(labels, yvals):
        ax.plot([0, 9.2], [y, y], color="#333333", lw=1.2)
        ax.text(-0.5, y, label, ha="right", va="center", fontsize=11)
        ax.text(0.25, y + 0.26, f"RY(x{label[1]})", ha="center", fontsize=8, color="#555555")
        ax.add_patch(plt.Rectangle((0.0, y - 0.17), 0.55, 0.34, color="#dceef2", ec="#00798c", lw=1.2))
    for layer, x in enumerate((2.1, 5.3), start=1):
        ax.text(x + 0.35, 3.52, f"Variational layer {layer}", ha="center", fontsize=9, color="#555555")
        for q, y in enumerate(yvals):
            ax.add_patch(plt.Rectangle((x, y - 0.19), 0.7, 0.38, color="#fde2e4", ec="#d1495b", lw=1.2))
            ax.text(x + 0.35, y, r"$R_y(\theta)$", ha="center", va="center", fontsize=8)
        for q in range(3):
            y1, y2 = yvals[q], yvals[q + 1]
            ax.plot([x + 1.05, x + 1.05], [y1, y2], color="#333333", lw=1.1)
            ax.scatter([x + 1.05], [y1], s=35, color="#333333", zorder=4)
            ax.add_patch(plt.Circle((x + 1.05, y2), 0.11, fill=False, ec="#333333", lw=1.2))
            ax.plot([x + 0.98, x + 1.12], [y2 - 0.07, y2 + 0.07], color="#333333", lw=1)
            ax.plot([x + 0.98, x + 1.12], [y2 + 0.07, y2 - 0.07], color="#333333", lw=1)
    ax.text(7.4, 1.5, "X + Z readout", ha="center", va="center", fontsize=10,
            bbox={"boxstyle": "round,pad=0.35", "fc": "#fff1d6", "ec": "#e5a000"})
    ax.annotate("", xy=(8.55, 1.5), xytext=(8.0, 1.5),
                arrowprops={"arrowstyle": "->", "color": "#333333"})
    ax.text(9.0, 1.5, "2 action logits", ha="center", va="center", fontsize=9)
    ax.set_title("Four-qubit variational policy", fontsize=14, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
