"""Phase 6/7: generate the results plots project.md asks for — learning curve, policy-vs-Nav2
success-rate comparison, and example trajectories overlaid on the map.

Runs on the Windows side (no ROS dependency, just matplotlib/numpy over already-saved
results/*.json and results/*/learning_curve.csv).

Usage (from rl-nav/):
    python eval/plots.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

POLICY_COLOR = "#2a78d6"  # validated categorical slot 1 (blue)
NAV2_COLOR = "#eb6834"    # validated categorical slot 2 (orange)
INK = "#333333"
MUTED = "#767676"
GRID = "#e3e3e3"


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED)


def plot_learning_curve(csv_path: Path, out_path: Path):
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    timesteps = data["timesteps"] / 1e6
    success_rate = data["success_rate"] * 100

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(timesteps, success_rate, color=POLICY_COLOR, linewidth=2)
    ax.set_xlabel("Training steps (millions)", color=INK)
    ax.set_ylabel("Success rate (%)", color=INK)
    ax.set_title("PPO learning curve (fast env, training maps)", color=INK, fontsize=13)
    ax.set_ylim(0, 100)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_success_bar(benchmark_json: Path, out_path: Path):
    with open(benchmark_json) as f:
        data = json.load(f)
    summary = data["summary"]

    metrics = [
        ("success_rate", "Success rate"),
        ("collision_rate", "Collision rate"),
        ("avg_spl", "Avg SPL"),
    ]
    labels = ["policy", "nav2"]
    display_labels = {"policy": "Learned Policy", "nav2": "Nav2"}
    colors = {"policy": POLICY_COLOR, "nav2": NAV2_COLOR}

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, key in enumerate(labels):
        if key not in summary:
            continue
        vals = [summary[key][m] * 100 if m != "avg_spl" else summary[key][m] * 100
                for m, _ in metrics]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=display_labels[key],
                       color=colors[key], zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=9, color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics], color=INK)
    ax.set_ylabel("% (SPL shown ×100)", color=INK)
    ax.set_ylim(0, 110)
    ax.set_title(f"Learned policy vs Nav2 ({summary.get('policy', summary.get('nav2', {})).get('n_episodes', '?')} episodes, turtlebot3_world)",
                 color=INK, fontsize=13)
    ax.legend(frameon=False, labelcolor=INK)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_trajectories(trajectories_json: Path, map_yaml: Path, out_path: Path):
    from eval.episodes import load_map

    with open(trajectories_json) as f:
        trajs = json.load(f)

    map_info = load_map(map_yaml)
    free = map_info["free_mask"]
    res = map_info["resolution"]
    ox, oy = map_info["origin_x"], map_info["origin_y"]
    h = map_info["height"]
    extent = [ox, ox + free.shape[1] * res, oy, oy + h * res]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(np.where(free, 1.0, 0.3), cmap="gray", origin="lower", extent=extent, zorder=0)

    for run in trajs:
        color = POLICY_COLOR if run["runner"] == "policy" else NAV2_COLOR
        xs = [p[0] for p in run["path"]]
        ys = [p[1] for p in run["path"]]
        ax.plot(xs, ys, color=color, linewidth=2, label=run["runner"], zorder=3)
        ax.scatter([xs[0]], [ys[0]], color=color, marker="o", s=60, zorder=4,
                   edgecolors="white", linewidths=1)
        ax.scatter([xs[-1]], [ys[-1]], color=color, marker="X", s=80, zorder=4,
                   edgecolors="white", linewidths=1)

    handles, unique_labels = [], []
    for h_, l_ in zip(*ax.get_legend_handles_labels()):
        if l_ not in unique_labels:
            handles.append(h_)
            unique_labels.append(l_)
    display = {"policy": "Learned Policy", "nav2": "Nav2"}
    ax.legend(handles, [display.get(l, l) for l in unique_labels], frameon=False, labelcolor=INK)
    ax.set_title("Example trajectories (o = start, x = goal)", color=INK, fontsize=13)
    ax.set_xlabel("x (m)", color=INK)
    ax.set_ylabel("y (m)", color=INK)
    ax.tick_params(colors=MUTED)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    results_dir = repo_root / args.results_dir

    lc_path = results_dir / "ppo_fastnav_v0" / "learning_curve.csv"
    if lc_path.exists():
        plot_learning_curve(lc_path, results_dir / "learning_curve.png")
    else:
        print(f"skip: {lc_path} not found")

    bench_path = results_dir / "benchmark.json"
    if bench_path.exists():
        plot_success_bar(bench_path, results_dir / "success_comparison.png")
    else:
        print(f"skip: {bench_path} not found")

    traj_path = results_dir / "example_trajectories.json"
    map_yaml = Path("/opt/ros/humble/share/turtlebot3_navigation2/map/map.yaml")
    if traj_path.exists() and map_yaml.exists():
        plot_trajectories(traj_path, map_yaml, results_dir / "trajectories.png")
    else:
        print(f"skip: trajectory plot ({traj_path} / {map_yaml} not both found)")


if __name__ == "__main__":
    main()
