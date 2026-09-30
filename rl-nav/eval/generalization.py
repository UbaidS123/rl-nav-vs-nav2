"""Phase 4: measure the generalization gap — success rate on training-map seeds vs a
held-out seed range the policy never saw during training (see training/configs/ppo_default.yaml:
training draws from seed_range [0, 800000], held-out uses [800000, 1_000_000]).

Usage (from rl-nav/):
    python eval/generalization.py --model results/ppo_fastnav_v0/final_model.zip
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from stable_baselines3 import PPO

from envs.fast_nav_env import FastNavEnv

N_EPISODES = 300
TRAIN_SEED_RANGE = (0, 800_000)
HELDOUT_SEED_RANGE = (800_000, 1_000_000)


def evaluate(model: PPO, seed_range: tuple, n_episodes: int) -> dict:
    env = FastNavEnv(randomize=True)
    # deterministic, evenly-spaced seeds spanning the range so repeat runs are comparable
    seeds = np.linspace(seed_range[0], seed_range[1] - 1, n_episodes).astype(int)

    successes = collisions = timeouts = 0
    path_lengths = []
    steps_to_goal = []

    for seed in seeds:
        obs, info = env.reset(seed=int(seed))
        terminated = truncated = False
        path_length = 0.0
        prev_pos = env.robot_pos.copy()
        n_steps = 0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            path_length += float(np.linalg.norm(env.robot_pos - prev_pos))
            prev_pos = env.robot_pos.copy()
            n_steps += 1

        if info["reached_goal"]:
            successes += 1
            path_lengths.append(path_length)
            steps_to_goal.append(n_steps)
        elif info["collided"]:
            collisions += 1
        else:
            timeouts += 1

    return {
        "n_episodes": n_episodes,
        "success_rate": successes / n_episodes,
        "collision_rate": collisions / n_episodes,
        "timeout_rate": timeouts / n_episodes,
        "avg_path_length": float(np.mean(path_lengths)) if path_lengths else None,
        "avg_steps_to_goal": float(np.mean(steps_to_goal)) if steps_to_goal else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="results/ppo_fastnav_v0/final_model.zip")
    parser.add_argument("--n-episodes", type=int, default=N_EPISODES)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = repo_root / args.model

    model = PPO.load(str(model_path))

    print(f"Evaluating {model_path} on {args.n_episodes} episodes per split...\n")

    train_results = evaluate(model, TRAIN_SEED_RANGE, args.n_episodes)
    heldout_results = evaluate(model, HELDOUT_SEED_RANGE, args.n_episodes)

    gap = train_results["success_rate"] - heldout_results["success_rate"]

    print("Training maps:")
    for k, v in train_results.items():
        print(f"  {k}: {v}")
    print("\nHeld-out maps:")
    for k, v in heldout_results.items():
        print(f"  {k}: {v}")
    print(f"\nGeneralization gap (train success - held-out success): {gap:.1%}")

    out = {
        "train": train_results,
        "held_out": heldout_results,
        "generalization_gap": gap,
    }
    out_path = model_path.parent / "generalization.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
