"""Phase 3: train PPO on FastNavEnv with SB3 SubprocVecEnv (many parallel CPU envs — see
project.md hardware constraint: no CUDA, env stepping is the bottleneck, not network compute).

Usage (from rl-nav/):
    python training/train_ppo.py --config training/configs/ppo_default.yaml
    python training/train_ppo.py --total-timesteps 20000 --n-envs 4   # quick smoke test
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv

from envs.fast_nav_env import FastNavEnv
from training.callbacks import NavMetricsCallback


def make_env(rank: int, seed_range: tuple, randomize: bool, seed: int):
    def _init():
        env = FastNavEnv(randomize=randomize, seed_range=tuple(seed_range))
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/configs/ppo_default.yaml")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    n_envs = args.n_envs or cfg["env"]["n_envs"]
    total_timesteps = args.total_timesteps or cfg["training"]["total_timesteps"]
    seed = cfg["training"]["seed"]
    set_random_seed(seed)

    run_dir = repo_root / "results" / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    vec_env = SubprocVecEnv([
        make_env(i, cfg["env"]["seed_range"], cfg["env"]["randomize"], seed)
        for i in range(n_envs)
    ])

    ppo_cfg = cfg["ppo"]
    model = PPO(
        ppo_cfg["policy"],
        vec_env,
        policy_kwargs=ppo_cfg["policy_kwargs"],
        n_steps=ppo_cfg["n_steps"],
        batch_size=ppo_cfg["batch_size"],
        n_epochs=ppo_cfg["n_epochs"],
        gamma=ppo_cfg["gamma"],
        gae_lambda=ppo_cfg["gae_lambda"],
        clip_range=ppo_cfg["clip_range"],
        ent_coef=ppo_cfg["ent_coef"],
        learning_rate=ppo_cfg["learning_rate"],
        max_grad_norm=ppo_cfg["max_grad_norm"],
        vf_coef=ppo_cfg["vf_coef"],
        seed=seed,
        verbose=1,
        tensorboard_log=str(run_dir / "tb"),
    )

    metrics_cb = NavMetricsCallback(log_every=cfg["training"]["eval_freq"])
    ckpt_cb = CheckpointCallback(
        save_freq=max(cfg["training"]["checkpoint_freq"] // n_envs, 1),
        save_path=str(run_dir / "checkpoints"),
        name_prefix="ppo_fastnav",
    )

    model.learn(total_timesteps=total_timesteps, callback=CallbackList([metrics_cb, ckpt_cb]))

    model.save(str(run_dir / "final_model"))

    with open(run_dir / "learning_curve.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timesteps", "success_rate", "collision_rate"])
        writer.writeheader()
        writer.writerows(metrics_cb.history)

    print(f"Done. Model: {run_dir / 'final_model'}.zip  Learning curve: {run_dir / 'learning_curve.csv'}")


if __name__ == "__main__":
    main()
