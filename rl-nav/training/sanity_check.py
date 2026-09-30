"""Phase 2 gate: random agent runs, reward signs make sense, obs matches the intended format.

Run from rl-nav/: python training/sanity_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from envs.fast_nav_env import FastNavEnv
from envs.obs_utils import OBS_DIM

N_EPISODES = 200


def main():
    env = FastNavEnv(randomize=True, seed_range=(0, 1_000_000))
    obs, info = env.reset(seed=0)

    assert obs.shape == (OBS_DIM,), f"obs shape {obs.shape} != expected ({OBS_DIM},)"
    assert env.observation_space.contains(obs), "initial obs outside declared observation_space"
    print(f"obs shape OK: {obs.shape}")

    successes = collisions = timeouts = 0
    episode_rewards = []
    goal_reward_seen = collision_penalty_seen = progress_reward_seen = False

    for ep in range(N_EPISODES):
        obs, info = env.reset()
        assert env.observation_space.contains(obs), "reset obs outside declared observation_space"
        total_reward = 0.0
        terminated = truncated = False

        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert env.observation_space.contains(obs), "step obs outside declared observation_space"
            assert np.isfinite(reward), "non-finite reward"
            total_reward += reward

            if reward > 50:
                goal_reward_seen = True
            if reward < -50:
                collision_penalty_seen = True
            if abs(reward) > 0.001 and abs(reward) < 50:
                progress_reward_seen = True

        episode_rewards.append(total_reward)
        if info["reached_goal"]:
            successes += 1
        elif info["collided"]:
            collisions += 1
        else:
            timeouts += 1

    print(f"\n{N_EPISODES} episodes with a RANDOM agent (expect near-zero success — sanity only):")
    print(f"  success rate:   {successes / N_EPISODES:.1%}")
    print(f"  collision rate: {collisions / N_EPISODES:.1%}")
    print(f"  timeout rate:   {timeouts / N_EPISODES:.1%}")
    print(f"  reward: mean={np.mean(episode_rewards):.2f} min={np.min(episode_rewards):.2f} "
          f"max={np.max(episode_rewards):.2f}")

    print("\nReward sign checks:")
    print(f"  saw large positive (goal) reward:   {goal_reward_seen}")
    print(f"  saw large negative (collision) rew: {collision_penalty_seen}")
    print(f"  saw small dense progress reward:    {progress_reward_seen}")

    assert collision_penalty_seen, "never observed a collision in 200 random episodes — suspicious"
    assert progress_reward_seen, "never observed dense progress shaping reward — suspicious"

    print("\nAll sanity checks passed.")


if __name__ == "__main__":
    main()
