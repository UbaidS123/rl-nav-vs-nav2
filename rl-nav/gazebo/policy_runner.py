"""Pure control-logic wrapper around the trained PPO policy — no ROS dependency itself, so it
can be driven identically by the live demo node (policy_node.py) and the episodic benchmark
harness (eval/benchmark.py). Keeps the "single source of truth" promise from obs_utils.py
extended to the control loop itself, not just observation construction.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from stable_baselines3 import PPO

from envs.obs_utils import GOAL_MAX_DIST, LIDAR_MAX_RANGE, build_observation, denormalize_action


class PolicyController:
    def __init__(self, model_path: str):
        self.model = PPO.load(model_path)
        self.prev_action = np.zeros(2, dtype=np.float32)

    def reset(self):
        self.prev_action = np.zeros(2, dtype=np.float32)

    def compute_action(
        self,
        lidar_ranges: np.ndarray,
        robot_x: float, robot_y: float, robot_theta: float,
        goal_x: float, goal_y: float,
    ) -> tuple[float, float]:
        """Returns (v, w) in real units (m/s, rad/s)."""
        obs = build_observation(
            lidar_ranges, robot_x, robot_y, robot_theta, goal_x, goal_y, self.prev_action,
            max_lidar_range=LIDAR_MAX_RANGE, max_goal_dist=GOAL_MAX_DIST,
        )
        action, _ = self.model.predict(obs, deterministic=True)
        v, w = denormalize_action(action)
        self.prev_action = np.array([v, w], dtype=np.float32)
        return v, w
