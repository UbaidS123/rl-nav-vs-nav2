"""Fast 2D kinematic point-goal navigation env for CPU-parallel RL training.

Pure NumPy, no rendering — designed to run at thousands of steps/sec so PPO can be trained
with many parallel SubprocVecEnv workers on CPU (see hardware constraint in project.md:
no CUDA, so env throughput matters more than network compute).

Obstacles are circles (closed-form ray-circle intersection keeps LiDAR simulation cheap).
Observation/action format is defined once in obs_utils.py and reused by gazebo/policy_node.py
so the trained policy sees an identical input format in training and deployment.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .obs_utils import (
    NUM_LIDAR_BEAMS,
    LIDAR_MAX_RANGE,
    MAX_LINEAR_VEL,
    MAX_ANGULAR_VEL,
    OBS_DIM,
    build_observation,
    wrap_to_pi,
)

WORLD_SIZE = 10.0  # meters, square world [0, WORLD_SIZE] x [0, WORLD_SIZE]
ROBOT_RADIUS = 0.18  # TurtleBot3 Waffle footprint (~0.354m diameter), rounded up for margin
GOAL_RADIUS = 0.25
DT = 0.1  # seconds per simulated step
MAX_STEPS = 500

MIN_OBSTACLES = 4
MAX_OBSTACLES = 12
MIN_OBSTACLE_RADIUS = 0.15
MAX_OBSTACLE_RADIUS = 0.5

LIDAR_NOISE_STD = 0.02  # meters, Gaussian noise on each beam
DROPPED_BEAM_PROB = 0.02  # probability a beam returns max range (no detection)

COLLISION_PENALTY = -100.0
GOAL_REWARD = 100.0
TIME_PENALTY = -0.01
PROGRESS_SCALE = 20.0  # reward per meter of progress toward goal (potential-based shaping)
PROXIMITY_PENALTY_DIST = 0.4  # start penalizing when nearest obstacle surface is within this
PROXIMITY_PENALTY_SCALE = -0.5


class FastNavEnv(gym.Env):
    """Mapless point-goal navigation with obstacle avoidance, randomized per episode.

    Action: normalized [v_norm in [0,1], w_norm in [-1,1]], scaled to real TurtleBot3 Waffle
    velocity bounds in step(). Observation: see obs_utils.build_observation.
    """

    metadata = {"render_modes": []}

    def __init__(self, randomize: bool = True, seed_range: tuple[int, int] = (0, 1_000_000)):
        super().__init__()
        self.randomize = randomize
        self.seed_range = seed_range

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._beam_angles = np.linspace(-np.pi, np.pi, NUM_LIDAR_BEAMS, endpoint=False)
        self._map_rng: np.random.Generator = np.random.default_rng()

        self.robot_pos = np.zeros(2, dtype=np.float64)
        self.robot_theta = 0.0
        self.goal_pos = np.zeros(2, dtype=np.float64)
        self.obstacles = np.zeros((0, 3), dtype=np.float64)  # rows: [x, y, radius]
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.step_count = 0
        self._prev_goal_dist = 0.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if self.randomize and seed is None:
            map_seed = int(self.np_random.integers(*self.seed_range))
            self._map_rng = np.random.default_rng(map_seed)
        else:
            self._map_rng = self.np_random

        self._spawn_obstacles()
        self._spawn_start_and_goal()

        self.prev_action = np.zeros(2, dtype=np.float32)
        self.step_count = 0
        self._prev_goal_dist = float(np.linalg.norm(self.goal_pos - self.robot_pos))

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        v = float(action[0]) * MAX_LINEAR_VEL
        w = float(action[1]) * MAX_ANGULAR_VEL

        self.robot_theta = wrap_to_pi(self.robot_theta + w * DT)
        heading_vec = np.array([np.cos(self.robot_theta), np.sin(self.robot_theta)])
        self.robot_pos = self.robot_pos + heading_vec * v * DT
        self.robot_pos = np.clip(self.robot_pos, 0.0, WORLD_SIZE)

        self.step_count += 1

        collided = self._in_collision(self.robot_pos) or self._hit_wall()
        goal_dist = float(np.linalg.norm(self.goal_pos - self.robot_pos))
        reached_goal = goal_dist < GOAL_RADIUS

        reward = TIME_PENALTY + PROGRESS_SCALE * (self._prev_goal_dist - goal_dist)
        nearest = self._nearest_obstacle_surface_dist()
        if nearest < PROXIMITY_PENALTY_DIST:
            reward += PROXIMITY_PENALTY_SCALE * (PROXIMITY_PENALTY_DIST - nearest)

        terminated = False
        if collided:
            reward += COLLISION_PENALTY
            terminated = True
        elif reached_goal:
            reward += GOAL_REWARD
            terminated = True

        truncated = self.step_count >= MAX_STEPS
        self._prev_goal_dist = goal_dist
        # stored in real units (m/s, rad/s) — build_observation() normalizes it, matching
        # what gazebo/policy_node.py will have available (real cmd_vel, not the raw policy output)
        self.prev_action = np.array([v, w], dtype=np.float32)

        obs = self._get_obs()
        info = {"collided": collided, "reached_goal": reached_goal, "goal_dist": goal_dist}
        return obs, reward, terminated, truncated, info

    # -- world setup -----------------------------------------------------------------

    def _spawn_obstacles(self):
        n = int(self._map_rng.integers(MIN_OBSTACLES, MAX_OBSTACLES + 1))
        xs = self._map_rng.uniform(0.5, WORLD_SIZE - 0.5, size=n)
        ys = self._map_rng.uniform(0.5, WORLD_SIZE - 0.5, size=n)
        rs = self._map_rng.uniform(MIN_OBSTACLE_RADIUS, MAX_OBSTACLE_RADIUS, size=n)
        self.obstacles = np.stack([xs, ys, rs], axis=1)

    def _spawn_start_and_goal(self):
        start = None
        for _ in range(100):
            candidate = self._map_rng.uniform(0.5, WORLD_SIZE - 0.5, size=2)
            if not self._in_collision(candidate):
                start = candidate
                break
        if start is None:
            start = np.array([WORLD_SIZE / 2, WORLD_SIZE / 2])

        goal = None
        for _ in range(100):
            candidate = self._map_rng.uniform(0.5, WORLD_SIZE - 0.5, size=2)
            if not self._in_collision(candidate, radius=GOAL_RADIUS) and \
                    np.linalg.norm(candidate - start) > WORLD_SIZE * 0.3:
                goal = candidate
                break
        if goal is None:
            goal = WORLD_SIZE - start

        self.robot_pos = start
        self.robot_theta = float(self._map_rng.uniform(-np.pi, np.pi))
        self.goal_pos = goal

    # -- collision / geometry ----------------------------------------------------------

    def _in_collision(self, pos: np.ndarray, radius: float = ROBOT_RADIUS) -> bool:
        if len(self.obstacles) == 0:
            return False
        d = np.linalg.norm(self.obstacles[:, :2] - pos, axis=1)
        return bool(np.any(d < (self.obstacles[:, 2] + radius)))

    def _hit_wall(self) -> bool:
        return bool(np.any(self.robot_pos <= 0.0) or np.any(self.robot_pos >= WORLD_SIZE))

    def _nearest_obstacle_surface_dist(self) -> float:
        wall_d = float(min(
            self.robot_pos[0], self.robot_pos[1],
            WORLD_SIZE - self.robot_pos[0], WORLD_SIZE - self.robot_pos[1],
        ))
        if len(self.obstacles) == 0:
            return wall_d
        d = np.linalg.norm(self.obstacles[:, :2] - self.robot_pos, axis=1) - self.obstacles[:, 2]
        return float(min(d.min(), wall_d))

    def _cast_lidar(self) -> np.ndarray:
        """Ray-circle intersection against obstacles + the four bounding walls, vectorized
        across all beams at once (loop only over the small obstacle count)."""
        angles = self._beam_angles + self.robot_theta
        dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (B, 2), unit vectors
        px, py = self.robot_pos

        with np.errstate(divide="ignore", invalid="ignore"):
            t_x0 = np.where(dirs[:, 0] < 0, -px / dirs[:, 0], np.inf)
            t_x1 = np.where(dirs[:, 0] > 0, (WORLD_SIZE - px) / dirs[:, 0], np.inf)
            t_y0 = np.where(dirs[:, 1] < 0, -py / dirs[:, 1], np.inf)
            t_y1 = np.where(dirs[:, 1] > 0, (WORLD_SIZE - py) / dirs[:, 1], np.inf)
        ranges = np.minimum(np.minimum(t_x0, t_x1), np.minimum(t_y0, t_y1))

        for ox, oy, orad in self.obstacles:
            ocx, ocy = ox - px, oy - py
            b = -2 * (dirs[:, 0] * ocx + dirs[:, 1] * ocy)
            c = ocx ** 2 + ocy ** 2 - orad ** 2
            disc = b ** 2 - 4 * c
            hit = disc >= 0
            sqrt_disc = np.sqrt(np.maximum(disc, 0))
            t_near = (-b - sqrt_disc) / 2
            t_hit = np.where(hit & (t_near >= 0), t_near, np.inf)
            ranges = np.minimum(ranges, t_hit)

        ranges = np.clip(ranges, 0.0, LIDAR_MAX_RANGE)

        if self.randomize:
            noise = self._map_rng.normal(0, LIDAR_NOISE_STD, size=NUM_LIDAR_BEAMS)
            ranges = np.clip(ranges + noise, 0.0, LIDAR_MAX_RANGE)
            dropped = self._map_rng.random(NUM_LIDAR_BEAMS) < DROPPED_BEAM_PROB
            ranges[dropped] = LIDAR_MAX_RANGE

        return ranges

    def _get_obs(self) -> np.ndarray:
        lidar = self._cast_lidar()
        return build_observation(
            lidar,
            self.robot_pos[0], self.robot_pos[1], self.robot_theta,
            self.goal_pos[0], self.goal_pos[1],
            self.prev_action,
        )
