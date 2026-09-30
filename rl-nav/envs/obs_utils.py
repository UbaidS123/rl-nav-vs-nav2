"""Shared observation/action construction — the single source of truth used by BOTH
fast_nav_env.py (training) and gazebo/policy_node.py (deployment). Any change here changes
what the policy sees in both places, which is exactly the point: it's the only way to
guarantee the fast-env-to-Gazebo transfer is a fair sim-to-sim comparison.
"""
from __future__ import annotations

import numpy as np

NUM_LIDAR_BEAMS = 24
# TurtleBot3 Waffle ships the LDS-01 (max ~3.5m); some newer units carry the LDS-02 (~8m).
# Verify against the actual driver/URDF in Phase 1 and adjust here if it differs.
LIDAR_MAX_RANGE = 3.5  # meters

GOAL_MAX_DIST = 10.0  # meters — must match the fast env world size for normalization to make sense

# TurtleBot3 Waffle velocity limits. NOTE: project.md's "roughly v in [0, 0.22], w in
# [-2.84, 2.84]" are actually the Burger's limits, not the Waffle's. Waffle's real limits
# (from ROBOTIS specs) are used here — reverify against the URDF/param files once TurtleBot3
# packages are installed in Phase 1.
MAX_LINEAR_VEL = 0.26  # m/s
MAX_ANGULAR_VEL = 1.82  # rad/s

OBS_DIM = NUM_LIDAR_BEAMS + 2 + 2  # lidar beams + goal(dist, heading) + prev_action(v, w)


def wrap_to_pi(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap an angle (radians) into [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def normalize_lidar(ranges: np.ndarray, max_range: float = LIDAR_MAX_RANGE) -> np.ndarray:
    """Clip raw ranges to [0, max_range] and scale to [0, 1]. NaN/inf (no return / dropped
    beam) is treated as max range, matching how out-of-range LiDAR returns behave."""
    ranges = np.nan_to_num(ranges, nan=max_range, posinf=max_range, neginf=max_range)
    return np.clip(ranges, 0.0, max_range) / max_range


def downsample_lidar(full_ranges: np.ndarray, num_beams: int = NUM_LIDAR_BEAMS) -> np.ndarray:
    """Downsample a dense scan (e.g. 360 beams from a real/sim driver) to num_beams by taking
    the minimum range in each angular bin — conservative, so a narrow obstacle is never
    averaged away."""
    full_ranges = np.asarray(full_ranges, dtype=np.float64)
    n = len(full_ranges)
    edges = np.linspace(0, n, num_beams + 1).astype(int)
    out = np.empty(num_beams, dtype=np.float64)
    for i in range(num_beams):
        lo, hi = edges[i], edges[i + 1]
        out[i] = np.min(full_ranges[lo:hi]) if hi > lo else full_ranges[min(lo, n - 1)]
    return out


def goal_obs_in_robot_frame(
    robot_x: float, robot_y: float, robot_theta: float,
    goal_x: float, goal_y: float, max_dist: float = GOAL_MAX_DIST,
) -> tuple[float, float]:
    """Relative goal distance (normalized to [0, 1], clipped) and heading (normalized to
    [-1, 1] via /pi), both expressed in the robot's own frame."""
    dx, dy = goal_x - robot_x, goal_y - robot_y
    dist = float(np.hypot(dx, dy))
    heading_world = float(np.arctan2(dy, dx))
    heading_rel = wrap_to_pi(heading_world - robot_theta)
    return min(dist, max_dist) / max_dist, heading_rel / np.pi


def normalize_action(v: float, w: float) -> np.ndarray:
    return np.array([v / MAX_LINEAR_VEL, w / MAX_ANGULAR_VEL], dtype=np.float32)


def denormalize_action(action: np.ndarray) -> tuple[float, float]:
    """Inverse of the policy's [-1,1]/[0,1]-normalized action -> real (v, w) in m/s, rad/s."""
    v = float(action[0]) * MAX_LINEAR_VEL
    w = float(action[1]) * MAX_ANGULAR_VEL
    return v, w


def build_observation(
    lidar_ranges: np.ndarray,
    robot_x: float, robot_y: float, robot_theta: float,
    goal_x: float, goal_y: float,
    prev_action: np.ndarray,
    *,
    max_lidar_range: float = LIDAR_MAX_RANGE,
    max_goal_dist: float = GOAL_MAX_DIST,
) -> np.ndarray:
    """Build the exact observation vector the policy is trained on / deployed with:
    [normalized lidar (NUM_LIDAR_BEAMS,), goal_dist, goal_heading, prev_v, prev_w].
    """
    lidar_obs = normalize_lidar(lidar_ranges, max_lidar_range).astype(np.float32)
    goal_dist, goal_heading = goal_obs_in_robot_frame(
        robot_x, robot_y, robot_theta, goal_x, goal_y, max_goal_dist
    )
    prev_v, prev_w = float(prev_action[0]), float(prev_action[1])
    return np.concatenate([
        lidar_obs,
        np.array([goal_dist, goal_heading], dtype=np.float32),
        normalize_action(prev_v, prev_w),
    ])
