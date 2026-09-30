"""Phase 6: run the same fixed episode set (eval/episodes.py) through both the learned policy
and Nav2 in Gazebo, logging success rate, collision rate, SPL, path length, and time-to-goal
for a head-to-head comparison (project.md section 6).

Each episode gets a fully fresh Gazebo (+ Nav2, for the Nav2 runs) instance — see
eval/gazebo_process.py's docstring for why. Collision is a proxy (minimum LiDAR range ever
seen during the episode, threshold COLLISION_RANGE_M) since neither the fast env nor this
Gazebo model has a true contact/bumper sensor. SPL uses straight-line start-goal distance as
the "shortest path" term (eval/episodes.py's `straight_line_dist`) rather than a true
obstacle-aware geodesic distance — documented simplification, not a true shortest path.

Usage (from rl-nav/, inside WSL):
    python3 eval/benchmark.py --episodes eval/fixed_episodes.json
    python3 eval/benchmark.py --skip-nav2      # policy only
    python3 eval/benchmark.py --skip-policy    # nav2 only
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan

from baseline.nav2_runner import Nav2EpisodeNode, Nav2Stack
from envs.obs_utils import NUM_LIDAR_BEAMS, downsample_lidar
from eval.gazebo_process import GazeboEpisode
from gazebo.policy_runner import PolicyController

COLLISION_RANGE_M = 0.12  # LiDAR's physical range_min - readings below this are true near-contact,
# not just a close pass (empirically, safe successful episodes dip to ~0.15-0.20m routinely)
POLICY_GOAL_RADIUS = 0.25
POLICY_TIMEOUT_S = 60.0  # in SIMULATED seconds (paced off /clock, not wall-clock - see below)
POLICY_CONTROL_DT = 0.1  # matches fast_nav_env.py's DT exactly
WALLCLOCK_SAFETY_TIMEOUT_S = 120.0  # hard real-time cutoff in case /clock stalls


class RLEpisodeNode(Node):
    """Paces the control loop off simulated time (/clock), not wall-clock. Gazebo's real-time
    factor under headless + software rendering measured ~0.76 in this environment (varies with
    system load) — a wall-clock-timed 10Hz loop doesn't actually deliver one action per 0.1s of
    SIMULATED time, which is what the policy was trained on (fast_nav_env.py's DT=0.1). Pacing
    off /clock instead makes the control frequency correct regardless of the actual RTF.
    """

    def __init__(self, model_path: str):
        super().__init__('rl_episode_runner')
        sensor_qos = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(LaserScan, '/scan', self._scan_cb, sensor_qos)
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self.create_subscription(Clock, '/clock', self._clock_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.controller = PolicyController(model_path)

        self.latest_scan: LaserScan | None = None
        self.latest_odom: Odometry | None = None
        self.sim_time: float | None = None
        self.path_points: list[tuple[float, float]] = []

    def _scan_cb(self, msg: LaserScan):
        self.latest_scan = msg

    def _odom_cb(self, msg: Odometry):
        self.latest_odom = msg
        pos = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        self.path_points.append(pos)

    def _clock_cb(self, msg: Clock):
        self.sim_time = msg.clock.sec + msg.clock.nanosec * 1e-9

    def run_episode(self, goal_odom_x: float, goal_odom_y: float) -> dict:
        self.controller.reset()
        self.latest_scan = None
        self.latest_odom = None
        self.sim_time = None
        self.path_points = []
        path_length = 0.0
        prev_pos = None
        min_scan = float('inf')

        wall_deadline = time.time() + 10.0
        while (self.latest_scan is None or self.latest_odom is None or self.sim_time is None) \
                and time.time() < wall_deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.latest_scan is None or self.latest_odom is None or self.sim_time is None:
            return {"success": False, "collided": False, "timed_out": True,
                    "path_length": 0.0, "time_s": 0.0, "reason": "no_sensor_data"}

        sim_start = self.sim_time
        next_control_sim_time = self.sim_time
        wall_deadline = time.time() + WALLCLOCK_SAFETY_TIMEOUT_S
        success = False

        while (self.sim_time - sim_start) < POLICY_TIMEOUT_S and time.time() < wall_deadline:
            rclpy.spin_once(self, timeout_sec=0.02)

            pos = (self.latest_odom.pose.pose.position.x, self.latest_odom.pose.pose.position.y)
            if prev_pos is not None:
                path_length += math.hypot(pos[0] - prev_pos[0], pos[1] - prev_pos[1])
            prev_pos = pos

            finite = [r for r in self.latest_scan.ranges if r == r and r > 0.0]
            if finite:
                min_scan = min(min_scan, min(finite))

            if self.sim_time < next_control_sim_time:
                continue
            next_control_sim_time += POLICY_CONTROL_DT

            robot_x, robot_y = pos
            ori = self.latest_odom.pose.pose.orientation
            robot_theta = 2.0 * math.atan2(ori.z, ori.w)

            goal_dist = math.hypot(goal_odom_x - robot_x, goal_odom_y - robot_y)
            if goal_dist < POLICY_GOAL_RADIUS:
                success = True
                self.cmd_pub.publish(Twist())
                break

            lidar = downsample_lidar(
                np.array(self.latest_scan.ranges, dtype=np.float64), NUM_LIDAR_BEAMS)
            v, w = self.controller.compute_action(
                lidar, robot_x, robot_y, robot_theta, goal_odom_x, goal_odom_y)
            cmd = Twist()
            cmd.linear.x = v
            cmd.angular.z = w
            self.cmd_pub.publish(cmd)

        self.cmd_pub.publish(Twist())
        elapsed = self.sim_time - sim_start
        collided = min_scan < COLLISION_RANGE_M
        timed_out = not success and not collided

        return {
            "success": bool(success and not collided),
            "collided": bool(collided),
            "timed_out": bool(timed_out),
            "path_length": path_length,
            "time_s": elapsed,
        }


def world_to_odom_goal(start: dict, goal: dict) -> tuple[float, float]:
    """Each episode respawns the robot with odom origin = start pose (world frame), heading 0
    aligned with start yaw. Rotate the world-frame goal into that frame."""
    dx = goal['x'] - start['x']
    dy = goal['y'] - start['y']
    yaw = start['yaw']
    cos_y, sin_y = math.cos(-yaw), math.sin(-yaw)
    gx = dx * cos_y - dy * sin_y
    gy = dx * sin_y + dy * cos_y
    return gx, gy


def compute_spl(episode: dict, metrics: dict) -> float:
    if not metrics['success']:
        return 0.0
    shortest = episode['straight_line_dist']
    actual = max(metrics['path_length'], shortest)
    return shortest / actual if actual > 0 else 0.0


def run_policy_batch(episodes: list[dict], model_path: str) -> list[dict]:
    print(f"\n=== Running {len(episodes)} episodes: RL POLICY ===")
    out = []
    for ep in episodes:
        print(f"  episode {ep['episode_id']}...", end=' ', flush=True)
        with GazeboEpisode(ep['start']['x'], ep['start']['y'], ep['start']['yaw']):
            node = RLEpisodeNode(model_path)
            goal_x, goal_y = world_to_odom_goal(ep['start'], ep['goal'])
            metrics = node.run_episode(goal_x, goal_y)
            node.destroy_node()
        metrics['episode_id'] = ep['episode_id']
        metrics['spl'] = compute_spl(ep, metrics)
        out.append(metrics)
        print(metrics)
    return out


def run_nav2_batch(episodes: list[dict]) -> list[dict]:
    print(f"\n=== Running {len(episodes)} episodes: NAV2 ===")
    out = []
    for ep in episodes:
        print(f"  episode {ep['episode_id']}...", end=' ', flush=True)
        with GazeboEpisode(ep['start']['x'], ep['start']['y'], ep['start']['yaw']):
            with Nav2Stack():
                node = Nav2EpisodeNode()
                metrics = node.run_episode(ep['start'], ep['goal'])
                node.destroy_node()
        metrics['episode_id'] = ep['episode_id']
        metrics['spl'] = compute_spl(ep, metrics)
        out.append(metrics)
        print(metrics)
    return out


def summarize(runs: list[dict]) -> dict:
    n = len(runs)
    return {
        'n_episodes': n,
        'success_rate': sum(r['success'] for r in runs) / n,
        'collision_rate': sum(r['collided'] for r in runs) / n,
        'timeout_rate': sum(r['timed_out'] for r in runs) / n,
        'avg_spl': sum(r['spl'] for r in runs) / n,
        'avg_path_length': sum(r['path_length'] for r in runs) / n,
        'avg_time_s': sum(r['time_s'] for r in runs) / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', default='eval/fixed_episodes.json')
    parser.add_argument('--model', default='results/ppo_fastnav_v0/final_model.zip')
    parser.add_argument('--out', default='results/benchmark.json')
    parser.add_argument('--skip-nav2', action='store_true')
    parser.add_argument('--skip-policy', action='store_true')
    parser.add_argument('--limit', type=int, default=None, help='only run first N episodes')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    ep_path = Path(args.episodes)
    if not ep_path.is_absolute():
        ep_path = repo_root / args.episodes
    episodes = json.loads(ep_path.read_text())
    if args.limit:
        episodes = episodes[:args.limit]

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = repo_root / args.model

    rclpy.init()
    results = {"policy": [], "nav2": []}
    try:
        if not args.skip_policy:
            results['policy'] = run_policy_batch(episodes, str(model_path))
        if not args.skip_nav2:
            results['nav2'] = run_nav2_batch(episodes)
    finally:
        rclpy.shutdown()

    summary = {k: summarize(v) for k, v in results.items() if v}

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({'episodes': results, 'summary': summary}, indent=2))

    print(f"\nSaved to {out_path}")
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
