#!/usr/bin/env python3
"""Phase 5: ROS2 node that runs the fast-env-trained PPO policy on the real (Gazebo) robot.

Subscribes /scan + /odom, builds the exact same observation format used in training (via
envs/obs_utils.py — the single source of truth shared with fast_nav_env.py), and publishes
/cmd_vel. This is the sim-to-sim transfer point: if this file and fast_nav_env.py ever disagree
on beam count, normalization, or goal-frame convention, the policy will misbehave here even
though it worked fine in training.

The actual control step (obs -> policy -> action) lives in policy_runner.PolicyController,
shared with eval/benchmark.py so the live demo and the benchmark harness can never drift apart.

Run inside WSL (needs rclpy + a sourced ROS2/Gazebo environment):
    python3 policy_node.py --ros-args -p model_path:=/path/to/final_model.zip \
        -p goal_x:=1.5 -p goal_y:=1.0 -p use_sim_time:=true

`use_sim_time:=true` matters: the control timer below is paced in real 0.1s ticks by
default, but Gazebo's real-time factor isn't guaranteed to be 1.0 (measured ~0.76 under
headless + forced software rendering in this environment — see eval/benchmark.py's
RLEpisodeNode for the full explanation). Setting use_sim_time makes rclpy's clock (and
therefore this node's timer) follow /clock instead of the wall clock, so each control tick
really is one simulated DT=0.1s step, matching what the policy was trained on.

A new goal can also be sent at runtime via the /goal_pose topic (geometry_msgs/PoseStamped),
matching rviz2's "2D Goal Pose" tool.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # rl-nav root -> envs.obs_utils

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan

from envs.obs_utils import NUM_LIDAR_BEAMS, downsample_lidar
from gazebo.policy_runner import PolicyController

CONTROL_HZ = 10.0  # matches DT=0.1 in fast_nav_env.py
GOAL_RADIUS = 0.25  # matches fast_nav_env.py


def yaw_from_quaternion(q) -> float:
    """Planar-motion shortcut: for a ground robot the quaternion is a pure yaw rotation."""
    return 2.0 * float(np.arctan2(q.z, q.w))


class PolicyNode(Node):
    def __init__(self):
        super().__init__('policy_node')

        self.declare_parameter('model_path', '')
        self.declare_parameter('goal_x', 1.5)
        self.declare_parameter('goal_y', 1.0)

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        if not model_path:
            raise RuntimeError('model_path parameter is required, e.g. '
                                '-p model_path:=/path/to/final_model.zip')
        self.get_logger().info(f'Loading policy from {model_path}')
        self.controller = PolicyController(model_path)

        self.goal = np.array([
            self.get_parameter('goal_x').get_parameter_value().double_value,
            self.get_parameter('goal_y').get_parameter_value().double_value,
        ])
        self.get_logger().info(f'Initial goal: {self.goal}')

        sensor_qos = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(LaserScan, '/scan', self._scan_cb, sensor_qos)
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self._goal_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.latest_scan: LaserScan | None = None
        self.latest_odom: Odometry | None = None
        self.goal_reached_logged = False

        self.timer = self.create_timer(1.0 / CONTROL_HZ, self._control_step)

    def _scan_cb(self, msg: LaserScan):
        self.latest_scan = msg

    def _odom_cb(self, msg: Odometry):
        self.latest_odom = msg

    def _goal_cb(self, msg: PoseStamped):
        self.goal = np.array([msg.pose.position.x, msg.pose.position.y])
        self.goal_reached_logged = False
        self.controller.reset()
        self.get_logger().info(f'New goal: {self.goal}')

    def _control_step(self):
        if self.latest_scan is None or self.latest_odom is None:
            return

        scan_ranges = np.array(self.latest_scan.ranges, dtype=np.float64)
        lidar = downsample_lidar(scan_ranges, NUM_LIDAR_BEAMS)

        pos = self.latest_odom.pose.pose.position
        ori = self.latest_odom.pose.pose.orientation
        robot_x, robot_y = pos.x, pos.y
        robot_theta = yaw_from_quaternion(ori)

        goal_dist = float(np.hypot(self.goal[0] - robot_x, self.goal[1] - robot_y))
        if goal_dist < GOAL_RADIUS:
            if not self.goal_reached_logged:
                self.get_logger().info(f'Goal reached (dist={goal_dist:.2f}m)')
                self.goal_reached_logged = True
            self.cmd_pub.publish(Twist())
            return

        v, w = self.controller.compute_action(
            lidar, robot_x, robot_y, robot_theta, self.goal[0], self.goal[1])

        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        self.cmd_pub.publish(cmd)


def main():
    rclpy.init()
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
