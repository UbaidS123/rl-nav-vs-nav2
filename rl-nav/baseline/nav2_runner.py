"""Phase 6: drives Nav2 through the same fixed (start, goal) episodes used for the RL policy
(see eval/episodes.py, eval/benchmark.py), so the two can be compared head-to-head.

Nav2 (and Gazebo) are fully relaunched per episode — slower than reusing one long-lived stack,
but avoids an entire class of "is Nav2/AMCL in a weird state left over from the last episode"
bugs. See eval/gazebo_process.py's docstring for why the Gazebo side in particular needs a
fresh spawn (not a teleport) each episode.
"""
from __future__ import annotations

import math
import os
import signal
import subprocess
import time

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan

NAV2_MAP_YAML = "/opt/ros/humble/share/turtlebot3_navigation2/map/map.yaml"
NAV2_PARAMS = "/opt/ros/humble/share/turtlebot3_navigation2/param/humble/waffle.yaml"

COLLISION_RANGE_M = 0.12  # LiDAR's physical range_min - see eval/benchmark.py for why not 0.15
GOAL_TIMEOUT_S = 60.0
NAV2_SETTLE_S = 12.0  # time to let AMCL/costmaps come up after relaunch
INITIALPOSE_SETTLE_S = 3.0


def yaw_to_quat(yaw: float) -> tuple[float, float]:
    return 0.0, math.sin(yaw / 2.0)  # (used as z,w for a planar rotation)


class Nav2Stack:
    """Context manager: fully relaunches the Nav2 bringup stack for one episode."""

    def __enter__(self):
        env = os.environ.copy()
        self.proc = subprocess.Popen(
            ["ros2", "launch", "turtlebot3_navigation2", "navigation2.launch.py",
             "use_sim_time:=true", f"map:={NAV2_MAP_YAML}", f"params_file:={NAV2_PARAMS}"],
            env=env, preexec_fn=os.setsid,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(NAV2_SETTLE_S)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            pass
        time.sleep(1.0)
        return False


class Nav2EpisodeNode(Node):
    def __init__(self):
        super().__init__('nav2_episode_runner')
        sensor_qos = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.initialpose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', 10)
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self.create_subscription(LaserScan, '/scan', self._scan_cb, sensor_qos)
        self.action_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')

        self.latest_odom_pos = None
        self.min_scan_range = float('inf')
        self.path_points: list[tuple[float, float]] = []

    def _odom_cb(self, msg: Odometry):
        self.latest_odom_pos = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        self.path_points.append(self.latest_odom_pos)

    def _scan_cb(self, msg: LaserScan):
        finite = [r for r in msg.ranges if r == r and r > 0.0]  # filter NaN
        if finite:
            self.min_scan_range = min(self.min_scan_range, min(finite))

    def set_initial_pose(self, x: float, y: float, yaw: float):
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        z, w = yaw_to_quat(yaw)
        msg.pose.pose.orientation.z = z
        msg.pose.pose.orientation.w = w
        msg.pose.covariance[0] = 0.25
        msg.pose.covariance[7] = 0.25
        msg.pose.covariance[35] = 0.06
        self.initialpose_pub.publish(msg)

    def run_episode(self, start: dict, goal: dict) -> dict:
        self.min_scan_range = float('inf')
        self.path_points = []
        path_length = 0.0
        prev_pos = None

        self.set_initial_pose(start['x'], start['y'], start['yaw'])
        deadline = time.time() + INITIALPOSE_SETTLE_S
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)

        if not self.action_client.wait_for_server(timeout_sec=10.0):
            return {"success": False, "collided": False, "timed_out": True,
                    "path_length": 0.0, "time_s": 0.0, "reason": "action_server_unavailable"}

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.pose.position.x = goal['x']
        goal_msg.pose.pose.position.y = goal['y']
        goal_msg.pose.pose.orientation.w = 1.0

        send_future = self.action_client.send_goal_async(goal_msg)
        start_time = time.time()
        while not send_future.done() and time.time() - start_time < 10.0:
            rclpy.spin_once(self, timeout_sec=0.1)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            return {"success": False, "collided": False, "timed_out": True,
                    "path_length": 0.0, "time_s": 0.0, "reason": "goal_rejected"}

        result_future = goal_handle.get_result_async()
        episode_start = time.time()
        while not result_future.done() and time.time() - episode_start < GOAL_TIMEOUT_S:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.latest_odom_pos is not None:
                if prev_pos is not None:
                    dx = self.latest_odom_pos[0] - prev_pos[0]
                    dy = self.latest_odom_pos[1] - prev_pos[1]
                    path_length += math.hypot(dx, dy)
                prev_pos = self.latest_odom_pos

        elapsed = time.time() - episode_start
        timed_out = not result_future.done()
        if timed_out:
            goal_handle.cancel_goal_async()
            success = False
        else:
            status = result_future.result().status
            success = status == GoalStatus.STATUS_SUCCEEDED

        collided = self.min_scan_range < COLLISION_RANGE_M

        return {
            "success": bool(success) and not collided,
            "collided": bool(collided),
            "timed_out": bool(timed_out),
            "path_length": path_length,
            "time_s": elapsed,
        }
