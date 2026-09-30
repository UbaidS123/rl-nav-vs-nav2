"""Manages a Gazebo + TurtleBot3 process group for one benchmark episode.

Each episode gets a fully fresh Gazebo server + spawned robot at the specified start pose —
NOT a teleport of an existing entity. This was a deliberate choice after finding that both
`/world/default/set_pose` (teleport) and remove+respawn leave the diff-drive plugin's internal
odometry state stale or break sensor plugins entirely (see [[rl-nav-phase6-benchmark]] in
memory). A full relaunch is slower (~10-15s) but guarantees odom starts at (0,0,0) exactly at
the commanded spawn pose, which both the RL policy (raw-odom-frame) and Nav2 (needs a correct
TF tree) depend on for correctness.

Only Gazebo + robot_state_publisher + the entity + the ros_gz bridge are restarted per
episode; a Nav2 stack, if used, is launched once and kept running across episodes (see
baseline/nav2_runner.py) — only its AMCL initial pose needs resetting per episode.
"""
from __future__ import annotations

import os
import signal
import subprocess
import time

TB3_SHARE = "/root/ros2_ws/install/turtlebot3_gazebo/share/turtlebot3_gazebo"
WORLD_PATH = f"{TB3_SHARE}/worlds/turtlebot3_world.world"
MODEL_SDF = f"{TB3_SHARE}/models/turtlebot3_waffle/model.sdf"
BRIDGE_PARAMS = f"{TB3_SHARE}/params/turtlebot3_waffle_bridge.yaml"


def _gz_env() -> dict:
    env = os.environ.copy()
    env["TURTLEBOT3_MODEL"] = "waffle"
    env["GZ_SIM_RESOURCE_PATH"] = f"{TB3_SHARE}/models"
    env["LIBGL_ALWAYS_SOFTWARE"] = "1"
    return env


def _popen(args: list[str], env: dict) -> subprocess.Popen:
    return subprocess.Popen(
        args, env=env, preexec_fn=os.setsid,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _kill(proc: subprocess.Popen | None):
    if proc is None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=5)
    except Exception:
        pass


class GazeboEpisode:
    """Context manager: fresh gzserver + spawned robot for the duration of one episode."""

    def __init__(self, x: float, y: float, yaw: float, settle_time: float = 8.0):
        self.x, self.y, self.yaw = x, y, yaw
        self.settle_time = settle_time
        self._procs: list[subprocess.Popen] = []

    def __enter__(self):
        env = _gz_env()

        gz_proc = _popen(
            ["ign", "gazebo", "-r", "-s", "-v1", "--headless-rendering", WORLD_PATH], env)
        self._procs.append(gz_proc)
        time.sleep(4.0)  # world load

        rsp_proc = _popen(
            ["ros2", "launch", "turtlebot3_gazebo", "robot_state_publisher.launch.py",
             "use_sim_time:=true"], env)
        self._procs.append(rsp_proc)

        bridge_proc = _popen(
            ["ros2", "run", "ros_gz_bridge", "parameter_bridge",
             "--ros-args", "-p", f"config_file:={BRIDGE_PARAMS}"], env)
        self._procs.append(bridge_proc)

        subprocess.run(
            ["ros2", "run", "ros_gz_sim", "create",
             "-name", "waffle", "-file", MODEL_SDF,
             "-x", str(self.x), "-y", str(self.y), "-z", "0.01", "-Y", str(self.yaw)],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15,
        )
        time.sleep(self.settle_time)
        return self

    def __exit__(self, exc_type, exc, tb):
        for p in reversed(self._procs):
            _kill(p)
        time.sleep(1.5)
        return False
