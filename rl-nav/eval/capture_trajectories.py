"""Captures a couple of example (policy, Nav2) trajectories on the same episode(s) for the
"example trajectories overlaid on a map" plot (project.md section 6). Run separately from the
main statistical benchmark (eval/benchmark.py) since it only needs 1-2 episodes, not the full
fixed set, and needs full path points rather than just aggregate path length.

Usage (from rl-nav/, inside WSL):
    python3 eval/capture_trajectories.py --episode-ids 0 3
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rclpy

from baseline.nav2_runner import Nav2EpisodeNode, Nav2Stack
from eval.benchmark import RLEpisodeNode, world_to_odom_goal
from eval.gazebo_process import GazeboEpisode


def odom_to_world(px: float, py: float, start: dict) -> tuple[float, float]:
    yaw = start['yaw']
    wx = px * math.cos(yaw) - py * math.sin(yaw)
    wy = px * math.sin(yaw) + py * math.cos(yaw)
    return start['x'] + wx, start['y'] + wy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', default='eval/fixed_episodes.json')
    parser.add_argument('--model', default='results/ppo_fastnav_v0/final_model.zip')
    parser.add_argument('--out', default='results/example_trajectories.json')
    parser.add_argument('--episode-ids', type=int, nargs='+', default=[0])
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    episodes = json.loads((repo_root / args.episodes).read_text())
    by_id = {e['episode_id']: e for e in episodes}
    model_path = str(repo_root / args.model)

    rclpy.init()
    out = []
    try:
        for eid in args.episode_ids:
            ep = by_id[eid]

            print(f"episode {eid}: policy run...")
            with GazeboEpisode(ep['start']['x'], ep['start']['y'], ep['start']['yaw']):
                node = RLEpisodeNode(model_path)
                goal_x, goal_y = world_to_odom_goal(ep['start'], ep['goal'])
                metrics = node.run_episode(goal_x, goal_y)
                world_path = [odom_to_world(px, py, ep['start']) for px, py in node.path_points]
                node.destroy_node()
            out.append({"runner": "policy", "episode_id": eid, "path": world_path,
                        "success": metrics['success']})

            print(f"episode {eid}: nav2 run...")
            with GazeboEpisode(ep['start']['x'], ep['start']['y'], ep['start']['yaw']):
                with Nav2Stack():
                    node = Nav2EpisodeNode()
                    metrics = node.run_episode(ep['start'], ep['goal'])
                    world_path = [odom_to_world(px, py, ep['start']) for px, py in node.path_points]
                    node.destroy_node()
            out.append({"runner": "nav2", "episode_id": eid, "path": world_path,
                        "success": metrics['success']})
    finally:
        rclpy.shutdown()

    out_path = repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")


if __name__ == '__main__':
    main()
