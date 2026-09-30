"""Generates a fixed set of (start, goal) episodes from the Nav2 occupancy map, so the exact
same episodes can be run through both the learned policy and Nav2 for a fair head-to-head
(project.md section 6: "Run the same set of episodes ... through both").

Free-space is sampled from map.pgm (thresholded per map.yaml's occupied_thresh/free_thresh,
matching the standard ROS map_server convention), with occupied cells dilated by the robot's
footprint radius so sampled points aren't flush against a wall.

Usage (from rl-nav/, inside WSL where the map lives):
    python3 eval/episodes.py --n-episodes 20 --out eval/fixed_episodes.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROBOT_RADIUS_M = 0.22  # TurtleBot3 Waffle footprint radius, with a little margin
MIN_START_GOAL_DIST_M = 2.0


def load_map(map_yaml_path: Path):
    with open(map_yaml_path) as f:
        meta = yaml.safe_load(f)
    img_path = map_yaml_path.parent / meta["image"]
    img = np.array(Image.open(img_path))
    resolution = meta["resolution"]
    origin_x, origin_y, _ = meta["origin"]
    occupied_thresh = meta["occupied_thresh"]
    free_thresh = meta["free_thresh"]
    negate = meta.get("negate", 0)

    occ_prob = (255 - img) / 255.0 if not negate else img / 255.0
    free_mask = occ_prob < free_thresh
    occupied_mask = occ_prob > occupied_thresh

    return {
        "free_mask": free_mask,
        "occupied_mask": occupied_mask,
        "resolution": resolution,
        "origin_x": origin_x,
        "origin_y": origin_y,
        "height": img.shape[0],
    }


def dilate_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    """Simple max-pool style dilation (no scipy dependency)."""
    if radius_px <= 0:
        return mask
    out = mask.copy()
    h, w = mask.shape
    ys, xs = np.where(mask)
    for dy in range(-radius_px, radius_px + 1):
        for dx in range(-radius_px, radius_px + 1):
            if dx * dx + dy * dy > radius_px * radius_px:
                continue
            ny, nx = ys + dy, xs + dx
            valid = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
            out[ny[valid], nx[valid]] = True
    return out


def pixel_to_world(px: int, py: int, map_info: dict) -> tuple[float, float]:
    # PGM row 0 is the top of the image, which corresponds to the MAX y in map frame
    # (map_server convention: image is flipped vertically relative to world y).
    x = map_info["origin_x"] + (px + 0.5) * map_info["resolution"]
    y = map_info["origin_y"] + (map_info["height"] - py - 0.5) * map_info["resolution"]
    return x, y


def generate_episodes(map_info: dict, n_episodes: int, seed: int = 0) -> list[dict]:
    radius_px = int(np.ceil(ROBOT_RADIUS_M / map_info["resolution"]))
    blocked = dilate_mask(map_info["occupied_mask"], radius_px)
    valid_mask = map_info["free_mask"] & ~blocked
    valid_ys, valid_xs = np.where(valid_mask)
    if len(valid_xs) == 0:
        raise RuntimeError("No valid free-space pixels found in map after dilation")

    rng = np.random.default_rng(seed)
    episodes = []
    attempts = 0
    while len(episodes) < n_episodes and attempts < n_episodes * 200:
        attempts += 1
        i_start = rng.integers(0, len(valid_xs))
        i_goal = rng.integers(0, len(valid_xs))
        start = pixel_to_world(int(valid_xs[i_start]), int(valid_ys[i_start]), map_info)
        goal = pixel_to_world(int(valid_xs[i_goal]), int(valid_ys[i_goal]), map_info)
        dist = float(np.hypot(goal[0] - start[0], goal[1] - start[1]))
        if dist < MIN_START_GOAL_DIST_M:
            continue
        yaw = float(rng.uniform(-np.pi, np.pi))
        episodes.append({
            "episode_id": len(episodes),
            "start": {"x": start[0], "y": start[1], "yaw": yaw},
            "goal": {"x": goal[0], "y": goal[1]},
            "straight_line_dist": dist,
        })

    if len(episodes) < n_episodes:
        raise RuntimeError(
            f"Only generated {len(episodes)}/{n_episodes} episodes after {attempts} attempts "
            "- try lowering MIN_START_GOAL_DIST_M or check the map")
    return episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-yaml",
                         default="/opt/ros/humble/share/turtlebot3_navigation2/map/map.yaml")
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="eval/fixed_episodes.json")
    args = parser.parse_args()

    map_info = load_map(Path(args.map_yaml))
    episodes = generate_episodes(map_info, args.n_episodes, args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(episodes, f, indent=2)
    print(f"Wrote {len(episodes)} episodes to {out_path}")


if __name__ == "__main__":
    main()
