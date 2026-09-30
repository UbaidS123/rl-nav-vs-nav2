# Learned Navigation vs Nav2

Training a reinforcement learning policy for mapless point-goal navigation with obstacle
avoidance, then benchmarking it against the ROS2 Nav2 stack on the same simulated robot and
world.

## Idea

Nav2 is the standard classical navigation stack for ROS2 robots: a global/local planner,
costmaps, and a controller, all hand-tuned. This project trains a small neural network policy
with reinforcement learning to do the same job — take LiDAR + goal direction as input, output
linear/angular velocity — and compares the two head to head in Gazebo on a TurtleBot3.

Training RL directly inside a real-time physics simulator is far too slow (Gazebo runs close
to real-time; RL needs millions of environment steps). So training happens in a lightweight
custom 2D kinematic environment that runs at thousands of steps/second on CPU, and the trained
policy is then transferred into Gazebo for evaluation and comparison against Nav2. The
observation and action format are shared between the two environments through one module, so
the policy sees an identical input format whether it's training or deployed.

## Stack

- ROS2 Humble + Gazebo Fortress (via `ros_gz`)
- TurtleBot3 Waffle
- Stable-Baselines3 (PPO) + Gymnasium
- Nav2 as the baseline

## Repo layout

```
rl-nav/
  envs/            fast 2D training environment + shared observation/action utilities
  training/        PPO training script, configs, sanity checks
  gazebo/          ROS2 node that runs the trained policy on the robot
  baseline/        Nav2 episode runner
  eval/            episode generation, benchmark harness, plots
  results/         trained models, learning curves, metrics
```

## Status

- Training environment built and validated (random-agent sanity checks, reward shaping).
- PPO trained to convergence: ~97% success rate / ~3% collision rate on training maps.
- Generalization check on held-out maps the policy never trained on: ~95% success rate,
  effectively no generalization gap.
- Policy successfully transferred to Gazebo — drives the real simulated robot via LiDAR +
  odometry, reaching goals in both open and obstacle-filled worlds.
- Head-to-head benchmark harness against Nav2 (matched start/goal episodes, success rate,
  collision rate, SPL, path length) is built; full comparison numbers are still being
  collected on a denser evaluation world than the training distribution, which has already
  surfaced an interesting result: the same episodes that are easy for the classical planner
  can be genuinely harder for the learned policy, since its training world's obstacles were
  sparser than the fixed benchmark world's layout — a real sim-to-sim generalization gap
  rather than a training bug.

## Running it

Training (CPU, no GPU required):

```
cd rl-nav
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
python training/sanity_check.py
python training/train_ppo.py --config training/configs/ppo_default.yaml
python eval/generalization.py
```

The Gazebo/ROS2 side needs ROS2 Humble + Gazebo Fortress + a source build of
`turtlebot3_simulations` targeting modern Gazebo (the stock TurtleBot3 packages only support
Gazebo Classic). From a sourced ROS2 workspace:

```
python3 gazebo/policy_node.py --ros-args -p model_path:=results/ppo_fastnav_v0/final_model.zip \
    -p goal_x:=1.5 -p goal_y:=1.0 -p use_sim_time:=true
```
