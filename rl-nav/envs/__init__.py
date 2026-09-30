from gymnasium.envs.registration import register

from .fast_nav_env import FastNavEnv

register(
    id="FastNav-v0",
    entry_point="envs.fast_nav_env:FastNavEnv",
)

__all__ = ["FastNavEnv"]
