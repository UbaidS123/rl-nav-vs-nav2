"""Custom SB3 callback that tracks navigation-specific metrics (success/collision rate)
that PPO's default logger doesn't know about — these come from FastNavEnv's info dict.
"""
from __future__ import annotations

from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class NavMetricsCallback(BaseCallback):
    def __init__(self, log_every: int = 50_000, buffer_size: int = 200, verbose: int = 0):
        super().__init__(verbose)
        self.log_every = log_every
        self._success_buf: deque = deque(maxlen=buffer_size)
        self._collision_buf: deque = deque(maxlen=buffer_size)
        self.history: list[dict] = []
        self._last_log = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info, done in zip(infos, dones):
            if done and "episode" in info:
                self._success_buf.append(bool(info.get("reached_goal", False)))
                self._collision_buf.append(bool(info.get("collided", False)))

        if self.num_timesteps - self._last_log >= self.log_every and self._success_buf:
            success_rate = float(np.mean(self._success_buf))
            collision_rate = float(np.mean(self._collision_buf))
            self.history.append({
                "timesteps": self.num_timesteps,
                "success_rate": success_rate,
                "collision_rate": collision_rate,
            })
            self.logger.record("nav/success_rate", success_rate)
            self.logger.record("nav/collision_rate", collision_rate)
            self._last_log = self.num_timesteps

        return True
