"""Baseline đi ngẫu nhiên."""

from __future__ import annotations

import numpy as np

from environment.observation_decoding import safe_actions_from_observation
from environment.vacuum_cleaning_env import ACTION_COUNT


class RandomWalkAgent:
    """Chọn hành động ngẫu nhiên đều.

    Với only_safe_actions=True chỉ chọn trong các hướng không đụng tường — đây là mốc "không có
    chiến lược gì" công bằng hơn so với đi ngẫu nhiên hoàn toàn (vốn đụng tường liên tục).
    """

    def __init__(self, seed: int = 0, only_safe_actions: bool = True) -> None:
        self.random_generator = np.random.default_rng(seed)
        self.only_safe_actions = only_safe_actions

    def act(self, observation: np.ndarray, info: dict | None = None) -> int:
        if self.only_safe_actions:
            safe_actions = safe_actions_from_observation(observation)
            if safe_actions:
                return int(self.random_generator.choice(safe_actions))
        return int(self.random_generator.integers(ACTION_COUNT))

    def reset(self) -> None:
        """Không đặt lại bộ sinh số ngẫu nhiên: các episode liên tiếp phải khác nhau."""
