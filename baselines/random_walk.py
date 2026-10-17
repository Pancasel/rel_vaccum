"""Baseline đi ngẫu nhiên."""

from __future__ import annotations

import numpy as np

from env.grid_utils import valid_actions


class RandomWalkAgent:
    """Chọn hành động ngẫu nhiên đều.

    Với valid_only=True chỉ chọn trong các hướng không đụng tường — đây là mốc "không có
    chiến lược gì" công bằng hơn so với đi ngẫu nhiên hoàn toàn (vốn đụng tường liên tục).
    """

    def __init__(self, seed: int = 0, valid_only: bool = True) -> None:
        self.rng = np.random.default_rng(seed)
        self.valid_only = valid_only

    def act(self, obs: np.ndarray, info: dict | None = None) -> int:
        if self.valid_only:
            choices = valid_actions(obs)
            if choices:
                return int(self.rng.choice(choices))
        return int(self.rng.integers(4))

    def reset(self) -> None:
        """Không đặt lại rng: các episode liên tiếp phải khác nhau."""
