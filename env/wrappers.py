"""Wrapper observation cho các thí nghiệm."""

from __future__ import annotations

import gymnasium
import numpy as np


class HideCleanedChannel(gymnasium.ObservationWrapper):
    """Xóa kênh 1 (ô đã dọn) khỏi observation — bản deep của thí nghiệm E1.

    Agent chỉ còn thấy vị trí của mình và vật cản, không biết mình đã dọn tới đâu,
    nên trạng thái mất tính Markov (Sutton & Barto 2018, mục 3.1 và 17.3).
    """

    def __init__(self, env: gymnasium.Env) -> None:
        super().__init__(env)
        if env.observation_space.shape is None or env.observation_space.shape[0] != 3:
            raise ValueError("HideCleanedChannel cần observation shape (3, H, W).")

    def observation(self, observation: np.ndarray) -> np.ndarray:
        hidden = np.array(observation, dtype=np.float32, copy=True)
        hidden[1] = 0.0
        return hidden
