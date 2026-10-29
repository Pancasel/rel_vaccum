"""Baseline tham lam: luôn đi tới ô chưa dọn gần nhất."""

from __future__ import annotations

import numpy as np

from environment.observation_decoding import decode_observation
from environment.shortest_path_search import first_action_towards_nearest_target


class GreedyNearestDirtyCellAgent:
    """Tìm ô chưa dọn GẦN NHẤT bằng BFS rồi đi một bước về phía đó; lặp lại. Tất định.

    Vì sao luôn kết thúc: khoảng cách tới ô chưa dọn gần nhất giảm ít nhất 1 mỗi bước
    cho tới khi bằng 0 (robot bước vào ô đó và dọn nó).
    Không tối ưu: chọn gần nhất trước có thể bỏ sót một góc, phải quay lại sau.
    """

    def act(self, observation: np.ndarray, info: dict | None = None) -> int:
        robot_position, cleaned, obstacles = decode_observation(observation)
        dirty_cells = (obstacles == 0) & ~cleaned
        action = first_action_towards_nearest_target(obstacles, robot_position, dirty_cells)
        return 0 if action is None else action  # None chỉ khi đã dọn hết
