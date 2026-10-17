"""Baseline tham lam: luôn đi tới ô chưa dọn gần nhất."""

from __future__ import annotations

import numpy as np

from env.grid_utils import bfs_first_action_towards, decode_obs


class GreedyBFSAgent:
    """Tìm ô chưa dọn GẦN NHẤT bằng BFS rồi đi một bước về phía đó; lặp lại. Tất định.

    Vì sao luôn kết thúc: khoảng cách tới ô chưa dọn gần nhất giảm ít nhất 1 mỗi bước
    cho tới khi bằng 0 (robot bước vào ô đó và dọn nó).
    Không tối ưu: chọn gần nhất trước có thể bỏ sót một góc, phải quay lại sau.
    """

    def act(self, obs: np.ndarray, info: dict | None = None) -> int:
        pos, cleaned, obstacles = decode_obs(obs)
        targets = (obstacles == 0) & ~cleaned
        action = bfs_first_action_towards(obstacles, pos, targets)
        return 0 if action is None else action  # None chỉ khi đã dọn hết
