"""Baseline quét zigzag (boustrophedon) — cách máy cắt cỏ / robot rẻ tiền hay dùng."""

from __future__ import annotations

import numpy as np

from env.grid_utils import bfs_first_action_towards, decode_obs
from env.maps import FREE


def snake_order(obstacles: np.ndarray) -> list[tuple[int, int]]:
    """Thứ tự rắn của các ô trống: hàng chẵn trái -> phải, hàng lẻ phải -> trái."""
    height, width = obstacles.shape
    order: list[tuple[int, int]] = []
    for r in range(height):
        cols = range(width) if r % 2 == 0 else range(width - 1, -1, -1)
        order.extend((r, c) for c in cols if obstacles[r, c] == FREE)
    return order


class ZigzagAgent:
    """Quét theo thứ tự rắn; gặp vật cản thì đi vòng bằng đường BFS ngắn nhất (D-19).

    Mỗi bước: mục tiêu = ô CHƯA DỌN đầu tiên theo thứ tự rắn; đi một bước theo đường ngắn
    nhất tới mục tiêu. Agent không có trạng thái nội bộ (tính lại từ observation).

    Vì sao luôn kết thúc: mục tiêu chỉ đổi khi nó đã được dọn; khoảng cách BFS tới mục tiêu
    giảm đúng 1 mỗi bước; số mục tiêu hữu hạn và bản đồ liên thông.
    """

    def __init__(self) -> None:
        self._order_cache: dict[bytes, list[tuple[int, int]]] = {}

    def act(self, obs: np.ndarray, info: dict | None = None) -> int:
        pos, cleaned, obstacles = decode_obs(obs)
        key = obstacles.tobytes()
        if key not in self._order_cache:
            self._order_cache[key] = snake_order(obstacles)
        target = next((cell for cell in self._order_cache[key] if not cleaned[cell]), None)
        if target is None:
            return 0  # đã dọn hết (evaluate không gọi act ở trạng thái này)
        targets = np.zeros(obstacles.shape, dtype=bool)
        targets[target] = True
        action = bfs_first_action_towards(obstacles, pos, targets)
        return 0 if action is None else action
