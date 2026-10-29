"""Baseline quét zigzag (boustrophedon) — cách máy cắt cỏ / robot rẻ tiền hay dùng."""

from __future__ import annotations

import numpy as np

from environment.observation_decoding import decode_observation
from environment.room_layouts import FREE_CELL
from environment.shortest_path_search import first_action_towards_nearest_target


def snake_scan_order(obstacles: np.ndarray) -> list[tuple[int, int]]:
    """Thứ tự rắn của các ô trống: hàng chẵn trái -> phải, hàng lẻ phải -> trái."""
    height, width = obstacles.shape
    scan_order: list[tuple[int, int]] = []
    for row in range(height):
        columns = range(width) if row % 2 == 0 else range(width - 1, -1, -1)
        scan_order.extend((row, column) for column in columns if obstacles[row, column] == FREE_CELL)
    return scan_order


class ZigzagSweepAgent:
    """Quét theo thứ tự rắn; gặp vật cản thì đi vòng bằng đường BFS ngắn nhất (D-19).

    Mỗi bước: mục tiêu = ô CHƯA DỌN đầu tiên theo thứ tự rắn; đi một bước theo đường ngắn
    nhất tới mục tiêu. Agent không có trạng thái nội bộ (tính lại từ observation).

    Vì sao luôn kết thúc: mục tiêu chỉ đổi khi nó đã được dọn; khoảng cách BFS tới mục tiêu
    giảm đúng 1 mỗi bước; số mục tiêu hữu hạn và bản đồ liên thông.
    """

    def __init__(self) -> None:
        # Lưu thứ tự quét theo từng bản đồ (khóa = nội dung bản đồ dạng bytes) để khỏi tính lại.
        self._scan_order_by_room_bytes: dict[bytes, list[tuple[int, int]]] = {}

    def act(self, observation: np.ndarray, info: dict | None = None) -> int:
        robot_position, cleaned, obstacles = decode_observation(observation)
        room_bytes = obstacles.tobytes()
        if room_bytes not in self._scan_order_by_room_bytes:
            self._scan_order_by_room_bytes[room_bytes] = snake_scan_order(obstacles)
        next_target_cell = next(
            (cell for cell in self._scan_order_by_room_bytes[room_bytes] if not cleaned[cell]), None
        )
        if next_target_cell is None:
            return 0  # đã dọn hết (bộ đánh giá không gọi act ở trạng thái này)
        target_cells = np.zeros(obstacles.shape, dtype=bool)
        target_cells[next_target_cell] = True
        action = first_action_towards_nearest_target(obstacles, robot_position, target_cells)
        return 0 if action is None else action
