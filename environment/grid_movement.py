"""Quy ước hướng đi trên lưới và kiểm tra ô có đi vào được không."""

from __future__ import annotations

import numpy as np

from environment.room_layouts import FREE_CELL

# Phải trùng thứ tự hành động D-02: 0 = lên, 1 = xuống, 2 = trái, 3 = phải.
# Mỗi phần tử là (độ lệch hàng, độ lệch cột) khi thực hiện hành động đó.
MOVE_OFFSETS_BY_ACTION: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


def is_free_cell_inside_grid(grid: np.ndarray, row: int, column: int) -> bool:
    """True nếu (row, column) nằm trong lưới và là ô trống."""
    inside_grid = 0 <= row < grid.shape[0] and 0 <= column < grid.shape[1]
    return inside_grid and grid[row, column] == FREE_CELL


def safe_actions_at_position(grid: np.ndarray, robot_position: tuple[int, int]) -> list[int]:
    """Các hành động không đụng tường/vật cản khi robot đứng ở robot_position."""
    robot_row, robot_column = robot_position
    safe_actions: list[int] = []
    for action, (row_offset, column_offset) in enumerate(MOVE_OFFSETS_BY_ACTION):
        if is_free_cell_inside_grid(grid, robot_row + row_offset, robot_column + column_offset):
            safe_actions.append(action)
    return safe_actions
