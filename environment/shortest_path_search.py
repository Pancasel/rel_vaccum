"""Tìm đường ngắn nhất trên lưới bằng BFS (Breadth-First Search: tìm kiếm theo chiều rộng)."""

from __future__ import annotations

from collections import deque

import numpy as np

from environment.grid_movement import MOVE_OFFSETS_BY_ACTION, is_free_cell_inside_grid
from environment.room_layouts import FREE_CELL


def distances_from_cell(grid: np.ndarray, source_cell: tuple[int, int]) -> np.ndarray:
    """Khoảng cách ngắn nhất (số bước 4 hướng, đi vòng vật cản) từ source_cell tới mọi ô.

    Returns:
        Mảng (H, W) int32; -1 với vật cản hoặc ô không tới được.

    Raises:
        ValueError: nếu source_cell không phải ô trống.
    """
    if not is_free_cell_inside_grid(grid, *source_cell):
        raise ValueError(f"Ô nguồn {source_cell} không phải ô trống trong lưới.")
    distance_map = np.full(grid.shape, -1, dtype=np.int32)
    distance_map[source_cell] = 0
    cells_to_expand: deque[tuple[int, int]] = deque([source_cell])
    while cells_to_expand:
        row, column = cells_to_expand.popleft()
        for row_offset, column_offset in MOVE_OFFSETS_BY_ACTION:
            next_row, next_column = row + row_offset, column + column_offset
            if is_free_cell_inside_grid(grid, next_row, next_column) and distance_map[next_row, next_column] < 0:
                distance_map[next_row, next_column] = distance_map[row, column] + 1
                cells_to_expand.append((next_row, next_column))
    return distance_map


def distance_to_nearest_dirty_cell(grid: np.ndarray, robot_position: tuple[int, int],
                                   cleaned: np.ndarray) -> int:
    """Khoảng cách BFS từ robot tới ô trống chưa dọn ("dirty") gần nhất; 0 nếu đã dọn hết.

    Dùng cho shaping thế năng (D-14) và reward "hackable" (D-15).
    """
    dirty_cells = (grid == FREE_CELL) & ~cleaned
    if not dirty_cells.any():
        return 0
    if dirty_cells[robot_position]:
        return 0
    already_seen = np.zeros(grid.shape, dtype=bool)
    already_seen[robot_position] = True
    # Mỗi phần tử: (hàng, cột, khoảng cách từ robot tới ô đó).
    cells_to_expand: deque[tuple[int, int, int]] = deque([(robot_position[0], robot_position[1], 0)])
    while cells_to_expand:
        row, column, distance = cells_to_expand.popleft()
        for row_offset, column_offset in MOVE_OFFSETS_BY_ACTION:
            next_row, next_column = row + row_offset, column + column_offset
            if is_free_cell_inside_grid(grid, next_row, next_column) and not already_seen[next_row, next_column]:
                if dirty_cells[next_row, next_column]:
                    return distance + 1
                already_seen[next_row, next_column] = True
                cells_to_expand.append((next_row, next_column, distance + 1))
    # Chỉ xảy ra nếu bản đồ không liên thông (VacuumCleaningEnv đã chặn trường hợp này).
    raise ValueError("Không tới được ô chưa dọn nào: bản đồ không liên thông.")


def first_action_towards_nearest_target(grid: np.ndarray, robot_position: tuple[int, int],
                                        target_cells: np.ndarray) -> int | None:
    """Hành động đầu tiên trên một đường ngắn nhất từ robot tới ô mục tiêu gần nhất.

    Bản thân ô robot đang đứng không được tính là mục tiêu. Phá hòa (D-20): BFS mở rộng
    hàng xóm theo thứ tự hành động 0,1,2,3 nên kết quả luôn tất định.

    Args:
        grid: bản đồ.
        robot_position: vị trí hiện tại.
        target_cells: mảng bool (H, W), True ở các ô mục tiêu.

    Returns:
        Chỉ số hành động 0..3, hoặc None nếu không tới được mục tiêu nào.
    """
    already_seen = np.zeros(grid.shape, dtype=bool)
    already_seen[robot_position] = True
    # Mỗi phần tử: (hàng, cột, hành động đầu tiên đã dùng để đi tới ô đó).
    cells_to_expand: deque[tuple[int, int, int]] = deque()
    for action, (row_offset, column_offset) in enumerate(MOVE_OFFSETS_BY_ACTION):
        next_row = robot_position[0] + row_offset
        next_column = robot_position[1] + column_offset
        if is_free_cell_inside_grid(grid, next_row, next_column) and not already_seen[next_row, next_column]:
            if target_cells[next_row, next_column]:
                return action
            already_seen[next_row, next_column] = True
            cells_to_expand.append((next_row, next_column, action))
    while cells_to_expand:
        row, column, first_action = cells_to_expand.popleft()
        for row_offset, column_offset in MOVE_OFFSETS_BY_ACTION:
            next_row, next_column = row + row_offset, column + column_offset
            if is_free_cell_inside_grid(grid, next_row, next_column) and not already_seen[next_row, next_column]:
                if target_cells[next_row, next_column]:
                    return first_action
                already_seen[next_row, next_column] = True
                cells_to_expand.append((next_row, next_column, first_action))
    return None
