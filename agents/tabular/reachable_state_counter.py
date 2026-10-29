"""Đếm số trạng thái đạt tới được của MDP (thí nghiệm E7: bùng nổ trạng thái)."""

from __future__ import annotations

import time

import numpy as np

from environment.grid_movement import MOVE_OFFSETS_BY_ACTION
from environment.room_layouts import FREE_CELL, list_free_cells

ACTION_COUNT: int = 4
# Giới hạn kích thước mảng "đã thăm" khi đếm trạng thái bằng numpy (1 byte mỗi trạng thái).
MAX_VISITED_ARRAY_SIZE: int = 2**28


def count_reachable_states(grid: np.ndarray, time_limit_seconds: float = 600.0) -> int | None:
    """Đếm số trạng thái đạt tới được (mọi ô xuất phát), dùng cho E7.

    Cài bằng BFS theo tầng trên mảng numpy (mã trạng thái = vị trí * 2^|F| + bitmask) để đếm
    được tới hàng triệu trạng thái. Chỉ đếm, không tính reward; động lực học giống hệt
    compute_transition() (đi vào ô trống, đụng tường thì đứng yên) — test so khớp với
    build_state_transition_graph.

    Returns:
        Số trạng thái, hoặc None nếu quá lớn (mảng > MAX_VISITED_ARRAY_SIZE) hoặc quá thời gian.
    """
    free_cells = list_free_cells(grid)
    free_cell_count = len(free_cells)
    if free_cell_count * (1 << free_cell_count) > MAX_VISITED_ARRAY_SIZE:
        return None
    free_cell_index = {cell: index for index, cell in enumerate(free_cells)}
    # next_cell_index[i, a] = chỉ số ô trống sau khi đi hướng a từ ô i (đụng tường -> giữ nguyên i).
    next_cell_index = np.zeros((free_cell_count, ACTION_COUNT), dtype=np.int64)
    for cell_index, (row, column) in enumerate(free_cells):
        for action, (row_offset, column_offset) in enumerate(MOVE_OFFSETS_BY_ACTION):
            next_row, next_column = row + row_offset, column + column_offset
            is_free_and_inside = (0 <= next_row < grid.shape[0] and 0 <= next_column < grid.shape[1]
                                  and grid[next_row, next_column] == FREE_CELL)
            next_cell_index[cell_index, action] = (
                free_cell_index[(next_row, next_column)] if is_free_and_inside else cell_index
            )
    all_cleaned_bitmask = (1 << free_cell_count) - 1
    is_state_visited = np.zeros(free_cell_count << free_cell_count, dtype=bool)
    start_cell_indices = np.arange(free_cell_count, dtype=np.int64)
    # Mã trạng thái: (chỉ số ô robot) ghép với (bitmask ô đã dọn) vào một số nguyên.
    current_layer = (start_cell_indices << free_cell_count) | (np.int64(1) << start_cell_indices)
    is_state_visited[current_layer] = True
    visited_state_count = len(current_layer)
    deadline = time.monotonic() + time_limit_seconds
    while len(current_layer):
        if time.monotonic() > deadline:
            return None
        bitmasks = current_layer & all_cleaned_bitmask
        current_layer = current_layer[bitmasks != all_cleaned_bitmask]  # trạng thái terminal không mở rộng tiếp
        robot_cell_indices = current_layer >> free_cell_count
        bitmasks = current_layer & all_cleaned_bitmask
        next_robot_cell_indices = next_cell_index[robot_cell_indices].reshape(-1)
        next_bitmasks = np.repeat(bitmasks, ACTION_COUNT) | (np.int64(1) << next_robot_cell_indices)
        next_state_codes = np.unique((next_robot_cell_indices << free_cell_count) | next_bitmasks)
        next_state_codes = next_state_codes[~is_state_visited[next_state_codes]]
        is_state_visited[next_state_codes] = True
        visited_state_count += len(next_state_codes)
        current_layer = next_state_codes
    return int(visited_state_count)
