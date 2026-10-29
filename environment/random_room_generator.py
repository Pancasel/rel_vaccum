"""Sinh căn phòng ngẫu nhiên luôn dọn được hết (các ô trống liên thông với nhau)."""

from __future__ import annotations

from collections import deque

import numpy as np

from environment.grid_movement import MOVE_OFFSETS_BY_ACTION
from environment.room_layouts import FREE_CELL, OBSTACLE_CELL, list_free_cells


def is_fully_connected(grid: np.ndarray) -> bool:
    """True nếu mọi ô trống liên thông với nhau theo 4 hướng.

    Vì sao cần: nếu có ô trống bị vật cản bao kín, robot không bao giờ dọn được 100%,
    episode không thể kết thúc bằng `terminated` và bài toán trở nên vô nghĩa.
    """
    free_cells = list_free_cells(grid)
    if not free_cells:
        return False
    height, width = grid.shape
    already_seen = np.zeros(grid.shape, dtype=bool)
    cells_to_expand: deque[tuple[int, int]] = deque([free_cells[0]])
    already_seen[free_cells[0]] = True
    reached_cell_count = 1
    while cells_to_expand:
        row, column = cells_to_expand.popleft()
        for row_offset, column_offset in MOVE_OFFSETS_BY_ACTION:
            next_row, next_column = row + row_offset, column + column_offset
            inside_grid = 0 <= next_row < height and 0 <= next_column < width
            if inside_grid and not already_seen[next_row, next_column] and grid[next_row, next_column] == FREE_CELL:
                already_seen[next_row, next_column] = True
                reached_cell_count += 1
                cells_to_expand.append((next_row, next_column))
    return reached_cell_count == len(free_cells)


def generate_random_room(height: int, width: int, obstacle_density: float,
                         random_generator: np.random.Generator, max_tries: int = 1000) -> np.ndarray:
    """Sinh bản đồ ngẫu nhiên có đúng round(density*H*W) vật cản và luôn liên thông.

    Thuật toán: duyệt các ô theo một hoán vị ngẫu nhiên, đặt thử vật cản và chỉ giữ lại
    nếu các ô trống vẫn liên thông. Cách này đảm bảo bản đồ luôn dọn được hết.

    Args:
        height, width: kích thước lưới (>= 1, tổng ô >= 2).
        obstacle_density: tỷ lệ ô là vật cản, trong [0, 1).
        random_generator: bộ sinh số ngẫu nhiên numpy (để tái lập theo seed).
        max_tries: số lần thử lại với hoán vị mới trước khi báo lỗi.

    Raises:
        ValueError: tham số không hợp lệ.
        RuntimeError: không sinh được bản đồ liên thông sau max_tries lần.
    """
    if height < 1 or width < 1:
        raise ValueError(f"Kích thước lưới phải dương, nhận được {height}x{width}.")
    if not 0.0 <= obstacle_density < 1.0:
        raise ValueError(f"obstacle_density phải trong [0, 1), nhận được {obstacle_density}.")
    total_cell_count = height * width
    obstacle_count = int(round(obstacle_density * total_cell_count))
    if obstacle_count > total_cell_count - 2:
        raise ValueError(
            f"Quá nhiều vật cản ({obstacle_count}) cho lưới {height}x{width}: cần còn ít nhất 2 ô trống."
        )
    for _ in range(max_tries):
        grid = np.zeros((height, width), dtype=np.int8)
        placed_obstacle_count = 0
        for flat_cell_index in random_generator.permutation(total_cell_count):
            if placed_obstacle_count == obstacle_count:
                break
            row, column = divmod(int(flat_cell_index), width)
            grid[row, column] = OBSTACLE_CELL
            if is_fully_connected(grid):
                placed_obstacle_count += 1
            else:
                grid[row, column] = FREE_CELL
        if placed_obstacle_count == obstacle_count:
            return grid
    raise RuntimeError(
        f"Không sinh được bản đồ liên thông {height}x{width} với {obstacle_count} vật cản sau {max_tries} lần thử."
    )


def generate_distinct_rooms(height: int, width: int, obstacle_density: float, room_count: int, seed: int,
                            excluded_rooms: list[np.ndarray] | None = None,
                            max_attempts_factor: int = 100) -> list[np.ndarray]:
    """Sinh room_count bản đồ đôi một khác nhau từ một seed (dùng làm tập train/test cho E3, E4).

    Args:
        excluded_rooms: các bản đồ không được trùng (ví dụ tập test khi sinh tập train).
        max_attempts_factor: giới hạn số lần sinh = room_count * hệ số này.

    Raises:
        ValueError: room_count < 1.
        RuntimeError: không đủ bản đồ khác nhau (ví dụ mật độ 0 chỉ có đúng 1 bản đồ).
    """
    if room_count < 1:
        raise ValueError(f"n_maps phải >= 1, nhận được {room_count}.")
    random_generator = np.random.default_rng(seed)
    seen_room_bytes: set[bytes] = {room.astype(np.int8).tobytes() for room in (excluded_rooms or [])}
    rooms: list[np.ndarray] = []
    for _ in range(room_count * max_attempts_factor):
        room = generate_random_room(height, width, obstacle_density, random_generator)
        room_bytes = room.tobytes()
        if room_bytes not in seen_room_bytes:
            seen_room_bytes.add(room_bytes)
            rooms.append(room)
            if len(rooms) == room_count:
                return rooms
    raise RuntimeError(
        f"Chỉ sinh được {len(rooms)}/{room_count} bản đồ khác nhau ({height}x{width}, mật độ {obstacle_density})."
    )
