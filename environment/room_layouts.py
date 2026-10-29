"""Bản đồ căn phòng cố định và cách đọc bản đồ từ chuỗi.

Quy ước (D-01): mảng numpy shape (H, W), dtype int8, 0 = sàn trống, 1 = vật cản.
Bản đồ cố định viết bằng chuỗi cho dễ đọc: '.' là sàn trống, '#' là vật cản.
"""

from __future__ import annotations

import numpy as np

FREE_CELL: int = 0
OBSTACLE_CELL: int = 1

FIXED_ROOM_LAYOUTS: dict[str, list[str]] = {
    # Lưới 3x3 trống: 9 ô trống. Dùng kiểm chứng Value Iteration và test tabular (chạy nhanh).
    "empty_3x3": [
        "...",
        "...",
        "...",
    ],
    # Lưới 4x4 trống: 16 ô trống. Dùng cho E7 (bùng nổ trạng thái) và Value Iteration cỡ trung bình.
    "empty_4x4": [
        "....",
        "....",
        "....",
        "....",
    ],
    # Lưới 4x4 có 3 vật cản: 13 ô trống. Bản đồ chính cho tabular (E1, E2, E6).
    "obstacle_4x4": [
        "....",
        ".#..",
        "...#",
        "#...",
    ],
    # Lưới 5x5 và 6x6 trống: chỉ dùng ở nhiệm vụ 5.3 (nơi bảng Q bắt đầu quá tải).
    "empty_5x5": ["....."] * 5,
    "empty_6x6": ["......"] * 6,
    # Phòng 6x6 có đồ đạc: 29 ô trống. Bản đồ trung gian cho DQN.
    "room_6x6": [
        "......",
        ".##...",
        ".##.#.",
        "....#.",
        "#.....",
        "......",
    ],
    # Phòng 8x8 có đồ đạc (bàn 2x2, tủ, chân ghế): 52 ô trống. Bản đồ chính cho deep RL.
    "furniture_8x8": [
        "........",
        ".##.....",
        ".##..##.",
        "........",
        "...#....",
        ".#.#..#.",
        ".#....#.",
        "........",
    ],
}

_CELL_VALUE_BY_CHARACTER: dict[str, int] = {".": FREE_CELL, "#": OBSTACLE_CELL}


def parse_room_layout(text_rows: list[str]) -> np.ndarray:
    """Chuyển danh sách chuỗi thành mảng bản đồ.

    Args:
        text_rows: mỗi phần tử là một hàng, ký tự '.' (trống) hoặc '#' (vật cản).

    Returns:
        Mảng (H, W) dtype int8.

    Raises:
        ValueError: nếu rỗng, các hàng khác độ dài, có ký tự lạ, hoặc không có ô trống nào.
    """
    if not text_rows:
        raise ValueError("Bản đồ rỗng: cần ít nhất một hàng.")
    width = len(text_rows[0])
    if width == 0:
        raise ValueError("Bản đồ có hàng rỗng.")
    cell_rows: list[list[int]] = []
    for row_index, text_row in enumerate(text_rows):
        if len(text_row) != width:
            raise ValueError(
                f"Hàng {row_index} dài {len(text_row)} ký tự, khác hàng 0 ({width} ký tự). "
                "Bản đồ phải là hình chữ nhật."
            )
        try:
            cell_rows.append([_CELL_VALUE_BY_CHARACTER[character] for character in text_row])
        except KeyError as unknown_character_error:
            raise ValueError(
                f"Ký tự không hợp lệ {unknown_character_error.args[0]!r} ở hàng {row_index}; "
                "chỉ cho phép '.' (trống) và '#' (vật cản)."
            ) from None
    grid = np.array(cell_rows, dtype=np.int8)
    if not (grid == FREE_CELL).any():
        raise ValueError("Bản đồ không có ô trống nào.")
    return grid


def get_room_layout(layout_name: str) -> np.ndarray:
    """Trả bản sao của một bản đồ cố định theo tên.

    Raises:
        KeyError: nếu tên không tồn tại (thông báo kèm danh sách tên hợp lệ).
    """
    if layout_name not in FIXED_ROOM_LAYOUTS:
        raise KeyError(f"Không có bản đồ '{layout_name}'. Các bản đồ hợp lệ: {sorted(FIXED_ROOM_LAYOUTS)}")
    # parse_room_layout luôn tạo mảng mới nên người gọi sửa thoải mái mà không ảnh hưởng bản gốc.
    return parse_room_layout(FIXED_ROOM_LAYOUTS[layout_name])


def list_free_cells(grid: np.ndarray) -> list[tuple[int, int]]:
    """Danh sách ô trống (hàng, cột) theo thứ tự hàng trước cột sau (row-major).

    Thứ tự này định nghĩa chỉ số bit trong bitmask trạng thái tabular (D-12), không được đổi.
    """
    rows, columns = np.nonzero(grid == FREE_CELL)
    return [(int(row), int(column)) for row, column in zip(rows, columns)]


def room_layout_to_text(grid: np.ndarray) -> str:
    """Chuỗi nhiều dòng biểu diễn bản đồ ('.' trống, '#' vật cản) để in ra màn hình."""
    return "\n".join(
        "".join("#" if cell == OBSTACLE_CELL else "." for cell in grid_row) for grid_row in grid
    )
