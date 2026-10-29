"""Vẽ trạng thái môi trường thành ảnh RGB hoặc chuỗi ký tự, và thêm chú thích lên ảnh (D-26: dùng Pillow).

Phần hiển thị chỉ để xem, không ảnh hưởng tới việc học.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from environment.room_layouts import OBSTACLE_CELL

CELL_PIXEL_SIZE: int = 32
COLORS: dict[str, tuple[int, int, int]] = {
    "free": (235, 235, 235),
    "cleaned": (160, 210, 255),
    "obstacle": (60, 60, 60),
    "robot": (220, 50, 50),
    "grid_line": (200, 200, 200),
    "trail": (90, 140, 220),
    "band": (255, 255, 255),
    "text": (30, 30, 30),
}
ROBOT_MARGIN_FRACTION: float = 0.2  # robot chiếm 60% ô, lề 20% mỗi bên
_FONT_CACHE_BY_SIZE: dict[int, ImageFont.ImageFont] = {}


def _load_font(font_size: int) -> ImageFont.ImageFont:
    """Font DejaVuSans đi kèm matplotlib (hỗ trợ tiếng Việt có dấu); dự phòng font mặc định."""
    if font_size not in _FONT_CACHE_BY_SIZE:
        try:
            import matplotlib

            font_path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
            _FONT_CACHE_BY_SIZE[font_size] = ImageFont.truetype(str(font_path), font_size)
        except (ImportError, OSError):
            _FONT_CACHE_BY_SIZE[font_size] = ImageFont.load_default()
    return _FONT_CACHE_BY_SIZE[font_size]


def _cell_center_in_pixels(row: int, column: int, cell_pixel_size: int) -> tuple[float, float]:
    """Tọa độ (x, y) tính bằng pixel của tâm ô (row, column)."""
    return (column * cell_pixel_size + cell_pixel_size / 2, row * cell_pixel_size + cell_pixel_size / 2)


def draw_rgb_frame(grid: np.ndarray, robot_position: tuple[int, int], cleaned: np.ndarray,
                   cell_pixel_size: int = CELL_PIXEL_SIZE,
                   recent_positions: list[tuple[int, int]] | None = None) -> np.ndarray:
    """Ảnh RGB (H*cell_pixel_size, W*cell_pixel_size, 3) uint8 của căn phòng nhìn từ trên xuống.

    Args:
        grid: bản đồ.
        robot_position: vị trí robot.
        cleaned: mảng bool ô đã dọn.
        cell_pixel_size: số pixel mỗi ô.
        recent_positions: các vị trí gần nhất (cũ -> mới) để vẽ vệt đường đi.
    """
    height, width = grid.shape
    image = Image.new("RGB", (width * cell_pixel_size, height * cell_pixel_size), COLORS["free"])
    drawer = ImageDraw.Draw(image, "RGBA")
    for row in range(height):
        for column in range(width):
            cell_box = (column * cell_pixel_size, row * cell_pixel_size,
                        (column + 1) * cell_pixel_size - 1, (row + 1) * cell_pixel_size - 1)
            if grid[row, column] == OBSTACLE_CELL:
                cell_color = COLORS["obstacle"]
            elif cleaned[row, column]:
                cell_color = COLORS["cleaned"]
            else:
                cell_color = COLORS["free"]
            drawer.rectangle(cell_box, fill=cell_color, outline=COLORS["grid_line"])
    has_trail = recent_positions is not None and len(recent_positions) >= 2
    if has_trail:
        trail_points = [_cell_center_in_pixels(row, column, cell_pixel_size) for row, column in recent_positions]
        drawer.line(trail_points, fill=(*COLORS["trail"], 140), width=max(2, cell_pixel_size // 8))
    margin = cell_pixel_size * ROBOT_MARGIN_FRACTION
    robot_row, robot_column = robot_position
    drawer.ellipse((robot_column * cell_pixel_size + margin, robot_row * cell_pixel_size + margin,
                    (robot_column + 1) * cell_pixel_size - margin, (robot_row + 1) * cell_pixel_size - margin),
                   fill=COLORS["robot"])
    if has_trail and recent_positions[-1] != recent_positions[-2]:
        # Chấm trắng nhỏ chỉ hướng di chuyển gần nhất của robot.
        (previous_row, previous_column), (current_row, current_column) = recent_positions[-2], recent_positions[-1]
        center_x, center_y = _cell_center_in_pixels(current_row, current_column, cell_pixel_size)
        dot_distance_from_center = cell_pixel_size * 0.18
        dot_radius = cell_pixel_size * 0.08
        dot_x = center_x + (current_column - previous_column) * dot_distance_from_center
        dot_y = center_y + (current_row - previous_row) * dot_distance_from_center
        drawer.ellipse((dot_x - dot_radius, dot_y - dot_radius, dot_x + dot_radius, dot_y + dot_radius),
                       fill=(255, 255, 255))
    return np.asarray(image, dtype=np.uint8).copy()


def draw_text_frame(grid: np.ndarray, robot_position: tuple[int, int], cleaned: np.ndarray) -> str:
    """Chuỗi nhiều dòng: '#' vật cản, '.' chưa dọn, 'o' đã dọn, 'R' robot."""
    text_rows = []
    for row in range(grid.shape[0]):
        row_characters = []
        for column in range(grid.shape[1]):
            if (row, column) == tuple(robot_position):
                row_characters.append("R")
            elif grid[row, column] == OBSTACLE_CELL:
                row_characters.append("#")
            elif cleaned[row, column]:
                row_characters.append("o")
            else:
                row_characters.append(".")
        text_rows.append("".join(row_characters))
    return "\n".join(text_rows)


def add_caption_to_frame(frame: np.ndarray, title: str, caption_lines: list[str],
                         font_size: int = 14) -> np.ndarray:
    """Thêm dải tiêu đề phía trên (tên agent) và dải chú thích phía dưới (số bước, % dọn...).

    Returns:
        Ảnh mới cao hơn ảnh gốc, cùng chiều rộng.
    """
    font = _load_font(font_size)
    line_height = font_size + 6
    header_height = line_height + 6
    footer_height = line_height * max(1, len(caption_lines)) + 6
    frame_height, frame_width = frame.shape[:2]
    canvas = Image.new("RGB", (frame_width, frame_height + header_height + footer_height), COLORS["band"])
    canvas.paste(Image.fromarray(frame), (0, header_height))
    drawer = ImageDraw.Draw(canvas)
    drawer.text((6, 4), title, fill=COLORS["text"], font=font)
    for line_index, caption_line in enumerate(caption_lines):
        drawer.text((6, frame_height + header_height + 3 + line_index * line_height), caption_line,
                    fill=COLORS["text"], font=font)
    return np.asarray(canvas, dtype=np.uint8).copy()
