"""Vẽ trạng thái môi trường thành ảnh, ghép khung hình, xuất GIF (D-26: dùng Pillow).

Phần hiển thị chỉ để xem, không ảnh hưởng tới việc học.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from env.maps import OBSTACLE
from utils.paths import assert_inside_root

CELL_PX: int = 32
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
ROBOT_MARGIN: float = 0.2  # robot chiếm 60% ô, lề 20% mỗi bên
_FONT_CACHE: dict[int, ImageFont.ImageFont] = {}


def _font(size: int) -> ImageFont.ImageFont:
    """Font DejaVuSans đi kèm matplotlib (hỗ trợ tiếng Việt có dấu); dự phòng font mặc định."""
    if size not in _FONT_CACHE:
        try:
            import matplotlib

            path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
            _FONT_CACHE[size] = ImageFont.truetype(str(path), size)
        except (ImportError, OSError):
            _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


def _cell_center(r: int, c: int, cell_px: int) -> tuple[float, float]:
    return (c * cell_px + cell_px / 2, r * cell_px + cell_px / 2)


def render_rgb(grid: np.ndarray, pos: tuple[int, int], cleaned: np.ndarray, cell_px: int = CELL_PX,
               trail: list[tuple[int, int]] | None = None) -> np.ndarray:
    """Ảnh RGB (H*cell_px, W*cell_px, 3) uint8 của căn phòng nhìn từ trên xuống.

    Args:
        grid: bản đồ.
        pos: vị trí robot.
        cleaned: mảng bool ô đã dọn.
        cell_px: số pixel mỗi ô.
        trail: các vị trí gần nhất (cũ -> mới) để vẽ vệt đường đi.
    """
    height, width = grid.shape
    img = Image.new("RGB", (width * cell_px, height * cell_px), COLORS["free"])
    draw = ImageDraw.Draw(img, "RGBA")
    for r in range(height):
        for c in range(width):
            box = (c * cell_px, r * cell_px, (c + 1) * cell_px - 1, (r + 1) * cell_px - 1)
            if grid[r, c] == OBSTACLE:
                color = COLORS["obstacle"]
            elif cleaned[r, c]:
                color = COLORS["cleaned"]
            else:
                color = COLORS["free"]
            draw.rectangle(box, fill=color, outline=COLORS["grid_line"])
    if trail and len(trail) >= 2:
        points = [_cell_center(r, c, cell_px) for r, c in trail]
        draw.line(points, fill=(*COLORS["trail"], 140), width=max(2, cell_px // 8))
    margin = cell_px * ROBOT_MARGIN
    r, c = pos
    draw.ellipse((c * cell_px + margin, r * cell_px + margin,
                  (c + 1) * cell_px - margin, (r + 1) * cell_px - margin), fill=COLORS["robot"])
    if trail and len(trail) >= 2 and trail[-1] != trail[-2]:
        # Chấm trắng nhỏ chỉ hướng di chuyển gần nhất của robot.
        (pr, pc), (cr, cc) = trail[-2], trail[-1]
        cx, cy = _cell_center(cr, cc, cell_px)
        offset = cell_px * 0.18
        dot = cell_px * 0.08
        dx, dy = (cc - pc) * offset, (cr - pr) * offset
        draw.ellipse((cx + dx - dot, cy + dy - dot, cx + dx + dot, cy + dy + dot), fill=(255, 255, 255))
    return np.asarray(img, dtype=np.uint8).copy()


def render_ansi(grid: np.ndarray, pos: tuple[int, int], cleaned: np.ndarray) -> str:
    """Chuỗi nhiều dòng: '#' vật cản, '.' chưa dọn, 'o' đã dọn, 'R' robot."""
    lines = []
    for r in range(grid.shape[0]):
        chars = []
        for c in range(grid.shape[1]):
            if (r, c) == tuple(pos):
                chars.append("R")
            elif grid[r, c] == OBSTACLE:
                chars.append("#")
            elif cleaned[r, c]:
                chars.append("o")
            else:
                chars.append(".")
        lines.append("".join(chars))
    return "\n".join(lines)


def annotate_frame(frame: np.ndarray, title: str, lines: list[str], font_size: int = 14) -> np.ndarray:
    """Thêm dải tiêu đề phía trên (tên agent) và dải chú thích phía dưới (số bước, % dọn...).

    Returns:
        Ảnh mới cao hơn ảnh gốc, cùng chiều rộng.
    """
    font = _font(font_size)
    line_h = font_size + 6
    header_h = line_h + 6
    footer_h = line_h * max(1, len(lines)) + 6
    h, w = frame.shape[:2]
    canvas = Image.new("RGB", (w, h + header_h + footer_h), COLORS["band"])
    canvas.paste(Image.fromarray(frame), (0, header_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 4), title, fill=COLORS["text"], font=font)
    for i, line in enumerate(lines):
        draw.text((6, h + header_h + 3 + i * line_h), line, fill=COLORS["text"], font=font)
    return np.asarray(canvas, dtype=np.uint8).copy()


def pad_to_size(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    """Đệm ảnh bằng màu trắng (căn trái-trên) cho đủ kích thước, để ghép các ảnh khác cỡ."""
    h, w = frame.shape[:2]
    if h > height or w > width:
        raise ValueError(f"Ảnh {h}x{w} lớn hơn kích thước đích {height}x{width}.")
    out = np.full((height, width, 3), 255, dtype=np.uint8)
    out[:h, :w] = frame
    return out


def compose_side_by_side(frames: list[np.ndarray], gap_px: int = 8) -> np.ndarray:
    """Ghép các ảnh theo hàng ngang, cách nhau một khoảng trắng (ảnh khác cỡ được đệm cho bằng)."""
    if not frames:
        raise ValueError("Cần ít nhất một ảnh để ghép.")
    height = max(f.shape[0] for f in frames)
    parts: list[np.ndarray] = []
    for i, frame in enumerate(frames):
        if i > 0:
            parts.append(np.full((height, gap_px, 3), 255, dtype=np.uint8))
        parts.append(pad_to_size(frame, height, frame.shape[1]))
    return np.concatenate(parts, axis=1)


def save_gif(frames: list[np.ndarray], path: str | os.PathLike, fps: int = 4,
             last_frame_hold: int = 3) -> Path:
    """Ghi danh sách khung hình thành GIF lặp vô hạn (khung cuối giữ lâu hơn cho dễ nhìn).

    Raises:
        ValueError: danh sách rỗng, fps không dương, hoặc đường dẫn ngoài thư mục dự án.
    """
    if not frames:
        raise ValueError("Không có khung hình nào để ghi GIF.")
    if fps <= 0:
        raise ValueError(f"fps phải dương, nhận được {fps}.")
    resolved = assert_inside_root(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(f) for f in frames]
    duration = [1000 // fps] * len(images)
    duration[-1] = duration[-1] * max(1, last_frame_hold)
    images[0].save(resolved, save_all=True, append_images=images[1:], duration=duration, loop=0,
                   optimize=False)
    return resolved


def record_episode(agent: Any, env: Any, path: str | os.PathLike, seed: int = 0,
                   max_frames: int = 500, fps: int = 4) -> dict[str, Any]:
    """Chạy một episode với env có render_mode="rgb_array", lưu GIF và trả các chỉ số.

    Raises:
        ValueError: nếu env không ở chế độ rgb_array.
    """
    if getattr(env, "render_mode", None) != "rgb_array":
        raise ValueError("record_episode cần env với render_mode='rgb_array'.")
    if hasattr(agent, "reset"):
        agent.reset()
    obs, info = env.reset(seed=seed)
    frames = [env.render()]
    total_reward = 0.0
    terminated = truncated = False
    while not (terminated or truncated) and len(frames) < max_frames:
        obs, reward, terminated, truncated, info = env.step(agent.act(obs, info))
        total_reward += reward
        frames.append(env.render())
    save_gif(frames, path, fps=fps)
    return {"steps": info["steps"], "coverage": info["coverage"], "return_undiscounted": total_reward,
            "terminated": terminated, "truncated": truncated, "frames": len(frames)}


def demo_random_gif(map_name: str, path: str | os.PathLike, seed: int = 0) -> Path:
    """Demo nhanh: robot đi ngẫu nhiên (chỉ các hướng không đụng tường) trên bản đồ cố định."""
    from baselines.random_walk import RandomWalkAgent
    from env.vacuum_env import VacuumEnv

    env = VacuumEnv(map_name, render_mode="rgb_array")
    record_episode(RandomWalkAgent(seed=seed), env, path, seed=seed)
    return assert_inside_root(path)


def visit_heatmap(visits: np.ndarray, grid: np.ndarray, ax: Any = None, vmax: float | None = None) -> Any:
    """Vẽ số lần robot đi qua mỗi ô (ô vật cản tô xám). Trả về đối tượng ảnh của matplotlib."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    data = np.ma.masked_where(grid == OBSTACLE, visits.astype(float))
    cmap = plt.get_cmap("Reds").copy()
    cmap.set_bad(color=np.array(COLORS["obstacle"]) / 255.0)
    image = ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax)
    for (r, c), v in np.ndenumerate(visits):
        if grid[r, c] != OBSTACLE:
            ax.text(c, r, int(v), ha="center", va="center", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    return image
