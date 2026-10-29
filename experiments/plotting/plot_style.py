"""Phong cách chung cho mọi hình: màu theo agent, kích thước, định dạng trục, lưu file.

Quy ước (mục 7.2 kế hoạch): tiêu đề và nhãn trục tiếng Việt, màu cố định theo agent,
ghi số seed trong tiêu đề phụ, backend Agg (không mở cửa sổ).
"""

from __future__ import annotations

import os
from pathlib import Path

from project_utils.project_paths import require_path_inside_project, setup_local_caches

setup_local_caches()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

AGENT_COLORS: dict[str, str] = {
    "random_walk": "#9e9e9e",
    "zigzag": "#ff9800",
    "greedy_bfs": "#4caf50",
    "q_learning": "#1f77b4",
    "dqn": "#d62728",
    "value_iteration": "#000000",
    "sarsa": "#9467bd",
}
_FALLBACK_COLORS = plt.get_cmap("tab10").colors
FIGURE_SIZE_INCHES: tuple[float, float] = (8, 4.5)
DOTS_PER_INCH: int = 150

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.grid": True, "grid.alpha": 0.3})


def color_for_agent(name: str, fallback_index: int = 0) -> str:
    """Màu cố định cho agent/cấu hình; tên lạ thì lấy màu theo thứ tự fallback_index."""
    fallback_color = matplotlib.colors.to_hex(_FALLBACK_COLORS[fallback_index % len(_FALLBACK_COLORS)])
    return AGENT_COLORS.get(name, fallback_color)


def format_compact_number(value: float, _tick_position: int | None = None) -> str:
    """Nhãn trục gọn: 20000 -> 20k, 1500000 -> 1.5M (tránh các số dài dính vào nhau)."""
    for scale, suffix in ((1e6, "M"), (1e3, "k")):
        if abs(value) >= scale:
            return f"{value / scale:g}{suffix}"
    return f"{value:g}"


def tidy_x_axis(axes: plt.Axes) -> None:
    """Tối đa ~5 vạch chia, số lớn viết gọn — cho trục số episode / số bước huấn luyện."""
    axes.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5, integer=True))
    axes.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_compact_number))


def save_figure(figure: plt.Figure, output_path: str | os.PathLike, tight_bounding_box: bool = False) -> Path:
    """Lưu hình vào thư mục dự án (tự tạo thư mục cha) rồi đóng hình để giải phóng bộ nhớ."""
    resolved_path = require_path_inside_project(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    if tight_bounding_box:
        figure.savefig(resolved_path, dpi=DOTS_PER_INCH, bbox_inches="tight")
    else:
        figure.tight_layout()
        figure.savefig(resolved_path, dpi=DOTS_PER_INCH)
    plt.close(figure)
    return resolved_path
