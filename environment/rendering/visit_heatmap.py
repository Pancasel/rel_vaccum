"""Vẽ bản đồ nhiệt số lần robot đi qua từng ô."""

from __future__ import annotations

from typing import Any

import numpy as np

from environment.rendering.frame_drawing import COLORS
from environment.room_layouts import OBSTACLE_CELL


def draw_visit_heatmap(visit_counts: np.ndarray, grid: np.ndarray, axes: Any = None,
                       color_scale_max: float | None = None) -> Any:
    """Vẽ số lần robot đi qua mỗi ô (ô vật cản tô xám). Trả về đối tượng ảnh của matplotlib."""
    import matplotlib.pyplot as plt

    if axes is None:
        _, axes = plt.subplots(figsize=(4, 4))
    masked_counts = np.ma.masked_where(grid == OBSTACLE_CELL, visit_counts.astype(float))
    color_map = plt.get_cmap("Reds").with_extremes(bad=np.array(COLORS["obstacle"]) / 255.0)
    image = axes.imshow(masked_counts, cmap=color_map, vmin=0, vmax=color_scale_max)
    for (row, column), visit_count in np.ndenumerate(visit_counts):
        if grid[row, column] != OBSTACLE_CELL:
            axes.text(column, row, int(visit_count), ha="center", va="center", fontsize=8)
    axes.set_xticks([])
    axes.set_yticks([])
    return image
