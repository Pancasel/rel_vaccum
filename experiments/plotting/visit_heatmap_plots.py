"""Heatmap số lần đi qua mỗi ô của nhiều agent, đặt cạnh nhau để so sánh mức lãng phí."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from experiments.plotting.plot_style import plt, save_figure


def plot_visit_heatmaps(visit_counts_by_agent: dict[str, np.ndarray], grid: np.ndarray,
                        output_path: str | os.PathLike, subtitles: dict[str, str] | None = None) -> Path:
    """Heatmap số lần đi qua mỗi ô, mỗi agent một ô con, cùng thang màu."""
    from environment.rendering.visit_heatmap import draw_visit_heatmap

    agent_names = list(visit_counts_by_agent)
    color_scale_max = max(float(counts.max()) for counts in visit_counts_by_agent.values())
    figure, axes_grid = plt.subplots(1, len(agent_names), figsize=(3.2 * len(agent_names), 3.6), squeeze=False)
    image = None
    for axes, agent_name in zip(axes_grid[0], agent_names):
        image = draw_visit_heatmap(visit_counts_by_agent[agent_name], grid, axes=axes,
                                   color_scale_max=color_scale_max)
        axes.set_title(f"{agent_name}\n{(subtitles or {}).get(agent_name, '')}", fontsize=10)
    if image is not None:
        figure.colorbar(image, ax=axes_grid[0].tolist(), shrink=0.8, label="số lần đi qua")
    figure.suptitle("Số lần robot đi qua mỗi ô (đỏ đậm = lãng phí)")
    return save_figure(figure, output_path, tight_bounding_box=True)
