"""Biểu đồ so sánh có thanh sai số: cột (so sánh agent) và đường (theo một biến số)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.plotting.plot_style import FIGURE_SIZE_INCHES, color_for_agent, plt, save_figure


def plot_bar_with_error(summary: pd.DataFrame, category_column: str, value_column: str, error_column: str,
                        output_path: str | os.PathLike, title: str, y_label: str,
                        group_column: str | None = None) -> Path:
    """Biểu đồ cột có thanh sai số (ví dụ số bước trung bình ± std của từng agent)."""
    figure, axes = plt.subplots(figsize=FIGURE_SIZE_INCHES)
    categories = list(dict.fromkeys(summary[category_column]))
    groups = list(dict.fromkeys(summary[group_column])) if group_column else [None]
    bar_width = 0.8 / len(groups)
    for group_index, group in enumerate(groups):
        rows = summary if group is None else summary[summary[group_column] == group]
        rows = rows.set_index(category_column).reindex(categories)
        bar_positions = np.arange(len(categories)) + (group_index - (len(groups) - 1) / 2) * bar_width
        if group is None:
            bar_colors = [color_for_agent(category, index) for index, category in enumerate(categories)]
        else:
            bar_colors = color_for_agent(str(group), group_index)
        axes.bar(bar_positions, rows[value_column], bar_width, yerr=rows[error_column], color=bar_colors,
                 capsize=4, label=None if group is None else str(group))
    axes.set_xticks(np.arange(len(categories)))
    axes.set_xticklabels(categories)
    axes.set_ylabel(y_label)
    axes.set_title(title)
    if group_column:
        axes.legend(fontsize=9)
    return save_figure(figure, output_path)


def plot_lines_with_error(summary: pd.DataFrame, x_column: str, panels: list[tuple[str, str, str]],
                          group_column: str, output_path: str | os.PathLike, title: str, x_label: str) -> Path:
    """Đường có thanh sai số theo một biến số (ví dụ theo mật độ vật cản, E3).

    Args:
        panels: danh sách (cột giá trị, cột sai số, nhãn trục y), mỗi phần tử một panel.
    """
    figure, axes_grid = plt.subplots(
        1, len(panels), figsize=(FIGURE_SIZE_INCHES[0] * 0.65 * len(panels), FIGURE_SIZE_INCHES[1]), squeeze=False)
    for axes, (value_column, error_column, y_label) in zip(axes_grid[0], panels):
        for group_index, (group_name, group_rows) in enumerate(summary.groupby(group_column, sort=False)):
            group_rows = group_rows.sort_values(x_column)
            axes.errorbar(group_rows[x_column], group_rows[value_column], yerr=group_rows[error_column],
                          marker="o", capsize=4, color=color_for_agent(str(group_name), group_index),
                          label=str(group_name))
        axes.set_xlabel(x_label)
        axes.set_ylabel(y_label)
        axes.legend(fontsize=9)
    figure.suptitle(title)
    return save_figure(figure, output_path)
