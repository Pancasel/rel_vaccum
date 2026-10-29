"""Hình cho thí nghiệm bùng nổ trạng thái (E7) và giới hạn của Q-learning dạng bảng (T5.3)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.plotting.plot_style import FIGURE_SIZE_INCHES, color_for_agent, plt, save_figure

# Bảng Q dạng float64 với 4 hành động: 4 giá trị x 8 byte = 32 byte mỗi trạng thái.
BYTES_PER_Q_TABLE_STATE: int = 32


def plot_state_explosion(counts_table: pd.DataFrame, output_path: str | os.PathLike, memory_limit_gb: float = 8.0,
                         title: str = "Bùng nổ không gian trạng thái của Q-learning dạng bảng") -> Path:
    """E7: số trạng thái theo số ô trống (thang log), đo thực tế và cận trên |F|·2^|F|.

    Cột bắt buộc: n_free (số ô trống |F|), upper_bound, reachable (NaN nếu không đếm được), grid (tên lưới).
    """
    figure, axes = plt.subplots(figsize=FIGURE_SIZE_INCHES)
    counts_table = counts_table.sort_values("n_free")
    axes.plot(counts_table["n_free"], counts_table["upper_bound"], "o--", color="#d62728",
              label="cận trên |F|·2^|F|")
    measured_rows = counts_table.dropna(subset=["reachable"])
    axes.plot(measured_rows["n_free"], measured_rows["reachable"], "o-", color="#1f77b4",
              label="đạt tới được (đo bằng BFS)")
    for _, row in counts_table.iterrows():
        axes.annotate(row["grid"], (row["n_free"], row["upper_bound"]), textcoords="offset points", xytext=(0, 6),
                      ha="center", fontsize=8)
    memory_limit_in_states = memory_limit_gb * 1e9 / BYTES_PER_Q_TABLE_STATE
    axes.axhspan(memory_limit_in_states, max(counts_table["upper_bound"].max() * 10, memory_limit_in_states * 10),
                 color="#d62728", alpha=0.08, label=f"bảng Q > {memory_limit_gb:g} GB RAM")
    axes.set_yscale("log")
    axes.set_xlabel("số ô trống |F|")
    axes.set_ylabel("số trạng thái (thang log)")
    axes.set_title(title)
    axes.legend(fontsize=9)
    return save_figure(figure, output_path)


def plot_tabular_scaling(training_table: pd.DataFrame, evaluation_table: pd.DataFrame,
                         output_path: str | os.PathLike) -> Path:
    """T5.3: số trạng thái bảng Q đã gặp (thang log) và coverage tham lam theo thời gian, mỗi lưới một đường."""
    figure, axes_pair = plt.subplots(1, 2, figsize=(FIGURE_SIZE_INCHES[0] * 1.3, FIGURE_SIZE_INCHES[1]))
    for group_index, (map_name, rows) in enumerate(training_table.groupby("map", sort=False)):
        axes_pair[0].plot(rows["wall_time_s"] / 60, rows["n_states_visited"],
                          color=color_for_agent(str(map_name), group_index), label=map_name)
    for group_index, (map_name, rows) in enumerate(evaluation_table.groupby("map", sort=False)):
        same_map_training = training_table[training_table["map"] == map_name]
        minutes = np.interp(rows["episode"], same_map_training["episode"], same_map_training["wall_time_s"]) / 60
        axes_pair[1].plot(minutes, rows["coverage_mean"], color=color_for_agent(str(map_name), group_index),
                          label=map_name)
    axes_pair[0].set_yscale("log")
    axes_pair[0].set_xlabel("thời gian huấn luyện (phút)")
    axes_pair[0].set_ylabel("số trạng thái trong bảng Q (log)")
    axes_pair[1].set_xlabel("thời gian huấn luyện (phút)")
    axes_pair[1].set_ylabel("tỷ lệ đã dọn (chính sách tham lam)")
    axes_pair[1].set_ylim(0, 1.05)
    for axes in axes_pair:
        axes.legend(fontsize=9)
    figure.suptitle("Q-learning dạng bảng khi lưới lớn dần: bảng phình to, chính sách không kịp học")
    return save_figure(figure, output_path)
