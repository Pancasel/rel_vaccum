"""Learning curve (đường cong học): trung bình ± độ lệch chuẩn (std) qua các seed."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

from experiments.plotting.plot_style import (FIGURE_SIZE_INCHES, color_for_agent, plt, save_figure,
                                             tidy_x_axis)
from project_utils.project_paths import PROJECT_ROOT


def load_seed_csv_files(glob_pattern: str, search_root: str | os.PathLike = PROJECT_ROOT) -> pd.DataFrame:
    """Đọc nhiều CSV theo mẫu glob (tính từ search_root), thêm cột 'seed' lấy từ tên file 'seed<N>'.

    Raises:
        FileNotFoundError: không có file nào khớp.
    """
    csv_paths = sorted(Path(search_root).glob(glob_pattern))
    if not csv_paths:
        raise FileNotFoundError(f"Không có file nào khớp '{glob_pattern}' trong {search_root}.")
    tables = []
    for csv_path in csv_paths:
        table = pd.read_csv(csv_path)
        seed_match = re.search(r"seed(\d+)", csv_path.as_posix())
        if "seed" not in table.columns:
            table["seed"] = int(seed_match.group(1)) if seed_match else 0
        table["source"] = csv_path.as_posix()
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def _mean_and_std_across_seeds(results_table: pd.DataFrame, x_column: str, y_column: str,
                               smoothing_window: int) -> pd.DataFrame:
    """Trung bình ± std qua seed tại mỗi x; làm mượt bằng trung bình trượt nếu smoothing_window > 1."""
    per_seed_values = results_table[[x_column, y_column, "seed"]].dropna(subset=[y_column]).sort_values(x_column)
    if smoothing_window > 1:
        per_seed_values = per_seed_values.copy()
        per_seed_values[y_column] = per_seed_values.groupby("seed")[y_column].transform(
            lambda values: values.rolling(smoothing_window, min_periods=1).mean())
    statistics = per_seed_values.groupby(x_column)[y_column].agg(["mean", "std", "count"]).reset_index()
    statistics["std"] = statistics["std"].fillna(0.0)
    return statistics


def plot_learning_curves(results_table: pd.DataFrame, x_column: str, panels: list[tuple[str, str]],
                         group_column: str, output_path: str | os.PathLike, title: str, x_label: str,
                         smoothing_window: int = 1,
                         reference_lines: dict[str, dict[str, float]] | None = None) -> Path:
    """Learning curve nhiều panel (mỗi panel một chỉ số), mỗi đường một giá trị của `group_column`.

    Args:
        results_table: dữ liệu dạng dài, có cột x, các cột chỉ số, cột group và cột seed.
        panels: danh sách (tên cột chỉ số, nhãn trục y).
        reference_lines: {tên cột chỉ số: {nhãn: giá trị}} các đường ngang tham chiếu (tối ưu, baseline...).
    """
    width_per_panel = FIGURE_SIZE_INCHES[0] * (0.65 if len(panels) > 1 else 1)
    figure, axes_grid = plt.subplots(1, len(panels), figsize=(width_per_panel * len(panels), FIGURE_SIZE_INCHES[1]),
                                     squeeze=False)
    seed_count = results_table["seed"].nunique()
    for axes, (metric_column, y_label) in zip(axes_grid[0], panels):
        # Tỷ lệ (coverage, success) nằm trong [0, 1]: cắt dải std để không vẽ giá trị vô nghĩa.
        is_bounded_ratio = "coverage" in metric_column or "success" in metric_column
        for group_index, (group_name, group_rows) in enumerate(results_table.groupby(group_column, sort=False)):
            statistics = _mean_and_std_across_seeds(group_rows, x_column, metric_column, smoothing_window)
            line_color = color_for_agent(str(group_name), group_index)
            lower_band = statistics["mean"] - statistics["std"]
            upper_band = statistics["mean"] + statistics["std"]
            if is_bounded_ratio:
                lower_band, upper_band = lower_band.clip(0.0, 1.0), upper_band.clip(0.0, 1.0)
            axes.plot(statistics[x_column], statistics["mean"], color=line_color, label=str(group_name))
            axes.fill_between(statistics[x_column], lower_band, upper_band, color=line_color, alpha=0.2)
        reference_values = (reference_lines or {}).get(metric_column, {})
        for line_index, (label, value) in enumerate(reference_values.items()):
            # Ghi giá trị vào chú thích: các đường tham chiếu trùng nhau vẫn đọc được.
            axes.axhline(value, linestyle="--", linewidth=1.2, color=color_for_agent(label, line_index + 5),
                         label=f"{label} = {value:.4g}")
        tidy_x_axis(axes)
        axes.set_xlabel(x_label)
        axes.set_ylabel(y_label)
        axes.legend(fontsize=9)
    figure.suptitle(f"{title}\n(trung bình ± std, {seed_count} seed)")
    return save_figure(figure, output_path)


def plot_learning_curve(results_table: pd.DataFrame, x_column: str, y_column: str, group_column: str,
                        output_path: str | os.PathLike, title: str = "", x_label: str = "", y_label: str = "",
                        smoothing_window: int = 1, reference_line_value: float | None = None,
                        reference_line_label: str = "") -> Path:
    """Learning curve một panel (bọc plot_learning_curves)."""
    reference_lines = None
    if reference_line_value is not None:
        reference_lines = {y_column: {reference_line_label or "tham chiếu": reference_line_value}}
    return plot_learning_curves(results_table, x_column, [(y_column, y_label or y_column)], group_column,
                                output_path, title or y_column, x_label or x_column, smoothing_window,
                                reference_lines)


def plot_q_convergence(training_logs: pd.DataFrame, optimal_start_value: float, output_path: str | os.PathLike,
                       smoothing_window: int = 50) -> Path:
    """max_a Q(s0, a) theo episode so với giá trị tối ưu thật V*(s0) từ Value Iteration."""
    logs = training_logs.copy()
    if "algo" not in logs.columns:
        logs["algo"] = "q_learning"
    return plot_learning_curves(logs, "episode", [("q_start_max", "max_a Q(s0, a)")], "algo", output_path,
                                "Ước lượng Q của trạng thái xuất phát hội tụ về V*", "episode",
                                smoothing_window=smoothing_window,
                                reference_lines={"q_start_max": {"V*(s0) (Value Iteration)": optimal_start_value}})
