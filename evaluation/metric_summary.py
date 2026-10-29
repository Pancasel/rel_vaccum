"""Cột chỉ số của mỗi episode và cách tổng hợp thành trung bình ± độ lệch chuẩn (chỉ số B6)."""

from __future__ import annotations

import pandas as pd

# Tên cột là một phần giao diện dữ liệu (CSV trong results/ và báo cáo đang trích), không đổi.
PER_EPISODE_METRIC_COLUMNS: list[str] = [
    "episode", "env_seed", "start_row", "start_col", "return_undiscounted", "return_discounted",
    "coverage", "steps", "steps_to_full", "redundancy", "collisions", "n_revisit", "terminated", "truncated",
]
SUMMARIZED_METRICS: list[str] = [
    "return_undiscounted", "return_discounted", "coverage", "steps", "steps_to_full", "redundancy",
    "collisions", "n_revisit",
]


def summarize_metrics(episode_table: pd.DataFrame, group_columns: list[str] | None = None) -> pd.DataFrame:
    """Trung bình và độ lệch chuẩn (ddof=1) của các chỉ số, kèm số episode n và success_rate.

    Cột đầu ra dạng "coverage_mean", "coverage_std" (std = standard deviation, độ lệch chuẩn).
    Độ lệch chuẩn của nhóm 1 phần tử ghi 0. steps_to_full chỉ tính trên các episode dọn xong
    (NaN bị bỏ qua). success_rate = tỷ lệ episode dọn xong 100% (terminated).
    """
    metrics = [metric for metric in SUMMARIZED_METRICS if metric in episode_table.columns]
    if group_columns:
        grouped = episode_table.groupby(group_columns, sort=False)
        means = grouped[metrics].mean().add_suffix("_mean")
        standard_deviations = grouped[metrics].std(ddof=1).fillna(0.0).add_suffix("_std")
        counts = pd.DataFrame({"n": grouped.size(), "success_rate": grouped["terminated"].mean()})
        summary = pd.concat([counts, means, standard_deviations], axis=1).reset_index()
    else:
        means = episode_table[metrics].mean().add_suffix("_mean")
        standard_deviations = episode_table[metrics].std(ddof=1).fillna(0.0).add_suffix("_std")
        counts = pd.Series({"n": len(episode_table), "success_rate": episode_table["terminated"].mean()})
        summary = pd.concat([counts, means, standard_deviations]).to_frame().T
    ordered_columns = [column for column in summary.columns if not column.endswith(("_mean", "_std"))]
    for metric in metrics:
        ordered_columns += [f"{metric}_mean", f"{metric}_std"]
    return summary[ordered_columns]
