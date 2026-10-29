"""Test cho experiments/plots.py: các hàm vẽ tạo file, đọc nhiều seed, bảng markdown."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from environment.room_layouts import get_room_layout
from experiments.plots import (color_for, df_to_markdown_table, load_runs, plot_bar_with_error, plot_learning_curve,
                               plot_learning_curves, plot_lines_with_error, plot_q_convergence, plot_state_explosion,
                               plot_tabular_scaling, plot_visit_heatmaps, write_text)


def _curve_df() -> pd.DataFrame:
    rows = [{"episode": e, "coverage_mean": min(1.0, e / 10 + s * 0.01), "steps_mean": 50 - e, "seed": s,
             "algo": "q_learning", "q_start_max": e / 2} for s in range(3) for e in range(1, 11)]
    return pd.DataFrame(rows)


def test_load_runs(tmp_path: Path) -> None:
    for seed in (0, 1):
        pd.DataFrame({"x": [1, 2]}).to_csv(tmp_path / f"seed{seed}_eval.csv", index=False)
    df = load_runs("seed*_eval.csv", root=tmp_path)
    assert sorted(df["seed"].unique()) == [0, 1] and len(df) == 4
    with pytest.raises(FileNotFoundError):
        load_runs("nothing*.csv", root=tmp_path)


def test_learning_curves_and_convergence(tmp_path: Path) -> None:
    df = _curve_df()
    p1 = plot_learning_curves(df, "episode", [("coverage_mean", "cov"), ("steps_mean", "steps")], "algo",
                              tmp_path / "a.png", "tiêu đề", "episode", hlines={"steps_mean": {"tối ưu": 13}})
    p2 = plot_learning_curve(df, "episode", "steps_mean", "algo", tmp_path / "b.png", hline=13.0, smooth=3)
    p3 = plot_q_convergence(df, v_star=5.0, out_path=tmp_path / "c.png", smooth=2)
    assert all(p.exists() and p.stat().st_size > 0 for p in (p1, p2, p3))


def test_bar_and_lines(tmp_path: Path) -> None:
    summary = pd.DataFrame({"K": ["1", "1", "∞", "∞"], "split": ["train", "test"] * 2,
                            "coverage_mean": [0.9, 0.5, 0.8, 0.8], "coverage_mean_sd": [0.05] * 4,
                            "density": [0.0, 0.0, 0.1, 0.1], "agent": ["dqn", "zigzag"] * 2})
    assert plot_bar_with_error(summary, "K", "coverage_mean", "coverage_mean_sd", tmp_path / "bar.png", "t", "y",
                               hue="split").exists()
    assert plot_lines_with_error(summary, "density", [("coverage_mean", "coverage_mean_sd", "y")], "agent",
                                 tmp_path / "lines.png", "t", "x").exists()


def test_state_explosion_and_scaling(tmp_path: Path) -> None:
    counts = pd.DataFrame({"grid": ["2x2", "3x3", "5x5"], "n_free": [4, 9, 25], "upper_bound": [64.0, 4608.0, 8.4e8],
                           "reachable": [28.0, 1081.0, np.nan]})
    assert plot_state_explosion(counts, tmp_path / "e7.png").exists()
    train = pd.DataFrame({"map": ["a"] * 3, "episode": [1, 2, 3], "wall_time_s": [1.0, 2.0, 3.0],
                          "n_states_visited": [10, 20, 30]})
    evals = pd.DataFrame({"map": ["a"], "episode": [3], "coverage_mean": [1.0]})
    assert plot_tabular_scaling(train, evals, tmp_path / "scaling.png").exists()


def test_visit_heatmaps(tmp_path: Path) -> None:
    grid = get_room_layout("obstacle_4x4")
    visits = {"a": np.ones((4, 4), dtype=int), "b": np.arange(16).reshape(4, 4)}
    assert plot_visit_heatmaps(visits, grid, tmp_path / "h.png", {"a": "x"}).exists()


def test_markdown_and_colors(tmp_path: Path) -> None:
    table = df_to_markdown_table(pd.DataFrame({"a": [1, 2], "b": [0.123, np.nan]}))
    assert table.splitlines() == ["| a | b |", "|---|---|", "| 1 | 0.12 |", "| 2 | nan |"]
    assert color_for("dqn") == "#d62728"
    assert color_for("something_new", 2).startswith("#")
    assert write_text("xin chào", tmp_path / "t.md").read_text(encoding="utf-8") == "xin chào"


def test_compact_axis_numbers() -> None:
    from experiments.plots import _compact_number

    assert [_compact_number(v) for v in (0, 500, 2500, 20000, 1_500_000)] == ["0", "500", "2.5k", "20k", "1.5M"]


def test_state_explosion_without_title(tmp_path: Path) -> None:
    counts = pd.DataFrame({"grid": ["2x2", "3x3"], "n_free": [4, 9], "upper_bound": [64.0, 4608.0],
                           "reachable": [28.0, 1081.0]})
    assert plot_state_explosion(counts, tmp_path / "e7_untitled.png", title="").exists()
