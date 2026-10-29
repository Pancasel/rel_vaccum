"""Vẽ lại các learning curve từ CSV đã lưu trong results/ (không huấn luyện lại).

Dùng khi sửa cách vẽ (ví dụ định dạng trục) mà muốn cập nhật hình của các lần chạy cũ.

    python -m experiments.replot_saved_results
"""

from __future__ import annotations

import re
from pathlib import Path

from evaluation.optimal_reference import optimal_value_of_start, reference_step_counts
from experiments.shared.command_line_options import resolve_output_directory

import pandas as pd  # noqa: E402

from agents.baselines import GreedyNearestDirtyCellAgent, ZigzagSweepAgent  # noqa: E402
from environment.random_room_generator import generate_distinct_rooms  # noqa: E402
from environment.room_layouts import get_room_layout  # noqa: E402
from environment.reward_settings import RewardSettings  # noqa: E402
from evaluation.agent_evaluation import evaluate_agent_on_rooms  # noqa: E402
from experiments.plotting.learning_curve_plots import plot_learning_curves, plot_q_convergence  # noqa: E402
from experiments.shared.evaluation_seeds import TEST_ROOMS_SEED  # noqa: E402

EVAL_PANELS = [("coverage_mean", "tỷ lệ đã dọn (chính sách tham lam)"), ("steps_mean", "số bước (chính sách tham lam)")]


def _read_seeds(paths: list[Path], **extra: str) -> pd.DataFrame:
    frames = []
    for path in paths:
        seed = int(re.search(r"seed(\d+)", path.parent.name).group(1))
        frames.append(pd.read_csv(path).assign(seed=seed, **extra))
    return pd.concat(frames, ignore_index=True)


def replot_tabular(results: Path) -> list[Path]:
    """Learning curve và hội tụ Q của các lượt run_tabular (reward dense, state full)."""
    out = []
    for run_dir in sorted((results / "tabular" / "dense" / "full").iterdir()):
        if not run_dir.is_dir():
            continue
        match = re.fullmatch(r"(.+?)(?:_s(\d)(\d))?", run_dir.name)
        map_name = match.group(1)
        start = (int(match.group(2)), int(match.group(3))) if match.group(2) else None
        grid = get_room_layout(map_name)
        evals = _read_seeds(sorted(run_dir.glob("q_learning/seed*/seed*_eval.csv")), algo="q_learning")
        start_txt = f"xuất phát {start}" if start else "xuất phát ngẫu nhiên"
        out.append(plot_learning_curves(
            evals, "episode", EVAL_PANELS, "algo", results / "tabular" / "figures" / f"tabular_learning_curve_{run_dir.name}.png",
            f"q_learning trên {map_name} ({start_txt}, reward dense, state full)", "episode",
            reference_lines={"steps_mean": reference_step_counts(grid, start)}))
        if start is not None:
            train = _read_seeds(sorted(run_dir.glob("q_learning/seed*/seed*_train.csv")), algo="q_learning")
            optimal_start_value = optimal_value_of_start(grid, start, RewardSettings.dense(), 0.99)
            if optimal_start_value is not None:
                episodes = int(train["episode"].max())
                out.append(plot_q_convergence(train, optimal_start_value, results / "tabular" / "figures" /
                                              f"q_convergence_{run_dir.name}.png", smoothing_window=max(1, episodes // 400)))
    return out


def replot_ablation(results: Path) -> list[Path]:
    """Hình E1 (tính Markov) và E2 (thiết kế reward) của run_ablation."""
    out = []
    titles = {"markov": "E1: state có / không có bản đồ đã dọn ({map}, xuất phát (0, 0))",
              "reward": "E2: bốn thiết kế reward ({map}, xuất phát (0, 0))"}
    names = {"markov": "e1_markov_{map}.png", "reward": "e2_reward_curves_{map}.png"}
    for exp in ("markov", "reward"):
        exp_dir = results / "ablation" / exp
        if not exp_dir.exists():
            continue
        for map_dir in sorted(path for path in exp_dir.iterdir() if path.is_dir()):
            frames = [_read_seeds(sorted(variant.glob("seed*/seed*_eval.csv")), variant=variant.name)
                      for variant in sorted(path for path in map_dir.iterdir() if path.is_dir())]
            results_table = pd.concat(frames, ignore_index=True)
            out.append(plot_learning_curves(
                results_table, "episode", [("coverage_mean", "tỷ lệ đã dọn (tham lam)"), ("steps_mean", "số bước (tham lam)")],
                "variant", results / "ablation" / "figures" / names[exp].format(map=map_dir.name),
                titles[exp].format(map=map_dir.name), "episode"))
    return out


def _deep_references(tag: str) -> dict[str, float]:
    """Đường tham chiếu: tối ưu/greedy/zigzag trên bản đồ cố định, hoặc trên tập test 50 phòng ngẫu nhiên."""
    random_match = re.match(r"rand(\d+)x(\d+)_d([\d.]+)_K0", tag)
    if random_match:
        height, width, density = int(random_match.group(1)), int(random_match.group(2)), float(random_match.group(3))
        test_maps = generate_distinct_rooms(height, width, density, 50, seed=TEST_ROOMS_SEED)
        return {name: float(evaluate_agent_on_rooms(agent, test_maps, 1, seed=0)["steps"].mean())
                for name, agent in (("greedy_bfs", GreedyNearestDirtyCellAgent()), ("zigzag", ZigzagSweepAgent()))}
    fixed = re.fullmatch(r"(.+?)(?:_s(\d)(\d))?", tag)
    start = (int(fixed.group(2)), int(fixed.group(3))) if fixed.group(2) else None
    return reference_step_counts(get_room_layout(fixed.group(1)), start)


def replot_deep(results: Path) -> list[Path]:
    """Learning curve của các lượt run_deep trong results/deep/dqn/."""
    out = []
    for run_dir in sorted(path for path in (results / "deep" / "dqn").iterdir() if path.is_dir()):
        paths = sorted(run_dir.glob("seed*/eval.csv"))
        if not paths:
            continue
        results_table = _read_seeds(paths, algo="dqn")
        out.append(plot_learning_curves(
            results_table, "global_step", EVAL_PANELS, "algo", results / "deep" / "figures" / f"deep_learning_curve_{run_dir.name}.png",
            f"DQN trên {run_dir.name}", "bước huấn luyện", reference_lines={"steps_mean": _deep_references(run_dir.name)}))
    return out


def main(argv: list[str] | None = None) -> list[Path]:
    results = resolve_output_directory("results")
    paths = replot_tabular(results) + replot_ablation(results) + replot_deep(results)
    for path in paths:
        print(f"Hình: {path}")
    return paths


if __name__ == "__main__":
    main()
