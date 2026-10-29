"""T4.4: huấn luyện Q-learning dạng bảng trên nhiều seed, vẽ learning curve và hội tụ về V*.

Ví dụ:
    python -m experiments.run_tabular_q_learning --map obstacle_4x4 --start 0 0 --workers 5
    python -m experiments.run_tabular_q_learning --map obstacle_4x4 --state-mode position_only
    python -m experiments.run_tabular_q_learning --quick
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from experiments.shared.command_line_options import add_shared_options, parse_start_option, resolve_output_directory
from experiments.shared.environment_factories import make_environment_factory
from experiments.shared.parallel_execution import run_tasks_in_parallel
from experiments.shared.result_folder_naming import build_fixed_room_folder_name
from evaluation.optimal_reference import optimal_steps_for_each_start, optimal_value_of_start, reference_step_counts

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from environment.room_layouts import FIXED_ROOM_LAYOUTS, get_room_layout  # noqa: E402
from environment.reward_settings import RewardSettings  # noqa: E402
from evaluation.agent_evaluation import evaluate_agent  # noqa: E402
from evaluation.metric_summary import summarize_metrics  # noqa: E402
from experiments.plotting.markdown_tables import dataframe_to_markdown_table  # noqa: E402
from experiments.plotting.learning_curve_plots import plot_learning_curves, plot_q_convergence  # noqa: E402
from agents.tabular.tabular_agent_base import TabularAgentBase  # noqa: E402
from agents.tabular.tabular_settings import TabularSettings  # noqa: E402
from agents.tabular.q_learning_agent import QLearningAgent  # noqa: E402
from project_utils.result_files import save_config_with_metadata, save_dataframe_csv  # noqa: E402

ALGOS: dict[str, type[TabularAgentBase]] = {"q_learning": QLearningAgent}
FINAL_EVAL_EPISODES_RANDOM_START: int = 50


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Huấn luyện Q-learning dạng bảng.")
    parser.add_argument("--algos", nargs="+", default=["q_learning"], choices=list(ALGOS))
    parser.add_argument("--map", default="obstacle_4x4", choices=sorted(FIXED_ROOM_LAYOUTS))
    parser.add_argument("--start", nargs="+", default=["0", "0"], help="'random' hoặc R C (mặc định 0 0)")
    parser.add_argument("--episodes", type=int, default=50_000)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--state-mode", default="full", choices=["full", "position_only"])
    parser.add_argument("--reward", default="dense", choices=list(RewardSettings.PRESETS))
    parser.add_argument("--eval-every", type=int, default=None, help="mặc định: episodes/200")
    parser.add_argument("--eval-episodes", type=int, default=None, help="mặc định: 1 nếu start cố định, 10 nếu ngẫu nhiên")
    parser.add_argument("--time-budget-min", type=float, default=None)
    parser.add_argument("--checkpoints", type=int, nargs="*", default=[0, 1000, 10000],
                        help="các episode lưu bảng Q (cho demo 'robot học dần')")
    add_shared_options(parser, "results/tabular")
    return parser


def train_one(algo: str, map_name: str, start: tuple[int, int] | None, reward: str, agent_settings: dict[str, Any],
              log_dir: str) -> dict[str, Any]:
    """Huấn luyện một (thuật toán, seed); trả log và kết quả đánh giá cuối (chạy được ở tiến trình con)."""
    factory = make_environment_factory(map_name, start=start, reward=reward)
    agent = ALGOS[algo](get_room_layout(map_name), TabularSettings(**agent_settings))
    train_log, eval_log = agent.train(factory, log_dir=log_dir)
    final_episode_count = 1 if start is not None else FINAL_EVAL_EPISODES_RANDOM_START
    final = evaluate_agent(agent, factory, final_episode_count, seed=1000 + agent_settings["seed"], gamma=agent_settings["gamma"])
    save_dataframe_csv(final, Path(log_dir) / f"seed{agent_settings['seed']}_final.csv")
    return {"algo": algo, "seed": agent_settings["seed"], "train": train_log, "eval": eval_log, "final": final}


def run_name(reward: str, state_mode: str) -> str:
    """Hậu tố tên hình: rỗng với cấu hình mặc định (dense, full) cho đúng tên trong kế hoạch."""
    return "" if (reward == "dense" and state_mode == "full") else f"_{reward}_{state_mode}"


def summarize_runs(results: list[dict[str, Any]], optimum: dict[tuple[int, int], int] | None) -> pd.DataFrame:
    """Bảng kết quả cuối: trung bình ± std QUA SEED của số bước, coverage, tỷ lệ thành công, gap tối ưu."""
    result_rows = []
    for seed_result in results:
        final = seed_result["final"].copy()
        row = {"algo": seed_result["algo"], "seed": seed_result["seed"], **summarize_metrics(final).iloc[0].to_dict()}
        if optimum is not None:
            done = final[final["terminated"]]
            optimal_steps = np.array([optimum[(start_row, start_column)] for start_row, start_column in zip(done["start_row"], done["start_col"])])
            row["gap_pct"] = float(np.mean((done["steps"] - optimal_steps) / optimal_steps) * 100) if len(done) else float("nan")
        result_rows.append(row)
    per_seed = pd.DataFrame(result_rows)
    metrics = ["coverage_mean", "steps_mean", "success_rate", "redundancy_mean"] + (
        ["gap_pct"] if optimum is not None else [])
    grouped = per_seed.groupby("algo")[metrics]
    table = grouped.mean().add_suffix("_avg").join(grouped.std(ddof=1).fillna(0.0).add_suffix("_sd"))
    table.insert(0, "n_seeds", per_seed.groupby("algo").size())
    return table.reset_index()


def make_figures(results: list[dict[str, Any]], parsed_arguments: argparse.Namespace, start: tuple[int, int] | None,
                 out_fig: Path) -> list[Path]:
    """Learning curve (coverage, số bước của chính sách tham lam) và hình hội tụ max Q(s0) về V*."""
    grid = get_room_layout(parsed_arguments.map)
    evaluation_table = pd.concat([seed_result["eval"].assign(seed=seed_result["seed"], algo=seed_result["algo"]) for seed_result in results], ignore_index=True)
    reference_steps = reference_step_counts(grid, start)
    tag = run_name(parsed_arguments.reward, parsed_arguments.state_mode)
    name = build_fixed_room_folder_name(parsed_arguments.map, start)
    start_txt = f"xuất phát {start}" if start else "xuất phát ngẫu nhiên"
    paths = [plot_learning_curves(
        evaluation_table, "episode", [("coverage_mean", "tỷ lệ đã dọn (chính sách tham lam)"),
                             ("steps_mean", "số bước (chính sách tham lam)")], "algo",
        out_fig / f"tabular_learning_curve_{name}{tag}.png",
        f"{', '.join(parsed_arguments.algos)} trên {parsed_arguments.map} ({start_txt}, reward {parsed_arguments.reward}, state {parsed_arguments.state_mode})", "episode",
        reference_lines={"steps_mean": reference_steps})]
    if start is not None and parsed_arguments.state_mode == "full":
        optimal_start_value = optimal_value_of_start(grid, start, RewardSettings.from_name(parsed_arguments.reward), parsed_arguments.gamma)
        if optimal_start_value is not None:
            training_table = pd.concat([seed_result["train"].assign(seed=seed_result["seed"], algo=seed_result["algo"]) for seed_result in results],
                                 ignore_index=True)
            paths.append(plot_q_convergence(training_table, optimal_start_value, out_fig / f"q_convergence_{name}{tag}.png",
                                            smoothing_window=max(1, parsed_arguments.episodes // 400)))
    return paths


def main(argv: list[str] | None = None) -> pd.DataFrame:
    parsed_arguments = build_parser().parse_args(argv)
    out = resolve_output_directory(parsed_arguments.out)
    start = parse_start_option(parsed_arguments.start)
    seeds = parsed_arguments.seeds[:2] if parsed_arguments.quick else parsed_arguments.seeds
    episodes = max(20, parsed_arguments.episodes // 10) if parsed_arguments.quick else parsed_arguments.episodes
    evaluate_every_episodes = parsed_arguments.eval_every or max(1, episodes // 200)
    evaluation_episode_count = parsed_arguments.eval_episodes or (1 if start is not None else 10)
    run_dir = out / parsed_arguments.reward / parsed_arguments.state_mode / build_fixed_room_folder_name(parsed_arguments.map, start)
    tasks = []
    for algo in parsed_arguments.algos:
        for seed in seeds:
            agent_settings = TabularSettings(episode_count=episodes, alpha=parsed_arguments.alpha, gamma=parsed_arguments.gamma, epsilon_end=parsed_arguments.epsilon_end,
                                state_mode=parsed_arguments.state_mode, evaluate_every_episodes=evaluate_every_episodes, evaluation_episode_count=evaluation_episode_count,
                                time_budget_minutes=parsed_arguments.time_budget_min,
                                checkpoint_episodes=tuple(checkpoint_episode for checkpoint_episode in parsed_arguments.checkpoints if checkpoint_episode <= episodes), seed=seed)
            tasks.append({"algo": algo, "map_name": parsed_arguments.map, "start": start, "reward": parsed_arguments.reward,
                          "agent_settings": agent_settings.__dict__.copy(), "log_dir": str(run_dir / algo / f"seed{seed}")})
    results = run_tasks_in_parallel(train_one, tasks, workers=parsed_arguments.workers)

    optimum = optimal_steps_for_each_start(get_room_layout(parsed_arguments.map))
    table = summarize_runs(results, optimum)
    save_dataframe_csv(table, run_dir / "summary.csv")
    figures = make_figures(results, parsed_arguments, start, out / "figures")
    save_config_with_metadata({"args": vars(parsed_arguments), "episodes": episodes, "seeds": seeds, "eval_every": evaluate_every_episodes,
                      "eval_episodes": evaluation_episode_count, "start": start}, run_dir / "config.json")
    print(dataframe_to_markdown_table(table))
    for path in figures:
        print(f"Hình: {path}")
    return table


if __name__ == "__main__":
    main()
