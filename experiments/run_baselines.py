"""T3.2: chạy các baseline trên các bản đồ cố định, ghi CSV và bảng tóm tắt.

Ví dụ:
    python -m experiments.run_baselines
    python -m experiments.run_baselines --maps furniture_8x8 --agents zigzag greedy_bfs --quick
"""

from __future__ import annotations

import argparse

from experiments.shared.command_line_options import add_shared_options, parse_start_option, resolve_output_directory
from experiments.shared.environment_factories import make_environment_factory

import pandas as pd  # noqa: E402

from agents.baselines import BASELINE_AGENT_CLASSES, create_baseline_agent  # noqa: E402
from environment.room_layouts import FIXED_ROOM_LAYOUTS  # noqa: E402
from evaluation.agent_evaluation import evaluate_agent  # noqa: E402
from evaluation.metric_summary import summarize_metrics  # noqa: E402
from experiments.plotting.markdown_tables import dataframe_to_markdown_table  # noqa: E402
from project_utils.result_files import save_config_with_metadata, save_dataframe_csv  # noqa: E402

DEFAULT_MAPS = ["empty_3x3", "obstacle_4x4", "furniture_8x8"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chạy baseline (random_walk, zigzag, greedy_bfs).")
    parser.add_argument("--maps", nargs="+", default=DEFAULT_MAPS, choices=sorted(FIXED_ROOM_LAYOUTS))
    parser.add_argument("--agents", nargs="+", default=list(BASELINE_AGENT_CLASSES), choices=list(BASELINE_AGENT_CLASSES))
    parser.add_argument("--episodes", type=int, default=20, help="số episode mỗi seed")
    parser.add_argument("--start", nargs="+", default=["random"], help="'random' hoặc R C")
    add_shared_options(parser, "results/baselines")
    return parser


def run(parsed_arguments: argparse.Namespace) -> pd.DataFrame:
    """Chạy toàn bộ tổ hợp map × agent × seed, trả bảng tóm tắt."""
    out = resolve_output_directory(parsed_arguments.out)
    seeds = parsed_arguments.seeds[:1] if parsed_arguments.quick else parsed_arguments.seeds
    episodes = min(parsed_arguments.episodes, 3) if parsed_arguments.quick else parsed_arguments.episodes
    start = parse_start_option(parsed_arguments.start)
    all_rows = []
    for map_name in parsed_arguments.maps:
        factory = make_environment_factory(map_name, start=start)
        for agent_name in parsed_arguments.agents:
            for seed in seeds:
                episode_table = evaluate_agent(create_baseline_agent(agent_name, seed=seed), factory, episodes, seed=seed)
                save_dataframe_csv(episode_table, out / map_name / agent_name / f"seed{seed}.csv")
                episode_table.insert(0, "seed", seed)
                episode_table.insert(0, "agent", agent_name)
                episode_table.insert(0, "map", map_name)
                all_rows.append(episode_table)
    full = pd.concat(all_rows, ignore_index=True)
    summary = summarize_metrics(full, ["map", "agent"])
    save_dataframe_csv(summary, out / "summary.csv")
    save_config_with_metadata({"maps": parsed_arguments.maps, "agents": parsed_arguments.agents, "seeds": seeds, "episodes": episodes,
                      "start": start}, out / "config.json")
    return summary


def main(argv: list[str] | None = None) -> pd.DataFrame:
    parsed_arguments = build_parser().parse_args(argv)
    summary = run(parsed_arguments)
    summary_columns = ["map", "agent", "n", "success_rate", "coverage_mean", "steps_mean", "steps_std",
            "redundancy_mean", "collisions_mean"]
    print(dataframe_to_markdown_table(summary[summary_columns]))
    return summary


if __name__ == "__main__":
    main()
