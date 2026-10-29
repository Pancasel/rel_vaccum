"""T3.2: chạy các baseline trên các bản đồ cố định, ghi CSV và bảng tóm tắt.

Ví dụ:
    python -m experiments.run_baselines
    python -m experiments.run_baselines --maps furniture_8x8 --agents zigzag greedy_bfs --quick
"""

from __future__ import annotations

import argparse

from experiments.common import add_common_args, make_env_factory, parse_start, resolve_out

import pandas as pd  # noqa: E402

from agents.baselines import BASELINE_AGENT_CLASSES, create_baseline_agent  # noqa: E402
from environment.room_layouts import FIXED_ROOM_LAYOUTS  # noqa: E402
from evaluation.agent_evaluation import evaluate_agent  # noqa: E402
from evaluation.metric_summary import summarize_metrics  # noqa: E402
from experiments.plots import df_to_markdown_table  # noqa: E402
from project_utils.result_files import save_config_with_metadata, save_dataframe_csv  # noqa: E402

DEFAULT_MAPS = ["empty_3x3", "obstacle_4x4", "furniture_8x8"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chạy baseline (random_walk, zigzag, greedy_bfs).")
    parser.add_argument("--maps", nargs="+", default=DEFAULT_MAPS, choices=sorted(FIXED_ROOM_LAYOUTS))
    parser.add_argument("--agents", nargs="+", default=list(BASELINE_AGENT_CLASSES), choices=list(BASELINE_AGENT_CLASSES))
    parser.add_argument("--episodes", type=int, default=20, help="số episode mỗi seed")
    parser.add_argument("--start", nargs="+", default=["random"], help="'random' hoặc R C")
    add_common_args(parser, "results/baselines")
    return parser


def run(args: argparse.Namespace) -> pd.DataFrame:
    """Chạy toàn bộ tổ hợp map × agent × seed, trả bảng tóm tắt."""
    out = resolve_out(args.out)
    seeds = args.seeds[:1] if args.quick else args.seeds
    episodes = min(args.episodes, 3) if args.quick else args.episodes
    start = parse_start(args.start)
    all_rows = []
    for map_name in args.maps:
        factory = make_env_factory(map_name, start=start)
        for agent_name in args.agents:
            for seed in seeds:
                df = evaluate_agent(create_baseline_agent(agent_name, seed=seed), factory, episodes, seed=seed)
                save_dataframe_csv(df, out / map_name / agent_name / f"seed{seed}.csv")
                df.insert(0, "seed", seed)
                df.insert(0, "agent", agent_name)
                df.insert(0, "map", map_name)
                all_rows.append(df)
    full = pd.concat(all_rows, ignore_index=True)
    summary = summarize_metrics(full, ["map", "agent"])
    save_dataframe_csv(summary, out / "summary.csv")
    save_config_with_metadata({"maps": args.maps, "agents": args.agents, "seeds": seeds, "episodes": episodes,
                      "start": start}, out / "config.json")
    return summary


def main(argv: list[str] | None = None) -> pd.DataFrame:
    args = build_parser().parse_args(argv)
    summary = run(args)
    cols = ["map", "agent", "n", "success_rate", "coverage_mean", "steps_mean", "steps_std",
            "redundancy_mean", "collisions_mean"]
    print(df_to_markdown_table(summary[cols]))
    return summary


if __name__ == "__main__":
    main()
