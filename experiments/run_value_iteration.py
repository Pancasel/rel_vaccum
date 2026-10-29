"""T3.3: tính đáp án tối ưu bằng Value Iteration và so với baseline từ mọi ô xuất phát.

Ví dụ:
    python -m experiments.run_value_iteration
    python -m experiments.run_value_iteration --maps empty_3x3 --quick
"""

from __future__ import annotations

import argparse
import json
import time

from evaluation.optimal_reference import can_compute_exact_optimum
from experiments.common import resolve_out

import pandas as pd  # noqa: E402

from agents.baselines import GreedyNearestDirtyCellAgent, ZigzagSweepAgent  # noqa: E402
from environment.room_layouts import FIXED_ROOM_LAYOUTS, list_free_cells, get_room_layout  # noqa: E402
from environment.vacuum_cleaning_env import VacuumCleaningEnv  # noqa: E402
from evaluation.episode_runner import run_single_episode  # noqa: E402
from experiments.plots import df_to_markdown_table  # noqa: E402
from agents.tabular.state_transition_graph import build_state_transition_graph, minimum_steps_to_clean_all  # noqa: E402
from agents.tabular.value_iteration import follow_optimal_policy, save_value_iteration_result, run_value_iteration  # noqa: E402
from project_utils.result_files import save_config_with_metadata, save_dataframe_csv  # noqa: E402

DEFAULT_MAPS = ["empty_3x3", "obstacle_4x4", "empty_4x4"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Value Iteration: đáp án tối ưu trên lưới nhỏ.")
    parser.add_argument("--maps", nargs="+", default=DEFAULT_MAPS, choices=sorted(FIXED_ROOM_LAYOUTS))
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--out", type=str, default="results/vi")
    parser.add_argument("--quick", action="store_true", help="chỉ chạy empty_3x3")
    return parser


def solve_map(map_name: str, gamma: float) -> tuple[pd.DataFrame, dict, object]:
    """VI trên một bản đồ; bảng so sánh từng ô xuất phát với greedy_bfs và zigzag."""
    grid = get_room_layout(map_name)
    if not can_compute_exact_optimum(grid):
        raise ValueError(f"Bản đồ {map_name} quá lớn để tính tối ưu chính xác.")
    t0 = time.monotonic()
    graph = build_state_transition_graph(grid)
    result = run_value_iteration(graph, gamma=gamma)
    runtime = time.monotonic() - t0
    env = VacuumCleaningEnv(grid)
    rows = []
    for start in list_free_cells(grid):
        rollout = follow_optimal_policy(result, start, max_steps=env.max_steps)
        greedy = run_single_episode(GreedyNearestDirtyCellAgent(), env, seed=0, gamma=gamma, reset_options={"start": start})
        zigzag = run_single_episode(ZigzagSweepAgent(), env, seed=0, gamma=gamma, reset_options={"start": start})
        rows.append({
            "start_row": start[0], "start_col": start[1],
            "v_star": result.optimal_value_of_start(start),
            "min_steps": minimum_steps_to_clean_all(graph, start),
            "vi_policy_steps": rollout["steps"],
            "greedy_bfs_steps": greedy["steps"],
            "zigzag_steps": zigzag["steps"],
            "greedy_bfs_return_disc": greedy["return_discounted"],
        })
    meta = {"map": map_name, "n_states": graph.state_count, "n_iterations": result.iteration_count,
            "runtime_s": round(runtime, 3), "gamma": gamma}
    return pd.DataFrame(rows), meta, result


def main(argv: list[str] | None = None) -> dict[str, pd.DataFrame]:
    args = build_parser().parse_args(argv)
    out = resolve_out(args.out)
    maps = ["empty_3x3"] if args.quick else args.maps
    tables: dict[str, pd.DataFrame] = {}
    for map_name in maps:
        table, meta, result = solve_map(map_name, args.gamma)
        save_dataframe_csv(table, out / f"{map_name}_summary.csv")
        (out / f"{map_name}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        save_value_iteration_result(result, out / f"{map_name}_vi.pkl")
        tables[map_name] = table
        print(f"\n## {map_name}: {meta['n_states']} trạng thái, {meta['n_iterations']} vòng lặp, "
              f"{meta['runtime_s']} s")
        print(df_to_markdown_table(table))
    save_config_with_metadata({"maps": maps, "gamma": args.gamma}, out / "config.json")
    return tables


if __name__ == "__main__":
    main()
