"""T5.1 (E7) + T5.3: giới hạn của Q-learning dạng bảng.

1. Đếm số trạng thái đạt tới được theo kích thước lưới (thang log) và so với cận trên |F|·2^|F|.
2. Chạy Q-learning có giới hạn thời gian trên lưới 3x3 -> 6x6 để thấy bảng Q phình to và
   chính sách không kịp học khi lưới lớn — động cơ chuyển sang DQN.

Ví dụ:
    python -m experiments.run_state_explosion --workers 4
    python -m experiments.run_state_explosion --quick
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

from experiments.common import make_env_factory, resolve_out, run_parallel
from evaluation.optimal_reference import optimal_steps_for_each_start

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from environment.room_layouts import get_room_layout, parse_room_layout  # noqa: E402
from evaluation.agent_evaluation import evaluate_agent  # noqa: E402
from experiments.plots import df_to_markdown_table, plot_state_explosion, plot_tabular_scaling  # noqa: E402
from agents.tabular.tabular_settings import TabularSettings  # noqa: E402
from agents.tabular.q_learning_agent import QLearningAgent  # noqa: E402
from agents.tabular.reachable_state_counter import count_reachable_states  # noqa: E402
from project_utils.result_files import save_config_with_metadata, save_dataframe_csv  # noqa: E402

COUNT_SIZES = [(2, 2), (2, 3), (3, 3), (3, 4), (4, 4), (4, 5)]
BOUND_ONLY_SIZES = [(5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10)]
SCALING_MAPS = ["empty_3x3", "empty_4x4", "empty_5x5", "empty_6x6"]
BYTES_PER_Q_STATE = 4 * 8  # bảng Q float64, 4 hành động


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bùng nổ trạng thái (E7) và giới hạn của tabular (T5.3).")
    parser.add_argument("--time-budget-min", type=float, default=15.0, help="thời gian huấn luyện mỗi lưới")
    parser.add_argument("--count-time-limit-s", type=float, default=600.0)
    parser.add_argument("--skip-scaling", action="store_true", help="chỉ đếm trạng thái, bỏ phần huấn luyện")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    return parser


def count_states(sizes: list[tuple[int, int]], bound_only: list[tuple[int, int]], time_limit_s: float
                 ) -> pd.DataFrame:
    """Bảng số trạng thái: đếm thực tế (BFS) cho lưới nhỏ, chỉ cận trên cho lưới lớn."""
    rows = []
    for h, w in sizes + bound_only:
        free_cell_count = h * w
        reachable, elapsed = None, None
        if (h, w) in sizes:
            t0 = time.monotonic()
            reachable = count_reachable_states(parse_room_layout(["." * w] * h), time_limit_seconds=time_limit_s)
            elapsed = round(time.monotonic() - t0, 3)
        states_for_memory = reachable if reachable is not None else free_cell_count * 2**free_cell_count
        rows.append({"grid": f"{h}x{w}", "n_free": free_cell_count, "upper_bound": float(free_cell_count * 2**free_cell_count),
                     "reachable": float(reachable) if reachable is not None else np.nan, "count_time_s": elapsed,
                     "q_table_mb": states_for_memory * BYTES_PER_Q_STATE / 1e6})
    return pd.DataFrame(rows)


def _q_table_bytes(agent: QLearningAgent) -> int:
    """Ước lượng bộ nhớ thực của bảng Q dạng dict Python (khóa tuple + mảng numpy + ô băm)."""
    if not agent.q_table:
        return 0
    key, row = next(iter(agent.q_table.items()))
    per_entry = sys.getsizeof(key) + sum(sys.getsizeof(k) for k in key) + sys.getsizeof(row) + 2 * 8 * 3
    return per_entry * len(agent.q_table) + sys.getsizeof(agent.q_table)


def train_with_budget(map_name: str, budget_min: float, seed: int) -> dict[str, Any]:
    """Huấn luyện Q-learning (xuất phát (0,0)) trong một khoảng thời gian cố định."""
    factory = make_env_factory(map_name, start=(0, 0))
    cfg = TabularSettings(episode_count=10_000_000, time_budget_minutes=budget_min, evaluate_every_episodes=200, evaluation_episode_count=1,
                        epsilon_decay_fraction=0.01, seed=seed)
    # Không biết trước số episode chạy được trong ngân sách: ε giảm trong 100.000 episode đầu
    # (1% của 10 triệu) rồi giữ 0.05; lưới lớn thường còn đang khám phá khi hết giờ.
    agent = QLearningAgent(get_room_layout(map_name), cfg)
    train_log, eval_log = agent.train(factory)
    final = evaluate_agent(agent, factory, 1, seed=seed)
    return {"map": map_name, "train": train_log, "eval": eval_log, "final": final.iloc[0].to_dict(),
            "q_bytes": _q_table_bytes(agent)}


def scaling_table(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for res in results:
        optimum = optimal_steps_for_each_start(get_room_layout(res["map"]))
        rows.append({
            "map": res["map"], "n_free": int((get_room_layout(res["map"]) == 0).sum()),
            "episodes_done": len(res["train"]), "minutes": res["train"]["wall_time_s"].iloc[-1] / 60,
            "n_states_visited": int(res["train"]["n_states_visited"].iloc[-1]),
            "final_coverage": res["final"]["coverage"], "final_steps": res["final"]["steps"],
            "min_steps": optimum[(0, 0)] if optimum is not None else np.nan,
            "q_table_mb_est": res["q_bytes"] / 1e6,
        })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> dict[str, pd.DataFrame]:
    args = build_parser().parse_args(argv)
    out = resolve_out(args.out)
    sizes = COUNT_SIZES[:3] if args.quick else COUNT_SIZES
    counts = count_states(sizes, BOUND_ONLY_SIZES, args.count_time_limit_s)
    save_dataframe_csv(counts, out / "state_explosion.csv")
    plot_state_explosion(counts, out / "figures" / "state_explosion.png")
    print(df_to_markdown_table(counts, float_fmt=".3g"))
    tables = {"counts": counts}
    if not args.skip_scaling:
        budget = 0.1 if args.quick else args.time_budget_min
        maps = SCALING_MAPS[:2] if args.quick else SCALING_MAPS
        results = run_parallel(train_with_budget, [{"map_name": m, "budget_min": budget, "seed": args.seed}
                                                   for m in maps], workers=args.workers)
        scaling = scaling_table(results)
        save_dataframe_csv(scaling, out / "tabular" / "scaling.csv")
        train_df = pd.concat([r["train"].assign(map=r["map"]) for r in results], ignore_index=True)
        eval_df = pd.concat([r["eval"].assign(map=r["map"]) for r in results], ignore_index=True)
        plot_tabular_scaling(train_df, eval_df, out / "figures" / "tabular_scaling.png")
        print(df_to_markdown_table(scaling, float_fmt=".3g"))
        tables["scaling"] = scaling
    save_config_with_metadata({"args": vars(args), "count_sizes": sizes}, out / "state_explosion_config.json")
    return tables


if __name__ == "__main__":
    main()
