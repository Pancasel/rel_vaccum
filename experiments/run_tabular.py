"""T4.4: huấn luyện Q-learning dạng bảng trên nhiều seed, vẽ learning curve và hội tụ về V*.

Ví dụ:
    python -m experiments.run_tabular --map obstacle_4x4 --start 0 0 --workers 5
    python -m experiments.run_tabular --map obstacle_4x4 --state-mode position_only
    python -m experiments.run_tabular --quick
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from experiments.common import (add_common_args, make_env_factory, optimal_steps_by_start, optimal_value,
                                parse_start, reference_steps, resolve_out, run_parallel)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from env.maps import FIXED_MAPS, get_map  # noqa: E402
from env.vacuum_env import RewardConfig  # noqa: E402
from experiments.evaluate import evaluate, summarize  # noqa: E402
from experiments.plots import df_to_markdown_table, plot_learning_curves, plot_q_convergence  # noqa: E402
from tabular.common import TabularAgent, TabularConfig  # noqa: E402
from tabular.q_learning import QLearningAgent  # noqa: E402
from utils.io import save_config_json, save_csv  # noqa: E402

ALGOS: dict[str, type[TabularAgent]] = {"q_learning": QLearningAgent}
FINAL_EVAL_EPISODES_RANDOM_START: int = 50


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Huấn luyện Q-learning dạng bảng.")
    parser.add_argument("--algos", nargs="+", default=["q_learning"], choices=list(ALGOS))
    parser.add_argument("--map", default="obstacle_4x4", choices=sorted(FIXED_MAPS))
    parser.add_argument("--start", nargs="+", default=["0", "0"], help="'random' hoặc R C (mặc định 0 0)")
    parser.add_argument("--episodes", type=int, default=50_000)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--state-mode", default="full", choices=["full", "position_only"])
    parser.add_argument("--reward", default="dense", choices=list(RewardConfig.PRESETS))
    parser.add_argument("--eval-every", type=int, default=None, help="mặc định: episodes/200")
    parser.add_argument("--eval-episodes", type=int, default=None, help="mặc định: 1 nếu start cố định, 10 nếu ngẫu nhiên")
    parser.add_argument("--time-budget-min", type=float, default=None)
    parser.add_argument("--checkpoints", type=int, nargs="*", default=[0, 1000, 10000],
                        help="các episode lưu bảng Q (cho demo 'robot học dần')")
    add_common_args(parser, "results/tabular")
    return parser


def train_one(algo: str, map_name: str, start: tuple[int, int] | None, reward: str, cfg: dict[str, Any],
              log_dir: str) -> dict[str, Any]:
    """Huấn luyện một (thuật toán, seed); trả log và kết quả đánh giá cuối (chạy được ở tiến trình con)."""
    factory = make_env_factory(map_name, start=start, reward=reward)
    agent = ALGOS[algo](get_map(map_name), TabularConfig(**cfg))
    train_log, eval_log = agent.train(factory, log_dir=log_dir)
    n_final = 1 if start is not None else FINAL_EVAL_EPISODES_RANDOM_START
    final = evaluate(agent, factory, n_final, seed=1000 + cfg["seed"], gamma=cfg["gamma"])
    save_csv(final, Path(log_dir) / f"seed{cfg['seed']}_final.csv")
    return {"algo": algo, "seed": cfg["seed"], "train": train_log, "eval": eval_log, "final": final}


def map_tag(map_name: str, start: tuple[int, int] | None) -> str:
    """Tên bản đồ kèm điểm xuất phát ('obstacle_4x4_s00'; xuất phát ngẫu nhiên giữ nguyên tên).

    Vì sao: bảng Q gắn với điểm xuất phát; thiếu hậu tố thì lượt xuất phát cố định và lượt
    xuất phát ngẫu nhiên ghi đè kết quả của nhau (cùng quy ước với run_deep).
    """
    return map_name + (f"_s{start[0]}{start[1]}" if start is not None else "")


def run_name(reward: str, state_mode: str) -> str:
    """Hậu tố tên hình: rỗng với cấu hình mặc định (dense, full) cho đúng tên trong kế hoạch."""
    return "" if (reward == "dense" and state_mode == "full") else f"_{reward}_{state_mode}"


def summarize_runs(results: list[dict[str, Any]], optimum: dict[tuple[int, int], int] | None) -> pd.DataFrame:
    """Bảng kết quả cuối: trung bình ± std QUA SEED của số bước, coverage, tỷ lệ thành công, gap tối ưu."""
    rows = []
    for res in results:
        final = res["final"].copy()
        row = {"algo": res["algo"], "seed": res["seed"], **summarize(final).iloc[0].to_dict()}
        if optimum is not None:
            done = final[final["terminated"]]
            opt = np.array([optimum[(r, c)] for r, c in zip(done["start_row"], done["start_col"])])
            row["gap_pct"] = float(np.mean((done["steps"] - opt) / opt) * 100) if len(done) else float("nan")
        rows.append(row)
    per_seed = pd.DataFrame(rows)
    metrics = ["coverage_mean", "steps_mean", "success_rate", "redundancy_mean"] + (
        ["gap_pct"] if optimum is not None else [])
    grouped = per_seed.groupby("algo")[metrics]
    table = grouped.mean().add_suffix("_avg").join(grouped.std(ddof=1).fillna(0.0).add_suffix("_sd"))
    table.insert(0, "n_seeds", per_seed.groupby("algo").size())
    return table.reset_index()


def make_figures(results: list[dict[str, Any]], args: argparse.Namespace, start: tuple[int, int] | None,
                 out_fig: Path) -> list[Path]:
    """Learning curve (coverage, số bước của chính sách tham lam) và hình hội tụ max Q(s0) về V*."""
    grid = get_map(args.map)
    eval_df = pd.concat([r["eval"].assign(seed=r["seed"], algo=r["algo"]) for r in results], ignore_index=True)
    refs = reference_steps(grid, start)
    tag = run_name(args.reward, args.state_mode)
    name = map_tag(args.map, start)
    start_txt = f"xuất phát {start}" if start else "xuất phát ngẫu nhiên"
    paths = [plot_learning_curves(
        eval_df, "episode", [("coverage_mean", "tỷ lệ đã dọn (chính sách tham lam)"),
                             ("steps_mean", "số bước (chính sách tham lam)")], "algo",
        out_fig / f"tabular_learning_curve_{name}{tag}.png",
        f"Q-learning trên {args.map} ({start_txt}, reward {args.reward}, state {args.state_mode})", "episode",
        hlines={"steps_mean": refs})]
    if start is not None and args.state_mode == "full":
        v_star = optimal_value(grid, start, RewardConfig.from_name(args.reward), args.gamma)
        if v_star is not None:
            train_df = pd.concat([r["train"].assign(seed=r["seed"], algo=r["algo"]) for r in results],
                                 ignore_index=True)
            paths.append(plot_q_convergence(train_df, v_star, out_fig / f"q_convergence_{name}{tag}.png",
                                            smooth=max(1, args.episodes // 400)))
    return paths


def main(argv: list[str] | None = None) -> pd.DataFrame:
    args = build_parser().parse_args(argv)
    out = resolve_out(args.out)
    start = parse_start(args.start)
    seeds = args.seeds[:2] if args.quick else args.seeds
    episodes = max(20, args.episodes // 10) if args.quick else args.episodes
    eval_every = args.eval_every or max(1, episodes // 200)
    eval_episodes = args.eval_episodes or (1 if start is not None else 10)
    run_dir = out / args.reward / args.state_mode / map_tag(args.map, start)
    tasks = []
    for algo in args.algos:
        for seed in seeds:
            cfg = TabularConfig(n_episodes=episodes, alpha=args.alpha, gamma=args.gamma, epsilon_end=args.epsilon_end,
                                state_mode=args.state_mode, eval_every=eval_every, eval_episodes=eval_episodes,
                                time_budget_min=args.time_budget_min,
                                checkpoint_episodes=tuple(e for e in args.checkpoints if e <= episodes), seed=seed)
            tasks.append({"algo": algo, "map_name": args.map, "start": start, "reward": args.reward,
                          "cfg": cfg.__dict__.copy(), "log_dir": str(run_dir / algo / f"seed{seed}")})
    results = run_parallel(train_one, tasks, workers=args.workers)

    optimum = optimal_steps_by_start(get_map(args.map))
    table = summarize_runs(results, optimum)
    save_csv(table, run_dir / "summary.csv")
    figures = make_figures(results, args, start, out / "figures")
    save_config_json({"args": vars(args), "episodes": episodes, "seeds": seeds, "eval_every": eval_every,
                      "eval_episodes": eval_episodes, "start": start}, run_dir / "config.json")
    print(df_to_markdown_table(table))
    for path in figures:
        print(f"Hình: {path}")
    return table


if __name__ == "__main__":
    main()
