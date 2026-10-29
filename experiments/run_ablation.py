"""T5.2 (E1) và T7.1 (E2): thí nghiệm thay đổi một yếu tố thiết kế, giữ nguyên mọi thứ khác.

--exp markov : state CÓ / KHÔNG có bản đồ đã dọn (kiểm tra vai trò của tính Markov).
--exp reward : 4 thiết kế reward (sparse, dense, shaped, hackable) + GIF reward hacking
               + kiểm chứng lý thuyết shaping thế năng bằng Value Iteration.

Ví dụ:
    python -m experiments.run_ablation --exp markov --map obstacle_4x4 --workers 5
    python -m experiments.run_ablation --exp reward --map obstacle_4x4 --workers 5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from experiments.common import add_common_args, make_env_factory, resolve_out, run_parallel

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from env.maps import FIXED_MAPS, get_map  # noqa: E402
from env.render import record_episode  # noqa: E402
from env.vacuum_env import RewardConfig  # noqa: E402
from experiments.evaluate import summarize  # noqa: E402
from experiments.plots import df_to_markdown_table, plot_learning_curves, write_text  # noqa: E402
from experiments.run_tabular import train_one  # noqa: E402
from tabular.common import TabularAgent, TabularConfig  # noqa: E402
from tabular.state_space import build_state_graph  # noqa: E402
from tabular.value_iteration import value_iteration  # noqa: E402
from project_utils.result_files import save_config_with_metadata, save_dataframe_csv  # noqa: E402

START = (0, 0)
REWARD_CONFIGS = ["sparse", "dense", "shaped", "hackable"]
STATE_MODES = ["full", "position_only"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ablation: tính Markov của state (E1) và thiết kế reward (E2).")
    parser.add_argument("--exp", required=True, choices=["markov", "reward"])
    parser.add_argument("--map", default="obstacle_4x4", choices=sorted(FIXED_MAPS))
    parser.add_argument("--episodes", type=int, default=20_000)
    add_common_args(parser, "results/ablation")
    return parser


def run_variants(variants: dict[str, dict[str, Any]], map_name: str, episodes: int, seeds: list[int],
                 run_dir: Path, workers: int) -> list[dict[str, Any]]:
    """Huấn luyện Q-learning (xuất phát (0,0)) cho mỗi biến thể × seed; trả kết quả kèm tên biến thể."""
    tasks, names = [], []
    for name, variant in variants.items():
        for seed in seeds:
            cfg = TabularConfig(n_episodes=episodes, eval_every=max(1, episodes // 200), eval_episodes=1,
                                state_mode=variant.get("state_mode", "full"),
                                checkpoint_episodes=(), seed=seed)
            tasks.append({"algo": "q_learning", "map_name": map_name, "start": START,
                          "reward": variant.get("reward", "dense"), "cfg": cfg.__dict__.copy(),
                          "log_dir": str(run_dir / name / f"seed{seed}")})
            names.append(name)
    results = run_parallel(train_one, tasks, workers=workers)
    for name, res in zip(names, results):
        res["variant"] = name
    return results


def curves_and_summary(results: list[dict[str, Any]], out_fig: Path, title: str) -> pd.DataFrame:
    """Learning curve theo biến thể và bảng kết quả cuối (trung bình ± std qua seed)."""
    eval_df = pd.concat([r["eval"].assign(seed=r["seed"], variant=r["variant"]) for r in results], ignore_index=True)
    plot_learning_curves(eval_df, "episode", [("coverage_mean", "tỷ lệ đã dọn (tham lam)"),
                                              ("steps_mean", "số bước (tham lam)")], "variant", out_fig, title,
                         "episode")
    finals = pd.concat([r["final"].assign(seed=r["seed"], variant=r["variant"]) for r in results], ignore_index=True)
    return summarize(finals, ["variant"])[["variant", "n", "success_rate", "coverage_mean", "coverage_std",
                                           "steps_mean", "steps_std", "redundancy_mean"]]


def record_greedy_gif(log_dir: Path, map_name: str, reward: str, path: Path) -> dict[str, Any]:
    """GIF chính sách tham lam của agent đã học (seed 0) — để thấy tận mắt hành vi."""
    agent = TabularAgent.load(log_dir / "checkpoints" / "final.pkl")
    env = make_env_factory(map_name, start=START, reward=reward, render_mode="rgb_array")()
    return record_episode(agent, env, path, seed=0)


def count_back_and_forth(agent: TabularAgent, map_name: str, reward: str) -> int:
    """Số lần robot quay lại đúng ô của 2 bước trước (dấu hiệu vòng lặp "ra-vào")."""
    env = make_env_factory(map_name, start=START, reward=reward)()
    obs, info = env.reset(seed=0)
    positions = [info["position"]]
    done = False
    while not done:
        obs, _, terminated, truncated, info = env.step(agent.act(obs, info))
        positions.append(info["position"])
        done = terminated or truncated
    return sum(1 for t in range(2, len(positions)) if positions[t] == positions[t - 2] != positions[t - 1])


def shaping_policy_check(map_name: str, gamma: float = 0.99) -> dict[str, int]:
    """Kiểm chứng lý thuyết (Ng et al. 1999): shaping thế năng không đổi chính sách tối ưu.

    Đếm số trạng thái mà hành động tối ưu dưới reward shaped KHÔNG còn tối ưu dưới reward dense.
    """
    grid = get_map(map_name)
    dense = value_iteration(build_state_graph(grid, RewardConfig.dense()), gamma=gamma)
    shaped = value_iteration(build_state_graph(grid, RewardConfig.shaped(1.0)), gamma=gamma)
    assert dense.graph.states == shaped.graph.states  # cùng động lực học -> cùng thứ tự trạng thái
    non_terminal = ~dense.graph.is_terminal
    rows = np.arange(dense.graph.n_states)
    q_of_shaped_choice = dense.Q[rows, shaped.policy]
    differs = (q_of_shaped_choice < dense.Q.max(axis=1) - 1e-6) & non_terminal
    return {"n_states": int(non_terminal.sum()), "n_policy_differs": int(differs.sum())}


def main(argv: list[str] | None = None) -> pd.DataFrame:
    args = build_parser().parse_args(argv)
    out = resolve_out(args.out)
    seeds = args.seeds[:2] if args.quick else args.seeds
    episodes = max(50, args.episodes // 20) if args.quick else args.episodes
    run_dir = out / args.exp / args.map
    fig_dir = out / "figures"
    gif_dir = out / "gifs"
    if args.exp == "markov":
        variants = {mode: {"state_mode": mode} for mode in STATE_MODES}
        results = run_variants(variants, args.map, episodes, seeds, run_dir, args.workers)
        table = curves_and_summary(results, fig_dir / f"e1_markov_{args.map}.png",
                                   f"E1: state có / không có bản đồ đã dọn ({args.map}, xuất phát {START})")
        gif = record_greedy_gif(run_dir / "position_only" / f"seed{seeds[0]}", args.map, "dense",
                                gif_dir / f"e1_position_only_{args.map}.gif")
        extra = {"position_only_gif": gif}
    else:
        variants = {name: {"reward": name} for name in REWARD_CONFIGS}
        results = run_variants(variants, args.map, episodes, seeds, run_dir, args.workers)
        table = curves_and_summary(results, fig_dir / f"e2_reward_curves_{args.map}.png",
                                   f"E2: bốn thiết kế reward ({args.map}, xuất phát {START})")
        hack_dir = run_dir / "hackable" / f"seed{seeds[0]}"
        gif = record_greedy_gif(hack_dir, args.map, "hackable", gif_dir / f"e2_reward_hacking_{args.map}.gif")
        hack_agent = TabularAgent.load(hack_dir / "checkpoints" / "final.pkl")
        extra = {"hackable_gif": gif,
                 "hackable_back_and_forth_moves": count_back_and_forth(hack_agent, args.map, "hackable"),
                 "shaping_theory_check": shaping_policy_check(args.map)}
    save_dataframe_csv(table, run_dir / "summary.csv")
    report = df_to_markdown_table(table) + "\n\n" + "\n".join(f"- {k}: {v}" for k, v in extra.items())
    write_text(report, run_dir / "summary.md")
    save_config_with_metadata({"args": vars(args), "episodes": episodes, "seeds": seeds, "start": START, **extra},
                     run_dir / "config.json")
    print(report)
    return table


if __name__ == "__main__":
    main()
