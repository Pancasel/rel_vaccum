"""T5.2 (E1) và T7.1 (E2): thí nghiệm thay đổi một yếu tố thiết kế, giữ nguyên mọi thứ khác.

--exp markov : state CÓ / KHÔNG có bản đồ đã dọn (kiểm tra vai trò của tính Markov).
--exp reward : 4 thiết kế reward (sparse, dense, shaped, hackable) + GIF reward hacking
               + kiểm chứng lý thuyết shaping thế năng bằng Value Iteration.

Ví dụ:
    python -m experiments.run_markov_and_reward_ablation --exp markov --map obstacle_4x4 --workers 5
    python -m experiments.run_markov_and_reward_ablation --exp reward --map obstacle_4x4 --workers 5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from experiments.shared.command_line_options import add_shared_options, resolve_output_directory
from experiments.shared.environment_factories import make_environment_factory
from experiments.shared.parallel_execution import run_tasks_in_parallel

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from environment.room_layouts import FIXED_ROOM_LAYOUTS, get_room_layout  # noqa: E402
from environment.rendering.gif_recording import record_episode_as_gif  # noqa: E402
from environment.reward_settings import RewardSettings  # noqa: E402
from evaluation.metric_summary import summarize_metrics  # noqa: E402
from experiments.plotting.markdown_tables import dataframe_to_markdown_table, write_text_file  # noqa: E402
from experiments.plotting.learning_curve_plots import plot_learning_curves  # noqa: E402
from experiments.run_tabular_q_learning import train_one  # noqa: E402
from agents.tabular.tabular_agent_base import TabularAgentBase  # noqa: E402
from agents.tabular.tabular_settings import TabularSettings  # noqa: E402
from agents.tabular.state_transition_graph import build_state_transition_graph  # noqa: E402
from agents.tabular.value_iteration import run_value_iteration  # noqa: E402
from project_utils.result_files import save_config_with_metadata, save_dataframe_csv  # noqa: E402

START = (0, 0)
REWARD_CONFIGS = ["sparse", "dense", "shaped", "hackable"]
STATE_MODES = ["full", "position_only"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ablation: tính Markov của state (E1) và thiết kế reward (E2).")
    parser.add_argument("--exp", required=True, choices=["markov", "reward"])
    parser.add_argument("--map", default="obstacle_4x4", choices=sorted(FIXED_ROOM_LAYOUTS))
    parser.add_argument("--episodes", type=int, default=20_000)
    add_shared_options(parser, "results/ablation")
    return parser


def run_variants(variants: dict[str, dict[str, Any]], map_name: str, episodes: int, seeds: list[int],
                 run_dir: Path, workers: int) -> list[dict[str, Any]]:
    """Huấn luyện Q-learning (xuất phát (0,0)) cho mỗi biến thể × seed; trả kết quả kèm tên biến thể."""
    tasks, names = [], []
    for name, variant in variants.items():
        for seed in seeds:
            agent_settings = TabularSettings(episode_count=episodes, evaluate_every_episodes=max(1, episodes // 200), evaluation_episode_count=1,
                                state_mode=variant.get("state_mode", "full"),
                                checkpoint_episodes=(), seed=seed)
            tasks.append({"algo": "q_learning", "map_name": map_name, "start": START,
                          "reward": variant.get("reward", "dense"), "agent_settings": agent_settings.__dict__.copy(),
                          "log_dir": str(run_dir / name / f"seed{seed}")})
            names.append(name)
    results = run_tasks_in_parallel(train_one, tasks, workers=workers)
    for name, variant_result in zip(names, results):
        variant_result["variant"] = name
    return results


def curves_and_summary(results: list[dict[str, Any]], out_fig: Path, title: str) -> pd.DataFrame:
    """Learning curve theo biến thể và bảng kết quả cuối (trung bình ± std qua seed)."""
    evaluation_table = pd.concat([variant_result["eval"].assign(seed=variant_result["seed"], variant=variant_result["variant"]) for variant_result in results], ignore_index=True)
    plot_learning_curves(evaluation_table, "episode", [("coverage_mean", "tỷ lệ đã dọn (tham lam)"),
                                              ("steps_mean", "số bước (tham lam)")], "variant", out_fig, title,
                         "episode")
    finals = pd.concat([variant_result["final"].assign(seed=variant_result["seed"], variant=variant_result["variant"]) for variant_result in results], ignore_index=True)
    return summarize_metrics(finals, ["variant"])[["variant", "n", "success_rate", "coverage_mean", "coverage_std",
                                           "steps_mean", "steps_std", "redundancy_mean"]]


def record_greedy_gif(log_dir: Path, map_name: str, reward: str, path: Path) -> dict[str, Any]:
    """GIF chính sách tham lam của agent đã học (seed 0) — để thấy tận mắt hành vi."""
    agent = TabularAgentBase.load(log_dir / "checkpoints" / "final.pkl")
    env = make_environment_factory(map_name, start=START, reward=reward, render_mode="rgb_array")()
    return record_episode_as_gif(agent, env, path, seed=0)


def count_back_and_forth(agent: TabularAgentBase, map_name: str, reward: str) -> int:
    """Số lần robot quay lại đúng ô của 2 bước trước (dấu hiệu vòng lặp "ra-vào")."""
    env = make_environment_factory(map_name, start=START, reward=reward)()
    observation, info = env.reset(seed=0)
    positions = [info["position"]]
    done = False
    while not done:
        observation, _, terminated, truncated, info = env.step(agent.act(observation, info))
        positions.append(info["position"])
        done = terminated or truncated
    return sum(1 for step_index in range(2, len(positions)) if positions[step_index] == positions[step_index - 2] != positions[step_index - 1])


def shaping_policy_check(map_name: str, gamma: float = 0.99) -> dict[str, int]:
    """Kiểm chứng lý thuyết (Ng et al. 1999): shaping thế năng không đổi chính sách tối ưu.

    Đếm số trạng thái mà hành động tối ưu dưới reward shaped KHÔNG còn tối ưu dưới reward dense.
    """
    grid = get_room_layout(map_name)
    dense = run_value_iteration(build_state_transition_graph(grid, RewardSettings.dense()), gamma=gamma)
    shaped = run_value_iteration(build_state_transition_graph(grid, RewardSettings.shaped(1.0)), gamma=gamma)
    assert dense.graph.state_keys == shaped.graph.state_keys  # cùng động lực học -> cùng thứ tự trạng thái
    non_terminal = ~dense.graph.is_terminal_state
    state_indices = np.arange(dense.graph.state_count)
    q_of_shaped_choice = dense.action_values[state_indices, shaped.policy]
    differs = (q_of_shaped_choice < dense.action_values.max(axis=1) - 1e-6) & non_terminal
    return {"n_states": int(non_terminal.sum()), "n_policy_differs": int(differs.sum())}


def main(argv: list[str] | None = None) -> pd.DataFrame:
    parsed_arguments = build_parser().parse_args(argv)
    out = resolve_output_directory(parsed_arguments.out)
    seeds = parsed_arguments.seeds[:2] if parsed_arguments.quick else parsed_arguments.seeds
    episodes = max(50, parsed_arguments.episodes // 20) if parsed_arguments.quick else parsed_arguments.episodes
    run_dir = out / parsed_arguments.exp / parsed_arguments.map
    fig_dir = out / "figures"
    gif_dir = out / "gifs"
    if parsed_arguments.exp == "markov":
        variants = {mode: {"state_mode": mode} for mode in STATE_MODES}
        results = run_variants(variants, parsed_arguments.map, episodes, seeds, run_dir, parsed_arguments.workers)
        table = curves_and_summary(results, fig_dir / f"e1_markov_{parsed_arguments.map}.png",
                                   f"E1: state có / không có bản đồ đã dọn ({parsed_arguments.map}, xuất phát {START})")
        gif = record_greedy_gif(run_dir / "position_only" / f"seed{seeds[0]}", parsed_arguments.map, "dense",
                                gif_dir / f"e1_position_only_{parsed_arguments.map}.gif")
        extra = {"position_only_gif": gif}
    else:
        variants = {name: {"reward": name} for name in REWARD_CONFIGS}
        results = run_variants(variants, parsed_arguments.map, episodes, seeds, run_dir, parsed_arguments.workers)
        table = curves_and_summary(results, fig_dir / f"e2_reward_curves_{parsed_arguments.map}.png",
                                   f"E2: bốn thiết kế reward ({parsed_arguments.map}, xuất phát {START})")
        hack_dir = run_dir / "hackable" / f"seed{seeds[0]}"
        gif = record_greedy_gif(hack_dir, parsed_arguments.map, "hackable", gif_dir / f"e2_reward_hacking_{parsed_arguments.map}.gif")
        hack_agent = TabularAgentBase.load(hack_dir / "checkpoints" / "final.pkl")
        extra = {"hackable_gif": gif,
                 "hackable_back_and_forth_moves": count_back_and_forth(hack_agent, parsed_arguments.map, "hackable"),
                 "shaping_theory_check": shaping_policy_check(parsed_arguments.map)}
    save_dataframe_csv(table, run_dir / "summary.csv")
    report = dataframe_to_markdown_table(table) + "\n\n" + "\n".join(f"- {name}: {value}" for name, value in extra.items())
    write_text_file(report, run_dir / "summary.md")
    save_config_with_metadata({"args": vars(parsed_arguments), "episodes": episodes, "seeds": seeds, "start": START, **extra},
                     run_dir / "config.json")
    print(report)
    return table


if __name__ == "__main__":
    main()
