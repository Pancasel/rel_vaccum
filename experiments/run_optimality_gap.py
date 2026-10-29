"""T7.3 (E6): khoảng cách tới tối ưu của từng phương pháp trên lưới nhỏ (3x3, 4x4).

Mỗi ô của bảng: số bước (trung bình ± std qua seed) và gap% = (số bước − tối thiểu) / tối thiểu.
- Q-learning: huấn luyện riêng cho từng ô xuất phát (bảng Q gắn với một điểm xuất phát).
- DQN: một mô hình cho mỗi bản đồ (huấn luyện với xuất phát ngẫu nhiên), đánh giá từ từng ô.
- VI, greedy_bfs, zigzag: tất định.

Ví dụ:
    python -m experiments.run_optimality_gap --workers 12
"""

from __future__ import annotations

import argparse
import functools
from typing import Any

from experiments.shared.command_line_options import add_shared_options, resolve_output_directory
from experiments.shared.environment_factories import make_environment_factory
from experiments.shared.parallel_execution import run_tasks_in_parallel

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from agents.baselines import GreedyNearestDirtyCellAgent, ZigzagSweepAgent  # noqa: E402
from agents.deep_q_network.deep_q_network_agent import DeepQNetworkAgent  # noqa: E402
from agents.deep_q_network.deep_q_network_settings import DeepQNetworkSettings  # noqa: E402
from environment.room_layouts import get_room_layout  # noqa: E402
from environment.vacuum_cleaning_env import VacuumCleaningEnv  # noqa: E402
from evaluation.episode_runner import run_single_episode  # noqa: E402
from experiments.plotting.markdown_tables import dataframe_to_markdown_table, write_text_file  # noqa: E402
from agents.tabular.tabular_settings import TabularSettings  # noqa: E402
from agents.tabular.q_learning_agent import QLearningAgent  # noqa: E402
from agents.tabular.state_transition_graph import build_state_transition_graph, minimum_steps_to_clean_all  # noqa: E402
from agents.tabular.value_iteration import ValueIterationPolicyAgent, run_value_iteration  # noqa: E402
from project_utils.result_files import save_config_with_metadata, save_dataframe_csv  # noqa: E402

STARTS: dict[str, list[tuple[int, int]]] = {
    "empty_3x3": [(0, 0), (0, 1), (1, 1)],  # góc, giữa cạnh, tâm
    "obstacle_4x4": [(0, 0), (0, 3), (1, 2), (2, 0), (3, 3)],
}
Q_EPISODES = {"empty_3x3": 5_000, "obstacle_4x4": 20_000}
DQN_TIMESTEPS = {"empty_3x3": 60_000, "obstacle_4x4": 200_000}


def train_q(map_name: str, start: tuple[int, int], episodes: int, seed: int) -> dict[str, Any]:
    """Q-learning cho một (bản đồ, ô xuất phát, seed); trả số bước của chính sách tham lam."""
    factory = make_environment_factory(map_name, start=start)
    agent = QLearningAgent(get_room_layout(map_name), TabularSettings(episode_count=episodes, evaluate_every_episodes=episodes, evaluation_episode_count=1,
                                                            seed=seed))
    agent.train(factory)
    return {"map": map_name, "start": start, "seed": seed, "steps": run_single_episode(agent, factory(), seed=0)["steps"]}


def train_dqn(map_name: str, timesteps: int, seed: int, log_dir: str, device: str = "auto") -> dict[str, Any]:
    """DQN cho một bản đồ (xuất phát ngẫu nhiên); trả số bước từ từng ô xuất phát của bảng E6."""
    factory = functools.partial(VacuumCleaningEnv, map_name)
    agent = DeepQNetworkAgent((3, *get_room_layout(map_name).shape),
                     DeepQNetworkSettings(total_timesteps=timesteps, random_warmup_steps=min(2_000, timesteps // 5),
                               evaluate_every_steps=max(1, timesteps // 10), evaluation_episode_count=10, device=device, seed=seed))
    agent.train(factory, log_dir)
    env = factory()
    return {"map": map_name, "seed": seed,
            "steps": {start_cell: run_single_episode(agent, env, seed=0, reset_options={"start": start_cell})["steps"] for start_cell in STARTS[map_name]}}


def dispatch_task(kind: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Điều phối tác vụ Q-learning/DQN cho run_parallel (hàm cấp module để pickle được ở tiến trình con)."""
    return train_q(**kwargs) if kind == "q" else train_dqn(**kwargs)


def cell(values: list[float], optimum: int) -> str:
    """Ô bảng 'trung bình ± std (gap%)'."""
    mean, sd = float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    gap = (mean - optimum) / optimum * 100
    return f"{mean:.1f} ± {sd:.1f} ({gap:+.0f}%)" if sd > 0 else f"{mean:.1f} ({gap:+.0f}%)"


def main(argv: list[str] | None = None) -> pd.DataFrame:
    parser = argparse.ArgumentParser(description="E6: khoảng cách tới tối ưu trên lưới nhỏ.")
    parser.add_argument("--maps", nargs="+", default=list(STARTS), choices=list(STARTS))
    parser.add_argument("--device", default="auto", help="auto (GPU nếu có) | cpu | cuda")
    add_shared_options(parser, "results/optimality")
    parsed_arguments = parser.parse_args(argv)
    out = resolve_output_directory(parsed_arguments.out)
    seeds = parsed_arguments.seeds[:1] if parsed_arguments.quick else parsed_arguments.seeds
    q_learning_episode_counts = {map_name: (200 if parsed_arguments.quick else Q_EPISODES[map_name]) for map_name in parsed_arguments.maps}
    dqn_steps = {map_name: (800 if parsed_arguments.quick else DQN_TIMESTEPS[map_name]) for map_name in parsed_arguments.maps}

    q_learning_tasks = [{"map_name": map_name, "start": start_cell, "episodes": q_learning_episode_counts[map_name], "seed": seed}
               for map_name in parsed_arguments.maps for start_cell in STARTS[map_name] for seed in seeds]
    dqn_tasks = [{"map_name": map_name, "timesteps": dqn_steps[map_name], "seed": seed, "log_dir": str(out / "dqn" / map_name / f"seed{seed}"), "device": parsed_arguments.device}
               for map_name in parsed_arguments.maps for seed in seeds]
    results = run_tasks_in_parallel(dispatch_task, [{"kind": "q", "kwargs": task} for task in q_learning_tasks] +
                           [{"kind": "dqn", "kwargs": task} for task in dqn_tasks], workers=parsed_arguments.workers)
    q_learning_results = [task_result for task_result in results if "start" in task_result]
    dqn_results = [task_result for task_result in results if "start" not in task_result]

    result_rows, raw = [], []
    for map_name in parsed_arguments.maps:
        grid = get_room_layout(map_name)
        vi = run_value_iteration(build_state_transition_graph(grid))
        env = VacuumCleaningEnv(grid)
        for start_cell in STARTS[map_name]:
            optimum = minimum_steps_to_clean_all(vi.graph, start_cell)
            values = {
                "VI": [run_single_episode(ValueIterationPolicyAgent(vi), env, seed=0, reset_options={"start": start_cell})["steps"]],
                "Q-learning": [task_result["steps"] for task_result in q_learning_results if task_result["map"] == map_name and task_result["start"] == start_cell],
                "DQN": [task_result["steps"][start_cell] for task_result in dqn_results if task_result["map"] == map_name],
                "greedy_bfs": [run_single_episode(GreedyNearestDirtyCellAgent(), env, seed=0, reset_options={"start": start_cell})["steps"]],
                "zigzag": [run_single_episode(ZigzagSweepAgent(), env, seed=0, reset_options={"start": start_cell})["steps"]],
            }
            result_rows.append({"map": map_name, "start": str(start_cell), "min_steps": optimum,
                         **{name: cell(step_counts, optimum) for name, step_counts in values.items()}})
            raw += [{"map": map_name, "start": str(start_cell), "method": name, "steps_mean": float(np.mean(step_counts)), "min_steps": optimum,
                     "gap_pct": (float(np.mean(step_counts)) - optimum) / optimum * 100, "n": len(step_counts)}
                    for name, step_counts in values.items()]
    table = pd.DataFrame(result_rows)
    save_dataframe_csv(pd.DataFrame(raw), out / "e6_optimality_gap.csv")
    text = (f"# E6: khoảng cách tới tối ưu (số bước, trung bình ± std qua {len(seeds)} seed, gap so với tối thiểu)\n\n"
            + dataframe_to_markdown_table(table))
    write_text_file(text, out / "e6_optimality_gap.md")
    save_config_with_metadata({"args": vars(parsed_arguments), "seeds": seeds, "q_episodes": q_learning_episode_counts, "dqn_timesteps": dqn_steps,
                      "starts": {map_name: [list(start_cell) for start_cell in start_cells] for map_name, start_cells in STARTS.items()}}, out / "config.json")
    print(text)
    return table


if __name__ == "__main__":
    main()
