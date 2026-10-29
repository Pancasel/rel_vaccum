"""T7.3 (E6): khoảng cách tới tối ưu của từng phương pháp trên lưới nhỏ (3x3, 4x4).

Mỗi ô của bảng: số bước (trung bình ± std qua seed) và gap% = (số bước − tối thiểu) / tối thiểu.
- Q-learning: huấn luyện riêng cho từng ô xuất phát (bảng Q gắn với một điểm xuất phát).
- DQN: một mô hình cho mỗi bản đồ (huấn luyện với xuất phát ngẫu nhiên), đánh giá từ từng ô.
- VI, greedy_bfs, zigzag: tất định.

Ví dụ:
    python -m experiments.run_optimality --workers 12
"""

from __future__ import annotations

import argparse
import functools
from typing import Any

from experiments.common import add_common_args, make_env_factory, resolve_out, run_parallel

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from agents.baselines import GreedyNearestDirtyCellAgent, ZigzagSweepAgent  # noqa: E402
from agents.deep_q_network.deep_q_network_agent import DeepQNetworkAgent  # noqa: E402
from agents.deep_q_network.deep_q_network_settings import DeepQNetworkSettings  # noqa: E402
from environment.room_layouts import get_room_layout  # noqa: E402
from environment.vacuum_cleaning_env import VacuumCleaningEnv  # noqa: E402
from evaluation.episode_runner import run_single_episode  # noqa: E402
from experiments.plots import df_to_markdown_table, write_text  # noqa: E402
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
    factory = make_env_factory(map_name, start=start)
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
            "steps": {s: run_single_episode(agent, env, seed=0, reset_options={"start": s})["steps"] for s in STARTS[map_name]}}


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
    add_common_args(parser, "results/optimality")
    args = parser.parse_args(argv)
    out = resolve_out(args.out)
    seeds = args.seeds[:1] if args.quick else args.seeds
    q_eps = {m: (200 if args.quick else Q_EPISODES[m]) for m in args.maps}
    dqn_steps = {m: (800 if args.quick else DQN_TIMESTEPS[m]) for m in args.maps}

    q_tasks = [{"map_name": m, "start": s, "episodes": q_eps[m], "seed": seed}
               for m in args.maps for s in STARTS[m] for seed in seeds]
    d_tasks = [{"map_name": m, "timesteps": dqn_steps[m], "seed": seed, "log_dir": str(out / "dqn" / m / f"seed{seed}"), "device": args.device}
               for m in args.maps for seed in seeds]
    results = run_parallel(dispatch_task, [{"kind": "q", "kwargs": t} for t in q_tasks] +
                           [{"kind": "dqn", "kwargs": t} for t in d_tasks], workers=args.workers)
    q_res = [r for r in results if "start" in r]
    d_res = [r for r in results if "start" not in r]

    rows, raw = [], []
    for m in args.maps:
        grid = get_room_layout(m)
        vi = run_value_iteration(build_state_transition_graph(grid))
        env = VacuumCleaningEnv(grid)
        for s in STARTS[m]:
            optimum = minimum_steps_to_clean_all(vi.graph, s)
            values = {
                "VI": [run_single_episode(ValueIterationPolicyAgent(vi), env, seed=0, reset_options={"start": s})["steps"]],
                "Q-learning": [r["steps"] for r in q_res if r["map"] == m and r["start"] == s],
                "DQN": [r["steps"][s] for r in d_res if r["map"] == m],
                "greedy_bfs": [run_single_episode(GreedyNearestDirtyCellAgent(), env, seed=0, reset_options={"start": s})["steps"]],
                "zigzag": [run_single_episode(ZigzagSweepAgent(), env, seed=0, reset_options={"start": s})["steps"]],
            }
            rows.append({"map": m, "start": str(s), "min_steps": optimum,
                         **{name: cell(v, optimum) for name, v in values.items()}})
            raw += [{"map": m, "start": str(s), "method": name, "steps_mean": float(np.mean(v)), "min_steps": optimum,
                     "gap_pct": (float(np.mean(v)) - optimum) / optimum * 100, "n": len(v)}
                    for name, v in values.items()]
    table = pd.DataFrame(rows)
    save_dataframe_csv(pd.DataFrame(raw), out / "e6_optimality_gap.csv")
    text = (f"# E6: khoảng cách tới tối ưu (số bước, trung bình ± std qua {len(seeds)} seed, gap so với tối thiểu)\n\n"
            + df_to_markdown_table(table))
    write_text(text, out / "e6_optimality_gap.md")
    save_config_with_metadata({"args": vars(args), "seeds": seeds, "q_episodes": q_eps, "dqn_timesteps": dqn_steps,
                      "starts": {k: [list(s) for s in v] for k, v in STARTS.items()}}, out / "config.json")
    print(text)
    return table


if __name__ == "__main__":
    main()
