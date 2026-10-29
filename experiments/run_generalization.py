"""T7.2 (E4) và T7.3 (E3): DQN trên phòng ngẫu nhiên, so với baseline và kiểm tra khái quát hóa.

--exp density        : E3 — RL vs random/zigzag/greedy khi tăng mật độ vật cản (0%, 10%, 20%, 30%).
--exp generalization : E4 — huấn luyện trên K bản đồ (1, 10, 100, vô hạn), đo trên 50 bản đồ chưa gặp.

Ví dụ:
    python -m experiments.run_generalization --exp density --size 8 8 --timesteps 500000 --seeds 0 1 2 --workers 12
    python -m experiments.run_generalization --exp generalization --timesteps 300000 --seeds 0 1 2 --workers 12
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from experiments.shared.command_line_options import add_shared_options, resolve_output_directory
from experiments.shared.parallel_execution import run_tasks_in_parallel

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from agents.baselines import create_baseline_agent  # noqa: E402
from agents.deep_q_network.deep_q_network_agent import DeepQNetworkAgent  # noqa: E402
from agents.deep_q_network.deep_q_network_settings import DeepQNetworkSettings  # noqa: E402
from environment.random_room_generator import generate_distinct_rooms  # noqa: E402
from evaluation.agent_evaluation import evaluate_agent_on_rooms  # noqa: E402
from evaluation.metric_summary import summarize_metrics  # noqa: E402
from experiments.plotting.markdown_tables import dataframe_to_markdown_table, write_text_file  # noqa: E402
from experiments.plotting.comparison_plots import plot_bar_with_error, plot_lines_with_error  # noqa: E402
from experiments.shared.evaluation_seeds import TEST_ROOMS_SEED, TRAIN_ROOM_POOL_SEED
from experiments.shared.command_line_options import validate_reward_arguments
from experiments.shared.environment_factories import reward_settings_from_spec
from experiments.run_deep_q_network import train_one
from experiments.shared.result_folder_naming import build_training_options_suffix, build_environment_variant_suffix
from project_utils.result_files import save_config_with_metadata, save_dataframe_csv  # noqa: E402

BASELINE_NAMES = ["random_walk", "zigzag", "greedy_bfs"]
EPISODES_PER_TEST_MAP = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DQN trên bản đồ ngẫu nhiên: E3 (mật độ) và E4 (khái quát hóa).")
    parser.add_argument("--exp", required=True, choices=["density", "generalization"])
    parser.add_argument("--size", type=int, nargs=2, default=None, help="H W (mặc định 8 8 cho E3, 6 6 cho E4)")
    parser.add_argument("--densities", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3])
    parser.add_argument("--density", type=float, default=0.15, help="mật độ cho E4")
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 10, 100, 0], help="K bản đồ train (0 = vô hạn)")
    parser.add_argument("--n-test-maps", type=int, default=None, help="mặc định 20 (E3), 50 (E4)")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--device", default="auto", help="auto (GPU nếu có) | cpu | cuda")
    parser.add_argument("--action-mask", action="store_true", help="cải tiến v3: không chọn hướng đâm vào tường")
    parser.add_argument("--egocentric", action="store_true", help="cải tiến v4: quan sát lấy robot làm tâm")
    parser.add_argument("--reward", default="dense", choices=["dense", "shaped"])
    parser.add_argument("--shaping-coef", type=float, default=None)
    parser.add_argument("--n-envs", type=int, default=1, help="số môi trường song song trong một tiến trình")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--keep-best", action="store_true",
                        help="giữ mô hình có kết quả đánh giá định kỳ tốt nhất (tập kiểm định)")
    parser.add_argument("--train-freq", type=int, default=4, help="cập nhật mạng mỗi N bước môi trường")
    parser.add_argument("--tag", default="", help="hậu tố thêm vào tên thư mục/hình (các tùy chọn khác mặc định "
                                                    "như mask/ego/reward/n-envs đã tự được thêm)")
    add_shared_options(parser, "results/generalization")
    parser.set_defaults(seeds=[0, 1, 2])
    return parser


def dqn_cfg(timesteps: int, seed: int, quick: bool, device: str, parsed_arguments: argparse.Namespace) -> dict[str, Any]:
    """Cấu hình DQN chung cho E3/E4 (giống run_deep để kết quả so sánh được)."""
    agent_settings = DeepQNetworkSettings(total_timesteps=timesteps, random_warmup_steps=min(DeepQNetworkSettings.random_warmup_steps, timesteps // 5),
                    evaluate_every_steps=max(1, timesteps // 20), evaluation_episode_count=2 if quick else 20,
                    log_every_steps=max(1, min(DeepQNetworkSettings.log_every_steps, timesteps // 10)), device=device,
                    use_action_mask=parsed_arguments.action_mask, use_egocentric_observation=parsed_arguments.egocentric, parallel_env_count=parsed_arguments.n_envs,
                    batch_size=parsed_arguments.batch_size, steps_per_network_update=parsed_arguments.train_freq, keep_best_checkpoint=parsed_arguments.keep_best, seed=seed)
    return agent_settings.__dict__.copy()


def evaluate_agents(test_maps: list[np.ndarray], dqn_dirs: list[tuple[int, Path]], seeds: list[int],
                    device: str = "auto", reward_settings: Any = None) -> pd.DataFrame:
    """Mọi agent trên cùng tập bản đồ test; mỗi seed cho một giá trị trung bình (để lấy std qua seed)."""
    result_rows = []
    for seed in seeds:
        for name in BASELINE_NAMES:
            results_table = evaluate_agent_on_rooms(create_baseline_agent(name, seed=seed), test_maps, EPISODES_PER_TEST_MAP, seed=seed,
                                  reward_settings=reward_settings)
            result_rows.append({"agent": name, "seed": seed, **summarize_metrics(results_table).iloc[0].to_dict()})
    for seed, log_dir in dqn_dirs:
        agent = DeepQNetworkAgent.load(log_dir / "model.pt", device=device)
        results_table = evaluate_agent_on_rooms(agent, test_maps, EPISODES_PER_TEST_MAP, seed=seed, reward_settings=reward_settings)
        result_rows.append({"agent": "dqn", "seed": seed, **summarize_metrics(results_table).iloc[0].to_dict()})
    return pd.DataFrame(result_rows)


def across_seeds(per_seed: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    """Trung bình ± std QUA SEED của các chỉ số đã tính trên từng seed."""
    metrics = ["coverage_mean", "success_rate", "steps_mean", "steps_to_full_mean", "redundancy_mean"]
    grouped = per_seed.groupby(group, sort=False)[metrics]
    out = grouped.mean().join(grouped.std(ddof=1).fillna(0.0), rsuffix="_sd").reset_index()
    out.insert(len(group), "n_seeds", per_seed.groupby(group, sort=False).size().values)
    return out


def run_density(parsed_arguments: argparse.Namespace, out: Path, seeds: list[int], timesteps: int) -> pd.DataFrame:
    """E3: với mỗi mật độ, huấn luyện DQN trên phòng ngẫu nhiên rồi so với baseline trên cùng tập test."""
    height, width = parsed_arguments.size or (8, 8)
    test_room_count = 3 if parsed_arguments.quick else (parsed_arguments.n_test_maps or 20)
    tasks, meta = [], []
    test_sets: dict[float, list[np.ndarray]] = {}
    for density in parsed_arguments.densities:
        if density == 0.0:
            # Mật độ 0 chỉ có đúng một bản đồ (phòng trống): train và test buộc phải trùng nhau.
            test_maps, excluded_rooms = [np.zeros((height, width), dtype=np.int8)], None
        else:
            test_maps = generate_distinct_rooms(height, width, density, test_room_count, seed=1000 + int(round(100 * density)))
            excluded_rooms = test_maps
        test_sets[density] = test_maps
        spec = {"map": None, "random": (height, width, density), "pool": None, "exclude": excluded_rooms, "reward": parsed_arguments.reward,
                "shaping_coef": parsed_arguments.shaping_coef, "start": None}
        for seed in seeds:
            log_dir = out / "density" / f"{height}x{width}_d{density:g}{parsed_arguments.tag}" / f"seed{seed}"
            tasks.append({"spec": spec, "agent_settings": dqn_cfg(timesteps, seed, parsed_arguments.quick, parsed_arguments.device, parsed_arguments), "log_dir": str(log_dir),
                          "test_maps": test_maps})
            meta.append((density, seed, log_dir))
    run_tasks_in_parallel(train_one, tasks, workers=parsed_arguments.workers)
    per_seed = []
    for density, test_maps in test_sets.items():
        dirs = [(seed, log_dir) for d, seed, log_dir in meta if d == density]
        per_seed.append(evaluate_agents(test_maps, dirs, seeds, parsed_arguments.device, reward_settings_from_spec(vars(parsed_arguments)))
                        .assign(density=density))
    table = across_seeds(pd.concat(per_seed, ignore_index=True), ["density", "agent"])
    plot_lines_with_error(table, "density",
                          [("coverage_mean", "coverage_mean_sd", "tỷ lệ đã dọn"),
                           ("steps_to_full_mean", "steps_to_full_mean_sd", "số bước (episode dọn xong)")],
                          "agent", out / "figures" / f"e3_density_{height}x{width}{parsed_arguments.tag}.png",
                          f"E3: DQN và baseline trên phòng {height}x{width} khi tăng mật độ vật cản "
                          f"(trung bình ± std qua {len(seeds)} seed)", "mật độ vật cản")
    save_dataframe_csv(table, out / "density" / f"e3_density_{height}x{width}{parsed_arguments.tag}.csv")
    return table


def run_generalization(parsed_arguments: argparse.Namespace, out: Path, seeds: list[int], timesteps: int) -> pd.DataFrame:
    """E4: huấn luyện trên K bản đồ cố định (hoặc vô hạn), đo trên tập train và tập test chưa gặp."""
    height, width = parsed_arguments.size or (6, 6)
    density = parsed_arguments.density
    test_room_count = 3 if parsed_arguments.quick else (parsed_arguments.n_test_maps or 50)
    k_values = ([training_room_count for training_room_count in parsed_arguments.k_values if training_room_count <= 3] or [1]) if parsed_arguments.quick else parsed_arguments.k_values
    test_maps = generate_distinct_rooms(height, width, density, test_room_count, seed=TEST_ROOMS_SEED)
    tasks, meta = [], []
    for training_room_count in k_values:
        pool = generate_distinct_rooms(height, width, density, training_room_count, seed=TRAIN_ROOM_POOL_SEED, excluded_rooms=test_maps) if training_room_count > 0 else None
        spec = {"map": None, "random": (height, width, density), "pool": pool, "exclude": test_maps, "reward": parsed_arguments.reward,
                "shaping_coef": parsed_arguments.shaping_coef, "start": None}
        for seed in seeds:
            log_dir = out / "generalization" / f"{height}x{width}_d{density:g}_K{training_room_count}{parsed_arguments.tag}" / f"seed{seed}"
            tasks.append({"spec": spec, "agent_settings": dqn_cfg(timesteps, seed, parsed_arguments.quick, parsed_arguments.device, parsed_arguments), "log_dir": str(log_dir),
                          "test_maps": test_maps})
            meta.append((training_room_count, seed, log_dir, pool))
    run_tasks_in_parallel(train_one, tasks, workers=parsed_arguments.workers)
    result_rows = []
    for training_room_count, seed, log_dir, pool in meta:
        agent = DeepQNetworkAgent.load(log_dir / "model.pt", device=parsed_arguments.device)
        splits = {"test (chưa gặp)": test_maps}
        if pool is not None:
            splits["train (đã gặp)"] = pool[:test_room_count]
        for split, maps in splits.items():
            results_table = evaluate_agent_on_rooms(agent, maps, EPISODES_PER_TEST_MAP, seed=seed,
                                  reward_settings=reward_settings_from_spec(vars(parsed_arguments)))
            result_rows.append({"K": "∞" if training_room_count == 0 else str(training_room_count), "split": split, "seed": seed, **summarize_metrics(results_table).iloc[0].to_dict()})
    table = across_seeds(pd.DataFrame(result_rows), ["K", "split"])
    plot_bar_with_error(table, "K", "coverage_mean", "coverage_mean_sd", out / "figures" / f"e4_generalization_{height}x{width}{parsed_arguments.tag}.png",
                        f"E4: DQN huấn luyện trên K phòng, đo trên phòng đã gặp và chưa gặp ({height}x{width}, mật độ {density:g})",
                        "tỷ lệ đã dọn", group_column="split")
    save_dataframe_csv(table, out / "generalization" / f"e4_generalization_{height}x{width}{parsed_arguments.tag}.csv")
    return table


def main(argv: list[str] | None = None) -> pd.DataFrame:
    parser = build_parser()
    parsed_arguments = parser.parse_args(argv)
    validate_reward_arguments(parser, parsed_arguments)
    # Tên thư mục/hình tự gồm mọi tùy chọn khác mặc định, để các cấu hình không ghi đè kết quả của nhau.
    parsed_arguments.tag = build_environment_variant_suffix(vars(parsed_arguments)) + build_training_options_suffix(parsed_arguments) + parsed_arguments.tag
    out = resolve_output_directory(parsed_arguments.out)
    seeds = parsed_arguments.seeds[:1] if parsed_arguments.quick else parsed_arguments.seeds
    timesteps = min(parsed_arguments.timesteps, 1_200) if parsed_arguments.quick else parsed_arguments.timesteps
    if parsed_arguments.quick and parsed_arguments.exp == "density":
        parsed_arguments.densities = parsed_arguments.densities[:2]
    table = (run_density if parsed_arguments.exp == "density" else run_generalization)(parsed_arguments, out, seeds, timesteps)
    text = dataframe_to_markdown_table(table)
    write_text_file(text, out / parsed_arguments.exp / f"summary{parsed_arguments.tag}.md")
    save_config_with_metadata({"args": vars(parsed_arguments), "seeds": seeds, "timesteps": timesteps}, out / parsed_arguments.exp / f"config{parsed_arguments.tag}.json")
    print(text)
    return table


if __name__ == "__main__":
    main()
