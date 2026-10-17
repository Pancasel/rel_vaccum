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

from experiments.common import add_common_args, resolve_out, run_parallel

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from baselines import make_baseline  # noqa: E402
from deep.dqn import DQNAgent, DQNConfig  # noqa: E402
from env.maps import generate_map_set  # noqa: E402
from experiments.evaluate import evaluate_on_maps, summarize  # noqa: E402
from experiments.plots import df_to_markdown_table, plot_bar_with_error, plot_lines_with_error, write_text  # noqa: E402
from experiments.run_deep import TEST_SET_SEED, TRAIN_POOL_SEED, train_one  # noqa: E402
from utils.io import save_config_json, save_csv  # noqa: E402

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
    add_common_args(parser, "results/generalization")
    parser.set_defaults(seeds=[0, 1, 2])
    return parser


def dqn_cfg(timesteps: int, seed: int, quick: bool, device: str) -> dict[str, Any]:
    """Cấu hình DQN chung cho E3/E4 (giống run_deep để kết quả so sánh được)."""
    cfg = DQNConfig(total_timesteps=timesteps, learning_starts=min(DQNConfig.learning_starts, timesteps // 5),
                    eval_every_steps=max(1, timesteps // 20), eval_episodes=2 if quick else 20,
                    log_every_steps=max(1, min(DQNConfig.log_every_steps, timesteps // 10)), device=device, seed=seed)
    return cfg.__dict__.copy()


def evaluate_agents(test_maps: list[np.ndarray], dqn_dirs: list[tuple[int, Path]], seeds: list[int],
                    device: str = "auto") -> pd.DataFrame:
    """Mọi agent trên cùng tập bản đồ test; mỗi seed cho một giá trị trung bình (để lấy std qua seed)."""
    rows = []
    for seed in seeds:
        for name in BASELINE_NAMES:
            df = evaluate_on_maps(make_baseline(name, seed=seed), test_maps, EPISODES_PER_TEST_MAP, seed=seed)
            rows.append({"agent": name, "seed": seed, **summarize(df).iloc[0].to_dict()})
    for seed, log_dir in dqn_dirs:
        agent = DQNAgent.load(log_dir / "model.pt", device=device)
        df = evaluate_on_maps(agent, test_maps, EPISODES_PER_TEST_MAP, seed=seed)
        rows.append({"agent": "dqn", "seed": seed, **summarize(df).iloc[0].to_dict()})
    return pd.DataFrame(rows)


def across_seeds(per_seed: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    """Trung bình ± std QUA SEED của các chỉ số đã tính trên từng seed."""
    metrics = ["coverage_mean", "success_rate", "steps_mean", "steps_to_full_mean", "redundancy_mean"]
    g = per_seed.groupby(group, sort=False)[metrics]
    out = g.mean().join(g.std(ddof=1).fillna(0.0), rsuffix="_sd").reset_index()
    out.insert(len(group), "n_seeds", per_seed.groupby(group, sort=False).size().values)
    return out


def run_density(args: argparse.Namespace, out: Path, seeds: list[int], timesteps: int) -> pd.DataFrame:
    """E3: với mỗi mật độ, huấn luyện DQN trên phòng ngẫu nhiên rồi so với baseline trên cùng tập test."""
    h, w = args.size or (8, 8)
    n_test = 3 if args.quick else (args.n_test_maps or 20)
    tasks, meta = [], []
    test_sets: dict[float, list[np.ndarray]] = {}
    for density in args.densities:
        if density == 0.0:
            # Mật độ 0 chỉ có đúng một bản đồ (phòng trống): train và test buộc phải trùng nhau.
            test_maps, exclude = [np.zeros((h, w), dtype=np.int8)], None
        else:
            test_maps = generate_map_set(h, w, density, n_test, seed=1000 + int(round(100 * density)))
            exclude = test_maps
        test_sets[density] = test_maps
        spec = {"map": None, "random": (h, w, density), "pool": None, "exclude": exclude, "reward": "dense",
                "hide": False, "start": None}
        for seed in seeds:
            log_dir = out / "density" / f"{h}x{w}_d{density:g}" / f"seed{seed}"
            tasks.append({"spec": spec, "cfg": dqn_cfg(timesteps, seed, args.quick, args.device), "log_dir": str(log_dir),
                          "test_maps": test_maps})
            meta.append((density, seed, log_dir))
    run_parallel(train_one, tasks, workers=args.workers)
    per_seed = []
    for density, test_maps in test_sets.items():
        dirs = [(seed, log_dir) for d, seed, log_dir in meta if d == density]
        per_seed.append(evaluate_agents(test_maps, dirs, seeds, args.device).assign(density=density))
    table = across_seeds(pd.concat(per_seed, ignore_index=True), ["density", "agent"])
    plot_lines_with_error(table, "density",
                          [("coverage_mean", "coverage_mean_sd", "tỷ lệ đã dọn"),
                           ("steps_to_full_mean", "steps_to_full_mean_sd", "số bước (episode dọn xong)")],
                          "agent", out / "figures" / f"e3_density_{h}x{w}.png",
                          f"E3: DQN và baseline trên phòng {h}x{w} khi tăng mật độ vật cản "
                          f"(trung bình ± std qua {len(seeds)} seed)", "mật độ vật cản")
    save_csv(table, out / "density" / f"e3_density_{h}x{w}.csv")
    return table


def run_generalization(args: argparse.Namespace, out: Path, seeds: list[int], timesteps: int) -> pd.DataFrame:
    """E4: huấn luyện trên K bản đồ cố định (hoặc vô hạn), đo trên tập train và tập test chưa gặp."""
    h, w = args.size or (6, 6)
    density = args.density
    n_test = 3 if args.quick else (args.n_test_maps or 50)
    k_values = ([k for k in args.k_values if k <= 3] or [1]) if args.quick else args.k_values
    test_maps = generate_map_set(h, w, density, n_test, seed=TEST_SET_SEED)
    tasks, meta = [], []
    for k in k_values:
        pool = generate_map_set(h, w, density, k, seed=TRAIN_POOL_SEED, exclude=test_maps) if k > 0 else None
        spec = {"map": None, "random": (h, w, density), "pool": pool, "exclude": test_maps, "reward": "dense",
                "hide": False, "start": None}
        for seed in seeds:
            log_dir = out / "generalization" / f"{h}x{w}_d{density:g}_K{k}" / f"seed{seed}"
            tasks.append({"spec": spec, "cfg": dqn_cfg(timesteps, seed, args.quick, args.device), "log_dir": str(log_dir),
                          "test_maps": test_maps})
            meta.append((k, seed, log_dir, pool))
    run_parallel(train_one, tasks, workers=args.workers)
    rows = []
    for k, seed, log_dir, pool in meta:
        agent = DQNAgent.load(log_dir / "model.pt", device=args.device)
        splits = {"test (chưa gặp)": test_maps}
        if pool is not None:
            splits["train (đã gặp)"] = pool[:n_test]
        for split, maps in splits.items():
            df = evaluate_on_maps(agent, maps, EPISODES_PER_TEST_MAP, seed=seed)
            rows.append({"K": "∞" if k == 0 else str(k), "split": split, "seed": seed, **summarize(df).iloc[0].to_dict()})
    table = across_seeds(pd.DataFrame(rows), ["K", "split"])
    plot_bar_with_error(table, "K", "coverage_mean", "coverage_mean_sd", out / "figures" / f"e4_generalization_{h}x{w}.png",
                        f"E4: DQN huấn luyện trên K phòng, đo trên phòng đã gặp và chưa gặp ({h}x{w}, mật độ {density:g})",
                        "tỷ lệ đã dọn", hue="split")
    save_csv(table, out / "generalization" / f"e4_generalization_{h}x{w}.csv")
    return table


def main(argv: list[str] | None = None) -> pd.DataFrame:
    args = build_parser().parse_args(argv)
    out = resolve_out(args.out)
    seeds = args.seeds[:1] if args.quick else args.seeds
    timesteps = min(args.timesteps, 1_200) if args.quick else args.timesteps
    if args.quick and args.exp == "density":
        args.densities = args.densities[:2]
    table = (run_density if args.exp == "density" else run_generalization)(args, out, seeds, timesteps)
    text = df_to_markdown_table(table)
    write_text(text, out / args.exp / "summary.md")
    save_config_json({"args": vars(args), "seeds": seeds, "timesteps": timesteps}, out / args.exp / "config.json")
    print(text)
    return table


if __name__ == "__main__":
    main()
