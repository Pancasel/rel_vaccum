"""T6.2: huấn luyện DQN (Deep Q-learning) trên bản đồ cố định hoặc bản đồ ngẫu nhiên.

Ví dụ (thang nghiệm thu 3 bậc của T6.2):
    python -m experiments.run_deep --map empty_3x3 --start 0 0 --timesteps 30000
    python -m experiments.run_deep --map obstacle_4x4 --timesteps 200000 --workers 5
    python -m experiments.run_deep --map furniture_8x8 --timesteps 500000 --workers 5
    python -m experiments.run_deep --random-maps 8 8 0.15 --timesteps 500000 --workers 5   # model cho demo phòng lạ
    python -m experiments.run_deep --quick
"""

from __future__ import annotations

import argparse
import functools
from pathlib import Path
from typing import Any, Callable

from experiments.common import add_common_args, parse_start, resolve_out, run_parallel
from evaluation.optimal_reference import optimal_steps_for_each_start, reference_step_counts

import gymnasium  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from agents.baselines import GreedyNearestDirtyCellAgent, ZigzagSweepAgent  # noqa: E402
from agents.deep_q_network.deep_q_network_agent import DeepQNetworkAgent  # noqa: E402
from agents.deep_q_network.deep_q_network_settings import DeepQNetworkSettings  # noqa: E402
from environment.room_layouts import FIXED_ROOM_LAYOUTS, get_room_layout  # noqa: E402
from environment.random_room_generator import generate_distinct_rooms  # noqa: E402
from environment.random_room_vacuum_env import RandomRoomVacuumCleaningEnv  # noqa: E402
from environment.reward_settings import RewardSettings  # noqa: E402
from environment.vacuum_cleaning_env import VacuumCleaningEnv  # noqa: E402
from evaluation.agent_evaluation import evaluate_agent, evaluate_agent_on_rooms  # noqa: E402
from evaluation.metric_summary import summarize_metrics  # noqa: E402
from experiments.plots import df_to_markdown_table, plot_learning_curves  # noqa: E402
from project_utils.result_files import save_config_with_metadata, save_dataframe_csv  # noqa: E402

TEST_SET_SEED = 3000
TRAIN_POOL_SEED = 2000
N_TEST_MAPS = 50
FINAL_EVAL_EPISODES_RANDOM_START = 50


def reward_config_of(spec: dict[str, Any]) -> RewardSettings:
    """Cấu hình reward theo spec; "shaping_coef" (nếu có) ghi đè hệ số shaping thế năng."""
    overrides = {"shaping_coef": spec["shaping_coef"]} if spec.get("shaping_coef") is not None else {}
    return RewardSettings.from_name(spec["reward"], **overrides)


def make_env(spec: dict[str, Any]) -> gymnasium.Env:
    """Tạo env từ một spec (dict pickle được, để dùng trong tiến trình con).

    spec: {"map": tên | None, "random": (H, W, D) | None, "pool": list | None, "exclude": list | None,
           "start": (r, c) | None, "reward": tên}
    """
    reward = reward_config_of(spec)
    if spec.get("random"):
        h, w, density = spec["random"]
        env: gymnasium.Env = RandomRoomVacuumCleaningEnv(h, w, density, room_pool=spec.get("pool"),
                                                excluded_rooms=spec.get("exclude"), reward_settings=reward)
    else:
        env = VacuumCleaningEnv(spec["map"], reward_settings=reward, start=spec.get("start"))
    return env


# Cờ dòng lệnh -> (trường DeepQNetworkSettings, hậu tố tên thư mục), chỉ thêm vào tên khi khác mặc định.
TRAIN_TAGS = {"learning_rate": ("learning_rate", "lr"),
              "target_update": ("target_network_sync_interval_steps", "tu"),
              "buffer_size": ("replay_buffer_capacity", "buf"), "n_envs": ("parallel_env_count", "ne"),
              "batch_size": ("batch_size", "bs"), "train_freq": ("steps_per_network_update", "tf")}


def env_tag(spec: dict[str, Any]) -> str:
    """Tên thư mục kết quả, ví dụ 'furniture_8x8', 'rand8x8_d0.15_K0', thêm hậu tố nếu ẩn kênh/đổi reward."""
    if spec.get("random"):
        h, w, density = spec["random"]
        tag = f"rand{h}x{w}_d{density:g}_K{len(spec['pool']) if spec.get('pool') else 0}"
    else:
        tag = spec["map"] + (f"_s{spec['start'][0]}{spec['start'][1]}" if spec.get("start") else "")
    return tag + variant_tag(spec)


def variant_tag(options: dict[str, Any]) -> str:
    """Hậu tố cho các tùy chọn môi trường/cải tiến (ẩn kênh, reward, shaping, mask, egocentric)."""
    tag = ""
    if options["reward"] != "dense":
        tag += f"_{options['reward']}"
    if options.get("shaping_coef") is not None:
        tag += f"_sc{options['shaping_coef']:g}"
    if options.get("action_mask"):
        tag += "_mask"
    if options.get("egocentric"):
        tag += "_ego"
    return tag


def train_tag(args: argparse.Namespace) -> str:
    """Hậu tố cho các siêu tham số huấn luyện khác mặc định, để các lần chạy khác nhau không ghi đè nhau."""
    tag = ""
    for flag, (field_name, short) in TRAIN_TAGS.items():
        value = getattr(args, flag, None)
        if value is not None and value != getattr(DeepQNetworkSettings, field_name):
            tag += f"_{short}{value:g}"
    return tag


def check_reward_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """--shaping-coef chỉ có nghĩa với --reward shaped (tránh vô tình bật shaping cho preset khác)."""
    if args.shaping_coef is not None and args.reward != "shaped":
        parser.error("--shaping-coef chỉ dùng được với --reward shaped.")


def final_evaluation(agent: Any, spec: dict[str, Any], seed: int, test_maps: list[np.ndarray] | None) -> pd.DataFrame:
    """Đánh giá cuối: trên tập bản đồ test (nếu huấn luyện bản đồ ngẫu nhiên) hoặc trên bản đồ cố định."""
    if test_maps is not None:
        return evaluate_agent_on_rooms(agent, test_maps, 1, seed=2000 + seed, reward_settings=reward_config_of(spec))
    n = 1 if spec.get("start") else FINAL_EVAL_EPISODES_RANDOM_START
    return evaluate_agent(agent, functools.partial(make_env, spec), n, seed=2000 + seed)


def train_one(spec: dict[str, Any], cfg: dict[str, Any], log_dir: str,
              test_maps: list[np.ndarray] | None) -> dict[str, Any]:
    """Huấn luyện một seed DQN và đánh giá cuối (chạy được ở tiến trình con)."""
    factory = functools.partial(make_env, spec)
    observation_shape = factory().observation_space.shape
    agent = DeepQNetworkAgent(observation_shape, DeepQNetworkSettings(**cfg))
    logs = agent.train(factory, log_dir)
    final = final_evaluation(agent, spec, cfg["seed"], test_maps)
    save_dataframe_csv(final, Path(log_dir) / "final_eval.csv")
    return {"seed": cfg["seed"], "eval": logs["eval"], "final": final}


def baseline_references(spec: dict[str, Any], test_maps: list[np.ndarray] | None) -> dict[str, float]:
    """Số bước tham chiếu (tối ưu nếu tính được, greedy_bfs, zigzag) cho learning curve."""
    if test_maps is None:
        return reference_step_counts(get_room_layout(spec["map"]), spec.get("start"))
    refs = {}
    for name, agent in (("greedy_bfs", GreedyNearestDirtyCellAgent()), ("zigzag", ZigzagSweepAgent())):
        refs[name] = float(evaluate_agent_on_rooms(agent, test_maps, 1, seed=0)["steps"].mean())
    return refs


def summarize_seeds(results: list[dict[str, Any]], optimum: dict[tuple[int, int], int] | None) -> pd.DataFrame:
    """Kết quả cuối, trung bình ± std qua seed."""
    rows = []
    for res in results:
        final = res["final"]
        row = {"seed": res["seed"], **summarize_metrics(final).iloc[0].to_dict()}
        if optimum is not None:
            done = final[final["terminated"]]
            opt = np.array([optimum[(r, c)] for r, c in zip(done["start_row"], done["start_col"])])
            row["gap_pct"] = float(np.mean((done["steps"] - opt) / opt) * 100) if len(done) else float("nan")
        rows.append(row)
    per_seed = pd.DataFrame(rows)
    metrics = [m for m in ("coverage_mean", "success_rate", "steps_mean", "steps_to_full_mean", "redundancy_mean",
                           "collisions_mean", "gap_pct") if m in per_seed.columns]
    out = {"n_seeds": len(per_seed)}
    for m in metrics:
        out[f"{m}_avg"] = per_seed[m].mean()
        out[f"{m}_sd"] = per_seed[m].std(ddof=1) if len(per_seed) > 1 else 0.0
    return pd.DataFrame([out])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Huấn luyện DQN (Deep Q-learning).")
    where = parser.add_mutually_exclusive_group()
    where.add_argument("--map", choices=sorted(FIXED_ROOM_LAYOUTS), help="bản đồ cố định (mặc định obstacle_4x4)")
    where.add_argument("--random-maps", nargs=3, metavar=("H", "W", "DENSITY"),
                       help="huấn luyện trên bản đồ ngẫu nhiên H x W với mật độ vật cản DENSITY")
    parser.add_argument("--n-train-maps", type=int, default=0, help="K bản đồ train cố định (0 = sinh mới mỗi episode)")
    parser.add_argument("--start", nargs="+", default=["random"], help="'random' hoặc R C (chỉ với --map)")
    parser.add_argument("--reward", default="dense", choices=list(RewardSettings.PRESETS))
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--learning-rate", type=float, default=DeepQNetworkSettings.learning_rate)
    parser.add_argument("--target-update", type=int, default=DeepQNetworkSettings.target_network_sync_interval_steps)
    parser.add_argument("--buffer-size", type=int, default=DeepQNetworkSettings.replay_buffer_capacity)
    parser.add_argument("--eval-every", type=int, default=None, help="mặc định: timesteps/40")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--device", default="auto", help="auto (GPU nếu có) | cpu | cuda")
    parser.add_argument("--action-mask", action="store_true", help="cải tiến v3: không chọn hướng đâm vào tường")
    parser.add_argument("--egocentric", action="store_true", help="cải tiến v4: quan sát lấy robot làm tâm")
    parser.add_argument("--shaping-coef", type=float, default=None,
                        help="hệ số shaping thế năng (dùng với --reward shaped), mặc định 1.0")
    parser.add_argument("--n-envs", type=int, default=1, help="số môi trường song song trong một tiến trình")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--keep-best", action="store_true",
                        help="giữ mô hình có kết quả đánh giá định kỳ tốt nhất (tập kiểm định)")
    parser.add_argument("--train-freq", type=int, default=4, help="cập nhật mạng mỗi N bước môi trường")
    parser.add_argument("--checkpoints", type=int, nargs="*", default=None,
                        help="các bước lưu mô hình (mặc định 0, 1/4 và toàn bộ số bước)")
    add_common_args(parser, "results/deep")
    return parser


def build_spec(args: argparse.Namespace) -> tuple[dict[str, Any], list[np.ndarray] | None]:
    """Spec môi trường + tập bản đồ test (nếu huấn luyện bản đồ ngẫu nhiên)."""
    spec: dict[str, Any] = {"reward": args.reward, "start": None,
                            "action_mask": args.action_mask, "egocentric": args.egocentric,
                            "shaping_coef": args.shaping_coef}
    if args.random_maps:
        h, w, density = int(args.random_maps[0]), int(args.random_maps[1]), float(args.random_maps[2])
        n_test = 5 if args.quick else N_TEST_MAPS
        test_maps = generate_distinct_rooms(h, w, density, n_test, seed=TEST_SET_SEED)
        pool = (generate_distinct_rooms(h, w, density, args.n_train_maps, seed=TRAIN_POOL_SEED, excluded_rooms=test_maps)
                if args.n_train_maps > 0 else None)
        spec.update({"map": None, "random": (h, w, density), "pool": pool, "exclude": test_maps})
        return spec, test_maps
    spec.update({"map": args.map or "obstacle_4x4", "random": None, "start": parse_start(args.start)})
    return spec, None


def main(argv: list[str] | None = None) -> pd.DataFrame:
    parser = build_parser()
    args = parser.parse_args(argv)
    check_reward_args(parser, args)
    out = resolve_out(args.out)
    spec, test_maps = build_spec(args)
    seeds = args.seeds[:1] if args.quick else args.seeds
    timesteps = min(args.timesteps, 1_500) if args.quick else args.timesteps
    evaluate_every_steps = args.eval_every or max(1, timesteps // 40)
    checkpoints = args.checkpoints if args.checkpoints is not None else [0, timesteps // 4, timesteps]
    tag = env_tag(spec) + train_tag(args)
    run_dir = out / "dqn" / tag
    tasks = []
    for seed in seeds:
        cfg = DeepQNetworkSettings(total_timesteps=timesteps, learning_rate=args.learning_rate,
                        target_network_sync_interval_steps=args.target_update, replay_buffer_capacity=args.buffer_size,
                        random_warmup_steps=min(DeepQNetworkSettings.random_warmup_steps, timesteps // 5),
                        evaluate_every_steps=evaluate_every_steps, evaluation_episode_count=2 if args.quick else args.eval_episodes,
                        log_every_steps=max(1, min(DeepQNetworkSettings.log_every_steps, timesteps // 10)),
                        checkpoint_steps=tuple(checkpoints), device=args.device, use_action_mask=args.action_mask,
                        use_egocentric_observation=args.egocentric, parallel_env_count=args.n_envs, batch_size=args.batch_size,
                        steps_per_network_update=args.train_freq, keep_best_checkpoint=args.keep_best, seed=seed)
        tasks.append({"spec": spec, "cfg": cfg.__dict__.copy(), "log_dir": str(run_dir / f"seed{seed}"),
                      "test_maps": test_maps})
    results = run_parallel(train_one, tasks, workers=args.workers)

    optimum = optimal_steps_for_each_start(get_room_layout(spec["map"])) if test_maps is None else None
    table = summarize_seeds(results, optimum)
    save_dataframe_csv(table, run_dir / "summary.csv")
    refs = baseline_references(spec, test_maps)
    eval_df = pd.concat([r["eval"].assign(seed=r["seed"], algo="dqn") for r in results], ignore_index=True)
    figure = plot_learning_curves(
        eval_df, "global_step", [("coverage_mean", "tỷ lệ đã dọn (chính sách tham lam)"),
                                 ("steps_mean", "số bước (chính sách tham lam)")], "algo",
        out / "figures" / f"deep_learning_curve_{tag}.png",
        f"DQN trên {tag}", "bước huấn luyện", hlines={"steps_mean": refs})
    save_config_with_metadata({"args": vars(args), "spec": {k: v for k, v in spec.items() if k not in ("pool", "exclude")},
                      "n_test_maps": len(test_maps) if test_maps else 0, "seeds": seeds, "timesteps": timesteps},
                     run_dir / "config.json")
    print(df_to_markdown_table(table))
    print("Tham chiếu số bước:", {k: round(v, 2) for k, v in refs.items()})
    print(f"Hình: {figure}")
    return table


if __name__ == "__main__":
    main()
