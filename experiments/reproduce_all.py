"""Chạy lại toàn bộ thí nghiệm theo đúng thứ tự, dừng ngay ở bước đầu tiên bị lỗi.

    python -m experiments.reproduce_all --quick            # kiểm tra luồng (vài phút), ghi vào results/_quick/
    python -m experiments.reproduce_all --full --workers 8 # chạy thật (nhiều giờ), ghi vào results/
    python -m experiments.reproduce_all --full --only deep_furniture demo
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

from experiments.common import configure_console

from project_utils.project_paths import PROJECT_ROOT  # noqa: E402

configure_console()

# Tham số tốc độ của v4 (16 môi trường song song, lô 256 mỗi 16 bước) và cấu hình v4 đầy đủ.
V4_SPEED = ["--n-envs", "16", "--batch-size", "256", "--train-freq", "16"]
V4 = [*V4_SPEED, "--action-mask", "--egocentric", "--keep-best"]

# (tên bước, module, tham số chạy thật, có nhận --workers không)
STEPS: list[tuple[str, str, list[str], bool]] = [
    ("baselines", "experiments.run_baselines", [], False),
    ("value_iteration", "experiments.run_value_iteration", [], False),
    ("tabular_3x3", "experiments.run_tabular", ["--map", "empty_3x3", "--start", "0", "0", "--episodes", "5000"], True),
    ("tabular_4x4", "experiments.run_tabular", ["--map", "obstacle_4x4", "--start", "0", "0", "--episodes", "50000"], True),
    ("tabular_4x4_random_start", "experiments.run_tabular",
     ["--map", "obstacle_4x4", "--start", "random", "--episodes", "50000"], True),
    ("state_explosion", "experiments.run_state_explosion", [], True),
    ("e1_markov", "experiments.run_ablation", ["--exp", "markov", "--map", "obstacle_4x4"], True),
    ("e2_reward", "experiments.run_ablation", ["--exp", "reward", "--map", "obstacle_4x4"], True),
    ("deep_3x3", "experiments.run_deep", ["--map", "empty_3x3", "--start", "0", "0", "--timesteps", "30000"], True),
    ("deep_4x4", "experiments.run_deep", ["--map", "obstacle_4x4", "--timesteps", "200000"], True),
    ("deep_furniture", "experiments.run_deep", ["--map", "furniture_8x8", "--timesteps", "500000"], True),
    ("deep_random_8x8", "experiments.run_deep", ["--random-maps", "8", "8", "0.15", "--timesteps", "500000"], True),
    # Cải tiến v3: action masking. A/B trên phòng ngẫu nhiên 6x6, rồi mô hình 8x8 dùng cho demo phòng lạ.
    ("mask_ab_nomask", "experiments.run_deep",
     ["--random-maps", "6", "6", "0.15", "--timesteps", "150000", "--seeds", "0", "1", "--out", "results/mask_ab"], True),
    ("mask_ab_mask", "experiments.run_deep",
     ["--random-maps", "6", "6", "0.15", "--timesteps", "150000", "--seeds", "0", "1", "--action-mask",
      "--out", "results/mask_ab"], True),
    ("deep_random_8x8_mask", "experiments.run_deep",
     ["--random-maps", "8", "8", "0.15", "--timesteps", "500000", "--action-mask"], True),
    ("e3_density", "experiments.run_generalization", ["--exp", "density", "--timesteps", "300000"], True),
    ("e4_generalization", "experiments.run_generalization", ["--exp", "generalization", "--timesteps", "200000"], True),
    ("e6_optimality", "experiments.run_optimality", [], True),
    # Cải tiến v4: A/B bốn cấu hình trên phòng ngẫu nhiên 8x8, rồi bản chính và E3/E4 với cấu hình được chọn.
    *[(f"v4_ab_{name}", "experiments.run_deep",
       ["--random-maps", "8", "8", "0.15", "--timesteps", "600000", "--seeds", "0", "1", *V4_SPEED, "--action-mask",
        *extra, "--out", "results/v4_ab"], True)
      for name, extra in (("mask", []), ("mask_ego", ["--egocentric"]),
                          ("mask_shaped", ["--reward", "shaped", "--shaping-coef", "2"]),
                          ("mask_ego_shaped", ["--egocentric", "--reward", "shaped", "--shaping-coef", "2"]))],
    ("deep_random_8x8_v4", "experiments.run_deep",
     ["--random-maps", "8", "8", "0.15", "--timesteps", "2000000", *V4, "--eval-every", "50000",
      "--eval-episodes", "50", "--checkpoints", "0", "100000", "500000", "2000000"], True),
    ("e3_density_v4", "experiments.run_generalization",
     ["--exp", "density", "--size", "8", "8", "--timesteps", "2000000", "--seeds", "0", "1", "2", *V4], True),
    ("e4_generalization_v4", "experiments.run_generalization",
     ["--exp", "generalization", "--timesteps", "2000000", "--seeds", "0", "1", "2", *V4], True),
    ("demo", "experiments.make_demo", [], False),
]


# Các script có huấn luyện DQN, nhận tham số --device.
DEVICE_MODULES = {"experiments.run_deep", "experiments.run_generalization", "experiments.run_optimality"}


def build_command(name: str, module: str, args: list[str], takes_workers: bool, quick: bool,
                  workers: int, device: str = "auto") -> list[str]:
    """Dòng lệnh cho một bước; chế độ quick ghi vào results/_quick/<tên bước>."""
    cmd = [sys.executable, "-m", module, *args]
    if quick:
        cmd += ["--quick", "--out", f"results/_quick/{name}"]
    if takes_workers and workers > 1:
        cmd += ["--workers", str(workers)]
    if module in DEVICE_MODULES and device != "auto":
        cmd += ["--device", device]
    return cmd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chạy lại mọi thí nghiệm.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true", help="chạy rút gọn để kiểm tra luồng")
    mode.add_argument("--full", action="store_true", help="chạy đầy đủ như báo cáo")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", default="auto", help="thiết bị cho DQN: auto | cpu | cuda")
    parser.add_argument("--only", nargs="+", choices=[s[0] for s in STEPS], help="chỉ chạy các bước này")
    args = parser.parse_args(argv)
    selected = [s for s in STEPS if not args.only or s[0] in args.only]
    for i, (name, module, step_args, takes_workers) in enumerate(selected, 1):
        cmd = build_command(name, module, step_args, takes_workers, args.quick, args.workers, args.device)
        print(f"\n[{i}/{len(selected)}] {name}: {' '.join(cmd[1:])}", flush=True)
        t0 = time.monotonic()
        code = subprocess.run(cmd, cwd=PROJECT_ROOT).returncode
        if code != 0:
            print(f"Bước '{name}' lỗi (mã {code}). Dừng lại.", flush=True)
            return code
        print(f"[{i}/{len(selected)}] {name}: xong sau {time.monotonic() - t0:.0f} s", flush=True)
    print("\nTất cả các bước đã chạy xong.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
