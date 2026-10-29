"""Chuẩn bị ảnh cho slide từ results/: cắt tiêu đề của hình matplotlib, cắt từng ô của ảnh ghép.

Chạy từ gốc dự án:  python slides/build/prepare_assets.py
Kết quả: slides/build/assets/*.png|gif và slides/build/assets/sizes.json (kích thước pixel để giữ tỷ lệ).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results"
OUT = Path(__file__).resolve().parent / "assets"

# Hình matplotlib: bỏ tiêu đề (đã có tiêu đề slide), giữ từ đường viền trên của trục trở xuống.
FIGURES = {
    "q_convergence.png": "tabular/figures/q_convergence_obstacle_4x4_s00.png",
    "e1_markov.png": "ablation/figures/e1_markov_obstacle_4x4.png",
    "e3_density.png": "generalization/figures/e3_density_8x8_mask_ego_ne16_bs256_tf16.png",
    "dqn_curve_unseen.png": "deep/figures/deep_learning_curve_rand8x8_d0.15_K0_mask_ego_ne16_bs256_tf16.png",
}
# Ảnh động và ảnh giữ nguyên.
COPIES = {
    "race_8x8.gif": "demo/compare_furniture_8x8.gif",
    "unseen_rooms.gif": "demo/unseen_rooms.gif",
    "reward_hacking.gif": "ablation/gifs/e2_reward_hacking_obstacle_4x4.gif",
    "progress_qlearning.gif": "demo/progress_qlearning.gif",
}


TOP_TICK_ROOM = 22  # số pixel giữ lại phía trên đường viền trục (nửa trên của nhãn số trên cùng)


def crop_title(img: Image.Image) -> Image.Image:
    """Bỏ tiêu đề của hình: cắt từ đường viền trên của trục đầu tiên trở xuống.

    Đường viền = hàng đầu tiên có vệt tối dài hơn 25% chiều ngang. Giữ lại TOP_TICK_ROOM pixel phía trên
    cho nhãn số trên cùng (nằm bên trái trục), rồi tô trắng phần còn sót của tiêu đề nằm phía trên trục.
    """
    gray = np.asarray(img.convert("L"))
    dark = gray < 90
    width = gray.shape[1]
    for row in range(gray.shape[0] // 2):
        run, best, best_end = 0, 0, 0
        for col, value in enumerate(dark[row]):
            run = run + 1 if value else 0
            if run > best:
                best, best_end = run, col
        if best > 0.25 * width:
            top = max(0, row - TOP_TICK_ROOM)
            out = img.crop((0, top, img.width, img.height))
            axis_left = best_end - best + 1
            # Chỉ xóa vùng ngay trên trục đầu tiên: nhãn số bên trái và trục thứ hai (nếu có) giữ nguyên.
            blank = Image.new("RGB", (best - 4, row - top - 3), "white")
            out.paste(blank, (axis_left + 2, 0))
            # Tiêu đề có thể rộng hơn trục: xóa thêm dải trên cùng trên toàn chiều ngang, phía trên nhãn số.
            out.paste(Image.new("RGB", (out.width, max(0, row - top - 12)), "white"), (0, 0))
            return out
    return img


def panel(img: Image.Image, index: int, count: int, gap: int = 8) -> Image.Image:
    """Ô thứ `index` trong ảnh ghép `count` ô theo hàng ngang (compose_side_by_side, khe 8 px)."""
    w = (img.width - gap * (count - 1)) // count
    left = index * (w + gap)
    return img.crop((left, 0, left + w, img.height))


def replot_untitled() -> None:
    """Hình một trục có tiêu đề nằm sát trục (cắt sẽ lẹm): vẽ lại từ CSV, không tiêu đề."""
    import sys

    import pandas as pd

    sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.plotting.comparison_plots import plot_bar_with_error
    from experiments.plotting.tabular_scaling_plots import plot_state_explosion

    plot_state_explosion(pd.read_csv(RESULTS / "state_explosion.csv"), OUT / "state_explosion.png", title="")
    e4 = pd.read_csv(RESULTS / "generalization/generalization/e4_generalization_6x6_mask_ego_ne16_bs256_tf16.csv")
    plot_bar_with_error(e4, "K", "coverage_mean", "coverage_mean_sd", OUT / "e4_generalization.png", title="",
                        y_label="tỷ lệ đã dọn", group_column="split")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sizes: dict[str, list[int]] = {}
    replot_untitled()
    for name in ("state_explosion.png", "e4_generalization.png"):
        with Image.open(OUT / name) as img:
            sizes[name] = [img.width, img.height]
    for name, src in FIGURES.items():
        img = crop_title(Image.open(RESULTS / src).convert("RGB"))
        img.save(OUT / name)
        sizes[name] = [img.width, img.height]
    for name, src in COPIES.items():
        shutil.copyfile(RESULTS / src, OUT / name)
        with Image.open(OUT / name) as img:
            sizes[name] = [img.width, img.height]
    # Ô "tối ưu (VI)" (ô cuối trong 5 ô) của cuộc đua 4x4, và ô DQN (ô cuối trong 4 ô) của phòng 8x8.
    race4 = Image.open(RESULTS / "demo/compare_obstacle_4x4.png").convert("RGB")
    race8 = Image.open(RESULTS / "demo/compare_furniture_8x8.png").convert("RGB")
    for name, img in (("room_4x4_optimal.png", panel(race4, 4, 5)), ("room_8x8_dqn.png", panel(race8, 3, 4)),
                      ("room_4x4_random.png", panel(race4, 0, 5))):
        img.save(OUT / name)
        sizes[name] = [img.width, img.height]
    (OUT / "sizes.json").write_text(json.dumps(sizes, indent=2), encoding="utf-8")
    print(json.dumps(sizes))


if __name__ == "__main__":
    main()
