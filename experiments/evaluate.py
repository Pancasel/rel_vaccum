"""Bộ đánh giá chung cho mọi agent (giao diện C3, chỉ số B6).

Mọi agent (baseline, Q-learning, DQN) chạy qua cùng một hàm nên so sánh luôn công bằng.
"""

from __future__ import annotations

from typing import Any, Callable

import gymnasium
import numpy as np
import pandas as pd

from env.vacuum_env import RewardConfig, VacuumEnv

METRIC_COLUMNS: list[str] = [
    "episode", "env_seed", "start_row", "start_col", "return_undiscounted", "return_discounted",
    "coverage", "steps", "steps_to_full", "redundancy", "collisions", "n_revisit", "terminated", "truncated",
]
SUMMARY_METRICS: list[str] = [
    "return_undiscounted", "return_discounted", "coverage", "steps", "steps_to_full", "redundancy",
    "collisions", "n_revisit",
]
SEED_STRIDE: int = 10_000  # D-23: episode i của evaluate(seed=s) dùng reset(seed=s*10_000+i)


def run_episode(agent: Any, env: gymnasium.Env, seed: int, gamma: float = 0.99,
                record_visits: bool = False, record_frames: bool = False,
                reset_options: dict[str, Any] | None = None) -> dict[str, Any]:
    """Chạy đúng một episode và tính các chỉ số B6.

    Args:
        agent: đối tượng có act(obs, info) -> int (và tùy chọn reset()).
        env: môi trường (VacuumEnv hoặc đã bọc wrapper).
        seed: seed cho env.reset.
        gamma: hệ số chiết khấu để tính return_discounted (cùng một thước đo cho mọi agent).
        record_visits: ghi số lần đi qua mỗi ô (để vẽ heatmap).
        record_frames: ghi ảnh từng bước (env phải có render_mode="rgb_array").
        reset_options: truyền vào env.reset (ví dụ {"start": (r, c)} hoặc {"grid": ...}).

    Returns:
        dict với các cột METRIC_COLUMNS (trừ "episode"), cộng "visits"/"frames" nếu yêu cầu.

    Raises:
        ValueError: yêu cầu ghi ảnh nhưng env không ở chế độ rgb_array.
    """
    if record_frames and getattr(env, "render_mode", None) != "rgb_array":
        raise ValueError("record_frames=True cần env với render_mode='rgb_array'.")
    if hasattr(agent, "reset"):
        agent.reset()
    obs, info = env.reset(seed=seed, options=reset_options)
    start = info["position"]
    visits = None
    if record_visits:
        visits = np.zeros(env.observation_space.shape[1:], dtype=np.int32)
        visits[start] += 1
    frames = [env.render()] if record_frames else None
    ret, ret_disc, discount = 0.0, 0.0, 1.0
    terminated = truncated = False
    while not (terminated or truncated):
        action = agent.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        ret += reward
        ret_disc += discount * reward
        discount *= gamma
        if visits is not None:
            visits[info["position"]] += 1
        if frames is not None:
            frames.append(env.render())
    steps = info["steps"]
    result: dict[str, Any] = {
        "env_seed": seed,
        "start_row": start[0],
        "start_col": start[1],
        "return_undiscounted": ret,
        "return_discounted": ret_disc,
        "coverage": info["coverage"],
        "steps": steps,
        "steps_to_full": float(steps) if terminated else float("nan"),
        # D-10: tỷ lệ bước đi vào ô đã dọn (va chạm không tính vào tử số).
        "redundancy": info["n_revisit"] / steps if steps > 0 else 0.0,
        "collisions": info["n_collisions"],
        "n_revisit": info["n_revisit"],
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }
    if visits is not None:
        result["visits"] = visits
    if frames is not None:
        result["frames"] = frames
    return result


def evaluate(agent: Any, env_factory: Callable[[], gymnasium.Env], n_episodes: int, seed: int,
             gamma: float = 0.99) -> pd.DataFrame:
    """Chạy agent trên n_episodes episode; mỗi dòng của DataFrame là một episode (C3).

    Env được tạo MỘT lần bằng env_factory(); episode i dùng reset(seed=seed*10_000+i) (D-23).

    Raises:
        ValueError: n_episodes < 1 hoặc seed < 0.
    """
    if n_episodes < 1:
        raise ValueError(f"n_episodes phải >= 1, nhận được {n_episodes}.")
    if seed < 0:
        raise ValueError(f"seed phải >= 0, nhận được {seed}.")
    env = env_factory()
    rows = []
    for i in range(n_episodes):
        row = run_episode(agent, env, seed=seed * SEED_STRIDE + i, gamma=gamma)
        row["episode"] = i
        rows.append(row)
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def evaluate_on_maps(agent: Any, maps: list[np.ndarray], episodes_per_map: int, seed: int,
                     reward_config: RewardConfig | None = None, max_steps_factor: int = 4,
                     gamma: float = 0.99) -> pd.DataFrame:
    """Đánh giá trên một danh sách bản đồ cố định (tập test của E3/E4), start ngẫu nhiên.

    Returns:
        DataFrame như evaluate(), thêm cột "map_id".
    """
    if not maps:
        raise ValueError("Danh sách bản đồ rỗng.")
    rows = []
    for map_id, grid in enumerate(maps):
        n_free = int((grid == 0).sum())
        env = VacuumEnv(grid, reward_config=reward_config, max_steps=max_steps_factor * n_free)
        for i in range(episodes_per_map):
            row = run_episode(agent, env, seed=seed * SEED_STRIDE + map_id * episodes_per_map + i, gamma=gamma)
            row["episode"] = i
            row["map_id"] = map_id
            rows.append(row)
    return pd.DataFrame(rows, columns=["map_id", *METRIC_COLUMNS])


def summarize(df: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    """Trung bình và độ lệch chuẩn (ddof=1) của các chỉ số, kèm số episode n và success_rate.

    Cột đầu ra dạng "coverage_mean", "coverage_std". Độ lệch chuẩn của nhóm 1 phần tử ghi 0.
    steps_to_full chỉ tính trên các episode dọn xong (NaN bị bỏ qua).
    """
    metrics = [m for m in SUMMARY_METRICS if m in df.columns]
    if group_cols:
        grouped = df.groupby(group_cols, sort=False)
        mean = grouped[metrics].mean().add_suffix("_mean")
        std = grouped[metrics].std(ddof=1).fillna(0.0).add_suffix("_std")
        extra = pd.DataFrame({"n": grouped.size(), "success_rate": grouped["terminated"].mean()})
        out = pd.concat([extra, mean, std], axis=1).reset_index()
    else:
        mean = df[metrics].mean().add_suffix("_mean")
        std = df[metrics].std(ddof=1).fillna(0.0).add_suffix("_std")
        extra = pd.Series({"n": len(df), "success_rate": df["terminated"].mean()})
        out = pd.concat([extra, mean, std]).to_frame().T
    ordered = [c for c in out.columns if not c.endswith(("_mean", "_std"))]
    for m in metrics:
        ordered += [f"{m}_mean", f"{m}_std"]
    return out[ordered]
