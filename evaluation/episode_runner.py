"""Chạy đúng một episode với một agent và tính các chỉ số của episode đó (chỉ số B6)."""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np


def run_single_episode(agent: Any, env: gymnasium.Env, seed: int, gamma: float = 0.99,
                       record_visit_counts: bool = False, record_frames: bool = False,
                       reset_options: dict[str, Any] | None = None) -> dict[str, Any]:
    """Chạy đúng một episode và tính các chỉ số B6.

    Args:
        agent: đối tượng có act(observation, info) -> int (và tùy chọn reset()).
        env: môi trường (VacuumCleaningEnv hoặc đã bọc wrapper).
        seed: seed cho env.reset.
        gamma: hệ số chiết khấu γ để tính return_discounted (cùng một thước đo cho mọi agent).
        record_visit_counts: ghi số lần đi qua mỗi ô (để vẽ heatmap), trả về trong khóa "visits".
        record_frames: ghi ảnh từng bước (env phải có render_mode="rgb_array"), trả về trong khóa "frames".
        reset_options: truyền vào env.reset (ví dụ {"start": (r, c)} hoặc {"grid": ...}).

    Returns:
        dict với các cột PER_EPISODE_METRIC_COLUMNS (trừ "episode"), cộng "visits"/"frames" nếu yêu cầu.

    Raises:
        ValueError: yêu cầu ghi ảnh nhưng env không ở chế độ rgb_array.
    """
    if record_frames and getattr(env, "render_mode", None) != "rgb_array":
        raise ValueError("record_frames=True cần env với render_mode='rgb_array'.")
    if hasattr(agent, "reset"):
        agent.reset()
    observation, info = env.reset(seed=seed, options=reset_options)
    start_position = info["position"]
    visit_counts = None
    if record_visit_counts:
        visit_counts = np.zeros(env.observation_space.shape[1:], dtype=np.int32)
        visit_counts[start_position] += 1
    frames = [env.render()] if record_frames else None
    undiscounted_return, discounted_return, discount_factor = 0.0, 0.0, 1.0
    terminated = truncated = False
    while not (terminated or truncated):
        action = agent.act(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        undiscounted_return += reward
        discounted_return += discount_factor * reward
        discount_factor *= gamma
        if visit_counts is not None:
            visit_counts[info["position"]] += 1
        if frames is not None:
            frames.append(env.render())
    step_count = info["steps"]
    # Tên khóa là tên cột CSV (báo cáo đang trích), không đổi.
    episode_metrics: dict[str, Any] = {
        "env_seed": seed,
        "start_row": start_position[0],
        "start_col": start_position[1],
        "return_undiscounted": undiscounted_return,
        "return_discounted": discounted_return,
        "coverage": info["coverage"],
        "steps": step_count,
        "steps_to_full": float(step_count) if terminated else float("nan"),
        # D-10: tỷ lệ bước đi vào ô đã dọn (va chạm không tính vào tử số).
        "redundancy": info["n_revisit"] / step_count if step_count > 0 else 0.0,
        "collisions": info["n_collisions"],
        "n_revisit": info["n_revisit"],
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }
    if visit_counts is not None:
        episode_metrics["visits"] = visit_counts
    if frames is not None:
        episode_metrics["frames"] = frames
    return episode_metrics
