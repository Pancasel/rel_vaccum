"""Ghi GIF từ danh sách khung hình, hoặc chạy một episode và ghi lại thành GIF."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from project_utils.project_paths import require_path_inside_project


def save_gif(frames: list[np.ndarray], path: str | os.PathLike, frames_per_second: int = 4,
             last_frame_hold: int = 3) -> Path:
    """Ghi danh sách khung hình thành GIF lặp vô hạn (khung cuối giữ lâu hơn cho dễ nhìn).

    Args:
        last_frame_hold: khung cuối hiển thị lâu gấp bấy nhiêu lần một khung thường.

    Raises:
        ValueError: danh sách rỗng, frames_per_second không dương, hoặc đường dẫn ngoài thư mục dự án.
    """
    if not frames:
        raise ValueError("Không có khung hình nào để ghi GIF.")
    if frames_per_second <= 0:
        raise ValueError(f"fps phải dương, nhận được {frames_per_second}.")
    resolved_path = require_path_inside_project(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(frame) for frame in frames]
    durations_in_milliseconds = [1000 // frames_per_second] * len(images)
    durations_in_milliseconds[-1] = durations_in_milliseconds[-1] * max(1, last_frame_hold)
    images[0].save(resolved_path, save_all=True, append_images=images[1:], duration=durations_in_milliseconds,
                   loop=0, optimize=False)
    return resolved_path


def record_episode_as_gif(agent: Any, env: Any, path: str | os.PathLike, seed: int = 0,
                          max_frames: int = 500, frames_per_second: int = 4) -> dict[str, Any]:
    """Chạy một episode với env có render_mode="rgb_array", lưu GIF và trả các chỉ số.

    Raises:
        ValueError: nếu env không ở chế độ rgb_array.
    """
    if getattr(env, "render_mode", None) != "rgb_array":
        raise ValueError("record_episode cần env với render_mode='rgb_array'.")
    if hasattr(agent, "reset"):
        agent.reset()
    observation, info = env.reset(seed=seed)
    frames = [env.render()]
    total_reward = 0.0
    terminated = truncated = False
    while not (terminated or truncated) and len(frames) < max_frames:
        observation, reward, terminated, truncated, info = env.step(agent.act(observation, info))
        total_reward += reward
        frames.append(env.render())
    save_gif(frames, path, frames_per_second=frames_per_second)
    return {"steps": info["steps"], "coverage": info["coverage"], "return_undiscounted": total_reward,
            "terminated": terminated, "truncated": truncated, "frames": len(frames)}


def record_random_policy_gif(room_layout_name: str, path: str | os.PathLike, seed: int = 0) -> Path:
    """Demo nhanh: robot đi ngẫu nhiên (chỉ các hướng không đụng tường) trên bản đồ cố định."""
    from agents.baselines.random_walk_agent import RandomWalkAgent
    from environment.vacuum_cleaning_env import VacuumCleaningEnv

    env = VacuumCleaningEnv(room_layout_name, render_mode="rgb_array")
    record_episode_as_gif(RandomWalkAgent(seed=seed), env, path, seed=seed)
    return require_path_inside_project(path)
