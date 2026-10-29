"""Bộ đánh giá chung cho mọi agent (giao diện C3).

Mọi agent (baseline, Q-learning, DQN) chạy qua cùng một hàm nên so sánh luôn công bằng.
"""

from __future__ import annotations

from typing import Any, Callable

import gymnasium
import numpy as np
import pandas as pd

from environment.reward_settings import RewardSettings
from environment.room_layouts import FREE_CELL
from environment.vacuum_cleaning_env import VacuumCleaningEnv
from evaluation.episode_runner import run_single_episode
from evaluation.metric_summary import PER_EPISODE_METRIC_COLUMNS

# D-23: episode thứ i của evaluate_agent(seed=s) dùng env.reset(seed=s*10_000+i).
EPISODE_SEED_STRIDE: int = 10_000


def evaluate_agent(agent: Any, env_factory: Callable[[], gymnasium.Env], episode_count: int, seed: int,
                   gamma: float = 0.99) -> pd.DataFrame:
    """Chạy agent trên episode_count episode; mỗi dòng của DataFrame là một episode (C3).

    Env được tạo MỘT lần bằng env_factory(); episode i dùng reset(seed=seed*10_000+i) (D-23).

    Raises:
        ValueError: episode_count < 1 hoặc seed < 0.
    """
    if episode_count < 1:
        raise ValueError(f"n_episodes phải >= 1, nhận được {episode_count}.")
    if seed < 0:
        raise ValueError(f"seed phải >= 0, nhận được {seed}.")
    env = env_factory()
    episode_rows = []
    for episode_index in range(episode_count):
        episode_metrics = run_single_episode(agent, env, seed=seed * EPISODE_SEED_STRIDE + episode_index,
                                             gamma=gamma)
        episode_metrics["episode"] = episode_index
        episode_rows.append(episode_metrics)
    return pd.DataFrame(episode_rows, columns=PER_EPISODE_METRIC_COLUMNS)


def evaluate_agent_on_rooms(agent: Any, rooms: list[np.ndarray], episodes_per_room: int, seed: int,
                            reward_settings: RewardSettings | None = None, max_steps_factor: int = 4,
                            gamma: float = 0.99) -> pd.DataFrame:
    """Đánh giá trên một danh sách bản đồ cố định (tập test của E3/E4), ô xuất phát ngẫu nhiên.

    Returns:
        DataFrame như evaluate_agent(), thêm cột "map_id" (chỉ số bản đồ trong danh sách).
    """
    if not rooms:
        raise ValueError("Danh sách bản đồ rỗng.")
    episode_rows = []
    for room_index, room in enumerate(rooms):
        free_cell_count = int((room == FREE_CELL).sum())
        env = VacuumCleaningEnv(room, reward_settings=reward_settings,
                                max_steps=max_steps_factor * free_cell_count)
        for episode_index in range(episodes_per_room):
            episode_seed = seed * EPISODE_SEED_STRIDE + room_index * episodes_per_room + episode_index
            episode_metrics = run_single_episode(agent, env, seed=episode_seed, gamma=gamma)
            episode_metrics["episode"] = episode_index
            episode_metrics["map_id"] = room_index
            episode_rows.append(episode_metrics)
    return pd.DataFrame(episode_rows, columns=["map_id", *PER_EPISODE_METRIC_COLUMNS])
