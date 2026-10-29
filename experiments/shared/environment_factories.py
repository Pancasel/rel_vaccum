"""Tạo môi trường cho thí nghiệm. Hàm tạo phải pickle được để gửi sang tiến trình con."""

from __future__ import annotations

import functools
from typing import Any, Callable

import gymnasium
import numpy as np

from environment.random_room_vacuum_env import RandomRoomVacuumCleaningEnv
from environment.reward_settings import RewardSettings
from environment.vacuum_cleaning_env import VacuumCleaningEnv


def make_environment_factory(room: str | np.ndarray, start: tuple[int, int] | None = None,
                             reward: str | RewardSettings = "dense", max_steps: int | None = None,
                             render_mode: str | None = None) -> Callable[[], VacuumCleaningEnv]:
    """Hàm tạo env pickle được (functools.partial) để dùng được trong nhiều tiến trình."""
    reward_settings = RewardSettings.from_name(reward) if isinstance(reward, str) else reward
    return functools.partial(VacuumCleaningEnv, room, reward_settings=reward_settings, max_steps=max_steps,
                             start=start, render_mode=render_mode)


# Environment spec: dict thuần (pickle được) mô tả một môi trường huấn luyện DQN:
#   "map": tên bản đồ cố định | None      "random": (H, W, mật độ vật cản) | None
#   "pool": danh sách K bản đồ train | None   "exclude": bản đồ không được sinh khi train (tập test) | None
#   "start": (hàng, cột) | None           "reward": tên preset reward ("dense"...)
#   "shaping_coef": hệ số shaping | None  "action_mask"/"egocentric": cờ cải tiến v3/v4 (chỉ để đặt tên thư mục)
EnvironmentSpec = dict[str, Any]


def reward_settings_from_spec(spec: EnvironmentSpec) -> RewardSettings:
    """Cấu hình reward theo spec; "shaping_coef" (nếu có) ghi đè hệ số shaping thế năng."""
    overrides = {}
    if spec.get("shaping_coef") is not None:
        overrides["shaping_coef"] = spec["shaping_coef"]
    return RewardSettings.from_name(spec["reward"], **overrides)


def make_environment_from_spec(spec: EnvironmentSpec) -> gymnasium.Env:
    """Tạo env từ một environment spec (dict pickle được, để dùng trong tiến trình con)."""
    reward_settings = reward_settings_from_spec(spec)
    if spec.get("random"):
        height, width, obstacle_density = spec["random"]
        return RandomRoomVacuumCleaningEnv(height, width, obstacle_density, room_pool=spec.get("pool"),
                                           excluded_rooms=spec.get("exclude"), reward_settings=reward_settings)
    return VacuumCleaningEnv(spec["map"], reward_settings=reward_settings, start=spec.get("start"))
