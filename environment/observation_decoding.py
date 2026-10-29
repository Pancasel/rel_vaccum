"""Giải mã observation (3, H, W) của môi trường thành vị trí robot, ô đã dọn và vật cản.

Observation gồm 3 kênh: kênh 0 = vị trí robot (one-hot), kênh 1 = ô đã dọn, kênh 2 = vật cản.
"""

from __future__ import annotations

import numpy as np

from environment.grid_movement import safe_actions_at_position


def decode_observation(observation: np.ndarray) -> tuple[tuple[int, int], np.ndarray, np.ndarray]:
    """Giải mã observation (3, H, W) thành (vị trí robot, mảng đã dọn, mảng vật cản).

    Raises:
        ValueError: nếu shape sai hoặc kênh 0 không có đúng một ô bằng 1.
    """
    if observation.ndim != 3 or observation.shape[0] != 3:
        raise ValueError(f"Observation phải có shape (3, H, W), nhận được {observation.shape}.")
    robot_channel = observation[0] > 0.5
    if int(robot_channel.sum()) != 1:
        raise ValueError(f"Kênh 0 phải có đúng một ô robot, đang có {int(robot_channel.sum())}.")
    robot_flat_index = int(np.argmax(robot_channel))
    robot_row, robot_column = divmod(robot_flat_index, observation.shape[2])
    cleaned = observation[1] > 0.5
    obstacles = (observation[2] > 0.5).astype(np.int8)
    return (int(robot_row), int(robot_column)), cleaned, obstacles


def safe_actions_from_observation(observation: np.ndarray) -> list[int]:
    """Các hành động không đụng tường/vật cản, tính trực tiếp từ observation."""
    robot_position, _, obstacles = decode_observation(observation)
    return safe_actions_at_position(obstacles, robot_position)
