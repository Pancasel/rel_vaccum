"""Mô hình MDP xác định: một hành động đưa robot sang trạng thái nào và nhận bao nhiêu thưởng.

`compute_transition()` là nguồn sự thật duy nhất về động lực học và phần thưởng (D-11):
cả `VacuumCleaningEnv.step` lẫn Value Iteration đều gọi hàm này, nên hai bên không thể lệch nhau.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from environment.grid_movement import MOVE_OFFSETS_BY_ACTION
from environment.reward_settings import RewardSettings
from environment.room_layouts import OBSTACLE_CELL
from environment.shortest_path_search import distance_to_nearest_dirty_cell


@dataclass(frozen=True)
class TransitionOutcome:
    """Kết quả một bước của mô hình MDP."""

    next_position: tuple[int, int]
    next_cleaned: np.ndarray  # mảng MỚI (H, W) bool, không dùng chung bộ nhớ với đầu vào
    reward: float
    terminated: bool
    event: str  # "new_cell" | "revisit" | "collision"


def _distance_used_for_shaping(grid: np.ndarray, robot_position: tuple[int, int], cleaned: np.ndarray,
                               is_terminal_state: bool) -> int:
    """Khoảng cách d(s) dùng cho shaping: 0 ở trạng thái kết thúc (quy ước Φ(terminal) = 0)."""
    if is_terminal_state:
        return 0
    return distance_to_nearest_dirty_cell(grid, robot_position, cleaned)


def compute_transition(grid: np.ndarray, robot_position: tuple[int, int], cleaned: np.ndarray, action: int,
                       reward_settings: RewardSettings) -> TransitionOutcome:
    """Mô hình MDP xác định p(s', r | s, a) (D-11).

    Hàm thuần: không đọc/ghi trạng thái toàn cục, không dùng số ngẫu nhiên, không sửa `cleaned`.

    Args:
        grid: bản đồ (H, W), 0 trống, 1 vật cản.
        robot_position: vị trí robot (hàng, cột).
        cleaned: mảng bool (H, W) các ô đã dọn.
        action: 0..3.
        reward_settings: cấu hình phần thưởng.

    Raises:
        ValueError: hành động ngoài 0..3.
    """
    if not 0 <= action < len(MOVE_OFFSETS_BY_ACTION):
        raise ValueError(f"Hành động phải trong 0..3, nhận được {action}.")
    height, width = grid.shape
    row_offset, column_offset = MOVE_OFFSETS_BY_ACTION[action]
    target_row = robot_position[0] + row_offset
    target_column = robot_position[1] + column_offset
    target_is_outside_grid = not (0 <= target_row < height and 0 <= target_column < width)
    if target_is_outside_grid or grid[target_row, target_column] == OBSTACLE_CELL:
        # D-04: đụng tường thì đứng yên nhưng vẫn mất một bước.
        next_position, event, event_reward = robot_position, "collision", reward_settings.collision
    elif cleaned[target_row, target_column]:
        next_position, event, event_reward = (target_row, target_column), "revisit", reward_settings.revisit
    else:
        next_position, event, event_reward = (target_row, target_column), "new_cell", reward_settings.new_cell

    next_cleaned = cleaned.copy()
    next_cleaned[next_position] = True
    # Đã dọn hết khi mọi ô trống đều True (ô vật cản được coi như "xong").
    terminated = bool((next_cleaned | (grid == OBSTACLE_CELL)).all())

    # D-03: reward = bước thời gian + đúng một sự kiện + bonus kết thúc (+ các thành phần tùy chọn).
    completion_bonus = reward_settings.completion if terminated else 0.0
    reward = reward_settings.step + event_reward + completion_bonus

    uses_shaping = reward_settings.shaping_coef != 0.0
    uses_naive_progress_bonus = reward_settings.naive_progress_bonus != 0.0
    if uses_shaping or uses_naive_progress_bonus:
        distance_before = _distance_used_for_shaping(grid, robot_position, cleaned, is_terminal_state=False)
        distance_after = _distance_used_for_shaping(grid, next_position, next_cleaned, terminated)
        if uses_shaping:
            # D-14: F = coef * (γΦ(s') − Φ(s)), Φ(s) = −d(s)/(H+W). Ng, Harada & Russell (1999)
            # chứng minh shaping dạng này không làm đổi chính sách tối ưu.
            potential_before = -distance_before / (height + width)
            potential_after = -distance_after / (height + width)
            reward += reward_settings.shaping_coef * (
                reward_settings.gamma * potential_after - potential_before
            )
        if uses_naive_progress_bonus and distance_after < distance_before:
            # D-15: chỉ thưởng khi lại gần, không phạt khi đi xa -> có thể bị lợi dụng.
            reward += reward_settings.naive_progress_bonus

    return TransitionOutcome(next_position=next_position, next_cleaned=next_cleaned, reward=float(reward),
                             terminated=terminated, event=event)
