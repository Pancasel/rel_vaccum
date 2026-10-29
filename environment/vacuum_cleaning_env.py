"""Môi trường robot hút bụi dạng lưới theo chuẩn Gymnasium.

MDP (PLAN_rel.md mục B2):
- State: vị trí robot + tập ô đã dọn + bản đồ vật cản (đủ thông tin để thỏa tính Markov).
- Action: 0=lên, 1=xuống, 2=trái, 3=phải.
- Transition: xác định (deterministic), tính bởi `compute_transition` (transition_model.py).
- Reward: cấu hình bằng RewardSettings (mặc định "dense").
- terminated: dọn hết ô trống. truncated: hết số bước cho phép (max_steps).

Observation: tensor (3, H, W) float32 — kênh 0 vị trí robot (one-hot), kênh 1 ô đã dọn, kênh 2 vật cản.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces

from environment.grid_movement import safe_actions_at_position
from environment.random_room_generator import is_fully_connected
from environment.reward_settings import RewardSettings
from environment.room_layouts import (FREE_CELL, OBSTACLE_CELL, get_room_layout, list_free_cells,
                                      parse_room_layout)
from environment.transition_model import compute_transition

ACTION_NAMES: tuple[str, ...] = ("up", "down", "left", "right")
ACTION_COUNT: int = len(ACTION_NAMES)

# Số vị trí gần nhất lưu lại để vẽ vệt đường đi khi render.
TRAIL_LENGTH: int = 30


def convert_to_grid(room: np.ndarray | list[str] | str) -> np.ndarray:
    """Chấp nhận mảng, danh sách chuỗi hoặc tên bản đồ cố định; luôn trả bản sao int8."""
    if isinstance(room, str):
        return get_room_layout(room)
    if isinstance(room, np.ndarray):
        if room.ndim != 2:
            raise ValueError(f"Bản đồ phải là mảng 2 chiều, nhận được shape {room.shape}.")
        if not np.isin(room, (FREE_CELL, OBSTACLE_CELL)).all():
            raise ValueError("Bản đồ chỉ được chứa giá trị 0 (trống) và 1 (vật cản).")
        return room.astype(np.int8, copy=True)
    return parse_room_layout(list(room))


class VacuumCleaningEnv(gymnasium.Env):
    """Robot hút bụi trên lưới 2D (giao diện C3 của PLAN_rel.md)."""

    metadata = {"render_modes": ["rgb_array", "ansi"], "render_fps": 4}

    def __init__(self, room: np.ndarray | list[str] | str, reward_settings: RewardSettings | None = None,
                 max_steps: int | None = None, start: tuple[int, int] | None = None,
                 render_mode: str | None = None) -> None:
        """
        Args:
            room: mảng (H, W), danh sách chuỗi, hoặc tên bản đồ cố định (ví dụ "obstacle_4x4").
            reward_settings: cấu hình phần thưởng; None = dense (B2).
            max_steps: số bước tối đa mỗi episode; None = 4 * số ô trống (D-07).
            start: ô xuất phát cố định; None = ngẫu nhiên đều trên ô trống (D-08).
            render_mode: None, "rgb_array" hoặc "ansi".

        Raises:
            ValueError: bản đồ không liên thông, ít hơn 2 ô trống, start không hợp lệ,
                max_steps không dương, hoặc render_mode lạ.
        """
        super().__init__()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"render_mode '{render_mode}' không hỗ trợ; chọn {self.metadata['render_modes']} hoặc None."
            )
        self.render_mode = render_mode
        self.reward_settings: RewardSettings = reward_settings or RewardSettings.dense()
        self._requested_max_steps = max_steps
        self._fixed_start: tuple[int, int] | None = None
        self._set_grid(convert_to_grid(room))
        if start is not None:
            self._fixed_start = self._validate_start(start)
        height, width = self._grid.shape
        self.action_space = spaces.Discrete(ACTION_COUNT)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3, height, width), dtype=np.float32)
        self._is_ready = False

    # ------------------------------------------------------------------ thiết lập bản đồ
    def _set_grid(self, grid: np.ndarray) -> None:
        """Gán bản đồ mới (kiểm tra hợp lệ) và tính lại các đại lượng phụ thuộc."""
        free_cell_count = int((grid == FREE_CELL).sum())
        if free_cell_count < 2:
            raise ValueError(f"Bản đồ cần ít nhất 2 ô trống, đang có {free_cell_count} (D-06).")
        if not is_fully_connected(grid):
            raise ValueError("Bản đồ không liên thông: có ô trống robot không thể tới, không bao giờ dọn xong được.")
        if self._requested_max_steps is not None and self._requested_max_steps <= 0:
            raise ValueError(f"max_steps phải dương, nhận được {self._requested_max_steps}.")
        self._grid = grid
        self._free_cells = list_free_cells(grid)
        self._free_cell_count = free_cell_count
        if self._requested_max_steps is not None:
            self.max_steps: int = self._requested_max_steps
        else:
            self.max_steps = 4 * free_cell_count

    def _validate_start(self, start: tuple[int, int]) -> tuple[int, int]:
        """Kiểm tra ô xuất phát nằm trong lưới và là ô trống."""
        start_row, start_column = int(start[0]), int(start[1])
        height, width = self._grid.shape
        inside_grid = 0 <= start_row < height and 0 <= start_column < width
        if not inside_grid or self._grid[start_row, start_column] != FREE_CELL:
            raise ValueError(f"Ô xuất phát {start} không phải ô trống của bản đồ.")
        return (start_row, start_column)

    # ------------------------------------------------------------------ thuộc tính chỉ đọc
    @property
    def grid(self) -> np.ndarray:
        """Bản sao bản đồ hiện tại."""
        return self._grid.copy()

    @property
    def height(self) -> int:
        return int(self._grid.shape[0])

    @property
    def width(self) -> int:
        return int(self._grid.shape[1])

    @property
    def free_cell_count(self) -> int:
        """Số ô trống |F| của bản đồ hiện tại."""
        return self._free_cell_count

    @property
    def position(self) -> tuple[int, int]:
        self._require_ready()
        return self._robot_position

    @property
    def cleaned(self) -> np.ndarray:
        """Bản sao mảng ô đã dọn."""
        self._require_ready()
        return self._cleaned.copy()

    @property
    def steps(self) -> int:
        self._require_ready()
        return self._step_count

    @property
    def fixed_start(self) -> tuple[int, int] | None:
        """Ô xuất phát cố định (None nếu xuất phát ngẫu nhiên)."""
        return self._fixed_start

    def _require_ready(self) -> None:
        if not self._is_ready:
            raise RuntimeError("Phải gọi reset() trước khi dùng môi trường.")

    # ------------------------------------------------------------------ API Gymnasium
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None
              ) -> tuple[np.ndarray, dict[str, Any]]:
        """Bắt đầu episode mới. options có thể chứa "start": (r, c) để ghi đè ô xuất phát."""
        super().reset(seed=seed)  # bắt buộc để self.np_random được seed đúng chuẩn Gymnasium
        options = options or {}
        if "start" in options and options["start"] is not None:
            start = self._validate_start(options["start"])
        elif self._fixed_start is not None:
            start = self._fixed_start
        else:
            start = self._free_cells[int(self.np_random.integers(len(self._free_cells)))]
        self._robot_position = start
        self._cleaned = np.zeros(self._grid.shape, dtype=bool)
        self._cleaned[start] = True  # D-05: ô xuất phát tính là đã dọn, không nhận thưởng
        self._step_count = 0
        self._revisit_count = 0
        self._collision_count = 0
        self._recent_positions: deque[tuple[int, int]] = deque([start], maxlen=TRAIL_LENGTH)
        self._is_ready = True
        return self._build_observation(), self._build_info(event="reset")

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Thực hiện một hành động. Trả (observation, reward, terminated, truncated, info)."""
        self._require_ready()
        is_integer_action = isinstance(action, (int, np.integer)) and not isinstance(action, (bool, np.bool_))
        if not is_integer_action or not 0 <= int(action) < ACTION_COUNT:
            raise ValueError(f"Hành động phải là số nguyên trong 0..3, nhận được {action!r}.")
        outcome = compute_transition(self._grid, self._robot_position, self._cleaned, int(action),
                                     self.reward_settings)
        self._robot_position = outcome.next_position
        self._cleaned = outcome.next_cleaned
        self._step_count += 1
        if outcome.event == "revisit":
            self._revisit_count += 1
        elif outcome.event == "collision":
            self._collision_count += 1
        self._recent_positions.append(self._robot_position)
        terminated = outcome.terminated
        # D-07: không bao giờ vừa terminated vừa truncated.
        truncated = self._step_count >= self.max_steps and not terminated
        info = self._build_info(event=outcome.event)
        return self._build_observation(), float(outcome.reward), terminated, truncated, info

    def _build_observation(self) -> np.ndarray:
        """Tạo observation MỚI mỗi lần gọi (không trả tham chiếu tới mảng nội bộ)."""
        observation = np.zeros((3, *self._grid.shape), dtype=np.float32)
        observation[0][self._robot_position] = 1.0
        observation[1] = self._cleaned
        observation[2] = self._grid == OBSTACLE_CELL
        return observation

    def _build_info(self, event: str) -> dict[str, Any]:
        """Thông tin phụ trả cùng observation. Tên khóa là một phần giao diện dữ liệu, không đổi."""
        cleaned_cell_count = int(self._cleaned.sum())
        safe_action_mask = np.zeros(ACTION_COUNT, dtype=np.int8)
        safe_action_mask[safe_actions_at_position(self._grid, self._robot_position)] = 1
        return {
            "coverage": cleaned_cell_count / self._free_cell_count,
            "collided": event == "collision",
            "new_cell": event == "new_cell",
            "event": event,
            "steps": self._step_count,
            "n_cleaned": cleaned_cell_count,
            "n_free": self._free_cell_count,
            "position": self._robot_position,
            "n_revisit": self._revisit_count,
            "n_collisions": self._collision_count,
            "action_mask": safe_action_mask,
        }

    def render(self) -> np.ndarray | str | None:
        """Vẽ trạng thái hiện tại theo render_mode ("rgb_array" -> ảnh, "ansi" -> chuỗi)."""
        if self.render_mode is None:
            return None
        self._require_ready()
        # Import muộn để môi trường không phụ thuộc Pillow/matplotlib khi không vẽ.
        from environment.rendering.frame_drawing import draw_rgb_frame, draw_text_frame

        if self.render_mode == "ansi":
            return draw_text_frame(self._grid, self._robot_position, self._cleaned)
        return draw_rgb_frame(self._grid, self._robot_position, self._cleaned,
                              recent_positions=list(self._recent_positions))
