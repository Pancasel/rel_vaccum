"""Mã hóa trạng thái thành khóa hashable (dùng làm khóa của bảng Q)."""

from __future__ import annotations

import numpy as np

from environment.observation_decoding import decode_observation
from environment.room_layouts import list_free_cells

# Bitmask dùng int64 nên tối đa 62 ô trống (chừa bit dấu và an toàn khi cộng).
MAX_TABULAR_FREE_CELLS: int = 62


class StateKeyEncoder:
    """Mã hóa trạng thái thành khóa hashable cho bảng Q (D-12, D-13).

    - state_mode="full": khóa (row, col, bitmask), bitmask là số nguyên mà bit thứ i bằng 1 khi
      ô trống thứ i đã dọn; ô trống được đánh số theo thứ tự `list_free_cells` (hàng trước cột sau).
    - state_mode="position_only": khóa (row, col), cố ý bỏ thông tin đã dọn cho thí nghiệm E1.
    """

    STATE_MODES: tuple[str, ...] = ("full", "position_only")

    def __init__(self, grid: np.ndarray, state_mode: str = "full") -> None:
        """
        Args:
            grid: bản đồ (H, W).
            state_mode: "full" hoặc "position_only".

        Raises:
            ValueError: state_mode lạ, hoặc quá nhiều ô trống cho bitmask.
        """
        if state_mode not in self.STATE_MODES:
            raise ValueError(f"state_mode phải là một trong {self.STATE_MODES}, nhận được '{state_mode}'.")
        free_cells = list_free_cells(grid)
        if state_mode == "full" and len(free_cells) > MAX_TABULAR_FREE_CELLS:
            raise ValueError(
                f"Bản đồ có {len(free_cells)} ô trống, vượt giới hạn {MAX_TABULAR_FREE_CELLS} của bitmask tabular."
            )
        self.grid: np.ndarray = grid.copy()
        self.state_mode: str = state_mode
        self.free_cells: list[tuple[int, int]] = free_cells
        self.free_cell_index: dict[tuple[int, int], int] = {cell: index for index, cell in enumerate(free_cells)}
        self.free_cell_count: int = len(free_cells)
        self.all_cleaned_bitmask: int = (1 << self.free_cell_count) - 1
        grid_width = grid.shape[1]
        self._free_cell_flat_indices = np.array([row * grid_width + column for row, column in free_cells],
                                                dtype=np.int64)
        self._bit_weights = np.left_shift(np.int64(1), np.arange(self.free_cell_count, dtype=np.int64))

    def cleaned_cells_to_bitmask(self, cleaned: np.ndarray) -> int:
        """Bitmask của mảng đã dọn (H, W) bool."""
        cleaned_bits = cleaned.ravel()[self._free_cell_flat_indices].astype(np.int64)
        return int((cleaned_bits * self._bit_weights).sum())

    def bitmask_to_cleaned_cells(self, bitmask: int) -> np.ndarray:
        """Ngược với cleaned_cells_to_bitmask: bitmask -> mảng đã dọn (H, W) bool."""
        if not 0 <= bitmask <= self.all_cleaned_bitmask:
            raise ValueError(f"mask {bitmask} ngoài khoảng [0, {self.all_cleaned_bitmask}].")
        cleaned = np.zeros(self.grid.shape, dtype=bool)
        for cell_index, (row, column) in enumerate(self.free_cells):
            if (bitmask >> cell_index) & 1:
                cleaned[row, column] = True
        return cleaned

    def encode_state(self, robot_position: tuple[int, int], cleaned: np.ndarray) -> tuple:
        """Khóa trạng thái từ vị trí robot và mảng đã dọn."""
        if self.state_mode == "position_only":
            return (int(robot_position[0]), int(robot_position[1]))
        return (int(robot_position[0]), int(robot_position[1]), self.cleaned_cells_to_bitmask(cleaned))

    def encode_observation(self, observation: np.ndarray) -> tuple:
        """Khóa trạng thái trực tiếp từ observation (3, H, W)."""
        robot_position, cleaned, _ = decode_observation(observation)
        return self.encode_state(robot_position, cleaned)
