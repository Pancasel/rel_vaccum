"""Đồ thị trạng thái của MDP trên một bản đồ cố định, và số bước tối thiểu để dọn hết.

Trạng thái tabular = (row, col, bitmask) (D-12). Mọi cạnh của đồ thị được sinh bằng hàm
`compute_transition()` của môi trường (D-11) nên đồ thị khớp tuyệt đối với VacuumCleaningEnv.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from agents.tabular.state_key_encoder import StateKeyEncoder
from environment.reward_settings import RewardSettings
from environment.room_layouts import list_free_cells
from environment.transition_model import compute_transition

ACTION_COUNT: int = 4
StateKey = tuple[int, int, int]  # (hàng robot, cột robot, bitmask các ô đã dọn)


@dataclass
class StateTransitionGraph:
    """Đồ thị trạng thái hữu hạn của MDP trên một bản đồ cố định.

    Trạng thái được đánh số 0..S-1. Các mảng bên dưới có một hàng cho mỗi trạng thái,
    mỗi cột là một hành động (0=lên, 1=xuống, 2=trái, 3=phải).
    """

    state_encoder: StateKeyEncoder
    state_keys: list[StateKey]  # chỉ số trạng thái -> khóa (hàng, cột, bitmask)
    state_index_by_key: dict[StateKey, int]  # khóa -> chỉ số trạng thái
    next_state_indices: np.ndarray  # (S, 4) int64: chỉ số trạng thái sau khi làm hành động
    rewards: np.ndarray  # (S, 4) float64: phần thưởng nhận được
    next_state_is_terminal: np.ndarray  # (S, 4) bool: trạng thái kế tiếp là terminal (đã dọn hết)
    is_terminal_state: np.ndarray  # (S,) bool

    @property
    def state_count(self) -> int:
        return len(self.state_keys)

    def start_state_index(self, start_cell: tuple[int, int]) -> int:
        """Chỉ số của trạng thái khởi đầu: robot ở start_cell, chỉ ô start_cell đã dọn."""
        state_key = (start_cell[0], start_cell[1], 1 << self.state_encoder.free_cell_index[tuple(start_cell)])
        if state_key not in self.state_index_by_key:
            raise KeyError(f"Ô xuất phát {start_cell} không có trong đồ thị (không phải ô trống?).")
        return self.state_index_by_key[state_key]


def build_state_transition_graph(grid: np.ndarray, reward_settings: RewardSettings | None = None,
                                 start_cells: list[tuple[int, int]] | None = None,
                                 max_states: int = 5_000_000) -> StateTransitionGraph:
    """BFS từ các trạng thái khởi đầu để liệt kê mọi trạng thái ĐẠT TỚI ĐƯỢC.

    Trạng thái terminal (đã dọn hết) là trạng thái hấp thụ: hàng của nó trỏ về chính nó với
    reward 0, để Value Iteration không cộng thêm gì sau khi kết thúc.

    Args:
        grid: bản đồ.
        reward_settings: cấu hình reward (None = dense).
        start_cells: các ô xuất phát (None = mọi ô trống).
        max_states: vượt quá thì dừng.

    Raises:
        RuntimeError: số trạng thái vượt max_states — đúng động cơ chuyển sang deep RL (Giai đoạn 5).
    """
    reward_settings = reward_settings or RewardSettings.dense()
    state_encoder = StateKeyEncoder(grid, "full")
    start_cells = start_cells if start_cells is not None else list_free_cells(grid)
    state_keys: list[StateKey] = []
    state_index_by_key: dict[StateKey, int] = {}
    states_to_expand: deque[int] = deque()
    for start_cell in start_cells:
        start_key = (start_cell[0], start_cell[1], 1 << state_encoder.free_cell_index[tuple(start_cell)])
        if start_key not in state_index_by_key:
            state_index_by_key[start_key] = len(state_keys)
            state_keys.append(start_key)
            states_to_expand.append(state_index_by_key[start_key])

    next_state_rows: list[list[int]] = []
    reward_rows: list[list[float]] = []
    next_is_terminal_rows: list[list[bool]] = []
    terminal_flags: list[bool] = []
    # BFS theo thứ tự chỉ số: trạng thái thứ i được xử lý đúng lần thứ i, nên hàng i của các
    # mảng luôn ứng với trạng thái i.
    while states_to_expand:
        state_index = states_to_expand.popleft()
        robot_row, robot_column, cleaned_bitmask = state_keys[state_index]
        if cleaned_bitmask == state_encoder.all_cleaned_bitmask:
            next_state_rows.append([state_index] * ACTION_COUNT)
            reward_rows.append([0.0] * ACTION_COUNT)
            next_is_terminal_rows.append([True] * ACTION_COUNT)
            terminal_flags.append(True)
            continue
        cleaned = state_encoder.bitmask_to_cleaned_cells(cleaned_bitmask)
        next_states_of_row: list[int] = []
        rewards_of_row: list[float] = []
        next_is_terminal_of_row: list[bool] = []
        for action in range(ACTION_COUNT):
            outcome = compute_transition(grid, (robot_row, robot_column), cleaned, action, reward_settings)
            next_key = state_encoder.encode_state(outcome.next_position, outcome.next_cleaned)
            if next_key not in state_index_by_key:
                if len(state_keys) >= max_states:
                    raise RuntimeError(
                        f"Vượt {max_states} trạng thái: lưới quá lớn cho tabular (đây là lý do cần deep RL)."
                    )
                state_index_by_key[next_key] = len(state_keys)
                state_keys.append(next_key)
                states_to_expand.append(state_index_by_key[next_key])
            next_states_of_row.append(state_index_by_key[next_key])
            rewards_of_row.append(outcome.reward)
            next_is_terminal_of_row.append(outcome.terminated)
        next_state_rows.append(next_states_of_row)
        reward_rows.append(rewards_of_row)
        next_is_terminal_rows.append(next_is_terminal_of_row)
        terminal_flags.append(False)

    return StateTransitionGraph(
        state_encoder=state_encoder, state_keys=state_keys, state_index_by_key=state_index_by_key,
        next_state_indices=np.array(next_state_rows, dtype=np.int64),
        rewards=np.array(reward_rows, dtype=np.float64),
        next_state_is_terminal=np.array(next_is_terminal_rows, dtype=bool),
        is_terminal_state=np.array(terminal_flags, dtype=bool),
    )


def minimum_steps_to_clean_all(graph: StateTransitionGraph, start_cell: tuple[int, int]) -> int:
    """Số bước ÍT NHẤT để dọn hết từ start_cell (BFS không trọng số trên đồ thị trạng thái, D-22).

    Raises:
        RuntimeError: không tới được trạng thái terminal (không xảy ra trên bản đồ liên thông).
    """
    source_state = graph.start_state_index(start_cell)
    steps_from_source = np.full(graph.state_count, -1, dtype=np.int64)
    steps_from_source[source_state] = 0
    states_to_expand: deque[int] = deque([source_state])
    while states_to_expand:
        state_index = states_to_expand.popleft()
        if graph.is_terminal_state[state_index]:
            return int(steps_from_source[state_index])
        for next_state in graph.next_state_indices[state_index]:
            if steps_from_source[next_state] < 0:
                steps_from_source[next_state] = steps_from_source[state_index] + 1
                states_to_expand.append(int(next_state))
    raise RuntimeError("Không tới được trạng thái đã dọn hết.")
