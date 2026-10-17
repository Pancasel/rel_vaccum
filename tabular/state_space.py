"""Liệt kê không gian trạng thái đạt tới được và tính số bước tối thiểu để dọn hết.

Trạng thái tabular = (row, col, mask) (D-12). Mọi cạnh của đồ thị được sinh bằng hàm
`transition()` của môi trường (D-11) nên đồ thị khớp tuyệt đối với VacuumEnv.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from env.grid_utils import ACTION_DELTAS, StateEncoder
from env.maps import FREE, free_cells
from env.vacuum_env import RewardConfig, transition

N_ACTIONS: int = 4
# Giới hạn kích thước mảng "đã thăm" khi đếm trạng thái bằng numpy (1 byte mỗi trạng thái).
MAX_COUNT_ARRAY: int = 2**28


@dataclass
class StateGraph:
    """Đồ thị trạng thái hữu hạn của MDP trên một bản đồ cố định."""

    encoder: StateEncoder
    states: list[tuple[int, int, int]]  # chỉ số -> khóa (r, c, mask)
    index: dict[tuple[int, int, int], int]  # khóa -> chỉ số
    next_state: np.ndarray  # (S, 4) int64
    reward: np.ndarray  # (S, 4) float64
    terminal_next: np.ndarray  # (S, 4) bool: trạng thái kế tiếp là terminal
    is_terminal: np.ndarray  # (S,) bool

    @property
    def n_states(self) -> int:
        return len(self.states)

    def start_index(self, start: tuple[int, int]) -> int:
        """Chỉ số của trạng thái khởi đầu: robot ở start, chỉ ô start đã dọn."""
        key = (start[0], start[1], 1 << self.encoder.free_index[tuple(start)])
        if key not in self.index:
            raise KeyError(f"Ô xuất phát {start} không có trong đồ thị (không phải ô trống?).")
        return self.index[key]


def build_state_graph(grid: np.ndarray, cfg: RewardConfig | None = None,
                      starts: list[tuple[int, int]] | None = None, max_states: int = 5_000_000) -> StateGraph:
    """BFS từ các trạng thái khởi đầu để liệt kê mọi trạng thái ĐẠT TỚI ĐƯỢC.

    Trạng thái terminal (đã dọn hết) là trạng thái hấp thụ: hàng của nó trỏ về chính nó với
    reward 0, để Value Iteration không cộng thêm gì sau khi kết thúc.

    Args:
        grid: bản đồ.
        cfg: cấu hình reward (None = dense).
        starts: các ô xuất phát (None = mọi ô trống).
        max_states: vượt quá thì dừng.

    Raises:
        RuntimeError: số trạng thái vượt max_states — đúng động cơ chuyển sang deep RL (Giai đoạn 5).
    """
    cfg = cfg or RewardConfig.dense()
    encoder = StateEncoder(grid, "full")
    starts = starts if starts is not None else free_cells(grid)
    states: list[tuple[int, int, int]] = []
    index: dict[tuple[int, int, int], int] = {}
    queue: deque[int] = deque()
    for start in starts:
        key = (start[0], start[1], 1 << encoder.free_index[tuple(start)])
        if key not in index:
            index[key] = len(states)
            states.append(key)
            queue.append(index[key])

    next_rows: list[list[int]] = []
    reward_rows: list[list[float]] = []
    term_rows: list[list[bool]] = []
    terminal_flags: list[bool] = []
    # BFS theo thứ tự chỉ số: trạng thái thứ i được xử lý đúng lần thứ i, nên hàng i của các
    # mảng luôn ứng với trạng thái i.
    while queue:
        s = queue.popleft()
        r, c, mask = states[s]
        if mask == encoder.full_mask:
            next_rows.append([s] * N_ACTIONS)
            reward_rows.append([0.0] * N_ACTIONS)
            term_rows.append([True] * N_ACTIONS)
            terminal_flags.append(True)
            continue
        cleaned = encoder.decode_mask(mask)
        row_next, row_rew, row_term = [], [], []
        for action in range(N_ACTIONS):
            result = transition(grid, (r, c), cleaned, action, cfg)
            nkey = encoder.encode(result.next_pos, result.next_cleaned)
            if nkey not in index:
                if len(states) >= max_states:
                    raise RuntimeError(
                        f"Vượt {max_states} trạng thái: lưới quá lớn cho tabular (đây là lý do cần deep RL)."
                    )
                index[nkey] = len(states)
                states.append(nkey)
                queue.append(index[nkey])
            row_next.append(index[nkey])
            row_rew.append(result.reward)
            row_term.append(result.terminated)
        next_rows.append(row_next)
        reward_rows.append(row_rew)
        term_rows.append(row_term)
        terminal_flags.append(False)

    return StateGraph(
        encoder=encoder, states=states, index=index,
        next_state=np.array(next_rows, dtype=np.int64),
        reward=np.array(reward_rows, dtype=np.float64),
        terminal_next=np.array(term_rows, dtype=bool),
        is_terminal=np.array(terminal_flags, dtype=bool),
    )


def min_steps_to_cover(graph: StateGraph, start: tuple[int, int]) -> int:
    """Số bước ÍT NHẤT để dọn hết từ start (BFS không trọng số trên đồ thị trạng thái, D-22).

    Raises:
        RuntimeError: không tới được trạng thái terminal (không xảy ra trên bản đồ liên thông).
    """
    source = graph.start_index(start)
    dist = np.full(graph.n_states, -1, dtype=np.int64)
    dist[source] = 0
    queue: deque[int] = deque([source])
    while queue:
        s = queue.popleft()
        if graph.is_terminal[s]:
            return int(dist[s])
        for s_next in graph.next_state[s]:
            if dist[s_next] < 0:
                dist[s_next] = dist[s] + 1
                queue.append(int(s_next))
    raise RuntimeError("Không tới được trạng thái đã dọn hết.")


def count_reachable_states(grid: np.ndarray, time_limit_s: float = 600.0) -> int | None:
    """Đếm số trạng thái đạt tới được (mọi ô xuất phát), dùng cho E7.

    Cài bằng BFS theo tầng trên mảng numpy (mã trạng thái = pos * 2^|F| + mask) để đếm
    được tới hàng triệu trạng thái. Chỉ đếm, không tính reward; động lực học giống hệt
    transition() (đi vào ô trống, đụng tường thì đứng yên) — test so khớp với build_state_graph.

    Returns:
        Số trạng thái, hoặc None nếu quá lớn (mảng > MAX_COUNT_ARRAY) hoặc quá thời gian.
    """
    cells = free_cells(grid)
    n_free = len(cells)
    if n_free * (1 << n_free) > MAX_COUNT_ARRAY:
        return None
    index_of = {cell: i for i, cell in enumerate(cells)}
    # neighbour[i, a] = chỉ số ô trống sau khi đi hướng a từ ô i (đụng tường -> giữ nguyên i).
    neighbour = np.zeros((n_free, N_ACTIONS), dtype=np.int64)
    for i, (r, c) in enumerate(cells):
        for a, (dr, dc) in enumerate(ACTION_DELTAS):
            nr, nc = r + dr, c + dc
            inside = 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] == FREE
            neighbour[i, a] = index_of[(nr, nc)] if inside else i
    full = (1 << n_free) - 1
    visited = np.zeros(n_free << n_free, dtype=bool)
    pos = np.arange(n_free, dtype=np.int64)
    frontier = (pos << n_free) | (np.int64(1) << pos)
    visited[frontier] = True
    count = len(frontier)
    deadline = time.monotonic() + time_limit_s
    while len(frontier):
        if time.monotonic() > deadline:
            return None
        masks = frontier & full
        frontier = frontier[masks != full]  # trạng thái terminal không mở rộng tiếp
        positions = frontier >> n_free
        masks = frontier & full
        nxt_pos = neighbour[positions].reshape(-1)
        nxt_mask = np.repeat(masks, N_ACTIONS) | (np.int64(1) << nxt_pos)
        codes = np.unique((nxt_pos << n_free) | nxt_mask)
        codes = codes[~visited[codes]]
        visited[codes] = True
        count += len(codes)
        frontier = codes
    return int(count)
