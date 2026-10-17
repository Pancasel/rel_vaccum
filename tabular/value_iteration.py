"""Value Iteration (Sutton & Barto 2018, mục 4.4) — đáp án tối ưu chính xác trên lưới nhỏ.

Value Iteration không phải agent "học" từ trải nghiệm: nó dùng mô hình đầy đủ p(s', r | s, a)
(đồ thị trạng thái) để tính V* và π*. Dự án dùng nó làm đáp án để chấm Q-learning và DQN.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from env.grid_utils import decode_obs
from tabular.state_space import StateGraph
from utils.paths import assert_inside_root

TIE_TOLERANCE: float = 1e-9


@dataclass
class VIResult:
    """Kết quả Value Iteration."""

    graph: StateGraph
    V: np.ndarray  # (S,)
    Q: np.ndarray  # (S, 4)
    policy: np.ndarray  # (S,) chỉ số hành động tối ưu, phá hòa chọn chỉ số nhỏ nhất (D-20)
    n_iterations: int
    delta_history: list[float]
    gamma: float

    def value_of(self, start: tuple[int, int]) -> float:
        """V*(s0) của trạng thái xuất phát tại start."""
        return float(self.V[self.graph.start_index(start)])


def deterministic_argmax(q: np.ndarray) -> np.ndarray:
    """argmax theo hàng, các giá trị bằng nhau (sai số 1e-9) chọn hành động chỉ số nhỏ nhất."""
    best = q.max(axis=-1, keepdims=True)
    return np.argmax(q >= best - TIE_TOLERANCE, axis=-1)


def value_iteration(graph: StateGraph, gamma: float = 0.99, theta: float = 1e-10,
                    max_iterations: int = 100_000) -> VIResult:
    """Value Iteration trên đồ thị trạng thái.

    Raises:
        ValueError: gamma ngoài (0, 1] hoặc theta không dương.
        RuntimeError: chưa hội tụ sau max_iterations vòng.
    """
    if not 0.0 < gamma <= 1.0:
        raise ValueError(f"gamma phải trong (0, 1], nhận được {gamma}.")
    if theta <= 0:
        raise ValueError(f"theta phải dương, nhận được {theta}.")
    # Sutton & Barto 2018, mục 4.4: "Initialize V(s) arbitrarily for all s, except V(terminal) = 0"
    V = np.zeros(graph.n_states, dtype=np.float64)
    deltas: list[float] = []
    for iteration in range(1, max_iterations + 1):
        # BIẾN THỂ (D-21): cập nhật đồng bộ dạng vector (hai mảng) thay vì quét in-place từng s;
        # mục 4.1 của sách xác nhận dạng hai mảng cũng hội tụ, và cách này nhanh hơn hàng trăm lần.
        # "V(s) <- max_a Σ_{s',r} p(s',r|s,a)[r + γV(s')]" — MDP xác định nên tổng chỉ có 1 số hạng;
        # nếu s' là terminal thì V(s') = 0 (không bootstrap).
        Q = graph.reward + gamma * np.where(graph.terminal_next, 0.0, V[graph.next_state])
        V_new = Q.max(axis=1)
        V_new[graph.is_terminal] = 0.0
        # "Δ <- max(Δ, |v − V(s)|)" ... "until Δ < θ"
        delta = float(np.abs(V_new - V).max())
        V = V_new
        deltas.append(delta)
        if delta < theta:
            break
    else:
        raise RuntimeError(f"Value Iteration chưa hội tụ sau {max_iterations} vòng (Δ = {deltas[-1]:.3e}).")
    Q = graph.reward + gamma * np.where(graph.terminal_next, 0.0, V[graph.next_state])
    # "Output a deterministic policy, π ≈ π*, such that π(s) = argmax_a Σ p(s',r|s,a)[r + γV(s')]"
    policy = deterministic_argmax(Q)
    return VIResult(graph=graph, V=V, Q=Q, policy=policy, n_iterations=iteration, delta_history=deltas,
                    gamma=gamma)


def greedy_rollout(result: VIResult, start: tuple[int, int], max_steps: int) -> dict[str, Any]:
    """Đi theo π* từ start trên đồ thị; trả số bước, return chiết khấu, coverage, quỹ đạo vị trí."""
    graph = result.graph
    s = graph.start_index(start)
    trajectory = [start]
    ret_disc, discount, steps = 0.0, 1.0, 0
    while not graph.is_terminal[s] and steps < max_steps:
        a = int(result.policy[s])
        ret_disc += discount * graph.reward[s, a]
        discount *= result.gamma
        s = int(graph.next_state[s, a])
        steps += 1
        r, c, _ = graph.states[s]
        trajectory.append((r, c))
    mask = graph.states[s][2]
    coverage = bin(mask).count("1") / graph.encoder.n_free
    return {"steps": steps, "return_discounted": ret_disc, "coverage": coverage,
            "terminated": bool(graph.is_terminal[s]), "trajectory": trajectory}


class VIAgent:
    """Bọc π* thành Agent (act(obs)) để chạy chung bộ đánh giá với các agent khác."""

    def __init__(self, result: VIResult) -> None:
        self.result = result

    def act(self, obs: np.ndarray, info: dict | None = None) -> int:
        pos, cleaned, _ = decode_obs(obs)
        key = self.result.graph.encoder.encode(pos, cleaned)
        return int(self.result.policy[self.result.graph.index[key]])


def save_vi_result(result: VIResult, path: str | os.PathLike) -> Path:
    """Lưu kết quả VI (pickle) để các thí nghiệm sau dùng lại mà không phải tính lại."""
    resolved = assert_inside_root(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("wb") as f:
        pickle.dump(result, f)
    return resolved


def load_vi_result(path: str | os.PathLike) -> VIResult:
    """Đọc kết quả VI đã lưu bằng save_vi_result."""
    with Path(path).open("rb") as f:
        result = pickle.load(f)
    if not isinstance(result, VIResult):
        raise TypeError(f"File {path} không chứa VIResult.")
    return result
