"""Value Iteration (Sutton & Barto 2018, mục 4.4) — đáp án tối ưu chính xác trên lưới nhỏ.

Value Iteration không phải agent "học" từ trải nghiệm: nó dùng mô hình đầy đủ p(s', r | s, a)
(đồ thị trạng thái) để tính V* và π*. Dự án dùng nó làm đáp án để chấm Q-learning và DQN.

Ký hiệu: V* = giá trị tối ưu của trạng thái, Q* = giá trị tối ưu của cặp (trạng thái, hành động),
π* = chính sách tối ưu, γ (gamma) = hệ số chiết khấu, θ (theta) = ngưỡng dừng.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from agents.tabular.state_transition_graph import StateTransitionGraph
from environment.observation_decoding import decode_observation
from project_utils.project_paths import require_path_inside_project

# Hai giá trị Q chênh nhau không quá ngưỡng này thì coi là bằng nhau khi chọn hành động tốt nhất.
TIE_TOLERANCE: float = 1e-9


@dataclass
class ValueIterationResult:
    """Kết quả Value Iteration."""

    graph: StateTransitionGraph
    state_values: np.ndarray  # V*, shape (S,)
    action_values: np.ndarray  # Q*, shape (S, 4)
    policy: np.ndarray  # π*, shape (S,): chỉ số hành động tối ưu, phá hòa chọn chỉ số nhỏ nhất (D-20)
    iteration_count: int
    max_change_history: list[float]  # Δ (mức thay đổi lớn nhất của V) sau mỗi vòng lặp
    gamma: float

    def optimal_value_of_start(self, start_cell: tuple[int, int]) -> float:
        """V*(s0) của trạng thái xuất phát tại start_cell."""
        return float(self.state_values[self.graph.start_state_index(start_cell)])


def argmax_with_lowest_index_tie_break(action_values: np.ndarray) -> np.ndarray:
    """argmax theo hàng, các giá trị bằng nhau (sai số 1e-9) chọn hành động chỉ số nhỏ nhất."""
    best_values = action_values.max(axis=-1, keepdims=True)
    return np.argmax(action_values >= best_values - TIE_TOLERANCE, axis=-1)


def run_value_iteration(graph: StateTransitionGraph, gamma: float = 0.99, theta: float = 1e-10,
                        max_iterations: int = 100_000) -> ValueIterationResult:
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
    state_values = np.zeros(graph.state_count, dtype=np.float64)
    max_change_history: list[float] = []
    for iteration in range(1, max_iterations + 1):
        # BIẾN THỂ (D-21): cập nhật đồng bộ dạng vector (hai mảng) thay vì quét in-place từng s;
        # mục 4.1 của sách xác nhận dạng hai mảng cũng hội tụ, và cách này nhanh hơn hàng trăm lần.
        # "V(s) <- max_a Σ_{s',r} p(s',r|s,a)[r + γV(s')]" — MDP xác định nên tổng chỉ có 1 số hạng;
        # nếu s' là terminal thì V(s') = 0 (không bootstrap).
        value_of_next_state = np.where(graph.next_state_is_terminal, 0.0,
                                       state_values[graph.next_state_indices])
        action_values = graph.rewards + gamma * value_of_next_state
        new_state_values = action_values.max(axis=1)
        new_state_values[graph.is_terminal_state] = 0.0
        # "Δ <- max(Δ, |v − V(s)|)" ... "until Δ < θ"
        max_change = float(np.abs(new_state_values - state_values).max())
        state_values = new_state_values
        max_change_history.append(max_change)
        if max_change < theta:
            break
    else:
        raise RuntimeError(
            f"Value Iteration chưa hội tụ sau {max_iterations} vòng (Δ = {max_change_history[-1]:.3e})."
        )
    value_of_next_state = np.where(graph.next_state_is_terminal, 0.0, state_values[graph.next_state_indices])
    action_values = graph.rewards + gamma * value_of_next_state
    # "Output a deterministic policy, π ≈ π*, such that π(s) = argmax_a Σ p(s',r|s,a)[r + γV(s')]"
    policy = argmax_with_lowest_index_tie_break(action_values)
    return ValueIterationResult(graph=graph, state_values=state_values, action_values=action_values,
                                policy=policy, iteration_count=iteration,
                                max_change_history=max_change_history, gamma=gamma)


def follow_optimal_policy(result: ValueIterationResult, start_cell: tuple[int, int],
                          max_steps: int) -> dict[str, Any]:
    """Đi theo π* từ start_cell trên đồ thị; trả số bước, return chiết khấu, coverage, quỹ đạo vị trí."""
    graph = result.graph
    state_index = graph.start_state_index(start_cell)
    trajectory = [start_cell]
    discounted_return, discount_factor, step_count = 0.0, 1.0, 0
    while not graph.is_terminal_state[state_index] and step_count < max_steps:
        action = int(result.policy[state_index])
        discounted_return += discount_factor * graph.rewards[state_index, action]
        discount_factor *= result.gamma
        state_index = int(graph.next_state_indices[state_index, action])
        step_count += 1
        robot_row, robot_column, _ = graph.state_keys[state_index]
        trajectory.append((robot_row, robot_column))
    cleaned_bitmask = graph.state_keys[state_index][2]
    coverage = bin(cleaned_bitmask).count("1") / graph.state_encoder.free_cell_count
    return {"steps": step_count, "return_discounted": discounted_return, "coverage": coverage,
            "terminated": bool(graph.is_terminal_state[state_index]), "trajectory": trajectory}


class ValueIterationPolicyAgent:
    """Bọc π* thành Agent (act(observation)) để chạy chung bộ đánh giá với các agent khác."""

    def __init__(self, result: ValueIterationResult) -> None:
        self.result = result

    def act(self, observation: np.ndarray, info: dict | None = None) -> int:
        robot_position, cleaned, _ = decode_observation(observation)
        state_key = self.result.graph.state_encoder.encode_state(robot_position, cleaned)
        return int(self.result.policy[self.result.graph.state_index_by_key[state_key]])


def save_value_iteration_result(result: ValueIterationResult, path: str | os.PathLike) -> Path:
    """Lưu kết quả Value Iteration (pickle) để các thí nghiệm sau dùng lại mà không phải tính lại."""
    resolved_path = require_path_inside_project(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("wb") as output_file:
        pickle.dump(result, output_file)
    return resolved_path


def load_value_iteration_result(path: str | os.PathLike) -> ValueIterationResult:
    """Đọc kết quả Value Iteration đã lưu bằng save_value_iteration_result."""
    with Path(path).open("rb") as input_file:
        result = pickle.load(input_file)
    if not isinstance(result, ValueIterationResult):
        raise TypeError(f"File {path} không chứa ValueIterationResult.")
    return result
