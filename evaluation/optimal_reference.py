"""Đáp án tối ưu chính xác (tính bằng đồ thị trạng thái) để chấm các agent, chỉ cho bản đồ nhỏ."""

from __future__ import annotations

import numpy as np

from environment.reward_settings import RewardSettings
from environment.room_layouts import list_free_cells
from environment.vacuum_cleaning_env import VacuumCleaningEnv
from evaluation.episode_runner import run_single_episode

# Giới hạn số ô trống để còn tính được đáp án tối ưu chính xác bằng BFS trên đồ thị trạng thái.
MAX_FREE_CELLS_FOR_EXACT_OPTIMUM: int = 16


def can_compute_exact_optimum(grid: np.ndarray) -> bool:
    """True nếu bản đồ đủ nhỏ để tính đáp án tối ưu chính xác."""
    return len(list_free_cells(grid)) <= MAX_FREE_CELLS_FOR_EXACT_OPTIMUM


def optimal_steps_for_each_start(grid: np.ndarray) -> dict[tuple[int, int], int] | None:
    """Số bước tối thiểu để dọn hết từ mỗi ô xuất phát (None nếu bản đồ quá lớn)."""
    if not can_compute_exact_optimum(grid):
        return None
    from agents.tabular.state_transition_graph import build_state_transition_graph, minimum_steps_to_clean_all

    graph = build_state_transition_graph(grid)
    return {start_cell: minimum_steps_to_clean_all(graph, start_cell) for start_cell in list_free_cells(grid)}


def optimal_value_of_start(grid: np.ndarray, start_cell: tuple[int, int], reward_settings: RewardSettings,
                           gamma: float) -> float | None:
    """V*(s0) bằng Value Iteration với đúng cấu hình reward và γ (None nếu bản đồ quá lớn)."""
    if not can_compute_exact_optimum(grid):
        return None
    from agents.tabular.state_transition_graph import build_state_transition_graph
    from agents.tabular.value_iteration import run_value_iteration

    graph = build_state_transition_graph(grid, reward_settings)
    return run_value_iteration(graph, gamma=gamma).optimal_value_of_start(start_cell)


def reference_step_counts(grid: np.ndarray, start_cell: tuple[int, int] | None) -> dict[str, float]:
    """Số bước tham chiếu để vẽ lên learning curve: tối ưu, greedy_bfs, zigzag.

    start_cell=None (xuất phát ngẫu nhiên) thì lấy trung bình trên mọi ô xuất phát.
    """
    from agents.baselines import GreedyNearestDirtyCellAgent, ZigzagSweepAgent

    start_cells = [start_cell] if start_cell is not None else list_free_cells(grid)
    env = VacuumCleaningEnv(grid)
    reference_steps: dict[str, float] = {}
    optimal_steps = optimal_steps_for_each_start(grid)
    if optimal_steps is not None:
        reference_steps["tối ưu (VI)"] = float(np.mean([optimal_steps[cell] for cell in start_cells]))
    for baseline_name, baseline_agent in (("greedy_bfs", GreedyNearestDirtyCellAgent()),
                                          ("zigzag", ZigzagSweepAgent())):
        steps_per_start = [
            run_single_episode(baseline_agent, env, seed=0, reset_options={"start": cell})["steps"]
            for cell in start_cells
        ]
        reference_steps[baseline_name] = float(np.mean(steps_per_start))
    return reference_steps
