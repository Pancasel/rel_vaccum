"""Danh mục baseline: tạo agent theo tên."""

from __future__ import annotations

from agents.agent_interface import Agent
from agents.baselines.greedy_nearest_dirty_cell_agent import GreedyNearestDirtyCellAgent
from agents.baselines.random_walk_agent import RandomWalkAgent
from agents.baselines.zigzag_sweep_agent import ZigzagSweepAgent

# Tên khóa ("greedy_bfs"...) được ghi vào kết quả CSV nên giữ nguyên dù tên lớp đã đổi.
BASELINE_AGENT_CLASSES = {
    "random_walk": RandomWalkAgent,
    "zigzag": ZigzagSweepAgent,
    "greedy_bfs": GreedyNearestDirtyCellAgent,
}


def create_baseline_agent(baseline_name: str, seed: int = 0) -> Agent:
    """Tạo baseline theo tên ("random_walk", "zigzag", "greedy_bfs").

    Raises:
        KeyError: tên không hợp lệ.
    """
    if baseline_name not in BASELINE_AGENT_CLASSES:
        raise KeyError(
            f"Baseline '{baseline_name}' không tồn tại; chọn một trong {list(BASELINE_AGENT_CLASSES)}."
        )
    if baseline_name == "random_walk":
        return RandomWalkAgent(seed=seed)
    return BASELINE_AGENT_CLASSES[baseline_name]()
