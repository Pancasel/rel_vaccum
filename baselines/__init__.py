"""Các baseline không dùng RL, cùng giao diện Agent với agent RL."""

from __future__ import annotations

from baselines.agent import Agent
from baselines.greedy_bfs import GreedyBFSAgent
from baselines.random_walk import RandomWalkAgent
from baselines.zigzag import ZigzagAgent

BASELINES = {"random_walk": RandomWalkAgent, "zigzag": ZigzagAgent, "greedy_bfs": GreedyBFSAgent}


def make_baseline(name: str, seed: int = 0) -> Agent:
    """Tạo baseline theo tên ("random_walk", "zigzag", "greedy_bfs").

    Raises:
        KeyError: tên không hợp lệ.
    """
    if name not in BASELINES:
        raise KeyError(f"Baseline '{name}' không tồn tại; chọn một trong {list(BASELINES)}.")
    if name == "random_walk":
        return RandomWalkAgent(seed=seed)
    return BASELINES[name]()


__all__ = ["Agent", "BASELINES", "GreedyBFSAgent", "RandomWalkAgent", "ZigzagAgent", "make_baseline"]
