"""Các baseline không dùng RL, cùng giao diện Agent với agent RL."""

from __future__ import annotations

from agents.agent_interface import Agent
from agents.baselines.baseline_registry import BASELINE_AGENT_CLASSES, create_baseline_agent
from agents.baselines.greedy_nearest_dirty_cell_agent import GreedyNearestDirtyCellAgent
from agents.baselines.random_walk_agent import RandomWalkAgent
from agents.baselines.zigzag_sweep_agent import ZigzagSweepAgent

__all__ = ["Agent", "BASELINE_AGENT_CLASSES", "GreedyNearestDirtyCellAgent", "RandomWalkAgent",
           "ZigzagSweepAgent", "create_baseline_agent"]
