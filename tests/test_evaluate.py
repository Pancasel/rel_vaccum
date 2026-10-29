"""Test cho experiments/evaluate.py."""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd
import pytest

from agents.baselines import GreedyNearestDirtyCellAgent
from environment.room_layouts import get_room_layout
from environment.vacuum_cleaning_env import VacuumCleaningEnv
from evaluation.metric_summary import PER_EPISODE_METRIC_COLUMNS, summarize_metrics
from evaluation.agent_evaluation import evaluate_agent, evaluate_agent_on_rooms
from evaluation.episode_runner import run_single_episode


def test_evaluate_greedy_3x3() -> None:
    factory = functools.partial(VacuumCleaningEnv, "empty_3x3", start=(0, 0))
    df = evaluate_agent(GreedyNearestDirtyCellAgent(), factory, episode_count=3, seed=0)
    assert list(df.columns) == PER_EPISODE_METRIC_COLUMNS
    assert len(df) == 3
    assert (df["coverage"] == 1.0).all()
    assert (df["steps"] == 8).all()
    assert (df["steps_to_full"] == 8).all()
    assert df["return_undiscounted"].iloc[0] == pytest.approx(17.92)
    assert df["return_discounted"].iloc[0] == pytest.approx(16.968928730705, abs=1e-6)
    assert list(df["env_seed"]) == [0, 1, 2]


def test_evaluate_validation() -> None:
    factory = functools.partial(VacuumCleaningEnv, "empty_3x3")
    with pytest.raises(ValueError):
        evaluate_agent(GreedyNearestDirtyCellAgent(), factory, episode_count=0, seed=0)
    with pytest.raises(ValueError):
        evaluate_agent(GreedyNearestDirtyCellAgent(), factory, episode_count=1, seed=-1)


class _CountingAgent:
    """Agent giả: đếm số lần reset, luôn đụng tường để kiểm tra truncated."""

    def __init__(self) -> None:
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def act(self, obs: np.ndarray, info: dict | None = None) -> int:
        return 0


def test_agent_reset_called_and_truncation_metrics() -> None:
    agent = _CountingAgent()
    factory = functools.partial(VacuumCleaningEnv, "empty_3x3", start=(0, 0), max_steps=5)
    df = evaluate_agent(agent, factory, episode_count=4, seed=1)
    assert agent.resets == 4
    assert (df["truncated"]).all() and not df["terminated"].any()
    assert df["steps_to_full"].isna().all()
    assert (df["collisions"] == 5).all()
    assert (df["redundancy"] == 0).all()


def test_run_episode_visits() -> None:
    result = run_single_episode(GreedyNearestDirtyCellAgent(), VacuumCleaningEnv("empty_3x3", start=(0, 0)), seed=0, record_visit_counts=True)
    assert result["visits"].sum() == 9  # ô xuất phát + 8 bước, mỗi ô đúng một lần
    with pytest.raises(ValueError):
        run_single_episode(GreedyNearestDirtyCellAgent(), VacuumCleaningEnv("empty_3x3"), seed=0, record_frames=True)


def test_evaluate_on_maps() -> None:
    maps = [get_room_layout("empty_3x3"), get_room_layout("obstacle_4x4")]
    df = evaluate_agent_on_rooms(GreedyNearestDirtyCellAgent(), maps, episodes_per_room=2, seed=0)
    assert len(df) == 4 and set(df["map_id"]) == {0, 1}
    assert (df["coverage"] == 1.0).all()


def test_summarize_values() -> None:
    df = pd.DataFrame({
        "group": ["a", "a", "b"],
        "coverage": [1.0, 0.5, 1.0],
        "steps": [10, 20, 30],
        "steps_to_full": [10.0, np.nan, 30.0],
        "terminated": [True, False, True],
    })
    out = summarize_metrics(df, ["group"]).set_index("group")
    assert out.loc["a", "coverage_mean"] == pytest.approx(0.75)
    assert out.loc["a", "steps_std"] == pytest.approx(np.std([10, 20], ddof=1))
    assert out.loc["a", "steps_to_full_mean"] == pytest.approx(10.0)
    assert out.loc["a", "success_rate"] == pytest.approx(0.5)
    assert out.loc["b", "steps_std"] == 0.0  # nhóm một phần tử
    overall = summarize_metrics(df)
    assert overall["n"].iloc[0] == 3


def test_run_parallel_order_and_env_restored() -> None:
    import os

    from experiments.common import run_parallel

    before = os.environ.get("OMP_NUM_THREADS")
    tasks = [{"i": i} for i in range(4)]
    assert run_parallel(dict, tasks, workers=2) == tasks  # dict(**task) == task, giữ đúng thứ tự
    assert run_parallel(dict, tasks, workers=1) == tasks
    assert os.environ.get("OMP_NUM_THREADS") == before
