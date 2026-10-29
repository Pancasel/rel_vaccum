"""Test cho các baseline."""

from __future__ import annotations

import numpy as np
import pytest

from agents.baselines import BASELINE_AGENT_CLASSES, GreedyNearestDirtyCellAgent, RandomWalkAgent, ZigzagSweepAgent, create_baseline_agent
from agents.agent_interface import Agent
from agents.baselines.zigzag_sweep_agent import snake_scan_order
from environment.room_layouts import FIXED_ROOM_LAYOUTS, list_free_cells, get_room_layout
from environment.random_room_generator import generate_random_room
from environment.vacuum_cleaning_env import VacuumCleaningEnv
from evaluation.episode_runner import run_single_episode

SMALL_MAPS = ["empty_3x3", "obstacle_4x4", "furniture_8x8"]


@pytest.mark.parametrize("map_name", SMALL_MAPS)
@pytest.mark.parametrize("agent_name", list(BASELINE_AGENT_CLASSES))
def test_every_baseline_runs_from_every_start(map_name: str, agent_name: str) -> None:
    env = VacuumCleaningEnv(map_name)
    for start in list_free_cells(get_room_layout(map_name))[:10]:
        agent = create_baseline_agent(agent_name, seed=0)
        assert isinstance(agent, Agent)
        result = run_single_episode(agent, env, seed=0, reset_options={"start": start})
        assert 0 < result["coverage"] <= 1


@pytest.mark.parametrize("map_name", sorted(FIXED_ROOM_LAYOUTS))
@pytest.mark.parametrize("agent_cls", [ZigzagSweepAgent, GreedyNearestDirtyCellAgent])
def test_systematic_baselines_cover_fixed_maps(map_name: str, agent_cls: type) -> None:
    env = VacuumCleaningEnv(map_name)
    for seed in range(5):
        result = run_single_episode(agent_cls(), env, seed=seed)
        assert result["coverage"] == 1.0 and result["terminated"]


@pytest.mark.parametrize("agent_cls", [ZigzagSweepAgent, GreedyNearestDirtyCellAgent])
def test_systematic_baselines_cover_random_maps(agent_cls: type) -> None:
    for seed in range(50):
        grid = generate_random_room(8, 8, 0.3, np.random.default_rng(seed))
        result = run_single_episode(agent_cls(), VacuumCleaningEnv(grid), seed=seed)
        assert result["terminated"], f"seed {seed}"


def test_zigzag_empty_3x3_is_perfect() -> None:
    result = run_single_episode(ZigzagSweepAgent(), VacuumCleaningEnv("empty_3x3", start=(0, 0)), seed=0)
    assert result["steps"] == 8 and result["collisions"] == 0


def test_snake_order() -> None:
    assert snake_scan_order(get_room_layout("empty_3x3")) == [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0),
                                                 (2, 0), (2, 1), (2, 2)]


def test_greedy_never_collides() -> None:
    env = VacuumCleaningEnv("furniture_8x8")
    for seed in range(10):
        assert run_single_episode(GreedyNearestDirtyCellAgent(), env, seed=seed)["collisions"] == 0


def test_random_walk_valid_only_never_collides() -> None:
    env = VacuumCleaningEnv("furniture_8x8")
    for seed in range(5):
        assert run_single_episode(RandomWalkAgent(seed=seed), env, seed=seed)["collisions"] == 0


def test_deterministic_trajectories() -> None:
    env = VacuumCleaningEnv("obstacle_4x4")
    for name in BASELINE_AGENT_CLASSES:
        a = run_single_episode(create_baseline_agent(name, seed=3), env, seed=11, record_visit_counts=True)
        b = run_single_episode(create_baseline_agent(name, seed=3), env, seed=11, record_visit_counts=True)
        assert np.array_equal(a["visits"], b["visits"]) and a["steps"] == b["steps"]


def test_make_baseline_unknown() -> None:
    with pytest.raises(KeyError):
        create_baseline_agent("teleport")
