"""Test cho các baseline."""

from __future__ import annotations

import numpy as np
import pytest

from baselines import BASELINES, GreedyBFSAgent, RandomWalkAgent, ZigzagAgent, make_baseline
from baselines.agent import Agent
from baselines.zigzag import snake_order
from env.maps import FIXED_MAPS, free_cells, generate_random_map, get_map
from env.vacuum_env import VacuumEnv
from experiments.evaluate import run_episode

SMALL_MAPS = ["empty_3x3", "obstacle_4x4", "furniture_8x8"]


@pytest.mark.parametrize("map_name", SMALL_MAPS)
@pytest.mark.parametrize("agent_name", list(BASELINES))
def test_every_baseline_runs_from_every_start(map_name: str, agent_name: str) -> None:
    env = VacuumEnv(map_name)
    for start in free_cells(get_map(map_name))[:10]:
        agent = make_baseline(agent_name, seed=0)
        assert isinstance(agent, Agent)
        result = run_episode(agent, env, seed=0, reset_options={"start": start})
        assert 0 < result["coverage"] <= 1


@pytest.mark.parametrize("map_name", sorted(FIXED_MAPS))
@pytest.mark.parametrize("agent_cls", [ZigzagAgent, GreedyBFSAgent])
def test_systematic_baselines_cover_fixed_maps(map_name: str, agent_cls: type) -> None:
    env = VacuumEnv(map_name)
    for seed in range(5):
        result = run_episode(agent_cls(), env, seed=seed)
        assert result["coverage"] == 1.0 and result["terminated"]


@pytest.mark.parametrize("agent_cls", [ZigzagAgent, GreedyBFSAgent])
def test_systematic_baselines_cover_random_maps(agent_cls: type) -> None:
    for seed in range(50):
        grid = generate_random_map(8, 8, 0.3, np.random.default_rng(seed))
        result = run_episode(agent_cls(), VacuumEnv(grid), seed=seed)
        assert result["terminated"], f"seed {seed}"


def test_zigzag_empty_3x3_is_perfect() -> None:
    result = run_episode(ZigzagAgent(), VacuumEnv("empty_3x3", start=(0, 0)), seed=0)
    assert result["steps"] == 8 and result["collisions"] == 0


def test_snake_order() -> None:
    assert snake_order(get_map("empty_3x3")) == [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0),
                                                 (2, 0), (2, 1), (2, 2)]


def test_greedy_never_collides() -> None:
    env = VacuumEnv("furniture_8x8")
    for seed in range(10):
        assert run_episode(GreedyBFSAgent(), env, seed=seed)["collisions"] == 0


def test_random_walk_valid_only_never_collides() -> None:
    env = VacuumEnv("furniture_8x8")
    for seed in range(5):
        assert run_episode(RandomWalkAgent(seed=seed), env, seed=seed)["collisions"] == 0


def test_deterministic_trajectories() -> None:
    env = VacuumEnv("obstacle_4x4")
    for name in BASELINES:
        a = run_episode(make_baseline(name, seed=3), env, seed=11, record_visits=True)
        b = run_episode(make_baseline(name, seed=3), env, seed=11, record_visits=True)
        assert np.array_equal(a["visits"], b["visits"]) and a["steps"] == b["steps"]


def test_make_baseline_unknown() -> None:
    with pytest.raises(KeyError):
        make_baseline("teleport")
