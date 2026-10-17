"""Test cho tabular/state_space.py và tabular/value_iteration.py (đáp án tối ưu)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from baselines import GreedyBFSAgent, ZigzagAgent
from env.maps import free_cells, get_map, parse_map
from env.vacuum_env import VacuumEnv
from experiments.evaluate import run_episode
from tabular.state_space import build_state_graph, count_reachable_states, min_steps_to_cover
from tabular.value_iteration import (VIAgent, greedy_rollout, load_vi_result, save_vi_result,
                                     value_iteration)

V_STAR_3X3_CORNER = 16.968928730705  # phụ lục 10.1


@pytest.fixture(scope="module")
def vi_3x3():
    return value_iteration(build_state_graph(get_map("empty_3x3")))


@pytest.fixture(scope="module")
def vi_4x4():
    return value_iteration(build_state_graph(get_map("obstacle_4x4")))


def test_state_keys_contain_robot_bit(vi_3x3) -> None:
    graph = vi_3x3.graph
    assert graph.n_states > 0
    for r, c, mask in graph.states:
        assert (mask >> graph.encoder.free_index[(r, c)]) & 1


def test_v_star_corner(vi_3x3) -> None:
    assert vi_3x3.value_of((0, 0)) == pytest.approx(V_STAR_3X3_CORNER, abs=1e-6)


def test_min_steps_checkerboard(vi_3x3) -> None:
    graph = vi_3x3.graph
    assert min_steps_to_cover(graph, (0, 0)) == 8
    assert min_steps_to_cover(graph, (1, 1)) == 8
    assert min_steps_to_cover(graph, (0, 1)) == 9


def test_greedy_rollout_optimal(vi_3x3) -> None:
    out = greedy_rollout(vi_3x3, (0, 0), max_steps=36)
    assert out["steps"] == 8 and out["coverage"] == 1.0 and out["terminated"]
    assert out["return_discounted"] == pytest.approx(V_STAR_3X3_CORNER, abs=1e-6)


@pytest.mark.parametrize("map_name", ["empty_3x3", "obstacle_4x4"])
def test_optimal_beats_baselines(map_name: str, vi_3x3, vi_4x4) -> None:
    result = vi_3x3 if map_name == "empty_3x3" else vi_4x4
    env = VacuumEnv(map_name)
    for start in free_cells(get_map(map_name)):
        greedy = run_episode(GreedyBFSAgent(), env, seed=0, reset_options={"start": start})
        zigzag = run_episode(ZigzagAgent(), env, seed=0, reset_options={"start": start})
        min_steps = min_steps_to_cover(result.graph, start)
        assert result.value_of(start) >= greedy["return_discounted"] - 1e-9
        assert min_steps <= greedy["steps"]
        assert min_steps <= zigzag["steps"]


def test_bellman_residual(vi_4x4) -> None:
    graph = vi_4x4.graph
    backup = (graph.reward + vi_4x4.gamma * np.where(graph.terminal_next, 0.0, vi_4x4.V[graph.next_state])).max(1)
    backup[graph.is_terminal] = 0.0
    assert np.abs(vi_4x4.V - backup).max() < 1e-8


def test_vi_agent_through_env(vi_4x4) -> None:
    env = VacuumEnv("obstacle_4x4", start=(0, 0))
    result = run_episode(VIAgent(vi_4x4), env, seed=0)
    assert result["terminated"]
    assert result["return_discounted"] == pytest.approx(vi_4x4.value_of((0, 0)), abs=1e-6)


def test_obstacle_4x4_runtime() -> None:
    t0 = time.monotonic()
    result = value_iteration(build_state_graph(get_map("obstacle_4x4")))
    assert time.monotonic() - t0 < 30
    assert result.graph.n_states > 0 and result.n_iterations > 0


def test_large_grid_guard() -> None:
    with pytest.raises(RuntimeError):
        build_state_graph(get_map("empty_6x6"), max_states=10_000)


def test_count_matches_graph() -> None:
    for grid in (get_map("empty_3x3"), get_map("obstacle_4x4"), parse_map([".#..", "...."])):
        assert count_reachable_states(grid) == build_state_graph(grid).n_states
    assert count_reachable_states(get_map("empty_6x6")) is None  # quá lớn để đếm


def test_save_load(vi_3x3, tmp_path: Path) -> None:
    path = save_vi_result(vi_3x3, tmp_path / "vi.pkl")
    loaded = load_vi_result(path)
    assert np.array_equal(loaded.V, vi_3x3.V) and np.array_equal(loaded.policy, vi_3x3.policy)


def test_invalid_args() -> None:
    graph = build_state_graph(get_map("empty_3x3"))
    with pytest.raises(ValueError):
        value_iteration(graph, gamma=0.0)
    with pytest.raises(ValueError):
        value_iteration(graph, theta=0.0)
    with pytest.raises(RuntimeError):
        value_iteration(graph, max_iterations=2)
