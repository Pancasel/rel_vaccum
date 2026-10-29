"""Test cho tabular/state_space.py và tabular/value_iteration.py (đáp án tối ưu)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from agents.baselines import GreedyNearestDirtyCellAgent, ZigzagSweepAgent
from environment.room_layouts import list_free_cells, get_room_layout, parse_room_layout
from environment.vacuum_cleaning_env import VacuumCleaningEnv
from evaluation.episode_runner import run_single_episode
from agents.tabular.state_transition_graph import build_state_transition_graph, minimum_steps_to_clean_all
from agents.tabular.reachable_state_counter import count_reachable_states
from agents.tabular.value_iteration import ValueIterationPolicyAgent, follow_optimal_policy, load_value_iteration_result, save_value_iteration_result, run_value_iteration

V_STAR_3X3_CORNER = 16.968928730705  # phụ lục 10.1


@pytest.fixture(scope="module")
def vi_3x3():
    return run_value_iteration(build_state_transition_graph(get_room_layout("empty_3x3")))


@pytest.fixture(scope="module")
def vi_4x4():
    return run_value_iteration(build_state_transition_graph(get_room_layout("obstacle_4x4")))


def test_state_keys_contain_robot_bit(vi_3x3) -> None:
    graph = vi_3x3.graph
    assert graph.state_count > 0
    for r, c, mask in graph.state_keys:
        assert (mask >> graph.state_encoder.free_cell_index[(r, c)]) & 1


def test_v_star_corner(vi_3x3) -> None:
    assert vi_3x3.optimal_value_of_start((0, 0)) == pytest.approx(V_STAR_3X3_CORNER, abs=1e-6)


def test_min_steps_checkerboard(vi_3x3) -> None:
    graph = vi_3x3.graph
    assert minimum_steps_to_clean_all(graph, (0, 0)) == 8
    assert minimum_steps_to_clean_all(graph, (1, 1)) == 8
    assert minimum_steps_to_clean_all(graph, (0, 1)) == 9


def test_greedy_rollout_optimal(vi_3x3) -> None:
    out = follow_optimal_policy(vi_3x3, (0, 0), max_steps=36)
    assert out["steps"] == 8 and out["coverage"] == 1.0 and out["terminated"]
    assert out["return_discounted"] == pytest.approx(V_STAR_3X3_CORNER, abs=1e-6)


@pytest.mark.parametrize("map_name", ["empty_3x3", "obstacle_4x4"])
def test_optimal_beats_baselines(map_name: str, vi_3x3, vi_4x4) -> None:
    result = vi_3x3 if map_name == "empty_3x3" else vi_4x4
    env = VacuumCleaningEnv(map_name)
    for start in list_free_cells(get_room_layout(map_name)):
        greedy = run_single_episode(GreedyNearestDirtyCellAgent(), env, seed=0, reset_options={"start": start})
        zigzag = run_single_episode(ZigzagSweepAgent(), env, seed=0, reset_options={"start": start})
        min_steps = minimum_steps_to_clean_all(result.graph, start)
        assert result.optimal_value_of_start(start) >= greedy["return_discounted"] - 1e-9
        assert min_steps <= greedy["steps"]
        assert min_steps <= zigzag["steps"]


def test_bellman_residual(vi_4x4) -> None:
    graph = vi_4x4.graph
    backup = (graph.rewards + vi_4x4.gamma * np.where(graph.next_state_is_terminal, 0.0, vi_4x4.state_values[graph.next_state_indices])).max(1)
    backup[graph.is_terminal_state] = 0.0
    assert np.abs(vi_4x4.state_values - backup).max() < 1e-8


def test_vi_agent_through_env(vi_4x4) -> None:
    env = VacuumCleaningEnv("obstacle_4x4", start=(0, 0))
    result = run_single_episode(ValueIterationPolicyAgent(vi_4x4), env, seed=0)
    assert result["terminated"]
    assert result["return_discounted"] == pytest.approx(vi_4x4.optimal_value_of_start((0, 0)), abs=1e-6)


def test_obstacle_4x4_runtime() -> None:
    t0 = time.monotonic()
    result = run_value_iteration(build_state_transition_graph(get_room_layout("obstacle_4x4")))
    assert time.monotonic() - t0 < 30
    assert result.graph.state_count > 0 and result.iteration_count > 0


def test_large_grid_guard() -> None:
    with pytest.raises(RuntimeError):
        build_state_transition_graph(get_room_layout("empty_6x6"), max_states=10_000)


def test_count_matches_graph() -> None:
    for grid in (get_room_layout("empty_3x3"), get_room_layout("obstacle_4x4"), parse_room_layout([".#..", "...."])):
        assert count_reachable_states(grid) == build_state_transition_graph(grid).state_count
    assert count_reachable_states(get_room_layout("empty_6x6")) is None  # quá lớn để đếm


def test_save_load(vi_3x3, tmp_path: Path) -> None:
    path = save_value_iteration_result(vi_3x3, tmp_path / "vi.pkl")
    loaded = load_value_iteration_result(path)
    assert np.array_equal(loaded.state_values, vi_3x3.state_values) and np.array_equal(loaded.policy, vi_3x3.policy)


def test_invalid_args() -> None:
    graph = build_state_transition_graph(get_room_layout("empty_3x3"))
    with pytest.raises(ValueError):
        run_value_iteration(graph, gamma=0.0)
    with pytest.raises(ValueError):
        run_value_iteration(graph, theta=0.0)
    with pytest.raises(RuntimeError):
        run_value_iteration(graph, max_iterations=2)
