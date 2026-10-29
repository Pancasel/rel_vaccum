"""Test cho environment/shortest_path_search.py và observation_decoding.py."""

from __future__ import annotations

import numpy as np
import pytest

from environment.shortest_path_search import distance_to_nearest_dirty_cell, distances_from_cell, first_action_towards_nearest_target
from environment.observation_decoding import decode_observation, safe_actions_from_observation
from environment.room_layouts import get_room_layout, parse_room_layout
from environment.vacuum_cleaning_env import VacuumCleaningEnv


def test_bfs_distances_around_obstacle() -> None:
    grid = parse_room_layout(["...", "##.", "..."])
    dist = distances_from_cell(grid, (0, 0))
    assert dist.tolist() == [[0, 1, 2], [-1, -1, 3], [6, 5, 4]]


def test_bfs_distances_rejects_obstacle_source() -> None:
    with pytest.raises(ValueError):
        distances_from_cell(parse_room_layout([".#"]), (0, 1))


def test_distance_to_nearest_uncleaned() -> None:
    grid = parse_room_layout(["...", "##.", "..."])
    cleaned = np.zeros(grid.shape, dtype=bool)
    cleaned[0, :] = True
    cleaned[1, 2] = True
    assert distance_to_nearest_dirty_cell(grid, (0, 0), cleaned) == 4
    all_clean = grid == 0
    assert distance_to_nearest_dirty_cell(grid, (0, 0), all_clean) == 0


def test_first_action_deterministic_tie_break() -> None:
    grid = get_room_layout("empty_3x3")
    targets = np.zeros((3, 3), dtype=bool)
    targets[0, 1] = targets[1, 0] = True  # cùng khoảng cách 1 từ (0,0): xuống (1) trước phải (3)
    assert first_action_towards_nearest_target(grid, (0, 0), targets) == 1
    targets[:] = False
    targets[2, 2] = True
    assert first_action_towards_nearest_target(grid, (0, 0), targets) == 1
    targets[:] = False
    assert first_action_towards_nearest_target(grid, (0, 0), targets) is None


def test_first_action_ignores_source() -> None:
    grid = get_room_layout("empty_3x3")
    targets = np.zeros((3, 3), dtype=bool)
    targets[0, 0] = True
    assert first_action_towards_nearest_target(grid, (0, 0), targets) is None


def test_decode_obs_and_valid_actions() -> None:
    env = VacuumCleaningEnv("obstacle_4x4", start=(0, 1))
    obs, _ = env.reset(seed=0)
    pos, cleaned, obstacles = decode_observation(obs)
    assert pos == (0, 1)
    assert cleaned.sum() == 1 and cleaned[0, 1]
    assert obstacles[1, 1] == 1
    assert safe_actions_from_observation(obs) == [2, 3]  # lên là tường, xuống là vật cản


def test_decode_obs_rejects_bad_channel() -> None:
    obs = np.zeros((3, 3, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        decode_observation(obs)
    with pytest.raises(ValueError):
        decode_observation(np.zeros((2, 3, 3), dtype=np.float32))
