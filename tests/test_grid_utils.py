"""Test cho env/grid_utils.py."""

from __future__ import annotations

import numpy as np
import pytest

from env.grid_utils import (StateEncoder, bfs_distance_to_nearest_uncleaned, bfs_distances,
                            bfs_first_action_towards, decode_obs, valid_actions)
from env.maps import get_map, parse_map
from env.vacuum_env import VacuumEnv


def test_bfs_distances_around_obstacle() -> None:
    grid = parse_map(["...", "##.", "..."])
    dist = bfs_distances(grid, (0, 0))
    assert dist.tolist() == [[0, 1, 2], [-1, -1, 3], [6, 5, 4]]


def test_bfs_distances_rejects_obstacle_source() -> None:
    with pytest.raises(ValueError):
        bfs_distances(parse_map([".#"]), (0, 1))


def test_distance_to_nearest_uncleaned() -> None:
    grid = parse_map(["...", "##.", "..."])
    cleaned = np.zeros(grid.shape, dtype=bool)
    cleaned[0, :] = True
    cleaned[1, 2] = True
    assert bfs_distance_to_nearest_uncleaned(grid, (0, 0), cleaned) == 4
    all_clean = grid == 0
    assert bfs_distance_to_nearest_uncleaned(grid, (0, 0), all_clean) == 0


def test_first_action_deterministic_tie_break() -> None:
    grid = get_map("empty_3x3")
    targets = np.zeros((3, 3), dtype=bool)
    targets[0, 1] = targets[1, 0] = True  # cùng khoảng cách 1 từ (0,0): xuống (1) trước phải (3)
    assert bfs_first_action_towards(grid, (0, 0), targets) == 1
    targets[:] = False
    targets[2, 2] = True
    assert bfs_first_action_towards(grid, (0, 0), targets) == 1
    targets[:] = False
    assert bfs_first_action_towards(grid, (0, 0), targets) is None


def test_first_action_ignores_source() -> None:
    grid = get_map("empty_3x3")
    targets = np.zeros((3, 3), dtype=bool)
    targets[0, 0] = True
    assert bfs_first_action_towards(grid, (0, 0), targets) is None


def test_decode_obs_and_valid_actions() -> None:
    env = VacuumEnv("obstacle_4x4", start=(0, 1))
    obs, _ = env.reset(seed=0)
    pos, cleaned, obstacles = decode_obs(obs)
    assert pos == (0, 1)
    assert cleaned.sum() == 1 and cleaned[0, 1]
    assert obstacles[1, 1] == 1
    assert valid_actions(obs) == [2, 3]  # lên là tường, xuống là vật cản


def test_decode_obs_rejects_bad_channel() -> None:
    obs = np.zeros((3, 3, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        decode_obs(obs)
    with pytest.raises(ValueError):
        decode_obs(np.zeros((2, 3, 3), dtype=np.float32))


def test_encoder_roundtrip_and_obs_agree() -> None:
    env = VacuumEnv("obstacle_4x4")
    encoder = StateEncoder(env.grid)
    rng = np.random.default_rng(0)
    for episode in range(20):
        obs, _ = env.reset(seed=episode)
        for _ in range(5):
            key = encoder.encode(env.position, env.cleaned)
            assert key == encoder.encode_obs(obs)
            assert np.array_equal(encoder.decode_mask(key[2]), env.cleaned)
            assert (key[2] >> encoder.free_index[env.position]) & 1
            obs, *_ = env.step(int(rng.integers(4)))


def test_encoder_position_only() -> None:
    encoder = StateEncoder(get_map("empty_3x3"), state_mode="position_only")
    assert encoder.encode((1, 2), np.zeros((3, 3), dtype=bool)) == (1, 2)


def test_encoder_validation() -> None:
    with pytest.raises(ValueError):
        StateEncoder(get_map("empty_3x3"), state_mode="bad")
    with pytest.raises(ValueError):
        StateEncoder(np.zeros((8, 8), dtype=np.int8))  # 64 ô > 62
    encoder = StateEncoder(get_map("empty_3x3"))
    assert encoder.full_mask == 2**9 - 1
    with pytest.raises(ValueError):
        encoder.decode_mask(2**9)
