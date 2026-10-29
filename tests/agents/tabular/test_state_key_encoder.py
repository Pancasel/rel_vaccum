"""Test cho agents/tabular/state_key_encoder.py."""

from __future__ import annotations

import numpy as np
import pytest

from agents.tabular.state_key_encoder import StateKeyEncoder
from environment.room_layouts import get_room_layout
from environment.vacuum_cleaning_env import VacuumCleaningEnv


def test_encoder_roundtrip_and_obs_agree() -> None:
    env = VacuumCleaningEnv("obstacle_4x4")
    encoder = StateKeyEncoder(env.grid)
    rng = np.random.default_rng(0)
    for episode in range(20):
        obs, _ = env.reset(seed=episode)
        for _ in range(5):
            key = encoder.encode_state(env.position, env.cleaned)
            assert key == encoder.encode_observation(obs)
            assert np.array_equal(encoder.bitmask_to_cleaned_cells(key[2]), env.cleaned)
            assert (key[2] >> encoder.free_cell_index[env.position]) & 1
            obs, *_ = env.step(int(rng.integers(4)))


def test_encoder_position_only() -> None:
    encoder = StateKeyEncoder(get_room_layout("empty_3x3"), state_mode="position_only")
    assert encoder.encode_state((1, 2), np.zeros((3, 3), dtype=bool)) == (1, 2)


def test_encoder_validation() -> None:
    with pytest.raises(ValueError):
        StateKeyEncoder(get_room_layout("empty_3x3"), state_mode="bad")
    with pytest.raises(ValueError):
        StateKeyEncoder(np.zeros((8, 8), dtype=np.int8))  # 64 ô > 62
    encoder = StateKeyEncoder(get_room_layout("empty_3x3"))
    assert encoder.all_cleaned_bitmask == 2**9 - 1
    with pytest.raises(ValueError):
        encoder.bitmask_to_cleaned_cells(2**9)
