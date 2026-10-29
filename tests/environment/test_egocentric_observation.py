"""Test cho environment/egocentric_observation.py."""

from __future__ import annotations

import numpy as np

from environment.observation_decoding import decode_observation, safe_actions_from_observation
from environment.vacuum_cleaning_env import VacuumCleaningEnv


def test_egocentric_view_centers_robot_and_keeps_map() -> None:
    from environment.egocentric_observation import egocentric_observation_shape, to_egocentric_observation

    env = VacuumCleaningEnv("obstacle_4x4")
    rng = np.random.default_rng(0)
    for seed in range(20):
        obs, _ = env.reset(seed=seed)
        for _ in range(5):
            view = to_egocentric_observation(obs)
            assert view.shape == egocentric_observation_shape(4, 4) == (3, 7, 7)
            assert view[0, 3, 3] == 1 and view[0].sum() == 1          # robot luôn ở tâm
            assert view[1].sum() == obs[1].sum()                        # không mất ô đã dọn nào
            assert view[2].sum() == obs[2].sum() + (49 - 16)            # ngoài bản đồ = tường
            assert decode_observation(view)[0] == (3, 3)
            assert safe_actions_from_observation(view) == safe_actions_from_observation(obs)            # hành động hợp lệ giữ nguyên
            obs, *_ = env.step(int(rng.integers(4)))
