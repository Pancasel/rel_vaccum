"""Test cho env/vacuum_env.py và env/wrappers.py (bảng 4.2.5 của kế hoạch)."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from env.grid_utils import bfs_distance_to_nearest_uncleaned
from env.maps import parse_map
from env.vacuum_env import RandomMapVacuumEnv, RewardConfig, VacuumEnv, transition
from env.wrappers import HideCleanedChannel

SNAKE_3X3 = [3, 3, 1, 2, 2, 1, 3, 3]


def test_reset_shapes(env_3x3_corner: VacuumEnv) -> None:
    obs, info = env_3x3_corner.reset(seed=0)
    assert obs.shape == (3, 3, 3)
    assert obs.dtype == np.float32
    assert obs[0].sum() == 1 and obs[0][0, 0] == 1
    assert obs[1][0, 0] == 1 and obs[1].sum() == 1
    assert info["coverage"] == pytest.approx(1 / 9)
    for key in ("coverage", "collided", "new_cell", "event", "steps", "n_cleaned", "n_free", "position",
                "n_revisit", "n_collisions", "action_mask"):
        assert key in info


def test_collision_wall(env_3x3_corner: VacuumEnv) -> None:
    env_3x3_corner.reset(seed=0)
    obs, reward, term, trunc, info = env_3x3_corner.step(0)
    assert info["position"] == (0, 0)
    assert reward == pytest.approx(-0.51)
    assert info["collided"] and info["event"] == "collision"
    assert info["steps"] == 1 and info["n_collisions"] == 1
    assert not term and not trunc


def test_collision_obstacle() -> None:
    env = VacuumEnv("obstacle_4x4", start=(0, 1))
    env.reset(seed=0)
    _, reward, _, _, info = env.step(1)
    assert info["position"] == (0, 1)
    assert reward == pytest.approx(-0.51)


def test_new_cell(env_3x3_corner: VacuumEnv) -> None:
    env_3x3_corner.reset(seed=0)
    _, reward, _, _, info = env_3x3_corner.step(3)
    assert info["position"] == (0, 1)
    assert reward == pytest.approx(0.99)
    assert info["new_cell"]


def test_revisit(env_3x3_corner: VacuumEnv) -> None:
    env_3x3_corner.reset(seed=0)
    env_3x3_corner.step(3)
    _, reward, _, _, info = env_3x3_corner.step(2)
    assert reward == pytest.approx(-0.11)
    assert info["event"] == "revisit" and info["n_revisit"] == 1


def test_terminated_on_full(env_3x3_corner: VacuumEnv) -> None:
    env_3x3_corner.reset(seed=0)
    total = 0.0
    for i, action in enumerate(SNAKE_3X3):
        _, reward, term, trunc, info = env_3x3_corner.step(action)
        total += reward
        assert not trunc
        assert term == (i == len(SNAKE_3X3) - 1)
    assert reward == pytest.approx(10.99)
    assert total == pytest.approx(17.92, abs=1e-6)
    assert info["coverage"] == 1.0


def test_truncated() -> None:
    env = VacuumEnv("empty_3x3", start=(0, 0), max_steps=3)
    env.reset(seed=0)
    results = [env.step(0) for _ in range(3)]
    assert [r[3] for r in results] == [False, False, True]
    assert not any(r[2] for r in results)


def test_default_max_steps() -> None:
    assert VacuumEnv("obstacle_4x4").max_steps == 4 * 13


def test_never_both_terminated_and_truncated() -> None:
    env = VacuumEnv("empty_3x3", max_steps=12)
    rng = np.random.default_rng(0)
    for episode in range(200):
        env.reset(seed=episode)
        done = False
        while not done:
            _, _, term, trunc, _ = env.step(int(rng.integers(4)))
            assert not (term and trunc)
            done = term or trunc


def test_step_before_reset() -> None:
    with pytest.raises(RuntimeError):
        VacuumEnv("empty_3x3").step(0)


def test_invalid_action(env_3x3_corner: VacuumEnv) -> None:
    env_3x3_corner.reset(seed=0)
    for bad in (4, -1, 1.5, True):
        with pytest.raises(ValueError):
            env_3x3_corner.step(bad)
    env_3x3_corner.step(np.int64(3))  # kiểu numpy hợp lệ


def test_seed_reproducible() -> None:
    env = VacuumEnv("obstacle_4x4")
    obs_a, info_a = env.reset(seed=42)
    obs_b, info_b = env.reset(seed=42)
    assert info_a["position"] == info_b["position"]
    assert np.array_equal(obs_a, obs_b)
    starts = {env.reset(seed=s)[1]["position"] for s in range(50)}
    assert len(starts) > 1


def test_reset_options_start() -> None:
    env = VacuumEnv("empty_3x3")
    _, info = env.reset(seed=0, options={"start": (2, 2)})
    assert info["position"] == (2, 2)


def test_obs_not_aliased(env_3x3_corner: VacuumEnv) -> None:
    obs, _ = env_3x3_corner.reset(seed=0)
    saved = obs.copy()
    env_3x3_corner.step(3)
    assert np.array_equal(obs, saved)
    cleaned = env_3x3_corner.cleaned
    cleaned[:] = True
    assert not env_3x3_corner.cleaned.all()


def test_transition_pure() -> None:
    grid = parse_map(["...", "...", "..."])
    cleaned = np.zeros((3, 3), dtype=bool)
    cleaned[0, 0] = True
    before = cleaned.copy()
    a = transition(grid, (0, 0), cleaned, 3, RewardConfig())
    b = transition(grid, (0, 0), cleaned, 3, RewardConfig())
    assert np.array_equal(cleaned, before)
    assert a.next_pos == b.next_pos and a.reward == b.reward
    assert np.array_equal(a.next_cleaned, b.next_cleaned)
    assert a.next_cleaned is not cleaned


def test_transition_invalid_action() -> None:
    grid = parse_map(["..", ".."])
    with pytest.raises(ValueError):
        transition(grid, (0, 0), np.zeros((2, 2), dtype=bool), 7, RewardConfig())


def test_invalid_start() -> None:
    with pytest.raises(ValueError):
        VacuumEnv("obstacle_4x4", start=(1, 1))
    with pytest.raises(ValueError):
        VacuumEnv("obstacle_4x4", start=(9, 9))


def test_disconnected_map() -> None:
    with pytest.raises(ValueError):
        VacuumEnv(parse_map([".#.", "#..", "..."]))


def test_too_few_free_cells() -> None:
    with pytest.raises(ValueError):
        VacuumEnv(parse_map([".#", "##"]))


def test_invalid_render_mode() -> None:
    with pytest.raises(ValueError):
        VacuumEnv("empty_3x3", render_mode="human")


def test_sparse_reward() -> None:
    env = VacuumEnv("empty_3x3", start=(0, 0), reward_config=RewardConfig.sparse())
    env.reset(seed=0)
    rewards = [env.step(a)[1] for a in SNAKE_3X3]
    assert rewards[:-1] == [0.0] * 7
    assert rewards[-1] == pytest.approx(10.0)


def test_potential_shaping_telescopes() -> None:
    """Với γ=1, tổng các số hạng shaping trên cả episode = Φ(terminal) − Φ(s0) = −Φ(s0)."""
    grid = parse_map(["....", ".#..", "...#", "#..."])
    shaped = RewardConfig(shaping_coef=1.0, gamma=1.0)
    plain = RewardConfig(shaping_coef=0.0)
    rng = np.random.default_rng(3)
    pos = (0, 0)
    cleaned = np.zeros(grid.shape, dtype=bool)
    cleaned[pos] = True
    phi_s0 = -bfs_distance_to_nearest_uncleaned(grid, pos, cleaned) / (4 + 4)
    shaping_sum = 0.0
    for _ in range(500):
        action = int(rng.integers(4))
        r_shaped = transition(grid, pos, cleaned, action, shaped)
        r_plain = transition(grid, pos, cleaned, action, plain)
        shaping_sum += r_shaped.reward - r_plain.reward
        pos, cleaned = r_shaped.next_pos, r_shaped.next_cleaned
        if r_shaped.terminated:
            break
    assert r_shaped.terminated
    assert shaping_sum == pytest.approx(-phi_s0, abs=1e-9)


def test_hackable_loop() -> None:
    """Đi xa ô bẩn không bị phạt, quay lại thì được thưởng -> vòng "ra-vào" cho điểm dương mãi.

    Hàng 5 ô, đã dọn 3 ô giữa, ô bẩn ở hai đầu. Từ (0,1) sang (0,2): khoảng cách tới ô bẩn
    gần nhất tăng 1 -> 2 (không phạt). Từ (0,2) về (0,1): giảm 2 -> 1 (thưởng 0.5).
    """
    cfg = RewardConfig.hackable(0.5)
    grid = parse_map(["....."])
    cleaned = np.array([[False, True, True, True, False]])
    away = transition(grid, (0, 1), cleaned, 3, cfg)
    back = transition(grid, (0, 2), cleaned, 2, cfg)
    assert away.event == back.event == "revisit"
    assert away.reward == pytest.approx(0.0)
    assert back.reward == pytest.approx(0.5)
    assert away.reward + back.reward > 0  # một vòng lặp 2 bước luôn có lãi


def test_action_mask() -> None:
    env = VacuumEnv("empty_3x3", start=(0, 0))
    _, info = env.reset(seed=0)
    assert info["action_mask"].tolist() == [0, 1, 0, 1]


def test_reward_from_name() -> None:
    assert RewardConfig.from_name("dense") == RewardConfig()
    assert RewardConfig.from_name("shaped").shaping_coef == 1.0
    assert RewardConfig.from_name("dense", step=-0.02).step == -0.02
    with pytest.raises(ValueError):
        RewardConfig.from_name("unknown")


@pytest.mark.parametrize("render_mode", [None, "rgb_array", "ansi"])
def test_check_env_vacuum(render_mode: str | None) -> None:
    check_env(VacuumEnv("empty_3x3", render_mode=render_mode), skip_render_check=render_mode is None)


@pytest.mark.parametrize("render_mode", [None, "rgb_array", "ansi"])
def test_check_env_random_map(render_mode: str | None) -> None:
    check_env(RandomMapVacuumEnv(5, 5, 0.2, render_mode=render_mode), skip_render_check=render_mode is None)


@pytest.mark.parametrize("render_mode", [None, "rgb_array", "ansi"])
def test_check_env_hide_cleaned(render_mode: str | None) -> None:
    check_env(HideCleanedChannel(VacuumEnv("obstacle_4x4", render_mode=render_mode)),
              skip_render_check=render_mode is None)


def test_ansi_render() -> None:
    env = VacuumEnv("obstacle_4x4", start=(0, 0), render_mode="ansi")
    env.reset(seed=0)
    env.step(3)
    assert env.render() == "oR..\n.#..\n...#\n#..."


def test_random_map_env() -> None:
    env = RandomMapVacuumEnv(6, 6, 0.2)
    obs_a, _ = env.reset(seed=1)
    grid_a = env.grid
    obs_b, _ = env.reset(seed=2)
    assert obs_a.shape == obs_b.shape == (3, 6, 6)
    assert not np.array_equal(grid_a, env.grid)
    assert env.max_steps == 4 * env.n_free


def test_random_map_env_pool_and_forced_grid() -> None:
    pool = [parse_map(["....", ".#..", "....", "...."]), parse_map(["....", "....", "..#.", "...."])]
    env = RandomMapVacuumEnv(4, 4, 0.1, map_pool=pool)
    seen = set()
    for seed in range(20):
        env.reset(seed=seed)
        seen.add(env.grid.tobytes())
    assert seen == {m.tobytes() for m in pool}
    forced = parse_map(["....", "....", "....", "#..."])
    env.reset(seed=0, options={"grid": forced})
    assert np.array_equal(env.grid, forced)
    with pytest.raises(ValueError):
        env.reset(seed=0, options={"grid": parse_map(["..."])})


def test_random_map_env_exclude() -> None:
    exclude = [parse_map(["...", "...", ".#."])]
    env = RandomMapVacuumEnv(3, 3, 1 / 9, exclude_maps=exclude)
    for seed in range(30):
        env.reset(seed=seed)
        assert env.grid.tobytes() != exclude[0].tobytes()


def test_hide_cleaned_channel() -> None:
    env = HideCleanedChannel(VacuumEnv("empty_3x3", start=(0, 0)))
    obs, _ = env.reset(seed=0)
    assert obs[1].sum() == 0 and obs[0].sum() == 1
    obs, *_ = env.step(3)
    assert obs[1].sum() == 0 and obs[0][0, 1] == 1
