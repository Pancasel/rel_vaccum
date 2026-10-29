"""Test cho Q-learning dạng bảng (so khớp với Value Iteration)."""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pytest

from environment.room_layouts import get_room_layout, parse_room_layout
from environment.vacuum_cleaning_env import VacuumCleaningEnv
from evaluation.episode_runner import run_single_episode
from agents.tabular.tabular_agent_base import TabularAgentBase
from agents.tabular.tabular_settings import TabularSettings, linear_epsilon_for_episode
from agents.tabular.q_learning_agent import QLearningAgent
from agents.tabular.state_transition_graph import build_state_transition_graph, minimum_steps_to_clean_all
from agents.tabular.value_iteration import run_value_iteration


class _ScriptedEnv:
    """Env giả chạy đúng một bước với kết quả định sẵn, để kiểm tra công thức cập nhật."""

    def __init__(self, grid_lines: list[str], start: tuple[int, int], action_result: tuple) -> None:
        self.inner = VacuumCleaningEnv(parse_room_layout(grid_lines), start=start)
        self.unwrapped = self.inner
        self._result = action_result

    def reset(self, seed=None, options=None):
        return self.inner.reset(seed=seed, options=options)

    def step(self, action):
        obs, _, _, _, info = self.inner.step(action)
        reward, terminated, truncated = self._result
        return obs, reward, terminated, truncated, info


def _agent(grid_lines: list[str], **kwargs) -> QLearningAgent:
    cfg = TabularSettings(episode_count=1, epsilon_start=0.0, epsilon_end=0.0, **kwargs)
    return QLearningAgent(parse_room_layout(grid_lines), cfg)


def test_q_update_single_step() -> None:
    lines = ["...."]
    agent = _agent(lines, alpha=0.5, gamma=0.9)
    # truncated=True để episode dừng sau đúng một bước (vẫn bootstrap như bước thường).
    env = _ScriptedEnv(lines, (0, 0), (1.0, False, True))
    s = agent.state_encoder.encode_state((0, 0), np.array([[True, False, False, False]]))
    s_next = agent.state_encoder.encode_state((0, 1), np.array([[True, True, False, False]]))
    agent.q_table[s] = np.array([0.0, 0.0, 0.0, 2.0])  # tham lam chọn "phải"
    agent.q_table[s_next] = np.array([0.0, 4.0, 0.0, 1.0])
    agent._run_one_episode(env, epsilon=0.0, seed=0)
    # target = R + γ max Q(S') = 1 + 0.9*4 = 4.6; Q <- 2 + 0.5*(4.6 - 2) = 3.3
    assert agent.q_table[s][3] == pytest.approx(3.3)


def test_terminal_no_bootstrap() -> None:
    lines = ["...."]
    agent = _agent(lines, alpha=1.0, gamma=0.9)
    env = _ScriptedEnv(lines, (0, 0), (5.0, True, False))
    s = agent.state_encoder.encode_state((0, 0), np.array([[True, False, False, False]]))
    s_next = agent.state_encoder.encode_state((0, 1), np.array([[True, True, False, False]]))
    agent.q_table[s] = np.array([0.0, 0.0, 0.0, 1.0])
    agent.q_table[s_next] = np.array([100.0, 100.0, 100.0, 100.0])
    agent._run_one_episode(env, epsilon=0.0, seed=0)
    assert agent.q_table[s][3] == pytest.approx(5.0)  # target = R, không cộng γ·100


def test_truncated_bootstraps() -> None:
    lines = ["...."]
    agent = _agent(lines, alpha=1.0, gamma=0.9)
    env = _ScriptedEnv(lines, (0, 0), (1.0, False, True))
    s = agent.state_encoder.encode_state((0, 0), np.array([[True, False, False, False]]))
    s_next = agent.state_encoder.encode_state((0, 1), np.array([[True, True, False, False]]))
    agent.q_table[s] = np.array([0.0, 0.0, 0.0, 1.0])
    agent.q_table[s_next] = np.array([10.0, 0.0, 0.0, 0.0])
    agent._run_one_episode(env, epsilon=0.0, seed=0)
    assert agent.q_table[s][3] == pytest.approx(1.0 + 0.9 * 10.0)


def test_linear_epsilon_schedule() -> None:
    cfg = TabularSettings(episode_count=100, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_fraction=0.5)
    assert linear_epsilon_for_episode(0, cfg) == pytest.approx(1.0)
    assert linear_epsilon_for_episode(25, cfg) == pytest.approx(0.55)
    assert linear_epsilon_for_episode(50, cfg) == pytest.approx(0.1)
    assert linear_epsilon_for_episode(99, cfg) == pytest.approx(0.1)


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        TabularSettings(alpha=0.0)
    with pytest.raises(ValueError):
        TabularSettings(state_mode="bad")
    with pytest.raises(ValueError):
        TabularSettings(episode_count=0)


def test_base_class_is_abstract() -> None:
    with pytest.raises(TypeError):
        TabularAgentBase(get_room_layout("empty_3x3"), TabularSettings())  # type: ignore[abstract]


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_q_learning_matches_vi_3x3(seed: int) -> None:
    factory = functools.partial(VacuumCleaningEnv, "empty_3x3", start=(0, 0))
    agent = QLearningAgent(get_room_layout("empty_3x3"), TabularSettings(episode_count=3000, seed=seed))
    agent.train(factory)
    result = run_single_episode(agent, factory(), seed=0)
    vi = run_value_iteration(build_state_transition_graph(get_room_layout("empty_3x3")))
    assert result["coverage"] == 1.0
    assert result["steps"] == minimum_steps_to_clean_all(vi.graph, (0, 0)) == 8
    start_cleaned = np.zeros((3, 3), dtype=bool)
    start_cleaned[0, 0] = True
    s0 = agent.state_encoder.encode_state((0, 0), start_cleaned)
    assert abs(agent.best_value_of_state(s0) - vi.optimal_value_of_start((0, 0))) < 0.5


def test_position_only_key() -> None:
    factory = functools.partial(VacuumCleaningEnv, "empty_3x3", start=(0, 0))
    agent = QLearningAgent(get_room_layout("empty_3x3"), TabularSettings(episode_count=50, state_mode="position_only"))
    agent.train(factory)
    assert agent.q_table and all(len(k) == 2 for k in agent.q_table)


def test_reproducible() -> None:
    factory = functools.partial(VacuumCleaningEnv, "obstacle_4x4")
    logs = []
    for _ in range(2):
        agent = QLearningAgent(get_room_layout("obstacle_4x4"), TabularSettings(episode_count=200, evaluate_every_episodes=50, seed=7))
        train, _ = agent.train(factory)
        logs.append(train.drop(columns=["wall_time_s"]))
    assert logs[0].equals(logs[1])


def test_save_load(tmp_path: Path) -> None:
    factory = functools.partial(VacuumCleaningEnv, "obstacle_4x4")
    agent = QLearningAgent(get_room_layout("obstacle_4x4"), TabularSettings(episode_count=300, seed=1))
    agent.train(factory)
    loaded = TabularAgentBase.load(agent.save(tmp_path / "q.pkl"))
    assert isinstance(loaded, QLearningAgent)
    for key in list(agent.q_table)[:50]:
        assert loaded.choose_greedy_action(key) == agent.choose_greedy_action(key)


def test_checkpoints_and_logs(tmp_path: Path) -> None:
    factory = functools.partial(VacuumCleaningEnv, "empty_3x3", start=(0, 0))
    cfg = TabularSettings(episode_count=20, evaluate_every_episodes=10, checkpoint_episodes=(0, 10))
    agent = QLearningAgent(get_room_layout("empty_3x3"), cfg)
    train, evals = agent.train(factory, log_dir=tmp_path / "run")
    for name in ("ep0.pkl", "ep10.pkl", "final.pkl"):
        assert (tmp_path / "run" / "checkpoints" / name).exists()
    assert TabularAgentBase.load(tmp_path / "run" / "checkpoints" / "ep0.pkl").q_table == {}
    assert (tmp_path / "run" / "seed0_train.csv").exists() and (tmp_path / "run" / "seed0_eval.csv").exists()
    assert len(train) == 20 and list(evals["episode"]) == [10, 20]
    assert train["q_start_max"].notna().all()


def test_random_start_logs_nan_start_value() -> None:
    factory = functools.partial(VacuumCleaningEnv, "empty_3x3")
    agent = QLearningAgent(get_room_layout("empty_3x3"), TabularSettings(episode_count=5))
    train, _ = agent.train(factory)
    assert train["q_start_max"].isna().all()


def test_time_budget_stops_early() -> None:
    factory = functools.partial(VacuumCleaningEnv, "empty_3x3")
    agent = QLearningAgent(get_room_layout("empty_3x3"), TabularSettings(episode_count=10_000_000, time_budget_minutes=0.01))
    train, evals = agent.train(factory)
    assert len(train) < 10_000_000 and len(evals) >= 1
