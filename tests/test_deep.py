"""Test cho deep/: mạng, replay buffer, DQN."""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from deep.dqn import DQNAgent, DQNConfig, compute_targets, linear_epsilon
from deep.networks import QNetwork, count_parameters
from deep.replay_buffer import ReplayBuffer
from env.vacuum_env import VacuumEnv
from experiments.evaluate import run_episode


@pytest.mark.parametrize("size", [4, 8, 10])
def test_network_output_shape(size: int) -> None:
    net = QNetwork(size, size)
    out = net(torch.zeros(5, 3, size, size))
    assert out.shape == (5, 4)
    out.sum().backward()
    assert all(p.grad is not None for p in net.parameters())
    assert count_parameters(net) > 0


def test_network_rejects_wrong_size() -> None:
    with pytest.raises(ValueError):
        QNetwork(4, 4)(torch.zeros(1, 3, 5, 5))


def _obs(value: int, shape: tuple[int, ...] = (3, 2, 2)) -> np.ndarray:
    return np.full(shape, value % 2, dtype=np.float32)


def test_buffer_len_and_wraparound() -> None:
    buf = ReplayBuffer(capacity=3, obs_shape=(3, 2, 2), seed=0)
    for i in range(2):
        buf.add(_obs(i), i, float(i), _obs(i + 1), False)
    assert len(buf) == 2
    for i in range(2, 5):
        buf.add(_obs(i), i, float(i), _obs(i + 1), i == 4)
    assert len(buf) == 3
    assert sorted(buf._actions.tolist()) == [2, 3, 4]  # 0 và 1 (cũ nhất) đã bị ghi đè


def test_buffer_sample_shapes_and_determinism() -> None:
    def filled(seed: int) -> ReplayBuffer:
        buf = ReplayBuffer(capacity=50, obs_shape=(3, 2, 2), seed=seed)
        for i in range(20):
            buf.add(_obs(i), i % 4, float(i), _obs(i + 1), i % 5 == 0)
        return buf

    batch = filled(1).sample(8)
    assert batch["obs"].shape == (8, 3, 2, 2) and batch["obs"].dtype == np.float32
    assert batch["next_obs"].shape == (8, 3, 2, 2)
    assert batch["actions"].dtype == np.int64 and batch["rewards"].dtype == np.float32
    assert batch["terminated"].dtype == np.float32
    assert np.array_equal(filled(1).sample(8)["rewards"], filled(1).sample(8)["rewards"])
    with pytest.raises(ValueError):
        filled(1).sample(21)


def test_buffer_copies_data() -> None:
    buf = ReplayBuffer(capacity=4, obs_shape=(3, 2, 2))
    obs = _obs(1)
    buf.add(obs, 0, 0.0, obs, False)
    obs[:] = 0
    assert buf.sample(1)["obs"].sum() == 12


class _ConstNet(nn.Module):
    """Mạng giả trả Q cố định cho mọi input (để tính tay mục tiêu)."""

    def __init__(self, values: list[float]) -> None:
        super().__init__()
        self.values = torch.tensor(values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.values.expand(x.shape[0], -1)


def test_compute_targets_hand_values() -> None:
    batch = {"next_obs": torch.zeros(2, 3, 2, 2), "rewards": torch.tensor([1.0, 2.0]),
             "terminated": torch.tensor([0.0, 1.0])}
    y = compute_targets(batch, _ConstNet([0.0, 5.0, 3.0, -1.0]), gamma=0.9)
    assert y.tolist() == pytest.approx([1.0 + 0.9 * 5.0, 2.0])  # terminated -> y = r


def test_targets_use_target_network_only() -> None:
    agent = DQNAgent((3, 3, 3), DQNConfig(total_timesteps=10, batch_size=2, buffer_size=10))
    batch = {"next_obs": torch.rand(4, 3, 3, 3), "rewards": torch.zeros(4), "terminated": torch.zeros(4)}
    before = compute_targets(batch, agent.target, 0.99)
    with torch.no_grad():
        for p in agent.online.parameters():
            p.add_(1.0)
    assert torch.allclose(before, compute_targets(batch, agent.target, 0.99))


def test_linear_epsilon() -> None:
    cfg = DQNConfig(total_timesteps=1000, epsilon_decay_fraction=0.5)
    assert linear_epsilon(0, cfg) == pytest.approx(1.0)
    assert linear_epsilon(500, cfg) == pytest.approx(0.05)
    assert linear_epsilon(999, cfg) == pytest.approx(0.05)


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        DQNConfig(loss="l1")
    with pytest.raises(ValueError):
        DQNConfig(batch_size=100, buffer_size=10)


def test_smoke_training_writes_logs(tmp_path: Path) -> None:
    cfg = DQNConfig(total_timesteps=500, learning_starts=100, eval_every_steps=250, eval_episodes=2,
                    log_every_steps=100, batch_size=16, buffer_size=1000, checkpoint_steps=(0, 300))
    agent = DQNAgent((3, 3, 3), cfg)
    logs = agent.train(functools.partial(VacuumEnv, "empty_3x3"), tmp_path / "run")
    for name in ("train_episodes.csv", "train_losses.csv", "eval.csv", "model.pt"):
        assert (tmp_path / "run" / name).exists()
    assert (tmp_path / "run" / "checkpoints" / "step0.pt").exists()
    assert (tmp_path / "run" / "checkpoints" / "step300.pt").exists()
    assert list(logs["eval"]["global_step"]) == [250, 500]
    assert {"loss", "mean_q", "mean_target", "grad_norm"} <= set(logs["losses"].columns)


def test_save_load_same_actions(tmp_path: Path) -> None:
    agent = DQNAgent((3, 4, 4), DQNConfig(seed=3))
    loaded = DQNAgent.load(agent.save(tmp_path / "m.pt"))
    env = VacuumEnv("obstacle_4x4")
    for seed in range(10):
        obs, _ = env.reset(seed=seed)
        assert loaded.act(obs) == agent.act(obs)


def test_rejects_wrong_env_shape(tmp_path: Path) -> None:
    agent = DQNAgent((3, 4, 4), DQNConfig(total_timesteps=10, batch_size=2, buffer_size=10))
    with pytest.raises(ValueError):
        agent.train(functools.partial(VacuumEnv, "empty_3x3"), tmp_path / "x")


@pytest.mark.slow
def test_dqn_learns_3x3(tmp_path: Path) -> None:
    factory = functools.partial(VacuumEnv, "empty_3x3", start=(0, 0))
    agent = DQNAgent((3, 3, 3), DQNConfig(total_timesteps=30_000, eval_every_steps=10_000, eval_episodes=1))
    agent.train(factory, tmp_path / "dqn3")
    assert run_episode(agent, factory(), seed=0)["coverage"] == 1.0
