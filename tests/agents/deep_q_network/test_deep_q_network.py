"""Test cho deep/: mạng, replay buffer, DQN."""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from agents.deep_q_network.deep_q_network_agent import DeepQNetworkAgent
from agents.deep_q_network.deep_q_network_settings import DeepQNetworkSettings, linear_epsilon_for_step
from agents.deep_q_network.bellman_targets import compute_bellman_targets
from agents.deep_q_network.compute_device import choose_compute_device
from agents.deep_q_network.wall_action_masking import safe_action_mask
from agents.deep_q_network.q_value_network import QValueNetwork, count_trainable_parameters
from agents.deep_q_network.replay_buffer import ReplayBuffer
from environment.vacuum_cleaning_env import VacuumCleaningEnv
from evaluation.episode_runner import run_single_episode


@pytest.mark.parametrize("size", [4, 8, 10])
def test_network_output_shape(size: int) -> None:
    net = QValueNetwork(size, size)
    out = net(torch.zeros(5, 3, size, size))
    assert out.shape == (5, 4)
    out.sum().backward()
    assert all(p.grad is not None for p in net.parameters())
    assert count_trainable_parameters(net) > 0


def test_network_rejects_wrong_size() -> None:
    with pytest.raises(ValueError):
        QValueNetwork(4, 4)(torch.zeros(1, 3, 5, 5))


def _obs(value: int, shape: tuple[int, ...] = (3, 2, 2)) -> np.ndarray:
    return np.full(shape, value % 2, dtype=np.float32)


def test_buffer_len_and_wraparound() -> None:
    buf = ReplayBuffer(capacity=3, observation_shape=(3, 2, 2), seed=0)
    for i in range(2):
        buf.add(_obs(i), i, float(i), _obs(i + 1), False)
    assert len(buf) == 2
    for i in range(2, 5):
        buf.add(_obs(i), i, float(i), _obs(i + 1), i == 4)
    assert len(buf) == 3
    assert sorted(buf._actions.tolist()) == [2, 3, 4]  # 0 và 1 (cũ nhất) đã bị ghi đè


def test_buffer_sample_shapes_and_determinism() -> None:
    def filled(seed: int) -> ReplayBuffer:
        buf = ReplayBuffer(capacity=50, observation_shape=(3, 2, 2), seed=seed)
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
    buf = ReplayBuffer(capacity=4, observation_shape=(3, 2, 2))
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
    y = compute_bellman_targets(batch, _ConstNet([0.0, 5.0, 3.0, -1.0]), gamma=0.9)
    assert y.tolist() == pytest.approx([1.0 + 0.9 * 5.0, 2.0])  # terminated -> y = r


def test_targets_use_target_network_only() -> None:
    agent = DeepQNetworkAgent((3, 3, 3), DeepQNetworkSettings(total_timesteps=10, batch_size=2, replay_buffer_capacity=10))
    batch = {"next_obs": torch.rand(4, 3, 3, 3), "rewards": torch.zeros(4), "terminated": torch.zeros(4)}
    batch = {k: v.to(agent.device) for k, v in batch.items()}  # cùng thiết bị với mạng (CPU hoặc GPU)
    before = compute_bellman_targets(batch, agent.target_network, 0.99)
    with torch.no_grad():
        for p in agent.online_network.parameters():
            p.add_(1.0)
    assert torch.allclose(before, compute_bellman_targets(batch, agent.target_network, 0.99))


def test_linear_epsilon() -> None:
    cfg = DeepQNetworkSettings(total_timesteps=1000, epsilon_decay_fraction=0.5)
    assert linear_epsilon_for_step(0, cfg) == pytest.approx(1.0)
    assert linear_epsilon_for_step(500, cfg) == pytest.approx(0.05)
    assert linear_epsilon_for_step(999, cfg) == pytest.approx(0.05)


def test_resolve_device() -> None:
    assert choose_compute_device("cpu") == torch.device("cpu")
    assert choose_compute_device("auto").type == ("cuda" if torch.cuda.is_available() else "cpu")
    with pytest.raises(ValueError):
        choose_compute_device("tpu")


def test_load_on_cpu(tmp_path: Path) -> None:
    agent = DeepQNetworkAgent((3, 3, 3), DeepQNetworkSettings(device="auto"))
    loaded = DeepQNetworkAgent.load(agent.save(tmp_path / "m.pt"), device="cpu")
    assert next(loaded.online_network.parameters()).device.type == "cpu"


def test_valid_action_mask_matches_env() -> None:
    from environment.observation_decoding import safe_actions_from_observation

    env = VacuumCleaningEnv("furniture_8x8")
    for seed in range(30):
        obs, _ = env.reset(seed=seed)
        mask = safe_action_mask(torch.as_tensor(obs).unsqueeze(0))[0]
        assert np.flatnonzero(mask.numpy()).tolist() == safe_actions_from_observation(obs)


def test_masked_targets_ignore_blocked_actions() -> None:
    obs = torch.zeros(1, 3, 2, 2)
    obs[0, 0, 0, 0] = 1.0  # robot ở góc trên-trái: chỉ đi xuống (1) hoặc phải (3) được
    batch = {"next_obs": obs, "rewards": torch.zeros(1), "terminated": torch.zeros(1)}
    net = _ConstNet([100.0, 1.0, 100.0, 2.0])  # hành động bị chặn có Q rất lớn
    assert compute_bellman_targets(batch, net, 1.0)[0].item() == pytest.approx(100.0)
    assert compute_bellman_targets(batch, net, 1.0, use_action_mask=True)[0].item() == pytest.approx(2.0)


def test_masked_agent_never_collides() -> None:
    agent = DeepQNetworkAgent((3, 8, 8), DeepQNetworkSettings(use_action_mask=True, seed=1))
    result = run_single_episode(agent, VacuumCleaningEnv("furniture_8x8"), seed=0)
    assert result["collisions"] == 0


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        DeepQNetworkSettings(loss_function="l1")
    with pytest.raises(ValueError):
        DeepQNetworkSettings(batch_size=100, replay_buffer_capacity=10)


def test_smoke_training_writes_logs(tmp_path: Path) -> None:
    cfg = DeepQNetworkSettings(total_timesteps=500, random_warmup_steps=100, evaluate_every_steps=250, evaluation_episode_count=2,
                    log_every_steps=100, batch_size=16, replay_buffer_capacity=1000, checkpoint_steps=(0, 300))
    agent = DeepQNetworkAgent((3, 3, 3), cfg)
    logs = agent.train(functools.partial(VacuumCleaningEnv, "empty_3x3"), tmp_path / "run")
    for name in ("train_episodes.csv", "train_losses.csv", "eval.csv", "model.pt"):
        assert (tmp_path / "run" / name).exists()
    assert (tmp_path / "run" / "checkpoints" / "step0.pt").exists()
    assert (tmp_path / "run" / "checkpoints" / "step300.pt").exists()
    assert list(logs["eval"]["global_step"]) == [250, 500]
    assert {"loss", "mean_q", "mean_target", "grad_norm"} <= set(logs["losses"].columns)


def test_save_load_same_actions(tmp_path: Path) -> None:
    agent = DeepQNetworkAgent((3, 4, 4), DeepQNetworkSettings(seed=3))
    loaded = DeepQNetworkAgent.load(agent.save(tmp_path / "m.pt"))
    env = VacuumCleaningEnv("obstacle_4x4")
    for seed in range(10):
        obs, _ = env.reset(seed=seed)
        assert loaded.act(obs) == agent.act(obs)


def test_rejects_wrong_env_shape(tmp_path: Path) -> None:
    agent = DeepQNetworkAgent((3, 4, 4), DeepQNetworkSettings(total_timesteps=10, batch_size=2, replay_buffer_capacity=10))
    with pytest.raises(ValueError):
        agent.train(functools.partial(VacuumCleaningEnv, "empty_3x3"), tmp_path / "x")


@pytest.mark.slow
def test_dqn_learns_3x3(tmp_path: Path) -> None:
    factory = functools.partial(VacuumCleaningEnv, "empty_3x3", start=(0, 0))
    agent = DeepQNetworkAgent((3, 3, 3), DeepQNetworkSettings(total_timesteps=30_000, evaluate_every_steps=10_000, evaluation_episode_count=1))
    agent.train(factory, tmp_path / "dqn3")
    assert run_single_episode(agent, factory(), seed=0)["coverage"] == 1.0


def test_vectorized_egocentric_training(tmp_path: Path) -> None:
    from environment.random_room_vacuum_env import RandomRoomVacuumCleaningEnv

    cfg = DeepQNetworkSettings(total_timesteps=640, random_warmup_steps=64, parallel_env_count=8, use_egocentric_observation=True, use_action_mask=True,
                    batch_size=32, replay_buffer_capacity=2000, steps_per_network_update=8, evaluate_every_steps=320, evaluation_episode_count=2,
                    log_every_steps=160, checkpoint_steps=(0, 320))
    agent = DeepQNetworkAgent((3, 5, 5), cfg)
    assert agent.network_input_shape == (3, 9, 9)
    logs = agent.train(functools.partial(RandomRoomVacuumCleaningEnv, 5, 5, 0.2), tmp_path / "vec")
    assert list(logs["eval"]["global_step"]) == [320, 640]
    assert (tmp_path / "vec" / "checkpoints" / "step320.pt").exists()
    assert len(logs["losses"]) > 0
    loaded = DeepQNetworkAgent.load(tmp_path / "vec" / "model.pt", device="cpu")
    assert loaded.settings.use_egocentric_observation and loaded.network_input_shape == (3, 9, 9)
    result = run_single_episode(loaded, RandomRoomVacuumCleaningEnv(5, 5, 0.2), seed=0)
    assert result["collisions"] == 0  # có mask thì không bao giờ đâm tường


def test_vectorized_update_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tỷ lệ cập nhật/bước môi trường giữ đúng 1/train_freq khi chạy nhiều môi trường."""
    calls = []
    agent = DeepQNetworkAgent((3, 3, 3), DeepQNetworkSettings(total_timesteps=400, random_warmup_steps=80, parallel_env_count=4, steps_per_network_update=4,
                                          batch_size=8, replay_buffer_capacity=500, evaluate_every_steps=400, evaluation_episode_count=1))
    original = agent.learn_from_replay_batch
    monkeypatch.setattr(agent, "learn_from_replay_batch", lambda buf: calls.append(1) or original(buf))
    agent.train(functools.partial(VacuumCleaningEnv, "empty_3x3"), tmp_path / "count")
    assert len(calls) == (400 - 80) // 4


def test_vectorized_steps_match_single_env_schedule(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """total_timesteps không chia hết cho n_envs: không đi quá số bước, target/checkpoint đúng mốc."""
    agent = DeepQNetworkAgent((3, 3, 3), DeepQNetworkSettings(total_timesteps=1000, random_warmup_steps=100, parallel_env_count=3, steps_per_network_update=1,
                                          target_network_sync_interval_steps=2, batch_size=8, replay_buffer_capacity=500,
                                          evaluate_every_steps=1000, evaluation_episode_count=1, checkpoint_steps=(500,)))
    copies = []
    original = agent.target_network.load_state_dict
    monkeypatch.setattr(agent.target_network, "load_state_dict", lambda sd: copies.append(1) or original(sd))
    logs = agent.train(functools.partial(VacuumCleaningEnv, "empty_3x3"), tmp_path / "sched")
    assert list(logs["eval"]["global_step"]) == [1000]
    assert logs["episodes"]["global_step"].max() <= 1000
    assert len(copies) == 1000 // 2
    assert (tmp_path / "sched" / "checkpoints" / "step500.pt").exists()


def test_learning_starts_must_cover_batch() -> None:
    with pytest.raises(ValueError):
        DeepQNetworkSettings(random_warmup_steps=100, batch_size=512)


def test_keep_best_restores_best_checkpoint(tmp_path: Path) -> None:
    cfg = DeepQNetworkSettings(total_timesteps=600, random_warmup_steps=50, batch_size=16, replay_buffer_capacity=1000, evaluate_every_steps=150,
                    evaluation_episode_count=2, keep_best_checkpoint=True)
    agent = DeepQNetworkAgent((3, 3, 3), cfg)
    logs = agent.train(functools.partial(VacuumCleaningEnv, "empty_3x3"), tmp_path / "best")
    assert (tmp_path / "best" / "best.pt").exists() and (tmp_path / "best" / "last.pt").exists()
    evals = logs["eval"]
    assert evals["selected"].sum() == 1
    chosen = evals[evals["selected"]].iloc[0]
    best_score = max(zip(evals["success_rate"], evals["coverage_mean"], -evals["steps_mean"]))
    assert (chosen["success_rate"], chosen["coverage_mean"], -chosen["steps_mean"]) == best_score
    model = DeepQNetworkAgent.load(tmp_path / "best" / "model.pt", device="cpu")
    best = DeepQNetworkAgent.load(tmp_path / "best" / "best.pt", device="cpu")
    for a, b in zip(model.online_network.state_dict().values(), best.online_network.state_dict().values()):
        assert torch.equal(a, b)
