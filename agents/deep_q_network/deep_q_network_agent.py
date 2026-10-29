"""Agent DQN: chọn hành động và làm một bước học. Vòng huấn luyện nằm ở training_loop.py."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Callable

import gymnasium
import numpy as np
import pandas as pd
import torch
from torch import nn

from agents.deep_q_network.bellman_targets import compute_bellman_targets
from agents.deep_q_network.compute_device import choose_compute_device
from agents.deep_q_network.deep_q_network_settings import DeepQNetworkSettings
from agents.deep_q_network.q_value_network import QValueNetwork
from agents.deep_q_network.replay_buffer import ReplayBuffer
from agents.deep_q_network.wall_action_masking import BLOCKED_ACTION_Q_VALUE, safe_action_mask
from environment.egocentric_observation import egocentric_observation_shape, to_egocentric_observation
from project_utils.random_seeding import set_global_seed

ACTION_COUNT: int = 4


class DeepQNetworkAgent:
    """Agent DQN, cùng giao diện Agent (act) với baseline và Q-learning dạng bảng."""

    def __init__(self, observation_shape: tuple[int, int, int], settings: DeepQNetworkSettings) -> None:
        if len(observation_shape) != 3 or observation_shape[0] != 3:
            raise ValueError(f"obs_shape phải là (3, H, W), nhận được {observation_shape}.")
        self.observation_shape = tuple(int(size) for size in observation_shape)  # observation gốc của môi trường
        self.settings = settings
        # Shape mà mạng thực sự nhận (khác observation_shape khi dùng quan sát lấy robot làm tâm).
        if settings.use_egocentric_observation:
            self.network_input_shape = egocentric_observation_shape(*self.observation_shape[1:])
        else:
            self.network_input_shape = self.observation_shape
        self.device = choose_compute_device(settings.device)
        set_global_seed(settings.seed)  # khởi tạo trọng số tái lập được
        self.online_network = QValueNetwork(self.network_input_shape[1], self.network_input_shape[2]).to(self.device)
        self.target_network = copy.deepcopy(self.online_network)
        self.target_network.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=settings.learning_rate)
        self.random_generator = np.random.default_rng(settings.seed)

    # ------------------------------------------------------------------ chọn hành động
    def prepare_network_input(self, observation: np.ndarray) -> np.ndarray:
        """Observation gốc -> đầu vào của mạng (dịch về tâm robot nếu bật egocentric)."""
        if self.settings.use_egocentric_observation:
            return to_egocentric_observation(observation)
        return observation

    def predict_q_values(self, observation: np.ndarray) -> np.ndarray:
        """Q(s, ·) của một observation gốc theo mạng online."""
        with torch.no_grad():
            network_input = torch.as_tensor(self.prepare_network_input(observation), dtype=torch.float32,
                                            device=self.device).unsqueeze(0)
            return self.online_network(network_input)[0].cpu().numpy()

    def _allowed_actions(self, observation: np.ndarray) -> np.ndarray:
        """Chỉ số các hành động được phép (tất cả nếu tắt action mask)."""
        if not self.settings.use_action_mask:
            return np.arange(ACTION_COUNT)
        observation_batch = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        return np.flatnonzero(safe_action_mask(observation_batch)[0].numpy())

    def choose_random_action(self, observation: np.ndarray) -> int:
        """Hành động ngẫu nhiên đều (trong các hành động hợp lệ nếu bật action mask)."""
        if not self.settings.use_action_mask:
            # Giữ đúng cách rút số cũ để các kết quả đã chạy (không mask) tái lập được theo seed.
            return int(self.random_generator.integers(ACTION_COUNT))
        return int(self.random_generator.choice(self._allowed_actions(observation)))

    def act(self, observation: np.ndarray, info: dict | None = None, epsilon: float = 0.0) -> int:
        """ε-greedy theo mạng online; epsilon=0 (mặc định) là chính sách tham lam để đánh giá."""
        if epsilon > 0.0 and self.random_generator.random() < epsilon:
            return self.choose_random_action(observation)
        network_input = self.prepare_network_input(observation)
        return int(self.choose_greedy_actions_for_batch(network_input[None])[0])

    def choose_greedy_actions_for_batch(self, network_inputs: np.ndarray) -> np.ndarray:
        """Hành động tham lam cho một lô đầu vào mạng (B, 3, H', W'), có che hành động nếu bật action mask."""
        input_tensor = torch.as_tensor(network_inputs, dtype=torch.float32)
        with torch.no_grad():
            q_values = self.online_network(input_tensor.to(self.device))
            if self.settings.use_action_mask:
                q_values = q_values.masked_fill(~safe_action_mask(input_tensor).to(self.device),
                                                BLOCKED_ACTION_Q_VALUE)
        return q_values.argmax(dim=1).cpu().numpy()

    # ------------------------------------------------------------------ một bước học
    def learn_from_replay_batch(self, replay_buffer: ReplayBuffer) -> dict[str, torch.Tensor]:
        """Lấy một lô từ replay buffer và làm một bước gradient (tương ứng dòng cập nhật Q của 6.5).

        Trả các số đo dạng tensor (chưa đọc về CPU): mỗi lần đọc `.item()` bắt CPU chờ GPU tính xong,
        nên chỉ đọc khi ghi log (xem `average_training_metrics`), không đọc ở mỗi bước cập nhật.
        """
        settings = self.settings
        raw_batch = replay_buffer.sample(settings.batch_size)
        batch = {name: torch.as_tensor(values, device=self.device) for name, values in raw_batch.items()}
        batch["rewards"] = batch["rewards"] * settings.reward_scale
        bellman_targets = compute_bellman_targets(batch, self.target_network, settings.gamma,
                                                  settings.use_action_mask)
        q_values_all_actions = self.online_network(batch["obs"])
        q_value_of_taken_action = q_values_all_actions.gather(1, batch["actions"].unsqueeze(1)).squeeze(1)
        # "Q(S,A) <- Q(S,A) + α[target − Q(S,A)]" trở thành: giảm sai khác (Q(S,A) − target) bằng gradient.
        if settings.loss_function == "huber":
            loss = nn.functional.smooth_l1_loss(q_value_of_taken_action, bellman_targets)
        else:
            loss = nn.functional.mse_loss(q_value_of_taken_action, bellman_targets)
        self.optimizer.zero_grad()
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(self.online_network.parameters(), settings.max_gradient_norm)
        self.optimizer.step()
        # Tên khóa là tên cột trong train_losses.csv, không đổi.
        return {"loss": loss.detach(), "mean_q": q_value_of_taken_action.detach().mean(),
                "mean_target": bellman_targets.mean(), "grad_norm": gradient_norm.detach()}

    @staticmethod
    def average_training_metrics(pending_metrics: list[dict[str, torch.Tensor]]) -> dict[str, float]:
        """Trung bình các số đo của nhiều lần cập nhật (một lần đồng bộ GPU duy nhất)."""
        return {name: float(torch.stack([metrics[name] for metrics in pending_metrics]).mean().item())
                for name in pending_metrics[0]}

    @staticmethod
    def checkpoint_selection_score(evaluation_row: dict[str, Any]) -> tuple[float, float, float]:
        """Thứ tự ưu tiên khi chọn mô hình: tỷ lệ dọn xong, rồi tỷ lệ đã dọn, rồi ít bước hơn."""
        return (float(evaluation_row["success_rate"]), float(evaluation_row["coverage_mean"]),
                -float(evaluation_row["steps_mean"]))

    # ------------------------------------------------------------------ huấn luyện, lưu / đọc
    def train(self, env_factory: Callable[[], gymnasium.Env], log_dir: str | os.PathLike,
              eval_env_factory: Callable[[], gymnasium.Env] | None = None) -> dict[str, pd.DataFrame]:
        """Huấn luyện theo Algorithm 1 của Mnih et al. (2015); xem training_loop.DeepQNetworkTrainer."""
        from agents.deep_q_network.training_loop import DeepQNetworkTrainer

        return DeepQNetworkTrainer(self, env_factory, log_dir, eval_env_factory).run()

    def save(self, path: str | os.PathLike) -> Path:
        """Lưu trọng số mạng online, cấu hình và shape observation."""
        from agents.deep_q_network.model_checkpoint import save_agent

        return save_agent(self, path)

    @classmethod
    def load(cls, path: str | os.PathLike, device: str = "auto") -> "DeepQNetworkAgent":
        """Đọc agent đã lưu (mạng target được đặt bằng mạng online)."""
        from agents.deep_q_network.model_checkpoint import load_agent

        return load_agent(path, device)
