"""Replay buffer (bộ nhớ kinh nghiệm) cho DQN (Mnih et al. 2015).

Vì sao cần: các bước liên tiếp trong một episode rất giống nhau (tương quan cao). Học trực
tiếp theo thứ tự đó làm mạng "quên" và dao động. Lưu lại rồi lấy mẫu NGẪU NHIÊN một lô
giúp các mẫu gần như độc lập, đồng thời dùng lại mỗi kinh nghiệm nhiều lần.
"""

from __future__ import annotations

import numpy as np


class ReplayBuffer:
    """Bộ đệm vòng (circular): đầy thì ghi đè kinh nghiệm cũ nhất. Mảng numpy được cấp phát trước."""

    def __init__(self, capacity: int, observation_shape: tuple[int, ...], seed: int = 0) -> None:
        if capacity < 1:
            raise ValueError(f"capacity phải >= 1, nhận được {capacity}.")
        self.capacity = capacity
        self.observation_shape = tuple(observation_shape)
        # Observation chỉ gồm 0/1 nên lưu uint8 (tiết kiệm 4 lần bộ nhớ), đổi sang float32 khi lấy mẫu.
        self._observations = np.zeros((capacity, *self.observation_shape), dtype=np.uint8)
        self._next_observations = np.zeros((capacity, *self.observation_shape), dtype=np.uint8)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._terminated_flags = np.zeros(capacity, dtype=np.float32)
        self._next_write_index = 0
        self._stored_count = 0
        self._random_generator = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._stored_count

    def add(self, observation: np.ndarray, action: int, reward: float, next_observation: np.ndarray,
            terminated: bool) -> None:
        """Thêm một transition. Chỉ lưu cờ `terminated`, KHÔNG lưu `truncated`.

        Vì sao: khi bị cắt do hết bước (truncated), trạng thái sau vẫn có giá trị tương lai,
        nên mục tiêu vẫn phải bootstrap; chỉ terminated mới đặt Q(s') = 0 (C2.11).
        Ghi bằng phép gán vào mảng có sẵn nên dữ liệu luôn được SAO CHÉP, sửa observation gốc sau đó
        không làm hỏng buffer.
        """
        if np.shape(observation) != self.observation_shape or np.shape(next_observation) != self.observation_shape:
            raise ValueError(f"Observation phải có shape {self.observation_shape}.")
        write_index = self._next_write_index
        self._observations[write_index] = observation
        self._next_observations[write_index] = next_observation
        self._actions[write_index] = action
        self._rewards[write_index] = reward
        self._terminated_flags[write_index] = float(terminated)
        self._next_write_index = (write_index + 1) % self.capacity
        self._stored_count = min(self._stored_count + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Lấy ngẫu nhiên (không lặp) một lô transition.

        Raises:
            ValueError: batch_size lớn hơn số phần tử đang có.
        """
        if not 1 <= batch_size <= self._stored_count:
            raise ValueError(f"batch_size={batch_size} không hợp lệ khi buffer có {self._stored_count} phần tử.")
        sampled_indices = self._random_generator.choice(self._stored_count, size=batch_size, replace=False)
        # Tên khóa: obs/next_obs/rewards/actions/terminated đúng thứ tự một transition (s, a, r, s', done).
        return {
            "obs": self._observations[sampled_indices].astype(np.float32),
            "actions": self._actions[sampled_indices],
            "rewards": self._rewards[sampled_indices],
            "next_obs": self._next_observations[sampled_indices].astype(np.float32),
            "terminated": self._terminated_flags[sampled_indices],
        }
