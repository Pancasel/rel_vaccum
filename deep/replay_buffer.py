"""Replay buffer cho DQN (Mnih et al. 2015).

Vì sao cần: các bước liên tiếp trong một episode rất giống nhau (tương quan cao). Học trực
tiếp theo thứ tự đó làm mạng "quên" và dao động. Lưu lại rồi lấy mẫu NGẪU NHIÊN một lô
giúp các mẫu gần như độc lập, đồng thời dùng lại mỗi kinh nghiệm nhiều lần.
"""

from __future__ import annotations

import numpy as np


class ReplayBuffer:
    """Bộ đệm vòng (circular) với mảng numpy cấp phát trước."""

    def __init__(self, capacity: int, obs_shape: tuple[int, ...], seed: int = 0) -> None:
        if capacity < 1:
            raise ValueError(f"capacity phải >= 1, nhận được {capacity}.")
        self.capacity = capacity
        self.obs_shape = tuple(obs_shape)
        # Observation chỉ gồm 0/1 nên lưu uint8 (tiết kiệm 4 lần bộ nhớ), đổi sang float32 khi lấy mẫu.
        self._obs = np.zeros((capacity, *self.obs_shape), dtype=np.uint8)
        self._next_obs = np.zeros((capacity, *self.obs_shape), dtype=np.uint8)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._terminated = np.zeros(capacity, dtype=np.float32)
        self._pos = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    def add(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, terminated: bool) -> None:
        """Thêm một transition. Chỉ lưu cờ `terminated`, KHÔNG lưu `truncated`.

        Vì sao: khi bị cắt do hết bước (truncated), trạng thái sau vẫn có giá trị tương lai,
        nên mục tiêu vẫn phải bootstrap; chỉ terminated mới đặt Q(s') = 0 (C2.11).
        Ghi bằng phép gán vào mảng có sẵn nên dữ liệu luôn được SAO CHÉP, sửa obs gốc sau đó
        không làm hỏng buffer.
        """
        if np.shape(obs) != self.obs_shape or np.shape(next_obs) != self.obs_shape:
            raise ValueError(f"Observation phải có shape {self.obs_shape}.")
        self._obs[self._pos] = obs
        self._next_obs[self._pos] = next_obs
        self._actions[self._pos] = action
        self._rewards[self._pos] = reward
        self._terminated[self._pos] = float(terminated)
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Lấy ngẫu nhiên (không lặp) một lô transition.

        Raises:
            ValueError: batch_size lớn hơn số phần tử đang có.
        """
        if not 1 <= batch_size <= self._size:
            raise ValueError(f"batch_size={batch_size} không hợp lệ khi buffer có {self._size} phần tử.")
        idx = self._rng.choice(self._size, size=batch_size, replace=False)
        return {
            "obs": self._obs[idx].astype(np.float32),
            "actions": self._actions[idx],
            "rewards": self._rewards[idx],
            "next_obs": self._next_obs[idx].astype(np.float32),
            "terminated": self._terminated[idx],
        }
