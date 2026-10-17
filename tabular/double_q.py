"""Double Q-learning dạng bảng — giảm độ chệch tối đa hóa của Q-learning (thí nghiệm E5).

Sutton & Barto 2018, mục 6.7 "Double Q-learning, for estimating Q1 ≈ Q2 ≈ q*".
Q-learning dùng cùng một bảng để vừa CHỌN hành động tốt nhất vừa ĐÁNH GIÁ nó (max), nên nhiễu
ước lượng bị cộng dồn thành thiên lệch dương (maximization bias). Double Q tách hai việc đó
ra hai bảng độc lập.
"""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np

from tabular.common import N_ACTIONS, TabularAgent, TabularConfig


class DoubleQLearningAgent(TabularAgent):
    """Double Q-learning: hai bảng Q1, Q2; mỗi bước cập nhật ngẫu nhiên một bảng.

    `self.q` giữ trung bình (Q1 + Q2) / 2, được cập nhật cùng lúc, nên chọn hành động, đánh giá,
    lưu/đọc dùng lại nguyên phần chung của TabularAgent (sách: hành động theo Q1 + Q2).
    """

    algo_name = "double_q"

    def __init__(self, grid: np.ndarray, cfg: TabularConfig) -> None:
        super().__init__(grid, cfg)
        self.q_tables: tuple[dict[tuple, np.ndarray], dict[tuple, np.ndarray]] = ({}, {})

    def _table_row(self, table: dict[tuple, np.ndarray], state_key: tuple) -> np.ndarray:
        row = table.get(state_key)
        if row is None:
            row = np.full(N_ACTIONS, self.cfg.q_init, dtype=np.float64)
            table[state_key] = row
        return row

    def _run_episode(self, env: gymnasium.Env, epsilon: float, seed: int | None) -> dict[str, Any]:
        cfg = self.cfg
        # "Initialize Q1(s,a) and Q2(s,a) ... arbitrarily, except Q(terminal,·) = 0"
        obs, info = env.reset(seed=seed)
        state = self.encoder.encode_obs(obs)
        start_key = state
        total_reward = 0.0
        while True:
            # "Choose A from S using the policy ε-greedy in Q1 + Q2"
            action = self.epsilon_greedy(state, epsilon)
            # "Take action A, observe R, S'"
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = self.encoder.encode_obs(obs)
            # "With 0.5 probability: Q1(S,A) <- Q1(S,A) + α[R + γ Q2(S', argmax_a Q1(S',a)) − Q1(S,A)]
            #  else: (đổi vai Q1 và Q2)"
            first = self.rng.random() < 0.5
            update, other = self.q_tables if first else self.q_tables[::-1]
            if terminated:
                target = reward  # Q(terminal,·) = 0
            else:
                best = int(np.argmax(self._table_row(update, next_state)))  # bảng đang cập nhật CHỌN
                target = reward + cfg.gamma * float(self._table_row(other, next_state)[best])  # bảng kia ĐÁNH GIÁ
            row = self._table_row(update, state)
            row[action] += cfg.alpha * (target - row[action])
            q1, q2 = (self._table_row(t, state) for t in self.q_tables)
            self._q_row(state)[:] = (q1 + q2) / 2
            total_reward += reward
            state = next_state
            if terminated or truncated:
                break
        return {"return": total_reward, "coverage": info["coverage"], "steps": info["steps"],
                "terminated": terminated, "truncated": truncated, "start_key": start_key}
