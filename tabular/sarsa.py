"""SARSA dạng bảng — đối chứng on-policy cho Q-learning (thí nghiệm E5).

Sutton & Barto 2018, mục 6.4 "Sarsa (on-policy TD control) for estimating Q ≈ q*".
Khác Q-learning đúng một chỗ: mục tiêu dùng Q(S', A') của hành động A' mà agent SẼ làm thật
(chọn bằng ε-greedy), thay vì max_a Q(S', a).
"""

from __future__ import annotations

from typing import Any

import gymnasium

from tabular.common import TabularAgent


class SarsaAgent(TabularAgent):
    """SARSA: học giá trị của chính sách ε-greedy đang dùng (on-policy).

    Vì mục tiêu tính cả các bước khám phá ngẫu nhiên, SARSA học chính sách "thận trọng" hơn
    và Q của nó hội tụ về giá trị của chính sách ε-greedy, không phải Q* (khi ε giữ > 0).
    """

    algo_name = "sarsa"

    def _run_episode(self, env: gymnasium.Env, epsilon: float, seed: int | None) -> dict[str, Any]:
        cfg = self.cfg
        # "Loop for each episode:"  "Initialize S"
        obs, info = env.reset(seed=seed)
        state = self.encoder.encode_obs(obs)
        start_key = state
        # "Choose A from S using policy derived from Q (e.g., ε-greedy)"
        action = self.epsilon_greedy(state, epsilon)
        total_reward = 0.0
        # "Loop for each step of episode:"
        while True:
            # "Take action A, observe R, S'"
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = self.encoder.encode_obs(obs)
            # "Choose A' from S' using policy derived from Q (e.g., ε-greedy)"
            next_action = self.epsilon_greedy(next_state, epsilon)
            # "Q(S,A) <- Q(S,A) + α[R + γ Q(S',A') − Q(S,A)]"
            #   terminated: Q(S',·) = 0 nên target = R; truncated vẫn bootstrap (C2.11), như Q-learning.
            target = reward if terminated else reward + cfg.gamma * float(self.q_values(next_state)[next_action])
            row = self._q_row(state)
            row[action] += cfg.alpha * (target - row[action])
            total_reward += reward
            # "S <- S'; A <- A'"
            state, action = next_state, next_action
            # "until S is terminal" (thêm: dừng khi hết số bước cho phép)
            if terminated or truncated:
                break
        return {"return": total_reward, "coverage": info["coverage"], "steps": info["steps"],
                "terminated": terminated, "truncated": truncated, "start_key": start_key}
