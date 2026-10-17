"""Q-learning dạng bảng — model chính của dự án.

Sutton & Barto 2018, mục 6.5 "Q-learning (off-policy TD control) for estimating π ≈ π*".
Mỗi dòng code bên dưới có comment trích đúng dòng pseudocode tương ứng trong sách.
"""

from __future__ import annotations

from typing import Any

import gymnasium

from tabular.common import TabularAgent


class QLearningAgent(TabularAgent):
    """Q-learning: học Q* trực tiếp, bất kể chính sách đang dùng để đi (off-policy).

    Mục tiêu cập nhật dùng max_a Q(S', a) — giá trị của hành động TỐT NHẤT ở trạng thái sau,
    không phải hành động agent sẽ thực sự làm (khác SARSA). Nhờ vậy dù agent vẫn khám phá
    bằng ε-greedy, bảng Q vẫn hội tụ về giá trị của chính sách tối ưu.
    """

    algo_name = "q_learning"

    def _run_episode(self, env: gymnasium.Env, epsilon: float, seed: int | None) -> dict[str, Any]:
        cfg = self.cfg
        # "Initialize Q(s,a) arbitrarily, except Q(terminal,·) = 0":
        #   bảng Q khởi tạo q_init khi gặp trạng thái lần đầu; trạng thái terminal không bao giờ
        #   được dùng để bootstrap (xem dòng target bên dưới) nên coi như Q(terminal,·) = 0.
        # "Loop for each episode:"  "Initialize S"
        obs, info = env.reset(seed=seed)
        state = self.encoder.encode_obs(obs)
        start_key = state
        total_reward = 0.0
        # "Loop for each step of episode:"
        while True:
            # "Choose A from S using policy derived from Q (e.g., ε-greedy)"
            action = self.epsilon_greedy(state, epsilon)
            # "Take action A, observe R, S'"
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = self.encoder.encode_obs(obs)
            # "Q(S,A) <- Q(S,A) + α[R + γ max_a Q(S',a) − Q(S,A)]"
            #   terminated: S' là terminal, Q(S',·) = 0 nên target = R (không bootstrap).
            #   truncated (hết bước): S' KHÔNG phải terminal thật, vẫn bootstrap (C2.11).
            target = reward if terminated else reward + cfg.gamma * float(self.q_values(next_state).max())
            row = self._q_row(state)
            row[action] += cfg.alpha * (target - row[action])
            total_reward += reward
            # "S <- S'"
            state = next_state
            # "until S is terminal" (thêm: dừng khi hết số bước cho phép)
            if terminated or truncated:
                break
        return {"return": total_reward, "coverage": info["coverage"], "steps": info["steps"],
                "terminated": terminated, "truncated": truncated, "start_key": start_key}
