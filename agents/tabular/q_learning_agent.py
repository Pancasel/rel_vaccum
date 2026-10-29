"""Q-learning dạng bảng — model chính của dự án.

Sutton & Barto 2018, mục 6.5 "Q-learning (off-policy TD control) for estimating π ≈ π*".
Mỗi dòng code bên dưới có comment trích đúng dòng pseudocode tương ứng trong sách.
"""

from __future__ import annotations

from typing import Any

import gymnasium

from agents.tabular.tabular_agent_base import TabularAgentBase


class QLearningAgent(TabularAgentBase):
    """Q-learning: học Q* trực tiếp, bất kể chính sách đang dùng để đi (off-policy).

    Mục tiêu cập nhật dùng max_a Q(S', a) — giá trị của hành động TỐT NHẤT ở trạng thái sau,
    không phải hành động agent sẽ thực sự làm (khác SARSA). Nhờ vậy dù agent vẫn khám phá
    bằng ε-greedy, bảng Q vẫn hội tụ về giá trị của chính sách tối ưu.
    """

    algo_name = "q_learning"

    def _run_one_episode(self, env: gymnasium.Env, epsilon: float, seed: int | None) -> dict[str, Any]:
        settings = self.settings
        # "Initialize Q(s,a) arbitrarily, except Q(terminal,·) = 0":
        #   bảng Q khởi tạo initial_q_value khi gặp trạng thái lần đầu; trạng thái terminal không bao giờ
        #   được dùng để bootstrap (xem dòng bellman_target bên dưới) nên coi như Q(terminal,·) = 0.
        # "Loop for each episode:"  "Initialize S"
        observation, info = env.reset(seed=seed)
        state = self.state_encoder.encode_observation(observation)
        start_key = state
        total_reward = 0.0
        # "Loop for each step of episode:"
        while True:
            # "Choose A from S using policy derived from Q (e.g., ε-greedy)"
            action = self.choose_epsilon_greedy_action(state, epsilon)
            # "Take action A, observe R, S'"
            observation, reward, terminated, truncated, info = env.step(action)
            next_state = self.state_encoder.encode_observation(observation)
            # "Q(S,A) <- Q(S,A) + α[R + γ max_a Q(S',a) − Q(S,A)]"
            #   terminated: S' là terminal, Q(S',·) = 0 nên target = R (không bootstrap).
            #   truncated (hết bước): S' KHÔNG phải terminal thật, vẫn bootstrap (C2.11).
            if terminated:
                bellman_target = reward
            else:
                bellman_target = reward + settings.gamma * float(self.read_q_values(next_state).max())
            q_values_of_state = self._q_values_for_update(state)
            q_values_of_state[action] += settings.alpha * (bellman_target - q_values_of_state[action])
            total_reward += reward
            # "S <- S'"
            state = next_state
            # "until S is terminal" (thêm: dừng khi hết số bước cho phép)
            if terminated or truncated:
                break
        return {"return": total_reward, "coverage": info["coverage"], "steps": info["steps"],
                "terminated": terminated, "truncated": truncated, "start_key": start_key}
