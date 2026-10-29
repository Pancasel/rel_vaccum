"""Mục tiêu Bellman y = r + γ (1 − terminated) max_a Q_target(s', a) cho một lô transition."""

from __future__ import annotations

import torch
from torch import nn

from agents.deep_q_network.wall_action_masking import BLOCKED_ACTION_Q_VALUE, safe_action_mask


def compute_bellman_targets(batch: dict[str, torch.Tensor], target_network: nn.Module, gamma: float,
                            use_action_mask: bool = False) -> torch.Tensor:
    """y = r + γ (1 − terminated) max_a Q_target(s', a)  (max chỉ trên hành động hợp lệ nếu use_action_mask).

    Hàm chỉ nhận target network nên không thể vô tình dùng nhầm mạng online.
    """
    with torch.no_grad():
        # Sutton & Barto 6.5: "R + γ max_a Q(S',a)"; max lấy trên target network (Mnih 2015).
        next_state_q_values = target_network(batch["next_obs"])
        if use_action_mask:
            next_state_q_values = next_state_q_values.masked_fill(
                ~safe_action_mask(batch["next_obs"]), BLOCKED_ACTION_Q_VALUE)
        best_next_q_value = next_state_q_values.max(dim=1).values
        # terminated = 1 -> S' là terminal -> Q(S',·) = 0 -> mục tiêu chỉ còn R.
        return batch["rewards"] + gamma * (1.0 - batch["terminated"]) * best_next_q_value
