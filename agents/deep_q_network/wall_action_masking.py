"""Action masking: che các hành động đâm vào tường/vật cản (cải tiến v3)."""

from __future__ import annotations

import torch
from torch import nn

# Giá trị Q gán cho hành động bị che. Dùng số hữu hạn (không phải -inf) để 0 * giá trị này = 0, không ra NaN.
BLOCKED_ACTION_Q_VALUE: float = -1e9


def safe_action_mask(observation_batch: torch.Tensor) -> torch.Tensor:
    """Mặt nạ (B, 4) bool: hành động nào KHÔNG đâm vào tường/vật cản, tính từ observation (B, 3, H, W).

    Đệm viền bằng vật cản rồi lấy ô vật cản ngay trên/dưới/trái/phải vị trí robot (thứ tự hành động D-02).
    """
    robot_channel = observation_batch[:, 0]
    padded_obstacles = nn.functional.pad(observation_batch[:, 2], (1, 1, 1, 1), value=1.0)
    # Bốn lát cắt: ô vật cản phía trên, dưới, trái, phải của mọi ô (cùng thứ tự với hành động 0..3).
    obstacle_above = padded_obstacles[:, :-2, 1:-1]
    obstacle_below = padded_obstacles[:, 2:, 1:-1]
    obstacle_left = padded_obstacles[:, 1:-1, :-2]
    obstacle_right = padded_obstacles[:, 1:-1, 2:]
    is_blocked = torch.stack(
        [(robot_channel * obstacles_in_direction).sum(dim=(1, 2))
         for obstacles_in_direction in (obstacle_above, obstacle_below, obstacle_left, obstacle_right)],
        dim=1,
    )
    return is_blocked < 0.5
