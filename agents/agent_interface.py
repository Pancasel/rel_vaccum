"""Giao diện chung cho mọi agent (baseline và RL), theo C3 của PLAN_rel.md."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Agent(Protocol):
    """Agent nhận observation (3, H, W) và trả về hành động 0..3.

    Nếu agent có thêm phương thức `reset()` thì bộ đánh giá gọi nó đầu mỗi episode (D-24).
    """

    def act(self, observation: np.ndarray, info: dict | None = None) -> int:
        """Chọn hành động cho observation hiện tại."""
        ...
