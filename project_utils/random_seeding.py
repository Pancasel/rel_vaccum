"""Đặt seed cho các nguồn ngẫu nhiên để thí nghiệm chạy lại ra đúng kết quả."""

from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Đặt seed cho `random`, numpy (trạng thái toàn cục) và torch (nếu có).

    Code của dự án luôn dùng bộ sinh số riêng (`np.random.default_rng(seed)`, `env.np_random`),
    nhưng vẫn đặt seed toàn cục để thư viện bên thứ ba (ví dụ khởi tạo trọng số của torch)
    cũng tái lập được.

    Args:
        seed: số nguyên không âm.

    Raises:
        ValueError: nếu seed âm.
    """
    if seed < 0:
        raise ValueError(f"seed phải >= 0, nhận được {seed}")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
