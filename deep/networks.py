"""Mạng nơ-ron xấp xỉ Q(s, ·) cho DQN."""

from __future__ import annotations

import torch
from torch import nn


class QNetwork(nn.Module):
    """CNN nhỏ: nhận observation (B, 3, H, W), trả Q của 4 hành động (B, 4).

    Vì sao CNN: observation là "ảnh" 3 kênh của căn phòng; tích chập học được các mẫu cục bộ
    (góc tường, ô bẩn cạnh robot) và dùng chung trọng số ở mọi vị trí, nên khái quát sang
    phòng chưa thấy tốt hơn mạng fully-connected thuần.
    Giữ nguyên kích thước không gian (padding=1), không pooling: lưới rất nhỏ, mất thông tin
    vị trí từng ô là mất thông tin quan trọng.
    Kích thước mặc định (16, 32) kênh, lớp ẩn 128: đủ cho lưới <= 10x10 và cập nhật nhanh gấp
    ~3 lần cấu hình (32, 64)/256 khi huấn luyện trên CPU (xem CHANGELOG_DESIGN.md).
    """

    def __init__(self, height: int, width: int, n_actions: int = 4, channels: tuple[int, int] = (16, 32),
                 hidden: int = 128) -> None:
        super().__init__()
        if height < 1 or width < 1:
            raise ValueError(f"Kích thước lưới phải dương, nhận được {height}x{width}.")
        self.height, self.width = height, width
        c1, c2 = channels
        self.features = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Linear(c2 * height * width, hidden), nn.ReLU(), nn.Linear(hidden, n_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (self.height, self.width):
            raise ValueError(f"Input có kích thước không gian {tuple(x.shape[-2:])}, mạng cần {(self.height, self.width)}.")
        return self.head(self.features(x))


def count_parameters(model: nn.Module) -> int:
    """Tổng số tham số học được."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
