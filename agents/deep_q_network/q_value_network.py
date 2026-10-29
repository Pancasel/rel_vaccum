"""Mạng nơ-ron xấp xỉ Q(s, ·) cho DQN."""

from __future__ import annotations

import torch
from torch import nn


class QValueNetwork(nn.Module):
    """CNN nhỏ: nhận observation (B, 3, H, W), trả Q của 4 hành động (B, 4).

    Vì sao CNN: observation là "ảnh" 3 kênh của căn phòng; tích chập học được các mẫu cục bộ
    (góc tường, ô bẩn cạnh robot) và dùng chung trọng số ở mọi vị trí, nên khái quát sang
    phòng chưa thấy tốt hơn mạng fully-connected thuần.
    Giữ nguyên kích thước không gian (padding=1), không pooling: lưới rất nhỏ, mất thông tin
    vị trí từng ô là mất thông tin quan trọng.
    Kích thước mặc định (16, 32) kênh, lớp ẩn 128: đủ cho lưới <= 10x10 và cập nhật nhanh gấp
    ~3 lần cấu hình (32, 64)/256 khi huấn luyện trên CPU (xem docs/CHANGELOG_DESIGN.md).

    Lưu ý: hai thuộc tính `features` và `head` là khóa trong file .pt đã lưu; đổi tên sẽ làm
    các mô hình đã huấn luyện không đọc lại được.
    """

    def __init__(self, input_height: int, input_width: int, action_count: int = 4,
                 convolution_channels: tuple[int, int] = (16, 32), hidden_size: int = 128) -> None:
        super().__init__()
        if input_height < 1 or input_width < 1:
            raise ValueError(f"Kích thước lưới phải dương, nhận được {input_height}x{input_width}.")
        self.input_height, self.input_width = input_height, input_width
        first_channels, second_channels = convolution_channels
        self.features = nn.Sequential(
            nn.Conv2d(3, first_channels, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(first_channels, second_channels, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(second_channels * input_height * input_width, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, action_count),
        )

    def forward(self, observation_batch: torch.Tensor) -> torch.Tensor:
        if observation_batch.shape[-2:] != (self.input_height, self.input_width):
            raise ValueError(
                f"Input có kích thước không gian {tuple(observation_batch.shape[-2:])}, "
                f"mạng cần {(self.input_height, self.input_width)}."
            )
        return self.head(self.features(observation_batch))


def count_trainable_parameters(model: nn.Module) -> int:
    """Tổng số tham số học được."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
