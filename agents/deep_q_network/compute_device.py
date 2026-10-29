"""Chọn thiết bị tính toán (CPU hoặc GPU) cho torch."""

from __future__ import annotations

import torch


def choose_compute_device(device_name: str) -> torch.device:
    """"auto" -> cuda nếu máy có GPU, ngược lại cpu. Báo lỗi rõ ràng nếu đòi "cuda" mà không có GPU."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(
            "Yêu cầu device 'cuda' nhưng torch không thấy GPU (cài bản torch CUDA hoặc dùng 'cpu')."
        )
    if device_name != "cpu" and not device_name.startswith("cuda"):
        raise ValueError(f"device phải là 'auto', 'cpu' hoặc 'cuda', nhận được '{device_name}'.")
    return torch.device(device_name)
