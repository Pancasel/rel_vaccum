"""Ghép nhiều khung hình thành một ảnh (đặt cạnh nhau hoặc xếp dọc) để so sánh các agent."""

from __future__ import annotations

import numpy as np

WHITE_PIXEL_VALUE: int = 255


def pad_frame_to_size(frame: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    """Đệm ảnh bằng màu trắng (căn trái-trên) cho đủ kích thước, để ghép các ảnh khác cỡ."""
    frame_height, frame_width = frame.shape[:2]
    if frame_height > target_height or frame_width > target_width:
        raise ValueError(
            f"Ảnh {frame_height}x{frame_width} lớn hơn kích thước đích {target_height}x{target_width}."
        )
    padded_frame = np.full((target_height, target_width, 3), WHITE_PIXEL_VALUE, dtype=np.uint8)
    padded_frame[:frame_height, :frame_width] = frame
    return padded_frame


def place_frames_side_by_side(frames: list[np.ndarray], gap_pixels: int = 8) -> np.ndarray:
    """Ghép các ảnh theo hàng ngang, cách nhau một khoảng trắng (ảnh khác cỡ được đệm cho bằng)."""
    if not frames:
        raise ValueError("Cần ít nhất một ảnh để ghép.")
    tallest_height = max(frame.shape[0] for frame in frames)
    parts: list[np.ndarray] = []
    for frame_index, frame in enumerate(frames):
        if frame_index > 0:
            parts.append(np.full((tallest_height, gap_pixels, 3), WHITE_PIXEL_VALUE, dtype=np.uint8))
        parts.append(pad_frame_to_size(frame, tallest_height, frame.shape[1]))
    return np.concatenate(parts, axis=1)


def stack_frames_vertically(frames: list[np.ndarray], gap_pixels: int = 8) -> np.ndarray:
    """Xếp các ảnh theo chiều dọc (ảnh hẹp hơn được đệm trắng bên phải)."""
    if not frames:
        raise ValueError("Cần ít nhất một ảnh để ghép.")
    widest_width = max(frame.shape[1] for frame in frames)
    parts: list[np.ndarray] = []
    for frame_index, frame in enumerate(frames):
        if frame_index > 0:
            parts.append(np.full((gap_pixels, widest_width, 3), WHITE_PIXEL_VALUE, dtype=np.uint8))
        parts.append(pad_frame_to_size(frame, frame.shape[0], widest_width))
    return np.concatenate(parts, axis=0)
