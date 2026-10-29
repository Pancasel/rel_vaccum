"""Quan sát lấy robot làm tâm (egocentric = "nhìn từ góc của chính robot")."""

from __future__ import annotations

import numpy as np

from environment.observation_decoding import decode_observation


def egocentric_observation_shape(height: int, width: int) -> tuple[int, int, int]:
    """Shape của quan sát lấy robot làm tâm cho lưới H x W: (3, 2H-1, 2W-1)."""
    return (3, 2 * height - 1, 2 * width - 1)


def to_egocentric_observation(observation: np.ndarray) -> np.ndarray:
    """Dịch observation (3, H, W) sao cho robot luôn nằm ở tâm ảnh (3, 2H-1, 2W-1).

    Vì sao: với quan sát gốc, mạng phải học lại cùng một quy tắc ("ô bẩn ngay bên trái thì đi
    trái") cho từng vị trí trên bản đồ. Khi robot luôn ở tâm, quy tắc đó là một và dùng chung
    cho mọi vị trí, mọi căn phòng — điều kiện để khái quát sang phòng chưa gặp.
    Vùng ngoài bản đồ được đánh dấu là vật cản (kênh 2 = 1), nên robot "thấy" tường bao quanh.
    Ảnh đủ lớn (2H-1) để từ tâm vẫn thấy toàn bộ căn phòng ở mọi vị trí, không mất thông tin.
    Kênh 0 vẫn là one-hot vị trí robot (luôn ở tâm) để mọi hàm đọc observation dùng lại được.
    """
    _, height, width = observation.shape
    (robot_row, robot_column), _, _ = decode_observation(observation)
    centered_view = np.zeros(egocentric_observation_shape(height, width), dtype=np.float32)
    centered_view[2] = 1.0  # mặc định mọi ô ngoài bản đồ là vật cản
    top_offset = height - 1 - robot_row
    left_offset = width - 1 - robot_column
    centered_view[:, top_offset:top_offset + height, left_offset:left_offset + width] = observation
    return centered_view
