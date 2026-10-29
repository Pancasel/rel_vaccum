"""Biến thể môi trường sinh căn phòng mới mỗi lần reset (D-17), dùng cho E3, E4 và demo phòng lạ."""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np

from environment.random_room_generator import generate_random_room
from environment.reward_settings import RewardSettings
from environment.vacuum_cleaning_env import VacuumCleaningEnv, convert_to_grid


class RandomRoomVacuumCleaningEnv(VacuumCleaningEnv):
    """Mỗi lần reset là một căn phòng mới; ô xuất phát luôn ngẫu nhiên.

    Kích thước H, W cố định nên observation_space không đổi.
    """

    def __init__(self, height: int, width: int, obstacle_density: float,
                 room_pool: list[np.ndarray] | None = None, excluded_rooms: list[np.ndarray] | None = None,
                 reward_settings: RewardSettings | None = None, max_steps_factor: int = 4,
                 render_mode: str | None = None) -> None:
        """
        Args:
            height, width: kích thước lưới.
            obstacle_density: mật độ vật cản khi sinh ngẫu nhiên.
            room_pool: nếu có, mỗi episode chọn ngẫu nhiên một bản đồ trong danh sách này
                (dùng cho E4: huấn luyện trên K bản đồ cố định).
            excluded_rooms: bản đồ không được sinh ra khi train (ví dụ tập test), chỉ áp dụng
                khi không có room_pool.
            max_steps_factor: max_steps = hệ số * số ô trống của bản đồ hiện tại.

        Raises:
            ValueError: tham số không hợp lệ hoặc bản đồ trong pool sai kích thước.
        """
        if max_steps_factor <= 0:
            raise ValueError(f"max_steps_factor phải dương, nhận được {max_steps_factor}.")
        if room_pool is not None:
            if not room_pool:
                raise ValueError("map_pool rỗng.")
            for room in room_pool:
                if room.shape != (height, width):
                    raise ValueError(f"Bản đồ trong pool có shape {room.shape}, cần {(height, width)}.")
        self.obstacle_density = obstacle_density
        self.max_steps_factor = max_steps_factor
        self._room_pool = [room.astype(np.int8, copy=True) for room in room_pool] if room_pool else None
        self._excluded_room_bytes = {room.astype(np.int8).tobytes() for room in (excluded_rooms or [])}
        if self._room_pool:
            initial_room = self._room_pool[0]
        else:
            initial_room = generate_random_room(height, width, obstacle_density, np.random.default_rng(0))
        super().__init__(initial_room, reward_settings=reward_settings, max_steps=None, start=None,
                         render_mode=render_mode)

    def _sample_room(self) -> np.ndarray:
        """Chọn bản đồ cho episode mới bằng self.np_random (tái lập theo seed của reset)."""
        if self._room_pool is not None:
            return self._room_pool[int(self.np_random.integers(len(self._room_pool)))]
        for _ in range(1000):
            room = generate_random_room(self.height, self.width, self.obstacle_density, self.np_random)
            if room.tobytes() not in self._excluded_room_bytes:
                return room
        raise RuntimeError("Không sinh được bản đồ ngoài danh sách loại trừ sau 1000 lần thử.")

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None
              ) -> tuple[np.ndarray, dict[str, Any]]:
        """Sinh (hoặc nhận qua options["grid"]) bản đồ mới rồi reset như VacuumCleaningEnv."""
        gymnasium.Env.reset(self, seed=seed)
        options = dict(options or {})
        if options.get("grid") is not None:
            room = convert_to_grid(options.pop("grid"))
            if room.shape != (self.height, self.width):
                raise ValueError(
                    f"Bản đồ truyền vào có shape {room.shape}, cần {(self.height, self.width)}."
                )
        else:
            room = self._sample_room()
        self._set_grid(room.copy())
        self.max_steps = self.max_steps_factor * self._free_cell_count
        # seed đã được dùng ở trên; gọi lại với seed=None để không reset np_random lần nữa.
        return super().reset(seed=None, options=options)
