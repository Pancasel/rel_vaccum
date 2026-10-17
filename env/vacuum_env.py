"""Môi trường robot hút bụi dạng lưới theo chuẩn Gymnasium.

MDP (PLAN_rel.md mục B2):
- State: vị trí robot + tập ô đã dọn + bản đồ vật cản (đủ thông tin để thỏa tính Markov).
- Action: 0=lên, 1=xuống, 2=trái, 3=phải.
- Transition: xác định (deterministic).
- Reward: cấu hình bằng RewardConfig (mặc định "dense").
- terminated: dọn hết ô trống. truncated: hết số bước cho phép (max_steps).

Hàm thuần `transition()` là nguồn sự thật duy nhất về động lực học và phần thưởng (D-11):
cả `VacuumEnv.step` lẫn Value Iteration đều gọi hàm này, nên hai bên không thể lệch nhau.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces

from env.grid_utils import ACTION_DELTAS, bfs_distance_to_nearest_uncleaned, valid_actions_for
from env.maps import FREE, OBSTACLE, free_cells, generate_random_map, get_map, is_connected, parse_map

ACTION_NAMES: tuple[str, ...] = ("up", "down", "left", "right")

# Số vị trí gần nhất lưu lại để vẽ vệt đường đi khi render.
TRAIL_LENGTH: int = 30


@dataclass(frozen=True)
class RewardConfig:
    """Tham số phần thưởng. frozen=True để cấu hình không bị sửa giữa chừng khi chạy ablation."""

    new_cell: float = 1.0  # vào ô chưa dọn
    revisit: float = -0.1  # vào ô đã dọn (đi lãng phí)
    collision: float = -0.5  # đụng tường/vật cản (robot đứng yên)
    step: float = -0.01  # mỗi bước thời gian (khuyến khích dọn nhanh)
    completion: float = 10.0  # thưởng thêm khi dọn xong 100%
    shaping_coef: float = 0.0  # hệ số shaping thế năng (D-14), 0 = tắt
    naive_progress_bonus: float = 0.0  # reward "hackable" để minh họa (D-15), 0 = tắt
    gamma: float = 0.99  # chỉ dùng trong công thức shaping thế năng

    PRESETS = ("dense", "sparse", "shaped", "hackable")

    @classmethod
    def dense(cls) -> "RewardConfig":
        """Cấu hình v1 trong B2: thưởng ô mới, phạt đi lại, phạt va chạm, phạt thời gian."""
        return cls()

    @classmethod
    def sparse(cls) -> "RewardConfig":
        """E2a: chỉ thưởng khi dọn xong, mọi sự kiện khác bằng 0."""
        return cls(new_cell=0.0, revisit=0.0, collision=0.0, step=0.0, completion=10.0)

    @classmethod
    def shaped(cls, coef: float = 1.0) -> "RewardConfig":
        """E2c: dense + shaping thế năng theo khoảng cách tới ô chưa dọn gần nhất."""
        return cls(shaping_coef=coef)

    @classmethod
    def hackable(cls, bonus: float = 0.5) -> "RewardConfig":
        """Reward thiết kế kém (D-15): thưởng khi lại gần ô bẩn nhưng không phạt khi đi xa.

        Không phạt đi lại, không phạt thời gian, nên agent có thể "đi ra rồi đi vào" mãi để
        ăn điểm thay vì dọn xong. Dùng để minh họa reward hacking trong E2.
        """
        return cls(new_cell=1.0, revisit=0.0, collision=0.0, step=0.0, completion=10.0,
                   naive_progress_bonus=bonus)

    @classmethod
    def from_name(cls, name: str, **overrides: float) -> "RewardConfig":
        """Tạo cấu hình theo tên preset, có thể ghi đè từng trường.

        Raises:
            ValueError: tên preset không hợp lệ.
        """
        factories = {"dense": cls.dense, "sparse": cls.sparse, "shaped": cls.shaped, "hackable": cls.hackable}
        if name not in factories:
            raise ValueError(f"Reward preset '{name}' không hợp lệ; chọn một trong {list(factories)}.")
        return replace(factories[name](), **overrides)


@dataclass(frozen=True)
class TransitionResult:
    """Kết quả một bước của mô hình MDP."""

    next_pos: tuple[int, int]
    next_cleaned: np.ndarray  # mảng MỚI (H, W) bool, không dùng chung bộ nhớ với đầu vào
    reward: float
    terminated: bool
    event: str  # "new_cell" | "revisit" | "collision"


def _potential_distance(grid: np.ndarray, pos: tuple[int, int], cleaned: np.ndarray,
                        terminated: bool) -> int:
    """d(s) dùng cho shaping: 0 ở trạng thái kết thúc (quy ước Φ(terminal) = 0)."""
    return 0 if terminated else bfs_distance_to_nearest_uncleaned(grid, pos, cleaned)


def transition(grid: np.ndarray, pos: tuple[int, int], cleaned: np.ndarray, action: int,
               cfg: RewardConfig) -> TransitionResult:
    """Mô hình MDP xác định p(s', r | s, a) (D-11).

    Hàm thuần: không đọc/ghi trạng thái toàn cục, không dùng số ngẫu nhiên, không sửa `cleaned`.

    Args:
        grid: bản đồ (H, W), 0 trống, 1 vật cản.
        pos: vị trí robot (hàng, cột).
        cleaned: mảng bool (H, W) các ô đã dọn.
        action: 0..3.
        cfg: cấu hình phần thưởng.

    Raises:
        ValueError: hành động ngoài 0..3.
    """
    if not 0 <= action < len(ACTION_DELTAS):
        raise ValueError(f"Hành động phải trong 0..3, nhận được {action}.")
    height, width = grid.shape
    dr, dc = ACTION_DELTAS[action]
    nr, nc = pos[0] + dr, pos[1] + dc
    if not (0 <= nr < height and 0 <= nc < width) or grid[nr, nc] == OBSTACLE:
        # D-04: đụng tường thì đứng yên nhưng vẫn mất một bước.
        next_pos, event, event_reward = pos, "collision", cfg.collision
    elif cleaned[nr, nc]:
        next_pos, event, event_reward = (nr, nc), "revisit", cfg.revisit
    else:
        next_pos, event, event_reward = (nr, nc), "new_cell", cfg.new_cell

    next_cleaned = cleaned.copy()
    next_cleaned[next_pos] = True
    # Đã dọn hết khi mọi ô trống đều True (ô vật cản được coi như "xong").
    terminated = bool((next_cleaned | (grid == OBSTACLE)).all())

    # D-03: reward = bước thời gian + đúng một sự kiện + bonus kết thúc (+ các thành phần tùy chọn).
    reward = cfg.step + event_reward + (cfg.completion if terminated else 0.0)

    if cfg.shaping_coef != 0.0 or cfg.naive_progress_bonus != 0.0:
        d_s = _potential_distance(grid, pos, cleaned, terminated=False)
        d_next = _potential_distance(grid, next_pos, next_cleaned, terminated)
        if cfg.shaping_coef != 0.0:
            # D-14: F = coef * (γΦ(s') − Φ(s)), Φ(s) = −d(s)/(H+W). Ng, Harada & Russell (1999)
            # chứng minh shaping dạng này không làm đổi chính sách tối ưu.
            phi_s = -d_s / (height + width)
            phi_next = -d_next / (height + width)
            reward += cfg.shaping_coef * (cfg.gamma * phi_next - phi_s)
        if cfg.naive_progress_bonus != 0.0 and d_next < d_s:
            # D-15: chỉ thưởng khi lại gần, không phạt khi đi xa -> có thể bị lợi dụng.
            reward += cfg.naive_progress_bonus

    return TransitionResult(next_pos=next_pos, next_cleaned=next_cleaned, reward=float(reward),
                            terminated=terminated, event=event)


def _to_grid(grid: np.ndarray | list[str] | str) -> np.ndarray:
    """Chấp nhận mảng, danh sách chuỗi hoặc tên bản đồ cố định; luôn trả bản sao int8."""
    if isinstance(grid, str):
        return get_map(grid)
    if isinstance(grid, np.ndarray):
        if grid.ndim != 2:
            raise ValueError(f"Bản đồ phải là mảng 2 chiều, nhận được shape {grid.shape}.")
        if not np.isin(grid, (FREE, OBSTACLE)).all():
            raise ValueError("Bản đồ chỉ được chứa giá trị 0 (trống) và 1 (vật cản).")
        return grid.astype(np.int8, copy=True)
    return parse_map(list(grid))


class VacuumEnv(gymnasium.Env):
    """Robot hút bụi trên lưới 2D (giao diện C3 của PLAN_rel.md).

    Observation: tensor (3, H, W) float32 — kênh 0 vị trí robot (one-hot), kênh 1 ô đã dọn,
    kênh 2 vật cản.
    """

    metadata = {"render_modes": ["rgb_array", "ansi"], "render_fps": 4}

    def __init__(self, grid: np.ndarray | list[str] | str, reward_config: RewardConfig | None = None,
                 max_steps: int | None = None, start: tuple[int, int] | None = None,
                 render_mode: str | None = None) -> None:
        """
        Args:
            grid: mảng (H, W), danh sách chuỗi, hoặc tên bản đồ cố định (ví dụ "obstacle_4x4").
            reward_config: cấu hình phần thưởng; None = dense (B2).
            max_steps: số bước tối đa mỗi episode; None = 4 * |F| (D-07).
            start: ô xuất phát cố định; None = ngẫu nhiên đều trên ô trống (D-08).
            render_mode: None, "rgb_array" hoặc "ansi".

        Raises:
            ValueError: bản đồ không liên thông, ít hơn 2 ô trống, start không hợp lệ,
                max_steps không dương, hoặc render_mode lạ.
        """
        super().__init__()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode '{render_mode}' không hỗ trợ; chọn {self.metadata['render_modes']} hoặc None.")
        self.render_mode = render_mode
        self.reward_config: RewardConfig = reward_config or RewardConfig.dense()
        self._max_steps_arg = max_steps
        self._start: tuple[int, int] | None = None
        self._set_grid(_to_grid(grid))
        if start is not None:
            self._start = self._validate_start(start)
        height, width = self._grid.shape
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3, height, width), dtype=np.float32)
        self._ready = False

    # ------------------------------------------------------------------ thiết lập bản đồ
    def _set_grid(self, grid: np.ndarray) -> None:
        """Gán bản đồ mới (kiểm tra hợp lệ) và tính lại các đại lượng phụ thuộc."""
        n_free = int((grid == FREE).sum())
        if n_free < 2:
            raise ValueError(f"Bản đồ cần ít nhất 2 ô trống, đang có {n_free} (D-06).")
        if not is_connected(grid):
            raise ValueError("Bản đồ không liên thông: có ô trống robot không thể tới, không bao giờ dọn xong được.")
        if self._max_steps_arg is not None and self._max_steps_arg <= 0:
            raise ValueError(f"max_steps phải dương, nhận được {self._max_steps_arg}.")
        self._grid = grid
        self._free_cells = free_cells(grid)
        self._n_free = n_free
        self.max_steps: int = self._max_steps_arg if self._max_steps_arg is not None else 4 * n_free

    def _validate_start(self, start: tuple[int, int]) -> tuple[int, int]:
        """Kiểm tra ô xuất phát nằm trong lưới và là ô trống."""
        r, c = int(start[0]), int(start[1])
        height, width = self._grid.shape
        if not (0 <= r < height and 0 <= c < width) or self._grid[r, c] != FREE:
            raise ValueError(f"Ô xuất phát {start} không phải ô trống của bản đồ.")
        return (r, c)

    # ------------------------------------------------------------------ thuộc tính chỉ đọc
    @property
    def grid(self) -> np.ndarray:
        """Bản sao bản đồ hiện tại."""
        return self._grid.copy()

    @property
    def height(self) -> int:
        return int(self._grid.shape[0])

    @property
    def width(self) -> int:
        return int(self._grid.shape[1])

    @property
    def n_free(self) -> int:
        """Số ô trống |F|."""
        return self._n_free

    @property
    def position(self) -> tuple[int, int]:
        self._require_ready()
        return self._pos

    @property
    def cleaned(self) -> np.ndarray:
        """Bản sao mảng ô đã dọn."""
        self._require_ready()
        return self._cleaned.copy()

    @property
    def steps(self) -> int:
        self._require_ready()
        return self._steps

    @property
    def fixed_start(self) -> tuple[int, int] | None:
        """Ô xuất phát cố định (None nếu xuất phát ngẫu nhiên)."""
        return self._start

    def _require_ready(self) -> None:
        if not self._ready:
            raise RuntimeError("Phải gọi reset() trước khi dùng môi trường.")

    # ------------------------------------------------------------------ API Gymnasium
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None
              ) -> tuple[np.ndarray, dict[str, Any]]:
        """Bắt đầu episode mới. options có thể chứa "start": (r, c) để ghi đè ô xuất phát."""
        super().reset(seed=seed)  # bắt buộc để self.np_random được seed đúng chuẩn Gymnasium
        options = options or {}
        if "start" in options and options["start"] is not None:
            start = self._validate_start(options["start"])
        elif self._start is not None:
            start = self._start
        else:
            start = self._free_cells[int(self.np_random.integers(len(self._free_cells)))]
        self._pos = start
        self._cleaned = np.zeros(self._grid.shape, dtype=bool)
        self._cleaned[start] = True  # D-05: ô xuất phát tính là đã dọn, không nhận thưởng
        self._steps = 0
        self._n_revisit = 0
        self._n_collisions = 0
        self._trail: deque[tuple[int, int]] = deque([start], maxlen=TRAIL_LENGTH)
        self._ready = True
        return self._get_obs(), self._get_info(collided=False, new_cell=False, event="reset")

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Thực hiện một hành động. Trả (obs, reward, terminated, truncated, info)."""
        self._require_ready()
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)) \
                or not 0 <= int(action) < 4:
            raise ValueError(f"Hành động phải là số nguyên trong 0..3, nhận được {action!r}.")
        result = transition(self._grid, self._pos, self._cleaned, int(action), self.reward_config)
        self._pos = result.next_pos
        self._cleaned = result.next_cleaned
        self._steps += 1
        if result.event == "revisit":
            self._n_revisit += 1
        elif result.event == "collision":
            self._n_collisions += 1
        self._trail.append(self._pos)
        terminated = result.terminated
        # D-07: không bao giờ vừa terminated vừa truncated.
        truncated = self._steps >= self.max_steps and not terminated
        info = self._get_info(collided=result.event == "collision", new_cell=result.event == "new_cell",
                              event=result.event)
        return self._get_obs(), float(result.reward), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Tạo observation MỚI mỗi lần gọi (không trả tham chiếu tới mảng nội bộ)."""
        obs = np.zeros((3, *self._grid.shape), dtype=np.float32)
        obs[0][self._pos] = 1.0
        obs[1] = self._cleaned
        obs[2] = self._grid == OBSTACLE
        return obs

    def _get_info(self, collided: bool, new_cell: bool, event: str) -> dict[str, Any]:
        n_cleaned = int(self._cleaned.sum())
        mask = np.zeros(4, dtype=np.int8)
        mask[valid_actions_for(self._grid, self._pos)] = 1
        return {
            "coverage": n_cleaned / self._n_free,
            "collided": collided,
            "new_cell": new_cell,
            "event": event,
            "steps": self._steps,
            "n_cleaned": n_cleaned,
            "n_free": self._n_free,
            "position": self._pos,
            "n_revisit": self._n_revisit,
            "n_collisions": self._n_collisions,
            "action_mask": mask,
        }

    def render(self) -> np.ndarray | str | None:
        """Vẽ trạng thái hiện tại theo render_mode ("rgb_array" -> ảnh, "ansi" -> chuỗi)."""
        if self.render_mode is None:
            return None
        self._require_ready()
        # Import muộn để môi trường không phụ thuộc Pillow/matplotlib khi không vẽ.
        from env.render import render_ansi, render_rgb

        if self.render_mode == "ansi":
            return render_ansi(self._grid, self._pos, self._cleaned)
        return render_rgb(self._grid, self._pos, self._cleaned, trail=list(self._trail))


class RandomMapVacuumEnv(VacuumEnv):
    """Biến thể sinh bản đồ mới mỗi lần reset (D-17), dùng cho E3, E4 và demo phòng lạ.

    Kích thước H, W cố định nên observation_space không đổi; ô xuất phát luôn ngẫu nhiên.
    """

    def __init__(self, height: int, width: int, obstacle_density: float,
                 map_pool: list[np.ndarray] | None = None, exclude_maps: list[np.ndarray] | None = None,
                 reward_config: RewardConfig | None = None, max_steps_factor: int = 4,
                 render_mode: str | None = None) -> None:
        """
        Args:
            height, width: kích thước lưới.
            obstacle_density: mật độ vật cản khi sinh ngẫu nhiên.
            map_pool: nếu có, mỗi episode chọn ngẫu nhiên một bản đồ trong danh sách này
                (dùng cho E4: huấn luyện trên K bản đồ cố định).
            exclude_maps: bản đồ không được sinh ra khi train (ví dụ tập test), chỉ áp dụng
                khi không có map_pool.
            max_steps_factor: max_steps = hệ số * |F| của bản đồ hiện tại.

        Raises:
            ValueError: tham số không hợp lệ hoặc bản đồ trong pool sai kích thước.
        """
        if max_steps_factor <= 0:
            raise ValueError(f"max_steps_factor phải dương, nhận được {max_steps_factor}.")
        if map_pool is not None:
            if not map_pool:
                raise ValueError("map_pool rỗng.")
            for m in map_pool:
                if m.shape != (height, width):
                    raise ValueError(f"Bản đồ trong pool có shape {m.shape}, cần {(height, width)}.")
        self.obstacle_density = obstacle_density
        self.max_steps_factor = max_steps_factor
        self._map_pool = [m.astype(np.int8, copy=True) for m in map_pool] if map_pool else None
        self._exclude = {m.astype(np.int8).tobytes() for m in (exclude_maps or [])}
        initial = self._map_pool[0] if self._map_pool else generate_random_map(
            height, width, obstacle_density, np.random.default_rng(0))
        super().__init__(initial, reward_config=reward_config, max_steps=None, start=None,
                         render_mode=render_mode)

    def _sample_grid(self) -> np.ndarray:
        """Chọn bản đồ cho episode mới bằng self.np_random (tái lập theo seed của reset)."""
        if self._map_pool is not None:
            return self._map_pool[int(self.np_random.integers(len(self._map_pool)))]
        for _ in range(1000):
            grid = generate_random_map(self.height, self.width, self.obstacle_density, self.np_random)
            if grid.tobytes() not in self._exclude:
                return grid
        raise RuntimeError("Không sinh được bản đồ ngoài danh sách loại trừ sau 1000 lần thử.")

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None
              ) -> tuple[np.ndarray, dict[str, Any]]:
        """Sinh (hoặc nhận qua options["grid"]) bản đồ mới rồi reset như VacuumEnv."""
        gymnasium.Env.reset(self, seed=seed)
        options = dict(options or {})
        if options.get("grid") is not None:
            grid = _to_grid(options.pop("grid"))
            if grid.shape != (self.height, self.width):
                raise ValueError(f"Bản đồ truyền vào có shape {grid.shape}, cần {(self.height, self.width)}.")
        else:
            grid = self._sample_grid()
        self._set_grid(grid.copy())
        self.max_steps = self.max_steps_factor * self._n_free
        # seed đã được dùng ở trên; gọi lại với seed=None để không reset np_random lần nữa.
        return super().reset(seed=None, options=options)
