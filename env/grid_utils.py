"""Các hàm tiện ích trên lưới: BFS, giải mã observation, mã hóa trạng thái cho tabular."""

from __future__ import annotations

from collections import deque

import numpy as np

from env.maps import FREE, free_cells

# Phải trùng thứ tự hành động D-02: 0=lên, 1=xuống, 2=trái, 3=phải.
ACTION_DELTAS: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Bitmask dùng int64 nên tối đa 62 ô trống (chừa bit dấu và an toàn khi cộng).
MAX_TABULAR_FREE_CELLS: int = 62


def _in_free(grid: np.ndarray, r: int, c: int) -> bool:
    """True nếu (r, c) nằm trong lưới và là ô trống."""
    return 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and grid[r, c] == FREE


def bfs_distances(grid: np.ndarray, source: tuple[int, int]) -> np.ndarray:
    """Khoảng cách ngắn nhất (số bước 4 hướng, đi vòng vật cản) từ source tới mọi ô.

    Returns:
        Mảng (H, W) int32; -1 với vật cản hoặc ô không tới được.

    Raises:
        ValueError: nếu source không phải ô trống.
    """
    if not _in_free(grid, *source):
        raise ValueError(f"Ô nguồn {source} không phải ô trống trong lưới.")
    dist = np.full(grid.shape, -1, dtype=np.int32)
    dist[source] = 0
    queue: deque[tuple[int, int]] = deque([source])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ACTION_DELTAS:
            nr, nc = r + dr, c + dc
            if _in_free(grid, nr, nc) and dist[nr, nc] < 0:
                dist[nr, nc] = dist[r, c] + 1
                queue.append((nr, nc))
    return dist


def bfs_distance_to_nearest_uncleaned(
    grid: np.ndarray, pos: tuple[int, int], cleaned: np.ndarray
) -> int:
    """Khoảng cách BFS từ robot tới ô trống chưa dọn gần nhất; 0 nếu đã dọn hết.

    Dùng cho shaping thế năng (D-14) và reward "hackable" (D-15).
    """
    targets = (grid == FREE) & ~cleaned
    if not targets.any():
        return 0
    if targets[pos]:
        return 0
    seen = np.zeros(grid.shape, dtype=bool)
    seen[pos] = True
    queue: deque[tuple[int, int, int]] = deque([(pos[0], pos[1], 0)])
    while queue:
        r, c, d = queue.popleft()
        for dr, dc in ACTION_DELTAS:
            nr, nc = r + dr, c + dc
            if _in_free(grid, nr, nc) and not seen[nr, nc]:
                if targets[nr, nc]:
                    return d + 1
                seen[nr, nc] = True
                queue.append((nr, nc, d + 1))
    # Chỉ xảy ra nếu bản đồ không liên thông (VacuumEnv đã chặn trường hợp này).
    raise ValueError("Không tới được ô chưa dọn nào: bản đồ không liên thông.")


def bfs_first_action_towards(
    grid: np.ndarray, pos: tuple[int, int], targets: np.ndarray
) -> int | None:
    """Hành động đầu tiên trên một đường ngắn nhất từ pos tới ô mục tiêu gần nhất.

    Bản thân ô pos không được tính là mục tiêu. Phá hòa (D-20): BFS mở rộng hàng xóm theo
    thứ tự hành động 0,1,2,3 nên kết quả luôn tất định.

    Args:
        grid: bản đồ.
        pos: vị trí hiện tại.
        targets: mảng bool (H, W), True ở các ô mục tiêu.

    Returns:
        Chỉ số hành động 0..3, hoặc None nếu không tới được mục tiêu nào.
    """
    seen = np.zeros(grid.shape, dtype=bool)
    seen[pos] = True
    queue: deque[tuple[int, int, int]] = deque()
    for action, (dr, dc) in enumerate(ACTION_DELTAS):
        nr, nc = pos[0] + dr, pos[1] + dc
        if _in_free(grid, nr, nc) and not seen[nr, nc]:
            if targets[nr, nc]:
                return action
            seen[nr, nc] = True
            queue.append((nr, nc, action))
    while queue:
        r, c, first_action = queue.popleft()
        for dr, dc in ACTION_DELTAS:
            nr, nc = r + dr, c + dc
            if _in_free(grid, nr, nc) and not seen[nr, nc]:
                if targets[nr, nc]:
                    return first_action
                seen[nr, nc] = True
                queue.append((nr, nc, first_action))
    return None


def decode_obs(obs: np.ndarray) -> tuple[tuple[int, int], np.ndarray, np.ndarray]:
    """Giải mã observation (3, H, W) thành (vị trí robot, mảng đã dọn, mảng vật cản).

    Raises:
        ValueError: nếu shape sai hoặc kênh 0 không có đúng một ô bằng 1.
    """
    if obs.ndim != 3 or obs.shape[0] != 3:
        raise ValueError(f"Observation phải có shape (3, H, W), nhận được {obs.shape}.")
    robot = obs[0] > 0.5
    if int(robot.sum()) != 1:
        raise ValueError(f"Kênh 0 phải có đúng một ô robot, đang có {int(robot.sum())}.")
    flat = int(np.argmax(robot))
    pos = divmod(flat, obs.shape[2])
    cleaned = obs[1] > 0.5
    obstacles = (obs[2] > 0.5).astype(np.int8)
    return (int(pos[0]), int(pos[1])), cleaned, obstacles


def valid_actions_for(grid: np.ndarray, pos: tuple[int, int]) -> list[int]:
    """Các hành động không đụng tường/vật cản khi đứng ở pos."""
    return [a for a, (dr, dc) in enumerate(ACTION_DELTAS) if _in_free(grid, pos[0] + dr, pos[1] + dc)]


def valid_actions(obs: np.ndarray) -> list[int]:
    """Các hành động không đụng tường/vật cản, tính trực tiếp từ observation."""
    pos, _, obstacles = decode_obs(obs)
    return valid_actions_for(obstacles, pos)


class StateEncoder:
    """Mã hóa trạng thái thành khóa hashable cho bảng Q (D-12, D-13).

    - state_mode="full": khóa (row, col, mask), mask là bitmask các ô trống đã dọn,
      bit i ứng với ô trống thứ i theo thứ tự `free_cells` (row-major).
    - state_mode="position_only": khóa (row, col), cố ý bỏ thông tin đã dọn cho thí nghiệm E1.
    """

    STATE_MODES: tuple[str, ...] = ("full", "position_only")

    def __init__(self, grid: np.ndarray, state_mode: str = "full") -> None:
        """
        Args:
            grid: bản đồ (H, W).
            state_mode: "full" hoặc "position_only".

        Raises:
            ValueError: state_mode lạ, hoặc quá nhiều ô trống cho bitmask.
        """
        if state_mode not in self.STATE_MODES:
            raise ValueError(f"state_mode phải là một trong {self.STATE_MODES}, nhận được '{state_mode}'.")
        cells = free_cells(grid)
        if state_mode == "full" and len(cells) > MAX_TABULAR_FREE_CELLS:
            raise ValueError(
                f"Bản đồ có {len(cells)} ô trống, vượt giới hạn {MAX_TABULAR_FREE_CELLS} của bitmask tabular."
            )
        self.grid: np.ndarray = grid.copy()
        self.state_mode: str = state_mode
        self.free_cells: list[tuple[int, int]] = cells
        self.free_index: dict[tuple[int, int], int] = {cell: i for i, cell in enumerate(cells)}
        self.n_free: int = len(cells)
        self.full_mask: int = (1 << self.n_free) - 1
        width = grid.shape[1]
        self._free_flat_idx = np.array([r * width + c for r, c in cells], dtype=np.int64)
        self._weights = np.left_shift(np.int64(1), np.arange(self.n_free, dtype=np.int64))

    def mask_of(self, cleaned: np.ndarray) -> int:
        """Bitmask của mảng đã dọn (H, W) bool."""
        bits = cleaned.ravel()[self._free_flat_idx].astype(np.int64)
        return int((bits * self._weights).sum())

    def encode(self, pos: tuple[int, int], cleaned: np.ndarray) -> tuple:
        """Khóa trạng thái từ vị trí và mảng đã dọn."""
        if self.state_mode == "position_only":
            return (int(pos[0]), int(pos[1]))
        return (int(pos[0]), int(pos[1]), self.mask_of(cleaned))

    def encode_obs(self, obs: np.ndarray) -> tuple:
        """Khóa trạng thái trực tiếp từ observation (3, H, W)."""
        pos, cleaned, _ = decode_obs(obs)
        return self.encode(pos, cleaned)

    def decode_mask(self, mask: int) -> np.ndarray:
        """Ngược với mask_of: bitmask -> mảng đã dọn (H, W) bool."""
        if not 0 <= mask <= self.full_mask:
            raise ValueError(f"mask {mask} ngoài khoảng [0, {self.full_mask}].")
        cleaned = np.zeros(self.grid.shape, dtype=bool)
        for i, (r, c) in enumerate(self.free_cells):
            if (mask >> i) & 1:
                cleaned[r, c] = True
        return cleaned
