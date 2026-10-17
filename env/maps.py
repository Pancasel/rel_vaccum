"""Bản đồ căn phòng: bản đồ cố định, đọc từ chuỗi, sinh ngẫu nhiên có kiểm tra liên thông.

Quy ước (D-01): mảng numpy shape (H, W), dtype int8, 0 = sàn trống, 1 = vật cản.
Bản đồ cố định viết bằng chuỗi cho dễ đọc: '.' là sàn trống, '#' là vật cản.
"""

from __future__ import annotations

from collections import deque

import numpy as np

FREE: int = 0
OBSTACLE: int = 1

# Bốn hướng lân cận dùng cho kiểm tra liên thông (thứ tự giống hành động: lên, xuống, trái, phải).
_NEIGHBOURS: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))

FIXED_MAPS: dict[str, list[str]] = {
    # Lưới 3x3 trống: |F| = 9. Dùng kiểm chứng Value Iteration và test tabular (chạy nhanh).
    "empty_3x3": [
        "...",
        "...",
        "...",
    ],
    # Lưới 4x4 trống: |F| = 16. Dùng cho E7 (bùng nổ trạng thái) và VI cỡ trung bình.
    "empty_4x4": [
        "....",
        "....",
        "....",
        "....",
    ],
    # Lưới 4x4 có 3 vật cản: |F| = 13. Bản đồ chính cho tabular (E1, E2, E6).
    "obstacle_4x4": [
        "....",
        ".#..",
        "...#",
        "#...",
    ],
    # Lưới 5x5 và 6x6 trống: chỉ dùng ở nhiệm vụ 5.3 (nơi bảng Q bắt đầu quá tải).
    "empty_5x5": ["....."] * 5,
    "empty_6x6": ["......"] * 6,
    # Phòng 6x6 có đồ đạc: |F| = 29. Bản đồ trung gian cho DQN.
    "room_6x6": [
        "......",
        ".##...",
        ".##.#.",
        "....#.",
        "#.....",
        "......",
    ],
    # Phòng 8x8 có đồ đạc (bàn 2x2, tủ, chân ghế): |F| = 52. Bản đồ chính cho deep RL.
    "furniture_8x8": [
        "........",
        ".##.....",
        ".##..##.",
        "........",
        "...#....",
        ".#.#..#.",
        ".#....#.",
        "........",
    ],
}

_CHAR_TO_CELL: dict[str, int] = {".": FREE, "#": OBSTACLE}


def parse_map(lines: list[str]) -> np.ndarray:
    """Chuyển danh sách chuỗi thành mảng bản đồ.

    Args:
        lines: mỗi phần tử là một hàng, ký tự '.' (trống) hoặc '#' (vật cản).

    Returns:
        Mảng (H, W) dtype int8.

    Raises:
        ValueError: nếu rỗng, các hàng khác độ dài, có ký tự lạ, hoặc không có ô trống nào.
    """
    if not lines:
        raise ValueError("Bản đồ rỗng: cần ít nhất một hàng.")
    width = len(lines[0])
    if width == 0:
        raise ValueError("Bản đồ có hàng rỗng.")
    rows: list[list[int]] = []
    for r, line in enumerate(lines):
        if len(line) != width:
            raise ValueError(
                f"Hàng {r} dài {len(line)} ký tự, khác hàng 0 ({width} ký tự). Bản đồ phải là hình chữ nhật."
            )
        try:
            rows.append([_CHAR_TO_CELL[ch] for ch in line])
        except KeyError as exc:
            raise ValueError(
                f"Ký tự không hợp lệ {exc.args[0]!r} ở hàng {r}; chỉ cho phép '.' (trống) và '#' (vật cản)."
            ) from None
    grid = np.array(rows, dtype=np.int8)
    if not (grid == FREE).any():
        raise ValueError("Bản đồ không có ô trống nào.")
    return grid


def get_map(name: str) -> np.ndarray:
    """Trả bản sao của một bản đồ cố định theo tên.

    Raises:
        KeyError: nếu tên không tồn tại (thông báo kèm danh sách tên hợp lệ).
    """
    if name not in FIXED_MAPS:
        raise KeyError(f"Không có bản đồ '{name}'. Các bản đồ hợp lệ: {sorted(FIXED_MAPS)}")
    # parse_map luôn tạo mảng mới nên người gọi sửa thoải mái mà không ảnh hưởng bản gốc.
    return parse_map(FIXED_MAPS[name])


def free_cells(grid: np.ndarray) -> list[tuple[int, int]]:
    """Danh sách ô trống theo thứ tự hàng (row-major).

    Thứ tự này định nghĩa chỉ số bit trong bitmask trạng thái tabular (D-12), không được đổi.
    """
    rows, cols = np.nonzero(grid == FREE)
    return [(int(r), int(c)) for r, c in zip(rows, cols)]


def is_connected(grid: np.ndarray) -> bool:
    """True nếu mọi ô trống liên thông với nhau theo 4 hướng.

    Vì sao cần: nếu có ô trống bị vật cản bao kín, robot không bao giờ dọn được 100%,
    episode không thể kết thúc bằng `terminated` và bài toán trở nên vô nghĩa.
    """
    cells = free_cells(grid)
    if not cells:
        return False
    height, width = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    queue: deque[tuple[int, int]] = deque([cells[0]])
    seen[cells[0]] = True
    n_seen = 1
    while queue:
        r, c = queue.popleft()
        for dr, dc in _NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and not seen[nr, nc] and grid[nr, nc] == FREE:
                seen[nr, nc] = True
                n_seen += 1
                queue.append((nr, nc))
    return n_seen == len(cells)


def generate_random_map(
    height: int,
    width: int,
    obstacle_density: float,
    rng: np.random.Generator,
    max_tries: int = 1000,
) -> np.ndarray:
    """Sinh bản đồ ngẫu nhiên có đúng round(density*H*W) vật cản và luôn liên thông.

    Thuật toán: duyệt các ô theo một hoán vị ngẫu nhiên, đặt thử vật cản và chỉ giữ lại
    nếu các ô trống vẫn liên thông. Cách này đảm bảo bản đồ luôn dọn được hết.

    Args:
        height, width: kích thước lưới (>= 1, tổng ô >= 2).
        obstacle_density: tỷ lệ ô là vật cản, trong [0, 1).
        rng: bộ sinh số ngẫu nhiên numpy (để tái lập theo seed).
        max_tries: số lần thử lại với hoán vị mới trước khi báo lỗi.

    Raises:
        ValueError: tham số không hợp lệ.
        RuntimeError: không sinh được bản đồ liên thông sau max_tries lần.
    """
    if height < 1 or width < 1:
        raise ValueError(f"Kích thước lưới phải dương, nhận được {height}x{width}.")
    if not 0.0 <= obstacle_density < 1.0:
        raise ValueError(f"obstacle_density phải trong [0, 1), nhận được {obstacle_density}.")
    n_cells = height * width
    n_obstacles = int(round(obstacle_density * n_cells))
    if n_obstacles > n_cells - 2:
        raise ValueError(
            f"Quá nhiều vật cản ({n_obstacles}) cho lưới {height}x{width}: cần còn ít nhất 2 ô trống."
        )
    for _ in range(max_tries):
        grid = np.zeros((height, width), dtype=np.int8)
        placed = 0
        for flat in rng.permutation(n_cells):
            if placed == n_obstacles:
                break
            r, c = divmod(int(flat), width)
            grid[r, c] = OBSTACLE
            if is_connected(grid):
                placed += 1
            else:
                grid[r, c] = FREE
        if placed == n_obstacles:
            return grid
    raise RuntimeError(
        f"Không sinh được bản đồ liên thông {height}x{width} với {n_obstacles} vật cản sau {max_tries} lần thử."
    )


def generate_map_set(
    height: int,
    width: int,
    obstacle_density: float,
    n_maps: int,
    seed: int,
    exclude: list[np.ndarray] | None = None,
    max_attempts_factor: int = 100,
) -> list[np.ndarray]:
    """Sinh n_maps bản đồ đôi một khác nhau từ một seed (dùng làm tập train/test cho E3, E4).

    Args:
        exclude: các bản đồ không được trùng (ví dụ tập test khi sinh tập train).
        max_attempts_factor: giới hạn số lần sinh = n_maps * hệ số này.

    Raises:
        ValueError: n_maps < 1.
        RuntimeError: không đủ bản đồ khác nhau (ví dụ mật độ 0 chỉ có đúng 1 bản đồ).
    """
    if n_maps < 1:
        raise ValueError(f"n_maps phải >= 1, nhận được {n_maps}.")
    rng = np.random.default_rng(seed)
    seen: set[bytes] = {m.astype(np.int8).tobytes() for m in (exclude or [])}
    maps: list[np.ndarray] = []
    for _ in range(n_maps * max_attempts_factor):
        grid = generate_random_map(height, width, obstacle_density, rng)
        key = grid.tobytes()
        if key not in seen:
            seen.add(key)
            maps.append(grid)
            if len(maps) == n_maps:
                return maps
    raise RuntimeError(
        f"Chỉ sinh được {len(maps)}/{n_maps} bản đồ khác nhau ({height}x{width}, mật độ {obstacle_density})."
    )


def map_to_string(grid: np.ndarray) -> str:
    """Chuỗi nhiều dòng biểu diễn bản đồ ('.' trống, '#' vật cản) để in ra màn hình."""
    return "\n".join("".join("#" if v == OBSTACLE else "." for v in row) for row in grid)
