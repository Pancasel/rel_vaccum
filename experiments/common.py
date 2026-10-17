"""Tiện ích dùng chung cho các script thí nghiệm: tham số dòng lệnh, chạy song song, tạo env."""

from __future__ import annotations

import argparse
import functools
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable

from utils.paths import ROOT, assert_inside_root, setup_local_caches


def configure_console() -> None:
    """In được tiếng Việt trên console Windows (mặc định cp1252 sẽ báo UnicodeEncodeError)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


setup_local_caches()
configure_console()

import numpy as np  # noqa: E402

from env.maps import free_cells, get_map  # noqa: E402
from env.vacuum_env import RewardConfig, VacuumEnv  # noqa: E402

# Giới hạn số ô trống để còn tính được đáp án tối ưu chính xác bằng BFS trên đồ thị trạng thái.
MAX_FREE_FOR_OPTIMUM: int = 16


def add_common_args(parser: argparse.ArgumentParser, default_out: str) -> None:
    """Thêm các tham số chung: --seeds, --out, --quick, --workers."""
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="danh sách seed (mặc định 0..4, theo quy tắc báo cáo >= 5 seed)")
    parser.add_argument("--out", type=str, default=default_out, help="thư mục ghi kết quả (trong dự án)")
    parser.add_argument("--quick", action="store_true", help="chạy rút gọn để kiểm tra nhanh (< ~2 phút)")
    parser.add_argument("--workers", type=int, default=1, help="số tiến trình chạy song song")


def resolve_out(out: str | os.PathLike) -> Path:
    """Đường dẫn tuyệt đối của thư mục kết quả (tương đối thì tính từ ROOT); tạo nếu chưa có."""
    path = Path(out)
    if not path.is_absolute():
        path = ROOT / path
    resolved = assert_inside_root(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def parse_start(values: list[str] | None) -> tuple[int, int] | None:
    """--start random -> None; --start R C -> (R, C)."""
    if not values or values == ["random"]:
        return None
    if len(values) != 2:
        raise argparse.ArgumentTypeError("--start nhận 'random' hoặc hai số nguyên R C.")
    return (int(values[0]), int(values[1]))


def make_env_factory(grid: str | np.ndarray, start: tuple[int, int] | None = None,
                     reward: str | RewardConfig = "dense", max_steps: int | None = None,
                     render_mode: str | None = None) -> Callable[[], VacuumEnv]:
    """Hàm tạo env pickle được (functools.partial) để dùng được trong nhiều tiến trình."""
    cfg = RewardConfig.from_name(reward) if isinstance(reward, str) else reward
    return functools.partial(VacuumEnv, grid, reward_config=cfg, max_steps=max_steps, start=start,
                             render_mode=render_mode)


def _limit_threads() -> None:
    """Mỗi tiến trình con dùng 1 luồng torch: mạng nhỏ, chạy nhiều tiến trình song song hiệu quả hơn."""
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass


def _call(fn: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
    setup_local_caches()
    _limit_threads()
    return fn(**kwargs)


def run_parallel(fn: Callable[..., Any], tasks: Iterable[dict[str, Any]], workers: int = 1) -> list[Any]:
    """Chạy fn(**task) cho từng task, tuần tự hoặc song song bằng nhiều tiến trình.

    fn phải là hàm cấp module (pickle được). Thứ tự kết quả giữ đúng thứ tự tasks.
    """
    task_list = list(tasks)
    if workers <= 1 or len(task_list) <= 1:
        return [fn(**t) for t in task_list]
    with ProcessPoolExecutor(max_workers=min(workers, len(task_list))) as pool:
        futures = [pool.submit(_call, fn, t) for t in task_list]
        return [f.result() for f in futures]


def grid_of(map_name: str) -> np.ndarray:
    """Bản đồ cố định theo tên (bọc lại để thông báo lỗi rõ ràng trong CLI)."""
    return get_map(map_name)


def optimum_available(grid: np.ndarray) -> bool:
    """True nếu bản đồ đủ nhỏ để tính đáp án tối ưu chính xác."""
    return len(free_cells(grid)) <= MAX_FREE_FOR_OPTIMUM
