"""Tiện ích dùng chung cho các script thí nghiệm: tham số dòng lệnh, chạy song song, tạo env."""

from __future__ import annotations

import argparse
import functools
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable

from project_utils.project_paths import PROJECT_ROOT, require_path_inside_project, setup_local_caches


def configure_console() -> None:
    """In được tiếng Việt trên console Windows (mặc định cp1252 sẽ báo UnicodeEncodeError)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


setup_local_caches()
configure_console()

import numpy as np  # noqa: E402

from environment.room_layouts import list_free_cells  # noqa: E402
from environment.reward_settings import RewardSettings  # noqa: E402
from environment.vacuum_cleaning_env import VacuumCleaningEnv  # noqa: E402

def add_common_args(parser: argparse.ArgumentParser, default_out: str) -> None:
    """Thêm các tham số chung: --seeds, --out, --quick, --workers."""
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="danh sách seed (mặc định 0..4, theo quy tắc báo cáo >= 5 seed)")
    parser.add_argument("--out", type=str, default=default_out, help="thư mục ghi kết quả (trong dự án)")
    parser.add_argument("--quick", action="store_true", help="chạy rút gọn để kiểm tra nhanh (< ~2 phút)")
    parser.add_argument("--workers", type=int, default=1, help="số tiến trình chạy song song")


def resolve_out(out: str | os.PathLike) -> Path:
    """Đường dẫn tuyệt đối của thư mục kết quả (tương đối thì tính từ PROJECT_ROOT); tạo nếu chưa có."""
    path = Path(out)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    resolved = require_path_inside_project(path)
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
                     reward: str | RewardSettings = "dense", max_steps: int | None = None,
                     render_mode: str | None = None) -> Callable[[], VacuumCleaningEnv]:
    """Hàm tạo env pickle được (functools.partial) để dùng được trong nhiều tiến trình."""
    cfg = RewardSettings.from_name(reward) if isinstance(reward, str) else reward
    return functools.partial(VacuumCleaningEnv, grid, reward_settings=cfg, max_steps=max_steps, start=start,
                             render_mode=render_mode)


_SINGLE_THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS")


def _limit_threads() -> None:
    """Tiến trình con dùng 1 luồng torch: mạng nhỏ, chạy nhiều tiến trình song song hiệu quả hơn.

    Chỉ chạm vào torch nếu tiến trình ĐÃ import torch (worker DQN). Không import torch ở đây:
    worker Q-learning dạng bảng không cần torch, import vào tốn thêm ~1.3 GB bộ nhớ mỗi tiến trình.
    """
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.set_num_threads(1)


def _call(fn: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
    setup_local_caches()
    _limit_threads()
    return fn(**kwargs)


def run_parallel(fn: Callable[..., Any], tasks: Iterable[dict[str, Any]], workers: int = 1) -> list[Any]:
    """Chạy fn(**task) cho từng task, tuần tự hoặc song song bằng nhiều tiến trình.

    fn phải là hàm cấp module (pickle được). Thứ tự kết quả giữ đúng thứ tự tasks.
    Số tiến trình = min(workers, số task): mỗi tiến trình DQN dùng ~0.8 GB RAM (CPU) hoặc ~1.4 GB (GPU),
    nên chọn workers theo RAM của máy.
    """
    task_list = list(tasks)
    if workers <= 1 or len(task_list) <= 1:
        return [fn(**t) for t in task_list]
    # Tiến trình con kế thừa biến môi trường lúc tạo: đặt 1 luồng TRƯỚC khi chúng import torch/numpy,
    # tránh mỗi tiến trình mở 12 luồng giành nhau CPU.
    saved = {k: os.environ.get(k) for k in _SINGLE_THREAD_ENV}
    os.environ.update({k: "1" for k in _SINGLE_THREAD_ENV})
    try:
        with ProcessPoolExecutor(max_workers=min(workers, len(task_list))) as pool:
            futures = [pool.submit(_call, fn, t) for t in task_list]
            return [f.result() for f in futures]
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
