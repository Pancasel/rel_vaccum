"""Chạy nhiều tác vụ (mỗi seed/cấu hình một tác vụ) tuần tự hoặc song song bằng nhiều tiến trình."""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Iterable

from project_utils.project_paths import setup_local_caches

_SINGLE_THREAD_ENVIRONMENT_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS")


def _limit_torch_to_one_thread() -> None:
    """Tiến trình con dùng 1 luồng torch: mạng nhỏ, chạy nhiều tiến trình song song hiệu quả hơn.

    Chỉ chạm vào torch nếu tiến trình ĐÃ import torch (worker DQN). Không import torch ở đây:
    worker Q-learning dạng bảng không cần torch, import vào tốn thêm ~1.3 GB bộ nhớ mỗi tiến trình.
    """
    torch_module = sys.modules.get("torch")
    if torch_module is not None:
        torch_module.set_num_threads(1)


def _run_task_in_worker(task_function: Callable[..., Any], task_arguments: dict[str, Any]) -> Any:
    setup_local_caches()
    _limit_torch_to_one_thread()
    return task_function(**task_arguments)


def run_tasks_in_parallel(task_function: Callable[..., Any], tasks: Iterable[dict[str, Any]],
                          workers: int = 1) -> list[Any]:
    """Chạy task_function(**task) cho từng task, tuần tự hoặc song song bằng nhiều tiến trình.

    task_function phải là hàm cấp module (pickle được). Thứ tự kết quả giữ đúng thứ tự tasks.
    Số tiến trình = min(workers, số task): mỗi tiến trình DQN dùng ~0.8 GB RAM (CPU) hoặc ~1.4 GB (GPU),
    nên chọn workers theo RAM của máy.
    """
    task_list = list(tasks)
    if workers <= 1 or len(task_list) <= 1:
        return [task_function(**task) for task in task_list]
    # Tiến trình con kế thừa biến môi trường lúc tạo: đặt 1 luồng TRƯỚC khi chúng import torch/numpy,
    # tránh mỗi tiến trình mở 12 luồng giành nhau CPU.
    saved_values = {name: os.environ.get(name) for name in _SINGLE_THREAD_ENVIRONMENT_VARIABLES}
    os.environ.update({name: "1" for name in _SINGLE_THREAD_ENVIRONMENT_VARIABLES})
    try:
        with ProcessPoolExecutor(max_workers=min(workers, len(task_list))) as process_pool:
            futures = [process_pool.submit(_run_task_in_worker, task_function, task) for task in task_list]
            return [future.result() for future in futures]
    finally:
        for name, saved_value in saved_values.items():
            if saved_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = saved_value
