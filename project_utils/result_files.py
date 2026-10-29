"""Ghi kết quả thí nghiệm (CSV, JSON cấu hình) vào thư mục dự án."""

from __future__ import annotations

import dataclasses
import datetime as datetime_module
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from project_utils.project_paths import PROJECT_ROOT, require_path_inside_project


def ensure_dir(path: str | os.PathLike) -> Path:
    """Tạo thư mục (kể cả thư mục cha) nếu chưa có; chỉ cho phép trong PROJECT_ROOT."""
    resolved = require_path_inside_project(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def save_dataframe_csv(dataframe: pd.DataFrame, path: str | os.PathLike) -> Path:
    """Ghi DataFrame ra CSV (không ghi cột index), tự tạo thư mục cha."""
    resolved = require_path_inside_project(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(resolved, index=False)
    return resolved


def _to_jsonable(value_to_convert: Any) -> Any:
    """Chuyển dataclass, numpy, Path, tuple... về kiểu JSON ghi được."""
    if dataclasses.is_dataclass(value_to_convert) and not isinstance(value_to_convert, type):
        return _to_jsonable(dataclasses.asdict(value_to_convert))
    if isinstance(value_to_convert, dict):
        return {str(key): _to_jsonable(item) for key, item in value_to_convert.items()}
    if isinstance(value_to_convert, (list, tuple)):
        return [_to_jsonable(item) for item in value_to_convert]
    if isinstance(value_to_convert, Path):
        return str(value_to_convert)
    if isinstance(value_to_convert, np.generic):
        return value_to_convert.item()
    if isinstance(value_to_convert, np.ndarray):
        return value_to_convert.tolist()
    return value_to_convert


def _current_git_commit_hash() -> str | None:
    """Mã commit hiện tại (để biết kết quả sinh ra từ phiên bản code nào), None nếu không có."""
    try:
        git_result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return git_result.stdout.strip() or None


def save_config_with_metadata(config: Any, path: str | os.PathLike) -> Path:
    """Ghi cấu hình thí nghiệm kèm thời điểm, commit git và phiên bản thư viện.

    Vì sao: Phần E của kế hoạch yêu cầu báo cáo trung thực siêu tham số và ngân sách
    huấn luyện; file này đặt cạnh CSV để luôn biết kết quả được tạo ra như thế nào.
    """
    resolved = require_path_inside_project(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {
        "timestamp": datetime_module.datetime.now().isoformat(timespec="seconds"),
        "git_commit": _current_git_commit_hash(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    try:
        import torch

        meta["torch"] = torch.__version__
    except ImportError:
        pass
    payload = {"config": _to_jsonable(config), "meta": meta}
    resolved.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return resolved
