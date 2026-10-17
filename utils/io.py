"""Ghi kết quả thí nghiệm (CSV, JSON cấu hình) vào thư mục dự án."""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.paths import ROOT, assert_inside_root


def ensure_dir(path: str | os.PathLike) -> Path:
    """Tạo thư mục (kể cả thư mục cha) nếu chưa có; chỉ cho phép trong ROOT."""
    resolved = assert_inside_root(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def save_csv(df: pd.DataFrame, path: str | os.PathLike) -> Path:
    """Ghi DataFrame ra CSV (không ghi cột index), tự tạo thư mục cha."""
    resolved = assert_inside_root(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(resolved, index=False)
    return resolved


def _to_jsonable(obj: Any) -> Any:
    """Chuyển dataclass, numpy, Path, tuple... về kiểu JSON ghi được."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _to_jsonable(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _git_commit() -> str | None:
    """Mã commit hiện tại (để biết kết quả sinh ra từ phiên bản code nào), None nếu không có."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def save_config_json(config: Any, path: str | os.PathLike) -> Path:
    """Ghi cấu hình thí nghiệm kèm thời điểm, commit git và phiên bản thư viện.

    Vì sao: Phần E của kế hoạch yêu cầu báo cáo trung thực siêu tham số và ngân sách
    huấn luyện; file này đặt cạnh CSV để luôn biết kết quả được tạo ra như thế nào.
    """
    resolved = assert_inside_root(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
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
