"""Đường dẫn gốc của dự án và thiết lập cache cục bộ.

Quy tắc dự án: mọi file (kết quả, cache, file tạm) phải nằm trong thư mục dự án.
Module này cung cấp hằng số PROJECT_ROOT và hàm kiểm tra một đường dẫn có nằm trong PROJECT_ROOT không.
"""

from __future__ import annotations

import os
from pathlib import Path

# Thư mục gốc dự án = thư mục cha của thư mục project_utils/.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
RESULTS_DIR: Path = PROJECT_ROOT / "results"
CACHE_DIR: Path = PROJECT_ROOT / ".cache"


def setup_local_caches() -> None:
    """Đặt cache của matplotlib vào PROJECT_ROOT/.cache và dùng backend không cửa sổ.

    Vì sao: mặc định matplotlib ghi cache vào thư mục home của người dùng, vi phạm quy tắc
    "không ghi ra ngoài thư mục dự án". Backend "Agg" giúp script chạy được cả trên máy
    không có màn hình (Colab, server) và không bật cửa sổ khi vẽ hình.
    """
    matplotlib_cache_dir = CACHE_DIR / "matplotlib"
    matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))
    os.environ.setdefault("MPLBACKEND", "Agg")


def require_path_inside_project(path: str | os.PathLike) -> Path:
    """Chuẩn hóa đường dẫn và báo lỗi nếu nó nằm ngoài thư mục dự án.

    Args:
        path: đường dẫn tương đối (tính từ thư mục làm việc) hoặc tuyệt đối.

    Returns:
        Đường dẫn tuyệt đối đã chuẩn hóa.

    Raises:
        ValueError: nếu đường dẫn nằm ngoài PROJECT_ROOT.
    """
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"Đường dẫn '{resolved}' nằm ngoài thư mục dự án '{PROJECT_ROOT}'. "
            "Mọi file kết quả phải được ghi bên trong thư mục dự án."
        ) from exc
    return resolved
