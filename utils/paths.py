"""Đường dẫn gốc của dự án và thiết lập cache cục bộ.

Quy tắc dự án: mọi file (kết quả, cache, file tạm) phải nằm trong thư mục dự án.
Module này cung cấp hằng số ROOT và hàm kiểm tra một đường dẫn có nằm trong ROOT không.
"""

from __future__ import annotations

import os
from pathlib import Path

# Thư mục gốc dự án = thư mục cha của thư mục utils/.
ROOT: Path = Path(__file__).resolve().parents[1]
RESULTS_DIR: Path = ROOT / "results"
CACHE_DIR: Path = ROOT / ".cache"


def setup_local_caches() -> None:
    """Đặt cache của matplotlib vào ROOT/.cache và dùng backend không cửa sổ.

    Vì sao: mặc định matplotlib ghi cache vào thư mục home của người dùng, vi phạm quy tắc
    "không ghi ra ngoài thư mục dự án". Backend "Agg" giúp script chạy được cả trên máy
    không có màn hình (Colab, server) và không bật cửa sổ khi vẽ hình.
    """
    mpl_dir = CACHE_DIR / "matplotlib"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("MPLBACKEND", "Agg")


def assert_inside_root(path: str | os.PathLike) -> Path:
    """Chuẩn hóa đường dẫn và báo lỗi nếu nó nằm ngoài thư mục dự án.

    Args:
        path: đường dẫn tương đối (tính từ thư mục làm việc) hoặc tuyệt đối.

    Returns:
        Đường dẫn tuyệt đối đã chuẩn hóa.

    Raises:
        ValueError: nếu đường dẫn nằm ngoài ROOT.
    """
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError(
            f"Đường dẫn '{resolved}' nằm ngoài thư mục dự án '{ROOT}'. "
            "Mọi file kết quả phải được ghi bên trong thư mục dự án."
        ) from exc
    return resolved
