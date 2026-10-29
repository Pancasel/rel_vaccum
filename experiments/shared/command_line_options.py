"""Tham số dòng lệnh dùng chung cho các script thí nghiệm."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from project_utils.project_paths import PROJECT_ROOT, require_path_inside_project


def add_shared_options(parser: argparse.ArgumentParser, default_output_directory: str) -> None:
    """Thêm các tham số chung: --seeds, --out, --quick, --workers."""
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="danh sách seed (mặc định 0..4, theo quy tắc báo cáo >= 5 seed)")
    parser.add_argument("--out", type=str, default=default_output_directory,
                        help="thư mục ghi kết quả (trong dự án)")
    parser.add_argument("--quick", action="store_true", help="chạy rút gọn để kiểm tra nhanh (< ~2 phút)")
    parser.add_argument("--workers", type=int, default=1, help="số tiến trình chạy song song")


def resolve_output_directory(output_directory: str | os.PathLike) -> Path:
    """Đường dẫn tuyệt đối của thư mục kết quả (tương đối thì tính từ gốc dự án); tạo nếu chưa có."""
    path = Path(output_directory)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    resolved_path = require_path_inside_project(path)
    resolved_path.mkdir(parents=True, exist_ok=True)
    return resolved_path


def parse_start_option(values: list[str] | None) -> tuple[int, int] | None:
    """--start random -> None; --start R C -> (R, C) với R là hàng, C là cột."""
    if not values or values == ["random"]:
        return None
    if len(values) != 2:
        raise argparse.ArgumentTypeError("--start nhận 'random' hoặc hai số nguyên R C.")
    return (int(values[0]), int(values[1]))


def validate_reward_arguments(parser: argparse.ArgumentParser, parsed_arguments: argparse.Namespace) -> None:
    """--shaping-coef chỉ có nghĩa với --reward shaped (tránh vô tình bật shaping cho preset khác)."""
    if parsed_arguments.shaping_coef is not None and parsed_arguments.reward != "shaped":
        parser.error("--shaping-coef chỉ dùng được với --reward shaped.")
