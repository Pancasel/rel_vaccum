"""Chuẩn bị môi trường tiến trình cho script thí nghiệm."""

from __future__ import annotations

import sys

from project_utils.project_paths import setup_local_caches


def enable_utf8_console() -> None:
    """In được tiếng Việt trên console Windows (mặc định cp1252 sẽ báo UnicodeEncodeError)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def prepare_process_environment() -> None:
    """Đặt cache cục bộ cho matplotlib và bật console UTF-8."""
    setup_local_caches()
    enable_utf8_console()
