"""Ghi bảng markdown và văn bản ra file (để dán vào README/slide)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from project_utils.project_paths import require_path_inside_project


def dataframe_to_markdown_table(table: pd.DataFrame, float_format: str = ".2f") -> str:
    """Bảng markdown (để dán vào README/slide) mà không cần thư viện tabulate."""

    def format_cell(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            return "nan" if np.isnan(value) else format(float(value), float_format)
        return str(value)

    header = "| " + " | ".join(str(column) for column in table.columns) + " |"
    separator = "|" + "|".join("---" for _ in table.columns) + "|"
    body_rows = ["| " + " | ".join(format_cell(value) for value in row) + " |"
                 for row in table.itertuples(index=False)]
    return "\n".join([header, separator, *body_rows])


def write_text_file(text: str, output_path: str | os.PathLike) -> Path:
    """Ghi văn bản (ví dụ bảng markdown) ra file trong thư mục dự án."""
    resolved_path = require_path_inside_project(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(text, encoding="utf-8")
    return resolved_path
