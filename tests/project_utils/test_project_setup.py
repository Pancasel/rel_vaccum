"""Kiểm tra môi trường cài đặt: các thư viện import được và PROJECT_ROOT đúng."""

from __future__ import annotations

from pathlib import Path


def test_imports() -> None:
    import gymnasium  # noqa: F401
    import matplotlib  # noqa: F401
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import PIL  # noqa: F401
    import torch  # noqa: F401


def test_root_is_repo_root() -> None:
    from project_utils.project_paths import PROJECT_ROOT

    assert (PROJECT_ROOT / "docs/IMPLEMENTATION_PLAN.md").exists()
    assert PROJECT_ROOT == Path(__file__).resolve().parents[2]


def test_require_path_inside_project_rejects_outside(tmp_path: Path) -> None:
    import pytest

    from project_utils.project_paths import PROJECT_ROOT, require_path_inside_project

    assert require_path_inside_project(PROJECT_ROOT / "results") == (PROJECT_ROOT / "results").resolve()
    with pytest.raises(ValueError):
        require_path_inside_project(PROJECT_ROOT.parent / "somewhere_else.txt")
