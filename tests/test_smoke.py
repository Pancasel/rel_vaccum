"""Kiểm tra môi trường cài đặt: các thư viện import được và ROOT đúng."""

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
    from utils.paths import ROOT

    assert (ROOT / "IMPLEMENTATION_PLAN.md").exists()
    assert ROOT == Path(__file__).resolve().parents[1]


def test_assert_inside_root_rejects_outside(tmp_path: Path) -> None:
    import pytest

    from utils.paths import ROOT, assert_inside_root

    assert assert_inside_root(ROOT / "results") == (ROOT / "results").resolve()
    with pytest.raises(ValueError):
        assert_inside_root(ROOT.parent / "somewhere_else.txt")
