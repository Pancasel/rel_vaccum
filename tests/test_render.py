"""Test cho env/render.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from env.maps import get_map
from env.render import (CELL_PX, COLORS, annotate_frame, compose_side_by_side, render_ansi, render_rgb,
                        save_gif)
from env.vacuum_env import VacuumEnv


def test_render_rgb_shape_and_robot_color() -> None:
    grid = get_map("obstacle_4x4")
    cleaned = np.zeros(grid.shape, dtype=bool)
    cleaned[0, 0] = True
    img = render_rgb(grid, (2, 1), cleaned)
    assert img.shape == (4 * CELL_PX, 4 * CELL_PX, 3)
    assert img.dtype == np.uint8
    center = img[2 * CELL_PX + CELL_PX // 2, 1 * CELL_PX + CELL_PX // 2]
    assert tuple(center) == COLORS["robot"]
    obstacle_px = img[1 * CELL_PX + CELL_PX // 2, 1 * CELL_PX + CELL_PX // 2]
    assert tuple(obstacle_px) == COLORS["obstacle"]
    cleaned_px = img[CELL_PX // 2, CELL_PX // 2]
    assert tuple(cleaned_px) == COLORS["cleaned"]


def test_env_rgb_render_with_trail() -> None:
    env = VacuumEnv("empty_3x3", start=(0, 0), render_mode="rgb_array")
    env.reset(seed=0)
    env.step(3)
    env.step(1)
    frame = env.render()
    assert frame.shape == (3 * CELL_PX, 3 * CELL_PX, 3)


def test_render_ansi() -> None:
    grid = get_map("empty_3x3")
    cleaned = np.zeros((3, 3), dtype=bool)
    cleaned[0, 0] = True
    assert render_ansi(grid, (1, 1), cleaned) == "o..\n.R.\n..."


def test_annotate_and_compose() -> None:
    frame = np.zeros((64, 48, 3), dtype=np.uint8)
    annotated = annotate_frame(frame, "Tiêu đề", ["bước 3 · 50%", "dòng 2"])
    assert annotated.shape[1] == 48 and annotated.shape[0] > 64
    other = np.zeros((30, 20, 3), dtype=np.uint8)
    composed = compose_side_by_side([annotated, other], gap_px=8)
    assert composed.shape == (annotated.shape[0], 48 + 8 + 20, 3)
    with pytest.raises(ValueError):
        compose_side_by_side([])


def test_save_gif(tmp_path: Path) -> None:
    frames = [np.full((16, 16, 3), v, dtype=np.uint8) for v in (0, 128, 255)]
    path = save_gif(frames, tmp_path / "sub" / "a.gif", fps=4)
    assert path.exists() and path.stat().st_size > 0
    with pytest.raises(ValueError):
        save_gif([], tmp_path / "b.gif")
