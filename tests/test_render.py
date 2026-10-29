"""Test cho env/render.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from environment.room_layouts import get_room_layout
from environment.rendering.frame_drawing import CELL_PIXEL_SIZE, COLORS, add_caption_to_frame, draw_text_frame, draw_rgb_frame
from environment.rendering.frame_layout import place_frames_side_by_side
from environment.rendering.gif_recording import save_gif
from environment.vacuum_cleaning_env import VacuumCleaningEnv


def test_render_rgb_shape_and_robot_color() -> None:
    grid = get_room_layout("obstacle_4x4")
    cleaned = np.zeros(grid.shape, dtype=bool)
    cleaned[0, 0] = True
    img = draw_rgb_frame(grid, (2, 1), cleaned)
    assert img.shape == (4 * CELL_PIXEL_SIZE, 4 * CELL_PIXEL_SIZE, 3)
    assert img.dtype == np.uint8
    center = img[2 * CELL_PIXEL_SIZE + CELL_PIXEL_SIZE // 2, 1 * CELL_PIXEL_SIZE + CELL_PIXEL_SIZE // 2]
    assert tuple(center) == COLORS["robot"]
    obstacle_px = img[1 * CELL_PIXEL_SIZE + CELL_PIXEL_SIZE // 2, 1 * CELL_PIXEL_SIZE + CELL_PIXEL_SIZE // 2]
    assert tuple(obstacle_px) == COLORS["obstacle"]
    cleaned_px = img[CELL_PIXEL_SIZE // 2, CELL_PIXEL_SIZE // 2]
    assert tuple(cleaned_px) == COLORS["cleaned"]


def test_env_rgb_render_with_trail() -> None:
    env = VacuumCleaningEnv("empty_3x3", start=(0, 0), render_mode="rgb_array")
    env.reset(seed=0)
    env.step(3)
    env.step(1)
    frame = env.render()
    assert frame.shape == (3 * CELL_PIXEL_SIZE, 3 * CELL_PIXEL_SIZE, 3)


def test_render_ansi() -> None:
    grid = get_room_layout("empty_3x3")
    cleaned = np.zeros((3, 3), dtype=bool)
    cleaned[0, 0] = True
    assert draw_text_frame(grid, (1, 1), cleaned) == "o..\n.R.\n..."


def test_annotate_and_compose() -> None:
    frame = np.zeros((64, 48, 3), dtype=np.uint8)
    annotated = add_caption_to_frame(frame, "Tiêu đề", ["bước 3 · 50%", "dòng 2"])
    assert annotated.shape[1] == 48 and annotated.shape[0] > 64
    other = np.zeros((30, 20, 3), dtype=np.uint8)
    composed = place_frames_side_by_side([annotated, other], gap_pixels=8)
    assert composed.shape == (annotated.shape[0], 48 + 8 + 20, 3)
    with pytest.raises(ValueError):
        place_frames_side_by_side([])


def test_save_gif(tmp_path: Path) -> None:
    frames = [np.full((16, 16, 3), v, dtype=np.uint8) for v in (0, 128, 255)]
    path = save_gif(frames, tmp_path / "sub" / "a.gif", frames_per_second=4)
    assert path.exists() and path.stat().st_size > 0
    with pytest.raises(ValueError):
        save_gif([], tmp_path / "b.gif")
