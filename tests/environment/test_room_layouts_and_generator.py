"""Test cho env/maps.py."""

from __future__ import annotations

import numpy as np
import pytest

from environment.room_layouts import FIXED_ROOM_LAYOUTS, OBSTACLE_CELL, list_free_cells, get_room_layout, room_layout_to_text, parse_room_layout
from environment.random_room_generator import generate_distinct_rooms, generate_random_room, is_fully_connected


def test_parse_map_basic() -> None:
    grid = parse_room_layout([".#", ".."])
    assert grid.shape == (2, 2)
    assert grid.dtype == np.int8
    assert grid.tolist() == [[0, 1], [0, 0]]


def test_parse_map_rejects_bad_char() -> None:
    with pytest.raises(ValueError):
        parse_room_layout([".x", ".."])


def test_parse_map_rejects_ragged() -> None:
    with pytest.raises(ValueError):
        parse_room_layout(["...", ".."])


def test_parse_map_rejects_no_free() -> None:
    with pytest.raises(ValueError):
        parse_room_layout(["##"])


@pytest.mark.parametrize("name", sorted(FIXED_ROOM_LAYOUTS))
def test_fixed_maps_connected(name: str) -> None:
    assert is_fully_connected(get_room_layout(name))


def test_get_map_returns_copy() -> None:
    grid = get_room_layout("empty_3x3")
    grid[0, 0] = OBSTACLE_CELL
    assert get_room_layout("empty_3x3")[0, 0] == 0


def test_get_map_unknown_name() -> None:
    with pytest.raises(KeyError):
        get_room_layout("no_such_map")


@pytest.mark.parametrize("name,expected", [("empty_3x3", 9), ("obstacle_4x4", 13), ("room_6x6", 29),
                                           ("furniture_8x8", 52), ("empty_4x4", 16)])
def test_free_cell_counts(name: str, expected: int) -> None:
    assert len(list_free_cells(get_room_layout(name))) == expected


def test_free_cells_row_major_order() -> None:
    assert list_free_cells(parse_room_layout([".#", ".."])) == [(0, 0), (1, 0), (1, 1)]


def test_is_connected_detects_isolated_cell() -> None:
    assert is_fully_connected(parse_room_layout([".#.", "...", "..."]))
    assert not is_fully_connected(parse_room_layout([".#.", "#..", "..."]))


@pytest.mark.parametrize("density", [0.0, 0.1, 0.2, 0.3])
def test_random_maps_connected_and_exact_count(density: float) -> None:
    for seed in range(200):
        grid = generate_random_room(8, 8, density, np.random.default_rng(seed))
        assert is_fully_connected(grid)
        assert int((grid == OBSTACLE_CELL).sum()) == round(density * 64)


def test_random_map_reproducible() -> None:
    a = generate_random_room(6, 6, 0.2, np.random.default_rng(7))
    b = generate_random_room(6, 6, 0.2, np.random.default_rng(7))
    c = generate_random_room(6, 6, 0.2, np.random.default_rng(8))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_random_map_invalid_args() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        generate_random_room(3, 3, 1.0, rng)
    with pytest.raises(ValueError):
        generate_random_room(0, 3, 0.1, rng)
    with pytest.raises(ValueError):
        generate_random_room(2, 1, 0.5, rng)  # còn 1 ô trống < 2


def test_generate_map_set_distinct_and_exclude() -> None:
    maps = generate_distinct_rooms(6, 6, 0.2, room_count=20, seed=1)
    assert len(maps) == 20
    assert len({m.tobytes() for m in maps}) == 20
    more = generate_distinct_rooms(6, 6, 0.2, room_count=10, seed=1, excluded_rooms=maps)
    assert not ({m.tobytes() for m in more} & {m.tobytes() for m in maps})


def test_generate_map_set_impossible() -> None:
    with pytest.raises(RuntimeError):
        generate_distinct_rooms(4, 4, 0.0, room_count=2, seed=0, max_attempts_factor=5)


def test_map_to_string_roundtrip() -> None:
    lines = FIXED_ROOM_LAYOUTS["furniture_8x8"]
    assert room_layout_to_text(get_room_layout("furniture_8x8")) == "\n".join(lines)
