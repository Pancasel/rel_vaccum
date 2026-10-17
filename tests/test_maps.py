"""Test cho env/maps.py."""

from __future__ import annotations

import numpy as np
import pytest

from env.maps import (FIXED_MAPS, OBSTACLE, free_cells, generate_map_set, generate_random_map, get_map,
                      is_connected, map_to_string, parse_map)


def test_parse_map_basic() -> None:
    grid = parse_map([".#", ".."])
    assert grid.shape == (2, 2)
    assert grid.dtype == np.int8
    assert grid.tolist() == [[0, 1], [0, 0]]


def test_parse_map_rejects_bad_char() -> None:
    with pytest.raises(ValueError):
        parse_map([".x", ".."])


def test_parse_map_rejects_ragged() -> None:
    with pytest.raises(ValueError):
        parse_map(["...", ".."])


def test_parse_map_rejects_no_free() -> None:
    with pytest.raises(ValueError):
        parse_map(["##"])


@pytest.mark.parametrize("name", sorted(FIXED_MAPS))
def test_fixed_maps_connected(name: str) -> None:
    assert is_connected(get_map(name))


def test_get_map_returns_copy() -> None:
    grid = get_map("empty_3x3")
    grid[0, 0] = OBSTACLE
    assert get_map("empty_3x3")[0, 0] == 0


def test_get_map_unknown_name() -> None:
    with pytest.raises(KeyError):
        get_map("no_such_map")


@pytest.mark.parametrize("name,expected", [("empty_3x3", 9), ("obstacle_4x4", 13), ("room_6x6", 29),
                                           ("furniture_8x8", 52), ("empty_4x4", 16)])
def test_free_cell_counts(name: str, expected: int) -> None:
    assert len(free_cells(get_map(name))) == expected


def test_free_cells_row_major_order() -> None:
    assert free_cells(parse_map([".#", ".."])) == [(0, 0), (1, 0), (1, 1)]


def test_is_connected_detects_isolated_cell() -> None:
    assert is_connected(parse_map([".#.", "...", "..."]))
    assert not is_connected(parse_map([".#.", "#..", "..."]))


@pytest.mark.parametrize("density", [0.0, 0.1, 0.2, 0.3])
def test_random_maps_connected_and_exact_count(density: float) -> None:
    for seed in range(200):
        grid = generate_random_map(8, 8, density, np.random.default_rng(seed))
        assert is_connected(grid)
        assert int((grid == OBSTACLE).sum()) == round(density * 64)


def test_random_map_reproducible() -> None:
    a = generate_random_map(6, 6, 0.2, np.random.default_rng(7))
    b = generate_random_map(6, 6, 0.2, np.random.default_rng(7))
    c = generate_random_map(6, 6, 0.2, np.random.default_rng(8))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_random_map_invalid_args() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        generate_random_map(3, 3, 1.0, rng)
    with pytest.raises(ValueError):
        generate_random_map(0, 3, 0.1, rng)
    with pytest.raises(ValueError):
        generate_random_map(2, 1, 0.5, rng)  # còn 1 ô trống < 2


def test_generate_map_set_distinct_and_exclude() -> None:
    maps = generate_map_set(6, 6, 0.2, n_maps=20, seed=1)
    assert len(maps) == 20
    assert len({m.tobytes() for m in maps}) == 20
    more = generate_map_set(6, 6, 0.2, n_maps=10, seed=1, exclude=maps)
    assert not ({m.tobytes() for m in more} & {m.tobytes() for m in maps})


def test_generate_map_set_impossible() -> None:
    with pytest.raises(RuntimeError):
        generate_map_set(4, 4, 0.0, n_maps=2, seed=0, max_attempts_factor=5)


def test_map_to_string_roundtrip() -> None:
    lines = FIXED_MAPS["furniture_8x8"]
    assert map_to_string(get_map("furniture_8x8")) == "\n".join(lines)
