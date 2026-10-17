"""Cấu hình chung cho pytest: cache cục bộ và các fixture bản đồ/môi trường."""

from __future__ import annotations

from utils.paths import setup_local_caches

# Phải gọi trước khi bất kỳ test nào import matplotlib.
setup_local_caches()

import pytest  # noqa: E402

from env.vacuum_env import VacuumEnv  # noqa: E402


@pytest.fixture
def env_3x3_corner() -> VacuumEnv:
    """Môi trường 3x3 trống, xuất phát cố định ở góc (0,0)."""
    return VacuumEnv("empty_3x3", start=(0, 0))


@pytest.fixture
def env_4x4_obstacle() -> VacuumEnv:
    """Môi trường 4x4 có vật cản, xuất phát ngẫu nhiên."""
    return VacuumEnv("obstacle_4x4")
