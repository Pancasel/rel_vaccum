"""Cấu hình chung cho pytest: cache cục bộ và các fixture bản đồ/môi trường."""

from __future__ import annotations

from project_utils.project_paths import setup_local_caches

# Phải gọi trước khi bất kỳ test nào import matplotlib.
setup_local_caches()

import pytest  # noqa: E402

from environment.vacuum_cleaning_env import VacuumCleaningEnv  # noqa: E402


@pytest.fixture
def env_3x3_corner() -> VacuumCleaningEnv:
    """Môi trường 3x3 trống, xuất phát cố định ở góc (0,0)."""
    return VacuumCleaningEnv("empty_3x3", start=(0, 0))


@pytest.fixture
def env_4x4_obstacle() -> VacuumCleaningEnv:
    """Môi trường 4x4 có vật cản, xuất phát ngẫu nhiên."""
    return VacuumCleaningEnv("obstacle_4x4")
