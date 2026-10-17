"""Chạy mọi script thí nghiệm ở chế độ --quick từ đầu đến cuối (quy tắc 0.3.6: luồng đầy đủ).

Mỗi test gọi trực tiếp main() của script với thư mục kết quả tạm (bên trong dự án) rồi kiểm tra
file đầu ra tồn tại. Các test này chậm nên được đánh dấu slow.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.slow


def test_run_baselines(tmp_path: Path) -> None:
    from experiments import run_baselines

    summary = run_baselines.main(["--quick", "--out", str(tmp_path)])
    assert (tmp_path / "summary.csv").exists() and len(summary) == 9


def test_run_value_iteration(tmp_path: Path) -> None:
    from experiments import run_value_iteration

    run_value_iteration.main(["--quick", "--out", str(tmp_path)])
    assert (tmp_path / "empty_3x3_summary.csv").exists() and (tmp_path / "empty_3x3_vi.pkl").exists()


def test_run_tabular(tmp_path: Path) -> None:
    from experiments import run_tabular

    run_tabular.main(["--quick", "--map", "empty_3x3", "--episodes", "300", "--out", str(tmp_path)])
    run_dir = tmp_path / "dense" / "full" / "empty_3x3_s00"
    assert (run_dir / "summary.csv").exists()
    assert (run_dir / "q_learning" / "seed0" / "checkpoints" / "final.pkl").exists()
    assert (tmp_path / "figures" / "tabular_learning_curve_empty_3x3_s00.png").exists()
    assert (tmp_path / "figures" / "q_convergence_empty_3x3_s00.png").exists()


def test_run_state_explosion(tmp_path: Path) -> None:
    from experiments import run_state_explosion

    run_state_explosion.main(["--quick", "--out", str(tmp_path)])
    assert (tmp_path / "state_explosion.csv").exists()
    assert (tmp_path / "figures" / "state_explosion.png").exists()
    assert (tmp_path / "tabular" / "scaling.csv").exists()


@pytest.mark.parametrize("exp", ["markov", "reward"])
def test_run_ablation(tmp_path: Path, exp: str) -> None:
    from experiments import run_ablation

    run_ablation.main(["--exp", exp, "--map", "empty_3x3", "--quick", "--out", str(tmp_path)])
    assert (tmp_path / exp / "empty_3x3" / "summary.md").exists()
    assert list((tmp_path / "gifs").glob("*.gif"))


def test_run_deep(tmp_path: Path) -> None:
    from experiments import run_deep

    run_deep.main(["--quick", "--map", "empty_3x3", "--out", str(tmp_path)])
    assert (tmp_path / "dqn" / "empty_3x3" / "seed0" / "model.pt").exists()
    assert (tmp_path / "figures" / "deep_learning_curve_empty_3x3.png").exists()


@pytest.mark.parametrize("exp", ["density", "generalization"])
def test_run_generalization(tmp_path: Path, exp: str) -> None:
    from experiments import run_generalization

    run_generalization.main(["--exp", exp, "--quick", "--size", "4", "4", "--out", str(tmp_path)])
    assert (tmp_path / exp / "summary.md").exists()
    assert list((tmp_path / "figures").glob("e*_*.png"))


def test_run_optimality(tmp_path: Path) -> None:
    from experiments import run_optimality

    run_optimality.main(["--quick", "--maps", "empty_3x3", "--out", str(tmp_path)])
    assert (tmp_path / "e6_optimality_gap.md").exists()


def test_make_demo(tmp_path: Path) -> None:
    from experiments import make_demo

    make_demo.main(["--quick", "--map", "obstacle_4x4", "--out", str(tmp_path)])
    for name in ("compare_obstacle_4x4.gif", "progress_qlearning.gif", "progress_dqn.gif",
                 "heatmap_obstacle_4x4.png", "unseen_rooms.gif", "demo_summary.md"):
        assert (tmp_path / name).exists(), name


def test_reproduce_all_commands() -> None:
    from experiments import reproduce_all

    cmd = reproduce_all.build_command("x", "experiments.run_deep", ["--map", "empty_3x3"], True, quick=True, workers=4)
    assert cmd[1:] == ["-m", "experiments.run_deep", "--map", "empty_3x3", "--quick", "--out", "results/_quick/x",
                       "--workers", "4"]
    cmd = reproduce_all.build_command("y", "experiments.run_deep", [], False, quick=False, workers=1, device="cpu")
    assert cmd[-2:] == ["--device", "cpu"]
    assert "--device" not in reproduce_all.build_command("z", "experiments.run_baselines", [], False, False, 1, "cpu")
