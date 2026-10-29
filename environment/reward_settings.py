"""Tham số phần thưởng của môi trường và các bộ cấu hình dựng sẵn (preset)."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class RewardSettings:
    """Tham số phần thưởng. frozen=True để cấu hình không bị sửa giữa chừng khi chạy ablation."""

    new_cell: float = 1.0  # vào ô chưa dọn
    revisit: float = -0.1  # vào ô đã dọn (đi lãng phí)
    collision: float = -0.5  # đụng tường/vật cản (robot đứng yên)
    step: float = -0.01  # phạt mỗi bước thời gian (khuyến khích dọn nhanh)
    completion: float = 10.0  # thưởng thêm khi dọn xong 100%
    shaping_coef: float = 0.0  # hệ số shaping thế năng (D-14), 0 = tắt
    naive_progress_bonus: float = 0.0  # reward "hackable" để minh họa (D-15), 0 = tắt
    gamma: float = 0.99  # hệ số chiết khấu γ, chỉ dùng trong công thức shaping thế năng

    PRESETS = ("dense", "sparse", "shaped", "hackable")

    @classmethod
    def dense(cls) -> "RewardSettings":
        """Cấu hình v1 trong B2: thưởng ô mới, phạt đi lại, phạt va chạm, phạt thời gian."""
        return cls()

    @classmethod
    def sparse(cls) -> "RewardSettings":
        """E2a: chỉ thưởng khi dọn xong, mọi sự kiện khác bằng 0."""
        return cls(new_cell=0.0, revisit=0.0, collision=0.0, step=0.0, completion=10.0)

    @classmethod
    def shaped(cls, coef: float = 1.0) -> "RewardSettings":
        """E2c: dense + shaping thế năng theo khoảng cách tới ô chưa dọn gần nhất."""
        return cls(shaping_coef=coef)

    @classmethod
    def hackable(cls, bonus: float = 0.5) -> "RewardSettings":
        """Reward thiết kế kém (D-15): thưởng khi lại gần ô bẩn nhưng không phạt khi đi xa.

        Không phạt đi lại, không phạt thời gian, nên agent có thể "đi ra rồi đi vào" mãi để
        ăn điểm thay vì dọn xong. Dùng để minh họa reward hacking trong E2.
        """
        return cls(new_cell=1.0, revisit=0.0, collision=0.0, step=0.0, completion=10.0,
                   naive_progress_bonus=bonus)

    @classmethod
    def from_name(cls, preset_name: str, **field_overrides: float) -> "RewardSettings":
        """Tạo cấu hình theo tên preset, có thể ghi đè từng trường.

        Raises:
            ValueError: tên preset không hợp lệ.
        """
        preset_factories = {"dense": cls.dense, "sparse": cls.sparse, "shaped": cls.shaped,
                            "hackable": cls.hackable}
        if preset_name not in preset_factories:
            raise ValueError(
                f"Reward preset '{preset_name}' không hợp lệ; chọn một trong {list(preset_factories)}."
            )
        return replace(preset_factories[preset_name](), **field_overrides)
