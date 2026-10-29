"""Quy ước đặt tên thư mục kết quả trong results/.

Tên thư mục là ĐỊNH DANH ĐÃ ĐƯỢC báo cáo, slide và README trích dẫn nên các viết tắt dưới đây
được giữ nguyên; bảng này giải nghĩa từng viết tắt.

Ví dụ: `rand8x8_d0.15_K0_mask_ego_ne16_bs256_tf16`
    rand8x8   rand = phòng ngẫu nhiên (random rooms), kích thước 8 x 8
    d0.15     d = density, mật độ vật cản 0.15
    K0        K = số bản đồ huấn luyện cố định (0 = sinh phòng mới mỗi episode)
    mask      dùng action masking (che hướng đâm vào tường)
    ego       dùng quan sát lấy robot làm tâm (egocentric)
    ne16      ne = number of parallel environments (số môi trường chạy song song) = 16
    bs256     bs = batch size = 256
    tf16      tf = train frequency: cập nhật mạng mỗi 16 bước môi trường

Các hậu tố khác:
    s00       s = start: ô xuất phát cố định (hàng 0, cột 0); thiếu hậu tố = xuất phát ngẫu nhiên
    sc2       sc = shaping coefficient: hệ số shaping thế năng = 2
    lr        lr = learning rate (tốc độ học)
    tu        tu = target update: số bước giữa hai lần chép mạng target
    buf       buf = replay buffer capacity (sức chứa bộ nhớ kinh nghiệm)
    _shaped / _sparse / _hackable   tên preset reward khác "dense"
"""

from __future__ import annotations

import argparse
from typing import Any

from agents.deep_q_network.deep_q_network_settings import DeepQNetworkSettings

# Cờ dòng lệnh -> (tên trường DeepQNetworkSettings, hậu tố viết tắt trong tên thư mục).
# Hậu tố chỉ được thêm vào tên khi giá trị khác mặc định.
TRAINING_OPTION_ABBREVIATIONS = {
    "learning_rate": ("learning_rate", "lr"),
    "target_update": ("target_network_sync_interval_steps", "tu"),
    "buffer_size": ("replay_buffer_capacity", "buf"),
    "n_envs": ("parallel_env_count", "ne"),
    "batch_size": ("batch_size", "bs"),
    "train_freq": ("steps_per_network_update", "tf"),
}


def build_fixed_room_folder_name(room_name: str, start: tuple[int, int] | None) -> str:
    """Tên bản đồ kèm ô xuất phát cố định ('obstacle_4x4_s00'; xuất phát ngẫu nhiên giữ nguyên tên).

    Vì sao: giá trị học được gắn với ô xuất phát; thiếu hậu tố thì lượt xuất phát cố định và lượt
    xuất phát ngẫu nhiên ghi đè kết quả của nhau.
    """
    if start is None:
        return room_name
    return f"{room_name}_s{start[0]}{start[1]}"


def build_environment_variant_suffix(options: dict[str, Any]) -> str:
    """Hậu tố cho các tùy chọn môi trường/cải tiến (reward, shaping, action mask, egocentric)."""
    suffix = ""
    if options["reward"] != "dense":
        suffix += f"_{options['reward']}"
    if options.get("shaping_coef") is not None:
        suffix += f"_sc{options['shaping_coef']:g}"
    if options.get("action_mask"):
        suffix += "_mask"
    if options.get("egocentric"):
        suffix += "_ego"
    return suffix


def build_environment_folder_name(spec: dict[str, Any]) -> str:
    """Tên thư mục kết quả, ví dụ 'furniture_8x8', 'rand8x8_d0.15_K0', thêm hậu tố nếu đổi reward/cải tiến."""
    if spec.get("random"):
        height, width, obstacle_density = spec["random"]
        training_room_count = len(spec["pool"]) if spec.get("pool") else 0
        base_name = f"rand{height}x{width}_d{obstacle_density:g}_K{training_room_count}"
    else:
        base_name = build_fixed_room_folder_name(spec["map"], spec.get("start"))
    return base_name + build_environment_variant_suffix(spec)


def build_training_options_suffix(parsed_arguments: argparse.Namespace) -> str:
    """Hậu tố cho siêu tham số huấn luyện khác mặc định, để các lần chạy khác nhau không ghi đè nhau."""
    suffix = ""
    for flag_name, (settings_field, abbreviation) in TRAINING_OPTION_ABBREVIATIONS.items():
        value = getattr(parsed_arguments, flag_name, None)
        if value is not None and value != getattr(DeepQNetworkSettings, settings_field):
            suffix += f"_{abbreviation}{value:g}"
    return suffix
