"""Tên thư mục kết quả phải giữ đúng quy ước cũ: báo cáo, slide và README trích dẫn các tên này."""

import argparse

from experiments.shared.result_folder_naming import (build_environment_folder_name,
                                                      build_environment_variant_suffix,
                                                      build_fixed_room_folder_name,
                                                      build_training_options_suffix)


def test_fixed_room_folder_name() -> None:
    assert build_fixed_room_folder_name("obstacle_4x4", (0, 0)) == "obstacle_4x4_s00"
    assert build_fixed_room_folder_name("obstacle_4x4", None) == "obstacle_4x4"


def test_random_room_folder_name_with_all_improvements() -> None:
    spec = {"map": None, "random": (8, 8, 0.15), "pool": None, "reward": "dense", "shaping_coef": None,
            "action_mask": True, "egocentric": True}
    arguments = argparse.Namespace(n_envs=16, batch_size=256, train_freq=16, learning_rate=None,
                                   target_update=None, buffer_size=None)
    folder_name = build_environment_folder_name(spec) + build_training_options_suffix(arguments)
    assert folder_name == "rand8x8_d0.15_K0_mask_ego_ne16_bs256_tf16"


def test_variant_suffix_for_shaped_reward() -> None:
    options = {"reward": "shaped", "shaping_coef": 2.0, "action_mask": False, "egocentric": False}
    assert build_environment_variant_suffix(options) == "_shaped_sc2"


def test_default_training_options_add_no_suffix() -> None:
    arguments = argparse.Namespace(n_envs=1, batch_size=64, train_freq=4, learning_rate=2.5e-4,
                                   target_update=500, buffer_size=100_000)
    assert build_training_options_suffix(arguments) == ""
