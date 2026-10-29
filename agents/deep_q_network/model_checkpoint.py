"""Lưu và đọc agent DQN (file .pt)."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import torch

from agents.deep_q_network.deep_q_network_agent import DeepQNetworkAgent
from agents.deep_q_network.deep_q_network_settings import DeepQNetworkSettings
from agents.deep_q_network.q_value_network import count_trainable_parameters
from project_utils.project_paths import require_path_inside_project


def save_agent(agent: DeepQNetworkAgent, path: str | os.PathLike) -> Path:
    """Lưu trọng số mạng online, cấu hình và shape observation.

    Tên các khóa trong file ("state_dict", "config", "obs_shape", "n_parameters") giữ nguyên để
    đọc được các mô hình đã huấn luyện trước đây.
    """
    resolved_path = require_path_inside_project(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": agent.online_network.state_dict(), "config": dataclasses.asdict(agent.settings),
                "obs_shape": agent.observation_shape,
                "n_parameters": count_trainable_parameters(agent.online_network)}, resolved_path)
    return resolved_path


def load_agent(path: str | os.PathLike, device: str = "auto") -> DeepQNetworkAgent:
    """Đọc agent đã lưu (mạng target được đặt bằng mạng online).

    Đọc được cả file lưu bằng tên siêu tham số cũ (n_envs, action_mask...) nhờ bảng
    DeepQNetworkSettings.LEGACY_FIELD_NAMES.

    Args:
        device: thiết bị chạy mô hình sau khi đọc (mô hình huấn luyện trên GPU vẫn đọc được trên CPU).
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    saved_settings = {DeepQNetworkSettings.LEGACY_FIELD_NAMES.get(name, name): value
                      for name, value in payload["config"].items()}
    saved_settings["checkpoint_steps"] = tuple(saved_settings.get("checkpoint_steps", ()))
    saved_settings["device"] = device
    agent = DeepQNetworkAgent(tuple(payload["obs_shape"]), DeepQNetworkSettings(**saved_settings))
    agent.online_network.load_state_dict(payload["state_dict"])
    agent.target_network.load_state_dict(payload["state_dict"])
    return agent
