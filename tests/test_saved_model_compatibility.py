"""Các mô hình đã huấn luyện và commit trong results/ (lưu bằng tên siêu tham số cũ) vẫn phải đọc được."""

from pathlib import Path

import pytest

from agents.deep_q_network.deep_q_network_agent import DeepQNetworkAgent
from agents.tabular.tabular_agent_base import TabularAgentBase
from project_utils.project_paths import RESULTS_DIR

SAVED_DEEP_MODEL = RESULTS_DIR / "deep" / "dqn" / "empty_3x3_s00" / "seed0" / "model.pt"
SAVED_Q_TABLE = RESULTS_DIR / "ablation" / "markov" / "obstacle_4x4" / "full" / "seed0" / "checkpoints" / "final.pkl"


@pytest.mark.skipif(not SAVED_DEEP_MODEL.exists(), reason="chưa có mô hình DQN đã lưu trong results/")
def test_load_deep_model_saved_with_old_setting_names() -> None:
    agent = DeepQNetworkAgent.load(SAVED_DEEP_MODEL, device="cpu")
    assert agent.settings.parallel_env_count >= 1
    assert agent.observation_shape[0] == 3


@pytest.mark.skipif(not SAVED_Q_TABLE.exists(), reason="chưa có bảng Q đã lưu trong results/")
def test_load_q_table_saved_with_old_setting_names() -> None:
    agent = TabularAgentBase.load(Path(SAVED_Q_TABLE))
    assert agent.settings.episode_count >= 1
    assert len(agent.q_table) > 0
