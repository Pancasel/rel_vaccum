"""Phần dùng chung cho agent dạng bảng: bảng Q, chọn hành động, vòng huấn luyện, lưu/đọc.

Lớp con (QLearningAgent) chỉ cài đặt `_run_one_episode` — đúng phần pseudocode trong sách —
còn mọi thứ khác (ghi log, đánh giá định kỳ, lưu checkpoint) nằm ở đây để code thuật toán
gọn và dễ đối chiếu với Sutton & Barto.
"""

from __future__ import annotations

import dataclasses
import os
import pickle
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, ClassVar

import gymnasium
import numpy as np
import pandas as pd

from agents.tabular.state_key_encoder import StateKeyEncoder
from agents.tabular.tabular_settings import TabularSettings, linear_epsilon_for_episode
from evaluation.agent_evaluation import evaluate_agent
from evaluation.metric_summary import summarize_metrics
from project_utils.project_paths import require_path_inside_project
from project_utils.result_files import save_dataframe_csv

ACTION_COUNT: int = 4
# Hai giá trị Q chênh nhau không quá ngưỡng này thì coi là bằng nhau khi tìm hành động tốt nhất.
TIE_TOLERANCE: float = 1e-9


class TabularAgentBase(ABC):
    """Lớp cơ sở (trừu tượng) cho agent bảng Q. Lớp con đặt `algo_name` và cài `_run_one_episode`."""

    algo_name: ClassVar[str] = "tabular"
    registry: ClassVar[dict[str, type["TabularAgentBase"]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        TabularAgentBase.registry[cls.algo_name] = cls

    def __init__(self, grid: np.ndarray, settings: TabularSettings) -> None:
        self.grid = np.array(grid, dtype=np.int8, copy=True)
        self.settings = settings
        self.state_encoder = StateKeyEncoder(self.grid, settings.state_mode)
        self.random_generator = np.random.default_rng(settings.seed)
        # Bảng Q: khóa trạng thái -> mảng 4 giá trị Q, mỗi hành động một giá trị.
        self.q_table: dict[tuple, np.ndarray] = {}

    # ------------------------------------------------------------------ bảng Q
    def read_q_values(self, state_key: tuple) -> np.ndarray:
        """Đọc Q(s, ·) (không thêm trạng thái mới vào bảng); trạng thái chưa gặp trả giá trị Q ban đầu."""
        q_values = self.q_table.get(state_key)
        if q_values is not None:
            return q_values
        return np.full(ACTION_COUNT, self.settings.initial_q_value)

    def _q_values_for_update(self, state_key: tuple) -> np.ndarray:
        """Hàng Q(s, ·) để GHI (tạo mới với giá trị Q ban đầu nếu chưa có)."""
        q_values = self.q_table.get(state_key)
        if q_values is None:
            q_values = np.full(ACTION_COUNT, self.settings.initial_q_value, dtype=np.float64)
            self.q_table[state_key] = q_values
        return q_values

    def choose_epsilon_greedy_action(self, state_key: tuple, epsilon: float) -> int:
        """ε-greedy (Sutton & Barto 2018, mục 2.2): xác suất ε chọn ngẫu nhiên, còn lại chọn tốt nhất.

        Khi nhiều hành động cùng tốt nhất thì chọn ngẫu nhiên trong số đó (D-20), để agent
        không bị thiên vị hành động "lên" lúc bảng Q còn toàn số 0.
        """
        if self.random_generator.random() < epsilon:
            return int(self.random_generator.integers(ACTION_COUNT))
        q_values = self.read_q_values(state_key)
        best_actions = np.flatnonzero(q_values >= q_values.max() - TIE_TOLERANCE)
        return int(self.random_generator.choice(best_actions))

    def choose_greedy_action(self, state_key: tuple) -> int:
        """Hành động tham lam tất định (phá hòa chọn chỉ số nhỏ nhất), dùng khi đánh giá."""
        q_values = self.read_q_values(state_key)
        return int(np.argmax(q_values >= q_values.max() - TIE_TOLERANCE))

    def act(self, observation: np.ndarray, info: dict | None = None) -> int:
        """Giao diện Agent: chính sách tham lam theo bảng Q."""
        return self.choose_greedy_action(self.state_encoder.encode_observation(observation))

    def best_value_of_state(self, state_key: tuple) -> float:
        """max_a Q(s, a) — với s là trạng thái xuất phát thì so với V*(s0) để thấy ước lượng hội tụ."""
        return float(self.read_q_values(state_key).max())

    # ------------------------------------------------------------------ thuật toán (lớp con)
    @abstractmethod
    def _run_one_episode(self, env: gymnasium.Env, epsilon: float, seed: int | None) -> dict[str, Any]:
        """Chạy một episode và cập nhật bảng Q. Trả return, coverage, steps, terminated, truncated, start_key."""

    # ------------------------------------------------------------------ vòng huấn luyện
    def train(self, env_factory: Callable[[], gymnasium.Env], log_dir: str | os.PathLike | None = None
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Huấn luyện episode_count episode, đánh giá tham lam định kỳ, lưu checkpoint.

        Returns:
            (train_log, eval_log). Ghi CSV vào log_dir nếu có.
        """
        settings = self.settings
        env = env_factory()
        fixed_start = getattr(env.unwrapped, "fixed_start", None)
        log_path = require_path_inside_project(log_dir) if log_dir is not None else None
        if log_path is not None and 0 in settings.checkpoint_episodes:
            self.save(log_path / "checkpoints" / "ep0.pkl")
        start_time = time.monotonic()
        train_rows: list[dict[str, Any]] = []
        eval_rows: list[dict[str, Any]] = []
        for episode_number in range(1, settings.episode_count + 1):
            epsilon = linear_epsilon_for_episode(episode_number - 1, settings)
            # Chỉ seed ở episode đầu; các episode sau dùng tiếp bộ sinh số của env -> vẫn tái lập được.
            episode_stats = self._run_one_episode(env, epsilon,
                                                  seed=settings.seed if episode_number == 1 else None)
            if fixed_start is not None:
                start_state_value = self.best_value_of_state(episode_stats["start_key"])
            else:
                start_state_value = float("nan")
            # Tên cột CSV là giao diện dữ liệu (báo cáo đang trích), không đổi.
            train_rows.append({
                "episode": episode_number,
                "epsilon": epsilon,
                "return_undiscounted": episode_stats["return"],
                "coverage": episode_stats["coverage"],
                "steps": episode_stats["steps"],
                "terminated": episode_stats["terminated"],
                "truncated": episode_stats["truncated"],
                "n_states_visited": len(self.q_table),
                "q_start_max": start_state_value,
                "wall_time_s": time.monotonic() - start_time,
            })
            has_exceeded_time_budget = (settings.time_budget_minutes is not None
                                        and time.monotonic() - start_time > 60 * settings.time_budget_minutes)
            is_evaluation_episode = (episode_number % settings.evaluate_every_episodes == 0
                                     or episode_number == settings.episode_count or has_exceeded_time_budget)
            if is_evaluation_episode:
                evaluation_summary = summarize_metrics(evaluate_agent(
                    self, env_factory, settings.evaluation_episode_count, seed=settings.seed,
                    gamma=settings.gamma))
                eval_rows.append({"episode": episode_number, "n_states_visited": len(self.q_table),
                                  **evaluation_summary.iloc[0].to_dict()})
            if log_path is not None and episode_number in settings.checkpoint_episodes:
                self.save(log_path / "checkpoints" / f"ep{episode_number}.pkl")
            if has_exceeded_time_budget:
                break
        train_log, eval_log = pd.DataFrame(train_rows), pd.DataFrame(eval_rows)
        if log_path is not None:
            self.save(log_path / "checkpoints" / "final.pkl")
            save_dataframe_csv(train_log, log_path / f"seed{settings.seed}_train.csv")
            save_dataframe_csv(eval_log, log_path / f"seed{settings.seed}_eval.csv")
        return train_log, eval_log

    # ------------------------------------------------------------------ lưu / đọc
    def save(self, path: str | os.PathLike) -> Path:
        """Lưu bảng Q, cấu hình và bản đồ (pickle; bảng Q nhỏ nên file nhỏ).

        Tên các khóa trong file ("algo", "config", "grid", "q") giữ nguyên để đọc được file cũ.
        """
        resolved_path = require_path_inside_project(path)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"algo": self.algo_name, "config": dataclasses.asdict(self.settings), "grid": self.grid,
                   "q": {state_key: q_values.copy() for state_key, q_values in self.q_table.items()}}
        with resolved_path.open("wb") as output_file:
            pickle.dump(payload, output_file)
        return resolved_path

    @classmethod
    def load(cls, path: str | os.PathLike) -> "TabularAgentBase":
        """Đọc agent đã lưu; trả đúng lớp con theo trường 'algo'. Đọc được cả file lưu bằng tên trường cũ."""
        # Import để các lớp con tự đăng ký vào `registry` (không cần người gọi tự import).
        import agents.tabular.q_learning_agent  # noqa: F401

        with Path(path).open("rb") as input_file:
            payload = pickle.load(input_file)
        agent_class = TabularAgentBase.registry.get(payload["algo"])
        if agent_class is None:
            raise ValueError(f"Thuật toán '{payload['algo']}' trong {path} chưa được đăng ký.")
        saved_settings = {TabularSettings.LEGACY_FIELD_NAMES.get(name, name): value
                          for name, value in payload["config"].items()}
        saved_settings["checkpoint_episodes"] = tuple(saved_settings.get("checkpoint_episodes", ()))
        agent = agent_class(payload["grid"], TabularSettings(**saved_settings))
        agent.q_table = {state_key: np.array(q_values, dtype=np.float64)
                         for state_key, q_values in payload["q"].items()}
        return agent
