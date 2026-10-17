"""Phần dùng chung cho agent dạng bảng: cấu hình, lịch ε, bảng Q, vòng huấn luyện, lưu/đọc.

Lớp con (QLearningAgent) chỉ cài đặt `_run_episode` — đúng phần pseudocode trong sách —
còn mọi thứ khác (ghi log, đánh giá định kỳ, lưu checkpoint) nằm ở đây để code thuật toán
gọn và dễ đối chiếu với Sutton & Barto.
"""

from __future__ import annotations

import dataclasses
import os
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar

import gymnasium
import numpy as np
import pandas as pd

from env.grid_utils import StateEncoder
from experiments.evaluate import evaluate, summarize
from utils.io import save_csv
from utils.paths import assert_inside_root

N_ACTIONS: int = 4
TIE_TOLERANCE: float = 1e-9


@dataclass
class TabularConfig:
    """Siêu tham số cho agent dạng bảng."""

    n_episodes: int = 20_000
    alpha: float = 0.5  # môi trường xác định nên α lớn vẫn ổn (mục 6.5 cho phép α bất kỳ trong (0, 1])
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_fraction: float = 0.8  # ε giảm tuyến tính trong 80% số episode đầu, sau đó giữ nguyên
    q_init: float = 0.0
    state_mode: str = "full"  # "full" | "position_only" (E1)
    eval_every: int = 100  # cứ N episode đánh giá chính sách tham lam một lần
    eval_episodes: int = 5
    time_budget_min: float | None = None  # dừng sớm khi vượt (nhiệm vụ 5.3)
    checkpoint_episodes: tuple[int, ...] = field(default_factory=tuple)  # lưu bảng Q cho demo "robot học dần"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_episodes < 1:
            raise ValueError(f"n_episodes phải >= 1, nhận được {self.n_episodes}.")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"alpha phải trong (0, 1], nhận được {self.alpha}.")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError(f"gamma phải trong (0, 1], nhận được {self.gamma}.")
        if not (0.0 <= self.epsilon_end <= 1.0 and 0.0 <= self.epsilon_start <= 1.0):
            raise ValueError("epsilon_start và epsilon_end phải trong [0, 1].")
        if not 0.0 < self.epsilon_decay_fraction <= 1.0:
            raise ValueError("epsilon_decay_fraction phải trong (0, 1].")
        if self.eval_every < 1 or self.eval_episodes < 1:
            raise ValueError("eval_every và eval_episodes phải >= 1.")
        if self.state_mode not in StateEncoder.STATE_MODES:
            raise ValueError(f"state_mode phải là một trong {StateEncoder.STATE_MODES}.")
        self.checkpoint_episodes = tuple(int(e) for e in self.checkpoint_episodes)


def linear_epsilon(episode: int, cfg: TabularConfig) -> float:
    """ε cho episode thứ `episode` (đếm từ 0): giảm tuyến tính rồi giữ ở epsilon_end.

    Vì sao: đầu quá trình cần khám phá nhiều (ε lớn); về sau khai thác cái đã học (ε nhỏ).
    Giữ ε > 0 đến cuối để mọi cặp (s, a) vẫn tiếp tục được thăm — điều kiện hội tụ của
    Q-learning (Sutton & Barto 2018, mục 6.5).
    """
    decay_episodes = max(1, int(cfg.epsilon_decay_fraction * cfg.n_episodes))
    frac = min(1.0, episode / decay_episodes)
    return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)


class TabularAgent(ABC):
    """Lớp cơ sở (trừu tượng) cho agent bảng Q. Lớp con đặt `algo_name` và cài `_run_episode`."""

    algo_name: ClassVar[str] = "tabular"
    registry: ClassVar[dict[str, type["TabularAgent"]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        TabularAgent.registry[cls.algo_name] = cls

    def __init__(self, grid: np.ndarray, cfg: TabularConfig) -> None:
        self.grid = np.array(grid, dtype=np.int8, copy=True)
        self.cfg = cfg
        self.encoder = StateEncoder(self.grid, cfg.state_mode)
        self.rng = np.random.default_rng(cfg.seed)
        self.q: dict[tuple, np.ndarray] = {}

    # ------------------------------------------------------------------ bảng Q
    def q_values(self, state_key: tuple) -> np.ndarray:
        """Đọc Q(s, ·) (không thêm trạng thái mới vào bảng); trạng thái chưa gặp trả q_init."""
        row = self.q.get(state_key)
        return row if row is not None else np.full(N_ACTIONS, self.cfg.q_init)

    def _q_row(self, state_key: tuple) -> np.ndarray:
        """Hàng Q(s, ·) để GHI (tạo mới với q_init nếu chưa có)."""
        row = self.q.get(state_key)
        if row is None:
            row = np.full(N_ACTIONS, self.cfg.q_init, dtype=np.float64)
            self.q[state_key] = row
        return row

    def epsilon_greedy(self, state_key: tuple, epsilon: float) -> int:
        """ε-greedy (Sutton & Barto 2018, mục 2.2): xác suất ε chọn ngẫu nhiên, còn lại chọn tốt nhất.

        Khi nhiều hành động cùng tốt nhất thì chọn ngẫu nhiên trong số đó (D-20), để agent
        không bị thiên vị hành động "lên" lúc bảng Q còn toàn số 0.
        """
        if self.rng.random() < epsilon:
            return int(self.rng.integers(N_ACTIONS))
        q = self.q_values(state_key)
        best = np.flatnonzero(q >= q.max() - TIE_TOLERANCE)
        return int(self.rng.choice(best))

    def greedy_action(self, state_key: tuple) -> int:
        """Hành động tham lam tất định (phá hòa chọn chỉ số nhỏ nhất), dùng khi đánh giá."""
        q = self.q_values(state_key)
        return int(np.argmax(q >= q.max() - TIE_TOLERANCE))

    def act(self, obs: np.ndarray, info: dict | None = None) -> int:
        """Giao diện Agent: chính sách tham lam theo bảng Q."""
        return self.greedy_action(self.encoder.encode_obs(obs))

    def start_value(self, state_key: tuple) -> float:
        """max_a Q(s0, a) — so với V*(s0) để thấy ước lượng hội tụ về giá trị thật."""
        return float(self.q_values(state_key).max())

    # ------------------------------------------------------------------ thuật toán (lớp con)
    @abstractmethod
    def _run_episode(self, env: gymnasium.Env, epsilon: float, seed: int | None) -> dict[str, Any]:
        """Chạy một episode và cập nhật bảng Q. Trả return, coverage, steps, terminated, truncated, start_key."""

    # ------------------------------------------------------------------ vòng huấn luyện
    def train(self, env_factory: Callable[[], gymnasium.Env], log_dir: str | os.PathLike | None = None
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Huấn luyện n_episodes episode, đánh giá tham lam định kỳ, lưu checkpoint.

        Returns:
            (train_log, eval_log). Ghi CSV vào log_dir nếu có.
        """
        cfg = self.cfg
        env = env_factory()
        fixed_start = getattr(env.unwrapped, "fixed_start", None)
        log_path = assert_inside_root(log_dir) if log_dir is not None else None
        if log_path is not None and 0 in cfg.checkpoint_episodes:
            self.save(log_path / "checkpoints" / "ep0.pkl")
        t0 = time.monotonic()
        train_rows: list[dict[str, Any]] = []
        eval_rows: list[dict[str, Any]] = []
        for episode in range(1, cfg.n_episodes + 1):
            epsilon = linear_epsilon(episode - 1, cfg)
            # Chỉ seed ở episode đầu; các episode sau dùng tiếp bộ sinh số của env -> vẫn tái lập được.
            stats = self._run_episode(env, epsilon, seed=cfg.seed if episode == 1 else None)
            train_rows.append({
                "episode": episode,
                "epsilon": epsilon,
                "return_undiscounted": stats["return"],
                "coverage": stats["coverage"],
                "steps": stats["steps"],
                "terminated": stats["terminated"],
                "truncated": stats["truncated"],
                "n_states_visited": len(self.q),
                "q_start_max": self.start_value(stats["start_key"]) if fixed_start is not None else float("nan"),
                "wall_time_s": time.monotonic() - t0,
            })
            out_of_time = cfg.time_budget_min is not None and time.monotonic() - t0 > 60 * cfg.time_budget_min
            if episode % cfg.eval_every == 0 or episode == cfg.n_episodes or out_of_time:
                summary = summarize(evaluate(self, env_factory, cfg.eval_episodes, seed=cfg.seed, gamma=cfg.gamma))
                eval_rows.append({"episode": episode, "n_states_visited": len(self.q),
                                  **summary.iloc[0].to_dict()})
            if log_path is not None and episode in cfg.checkpoint_episodes:
                self.save(log_path / "checkpoints" / f"ep{episode}.pkl")
            if out_of_time:
                break
        train_log, eval_log = pd.DataFrame(train_rows), pd.DataFrame(eval_rows)
        if log_path is not None:
            self.save(log_path / "checkpoints" / "final.pkl")
            save_csv(train_log, log_path / f"seed{cfg.seed}_train.csv")
            save_csv(eval_log, log_path / f"seed{cfg.seed}_eval.csv")
        return train_log, eval_log

    # ------------------------------------------------------------------ lưu / đọc
    def save(self, path: str | os.PathLike) -> Path:
        """Lưu bảng Q, cấu hình và bản đồ (pickle; bảng Q nhỏ nên file nhỏ)."""
        resolved = assert_inside_root(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        payload = {"algo": self.algo_name, "config": dataclasses.asdict(self.cfg), "grid": self.grid,
                   "q": {k: v.copy() for k, v in self.q.items()}}
        with resolved.open("wb") as f:
            pickle.dump(payload, f)
        return resolved

    @classmethod
    def load(cls, path: str | os.PathLike) -> "TabularAgent":
        """Đọc agent đã lưu; trả đúng lớp con theo trường 'algo' trong file."""
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        agent_cls = TabularAgent.registry.get(payload["algo"])
        if agent_cls is None:
            raise ValueError(f"Thuật toán '{payload['algo']}' trong {path} chưa được đăng ký.")
        config = dict(payload["config"])
        config["checkpoint_episodes"] = tuple(config.get("checkpoint_episodes", ()))
        agent = agent_cls(payload["grid"], TabularConfig(**config))
        agent.q = {k: np.array(v, dtype=np.float64) for k, v in payload["q"].items()}
        return agent
