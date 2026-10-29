"""Siêu tham số cho agent dạng bảng và lịch giảm ε."""

from __future__ import annotations

from dataclasses import dataclass, field

from agents.tabular.state_key_encoder import StateKeyEncoder


@dataclass
class TabularSettings:
    """Siêu tham số cho agent dạng bảng.

    Ký hiệu trong sách Sutton & Barto: alpha = α (tốc độ học), gamma = γ (hệ số chiết khấu),
    epsilon = ε (xác suất khám phá ngẫu nhiên).
    """

    episode_count: int = 20_000
    alpha: float = 0.5  # môi trường xác định nên α lớn vẫn ổn (mục 6.5 cho phép α bất kỳ trong (0, 1])
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_fraction: float = 0.8  # ε giảm tuyến tính trong 80% số episode đầu, sau đó giữ nguyên
    initial_q_value: float = 0.0  # giá trị Q ban đầu của mọi cặp (trạng thái, hành động)
    state_mode: str = "full"  # "full" | "position_only" (E1)
    evaluate_every_episodes: int = 100  # cứ N episode đánh giá chính sách tham lam một lần
    evaluation_episode_count: int = 5
    time_budget_minutes: float | None = None  # dừng sớm khi vượt (nhiệm vụ 5.3)
    # Các episode cần lưu bảng Q để làm demo "robot học dần".
    checkpoint_episodes: tuple[int, ...] = field(default_factory=tuple)
    seed: int = 0

    # Tên trường cũ (đã lưu trong các file .pkl) -> tên mới, dùng khi đọc lại agent đã lưu.
    LEGACY_FIELD_NAMES = {
        "n_episodes": "episode_count",
        "q_init": "initial_q_value",
        "eval_every": "evaluate_every_episodes",
        "eval_episodes": "evaluation_episode_count",
        "time_budget_min": "time_budget_minutes",
    }

    def __post_init__(self) -> None:
        if self.episode_count < 1:
            raise ValueError(f"n_episodes phải >= 1, nhận được {self.episode_count}.")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"alpha phải trong (0, 1], nhận được {self.alpha}.")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError(f"gamma phải trong (0, 1], nhận được {self.gamma}.")
        if not (0.0 <= self.epsilon_end <= 1.0 and 0.0 <= self.epsilon_start <= 1.0):
            raise ValueError("epsilon_start và epsilon_end phải trong [0, 1].")
        if not 0.0 < self.epsilon_decay_fraction <= 1.0:
            raise ValueError("epsilon_decay_fraction phải trong (0, 1].")
        if self.evaluate_every_episodes < 1 or self.evaluation_episode_count < 1:
            raise ValueError("eval_every và eval_episodes phải >= 1.")
        if self.state_mode not in StateKeyEncoder.STATE_MODES:
            raise ValueError(f"state_mode phải là một trong {StateKeyEncoder.STATE_MODES}.")
        self.checkpoint_episodes = tuple(int(episode) for episode in self.checkpoint_episodes)


def linear_epsilon_for_episode(episode_index: int, settings: TabularSettings) -> float:
    """ε cho episode thứ `episode_index` (đếm từ 0): giảm tuyến tính rồi giữ ở epsilon_end.

    Vì sao: đầu quá trình cần khám phá nhiều (ε lớn); về sau khai thác cái đã học (ε nhỏ).
    Giữ ε > 0 đến cuối để mọi cặp (s, a) vẫn tiếp tục được thăm — điều kiện hội tụ của
    Q-learning (Sutton & Barto 2018, mục 6.5).
    """
    decay_episode_count = max(1, int(settings.epsilon_decay_fraction * settings.episode_count))
    decay_progress = min(1.0, episode_index / decay_episode_count)
    return settings.epsilon_start + decay_progress * (settings.epsilon_end - settings.epsilon_start)
