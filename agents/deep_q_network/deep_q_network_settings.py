"""Siêu tham số của DQN (Deep Q-Network = Q-learning dùng mạng nơ-ron thay cho bảng Q).

Bảng đối chiếu với Q-learning dạng bảng (Sutton & Barto 2018, mục 6.5), để giải thích khi vấn đáp:

| Q-learning dạng bảng (agents/tabular/q_learning_agent.py) | DQN (thư mục này) | Vì sao đổi |
|---|---|---|
| Q[S][A] tra trong dict | QValueNetwork(observation)[A] do CNN tính | Quá nhiều trạng thái cho một bảng; mạng khái quát sang trạng thái/phòng chưa gặp (S&B 9.1) |
| target = R + γ max_a Q(S', a) | cùng công thức, max lấy trên BẢN SAO ĐÓNG BĂNG của mạng (target network) | Mục tiêu thay đổi mỗi bước làm học bất ổn (deadly triad, S&B 11.3; Mnih 2015) |
| cập nhật ngay sau mỗi bước | lưu bước vào replay buffer, học trên lô ngẫu nhiên | Các bước liên tiếp tương quan mạnh; lô ngẫu nhiên phá tương quan (Mnih 2015) |
| Q[S][A] += α (target − Q[S][A]) | một bước gradient trên Huber loss(Q(S,A), target) | Cùng ý "kéo ước lượng về phía mục tiêu"; α trở thành learning rate |
| ε-greedy, ε giảm tuyến tính | ε-greedy, ε giảm tuyến tính | Giống hệt |
| terminated -> không bootstrap | nhân (1 − terminated) | Giống hệt (C2.11) |
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DeepQNetworkSettings:
    """Siêu tham số DQN. gamma = γ (hệ số chiết khấu), epsilon = ε (xác suất khám phá ngẫu nhiên)."""

    total_timesteps: int = 200_000  # tổng số bước tương tác với môi trường
    # Ba giá trị dưới được chọn bằng thử nghiệm trên obstacle_4x4 (results/tuning/): cấu hình này
    # không sụp giữa chừng như lr 5e-4 / target 1000 / buffer 50k.
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    replay_buffer_capacity: int = 100_000
    batch_size: int = 64
    random_warmup_steps: int = 2_000  # số bước đi ngẫu nhiên để lấp replay buffer trước khi học
    steps_per_network_update: int = 4  # mỗi 4 bước môi trường cập nhật mạng một lần (Mnih 2015)
    target_network_sync_interval_steps: int = 500  # chép trọng số mạng online -> mạng target sau mỗi N bước
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_fraction: float = 0.3  # ε giảm tuyến tính trong 30% tổng số bước
    max_gradient_norm: float = 10.0  # cắt gradient để học ổn định
    loss_function: str = "huber"  # "huber" | "mse"
    reward_scale: float = 1.0  # nhân reward trước khi lưu vào buffer (chỉ khi cần ổn định)
    checkpoint_steps: tuple[int, ...] = field(default_factory=tuple)  # lưu mô hình cho demo "robot học dần"
    evaluate_every_steps: int = 5_000
    evaluation_episode_count: int = 20
    log_every_steps: int = 500
    device: str = "auto"  # "auto" = GPU (CUDA) nếu có, ngược lại CPU; hoặc "cpu" / "cuda"
    # Cải tiến v3 (B2 "action masking"): bỏ các hướng đâm vào tường/vật cản khi chọn hành động và khi
    # lấy max trong mục tiêu. Va chạm không bao giờ có ích (chỉ bị phạt và đứng yên), nên che chúng đi
    # không làm mất chính sách tối ưu mà loại được vòng lặp "đâm tường mãi" của chính sách tất định.
    use_action_mask: bool = False
    # Cải tiến v4: quan sát lấy robot làm tâm (xem environment/egocentric_observation.py) để khái quát sang phòng lạ.
    use_egocentric_observation: bool = False
    # Số môi trường chạy song song trong một tiến trình: chọn hành động theo lô (nhanh hơn nhiều trên GPU).
    # Các mốc (cập nhật mạng, target, đánh giá, checkpoint) vẫn tính theo từng bước môi trường.
    parallel_env_count: int = 1
    # Chọn mô hình theo tập kiểm định: giữ trọng số có kết quả đánh giá định kỳ tốt nhất (đánh giá trên
    # phòng ngẫu nhiên tách biệt tập test) thay vì trọng số ở bước cuối, vốn có thể vừa dao động xuống.
    keep_best_checkpoint: bool = False
    seed: int = 0

    # Tên trường cũ (đã lưu trong các file .pt) -> tên mới, dùng khi đọc lại agent đã lưu.
    LEGACY_FIELD_NAMES = {
        "buffer_size": "replay_buffer_capacity",
        "learning_starts": "random_warmup_steps",
        "train_freq": "steps_per_network_update",
        "target_update_interval": "target_network_sync_interval_steps",
        "max_grad_norm": "max_gradient_norm",
        "loss": "loss_function",
        "eval_every_steps": "evaluate_every_steps",
        "eval_episodes": "evaluation_episode_count",
        "action_mask": "use_action_mask",
        "egocentric": "use_egocentric_observation",
        "n_envs": "parallel_env_count",
        "keep_best": "keep_best_checkpoint",
    }

    def __post_init__(self) -> None:
        if self.total_timesteps < 1:
            raise ValueError("total_timesteps phải >= 1.")
        if self.batch_size < 1 or self.replay_buffer_capacity < self.batch_size:
            raise ValueError("Cần 1 <= batch_size <= buffer_size.")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma phải trong (0, 1].")
        if self.loss_function not in ("huber", "mse"):
            raise ValueError("loss phải là 'huber' hoặc 'mse'.")
        periods = (self.steps_per_network_update, self.target_network_sync_interval_steps,
                   self.evaluate_every_steps, self.log_every_steps, self.evaluation_episode_count)
        if min(periods) < 1:
            raise ValueError("Các chu kỳ (train_freq, target_update_interval, eval/log) phải >= 1.")
        if not 0.0 < self.epsilon_decay_fraction <= 1.0:
            raise ValueError("epsilon_decay_fraction phải trong (0, 1].")
        if self.parallel_env_count < 1:
            raise ValueError("n_envs phải >= 1.")
        if self.random_warmup_steps < self.batch_size:
            raise ValueError("learning_starts phải >= batch_size (buffer cần đủ mẫu cho lô đầu tiên).")
        self.checkpoint_steps = tuple(int(step) for step in self.checkpoint_steps)


def linear_epsilon_for_step(global_step: int, settings: DeepQNetworkSettings) -> float:
    """ε tại bước `global_step` (đếm từ 0): giảm tuyến tính rồi giữ ở epsilon_end (giống bản tabular)."""
    decay_step_count = max(1, int(settings.epsilon_decay_fraction * settings.total_timesteps))
    decay_progress = min(1.0, global_step / decay_step_count)
    return settings.epsilon_start + decay_progress * (settings.epsilon_end - settings.epsilon_start)
