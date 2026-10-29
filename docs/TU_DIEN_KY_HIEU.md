# Từ điển ký hiệu, mã và viết tắt

Dùng để tra nhanh khi đọc code, comment, báo cáo. Tên hàm/biến trong code đã viết đầy đủ; những gì **không thể**
viết đầy đủ (ký hiệu toán học trong sách, mã tham chiếu tới kế hoạch, tên cột dữ liệu đã được báo cáo trích dẫn)
được giữ nguyên và giải nghĩa ở đây.

## 1. Mã quyết định thiết kế `D-xx`
Nguồn: bảng "Locked technical decisions" trong [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) mục 1.

| Mã | Ý nghĩa ngắn |
|---|---|
| D-01 | Bản đồ là mảng numpy (H, W) kiểu int8: 0 = sàn trống, 1 = vật cản; bản đồ cố định viết bằng chuỗi `.` và `#` |
| D-02 | Hành động: 0 = lên, 1 = xuống, 2 = trái, 3 = phải; hàng 0 là hàng trên cùng |
| D-03 | Reward = phạt bước + đúng một sự kiện (ô mới / đi lại / va chạm) + thưởng hoàn thành + (shaping) + (bonus ngây thơ) |
| D-04 | Đụng tường: robot đứng yên nhưng vẫn mất một bước và bị phạt |
| D-05 | Ô xuất phát tính là đã dọn, không nhận thưởng |
| D-06 | Bản đồ có dưới 2 ô trống thì báo lỗi |
| D-07 | `max_steps` mặc định = 4 × số ô trống; `truncated` chỉ bật khi chưa `terminated` |
| D-08 | `start=None` → ô xuất phát ngẫu nhiên đều; truyền tuple → cố định |
| D-09 | `coverage` = ô đã dọn / số ô trống, tính cả ô xuất phát |
| D-10 | `redundancy` = số bước đi vào ô đã dọn / tổng số bước (va chạm không tính ở tử số) |
| D-11 | `compute_transition()` là nguồn sự thật duy nhất về động lực học; env và Value Iteration cùng gọi nó |
| D-12 | Khóa trạng thái bảng Q = (hàng, cột, bitmask ô đã dọn); tối đa 62 ô trống |
| D-13 | E1: khóa chỉ gồm (hàng, cột) (`state_mode="position_only"`) |
| D-14 | Shaping thế năng: Φ(s) = −d(s)/(H+W), d = khoảng cách BFS tới ô chưa dọn gần nhất |
| D-15 | Reward "hackable": thưởng khi lại gần ô bẩn, không phạt khi đi xa → minh họa reward hacking |
| D-16 | Môi trường hoàn toàn tất định |
| D-17 | `RandomRoomVacuumCleaningEnv`: sinh phòng mới mỗi lần reset hoặc lấy từ nhóm K phòng cố định |
| D-18 | Mỗi mô hình DQN huấn luyện cho một kích thước lưới cố định |
| D-19 | Zigzag có vật cản: quét theo thứ tự rắn, mục tiêu = ô chưa dọn đầu tiên, đi theo đường BFS |
| D-20 | Phá hòa tất định = chọn hành động có chỉ số nhỏ nhất; khi huấn luyện ε-greedy phá hòa ngẫu nhiên |
| D-21 | Value Iteration cập nhật đồng bộ dạng vector (nhanh hơn vòng lặp hàng trăm lần) |
| D-22 | Số bước tối thiểu thật tính bằng BFS trên đồ thị trạng thái (`minimum_steps_to_clean_all`) |
| D-23 | Episode i của `evaluate_agent(seed=s)` dùng `env.reset(seed=s*10_000+i)` |
| D-24 | Agent có `reset()` thì bộ đánh giá gọi nó đầu mỗi episode |
| D-25 | Hỗ trợ Python 3.11–3.13 |
| D-26 | Vẽ khung hình bằng Pillow |
| D-27 | Phạm vi mô hình: Q-learning là chính, DQN là bản mở rộng, Value Iteration làm đáp án |

## 2. Mã tham chiếu tài liệu
| Mã | Nghĩa | Nơi tra |
|---|---|---|
| B2 | Định nghĩa MDP (state, action, reward...) | [PLAN_rel.md](PLAN_rel.md) mục B2 |
| B6 | Các chỉ số đánh giá (coverage, steps, redundancy...) | PLAN_rel.md mục B6 |
| B8 | Bảng nhật ký thay đổi thiết kế | PLAN_rel.md mục B8, [CHANGELOG_DESIGN.md](CHANGELOG_DESIGN.md) |
| C2.5 / C2.11 | Quy tắc làm việc: C2.5 comment tiếng Việt, tên tiếng Anh; C2.11 phân biệt `terminated` (không bootstrap) và `truncated` (vẫn bootstrap) | PLAN_rel.md mục C2 |
| C3 | Giao diện bắt buộc (env, agent, đánh giá) | PLAN_rel.md mục C3 |
| E1 | Thí nghiệm: trạng thái có / không có bản đồ ô đã dọn (tính Markov) | `run_markov_and_reward_ablation.py --exp markov` |
| E2 | Thí nghiệm: thiết kế reward (thưa / dày / shaping / hackable) | `run_markov_and_reward_ablation.py --exp reward` |
| E3 | Thí nghiệm: tăng mật độ vật cản | `run_generalization.py --exp density` |
| E4 | Thí nghiệm: khái quát hóa sang phòng chưa gặp | `run_generalization.py --exp generalization` |
| E5 | Q-learning vs SARSA vs Double Q (đã bỏ khỏi phạm vi) | PLAN_rel.md |
| E6 | Khoảng cách tới đáp án tối ưu của Value Iteration | `run_optimality_gap.py` |
| E7 | Bùng nổ trạng thái theo kích thước lưới | `run_state_explosion.py` |
| T5.3, T6.2, T8.4... | Mã thẻ nhiệm vụ trong kế hoạch triển khai | IMPLEMENTATION_PLAN.md phần 5 |
| S&B | Sutton & Barto (2018), *Reinforcement Learning: An Introduction*; "S&B 6.5" = mục 6.5 | giáo trình |
| Mnih 2015 | Mnih et al. (2015), bài báo DQN | — |

## 3. Ký hiệu toán học (giữ nguyên theo sách)
| Ký hiệu | Tên trong code | Nghĩa |
|---|---|---|
| s, s' | `state`, `next_state` | trạng thái hiện tại, trạng thái kế tiếp |
| a, A | `action` | hành động |
| r, R | `reward` | phần thưởng nhận được |
| γ (gamma) | `gamma` | hệ số chiết khấu (0, 1]; càng gần 1 càng coi trọng tương lai |
| α (alpha) | `alpha` | tốc độ học của Q-learning dạng bảng |
| ε (epsilon) | `epsilon` | xác suất chọn hành động ngẫu nhiên để khám phá |
| θ (theta) | `theta` | ngưỡng dừng của Value Iteration |
| Δ (delta) | `max_change` | mức thay đổi lớn nhất của V trong một vòng lặp |
| Q(s, a) | `q_values`, `action_values` | giá trị kỳ vọng khi làm a ở s rồi đi tiếp |
| V*(s), Q*, π* | `state_values`, `action_values`, `policy` | giá trị / Q / chính sách **tối ưu** |
| \|F\| | `free_cell_count` | số ô trống |
| Φ(s) | `potential_*` | hàm thế năng trong potential-based shaping |
| s0 | `start_*` | trạng thái xuất phát |

## 4. Viết tắt thuật ngữ
| Viết tắt | Đầy đủ |
|---|---|
| RL | Reinforcement Learning (học tăng cường) |
| MDP | Markov Decision Process |
| DQN | Deep Q-Network: Q-learning dùng mạng nơ-ron thay cho bảng Q |
| CNN | Convolutional Neural Network (mạng tích chập) |
| BFS | Breadth-First Search (tìm kiếm theo chiều rộng, ra đường ngắn nhất trên lưới) |
| VI | Value Iteration (trong tên thư mục/cột kết quả) |
| ε-greedy | chọn ngẫu nhiên với xác suất ε, còn lại chọn hành động tốt nhất |
| egocentric | quan sát lấy robot làm tâm ảnh |
| action masking | che các hành động đâm vào tường khi chọn hành động và khi tính mục tiêu |
| replay buffer | bộ nhớ kinh nghiệm để học trên lô ngẫu nhiên |
| target network | bản sao đóng băng của mạng Q dùng tính mục tiêu |
| std / sd | standard deviation, độ lệch chuẩn |
| avg | average, trung bình |
| GIF | ảnh động |

## 5. Tên cột CSV và khóa `info` (giữ nguyên vì báo cáo, slide trích dẫn)
| Tên | Nghĩa |
|---|---|
| `coverage` | tỷ lệ ô trống đã dọn (D-09) |
| `steps` | số bước đã đi trong episode |
| `steps_to_full` | số bước để dọn hết 100% (NaN nếu chưa dọn xong) |
| `redundancy` | tỷ lệ bước đi lại ô đã dọn (D-10) |
| `collisions` / `n_collisions` | số lần đụng tường/vật cản |
| `n_revisit` | số lần đi vào ô đã dọn |
| `n_cleaned`, `n_free` | số ô đã dọn, số ô trống |
| `return_undiscounted`, `return_discounted` | tổng thưởng không chiết khấu / có chiết khấu γ |
| `terminated`, `truncated` | dọn xong 100% / hết số bước tối đa (chuẩn Gymnasium) |
| `success_rate` | tỷ lệ episode dọn xong 100% |
| `gap_pct` | chênh lệch phần trăm so với số bước tối thiểu (E6) |
| `*_mean`, `*_std`, `*_avg`, `*_sd` | trung bình / độ lệch chuẩn (qua episode hoặc qua seed) |
| `q_start_max` | max_a Q(s0, a) tại trạng thái xuất phát |
| `n_states_visited` | số trạng thái đã gặp trong bảng Q |
| `wall_time_s` | thời gian thực tế đã chạy, tính bằng giây |
| `global_step` | số bước môi trường tích lũy khi huấn luyện DQN |
| `mean_q`, `mean_target`, `grad_norm`, `loss` | Q trung bình, mục tiêu trung bình, độ lớn gradient, hàm mất mát |
| `env_seed`, `start_row`, `start_col` | seed của env và ô xuất phát của episode |
| `map_id` | chỉ số bản đồ trong tập test |

## 6. Viết tắt trong tên thư mục kết quả `results/`
Xem bảng đầy đủ ở đầu [experiments/shared/result_folder_naming.py](../experiments/shared/result_folder_naming.py).
Ví dụ `rand8x8_d0.15_K0_mask_ego_ne16_bs256_tf16`: phòng ngẫu nhiên 8×8, mật độ vật cản 0.15, sinh phòng mới mỗi
episode (K = 0), có action masking, quan sát egocentric, 16 môi trường song song, batch 256, cập nhật mạng mỗi 16 bước.

## 7. Bảng đổi tên (báo cáo và slide dùng tên cũ)
| Tên cũ | Tên mới |
|---|---|
| `env/` → `env/maps.py` | `environment/room_layouts.py`, `environment/random_room_generator.py` |
| `env/grid_utils.py` | `environment/grid_movement.py`, `shortest_path_search.py`, `observation_decoding.py`, `egocentric_observation.py` (và `StateEncoder` → `agents/tabular/state_key_encoder.py`) |
| `env/vacuum_env.py` | `environment/vacuum_cleaning_env.py`, `random_room_vacuum_env.py`, `reward_settings.py`, `transition_model.py` |
| `env/render.py` | `environment/rendering/` (4 file) |
| `baselines/` | `agents/baselines/` |
| `tabular/` | `agents/tabular/` |
| `deep/` | `agents/deep_q_network/` |
| `utils/` | `project_utils/` |
| `experiments/evaluate.py` | `evaluation/` |
| `experiments/common.py` | `experiments/shared/` |
| `experiments/plots.py` | `experiments/plotting/` |
| `run_tabular`, `run_deep`, `run_ablation`, `run_optimality`, `make_demo`, `replot` | `run_tabular_q_learning`, `run_deep_q_network`, `run_markov_and_reward_ablation`, `run_optimality_gap`, `make_demo_gifs`, `replot_saved_results` |
| `VacuumEnv`, `RandomMapVacuumEnv`, `RewardConfig` | `VacuumCleaningEnv`, `RandomRoomVacuumCleaningEnv`, `RewardSettings` |
| `transition`, `TransitionResult` | `compute_transition`, `TransitionOutcome` |
| `TabularAgent`, `TabularConfig`, `StateEncoder` | `TabularAgentBase`, `TabularSettings`, `StateKeyEncoder` |
| `StateGraph`, `build_state_graph`, `min_steps_to_cover` | `StateTransitionGraph`, `build_state_transition_graph`, `minimum_steps_to_clean_all` |
| `VIResult`, `VIAgent`, `value_iteration`, `greedy_rollout` | `ValueIterationResult`, `ValueIterationPolicyAgent`, `run_value_iteration`, `follow_optimal_policy` |
| `DQNConfig`, `DQNAgent`, `QNetwork` | `DeepQNetworkSettings`, `DeepQNetworkAgent`, `QValueNetwork` |
| `GreedyBFSAgent`, `ZigzagAgent` | `GreedyNearestDirtyCellAgent`, `ZigzagSweepAgent` (khóa baseline `"greedy_bfs"`, `"zigzag"` giữ nguyên) |
| `run_episode`, `evaluate`, `evaluate_on_maps`, `summarize` | `run_single_episode`, `evaluate_agent`, `evaluate_agent_on_rooms`, `summarize_metrics` |
| `assert_inside_root`, `ROOT`, `save_csv`, `save_config_json` | `require_path_inside_project`, `PROJECT_ROOT`, `save_dataframe_csv`, `save_config_with_metadata` |
| tham số `cfg`, `obs`, `pos`, `n_envs`, `train_freq`... | `settings`, `observation`, `robot_position`, `parallel_env_count`, `steps_per_network_update`... (cờ dòng lệnh như `--n-envs`, `--train-freq` giữ nguyên) |

Các file mô hình `.pt` và `.pkl` đã lưu bằng tên tham số cũ vẫn đọc được: `DeepQNetworkSettings.LEGACY_FIELD_NAMES`
và `TabularSettings.LEGACY_FIELD_NAMES` ánh xạ tên cũ sang tên mới khi đọc.
