# Cấu trúc thư mục

Mỗi file Python chỉ làm **một việc**; dòng mô tả bên dưới lấy từ docstring đầu file. Tên viết tắt và mã (D-xx, E1..E7, B2...) được giải nghĩa trong [TU_DIEN_KY_HIEU.md](TU_DIEN_KY_HIEU.md).

Phân tầng: `project_utils/` và `environment/` là tầng thấp (không phụ thuộc agent); `agents/` và `evaluation/` dùng lẫn nhau (agent gọi bộ đánh giá khi huấn luyện, `evaluation/optimal_reference.py` dùng agent Value Iteration và baseline); `experiments/` là tầng cao nhất. Hai chỗ import ngược chiều được cố ý để muộn (lazy import trong hàm) để tránh vòng lặp import: `environment/rendering/gif_recording.py` (demo robot đi ngẫu nhiên) và `evaluation/optimal_reference.py`.

Ngoài code: `docs/` (kế hoạch, nhật ký thiết kế, báo cáo HTML), `results/` (CSV, hình, GIF, mô hình đã huấn luyện), `slides/` (bộ slide), `scripts/memory_guard.ps1` (canh bộ nhớ khi chạy thí nghiệm dài).

## environment/ — căn phòng, luật chơi và hiển thị (cũ: env/)

- `environment/egocentric_observation.py` — Quan sát lấy robot làm tâm (egocentric = "nhìn từ góc của chính robot").
- `environment/grid_movement.py` — Quy ước hướng đi trên lưới và kiểm tra ô có đi vào được không.
- `environment/observation_decoding.py` — Giải mã observation (3, H, W) của môi trường thành vị trí robot, ô đã dọn và vật cản.
- `environment/random_room_generator.py` — Sinh căn phòng ngẫu nhiên luôn dọn được hết (các ô trống liên thông với nhau).
- `environment/random_room_vacuum_env.py` — Biến thể môi trường sinh căn phòng mới mỗi lần reset (D-17), dùng cho E3, E4 và demo phòng lạ.
- `environment/rendering/frame_drawing.py` — Vẽ trạng thái môi trường thành ảnh RGB hoặc chuỗi ký tự, và thêm chú thích lên ảnh (D-26: dùng Pillow).
- `environment/rendering/frame_layout.py` — Ghép nhiều khung hình thành một ảnh (đặt cạnh nhau hoặc xếp dọc) để so sánh các agent.
- `environment/rendering/gif_recording.py` — Ghi GIF từ danh sách khung hình, hoặc chạy một episode và ghi lại thành GIF.
- `environment/rendering/visit_heatmap.py` — Vẽ bản đồ nhiệt số lần robot đi qua từng ô.
- `environment/reward_settings.py` — Tham số phần thưởng của môi trường và các bộ cấu hình dựng sẵn (preset).
- `environment/room_layouts.py` — Bản đồ căn phòng cố định và cách đọc bản đồ từ chuỗi.
- `environment/shortest_path_search.py` — Tìm đường ngắn nhất trên lưới bằng BFS (Breadth-First Search: tìm kiếm theo chiều rộng).
- `environment/transition_model.py` — Mô hình MDP xác định: một hành động đưa robot sang trạng thái nào và nhận bao nhiêu thưởng.
- `environment/vacuum_cleaning_env.py` — Môi trường robot hút bụi dạng lưới theo chuẩn Gymnasium.

## agents/ — các agent: baseline, Q-learning dạng bảng, Value Iteration, DQN (cũ: baselines/, tabular/, deep/)

- `agents/agent_interface.py` — Giao diện chung cho mọi agent (baseline và RL), theo C3 của PLAN_rel.md.
- `agents/baselines/baseline_registry.py` — Danh mục baseline: tạo agent theo tên.
- `agents/baselines/greedy_nearest_dirty_cell_agent.py` — Baseline tham lam: luôn đi tới ô chưa dọn gần nhất.
- `agents/baselines/random_walk_agent.py` — Baseline đi ngẫu nhiên.
- `agents/baselines/zigzag_sweep_agent.py` — Baseline quét zigzag (boustrophedon) — cách máy cắt cỏ / robot rẻ tiền hay dùng.
- `agents/deep_q_network/bellman_targets.py` — Mục tiêu Bellman y = r + γ (1 − terminated) max_a Q_target(s', a) cho một lô transition.
- `agents/deep_q_network/compute_device.py` — Chọn thiết bị tính toán (CPU hoặc GPU) cho torch.
- `agents/deep_q_network/deep_q_network_agent.py` — Agent DQN: chọn hành động và làm một bước học. Vòng huấn luyện nằm ở training_loop.py.
- `agents/deep_q_network/deep_q_network_settings.py` — Siêu tham số của DQN (Deep Q-Network = Q-learning dùng mạng nơ-ron thay cho bảng Q).
- `agents/deep_q_network/model_checkpoint.py` — Lưu và đọc agent DQN (file .pt).
- `agents/deep_q_network/q_value_network.py` — Mạng nơ-ron xấp xỉ Q(s, ·) cho DQN.
- `agents/deep_q_network/replay_buffer.py` — Replay buffer (bộ nhớ kinh nghiệm) cho DQN (Mnih et al. 2015).
- `agents/deep_q_network/training_loop.py` — Vòng huấn luyện DQN theo Algorithm 1 của Mnih et al. (2015).
- `agents/deep_q_network/wall_action_masking.py` — Action masking: che các hành động đâm vào tường/vật cản (cải tiến v3).
- `agents/tabular/q_learning_agent.py` — Q-learning dạng bảng — model chính của dự án.
- `agents/tabular/reachable_state_counter.py` — Đếm số trạng thái đạt tới được của MDP (thí nghiệm E7: bùng nổ trạng thái).
- `agents/tabular/state_key_encoder.py` — Mã hóa trạng thái thành khóa hashable (dùng làm khóa của bảng Q).
- `agents/tabular/state_transition_graph.py` — Đồ thị trạng thái của MDP trên một bản đồ cố định, và số bước tối thiểu để dọn hết.
- `agents/tabular/tabular_agent_base.py` — Phần dùng chung cho agent dạng bảng: bảng Q, chọn hành động, vòng huấn luyện, lưu/đọc.
- `agents/tabular/tabular_settings.py` — Siêu tham số cho agent dạng bảng và lịch giảm ε.
- `agents/tabular/value_iteration.py` — Value Iteration (Sutton & Barto 2018, mục 4.4) — đáp án tối ưu chính xác trên lưới nhỏ.

## evaluation/ — chạy episode, tính và tổng hợp chỉ số, đáp án tối ưu để so sánh (cũ: experiments/evaluate.py)

- `evaluation/agent_evaluation.py` — Bộ đánh giá chung cho mọi agent (giao diện C3).
- `evaluation/episode_runner.py` — Chạy đúng một episode với một agent và tính các chỉ số của episode đó (chỉ số B6).
- `evaluation/metric_summary.py` — Cột chỉ số của mỗi episode và cách tổng hợp thành trung bình ± độ lệch chuẩn (chỉ số B6).
- `evaluation/optimal_reference.py` — Đáp án tối ưu chính xác (tính bằng đồ thị trạng thái) để chấm các agent, chỉ cho bản đồ nhỏ.

## experiments/ — các script chạy thí nghiệm và vẽ hình

- `experiments/make_demo_gifs.py` — T8.4: tư liệu trình diễn — robot hút bụi chạy trong mô phỏng, dùng các mô hình đã huấn luyện.
- `experiments/plotting/comparison_plots.py` — Biểu đồ so sánh có thanh sai số: cột (so sánh agent) và đường (theo một biến số).
- `experiments/plotting/learning_curve_plots.py` — Learning curve (đường cong học): trung bình ± độ lệch chuẩn (std) qua các seed.
- `experiments/plotting/markdown_tables.py` — Ghi bảng markdown và văn bản ra file (để dán vào README/slide).
- `experiments/plotting/plot_style.py` — Phong cách chung cho mọi hình: màu theo agent, kích thước, định dạng trục, lưu file.
- `experiments/plotting/tabular_scaling_plots.py` — Hình cho thí nghiệm bùng nổ trạng thái (E7) và giới hạn của Q-learning dạng bảng (T5.3).
- `experiments/plotting/visit_heatmap_plots.py` — Heatmap số lần đi qua mỗi ô của nhiều agent, đặt cạnh nhau để so sánh mức lãng phí.
- `experiments/replot_saved_results.py` — Vẽ lại các learning curve từ CSV đã lưu trong results/ (không huấn luyện lại).
- `experiments/reproduce_all.py` — Chạy lại toàn bộ thí nghiệm theo đúng thứ tự, dừng ngay ở bước đầu tiên bị lỗi.
- `experiments/run_baselines.py` — T3.2: chạy các baseline trên các bản đồ cố định, ghi CSV và bảng tóm tắt.
- `experiments/run_deep_q_network.py` — T6.2: huấn luyện DQN (Deep Q-learning) trên bản đồ cố định hoặc bản đồ ngẫu nhiên.
- `experiments/run_generalization.py` — T7.2 (E4) và T7.3 (E3): DQN trên phòng ngẫu nhiên, so với baseline và kiểm tra khái quát hóa.
- `experiments/run_markov_and_reward_ablation.py` — T5.2 (E1) và T7.1 (E2): thí nghiệm thay đổi một yếu tố thiết kế, giữ nguyên mọi thứ khác.
- `experiments/run_optimality_gap.py` — T7.3 (E6): khoảng cách tới tối ưu của từng phương pháp trên lưới nhỏ (3x3, 4x4).
- `experiments/run_state_explosion.py` — T5.1 (E7) + T5.3: giới hạn của Q-learning dạng bảng.
- `experiments/run_tabular_q_learning.py` — T4.4: huấn luyện Q-learning dạng bảng trên nhiều seed, vẽ learning curve và hội tụ về V*.
- `experiments/run_value_iteration.py` — T3.3: tính đáp án tối ưu bằng Value Iteration và so với baseline từ mọi ô xuất phát.
- `experiments/shared/command_line_options.py` — Tham số dòng lệnh dùng chung cho các script thí nghiệm.
- `experiments/shared/console_setup.py` — Chuẩn bị môi trường tiến trình cho script thí nghiệm.
- `experiments/shared/environment_factories.py` — Tạo môi trường cho thí nghiệm. Hàm tạo phải pickle được để gửi sang tiến trình con.
- `experiments/shared/evaluation_seeds.py` — Seed cố định để tập bản đồ train/test giống nhau giữa mọi script (so sánh công bằng).
- `experiments/shared/parallel_execution.py` — Chạy nhiều tác vụ (mỗi seed/cấu hình một tác vụ) tuần tự hoặc song song bằng nhiều tiến trình.
- `experiments/shared/result_folder_naming.py` — Quy ước đặt tên thư mục kết quả trong results/.

## project_utils/ — đường dẫn, seed, ghi file kết quả (cũ: utils/)

- `project_utils/project_paths.py` — Đường dẫn gốc của dự án và thiết lập cache cục bộ.
- `project_utils/random_seeding.py` — Đặt seed cho các nguồn ngẫu nhiên để thí nghiệm chạy lại ra đúng kết quả.
- `project_utils/result_files.py` — Ghi kết quả thí nghiệm (CSV, JSON cấu hình) vào thư mục dự án.

## tests/ — kiểm thử, chia thư mục theo package đang được kiểm thử

- `tests/agents/baselines/test_baseline_agents.py` — Test cho các baseline.
- `tests/agents/deep_q_network/test_deep_q_network.py` — Test cho deep/: mạng, replay buffer, DQN.
- `tests/agents/tabular/test_q_learning_agent.py` — Test cho Q-learning dạng bảng (so khớp với Value Iteration).
- `tests/agents/tabular/test_state_key_encoder.py` — Test cho agents/tabular/state_key_encoder.py.
- `tests/agents/tabular/test_value_iteration.py` — Test cho tabular/state_space.py và tabular/value_iteration.py (đáp án tối ưu).
- `tests/agents/test_saved_model_compatibility.py` — Các mô hình đã huấn luyện và commit trong results/ (lưu bằng tên siêu tham số cũ) vẫn phải đọc được.
- `tests/conftest.py` — Cấu hình chung cho pytest: cache cục bộ và các fixture bản đồ/môi trường.
- `tests/environment/test_egocentric_observation.py` — Test cho environment/egocentric_observation.py.
- `tests/environment/test_rendering.py` — Test cho env/render.py.
- `tests/environment/test_room_layouts_and_generator.py` — Test cho env/maps.py.
- `tests/environment/test_shortest_path_and_observation_decoding.py` — Test cho environment/shortest_path_search.py và observation_decoding.py.
- `tests/environment/test_vacuum_cleaning_env.py` — Test cho env/vacuum_env.py (bảng 4.2.5 của kế hoạch).
- `tests/evaluation/test_agent_evaluation.py` — Test cho experiments/evaluate.py.
- `tests/experiments/test_pipeline.py` — Chạy mọi script thí nghiệm ở chế độ --quick từ đầu đến cuối (quy tắc 0.3.6: luồng đầy đủ).
- `tests/experiments/test_plotting.py` — Test cho experiments/plots.py: các hàm vẽ tạo file, đọc nhiều seed, bảng markdown.
- `tests/experiments/test_result_folder_naming.py` — Tên thư mục kết quả phải giữ đúng quy ước cũ: báo cáo, slide và README trích dẫn các tên này.
- `tests/project_utils/test_project_setup.py` — Kiểm tra môi trường cài đặt: các thư viện import được và PROJECT_ROOT đúng.
