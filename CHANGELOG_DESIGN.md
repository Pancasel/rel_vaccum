# CHANGELOG_DESIGN.md

Nhật ký thay đổi thiết kế so với `PLAN_rel.md` / `IMPLEMENTATION_PLAN.md`. Mỗi mục có đề xuất dòng cho bảng B8 của `PLAN_rel.md` (nhóm tự chép sang).

## 2026-10-17 — T1.2: chốt phạm vi model

- Change: Q-learning là model chính; DQN được trình bày là "Q-learning dùng mạng nơ-ron" để mở rộng lên phòng lớn và phòng lạ. Value Iteration là đáp án tối ưu, 3 baseline (random, zigzag, greedy BFS) để so sánh. SARSA và cửa sổ demo tương tác là tùy chọn. Bỏ: Double Q-learning + E5, Double DQN, PPO, curriculum, tùy chọn nhiễu reward, wrapper đệm bản đồ.
- Reason: tập trung vào một câu chuyện dễ giải thích khi vấn đáp; các thuật toán bị bỏ thêm khái niệm mới mà không cần cho câu hỏi chính của dự án.
- Evidence/result: IMPLEMENTATION_PLAN.md mục 1.1.
- Proposed B8 row: | v1 | 2026-10-17 | Chốt model: Q-learning (chính) + DQN (mở rộng); bỏ Double Q, PPO, Double DQN, curriculum, E5 | Tập trung, dễ giải thích | — |

## 2026-10-17 — T1.2: môi trường Python

- Change: venv cô lập trong `.venv/` (không dùng `--system-site-packages`).
- Reason: thử dùng chung thư viện với Anaconda thì pytest tự nạp plugin `deepeval` bị lỗi import; venv cô lập tránh được và đúng quy tắc 0.1.
- Evidence/result: `python -m pytest` chạy sạch trong venv cô lập.
- Proposed B8 row: không cần (chỉ là môi trường chạy).

## 2026-10-17 — T2.2 / T2.1: bổ sung nhỏ cho bản đồ và môi trường

- Change: `generate_map_set(..., exclude=...)` và `RandomMapVacuumEnv(..., exclude_maps=...)` để tập train không trùng tập test; `map_to_string` để in bản đồ; thuộc tính `VacuumEnv.fixed_start`.
- Reason: E3/E4 yêu cầu bản đồ train và test tách rời; Q-learning cần biết start có cố định không để ghi `q_start_max`.
- Evidence/result: test `test_generate_map_set_distinct_and_exclude`, `test_random_map_env_exclude`.
- Proposed B8 row: không cần (chi tiết cài đặt).

## 2026-10-17 — T6.2: siêu tham số DQN và kích thước mạng

- Change: DQNConfig mặc định learning_rate 5e-4 -> 2.5e-4, target_update_interval 1000 -> 500, buffer_size 50k -> 100k. QNetwork mặc định (32, 64) kênh / ẩn 256 -> (16, 32) kênh / ẩn 128.
- Reason: thử 4 cấu hình trên obstacle_4x4 (xuất phát ngẫu nhiên, 150k bước): cấu hình ban đầu đạt tối ưu nhưng sụp ở 30k–40k bước (coverage 0.72); cấu hình mới ổn định từ 20k bước. Mạng nhỏ hơn cập nhật nhanh gấp 3.2 lần trên CPU (8.8 ms so với 28.3 ms mỗi lô 64 trên lưới 8x8) mà vẫn có 268k tham số.
- Evidence/result: results/tuning/dqn_obstacle_4x4_config_{A,B,D}.log (A = cũ, D = chọn). Cấu hình D: 12.67 bước trung bình, tối ưu trung bình 12.69.
- Proposed B8 row: | v2 | 2026-10-17 | DQN: lr 2.5e-4, target 500, buffer 100k, mạng nhỏ hơn | Cấu hình cũ sụp giữa chừng; huấn luyện CPU chậm | DQN 4x4 đạt gần tối ưu (12.67 vs 12.69 bước), ổn định |

## 2026-10-17 — T4.4/T7: bổ sung script và helper

- Change: thêm `experiments/common.py` (tham số CLI chung, chạy song song, tham chiếu tối ưu/baseline), `experiments/run_optimality.py` (E6 có script riêng), `env/render.py::compose_vertical`, tham số `wrap` cho `evaluate_on_maps`; checkpoint Q-learning lưu theo thư mục seed (`<algo>/seed<N>/checkpoints/`) để các seed không ghi đè nhau.
- Reason: tránh lặp code giữa các script; E6 cần huấn luyện riêng cho từng ô xuất phát.
- Proposed B8 row: không cần (chi tiết cài đặt).

## 2026-10-17 — T5.3: kết quả khác giả thuyết

- Change: không đổi code; ghi nhận kết quả.
- Reason/Evidence: với xuất phát cố định trên phòng TRỐNG, Q-learning dạng bảng vẫn học tới 6x6 trong 15 phút (35 bước = tối ưu), nhưng bảng Q phình tới ~1.95 triệu trạng thái (~746 MB) và chỉ dùng được cho đúng một phòng + một điểm xuất phát. Giới hạn thật của tabular là bộ nhớ và không khái quát được, không phải "không học được" trên phòng nhỏ. (results/tabular/scaling.csv)
- Proposed B8 row: | — | 2026-10-17 | Ghi nhận: tabular học được 6x6 trống khi start cố định | Kết quả thực nghiệm | Giới hạn là bộ nhớ + không khái quát, lý do cần DQN |

## 2026-10-17 — Huấn luyện trên GPU

- Change: cài torch bản CUDA (`torch 2.14.1+cu130`) thay bản CPU; `DQNConfig.device = "auto"` (GPU nếu có, ngược lại CPU), tham số `--device` cho run_deep / run_generalization / run_optimality / reproduce_all; `DQNAgent.load(path, device=...)`.
- Reason: máy có RTX 4050 Laptop 6 GB; lần cài đầu dùng bản CPU theo hướng dẫn trong kế hoạch mà không kiểm tra GPU (lỗi của agent).
- Evidence/result (đo trên furniture_8x8, 12k bước): 1 tiến trình — CPU 1 luồng 2.65 ms/bước, CPU 4 luồng 1.96, GPU 1.92. Sáu tiến trình song song — CPU 4.0 ms/bước mỗi tiến trình, GPU 3.7. Mạng nhỏ nên thời gian chủ yếu nằm ở chạy môi trường và gọi hàm Python mỗi bước, GPU chỉ nhanh hơn ~10–40%. Cách dùng tốt nhất: chia thí nghiệm cho cả GPU và CPU chạy đồng thời.
- Proposed B8 row: | — | 2026-10-17 | Huấn luyện DQN trên GPU (auto) | Máy có RTX 4050 | Nhanh hơn ~10–40% mỗi tiến trình; chạy song song GPU + CPU để tăng thông lượng |

## 2026-10-17 — Cải tiến v3: action masking cho DQN

- Change: `DQNConfig.action_mask` (mặc định tắt) và cờ `--action-mask` trong run_deep: không chọn hướng đâm vào tường/vật cản khi hành động (kể cả khám phá ngẫu nhiên) và khi lấy max trong mục tiêu Q. Mô hình phòng ngẫu nhiên dùng cho demo chuyển sang bản có mask.
- Reason: DQN huấn luyện trên phòng ngẫu nhiên 8x8 (500k bước, 5 seed) chỉ dọn 14% phòng chưa gặp; log cho thấy ~101/216 bước là va chạm — chính sách tham lam tất định đâm tường mãi khi ước lượng sai ở phòng lạ (trạng thái không đổi -> hành động không đổi). Đường cong đi ngang suốt 500k bước nên không phải do thiếu bước huấn luyện. Va chạm không bao giờ có ích, nên che đi không làm mất chính sách tối ưu.
- Evidence/result: A/B trên phòng ngẫu nhiên 6x6 (150k bước, 2 seed, results/mask_ab/): không mask 31% đã dọn, 0% dọn xong, 36 va chạm/episode; có mask 69% đã dọn, 7% dọn xong, 0 va chạm. Vẫn kém greedy BFS (100%) — robot còn lặp qua lại giữa các ô đã dọn.
- Proposed B8 row: | v3 | 2026-10-17 | Action masking cho DQN | DQN đâm tường mãi ở phòng chưa gặp | Phòng lạ 6x6: 31% -> 69% đã dọn; vẫn kém greedy BFS |
