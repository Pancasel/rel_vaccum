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
