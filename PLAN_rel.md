# PLAN.md: Robot hút bụi phủ kín không gian bằng Học tăng cường

> Môn REL301, học kỳ FA26. Hình thức nộp: **Applications** (ứng dụng thực tế, không phải trò chơi).
> File này là **nguồn sự thật duy nhất** của dự án. Thành viên mới đọc Phần A. Claude đọc toàn bộ file, đặc biệt Phần B và Phần C, trước khi viết bất kỳ dòng code nào.

---

## Mục lục

- [Phần A. Dành cho thành viên nhóm (đọc 10 phút là hiểu)](#phần-a)
- [Phần B. Đặc tả kỹ thuật (cho cả nhóm và Claude)](#phần-b)
- [Phần C. Hướng dẫn riêng cho Claude](#phần-c)
- [Phần D. Lộ trình theo giai đoạn, kèm tiêu chí nghiệm thu](#phần-d)
- [Phần E. Thí nghiệm và đánh giá](#phần-e)
- [Phần F. Báo cáo pptx và nộp bài](#phần-f)
- [Phần G. Rủi ro, phân công, thuật ngữ](#phần-g)

---

<a id="phần-a"></a>
# PHẦN A. Dành cho thành viên nhóm

## A1. Dự án này làm gì (giải thích đơn giản)

Chúng ta dạy một **robot hút bụi ảo** tự học cách dọn hết căn phòng.

- Căn phòng là một **lưới ô vuông** (ví dụ 8×8). Một số ô là tường hoặc đồ đạc (robot không đi vào được).
- Robot bắt đầu ở một ô. Mỗi bước nó đi lên, xuống, trái hoặc phải một ô.
- Ô nào robot đi qua thì thành "đã dọn".
- **Mục tiêu:** dọn hết mọi ô trống bằng **ít bước nhất**, ít đi lại ô cũ nhất.

Robot **không được lập trình sẵn** cách đi. Nó thử, được thưởng khi dọn ô mới, bị phạt khi đi lãng phí, rồi tự rút kinh nghiệm. Đó là **học tăng cường** (Reinforcement Learning, RL).

## A2. Vì sao chọn đề tài này

- Là **ứng dụng thực tế** (robot hút bụi, robot dọn sàn, máy cắt cỏ, máy bay không người lái quét ảnh), đúng yêu cầu "không phải trò chơi" của thầy.
- Bài toán thật sự khó: tìm đường đi qua mọi ô với số bước ít nhất gần với bài toán đường đi Hamiltonian, thuộc lớp **NP-khó**. Do đó không có công thức nhanh cho bản đồ lớn, và RL có lý do để tồn tại.
- Chạy được trên laptop hoặc Google Colab, không cần phần cứng robot.
- Có nhiều thí nghiệm thú vị để viết báo cáo (xem Phần E).

## A3. Chúng ta nộp gì

Theo đề bài của thầy:

1. **Thư mục code** (Google Drive, share cho `hungbkhn@gmail.com`): thiết lập RL cho ứng dụng và khám phá ứng dụng, **chú thích chi tiết**.
2. **Một file pptx**: giải thích ứng dụng, cách định hình thành bài toán RL, các phương án giải và lựa chọn, quá trình thực hiện và cải tiến, đánh giá.
3. **Gửi email** tới `hungbkhn@gmail.com`, tiêu đề `REL301_FA26_Nhóm <số/tên nhóm>`, nội dung là link Drive.
4. **Deadline: 21h thứ 4 tuần thứ 9 của kỳ FA26. Không có gia hạn, không có ngày trễ.**
5. **Thuyết trình** ở 3 buổi cuối.

## A4. Phải biết trước khi bắt đầu (tối thiểu)

Không cần học hết mới bắt tay làm. Mỗi người đọc phần liên quan đến vai trò của mình trong sách **Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed., 2018** (bản đầy đủ miễn phí trên trang của tác giả).

| Ai cần | Đọc gì | Để làm gì |
|---|---|---|
| Tất cả | Ch.3 (Finite MDPs) | Hiểu state, action, reward, return, hàm giá trị, Bellman |
| Tất cả | Mục 6.4, 6.5 (SARSA, Q-learning) | Thuật toán chính |
| Phụ trách tabular | Ch.4 (DP, đặc biệt 4.4 Value Iteration) | Tạo đáp án tối ưu để kiểm chứng |
| Phụ trách tabular | Mục 6.7 (Maximization Bias, Double Q-learning) | Cải tiến |
| Phụ trách deep | Ch.9, 10, 11 (xấp xỉ hàm, deadly triad) | Hiểu vì sao cần DQN |
| Phụ trách deep | Ch.13 (Policy Gradient) | Nền cho PPO |
| Phụ trách deep | Bài báo DQN (Mnih et al., Nature 2015) | Replay buffer, target network |
| Thí nghiệm, báo cáo | Ch.3 phần reward hypothesis, Ch.16 (ví dụ ứng dụng) | Viết phần thiết kế reward và động lực |

Kỹ năng lập trình: Python, NumPy, Matplotlib, PyTorch (cơ bản), Git.

## A5. Từ điển 1 phút

| Thuật ngữ | Nghĩa trong dự án này |
|---|---|
| **Agent** | Robot hút bụi |
| **Environment** | Căn phòng dạng lưới, chạy bằng code Python |
| **State (trạng thái)** | Robot ở đâu, ô nào đã dọn, ô nào là vật cản |
| **Action (hành động)** | Lên, xuống, trái, phải |
| **Reward (phần thưởng)** | Số điểm nhận sau mỗi bước (thưởng ô mới, phạt lãng phí) |
| **Episode** | Một lần dọn từ đầu đến khi dọn xong hoặc hết bước |
| **Policy (chính sách)** | Quy tắc "ở trạng thái này thì đi hướng nào" |
| **Q(s,a)** | Điểm kỳ vọng tổng cộng nếu ở trạng thái s làm hành động a rồi đi tiếp tốt |
| **Baseline** | Cách làm không dùng RL để so sánh (đi ngẫu nhiên, zigzag, tham lam) |
| **Tabular** | RL lưu Q trong bảng, dùng được khi số trạng thái ít |
| **Deep RL** | Dùng mạng nơ-ron thay cho bảng, dùng được khi số trạng thái rất lớn |
| **Seed** | Số khởi tạo ngẫu nhiên, cố định để chạy lại ra đúng kết quả |

## A6. Quy tắc làm việc chung

1. Mọi thuật toán phải **khớp một mục trong sách** và có chú thích dẫn chiếu (ví dụ `# Sutton & Barto 2018, mục 6.5, Q-learning`).
2. **Làm theo vòng nhỏ:** làm một việc, chạy được, kiểm tra được, commit, mới sang việc tiếp.
3. Dùng AI để viết code **được**, nhưng **mỗi người phải giải thích được phần mình làm** mà không mở AI. Thầy sẽ hỏi vấn đáp.
4. Code của mình **phải có test**. Code AI viết hay chạy được nhưng sai ngầm.
5. Dùng Git, mỗi giai đoạn một nhánh hoặc ít nhất một commit rõ ràng. Cố định seed.
6. Không commit file nặng (model lớn, dữ liệu thô) hoặc khóa bí mật.

---

<a id="phần-b"></a>
# PHẦN B. Đặc tả kỹ thuật

## B1. Phạm vi

**Trong phạm vi**
- Một agent.
- Môi trường lưới 2D rời rạc, tự xây theo chuẩn **Gymnasium**.
- Vật cản **tĩnh**. Có bản đồ cố định (để so sánh công bằng) và bản đồ sinh ngẫu nhiên (để test khái quát hóa).
- Bốn hành động rời rạc.
- Bản đồ từ 3×3 (kiểm chứng) đến khoảng 10×10 (deep RL).

**Ngoài phạm vi (ghi rõ trong báo cáo là giới hạn)**
- Cảm biến nhiễu, trượt bánh, động lực học, SLAM.
- Vật cản động (người, thú cưng).
- Nhiều robot.
- Phần cứng thật.
- **Mở rộng tùy chọn nếu còn thời gian:** pin và trạm sạc, quan sát một phần (POMDP).

## B2. Định nghĩa MDP (v1, có thể cải tiến, ghi lại mọi thay đổi ở mục B8)

Gọi lưới là `H × W`, tập ô trống (không phải vật cản) là `F`.

**State**
- Đủ thông tin để thỏa tính Markov: **vị trí robot** + **tập ô đã dọn** + bản đồ vật cản.
- Biểu diễn cho deep RL: tensor `(3, H, W)`
  - kênh 0: vị trí robot (one-hot)
  - kênh 1: ô đã dọn (0/1)
  - kênh 2: vật cản (0/1)
- Biểu diễn cho tabular: tuple `(row, col, frozenset hoặc bitmask ô đã dọn)` trên bản đồ cố định.
- **Lý do bắt buộc có "ô đã dọn" trong state:** nếu chỉ có vị trí, hai tình huống có cùng vị trí nhưng khác tiến độ dọn bị gộp làm một, phần thưởng tương lai khác nhau, tính Markov bị phá (Sutton & Barto, mục 3.5). Đây là một thí nghiệm bắt buộc (E1).

**Action** (rời rạc, 4 giá trị): `0=lên, 1=xuống, 2=trái, 3=phải`.
- Đi vào tường hoặc vật cản: robot **đứng yên** và bị phạt.
- (Tùy chọn) hỗ trợ **action masking** để loại hành động vô nghĩa.

**Transition P:** xác định (deterministic) trong v1. Có thể thêm nhiễu (ví dụ 10% trượt sang hướng khác) ở bản mở rộng.

**Reward (cấu hình v1 "dense")**

| Sự kiện | Reward |
|---|---|
| Vào ô chưa dọn | `+1.0` |
| Vào ô đã dọn | `-0.1` |
| Đụng tường hoặc vật cản | `-0.5` |
| Mỗi bước thời gian | `-0.01` |
| Dọn xong 100% ô trống | `+10.0` (bonus kết thúc) |

Reward phải **tham số hóa** (dataclass `RewardConfig`) để chạy ablation mà không sửa code lõi.

**Điều kiện kết thúc** (phân biệt rõ theo Gymnasium)
- `terminated = True`: dọn xong toàn bộ ô trống.
- `truncated = True`: chạm giới hạn `max_steps` (mặc định `4 * |F|`, tham số hóa).

**Discount γ:** mặc định `0.99`. Nêu rõ trong báo cáo vì sao chọn.

**Khởi đầu:** vị trí xuất phát cố định hoặc ngẫu nhiên trên ô trống (tham số `start`). Ô xuất phát tính là đã dọn.

## B3. Bài toán dùng làm mốc kiểm chứng

Số trạng thái tabular ≈ `|F| × 2^|F|` (vị trí × tập ô đã dọn), thực tế ít hơn vì chỉ tính trạng thái **đạt tới được**. Với lưới 3×3 và 4×4 (có thể thêm vài vật cản) vẫn liệt kê được bằng BFS, nên **Value Iteration** cho đáp án tối ưu chính xác. Với lưới lớn, số trạng thái bùng nổ theo hàm mũ. Đây chính là động cơ chuyển sang Deep RL và là một hình minh họa bắt buộc trong báo cáo.

## B4. Baseline (không dùng RL)

| Baseline | Mô tả |
|---|---|
| `random_walk` | Chọn hành động hợp lệ ngẫu nhiên đều |
| `zigzag` | Quét theo hàng (boustrophedon), gặp vật cản thì rẽ/né theo luật đơn giản |
| `greedy_bfs` | Dùng BFS tìm ô **chưa dọn gần nhất** rồi đi tới đó, lặp lại |
| (tùy chọn) `spiral` | Xoáy ốc từ ngoài vào |

Mọi baseline dùng **cùng giao diện** như agent RL (nhận observation, trả action) để chạy chung bộ đánh giá.

## B5. Thuật toán RL

**Tabular (Sutton & Barto)**
- `value_iteration`: mục 4.4. Chỉ trên lưới nhỏ, cho đáp án tối ưu `V*`, `π*`.
- `q_learning`: mục 6.5, ε-greedy, ε giảm dần.
- `sarsa`: mục 6.4.
- `double_q_learning`: mục 6.7.

**Deep**
- `dqn`: replay buffer + target network, CNN nhỏ nhận tensor `(3,H,W)`. Sau đó thử Double DQN.
- `ppo_sb3`: PPO của Stable-Baselines3 làm đối chiếu độc lập (giúp phân biệt "code DQN của mình sai" với "bài toán khó").

## B6. Chỉ số đánh giá (metrics)

Với mỗi episode, ghi:
- `coverage`: phần trăm ô trống đã dọn khi kết thúc.
- `steps_to_full`: số bước để dọn xong (nếu xong).
- `redundancy`: tỷ lệ bước đi vào ô đã dọn.
- `collisions`: số lần đụng tường.
- `return`: tổng reward (có chiết khấu và không chiết khấu).

Báo cáo **trung bình ± độ lệch chuẩn qua ≥ 5 seed** (seed 0..4), không báo cáo "lần chạy tốt nhất".

## B7. Công nghệ

- Python 3.11
- `numpy`, `matplotlib`, `pandas`
- `gymnasium` (API: `reset() -> (obs, info)`, `step() -> (obs, reward, terminated, truncated, info)`)
- `torch`
- `stable-baselines3` (PPO, đối chiếu)
- `pytest`
- (tùy chọn) `tensorboard`, `pygame` cho render

## B8. Nhật ký thay đổi thiết kế (nhóm cập nhật)

| Phiên bản | Ngày | Thay đổi | Lý do | Kết quả |
|---|---|---|---|---|
| v1 | | Reward dense như B2 | Khởi đầu | |
| v2 | | | | |
| v3 | | | | |

Bảng này là nguyên liệu trực tiếp cho slide "quá trình thực hiện và cải tiến".

## B9. Cấu trúc repo

```
vacuum-rl/
├── PLAN.md                 # file này
├── README.md               # cách cài và chạy, kết quả chính
├── requirements.txt
├── env/
│   ├── __init__.py
│   ├── vacuum_env.py       # lớp VacuumEnv(gymnasium.Env), RewardConfig
│   ├── maps.py             # bản đồ cố định + hàm sinh ngẫu nhiên (đảm bảo ô trống liên thông)
│   └── render.py           # vẽ khung hình, xuất GIF/MP4
├── baselines/
│   ├── random_walk.py
│   ├── zigzag.py
│   └── greedy_bfs.py
├── tabular/
│   ├── value_iteration.py
│   ├── q_learning.py
│   ├── sarsa.py
│   └── double_q.py
├── deep/
│   ├── networks.py         # CNN nhỏ
│   ├── replay_buffer.py
│   ├── dqn.py              # DQN + tùy chọn Double DQN
│   └── ppo_sb3.py
├── experiments/
│   ├── evaluate.py         # chạy 1 agent trên N episode, ghi metrics ra CSV
│   ├── run_baselines.py
│   ├── run_tabular.py
│   ├── run_deep.py
│   ├── run_ablation.py
│   └── plots.py            # learning curve, bảng, heatmap
├── tests/
│   ├── test_env.py
│   ├── test_baselines.py
│   ├── test_tabular.py     # so khớp với value iteration
│   └── test_deep.py        # smoke test (chạy vài bước, không crash)
├── results/                # CSV, hình, video (không commit file nặng)
└── slides/                 # file pptx, hình dùng cho slide
```

---

<a id="phần-c"></a>
# PHẦN C. Hướng dẫn riêng cho Claude

> Claude: nếu bạn đang đọc file này để giúp viết code, hãy tuân thủ phần này.

## C1. Bối cảnh

Đây là bài tập cuối kỳ môn học tăng cường, có chấm điểm và vấn đáp. Người dùng là sinh viên, cần code **dễ hiểu, đúng với giáo trình Sutton & Barto (2018), chú thích chi tiết**, không cần tối ưu hiệu năng cực đoan.

## C2. Quy tắc làm việc

1. **Mỗi lần chỉ làm một nhiệm vụ** của giai đoạn hiện tại (xem Phần D). Không viết trước code của giai đoạn sau, không sửa file ngoài phạm vi được yêu cầu.
2. **Đọc Phần B trước.** Tên lớp, tên file, giao diện, reward mặc định, cấu trúc thư mục phải khớp. Nếu thấy mâu thuẫn hoặc thiếu thông tin, **hỏi lại**, đừng tự bịa.
3. **Đối chiếu giáo trình.** Với thuật toán RL, ghi trong comment số mục và tên bước trong pseudocode của Sutton & Barto. Viết đúng pseudocode, không "cải tiến" âm thầm. Nếu muốn biến thể, nói rõ và hỏi.
4. **Viết test cùng với code.** Mỗi module có ít nhất vài test trong `tests/`. Test tabular phải so với đáp án Value Iteration.
5. **Comment giải thích "vì sao", không chỉ "cái gì".** Ngôn ngữ comment và docstring: **tiếng Việt**. Tên biến, hàm, lớp: **tiếng Anh**, kiểu `snake_case`.
6. **Type hints và docstring** cho hàm công khai.
7. **Tái lập được:** mọi script nhận tham số `--seed`, đặt seed cho `random`, `numpy`, `torch`, và `env.reset(seed=...)`.
8. **Ghi log ra CSV** (episode, return, coverage, steps, redundancy, collisions, ...), không chỉ in màn hình.
9. **Không phụ thuộc ngầm:** dùng phiên bản thư viện trong `requirements.txt`. Không thêm thư viện lớn khi chưa hỏi.
10. **Báo cáo ngắn sau mỗi nhiệm vụ:** đã làm gì, chạy lệnh nào để kiểm tra, giới hạn đã biết. Không tuyên bố "đã hoạt động" nếu chưa chạy test.
11. **Cảnh giác lỗi thường gặp ở RL:**
    - nhầm `terminated` và `truncated` khi tính target (truncated vẫn được bootstrap, terminated thì không);
    - quên đặt `done` làm mask target `(1 - terminated)`;
    - rò rỉ trạng thái giữa các episode do không copy mảng;
    - dùng `Q` của mạng online thay vì target network;
    - reward quá lớn hoặc không chuẩn hóa làm DQN bất ổn.

## C3. Giao diện bắt buộc

**Môi trường**
```python
class VacuumEnv(gymnasium.Env):
    def __init__(self, grid, reward_config: RewardConfig | None = None,
                 max_steps: int | None = None, start: tuple[int,int] | None = None,
                 render_mode: str | None = None): ...
    def reset(self, *, seed: int | None = None, options: dict | None = None): ...
        # trả (obs, info); obs shape (3, H, W), dtype float32
    def step(self, action: int): ...
        # trả (obs, reward, terminated, truncated, info)
        # info chứa: coverage, collided, new_cell
    def render(self): ...
```

**Agent (baseline và RL cùng một giao diện)**
```python
class Agent(Protocol):
    def act(self, obs, info=None) -> int: ...
```

**Đánh giá**
```python
def evaluate(agent, env_factory, n_episodes: int, seed: int) -> pandas.DataFrame: ...
    # mỗi dòng một episode với các metric ở B6
```

## C4. Mẫu prompt người dùng có thể đưa cho Claude

> "Đọc PLAN.md. Hiện đang ở Giai đoạn 2. Hãy làm nhiệm vụ 2.1: viết `env/vacuum_env.py` đúng đặc tả B2 và giao diện C3. Kèm test trong `tests/test_env.py` cho: đụng tường, vào ô mới, vào ô đã dọn, dọn xong (terminated), hết bước (truncated), reset có seed lặp lại được. Chưa viết gì ngoài các file này."

> "Đọc PLAN.md, Giai đoạn 4, nhiệm vụ 4.2: viết Q-learning dạng bảng đúng pseudocode mục 6.5, kèm script so sánh chính sách tìm được với Value Iteration trên bản đồ 3×3. Comment đối chiếu từng dòng với pseudocode."

> "Agent DQN của mình có reward đi ngang. Đây là log và code. Hãy liệt kê tối đa 5 nguyên nhân khả dĩ theo xác suất giảm dần và cách kiểm tra từng cái, đừng sửa code vội."

---

<a id="phần-d"></a>
# PHẦN D. Lộ trình theo giai đoạn

Tính lùi từ **deadline 21h thứ 4 tuần 9**. Nếu đã qua vài tuần, **nén** các giai đoạn đầu, đừng bỏ giai đoạn nào. Giai đoạn 1 đến 4 + 7 là **bản nộp tối thiểu**. Giai đoạn 5 và 6 (deep RL) là phần nâng cấp.

Mỗi nhiệm vụ có **tiêu chí nghiệm thu** (Definition of Done). Chưa đạt thì chưa sang việc tiếp.

## Giai đoạn 1. Chốt đề tài và khởi tạo (tuần 1)

| # | Nhiệm vụ | Nghiệm thu |
|---|---|---|
| 1.1 | Cả nhóm đọc Phần A, Ch.3 sách, thống nhất đặc tả MDP (B2) | Mỗi người tự viết lại được S, A, R, γ của dự án |
| 1.2 | Tạo repo, cấu trúc thư mục (B9), `requirements.txt`, README rỗng | `pip install -r requirements.txt` chạy sạch |
| 1.3 | Phân công (G2), lập lịch họp cố định hằng tuần | Có bảng phân công và lịch |
| 1.4 | Gửi thầy hoặc hỏi nhanh trên lớp về ý tưởng và phạm vi | Có phản hồi (ghi lại vào B8) |

## Giai đoạn 2. Môi trường (tuần 2)

| # | Nhiệm vụ | Nghiệm thu |
|---|---|---|
| 2.1 | `vacuum_env.py` với `RewardConfig` theo B2 | `pytest tests/test_env.py` xanh; có đủ test các tình huống biên |
| 2.2 | `maps.py`: 3 bản đồ cố định (3×3 trống, 4×4 có vật cản, 8×8 có đồ đạc) + hàm sinh ngẫu nhiên có kiểm tra ô trống liên thông | Bản đồ ngẫu nhiên luôn dọn được hết (không có ô bị chắn) |
| 2.3 | `render.py`: vẽ khung hình, xuất GIF | Xem được GIF robot chạy hành động ngẫu nhiên |
| 2.4 | Kiểm tra `gymnasium.utils.env_checker.check_env` | Không lỗi |

## Giai đoạn 3. Baseline và đáp án tối ưu (tuần 3)

| # | Nhiệm vụ | Nghiệm thu |
|---|---|---|
| 3.1 | `random_walk`, `zigzag`, `greedy_bfs` theo giao diện Agent | Mỗi baseline chạy được trên cả 3 bản đồ cố định |
| 3.2 | `evaluate.py` + `run_baselines.py` ghi CSV theo metrics B6 | Có bảng so sánh baseline (≥ 5 seed hoặc ≥ 100 episode) |
| 3.3 | `value_iteration.py` trên bản đồ 3×3 và 4×4: liệt kê trạng thái đạt tới được bằng BFS, lặp đến hội tụ | Có `V*`, `π*`, số bước tối ưu; `greedy_bfs` không tốt hơn tối ưu (kiểm tra tính đúng) |

## Giai đoạn 4. Tabular RL (tuần 4)

| # | Nhiệm vụ | Nghiệm thu |
|---|---|---|
| 4.1 | `q_learning.py` (mục 6.5) | Trên 3×3, chính sách hội tụ về trùng hoặc tương đương `π*` (test tự động) |
| 4.2 | `sarsa.py` (mục 6.4) | Cùng test; so sánh learning curve với Q-learning |
| 4.3 | `double_q.py` (mục 6.7) | Cùng test; có hình so sánh độ chệch ước lượng Q |
| 4.4 | `run_tabular.py` + learning curve (≥ 5 seed) | Hình đường cong trung bình ± độ lệch chuẩn |

## Giai đoạn 5. Giới hạn của bảng (tuần 5)

| # | Nhiệm vụ | Nghiệm thu |
|---|---|---|
| 5.1 | Đo số trạng thái đạt tới được theo kích thước lưới, vẽ đồ thị (thang log) | Hình minh họa bùng nổ trạng thái |
| 5.2 | **Thí nghiệm E1:** state có và không có bản đồ đã dọn | Có bằng chứng thực nghiệm vì sao thiếu thì không học được tốt |
| 5.3 | Chạy thử Q-learning trên 5×5, 6×6 | Chỉ ra điểm tabular bắt đầu quá chậm hoặc quá tốn bộ nhớ |

## Giai đoạn 6. Deep RL (tuần 6)

| # | Nhiệm vụ | Nghiệm thu |
|---|---|---|
| 6.1 | `networks.py`, `replay_buffer.py` (có test) | Buffer thêm, lấy mẫu đúng shape |
| 6.2 | `dqn.py` đúng Mnih et al. 2015, ghi log loss và Q trung bình | Học được trên 4×4, đạt gần `π*`; sau đó 8×8 |
| 6.3 | `ppo_sb3.py` làm đối chiếu | PPO học được trên cùng bản đồ; so sánh với DQN |
| 6.4 | Double DQN (biến thể mục 6.7 ở dạng sâu) | Có so sánh với DQN thường |

## Giai đoạn 7. Cải tiến, ablation, tổng hợp (tuần 7)

| # | Nhiệm vụ | Nghiệm thu |
|---|---|---|
| 7.1 | Ablation reward (E2), ghi lại ít nhất một trường hợp reward hacking kèm GIF | Bảng và GIF |
| 7.2 | Curriculum (lưới nhỏ rồi lớn) và bản đồ ngẫu nhiên (E4, E5) | Đường cong và bảng |
| 7.3 | So sánh tổng thể RL và baseline khi tăng mật độ vật cản (E3) | Hình tổng hợp |
| 7.4 | Cập nhật bảng B8 | Mỗi thay đổi có lý do và kết quả |

## Giai đoạn 8. Slide và dọn code (tuần 8)

| # | Nhiệm vụ | Nghiệm thu |
|---|---|---|
| 8.1 | Làm pptx theo khung Phần F | Bản đầu hoàn chỉnh, có hình và GIF/video |
| 8.2 | Rà soát comment, docstring, README (cài đặt và chạy lại từng thí nghiệm bằng một lệnh) | Người ngoài làm theo README chạy được |
| 8.3 | Tập thuyết trình, tập hỏi đáp chéo trong nhóm | Mỗi người trả lời được câu hỏi về phần của mình |

## Giai đoạn 9. Nghiệm thu và nộp (tuần 9)

| # | Nhiệm vụ | Nghiệm thu |
|---|---|---|
| 9.1 | Chạy lại toàn bộ test và script chính trên máy sạch | Tất cả xanh |
| 9.2 | Upload Drive, share cho `hungbkhn@gmail.com`, **mở link bằng tài khoản khác để kiểm tra quyền** | Mở được, thấy thư mục code và pptx |
| 9.3 | Gửi email đúng tiêu đề `REL301_FA26_Nhóm <số/tên>`, **nộp sớm ít nhất vài giờ trước 21h thứ 4** | Email đã gửi, lưu ảnh chụp |

---

<a id="phần-e"></a>
# PHẦN E. Thí nghiệm và đánh giá

Đây là nguồn điểm sáng tạo và "hiểu biết mới" của báo cáo. Mỗi thí nghiệm cần: **giả thuyết, cách làm, kết quả, giải thích**.

| Mã | Thí nghiệm | Giả thuyết | Đầu ra |
|---|---|---|---|
| **E1** | State **có / không có** bản đồ đã dọn | Thiếu thông tin tiến độ làm agent không học được chiến lược phủ kín (mất tính Markov) | Learning curve hai cấu hình |
| **E2** | Ba thiết kế reward: (a) thưa: chỉ thưởng dọn xong; (b) dày như B2; (c) dày + shaping dạng thế năng theo khoảng cách tới ô chưa dọn gần nhất | Reward thưa học chậm hoặc không học; reward thiết kế kém có thể bị lợi dụng; shaping dạng thế năng không đổi chính sách tối ưu | Đường cong, GIF reward hacking nếu có |
| **E3** | RL vs random vs zigzag vs greedy BFS khi tăng mật độ vật cản (0%, 10%, 20%, 30%) | Zigzag tốt trên phòng trống nhưng yếu dần khi nhiều vật cản; RL có thể tiệm cận hoặc vượt trong vùng giữa | Hình tổng hợp, kể cả trường hợp RL thua |
| **E4** | Khái quát hóa: huấn luyện trên K bản đồ ngẫu nhiên, test trên bản đồ chưa gặp | Mạng CNN khái quát được ở mức nào đó; có giới hạn | Bảng train và test |
| **E5** | Q-learning vs SARSA vs Double Q-learning | Double Q giảm độ chệch tối đa hóa (mục 6.7) | Learning curve, đồ thị bias |
| **E6** | Đối chiếu với đáp án tối ưu của Value Iteration (3×3, 4×4) | Q-learning/DQN tiến sát tối ưu | Bảng khoảng cách tới tối ưu |
| **E7** | Bùng nổ trạng thái theo kích thước lưới | Tabular không scale | Đồ thị log |

**Quy tắc báo cáo trung thực**
- Báo cáo trung bình ± độ lệch chuẩn qua ≥ 5 seed.
- So sánh công bằng: cùng bản đồ, cùng `max_steps`, cùng số episode đánh giá.
- Nếu RL thua baseline, vẫn báo cáo và phân tích nguyên nhân. Đó là kết quả hợp lệ.
- Ghi rõ siêu tham số và ngân sách huấn luyện của mỗi agent.

---

<a id="phần-f"></a>
# PHẦN F. Báo cáo pptx và nộp bài

## F1. Khung slide

1. **Giới thiệu ứng dụng và động lực** (robot hút bụi, Coverage Path Planning, vì sao khó)
2. **Định hình thành bài toán RL:** state, action, reward, kết thúc (hình minh họa môi trường)
3. **Các phương án giải và lý do chọn:** baseline, tabular, deep (bảng so sánh ưu nhược)
4. **Quá trình thực hiện và cải tiến:** v1 → v2 → v3, lỗi gặp, cách sửa (lấy từ bảng B8)
5. **Thí nghiệm và kết quả:** E1 đến E7 chọn lọc, learning curve, bảng, GIF
6. **Đánh giá, hạn chế, hướng phát triển** (sim-to-real, vật cản động, pin, nhiều robot)
7. **Phân công và tài liệu tham khảo** (Sutton & Barto 2018, Mnih et al. 2015, Schulman et al. 2017 cho PPO)

## F2. Về lời giới thiệu cho giảng viên

Đặt vào README hoặc slide 1: mục tiêu, phạm vi, giới hạn, sản phẩm nộp, cách đối chiếu tiêu chí đề bài, cách nhóm dùng công cụ AI. Giữ phần **Giới hạn** thẳng thắn. Chỉ ghi biện pháp nhóm thực sự đã làm.

## F3. Checklist nộp bài

- [ ] Repo chạy được từ đầu theo README
- [ ] Code có chú thích, không còn file rác hoặc khóa bí mật
- [ ] `pytest` xanh
- [ ] pptx nằm trong `slides/`
- [ ] Link Drive share cho `hungbkhn@gmail.com`, đã kiểm tra bằng tài khoản khác
- [ ] Email đúng tiêu đề `REL301_FA26_Nhóm ...`, gửi trước **21h thứ 4 tuần 9**
- [ ] Đã tập thuyết trình cho 3 buổi cuối và tập vấn đáp

---

<a id="phần-g"></a>
# PHẦN G. Rủi ro, phân công, thuật ngữ

## G1. Rủi ro và cách phòng

| Rủi ro | Cách xử lý |
|---|---|
| DQN không học | Quay lại lưới 3×3 và 4×4, kiểm tra reward và mask `terminated`, thử PPO của SB3 để loại trừ lỗi code |
| Code AI viết sai ngầm | Test với đáp án Value Iteration, đối chiếu pseudocode sách, đọc code từng dòng |
| Không kịp deep RL | Giai đoạn 1 đến 4 + 7 + 8 là bản nộp tối thiểu hợp lệ; deep RL là nâng cấp |
| Kết quả không ổn định | ≥ 5 seed, báo cáo trung bình ± độ lệch chuẩn |
| Chỉ nhóm trưởng hiểu code | Mỗi người trình bày một phần, tập hỏi đáp chéo trước tuần 9 |
| Máy yếu, huấn luyện chậm | Dùng Colab, giữ lưới ≤ 10×10, giảm số seed khi thử nghiệm và tăng lại khi chạy chính thức |
| Nộp lỗi (quyền Drive, sai tiêu đề) | Nộp sớm, kiểm tra bằng tài khoản khác, chụp ảnh email đã gửi |
| Phạm vi phình to | Mọi tính năng mới (pin, vật cản động) chỉ làm sau khi bản tối thiểu xong |

## G2. Phân công (điền tên)

| Vai trò | Người | Phụ trách chính | Sách cần đọc |
|---|---|---|---|
| A. Môi trường và trực quan hóa | [ ] | `env/`, test, GIF, video demo | Ch.3 |
| B. Baseline và tabular | [ ] | `baselines/`, `tabular/`, Value Iteration | Ch.4, 6 |
| C. Deep RL | [ ] | `deep/`, PPO đối chiếu | Ch.9–11, 13, bài DQN |
| D. Thí nghiệm và báo cáo | [ ] | `experiments/`, biểu đồ, slide, README | Ch.3, 16 |

Nhóm 3 người: gộp B với D. Dù phân công thế nào, **mọi người đều phải hiểu MDP của dự án** và trả lời được vì sao state phải chứa bản đồ đã dọn.

## G3. Thông tin nhóm (điền)

- Tên hoặc số nhóm: [ ]
- Thành viên và MSSV: [ ]
- Email nhóm trưởng: [ ]
- Link repo và Drive: [ ]
- Lịch họp cố định: [ ]

## G4. Tài liệu tham khảo

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518.
- van Hasselt, H., Guez, A., & Silver, D. (2016). Deep Reinforcement Learning with Double Q-learning. *AAAI*.
- Schulman, J. et al. (2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347*.
- Gymnasium documentation; Stable-Baselines3 documentation.
- Galceran, E., & Carreras, M. (2013). A survey on coverage path planning for robotics. *Robotics and Autonomous Systems*.
