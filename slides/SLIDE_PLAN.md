# Kế hoạch slide thuyết trình — Robot hút bụi học tăng cường (REL301 FA26)

> Dùng cho cả nhóm khi dựng file pptx nộp kèm code (yêu cầu ở PLAN_rel.md mục A3 và Phần F).
> **Bản dựng sẵn theo đúng plan này:** `slides/REL301_FA26_RobotHutBui_RL.pptx` (24 slide: 18 slide chính, slide cảm ơn,
> 5 slide phụ lục; mỗi slide có ghi chú người nói). Nhóm chỉ cần điền tên nhóm, thành viên, giảng viên ở slide 1.
> Mọi số liệu dưới đây lấy từ `results/` và `BAO_CAO_DU_AN.html`; không tự gõ lại số khác.

---

## 1. Thông số chung

| Mục | Quy định |
|---|---|
| Thời lượng | 15 phút trình bày + 5 phút hỏi đáp |
| Số slide chính | 18 slide (khoảng 50 giây/slide), cộng 5 slide dự phòng cho hỏi đáp |
| Tỷ lệ | 16:9 |
| Công cụ | PowerPoint hoặc Google Slides (xuất .pptx); GIF chèn trực tiếp vẫn chạy trong PowerPoint |
| Người trình bày | chia theo vai trò G2: A môi trường, B Q-learning, C DQN, D thí nghiệm |

---

## 2. Phong cách: học thuật, thoáng

Tinh thần: giống một bài báo hội nghị được trình bày bằng slide. Nền trắng, nhiều khoảng trống, mỗi slide một ý, hình to, chữ ít.

### 2.1 Màu

Lấy từ chính mô phỏng để slide và hình minh họa thống nhất với nhau.

| Vai trò | Màu | Mã | Dùng cho |
|---|---|---|---|
| Nền | trắng | `#FFFFFF` | mọi slide |
| Chữ chính | xanh than | `#1D2733` | tiêu đề, nội dung |
| Chữ phụ | xám xanh | `#5B6878` | chú thích, nguồn, số trang |
| Nhấn chính | xanh dương | `#1F6FB2` | số liệu quan trọng, đường kẻ tiêu đề |
| Nền nhấn nhẹ | xanh rất nhạt | `#E8F3FF` | ô "kết luận", hàng đầu bảng |
| Điểm nhấn hiếm | đỏ robot | `#C8322F` | chỉ dùng 1 chỗ/slide: con số "wow" hoặc chữ "Vì sao?" |
| Tốt / kém | xanh lá / đỏ gạch | `#2E7D4F` / `#B3261E` | ô tốt nhất / ô yếu nhất trong bảng |

Màu từng agent phải giống hệt trong các hình đã vẽ: random `#9E9E9E`, zigzag `#FF9800`, greedy BFS `#4CAF50`, Q-learning `#1F77B4`, DQN `#D62728`.

### 2.2 Chữ

| Vai trò | Font | Cỡ |
|---|---|---|
| Tiêu đề slide | **Cambria** Bold (serif, gợi cảm giác học thuật) | 28 pt |
| Nội dung | **Calibri** | 15–20 pt |
| Con số lớn | Calibri Bold | 40–60 pt |
| Chú thích, nguồn | Calibri Italic | 12 pt |
| Công thức | Cambria Math | 20–32 pt |

Cả ba font có sẵn trong mọi bản Microsoft Office và hiển thị tiếng Việt có dấu chuẩn, nên không cần nhúng font.

### 2.3 Bố cục

- Lề 0,6 inch mỗi bên. Không đặt gì sát mép.
- Tiêu đề căn trái ở trên cùng, phía trên là nhãn mục nhỏ (chữ in hoa giãn cách) kèm một ô vuông nhỏ màu "ô đã dọn" `#A0D2FF` — họa tiết lặp lại ở mọi slide. Không kẻ gạch dưới tiêu đề, không dùng dải màu trang trí.
- Tiêu đề viết thành **một câu kết luận**, không phải tên chủ đề. Ví dụ: "Q-learning đạt đúng lời giải tối ưu" thay vì "Kết quả Q-learning".
- Tối đa 3 gạch đầu dòng, mỗi dòng không quá 12 chữ. Ý dài để trong ghi chú người nói.
- Chân trang: bên trái "REL301 · Robot hút bụi RL", bên phải số trang, cỡ 12 pt màu xám.
- Hình chiếm ít nhất 55% diện tích ở các slide kết quả. Mỗi hình có một dòng nguồn nhỏ bên dưới (ví dụ "5 seed, trung bình ± std").
- Không dùng clip-art, biểu tượng cảm xúc, hiệu ứng chuyển slide lòe loẹt. Chỉ dùng hiệu ứng "Appear" khi cần lộ dần từng ý.

### 2.4 Ba loại slide dùng lặp lại

1. **Slide hình lớn:** tiêu đề kết luận, một hình to, 1–2 dòng chú thích bên dưới.
2. **Slide số liệu:** tiêu đề kết luận, 2–3 con số lớn xếp ngang, mỗi số kèm một dòng giải thích.
3. **Slide so sánh:** chia đôi trái/phải hoặc bảng nhỏ (tối đa 5 hàng × 5 cột), ô tốt nhất tô xanh lá.

---

## 3. Dàn ý từng slide

Bố cục ghi bằng sơ đồ chữ. "Hình:" là đường dẫn file trong thư mục dự án.

### Phần 1 — Ứng dụng và động lực (người D, ~2 phút)

**Slide 1. Trang bìa**
- Tiêu đề: *Robot hút bụi tự học cách dọn kín căn phòng*
- Phụ đề: *Q-learning và Deep Q-learning cho bài toán Coverage Path Planning*
- Tên nhóm, thành viên, MSSV, REL301 · FA26, giảng viên.
- Hình: khung cuối của `results/demo/compare_furniture_8x8.png` đặt mờ (độ trong 15%) ở nửa phải.
```
┌───────────────────────────────────────────────┐
│ Robot hút bụi tự học            [hình mờ    ] │
│ cách dọn kín căn phòng          [phòng 8x8  ] │
│ ─────                           [           ] │
│ Q-learning & Deep Q-learning                  │
│ Nhóm X · tên – MSSV …                         │
└───────────────────────────────────────────────┘
```

**Slide 2. Bài toán thực tế khó hơn ta nghĩ**
- Bên trái 3 dòng: robot hút bụi, máy cắt cỏ, drone khảo sát đều cần **đi qua mọi điểm với quãng đường ngắn nhất**.
- Bên phải: một con số lớn màu đỏ "**NP-khó**", dưới ghi "gần với bài toán đường đi Hamilton".
- Ghi chú người nói: robot rẻ tiền đi zigzag hoặc đi tới chỗ bẩn gần nhất, cả hai lãng phí khi phòng nhiều đồ đạc.

**Slide 3. Câu hỏi của dự án**
- Một câu lớn ở giữa slide: *"Robot có thể tự học một chiến lược dọn tốt hơn cách làm thủ công không — kể cả ở căn phòng chưa từng thấy?"*
- Dưới, 3 ô nhỏ: Phòng nhỏ → Phòng lớn → Phòng lạ (chính là mạch của cả bài).

### Phần 2 — Định hình bài toán RL (người A, ~2,5 phút)

**Slide 4. Môi trường: căn phòng là một lưới ô**
- Hình: `results/demo/compare_obstacle_4x4.png` (cắt lấy một ô) hoặc chụp một khung GIF.
- Chú thích màu: xám = sàn chưa dọn, xanh = đã dọn, đen = đồ đạc, đỏ = robot.
- 2 dòng: tự viết theo chuẩn Gymnasium; một hàm `transition()` duy nhất dùng chung cho môi trường và đáp án tối ưu.

**Slide 5. MDP của robot hút bụi**
- Bảng 5 hàng: State, Action, Reward, Kết thúc, γ (lấy từ mục 3 của báo cáo).
- Reward viết gọn: ô mới **+1**, ô cũ −0,1, đụng tường −0,5, mỗi bước −0,01, xong **+10**.
- Ô nhấn nền xanh nhạt: "State phải chứa *ô nào đã dọn* — nếu không, mất tính Markov (S&B 3.1)".

**Slide 6. Robot "nhìn thấy" gì**
- Ba hình vuông nhỏ cạnh nhau: kênh vị trí robot, kênh đã dọn, kênh vật cản (vẽ lại bằng ô vuông trong PowerPoint, 4×4 là đủ).
- Mũi tên sang phải: "→ Bảng Q (phòng nhỏ)" và "→ Mạng CNN (phòng lớn)".

### Phần 3 — Phương án giải và lựa chọn (người B, ~2 phút)

**Slide 7. Bốn nhóm phương pháp, mỗi nhóm một vai trò**
- Bảng 4 hàng (Vai trò | Phương pháp | Vì sao chọn):
  - Mô hình chính | Q-learning (S&B 6.5) | đơn giản, đúng giáo trình, học được tối ưu
  - Mở rộng | DQN | phòng 8×8 có ~10²¹ trạng thái, bảng không chứa nổi
  - Đáp án | Value Iteration (S&B 4.4) | lời giải tối ưu tuyệt đối để chấm điểm
  - Đối chứng | random, zigzag, greedy BFS | cách làm thủ công
- Dòng nhỏ dưới: "Không dùng hết sách: bỏ SARSA, Double Q, PPO để tập trung một mạch".

**Slide 8. Q-learning trong một dòng công thức**
- Công thức lớn ở giữa (JetBrains Mono hoặc Equation):
  `Q(S,A) ← Q(S,A) + α [ R + γ · max_a Q(S',a) − Q(S,A) ]`
- Dưới, 3 nhãn chỉ vào công thức: "phần thưởng vừa nhận", "giá trị tốt nhất phía trước", "sai số cần sửa".
- Ghi chú người nói: ε-greedy giảm từ 1 xuống 0,05; terminated thì không cộng γ·max.

**Slide 9. DQN = Q-learning + mạng nơ-ron**
- Bảng đối chiếu 3 hàng (lấy từ mục 5.4 báo cáo): bảng → CNN; target thay đổi → target network; cập nhật ngay → replay buffer.
- Một dòng nhấn: "Cùng công thức, chỉ thay nơi lưu Q".

### Phần 4 — Kết quả chính (người B và C, ~5 phút)

**Slide 10. Q-learning đạt đúng lời giải tối ưu**
- Hình trái: `results/tabular/figures/q_convergence_obstacle_4x4_s00.png`.
- Phải: 2 con số lớn: "**0%** khoảng cách tới tối ưu (8/8 điểm xuất phát)" và "**V\* = 16,969**, khớp tính tay".
- Nguồn: 5 seed; bảng E6.

**Slide 11. Bảng Q bùng nổ khi phòng lớn dần**
- Hình: `results/figures/state_explosion.png` (to, gần hết slide).
- Một dòng: "4×5 đã có 1,4 triệu trạng thái; 8×8 ~10²¹ → cần mạng nơ-ron".
- Ghi chú người nói: phòng trống 6×6 Q-learning vẫn học được nhưng bảng tốn 746 MB và chỉ dùng cho đúng một phòng.

**Slide 12. DQN vượt cách làm thủ công trên phòng 8×8**
- GIF lớn: `results/demo/compare_furniture_8x8.gif` (4 robot đua).
- Dưới, 3 con số xếp ngang: DQN **51,9** bước · greedy 58,9 · zigzag 80,0 (cận dưới 51).

**Slide 13. Thách thức: căn phòng chưa từng thấy** *(slide chuyển ý, để khán giả thấy vấn đề)*
- Con số lớn màu đỏ: "**14%**" — DQN bản đầu chỉ dọn được 14% phòng lạ 8×8.
- Một dòng: "Robot dành gần nửa số bước để đâm tường".

**Slide 14. Ba cải tiến đưa 14% lên 98%**
- Biểu đồ bậc thang hoặc 4 cột (vẽ trong PowerPoint): v2 **14%** → v3 action masking **37%** → v4 quan sát lấy robot làm tâm + huấn luyện song song + chọn mô hình **98%**.
- Dưới mỗi cột một dòng giải thích ngắn.
- Ô nhấn: "Cải tiến quyết định: quan sát lấy robot làm tâm (A/B: 60% → 91%)".

**Slide 15. DQN dọn phòng lạ nhanh hơn greedy**
- Trái: GIF `results/demo/unseen_rooms.gif`.
- Phải, bảng nhỏ (50 phòng lạ, 5 seed):

| | Đã dọn | Dọn xong | Bước khi xong |
|---|---|---|---|
| DQN v4 | 98% | 80% | **69,0** |
| greedy BFS | 100% | 100% | 71,4 |
| zigzag | 100% | 100% | 80,8 |

- Dòng nhỏ: "Phòng lạ 6×6: DQN dọn xong 99%, 33,8 bước (greedy 38,8)".

### Phần 5 — Thí nghiệm hiểu sâu (người D, ~2 phút)

**Slide 16. Hai bài học về thiết kế bài toán RL**
- Chia đôi:
  - Trái "**E1 — Tính Markov**": bỏ thông tin ô đã dọn → 0% dọn xong (đủ thông tin: 100%). Hình nhỏ: `results/ablation/figures/e1_markov_obstacle_4x4.png`.
  - Phải "**E2 — Reward hacking**": GIF `results/ablation/gifs/e2_reward_hacking_obstacle_4x4.gif`, dòng "dọn 92% rồi đi qua đi lại để ăn điểm, cả 5 seed".
- Dòng nhỏ dưới: "Shaping thế năng: 0/4.676 trạng thái đổi lời giải tối ưu — đúng định lý Ng et al. 1999".

**Slide 17. Khái quát: càng nhiều phòng huấn luyện càng tốt**
- Hình: `results/generalization/figures/e4_generalization_6x6_mask_ego_ne16_bs256_tf16.png`.
- 1 dòng: "1 phòng: học thuộc (61% phòng lạ) → vô hạn phòng: 100% phòng lạ".
- Ghi chú người nói: E3 — phòng càng nhiều vật cản càng khó (30%: dọn xong 34%).

### Phần 6 — Đánh giá và kết luận (người C, ~1,5 phút)

**Slide 18. Kết luận, hạn chế, hướng phát triển**
- Ba cột đều nhau, tiêu đề cột nhỏ in hoa:
  - **Đạt được:** tối ưu tuyệt đối ở phòng nhỏ; thắng greedy ở phòng 8×8; khái quát sang phòng lạ.
  - **Hạn chế:** phòng lạ nhiều vật cản (34% dọn xong); mô phỏng đơn giản hóa; mỗi mô hình một kích thước.
  - **Hướng tiếp:** chuyển sang greedy khi phát hiện lặp vòng; huấn luyện lâu hơn; pin, vật cản động, robot thật.
- Dòng cuối nhỏ: link GitHub, tài liệu tham khảo chính (Sutton & Barto 2018; Mnih et al. 2015).

**(Slide cuối) Cảm ơn — Hỏi đáp**
- Chỉ một dòng lớn ở giữa và GIF nhỏ `results/demo/progress_qlearning.gif` chạy lặp.

---

## 4. Slide dự phòng cho hỏi đáp

Đặt sau slide cảm ơn, chỉ mở khi được hỏi.

| # | Câu hỏi dự đoán | Nội dung slide |
|---|---|---|
| D1 | Q-learning khác SARSA? | Hai công thức đặt chồng nhau, khoanh chỗ `max_a Q(S',a)` vs `Q(S',A')` |
| D2 | Terminated và truncated? | Sơ đồ 2 nhánh: dọn xong → target = R; hết bước → vẫn cộng γ·max Q |
| D3 | Siêu tham số chọn thế nào? | Bảng A/B 4 cấu hình (`results/tuning/`, `results/v4_ab/`) |
| D4 | Chọn mô hình có gian lận không? | Sơ đồ 3 tập: huấn luyện / kiểm định (chọn mô hình) / test 50 phòng (chỉ báo cáo) |
| D5 | Bảng E6 đầy đủ | Bảng khoảng cách tối ưu 8 điểm xuất phát (`results/optimality/e6_optimality_gap.md`) |

---

## 5. Danh sách hình cần chuẩn bị

| Slide | File | Ghi chú |
|---|---|---|
| 1 | `results/demo/compare_furniture_8x8.png` | làm mờ 15% |
| 4 | `results/demo/compare_obstacle_4x4.png` | cắt lấy 1 phòng |
| 10 | `results/tabular/figures/q_convergence_obstacle_4x4_s00.png` | |
| 11 | `results/figures/state_explosion.png` | |
| 12 | `results/demo/compare_furniture_8x8.gif` | GIF, 1,4 MB |
| 15 | `results/demo/unseen_rooms.gif` | GIF, 2,1 MB |
| 16 | `results/ablation/figures/e1_markov_obstacle_4x4.png`, `results/ablation/gifs/e2_reward_hacking_obstacle_4x4.gif` | |
| 17 | `results/generalization/figures/e4_generalization_6x6_mask_ego_ne16_bs256_tf16.png` | |
| cuối | `results/demo/progress_qlearning.gif` | |

Hình vẽ bằng matplotlib có tiêu đề riêng ở trên. Khi đặt vào slide, nên **cắt bỏ tiêu đề của hình** để không trùng với tiêu đề slide. Trong PowerPoint dùng Crop, kéo cạnh trên xuống.

---

## 6. Phân công gợi ý

| Người | Slide | Cần đọc kỹ |
|---|---|---|
| D | 1–3, 16–17 | E1, E2, E4 trong báo cáo; Phần A của plan |
| A | 4–6 | `env/vacuum_env.py`, mục 4 báo cáo |
| B | 7–8, 10–11 | `tabular/q_learning.py`, Value Iteration, mục 8.2–8.3 |
| C | 9, 12–15, 18 | `deep/dqn.py` (bảng đối chiếu đầu file), mục 5.4–5.5, 8.6–8.9 |

Mỗi người tập trả lời 2 câu ở mục 11 của báo cáo ("Chuẩn bị vấn đáp") liên quan tới phần mình.

---

## 7. Kiểm tra trước khi nộp

- [ ] 18 slide chính + slide cảm ơn + 5 slide dự phòng
- [ ] Mọi tiêu đề là một câu kết luận
- [ ] Không slide nào quá 3 gạch đầu dòng
- [ ] Mọi số liệu khớp với `BAO_CAO_DU_AN.html`
- [ ] Mọi hình có dòng nguồn (số seed)
- [ ] GIF chạy được khi trình chiếu (thử ở chế độ Slide Show)
- [ ] Đã nhúng font hoặc máy trình chiếu có đủ font
- [ ] Thử trình bày một lượt, bấm giờ: không quá 15 phút
- [ ] Lưu thành `slides/REL301_FA26_Nhom<X>.pptx`, đưa lên Drive cùng thư mục code

---

## 8. Dựng lại file pptx bằng script

File pptx được sinh tự động từ kết quả thật, nên khi số liệu thay đổi chỉ cần chạy lại:

```powershell
# từ gốc dự án
python slides/build/prepare_assets.py          # cắt/vẽ lại hình cho slide vào slides/build/assets/
cd slides/build
npm install                                    # lần đầu: cài pptxgenjs cục bộ
node build_deck.js                             # ghi slides/REL301_FA26_RobotHutBui_RL.pptx
```

Chỉnh nội dung slide trong `slides/build/build_deck.js` (mỗi slide một khối `{ ... }`, có ghi chú người nói).
