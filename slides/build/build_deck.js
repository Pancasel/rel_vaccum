// Dựng slide thuyết trình REL301 (pptx) từ kết quả thật trong results/.
// Chạy:  python slides/build/prepare_assets.py   rồi   node slides/build/build_deck.js
// Kết quả: slides/REL301_FA26_RobotHutBui_RL.pptx
//
// Phong cách: học thuật, thoáng — nền trắng, một ý mỗi slide, hình lớn; màu lấy từ chính mô phỏng
// (ô đã dọn xanh, robot đỏ). Họa tiết lặp lại: ô vuông nhỏ màu "đã dọn" trước nhãn mục.

const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");

const ASSETS = path.join(__dirname, "assets");
const SIZES = JSON.parse(fs.readFileSync(path.join(ASSETS, "sizes.json"), "utf-8"));
const OUT = path.join(__dirname, "..", "REL301_FA26_RobotHutBui_RL.pptx");

const THEME = {
  name: "REL301 Robot Hut Bui",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1D2733", lt1: "FFFFFF", dk2: "5B6878", lt2: "EEF4FA",
    accent1: "1F6FB2", accent2: "C8322F", accent3: "2E7D4F", accent4: "FF9800", accent5: "4CAF50", accent6: "9E9E9E",
    hlink: "1F6FB2", folHlink: "5B6878",
  },
};
// Mã hex cho các tùy chọn chỉ nhận hex (bảng, biểu đồ, nền layout).
const HEX = { ink: "1D2733", muted: "5B6878", tint: "EEF4FA", line: "D5DEE8", blue: "1F6FB2", red: "C8322F",
  green: "2E7D4F", white: "FFFFFF", softText: "B8C7D9", tile: "A0D2FF", good: "E3F2E9" };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333" x 7.5"
pres.title = "Robot hút bụi tự học cách dọn kín căn phòng";
pres.subject = "REL301 FA26 — Q-learning và Deep Q-learning";
pres.author = "Nhóm REL301";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
// Màu cố định (hex) thay vì màu theo theme: ai chạy lại script cũng ra đúng màu, kể cả khi không áp được theme.
const C = { text1: THEME.colors.dk1, text2: THEME.colors.dk2, background1: THEME.colors.lt1, background2: THEME.colors.lt2,
  accent1: THEME.colors.accent1, accent2: THEME.colors.accent2, accent3: THEME.colors.accent3, accent4: THEME.colors.accent4,
  accent5: THEME.colors.accent5, accent6: THEME.colors.accent6 };
const W = 13.333;
const MARGIN = 0.6;

// ------------------------------------------------------------------ layouts
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: HEX.ink },
  objects: [
    { rect: { x: 0.8, y: 1.38, w: 0.16, h: 0.16, fill: { color: HEX.tile } } },
    { placeholder: { options: { name: "kicker", type: "body", x: 1.05, y: 1.25, w: 8, h: 0.4, fontSize: 14, color: HEX.softText, charSpacing: 2, margin: 0 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 1.8, w: 7.6, h: 2.0, fontFace: THEME.headFontFace, fontSize: 40, bold: true, color: HEX.white, align: "left", valign: "top", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "subtitle", type: "body", x: 0.8, y: 3.95, w: 7.6, h: 0.9, fontSize: 20, color: HEX.softText, valign: "top", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "team", type: "body", x: 0.8, y: 5.55, w: 7.6, h: 1.1, fontSize: 14, color: HEX.softText, valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: HEX.white },
  margin: [0.5, MARGIN, 0.6, MARGIN],
  objects: [
    { rect: { x: MARGIN, y: 0.44, w: 0.13, h: 0.13, fill: { color: HEX.tile } } },
    { placeholder: { options: { name: "kicker", type: "body", x: 0.84, y: 0.33, w: 9, h: 0.34, fontSize: 12, color: HEX.muted, charSpacing: 2, margin: 0 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: MARGIN, y: 0.7, w: W - 2 * MARGIN, h: 0.8, fontFace: THEME.headFontFace, fontSize: 28, bold: true, color: HEX.ink, align: "left", valign: "top", margin: 0 }, text: "" } },
    { text: { text: "REL301 · Robot hút bụi học tăng cường", options: { x: MARGIN, y: 7.0, w: 6, h: 0.3, fontSize: 11, color: HEX.muted, margin: 0 } } },
  ],
  slideNumber: { x: W - MARGIN - 0.8, y: 7.0, w: 0.8, h: 0.3, fontSize: 11, color: HEX.muted, align: "right" },
});
pres.defineSlideMaster({
  title: "END_DARK",
  background: { color: HEX.ink },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.3, w: 6.6, h: 1.4, fontFace: THEME.headFontFace, fontSize: 44, bold: true, color: HEX.white, align: "left", valign: "top", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "subtitle", type: "body", x: 0.8, y: 3.85, w: 6.6, h: 1.6, fontSize: 18, color: HEX.softText, valign: "top", margin: 0 }, text: "" } },
  ],
});

// ------------------------------------------------------------------ helpers
let currentSection = "";
function section(title) { pres.addSection({ title }); currentSection = title; }

function contentSlide(kicker, title, notes) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: currentSection });
  s.addText(kicker.toUpperCase(), { placeholder: "kicker" });
  s.addText(title, { placeholder: "title" });
  if (notes) s.addNotes(notes);
  return s;
}

function fit(name, x, y, w, h, align = "center", valign = "top") {
  const [pw, ph] = SIZES[name];
  const r = Math.min(w / pw, h / ph);
  const iw = pw * r, ih = ph * r;
  const dx = align === "center" ? (w - iw) / 2 : align === "right" ? w - iw : 0;
  const dy = valign === "middle" ? (h - ih) / 2 : valign === "bottom" ? h - ih : 0;
  return { x: x + dx, y: y + dy, w: iw, h: ih };
}

function image(s, name, x, y, w, h, opts = {}) {
  const box = fit(name, x, y, w, h, opts.align, opts.valign);
  s.addImage({ path: path.join(ASSETS, name), ...box, altText: opts.alt || name, objectName: opts.objectName || name });
  return box;
}

function text(s, value, x, y, w, h, opts = {}) {
  s.addText(value, { x, y, w, h, isTextBox: true, margin: 0, fontSize: 18, color: C.text1, valign: "top", ...opts });
}

function caption(s, value, x, y, w, opts = {}) {
  text(s, value, x, y, w, 0.3, { fontSize: 12, color: C.text2, italic: true, ...opts });
}

function card(s, x, y, w, h, name, fill = C.background2) {
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h, fill: { color: fill }, line: { color: fill, width: 0 }, rectRadius: 0.08, objectName: name });
}

function stat(s, x, y, w, value, label, color = C.accent1, size = 54) {
  text(s, value, x, y, w, 0.95, { fontSize: size, bold: true, color, fontFace: THEME.bodyFontFace, valign: "bottom" });
  text(s, label, x, y + 1.0, w, 0.85, { fontSize: 15, color: C.text2 });
}

function bullets(s, items, x, y, w, h, size = 18) {
  s.addText(items.map((t, i) => ({ text: t, options: { bullet: { indent: 18 }, breakLine: i < items.length - 1 } })),
    { x, y, w, h, isTextBox: true, margin: 0, fontSize: size, color: C.text1, paraSpaceAfter: 10, valign: "top" });
}

function chip(s, x, y, label) {
  s.addShape(pres.ShapeType.roundRect, { x, y, w: 0.42, h: 0.42, fill: { color: C.accent1 }, line: { color: C.accent1, width: 0 }, rectRadius: 0.06 });
  text(s, label, x, y, 0.42, 0.42, { fontSize: 15, bold: true, color: C.background1, align: "center", valign: "middle" });
}

const TH = { bold: true, fill: { color: HEX.tint }, color: HEX.ink, fontSize: 15 };
const TD = { color: HEX.ink, fontSize: 15 };
function table(s, rows, x, y, w, colW, opts = {}) {
  const body = rows.map((row, r) => row.map((cell) => {
    const base = r === 0 ? TH : TD;
    return typeof cell === "string" ? { text: cell, options: { ...base } } : { text: cell.text, options: { ...base, ...cell.options } };
  }));
  s.addTable(body, { x, y, w, colW, border: { type: "solid", color: HEX.line, pt: 0.75 }, margin: [4, 8, 4, 8], valign: "middle", rowH: opts.rowH || 0.42, fontFace: THEME.bodyFontFace, ...opts });
}
const best = (t) => ({ text: t, options: { bold: true, color: HEX.green, fill: { color: HEX.good } } });
const weak = (t) => ({ text: t, options: { bold: true, color: HEX.red } });

// Lưới mini (vẽ bằng hình vuông) cho slide "robot nhìn thấy gì".
const MAP4 = ["....", ".#..", "...#", "#..."];
const CLEANED4 = [[0, 0], [0, 1], [0, 2], [1, 0], [2, 0], [2, 1]];
function miniGrid(s, x, y, cell, kind) {
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) {
    let on = false;
    if (kind === "robot") on = r === 2 && c === 1;
    if (kind === "cleaned") on = CLEANED4.some(([a, b]) => a === r && b === c);
    if (kind === "obstacle") on = MAP4[r][c] === "#";
    const color = !on ? "F4F6F8" : kind === "robot" ? HEX.red : kind === "cleaned" ? HEX.tile : HEX.ink;
    s.addShape(pres.ShapeType.rect, { x: x + c * cell, y: y + r * cell, w: cell, h: cell, fill: { color }, line: { color: "FFFFFF", width: 1.5 } });
  }
}

// ================================================================== Phần 1: ứng dụng và động lực
section("1. Ứng dụng và động lực");
{
  const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: currentSection });
  s.addText("REL301 · FA26 · HÌNH THỨC APPLICATIONS", { placeholder: "kicker" });
  s.addText("Robot hút bụi tự học cách dọn kín căn phòng", { placeholder: "title" });
  s.addText("Q-learning và Deep Q-learning cho bài toán phủ kín không gian", { placeholder: "subtitle" });
  s.addText([
    { text: "Nhóm: (điền tên/số nhóm)", options: { breakLine: true } },
    { text: "Thành viên: (điền họ tên – MSSV)", options: { breakLine: true } },
    { text: "Giảng viên: (điền tên giảng viên)" },
  ], { placeholder: "team" });
  s.addShape(pres.ShapeType.roundRect, { x: 9.0, y: 1.25, w: 3.6, h: 4.95, fill: { color: HEX.white }, line: { color: HEX.white, width: 0 }, rectRadius: 0.1, objectName: "photo frame" });
  image(s, "room_8x8_dqn.png", 9.15, 1.4, 3.3, 4.65, { valign: "middle", alt: "Robot DQN dọn xong phòng 8x8 có đồ đạc" });
  text(s, "DQN dọn phòng 8×8 có đồ đạc", 9.0, 6.35, 3.6, 0.35, { fontSize: 12, color: HEX.softText, align: "center", italic: true });
  s.addNotes("Chào thầy và các bạn. Nhóm em trình bày dự án dạy một robot hút bụi ảo tự học cách dọn hết căn phòng với ít bước nhất, bằng Q-learning cho phòng nhỏ và Deep Q-learning cho phòng lớn và phòng chưa từng thấy.");
}
{
  const s = contentSlide("Ứng dụng", "Phủ kín không gian là bài toán thực tế và khó",
    "Robot hút bụi, máy cắt cỏ, drone khảo sát đều phải đi qua mọi điểm của một vùng với quãng đường ngắn nhất. Tìm đường đi qua mọi ô ngắn nhất gần với bài toán đường đi Hamilton, thuộc lớp NP-khó, nên không có công thức nhanh cho phòng lớn. Robot rẻ tiền thường đi zigzag hoặc đi tới chỗ bẩn gần nhất; cả hai lãng phí khi phòng có nhiều đồ đạc.");
  const rows = [["1", "Robot hút bụi, lau sàn", "nhà ở, văn phòng, bệnh viện"],
                ["2", "Máy cắt cỏ, máy nông nghiệp", "cắt cỏ, phun thuốc theo luống"],
                ["3", "Drone, robot kiểm tra", "khảo sát bản đồ, tấm pin mặt trời"]];
  rows.forEach(([n, head, sub], i) => {
    const y = 1.95 + i * 1.35;
    chip(s, MARGIN, y, n);
    text(s, head, MARGIN + 0.65, y - 0.05, 6.2, 0.45, { fontSize: 20, bold: true });
    text(s, sub, MARGIN + 0.65, y + 0.42, 6.2, 0.4, { fontSize: 16, color: C.text2 });
  });
  card(s, 7.7, 1.85, 5.03, 4.35, "stat card");
  text(s, "NP-khó", 8.1, 2.15, 4.3, 1.1, { fontSize: 60, bold: true, color: C.accent2 });
  text(s, "Đường đi qua mọi ô ngắn nhất gần với bài toán đường đi Hamilton: không có công thức nhanh cho phòng lớn.", 8.1, 3.35, 4.3, 1.2, { fontSize: 17 });
  text(s, "Robot rẻ tiền đi zigzag hoặc tới chỗ bẩn gần nhất, lãng phí khi phòng nhiều đồ đạc.", 8.1, 4.75, 4.3, 1.1, { fontSize: 16, color: C.text2 });
}
{
  const s = contentSlide("Câu hỏi nghiên cứu", "Robot có tự học được cách dọn tốt hơn cách làm thủ công",
    "Câu hỏi xuyên suốt: robot có thể tự học một chiến lược dọn tốt hơn zigzag hay đi tới chỗ bẩn gần nhất không, kể cả ở căn phòng chưa từng thấy? Bài trình bày đi theo ba bước: phòng nhỏ với Q-learning, phòng lớn với DQN, rồi phòng chưa từng thấy.");
  text(s, "“Robot có thể tự học một chiến lược dọn kín căn phòng tốt hơn cách làm thủ công — kể cả ở căn phòng nó chưa từng thấy?”",
    MARGIN, 1.85, 12.1, 1.4, { fontSize: 26, italic: true, fontFace: THEME.headFontFace, color: C.text1 });
  const steps = [["Phòng nhỏ", "3×3, 4×4", "Q-learning dạng bảng"], ["Phòng lớn", "8×8 có đồ đạc", "DQN (Deep Q-learning)"], ["Phòng lạ", "chưa từng thấy", "DQN v4 + 3 cải tiến"]];
  steps.forEach(([head, size, method], i) => {
    const x = MARGIN + i * 4.15;
    card(s, x, 3.75, 3.55, 2.35, `step ${i + 1}`);
    text(s, head, x + 0.3, 3.95, 3.0, 0.5, { fontSize: 22, bold: true, color: C.accent1 });
    text(s, size, x + 0.3, 4.5, 3.0, 0.4, { fontSize: 16, color: C.text2 });
    text(s, method, x + 0.3, 5.15, 3.0, 0.7, { fontSize: 18, bold: true });
    if (i < 2) s.addShape(pres.ShapeType.rightArrow, { x: x + 3.62, y: 4.75, w: 0.48, h: 0.36, fill: { color: C.accent6 }, line: { color: C.accent6, width: 0 } });
  });
}

// ================================================================== Phần 2: định hình bài toán RL
section("2. Định hình bài toán RL");
{
  const s = contentSlide("Môi trường mô phỏng", "Căn phòng là một lưới ô, robot đi từng bước",
    "Nhóm tự viết môi trường bằng Python theo chuẩn Gymnasium. Căn phòng là ma trận 0 là sàn, 1 là đồ đạc. Mỗi bước robot đi lên, xuống, trái hoặc phải một ô; ô đi qua thành đã dọn. Toàn bộ quy luật nằm trong một hàm transition thuần, dùng chung cho môi trường và cho Value Iteration nên hai bên không bao giờ tính phần thưởng khác nhau.");
  card(s, MARGIN, 1.75, 4.2, 4.95, "image card");
  image(s, "room_4x4_optimal.png", MARGIN + 0.35, 1.95, 3.5, 4.55, { valign: "middle", alt: "Đường đi tối ưu trên phòng 4x4 có vật cản" });
  const legend = [[HEX.white, "F0F0F0", "Sàn chưa dọn"], [HEX.tile, HEX.tile, "Đã dọn"], [HEX.ink, HEX.ink, "Đồ đạc, tường"], [HEX.red, HEX.red, "Robot"]];
  legend.forEach(([, color, label], i) => {
    const y = 1.95 + i * 0.5;
    s.addShape(pres.ShapeType.rect, { x: 5.35, y: y + 0.06, w: 0.28, h: 0.28, fill: { color }, line: { color: HEX.line, width: 0.75 } });
    text(s, label, 5.8, y, 3, 0.4, { fontSize: 17 });
  });
  bullets(s, [
    "Lưới H×W: 0 = sàn, 1 = đồ đạc; bản đồ ngẫu nhiên luôn dọn hết được",
    "Mỗi bước: lên, xuống, trái, phải một ô; đụng tường thì đứng yên",
    "Tự viết theo chuẩn Gymnasium, qua bộ kiểm tra check_env",
    "Một hàm transition() dùng chung cho môi trường và đáp án tối ưu",
  ], 5.35, 4.15, 7.3, 2.6, 17);
  caption(s, "Hình: đường đi tối ưu (Value Iteration) trên phòng 4×4 có vật cản — 13 bước", MARGIN, 6.72, 6);
}
{
  const s = contentSlide("Quá trình quyết định Markov", "Bài toán được định hình thành một MDP rõ ràng",
    "Trạng thái gồm vị trí robot, tập ô đã dọn và bản đồ vật cản. Phần thưởng cấu hình dense: ô mới cộng 1, ô cũ trừ 0,1, đụng tường trừ 0,5, mỗi bước trừ 0,01, dọn xong cộng 10. Gamma 0,99 vì robot cần nhìn trước nhiều bước. Điểm quan trọng: trạng thái phải chứa thông tin ô nào đã dọn, nếu không sẽ mất tính Markov.");
  table(s, [
    ["Thành phần", "Định nghĩa trong dự án"],
    ["Trạng thái", "vị trí robot + tập ô đã dọn + bản đồ vật cản"],
    ["Hành động", "lên, xuống, trái, phải"],
    ["Phần thưởng", "ô mới +1 · ô cũ −0,1 · đụng tường −0,5 · mỗi bước −0,01 · dọn xong +10"],
    ["Kết thúc", "terminated: dọn hết · truncated: quá 4·|F| bước (hết pin)"],
    ["Chiết khấu γ", "0,99 — cần nhìn trước hàng chục bước để dọn hết"],
  ], MARGIN, 1.85, 8.3, [2.1, 6.2], { rowH: 0.62 });
  card(s, 9.3, 1.85, 3.43, 3.72, "markov card");
  text(s, "Vì sao?", 9.6, 2.1, 2.9, 0.5, { fontSize: 20, bold: true, color: C.accent2 });
  text(s, "Trạng thái phải chứa ô nào đã dọn. Cùng vị trí nhưng khác tiến độ thì phần thưởng tương lai khác nhau — thiếu nó là mất tính Markov (S&B 3.1).", 9.6, 2.7, 2.85, 2.7, { fontSize: 17 });
}
{
  const s = contentSlide("Quan sát", "Robot nhìn căn phòng như một ảnh ba lớp",
    "Mạng nơ-ron nhận một tensor ba kênh giống một ảnh: kênh vị trí robot, kênh ô đã dọn, kênh vật cản. Q-learning dạng bảng thì mã hóa cùng thông tin thành khóa hàng, cột và một số nguyên mà mỗi bit là một ô đã dọn hay chưa.");
  const labels = [["robot", "Kênh 0: vị trí robot"], ["cleaned", "Kênh 1: ô đã dọn"], ["obstacle", "Kênh 2: vật cản"]];
  labels.forEach(([kind, label], i) => {
    const x = MARGIN + i * 2.85;
    miniGrid(s, x, 1.95, 0.62, kind);
    text(s, label, x, 4.6, 2.48, 0.4, { fontSize: 16, align: "center", color: C.text2 });
  });
  s.addShape(pres.ShapeType.rightArrow, { x: 9.0, y: 3.0, w: 0.55, h: 0.42, fill: { color: C.accent6 }, line: { color: C.accent6, width: 0 } });
  card(s, 9.75, 1.95, 2.98, 1.75, "table card");
  text(s, "Bảng Q — phòng nhỏ", 10.0, 2.12, 2.6, 0.4, { fontSize: 18, bold: true, color: C.accent1 });
  text(s, "khóa (hàng, cột, mask), mỗi bit của mask là một ô đã dọn", 10.0, 2.6, 2.6, 1.0, { fontSize: 15 });
  card(s, 9.75, 3.9, 2.98, 1.75, "cnn card");
  text(s, "Mạng CNN — phòng lớn", 10.0, 4.07, 2.6, 0.4, { fontSize: 18, bold: true, color: C.accent1 });
  text(s, "tensor 3×H×W, đọc như một ảnh nhỏ", 10.0, 4.55, 2.6, 1.0, { fontSize: 15 });
  text(s, "Ví dụ: phòng 4×4 sau 5 bước — robot ở hàng 3, cột 2; đã dọn 6 ô; 3 ô là đồ đạc.", MARGIN, 5.55, 8.4, 0.5, { fontSize: 16, color: C.text2 });
}

// ================================================================== Phần 3: phương án và lựa chọn
section("3. Phương án giải và lựa chọn");
{
  const s = contentSlide("Phương pháp", "Bốn nhóm phương pháp, mỗi nhóm một vai trò",
    "Nhóm không dùng hết các thuật toán trong sách. Q-learning là mô hình chính. DQN là Q-learning dùng mạng nơ-ron cho phòng lớn. Value Iteration không học mà tính lời giải tối ưu tuyệt đối để chấm điểm. Ba baseline là cách làm thủ công để so sánh. Nhóm bỏ SARSA, Double Q, PPO để tập trung một mạch.");
  table(s, [
    ["Vai trò", "Phương pháp", "Vì sao chọn"],
    [{ text: "Mô hình chính", options: { bold: true } }, "Q-learning dạng bảng (S&B 6.5)", "đơn giản, đúng giáo trình, học được lời giải tối ưu"],
    [{ text: "Mở rộng", options: { bold: true } }, "DQN = Q-learning + mạng CNN", "phòng 8×8 có ~10²¹ trạng thái, bảng không chứa nổi"],
    ["Đáp án", "Value Iteration (S&B 4.4)", "lời giải tối ưu tuyệt đối để chấm điểm, không học"],
    ["Đối chứng", "đi ngẫu nhiên, zigzag, greedy BFS", "cách làm thủ công của robot thông thường"],
  ], MARGIN, 1.9, 12.1, [2.2, 4.2, 5.7], { rowH: 0.7 });
  caption(s, "Không dùng hết sách: bỏ SARSA, Double Q-learning, PPO để tập trung vào một mạch Q-learning → DQN.", MARGIN, 5.75, 12.1, { fontSize: 14 });
}
{
  const s = contentSlide("Q-learning", "Q-learning học từ một công thức cập nhật duy nhất",
    "Q của S, A là tổng phần thưởng kỳ vọng nếu ở trạng thái S làm hành động A rồi sau đó đi tốt nhất. Sau mỗi bước robot cộng thêm vào Q một phần sai số giữa mục tiêu và ước lượng cũ. Mục tiêu dùng max theo a của Q tại trạng thái sau, nên Q-learning là off-policy. Khi dọn xong thì không cộng giá trị tương lai; khi bị cắt do hết bước thì vẫn cộng.");
  card(s, MARGIN, 1.95, 12.1, 1.5, "formula card");
  s.addText([
    { text: "Q(S, A)  ←  Q(S, A)  +  α [ ", options: { color: C.text1 } },
    { text: "R", options: { color: C.accent2, bold: true } },
    { text: "  +  γ · ", options: { color: C.text1 } },
    { text: "maxₐ Q(S′, a)", options: { color: C.accent1, bold: true } },
    { text: "  −  Q(S, A) ]", options: { color: C.text1 } },
  ], { x: MARGIN + 0.3, y: 2.15, w: 11.5, h: 1.1, isTextBox: true, margin: 0, fontSize: 32, fontFace: "Cambria Math", align: "center", valign: "middle" });
  const parts = [[C.accent2, "R", "phần thưởng vừa nhận sau bước đi"],
                 [C.accent1, "γ · maxₐ Q(S′, a)", "giá trị tốt nhất có thể đạt được từ trạng thái sau"],
                 [C.text1, "[ … ] · α", "sai số cần sửa, nhân với tốc độ học α = 0,5"]];
  parts.forEach(([color, head, body], i) => {
    const x = MARGIN + i * 4.1;
    text(s, head, x, 3.85, 3.8, 0.5, { fontSize: 20, bold: true, color, fontFace: "Cambria Math" });
    text(s, body, x, 4.4, 3.7, 0.9, { fontSize: 16 });
  });
  text(s, "ε-greedy: với xác suất ε chọn ngẫu nhiên để khám phá (ε giảm từ 1 xuống 0,05), còn lại chọn hành động có Q lớn nhất.", MARGIN, 5.6, 12.1, 0.8, { fontSize: 16, color: C.text2 });
}
{
  const s = contentSlide("Deep Q-learning", "DQN giữ nguyên công thức, chỉ thay nơi lưu Q",
    "DQN là đúng thuật toán Q-learning nhưng thay bảng bằng mạng CNN nhỏ. Hai thay đổi đi kèm: target network giữ mục tiêu ổn định vì mục tiêu thay đổi mỗi bước làm học bất ổn, và replay buffer lấy mẫu ngẫu nhiên để phá tương quan giữa các bước liên tiếp.");
  table(s, [
    ["Q-learning dạng bảng", "DQN", "Vì sao đổi"],
    ["Q[S][A] tra trong bảng", "QNetwork(obs)[A] do CNN tính", "quá nhiều trạng thái; mạng khái quát được (S&B 9.1)"],
    ["mục tiêu R + γ·max Q(S′, a)", "cùng công thức, max trên target network", "mục tiêu thay đổi mỗi bước gây bất ổn (S&B 11.3)"],
    ["cập nhật ngay sau mỗi bước", "lưu vào replay buffer, học trên lô ngẫu nhiên", "các bước liên tiếp tương quan mạnh"],
  ], MARGIN, 1.9, 12.1, [3.5, 4.1, 4.5], { rowH: 0.75 });
  card(s, MARGIN, 5.25, 12.1, 0.9, "takeaway card");
  text(s, "Mạng CNN 2 lớp tích chập + 1 lớp ẩn, ~268 nghìn tham số; learning rate 2,5·10⁻⁴, buffer 100k, chép target network mỗi 500 bước.", MARGIN + 0.3, 5.45, 11.5, 0.6, { fontSize: 16 });
}

// ================================================================== Phần 4: kết quả chính
section("4. Kết quả");
{
  const s = contentSlide("Kết quả · Q-learning", "Q-learning đạt đúng lời giải tối ưu",
    "Trên phòng 3×3 và 4×4, Q-learning đạt đúng số bước tối thiểu ở cả 8 điểm xuất phát thử nghiệm, độ lệch chuẩn bằng 0 qua 5 seed. Giá trị max Q tại trạng thái xuất phát hội tụ về đúng V* mà Value Iteration tính; trên 3×3 đó là 16,969, khớp giá trị tính tay.");
  image(s, "q_convergence.png", MARGIN, 1.8, 7.6, 4.6, { valign: "middle", alt: "max Q(s0) hội tụ về V*" });
  caption(s, "Phòng 4×4 có vật cản, xuất phát (0,0); trung bình ± std qua 5 seed", MARGIN, 6.5, 7.6);
  stat(s, 8.75, 1.95, 4.0, "0%", "khoảng cách tới tối ưu ở 8/8 điểm xuất phát (3×3, 4×4)");
  stat(s, 8.75, 4.05, 4.0, "16,969", "V* học được trên 3×3, khớp đáp án tính tay", C.accent3);
}
{
  const s = contentSlide("Giới hạn của bảng Q", "Bảng Q bùng nổ khi căn phòng lớn dần",
    "Số trạng thái tăng theo hàm mũ với số ô trống: lưới 4×5 đã có 1,4 triệu trạng thái đạt tới được, phòng 8×8 cận trên khoảng 10 mũ 21. Với phòng trống 6×6, Q-learning vẫn học được nhưng bảng tốn khoảng 746 MB và chỉ dùng được cho đúng một căn phòng, một điểm xuất phát. Đó là lý do cần mạng nơ-ron.");
  image(s, "state_explosion.png", MARGIN, 1.8, 8.2, 4.7, { valign: "middle", alt: "Bùng nổ không gian trạng thái" });
  stat(s, 9.3, 1.85, 3.43, "1,4 triệu", "trạng thái ở lưới 4×5", C.accent1, 44);
  stat(s, 9.3, 3.45, 3.43, "~10²¹", "cận trên ở phòng 8×8", C.accent2, 44);
  stat(s, 9.3, 5.05, 3.43, "746 MB", "bảng Q cho một phòng 6×6", C.text1, 44);
}
{
  const s = contentSlide("Kết quả · DQN", "DQN dọn phòng 8×8 nhanh hơn cách làm thủ công",
    "Bốn robot cùng dọn một phòng 8×8 có đồ đạc từ cùng điểm xuất phát. Qua 5 seed với điểm xuất phát ngẫu nhiên, DQN dọn xong trong 51,9 bước, so với greedy BFS 58,9 và zigzag 80,0. Phòng có 52 ô trống nên không thể dọn ít hơn 51 bước: DQN gần như tối ưu.");
  image(s, "race_8x8.gif", MARGIN, 1.75, 12.1, 3.75, { alt: "Bốn robot cùng dọn phòng 8x8" });
  const stats = [["51,9", "bước — DQN", C.accent2], ["58,9", "bước — greedy BFS", C.accent5], ["80,0", "bước — zigzag", C.accent4]];
  stats.forEach(([v, l, col], i) => {
    const x = MARGIN + 0.6 + i * 4.0;
    text(s, v, x, 5.6, 3.2, 0.75, { fontSize: 40, bold: true, color: col });
    text(s, l, x, 6.3, 3.2, 0.35, { fontSize: 15, color: C.text2 });
  });
}
{
  const s = contentSlide("Thách thức", "Căn phòng chưa từng thấy là thử thách thật",
    "Học thuộc một căn phòng chưa đủ: robot thật gặp phòng mới. DQN bản đầu tiên, huấn luyện trên phòng ngẫu nhiên, chỉ dọn được 14% diện tích phòng 8×8 chưa từng thấy, và gần nửa số bước là đâm vào tường vì chính sách tất định cứ lặp lại hành động sai.");
  text(s, "14%", MARGIN, 1.9, 6, 2.4, { fontSize: 150, bold: true, color: C.accent2 });
  text(s, "diện tích phòng 8×8 chưa từng thấy được DQN bản đầu dọn sạch", MARGIN, 4.4, 5.6, 1.0, { fontSize: 22 });
  card(s, 7.3, 2.1, 5.43, 3.6, "diagnosis card");
  text(s, "Chẩn đoán", 7.65, 2.35, 4.8, 0.45, { fontSize: 20, bold: true, color: C.accent1 });
  bullets(s, [
    "≈ 47% số bước là đâm vào tường",
    "Chính sách tất định: trạng thái không đổi → hành động sai lặp mãi",
    "Mạng phải học lại cùng một quy tắc ở mọi vị trí",
  ], 7.65, 2.95, 4.8, 2.6, 17);
}
{
  const s = contentSlide("Ba cải tiến", "Ba cải tiến đưa tỷ lệ dọn phòng lạ từ 14% lên 98%",
    "Mỗi cải tiến được đo bằng A/B. Action masking bỏ các hướng đâm tường, lên 37%. Quan sát lấy robot làm tâm là cải tiến quyết định: robot luôn ở giữa ảnh nên quy tắc dùng chung cho mọi vị trí, A/B tăng từ 60% lên 91%. Huấn luyện 16 môi trường song song nhanh gấp 4 lần và chọn mô hình theo tập kiểm định cho kết quả ổn định: 98% diện tích, 80% số phòng dọn xong.");
  s.addChart(pres.ChartType.bar, [
    { name: "Đã dọn (%)", labels: ["v2 DQN gốc", "v3 + action mask", "v4 + 2 cải tiến"], values: [14, 37, 98] },
    { name: "Dọn xong (%)", labels: ["v2 DQN gốc", "v3 + action mask", "v4 + 2 cải tiến"], values: [0, 0, 80] },
  ], {
    x: MARGIN, y: 1.75, w: 6.9, h: 4.85, barDir: "col", barGrouping: "clustered", barGapWidthPct: 60,
    chartColors: [HEX.blue, HEX.red], showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 13, dataLabelColor: HEX.ink,
    dataLabelFontFace: "+mn-lt", catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt",
    catAxisLabelColor: HEX.muted, catAxisLabelFontSize: 13, valAxisLabelColor: HEX.muted, valAxisLabelFontSize: 12,
    valAxisMaxVal: 100, valAxisMinVal: 0, valGridLine: { color: "E6EBF0", size: 0.75 }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "t", legendFontSize: 13, legendColor: HEX.ink,
    showTitle: true, title: "50 phòng 8×8 chưa từng thấy, 5 seed", titleFontSize: 14, titleColor: HEX.muted, titleFontFace: "+mn-lt",
  });
  const items = [["1", "Action masking", "không chọn hướng đâm tường"],
                 ["2", "Robot làm tâm", "quy tắc dùng chung mọi vị trí — A/B 60% → 91%"],
                 ["3", "Song song + chọn mô hình", "16 môi trường, nhanh ×4; giữ bản tốt nhất trên tập kiểm định"]];
  items.forEach(([n, head, body], i) => {
    const y = 1.95 + i * 1.5;
    chip(s, 7.95, y, n);
    text(s, head, 8.6, y - 0.05, 4.1, 0.45, { fontSize: 19, bold: true });
    text(s, body, 8.6, y + 0.42, 4.1, 0.9, { fontSize: 15, color: C.text2 });
  });
}
{
  const s = contentSlide("Kết quả · phòng lạ", "Ở phòng lạ, DQN v4 dọn xong nhanh hơn greedy BFS",
    "Đo trên 50 phòng 8×8 mà mô hình chưa từng thấy, qua 5 seed. DQN v4 dọn 98% diện tích và dọn xong 80% số phòng. Khi dọn xong, DQN dùng 69,0 bước, ít hơn greedy BFS 71,4 ở cả 5 seed. Trên phòng lạ 6×6, DQN dọn xong 99% với 33,8 bước, so với greedy 38,8. Hai mươi phần trăm còn lại là robot lặp vòng ở vùng đã dọn.");
  card(s, MARGIN, 1.75, 3.25, 4.95, "gif card");
  image(s, "unseen_rooms.gif", MARGIN + 0.2, 1.85, 2.85, 4.75, { valign: "middle", alt: "DQN và greedy BFS dọn 3 phòng chưa từng thấy" });
  table(s, [
    ["50 phòng 8×8 chưa gặp", "Đã dọn", "Dọn xong", "Bước khi xong"],
    [{ text: "DQN v4", options: { bold: true } }, "98% ± 1%", "80% ± 9%", best("69,0")],
    ["greedy BFS", "100%", "100%", "71,4"],
    ["zigzag", "100%", "100%", "80,8"],
  ], 4.25, 1.95, 8.48, [2.9, 1.8, 1.8, 1.98], { rowH: 0.62 });
  card(s, 4.25, 4.75, 8.48, 1.35, "note card");
  text(s, "Phòng lạ 6×6: DQN v4 dọn xong 99% số phòng với 33,8 bước (greedy BFS 38,8; zigzag 44,2).", 4.55, 4.95, 7.9, 0.95, { fontSize: 17 });
  caption(s, "Hình: 3 phòng chưa từng thấy — DQN (trái) và greedy BFS (phải)", MARGIN, 6.75, 6);
}

// ================================================================== Phần 5: thí nghiệm hiểu sâu
section("5. Thí nghiệm");
{
  const s = contentSlide("E1 · E2", "Hai bài học về cách thiết kế bài toán RL",
    "E1: nếu trạng thái chỉ có vị trí, chính sách tất định chỉ có một hướng cho mỗi ô nên robot lặp vòng: không lần nào dọn xong, chỉ 45% diện tích. E2: với phần thưởng thưởng khi lại gần ô bẩn mà không phạt khi đi xa, robot học cách đi qua đi lại mãi để ăn điểm — reward hacking ở cả 5 seed. Shaping dạng thế năng không đổi lời giải tối ưu ở trạng thái nào, đúng định lý Ng và cộng sự 1999.");
  text(s, "E1 · Tính Markov", MARGIN, 1.75, 6.3, 0.45, { fontSize: 20, bold: true, color: C.accent1 });
  image(s, "e1_markov.png", MARGIN, 2.25, 6.3, 2.6, { align: "left", alt: "E1: state đầy đủ và chỉ vị trí" });
  text(s, "Bỏ thông tin ô đã dọn → 0% dọn xong (đủ thông tin: 100%, tối ưu 13 bước)", MARGIN, 5.0, 6.1, 0.8, { fontSize: 16 });
  text(s, "E2 · Reward hacking", 7.3, 1.75, 5.4, 0.45, { fontSize: 20, bold: true, color: C.accent2 });
  card(s, 7.3, 2.25, 2.55, 2.6, "gif card");
  image(s, "reward_hacking.gif", 7.45, 2.4, 2.25, 2.3, { valign: "middle", alt: "Robot đi qua đi lại để ăn điểm" });
  text(s, "Dọn 92% rồi đi qua đi lại để ăn điểm — cả 5 seed", 10.05, 2.35, 2.7, 2.4, { fontSize: 16 });
  card(s, MARGIN, 5.95, 12.1, 0.75, "theorem card");
  text(s, "Shaping thế năng: 0 / 4.676 trạng thái đổi lời giải tối ưu — đúng định lý Ng, Harada & Russell (1999).", MARGIN + 0.3, 6.1, 11.5, 0.5, { fontSize: 16 });
}
{
  const s = contentSlide("E4 · E3", "Càng huấn luyện trên nhiều phòng, càng khái quát tốt",
    "E4: huấn luyện DQN v4 trên một phòng thì mạng học thuộc phòng đó, chỉ dọn 61% phòng lạ; từ 100 phòng trở lên thì kết quả trên phòng lạ gần bằng trên phòng đã gặp, và huấn luyện vô hạn phòng thì dọn 100% phòng lạ. E3: phòng càng nhiều vật cản càng khó; ở mật độ 30% DQN chỉ dọn xong 34% số phòng trong khi greedy BFS luôn dọn xong.");
  image(s, "e4_generalization.png", MARGIN, 1.8, 7.6, 4.6, { valign: "middle", alt: "E4 khái quát hóa theo số phòng huấn luyện" });
  caption(s, "Phòng 6×6, 15% vật cản, 50 phòng test chưa gặp, 3 seed", MARGIN, 6.5, 7.6);
  stat(s, 8.75, 1.95, 4.0, "61% → 100%", "diện tích phòng lạ: huấn luyện 1 phòng → vô hạn phòng", C.accent1, 40);
  text(s, "E3 · Mật độ vật cản (8×8)", 8.75, 4.2, 4.0, 0.4, { fontSize: 18, bold: true });
  text(s, "Dọn xong: 100% · 87% · 61% · 34% ở mật độ 0 · 10 · 20 · 30%. Greedy BFS luôn dọn xong.", 8.75, 4.65, 3.95, 1.4, { fontSize: 16, color: C.text2 });
}

// ================================================================== Phần 6: đánh giá
section("6. Đánh giá và kết luận");
{
  const s = contentSlide("Kết luận", "Đạt được, hạn chế và hướng phát triển",
    "Tóm lại: Q-learning đạt tối ưu tuyệt đối ở phòng nhỏ; DQN thắng cách làm thủ công ở phòng 8×8 và khái quát được sang phòng chưa từng thấy. Hạn chế: phòng lạ nhiều vật cản, mô phỏng đơn giản hóa, mỗi mô hình một kích thước. Hướng tiếp: chuyển sang greedy khi phát hiện lặp vòng, huấn luyện lâu hơn, thêm pin và vật cản động, thử trên robot thật.");
  const cols = [["Đạt được", C.accent3, ["Tối ưu tuyệt đối ở phòng 3×3, 4×4", "Thắng greedy BFS ở phòng 8×8 (51,9 vs 58,9)", "Dọn 98% phòng lạ 8×8, 99% phòng lạ 6×6 dọn xong"]],
                ["Hạn chế", C.accent2, ["Phòng lạ 30% vật cản: dọn xong 34%", "Mô phỏng tất định, vật cản tĩnh", "Mỗi mô hình một kích thước phòng"]],
                ["Hướng tiếp", C.accent1, ["Chuyển sang greedy khi phát hiện lặp vòng", "Huấn luyện lâu hơn, nhiều mật độ", "Pin, vật cản động, robot thật"]]];
  cols.forEach(([head, color, items], i) => {
    const x = MARGIN + i * 4.1;
    card(s, x, 1.85, 3.8, 4.3, `${head} card`);
    text(s, head.toUpperCase(), x + 0.3, 2.05, 3.2, 0.4, { fontSize: 15, bold: true, color, charSpacing: 2 });
    bullets(s, items, x + 0.3, 2.6, 3.25, 3.4, 16);
  });
  caption(s, "Mã nguồn: github.com/Pancasel/rel_vaccum · Sutton & Barto (2018); Mnih et al. (2015); Ng, Harada & Russell (1999)", MARGIN, 6.45, 12.1);
}
{
  const s = pres.addSlide({ masterName: "END_DARK", sectionTitle: currentSection });
  s.addText("Cảm ơn thầy và các bạn", { placeholder: "title" });
  s.addText([
    { text: "Hỏi đáp", options: { breakLine: true, bold: true } },
    { text: "Báo cáo đầy đủ: BAO_CAO_DU_AN.html", options: { breakLine: true } },
    { text: "Mã nguồn: github.com/Pancasel/rel_vaccum" },
  ], { placeholder: "subtitle" });
  s.addShape(pres.ShapeType.roundRect, { x: 7.95, y: 2.25, w: 4.75, h: 2.25, fill: { color: HEX.white }, line: { color: HEX.white, width: 0 }, rectRadius: 0.08 });
  image(s, "progress_qlearning.gif", 8.05, 2.35, 4.55, 2.05, { valign: "middle", alt: "Robot học dần qua các episode" });
  text(s, "Robot học dần: episode 0 → 1.000 → 10.000 → học xong", 7.95, 4.65, 4.75, 0.4, { fontSize: 12, italic: true, color: HEX.softText, align: "center" });
  s.addNotes("Cảm ơn thầy và các bạn đã lắng nghe. Nhóm em sẵn sàng trả lời câu hỏi; các slide phụ lục phía sau dùng khi được hỏi chi tiết.");
}

// ================================================================== Phụ lục
section("Phụ lục");
{
  const s = contentSlide("Phụ lục · D1", "Q-learning và SARSA khác nhau đúng một chỗ",
    "Q-learning dùng max theo a của Q tại trạng thái sau nên học giá trị của chính sách tối ưu dù đang khám phá, gọi là off-policy. SARSA dùng Q của hành động A phẩy mà agent sẽ làm thật, nên học giá trị của chính sách epsilon-greedy đang dùng, gọi là on-policy, và thường thận trọng hơn.");
  const rows = [["Q-learning (off-policy)", "R + γ · maxₐ Q(S′, a)", C.accent1], ["SARSA (on-policy)", "R + γ · Q(S′, A′)", C.accent2]];
  rows.forEach(([head, formula, color], i) => {
    const y = 2.0 + i * 2.0;
    card(s, MARGIN, y, 12.1, 1.6, head);
    text(s, head, MARGIN + 0.35, y + 0.2, 4.5, 0.5, { fontSize: 20, bold: true, color });
    text(s, `mục tiêu = ${formula}`, MARGIN + 0.35, y + 0.75, 11, 0.65, { fontSize: 26, fontFace: "Cambria Math" });
  });
  text(s, "A′ là hành động agent sẽ thật sự làm ở S′ (chọn bằng ε-greedy).", MARGIN, 6.1, 12, 0.4, { fontSize: 16, color: C.text2 });
}
{
  const s = contentSlide("Phụ lục · D2", "Terminated và truncated được xử lý khác nhau",
    "Terminated là kết thúc thật khi đã dọn hết: giá trị tương lai bằng 0 nên mục tiêu chỉ còn R. Truncated là bị cắt ngang khi hết số bước: trạng thái sau vẫn còn giá trị nên vẫn phải cộng gamma nhân max Q. Nhầm hai trường hợp này là lỗi phổ biến nhất khi tự viết RL; dự án có test riêng cho cả hai.");
  const boxes = [["terminated — đã dọn hết", "mục tiêu = R", "trạng thái kết thúc thật, Q(terminal) = 0", C.accent3],
                 ["truncated — hết số bước", "mục tiêu = R + γ · maxₐ Q(S′, a)", "chỉ bị cắt ngang, S′ vẫn còn giá trị", C.accent4]];
  boxes.forEach(([head, formula, body, color], i) => {
    const x = MARGIN + i * 6.15;
    card(s, x, 2.0, 5.95, 3.4, head);
    text(s, head, x + 0.35, 2.25, 5.3, 0.5, { fontSize: 20, bold: true, color });
    text(s, formula, x + 0.35, 2.95, 5.3, 0.8, { fontSize: 22, fontFace: "Cambria Math" });
    text(s, body, x + 0.35, 3.95, 5.3, 1.1, { fontSize: 17, color: C.text2 });
  });
  text(s, "Replay buffer của DQN chỉ lưu cờ terminated, không lưu truncated — đúng theo nguyên tắc trên.", MARGIN, 5.8, 12.1, 0.5, { fontSize: 16 });
}
{
  const s = contentSlide("Phụ lục · D3", "Siêu tham số và cải tiến được chọn bằng thử nghiệm A/B",
    "Bảng trái: DQN v2, so bốn cấu hình trên phòng 4×4; cấu hình cũ sụp giữa chừng, cấu hình chọn ổn định. Bảng phải: v4, so bốn cấu hình trên phòng lạ 8×8; quan sát lấy robot làm tâm là cải tiến quyết định, reward shaping không giúp thêm.");
  text(s, "v2 · ổn định huấn luyện (4×4, 150k bước)", MARGIN, 1.8, 5.9, 0.4, { fontSize: 17, bold: true });
  table(s, [
    ["Cấu hình", "Kết quả"],
    ["lr 5·10⁻⁴, target 1000, buffer 50k", weak("sụp ở 30–40k bước")],
    [{ text: "lr 2,5·10⁻⁴, target 500, buffer 100k", options: { bold: true } }, best("12,67 bước (tối ưu 12,69)")],
  ], MARGIN, 2.3, 5.9, [3.3, 2.6], { rowH: 0.7 });
  text(s, "v4 · phòng lạ 8×8 (600k bước, 2 seed)", 6.85, 1.8, 5.9, 0.4, { fontSize: 17, bold: true });
  table(s, [
    ["Cấu hình (có action mask)", "Đã dọn", "Dọn xong"],
    ["quan sát gốc", "60%", "1%"],
    [{ text: "robot làm tâm", options: { bold: true } }, best("91%"), best("42%")],
    ["quan sát gốc + shaping", "59%", "0%"],
    ["robot làm tâm + shaping", "89%", "33%"],
  ], 6.85, 2.3, 5.88, [3.1, 1.39, 1.39], { rowH: 0.55 });
}
{
  const s = contentSlide("Phụ lục · D4", "Mô hình được chọn trên tập kiểm định, đánh giá trên tập test riêng",
    "Ba nhóm phòng tách biệt hoàn toàn. Phòng huấn luyện sinh ngẫu nhiên mỗi episode và loại trừ tập test. Phòng kiểm định dùng để đánh giá định kỳ và giữ lại mô hình tốt nhất. Năm mươi phòng test chỉ dùng để báo cáo kết quả cuối, không dùng để chọn mô hình — nên cách chọn này không phải gian lận.");
  const sets = [["Huấn luyện", "phòng mới mỗi episode, đã loại trừ tập test", C.accent6],
                ["Kiểm định", "đánh giá định kỳ, giữ mô hình tốt nhất (best.pt)", C.accent1],
                ["Test — 50 phòng", "chỉ dùng để báo cáo số liệu cuối", C.accent2]];
  sets.forEach(([head, body, color], i) => {
    const x = MARGIN + i * 4.15;
    card(s, x, 2.2, 3.6, 2.6, head);
    text(s, head, x + 0.3, 2.45, 3.0, 0.5, { fontSize: 21, bold: true, color });
    text(s, body, x + 0.3, 3.1, 3.0, 1.5, { fontSize: 17 });
    if (i < 2) s.addShape(pres.ShapeType.rightArrow, { x: x + 3.66, y: 3.3, w: 0.44, h: 0.36, fill: { color: C.accent6 }, line: { color: C.accent6, width: 0 } });
  });
  text(s, "Mô hình cuối (model.pt) = bản tốt nhất trên tập kiểm định; last.pt = trọng số ở bước cuối để đối chiếu.", MARGIN, 5.4, 12.1, 0.6, { fontSize: 16, color: C.text2 });
}
{
  const s = contentSlide("Phụ lục · D5", "Bảng khoảng cách tới tối ưu (E6)",
    "Số bước của từng phương pháp so với số bước tối thiểu tính bằng BFS trên đồ thị trạng thái. Q-learning và DQN đạt đúng tối ưu ở cả 8 điểm xuất phát; greedy BFS thừa tới 42%, zigzag thừa tới 50%.");
  const r = (map, start, opt, g, z) => [map, start, opt, best(`${opt} (+0%)`), best(`${opt} (+0%)`), g, z];
  table(s, [
    ["Phòng", "Xuất phát", "Tối thiểu", "Q-learning", "DQN", "greedy BFS", "zigzag"],
    r("3×3", "(0,0) góc", "8", "8 (+0%)", "8 (+0%)"),
    r("3×3", "(0,1) giữa cạnh", "9", "9 (+0%)", "9 (+0%)"),
    r("3×3", "(1,1) tâm", "8", "8 (+0%)", "10 (+25%)"),
    r("4×4", "(0,0)", "13", "16 (+23%)", "16 (+23%)"),
    r("4×4", "(0,3)", "13", "13 (+0%)", "19 (+46%)"),
    r("4×4", "(1,2)", "13", "17 (+31%)", "19 (+46%)"),
    r("4×4", "(2,0)", "12", "17 (+42%)", "12 (+0%)"),
    r("4×4", "(3,3)", "12", "15 (+25%)", "18 (+50%)"),
  ], MARGIN, 1.85, 12.1, [1.2, 2.2, 1.4, 1.9, 1.9, 1.75, 1.75], { rowH: 0.5 });
  caption(s, "Q-learning và DQN: 5 seed, độ lệch chuẩn bằng 0", MARGIN, 6.55, 12.1);
}

// ------------------------------------------------------------------ ghi file + áp theme
(async () => {
  await pres.writeFile({ fileName: OUT });
  // Tùy chọn: ghi bảng màu/tên theme vào file để bảng chọn màu trong PowerPoint khớp với slide.
  // Slide đã dùng mã màu cố định nên bỏ qua bước này vẫn hiển thị đúng.
  if (process.env.PPTX_APPLY_THEME) {
    const { applyTheme } = require(process.env.PPTX_APPLY_THEME);
    await applyTheme(OUT, THEME);
  }
  console.log("Đã tạo", OUT);
})();
