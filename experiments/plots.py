"""Vẽ hình cho báo cáo: learning curve, cột/đường có sai số, bùng nổ trạng thái, heatmap.

Quy ước (mục 7.2 kế hoạch): tiêu đề và nhãn trục tiếng Việt, màu cố định theo agent,
ghi số seed trong tiêu đề phụ, backend Agg (không mở cửa sổ).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from project_utils.project_paths import PROJECT_ROOT, require_path_inside_project, setup_local_caches

setup_local_caches()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

AGENT_COLORS: dict[str, str] = {
    "random_walk": "#9e9e9e",
    "zigzag": "#ff9800",
    "greedy_bfs": "#4caf50",
    "q_learning": "#1f77b4",
    "dqn": "#d62728",
    "value_iteration": "#000000",
    "sarsa": "#9467bd",
}
_FALLBACK_COLORS = plt.get_cmap("tab10").colors
FIG_SIZE: tuple[float, float] = (8, 4.5)
DPI: int = 150

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.grid": True, "grid.alpha": 0.3})


def color_for(name: str, index: int = 0) -> str:
    """Màu cố định cho agent/cấu hình; tên lạ thì lấy màu theo thứ tự."""
    return AGENT_COLORS.get(name, matplotlib.colors.to_hex(_FALLBACK_COLORS[index % len(_FALLBACK_COLORS)]))


def _compact_number(value: float, _pos: int | None = None) -> str:
    """Nhãn trục gọn: 20000 -> 20k, 1500000 -> 1.5M (tránh các số dài dính vào nhau)."""
    for scale, suffix in ((1e6, "M"), (1e3, "k")):
        if abs(value) >= scale:
            return f"{value / scale:g}{suffix}"
    return f"{value:g}"


def _tidy_x_axis(ax: plt.Axes) -> None:
    """Tối đa ~5 vạch chia, số lớn viết gọn — cho trục số episode / số bước huấn luyện."""
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5, integer=True))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_compact_number))


def _save(fig: plt.Figure, out_path: str | os.PathLike) -> Path:
    path = require_path_inside_project(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def load_runs(pattern: str, root: str | os.PathLike = PROJECT_ROOT) -> pd.DataFrame:
    """Đọc nhiều CSV theo mẫu glob (tính từ root), thêm cột 'seed' lấy từ tên file 'seed<N>'.

    Raises:
        FileNotFoundError: không có file nào khớp.
    """
    paths = sorted(Path(root).glob(pattern))
    if not paths:
        raise FileNotFoundError(f"Không có file nào khớp '{pattern}' trong {root}.")
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        match = re.search(r"seed(\d+)", p.as_posix())
        if "seed" not in df.columns:
            df["seed"] = int(match.group(1)) if match else 0
        df["source"] = p.as_posix()
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _mean_std(df: pd.DataFrame, x: str, y: str, smooth: int) -> pd.DataFrame:
    """Trung bình ± std qua seed tại mỗi x; làm mượt bằng trung bình trượt nếu smooth > 1."""
    per_seed = df[[x, y, "seed"]].dropna(subset=[y]).sort_values(x)
    if smooth > 1:
        per_seed = per_seed.copy()
        per_seed[y] = per_seed.groupby("seed")[y].transform(lambda s: s.rolling(smooth, min_periods=1).mean())
    stats = per_seed.groupby(x)[y].agg(["mean", "std", "count"]).reset_index()
    stats["std"] = stats["std"].fillna(0.0)
    return stats


def plot_learning_curves(df: pd.DataFrame, x: str, panels: list[tuple[str, str]], hue: str,
                         out_path: str | os.PathLike, title: str, xlabel: str, smooth: int = 1,
                         hlines: dict[str, dict[str, float]] | None = None) -> Path:
    """Learning curve nhiều panel (mỗi panel một chỉ số), mỗi đường một giá trị của `hue`.

    Args:
        df: dữ liệu dạng dài, có cột x, các cột chỉ số, cột hue và cột seed.
        panels: danh sách (tên cột, nhãn trục y).
        hlines: {tên cột: {nhãn: giá trị}} các đường ngang tham chiếu (tối ưu, baseline...).
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(FIG_SIZE[0] * (0.65 if len(panels) > 1 else 1) * len(panels),
                                                      FIG_SIZE[1]), squeeze=False)
    n_seeds = df["seed"].nunique()
    for ax, (col, ylabel) in zip(axes[0], panels):
        # Tỷ lệ (coverage, success) nằm trong [0, 1]: cắt dải std để không vẽ giá trị vô nghĩa.
        bounded = "coverage" in col or "success" in col
        for i, (name, group) in enumerate(df.groupby(hue, sort=False)):
            stats = _mean_std(group, x, col, smooth)
            color = color_for(str(name), i)
            low, high = stats["mean"] - stats["std"], stats["mean"] + stats["std"]
            if bounded:
                low, high = low.clip(0.0, 1.0), high.clip(0.0, 1.0)
            ax.plot(stats[x], stats["mean"], color=color, label=str(name))
            ax.fill_between(stats[x], low, high, color=color, alpha=0.2)
        for j, (label, value) in enumerate((hlines or {}).get(col, {}).items()):
            # Ghi giá trị vào chú thích: các đường tham chiếu trùng nhau vẫn đọc được.
            ax.axhline(value, linestyle="--", linewidth=1.2, color=color_for(label, j + 5),
                       label=f"{label} = {value:.4g}")
        _tidy_x_axis(ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
    fig.suptitle(f"{title}\n(trung bình ± std, {n_seeds} seed)")
    return _save(fig, out_path)


def plot_learning_curve(df: pd.DataFrame, x: str, y: str, hue: str, out_path: str | os.PathLike,
                        title: str = "", xlabel: str = "", ylabel: str = "", smooth: int = 1,
                        hline: float | None = None, hline_label: str = "") -> Path:
    """Learning curve một panel (bọc plot_learning_curves)."""
    hlines = {y: {hline_label or "tham chiếu": hline}} if hline is not None else None
    return plot_learning_curves(df, x, [(y, ylabel or y)], hue, out_path, title or y, xlabel or x, smooth, hlines)


def plot_bar_with_error(summary: pd.DataFrame, category: str, value: str, error: str, out_path: str | os.PathLike,
                        title: str, ylabel: str, hue: str | None = None) -> Path:
    """Biểu đồ cột có thanh sai số (ví dụ số bước trung bình ± std của từng agent)."""
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    categories = list(dict.fromkeys(summary[category]))
    hues = list(dict.fromkeys(summary[hue])) if hue else [None]
    width = 0.8 / len(hues)
    for i, h in enumerate(hues):
        part = summary if h is None else summary[summary[hue] == h]
        part = part.set_index(category).reindex(categories)
        xs = np.arange(len(categories)) + (i - (len(hues) - 1) / 2) * width
        colors = [color_for(c, k) for k, c in enumerate(categories)] if h is None else color_for(str(h), i)
        ax.bar(xs, part[value], width, yerr=part[error], color=colors, capsize=4, label=None if h is None else str(h))
    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels(categories)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if hue:
        ax.legend(fontsize=9)
    return _save(fig, out_path)


def plot_lines_with_error(summary: pd.DataFrame, x: str, panels: list[tuple[str, str, str]], hue: str,
                          out_path: str | os.PathLike, title: str, xlabel: str) -> Path:
    """Đường có thanh sai số theo một biến số (ví dụ theo mật độ vật cản, E3).

    Args:
        panels: danh sách (cột giá trị, cột sai số, nhãn trục y), mỗi phần tử một panel.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(FIG_SIZE[0] * 0.65 * len(panels), FIG_SIZE[1]), squeeze=False)
    for ax, (y, error, ylabel) in zip(axes[0], panels):
        for i, (name, group) in enumerate(summary.groupby(hue, sort=False)):
            group = group.sort_values(x)
            ax.errorbar(group[x], group[y], yerr=group[error], marker="o", capsize=4,
                        color=color_for(str(name), i), label=str(name))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
    fig.suptitle(title)
    return _save(fig, out_path)


def plot_state_explosion(df: pd.DataFrame, out_path: str | os.PathLike, memory_limit_gb: float = 8.0,
                         title: str = "Bùng nổ không gian trạng thái của Q-learning dạng bảng") -> Path:
    """E7: số trạng thái theo số ô trống (thang log), đo thực tế và cận trên |F|·2^|F|."""
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    df = df.sort_values("n_free")
    ax.plot(df["n_free"], df["upper_bound"], "o--", color="#d62728", label="cận trên |F|·2^|F|")
    measured = df.dropna(subset=["reachable"])
    ax.plot(measured["n_free"], measured["reachable"], "o-", color="#1f77b4", label="đạt tới được (đo bằng BFS)")
    for _, row in df.iterrows():
        ax.annotate(row["grid"], (row["n_free"], row["upper_bound"]), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=8)
    # Bảng Q dạng float64 với 4 hành động: 32 byte mỗi trạng thái.
    limit_states = memory_limit_gb * 1e9 / 32
    ax.axhspan(limit_states, max(df["upper_bound"].max() * 10, limit_states * 10), color="#d62728", alpha=0.08,
               label=f"bảng Q > {memory_limit_gb:g} GB RAM")
    ax.set_yscale("log")
    ax.set_xlabel("số ô trống |F|")
    ax.set_ylabel("số trạng thái (thang log)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    return _save(fig, out_path)


def plot_tabular_scaling(train_df: pd.DataFrame, eval_df: pd.DataFrame, out_path: str | os.PathLike) -> Path:
    """T5.3: số trạng thái bảng Q đã gặp (thang log) và coverage tham lam theo thời gian, mỗi lưới một đường."""
    fig, axes = plt.subplots(1, 2, figsize=(FIG_SIZE[0] * 1.3, FIG_SIZE[1]))
    for i, (name, group) in enumerate(train_df.groupby("map", sort=False)):
        axes[0].plot(group["wall_time_s"] / 60, group["n_states_visited"], color=color_for(str(name), i), label=name)
    for i, (name, group) in enumerate(eval_df.groupby("map", sort=False)):
        minutes = np.interp(group["episode"], train_df.loc[train_df["map"] == name, "episode"],
                            train_df.loc[train_df["map"] == name, "wall_time_s"]) / 60
        axes[1].plot(minutes, group["coverage_mean"], color=color_for(str(name), i), label=name)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("thời gian huấn luyện (phút)")
    axes[0].set_ylabel("số trạng thái trong bảng Q (log)")
    axes[1].set_xlabel("thời gian huấn luyện (phút)")
    axes[1].set_ylabel("tỷ lệ đã dọn (chính sách tham lam)")
    axes[1].set_ylim(0, 1.05)
    for ax in axes:
        ax.legend(fontsize=9)
    fig.suptitle("Q-learning dạng bảng khi lưới lớn dần: bảng phình to, chính sách không kịp học")
    return _save(fig, out_path)


def plot_q_convergence(train_logs: pd.DataFrame, v_star: float, out_path: str | os.PathLike,
                       smooth: int = 50) -> Path:
    """max_a Q(s0, a) theo episode so với giá trị tối ưu thật V*(s0) từ Value Iteration."""
    hue = "algo" if "algo" in train_logs.columns else None
    df = train_logs.copy()
    if hue is None:
        df["algo"] = "q_learning"
    return plot_learning_curves(df, "episode", [("q_start_max", "max_a Q(s0, a)")], "algo", out_path,
                                "Ước lượng Q của trạng thái xuất phát hội tụ về V*", "episode", smooth=smooth,
                                hlines={"q_start_max": {"V*(s0) (Value Iteration)": v_star}})


def plot_visit_heatmaps(visits: dict[str, np.ndarray], grid: np.ndarray, out_path: str | os.PathLike,
                        subtitles: dict[str, str] | None = None) -> Path:
    """Heatmap số lần đi qua mỗi ô, mỗi agent một ô con, cùng thang màu."""
    from env.render import visit_heatmap

    names = list(visits)
    vmax = max(float(v.max()) for v in visits.values())
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.6), squeeze=False)
    image = None
    for ax, name in zip(axes[0], names):
        image = visit_heatmap(visits[name], grid, ax=ax, vmax=vmax)
        ax.set_title(f"{name}\n{(subtitles or {}).get(name, '')}", fontsize=10)
    if image is not None:
        fig.colorbar(image, ax=axes[0].tolist(), shrink=0.8, label="số lần đi qua")
    fig.suptitle("Số lần robot đi qua mỗi ô (đỏ đậm = lãng phí)")
    path = require_path_inside_project(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def df_to_markdown_table(df: pd.DataFrame, float_fmt: str = ".2f") -> str:
    """Bảng markdown (để dán vào README/slide) mà không cần thư viện tabulate."""

    def fmt(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            return "nan" if np.isnan(value) else format(float(value), float_fmt)
        return str(value)

    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def write_text(text: str, out_path: str | os.PathLike) -> Path:
    """Ghi văn bản (ví dụ bảng markdown) ra file trong thư mục dự án."""
    path = require_path_inside_project(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
