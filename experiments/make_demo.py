"""T8.4: tư liệu trình diễn — robot hút bụi chạy trong mô phỏng, dùng các mô hình đã huấn luyện.

Đầu ra (results/demo/):
- compare_<map>.gif / .png       : "cuộc đua" các robot trên cùng căn phòng, cùng điểm xuất phát
- compare_obstacle_4x4.gif / .png: như trên ở phòng 4x4, thêm Q-learning và đường tối ưu (Value Iteration)
- progress_qlearning.gif, progress_dqn.gif : "robot học dần" qua các checkpoint
- heatmap_<map>.png              : số lần đi qua mỗi ô của từng robot
- unseen_rooms.gif               : DQN (huấn luyện trên phòng ngẫu nhiên) dọn 3 phòng chưa từng thấy, cạnh greedy_bfs
- demo_summary.md                : bảng số liệu của các GIF, để dán vào slide

Ví dụ:
    python -m experiments.make_demo                 # dùng mô hình trong results/ (chạy run_tabular, run_deep trước)
    python -m experiments.make_demo --quick         # tự huấn luyện mô hình tí hon để kiểm tra luồng
"""

from __future__ import annotations

import argparse
import functools
from pathlib import Path
from typing import Any

from experiments.common import make_env_factory, resolve_out

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from baselines import GreedyBFSAgent, RandomWalkAgent, ZigzagAgent  # noqa: E402
from deep.dqn import DQNAgent, DQNConfig  # noqa: E402
from env.maps import FIXED_MAPS, generate_map_set, get_map  # noqa: E402
from env.render import annotate_frame, compose_side_by_side, compose_vertical, save_gif  # noqa: E402
from env.vacuum_env import RandomMapVacuumEnv, VacuumEnv  # noqa: E402
from experiments.evaluate import run_episode  # noqa: E402
from experiments.plots import df_to_markdown_table, plot_visit_heatmaps, write_text  # noqa: E402
from tabular.common import TabularAgent, TabularConfig  # noqa: E402
from tabular.q_learning import QLearningAgent  # noqa: E402
from tabular.state_space import build_state_graph  # noqa: E402
from tabular.value_iteration import VIAgent, value_iteration  # noqa: E402
from utils.paths import ROOT  # noqa: E402

FPS = 6
MAX_GIF_FRAMES = 160  # giới hạn số khung để GIF nhẹ (< 5 MB)
SMALL_MAP = "obstacle_4x4"
SMALL_START = (0, 0)
UNSEEN_SEED = 777
DEFAULT_QTABLE = "results/tabular/dense/full/obstacle_4x4/q_learning/seed0/checkpoints"
DEFAULT_DQN_MAP = "results/deep/dqn/furniture_8x8/seed0"
DEFAULT_DQN_RANDOM = "results/deep/dqn/rand8x8_d0.15_K0/seed0"


def rollout(agent: Any, env: VacuumEnv, title: str, seed: int, start: tuple[int, int] | None,
            cell_px: int) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Chạy một episode, trả các khung hình có chú thích và số liệu cuối."""
    if hasattr(agent, "reset"):
        agent.reset()
    obs, info = env.reset(seed=seed, options={"start": start} if start else None)
    from env.render import render_rgb

    trail = [info["position"]]

    def frame(info: dict[str, Any], status: str = "") -> np.ndarray:
        img = render_rgb(env.grid, info["position"], env.cleaned, cell_px=cell_px, trail=trail[-30:])
        # Hai dòng ngắn để vừa chiều rộng của phòng nhỏ.
        lines = [f"bước {info['steps']} · dọn {info['coverage']:.0%}",
                 f"lặp {info['n_revisit']} · đụng {info['n_collisions']}{status}"]
        return annotate_frame(img, title, lines, font_size=max(11, cell_px // 3))

    frames = [frame(info)]
    terminated = truncated = False
    while not (terminated or truncated):
        obs, _, terminated, truncated, info = env.step(agent.act(obs, info))
        trail.append(info["position"])
        frames.append(frame(info))
    frames[-1] = frame(info, " ✓ xong" if terminated else " ✗ hết giờ")
    stats = {"agent": title, "steps": info["steps"], "coverage": info["coverage"], "revisits": info["n_revisit"],
             "collisions": info["n_collisions"], "finished": terminated}
    return frames, stats


def synchronized(frame_lists: list[list[np.ndarray]]) -> list[np.ndarray]:
    """Ghép các rollout cạnh nhau theo từng bước; robot xong trước đứng yên ở khung cuối.

    Nếu quá dài thì lấy mẫu đều, luôn giữ khung cuối.
    """
    n = max(len(f) for f in frame_lists)
    indices = list(range(n))
    if n > MAX_GIF_FRAMES:
        indices = sorted(set(np.linspace(0, n - 1, MAX_GIF_FRAMES).round().astype(int).tolist()))
    return [compose_side_by_side([f[min(t, len(f) - 1)] for f in frame_lists]) for t in indices]


def race(agents: dict[str, Any], grid: np.ndarray, start: tuple[int, int] | None, seed: int, out_path: Path,
         cell_px: int = 32) -> list[dict[str, Any]]:
    """"Cuộc đua" nhiều robot trên cùng căn phòng; lưu GIF và ảnh khung cuối."""
    frame_lists, stats = [], []
    for name, agent in agents.items():
        frames, s = rollout(agent, VacuumEnv(grid), name, seed, start, cell_px)
        frame_lists.append(frames)
        stats.append(s)
    frames = synchronized(frame_lists)
    save_gif(frames, out_path, fps=FPS, last_frame_hold=FPS * 2)
    from PIL import Image

    Image.fromarray(frames[-1]).save(out_path.with_suffix(".png"))
    return stats


def load_qtable(checkpoint_dir: Path, name: str) -> TabularAgent:
    path = checkpoint_dir / name
    if not path.exists():
        raise FileNotFoundError(
            f"Không thấy {path}. Hãy chạy trước: python -m experiments.run_tabular --map {SMALL_MAP} --start 0 0")
    return TabularAgent.load(path)


def load_dqn(path: Path, hint: str) -> DQNAgent:
    if not path.exists():
        raise FileNotFoundError(f"Không thấy {path}. Hãy chạy trước: {hint}")
    return DQNAgent.load(path)


def train_quick_models(model_dir: Path, map_name: str) -> dict[str, Path]:
    """--quick: huấn luyện mô hình tí hon (vài giây) để kiểm tra toàn bộ luồng demo."""
    q_dir = model_dir / "qtable"
    agent = QLearningAgent(get_map(SMALL_MAP), TabularConfig(n_episodes=300, eval_every=150, eval_episodes=1,
                                                             checkpoint_episodes=(0, 50, 150)))
    agent.train(make_env_factory(SMALL_MAP, start=SMALL_START), log_dir=q_dir)
    cfg = dict(total_timesteps=600, learning_starts=100, batch_size=16, buffer_size=1000, eval_every_steps=600,
               eval_episodes=1, log_every_steps=100)
    grid = get_map(map_name)
    dqn_map = DQNAgent((3, *grid.shape), DQNConfig(checkpoint_steps=(0, 150), **cfg))
    dqn_map.train(make_env_factory(map_name), model_dir / "dqn_map")
    dqn_rand = DQNAgent((3, 8, 8), DQNConfig(**cfg))
    dqn_rand.train(functools.partial(RandomMapVacuumEnv, 8, 8, 0.15), model_dir / "dqn_random")
    return {"qtable": q_dir / "checkpoints", "dqn_map": model_dir / "dqn_map", "dqn_random": model_dir / "dqn_random"}


def checkpoint_sequence(directory: Path, pattern: str, label: str) -> list[tuple[str, Path]]:
    """Các checkpoint theo thứ tự (ep0, ep1000, ..., final) kèm nhãn hiển thị."""
    def number(p: Path) -> int:
        return int("".join(ch for ch in p.stem if ch.isdigit()) or -1)

    paths = sorted(directory.glob(pattern), key=number)
    return [(f"{label} {number(p)}", p) for p in paths]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tạo GIF/hình trình diễn robot hút bụi.")
    parser.add_argument("--what", nargs="+", default=["all"],
                        choices=["all", "compare", "progress", "heatmap", "unseen"])
    parser.add_argument("--map", default="furniture_8x8", choices=sorted(FIXED_MAPS))
    parser.add_argument("--start", type=int, nargs=2, default=[0, 0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--qtable-dir", default=DEFAULT_QTABLE, help="thư mục checkpoints của Q-learning (4x4)")
    parser.add_argument("--dqn-map-dir", default=DEFAULT_DQN_MAP, help="thư mục DQN huấn luyện trên --map")
    parser.add_argument("--dqn-random-dir", default=DEFAULT_DQN_RANDOM, help="thư mục DQN huấn luyện trên phòng ngẫu nhiên 8x8")
    parser.add_argument("--out", default="results/demo")
    parser.add_argument("--quick", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> pd.DataFrame:
    args = build_parser().parse_args(argv)
    out = resolve_out(args.out)
    what = {"compare", "progress", "heatmap", "unseen"} if "all" in args.what else set(args.what)
    if args.quick:
        paths = train_quick_models(out / "_quick_models", args.map)
    else:
        paths = {"qtable": ROOT / args.qtable_dir, "dqn_map": ROOT / args.dqn_map_dir,
                 "dqn_random": ROOT / args.dqn_random_dir}
    grid, start = get_map(args.map), tuple(args.start)
    hint_map = f"python -m experiments.run_deep --map {args.map} --timesteps 500000"
    rows: list[dict[str, Any]] = []

    if "compare" in what:
        dqn = load_dqn(paths["dqn_map"] / "model.pt", hint_map)
        agents = {"random_walk": RandomWalkAgent(seed=args.seed), "zigzag": ZigzagAgent(),
                  "greedy_bfs": GreedyBFSAgent(), "DQN": dqn}
        rows += [{"demo": f"compare_{args.map}", **s} for s in
                 race(agents, grid, start, args.seed, out / f"compare_{args.map}.gif")]
        small = get_map(SMALL_MAP)
        vi = value_iteration(build_state_graph(small))
        agents_small = {"random_walk": RandomWalkAgent(seed=args.seed), "zigzag": ZigzagAgent(),
                        "greedy_bfs": GreedyBFSAgent(), "Q-learning": load_qtable(paths["qtable"], "final.pkl"),
                        "tối ưu (VI)": VIAgent(vi)}
        rows += [{"demo": f"compare_{SMALL_MAP}", **s} for s in
                 race(agents_small, small, SMALL_START, args.seed, out / f"compare_{SMALL_MAP}.gif", cell_px=48)]

    if "progress" in what:
        q_seq = checkpoint_sequence(paths["qtable"], "ep*.pkl", "episode") + [("đã học xong", paths["qtable"] / "final.pkl")]
        agents = {label: TabularAgent.load(p) for label, p in q_seq}
        rows += [{"demo": "progress_qlearning", **s} for s in
                 race(agents, get_map(SMALL_MAP), SMALL_START, args.seed, out / "progress_qlearning.gif", cell_px=48)]
        d_seq = checkpoint_sequence(paths["dqn_map"] / "checkpoints", "step*.pt", "bước")
        d_seq = [(label, p) for label, p in d_seq] + [("đã học xong", paths["dqn_map"] / "model.pt")]
        agents = {label: load_dqn(p, hint_map) for label, p in d_seq}
        rows += [{"demo": "progress_dqn", **s} for s in
                 race(agents, grid, start, args.seed, out / "progress_dqn.gif")]

    if "heatmap" in what:
        dqn = load_dqn(paths["dqn_map"] / "model.pt", hint_map)
        agents = {"random_walk": RandomWalkAgent(seed=args.seed), "zigzag": ZigzagAgent(),
                  "greedy_bfs": GreedyBFSAgent(), "DQN": dqn}
        visits, subtitles = {}, {}
        for name, agent in agents.items():
            res = run_episode(agent, VacuumEnv(grid), seed=args.seed, record_visits=True,
                              reset_options={"start": start})
            visits[name] = res["visits"]
            subtitles[name] = f"{res['steps']} bước, lặp {res['redundancy']:.0%}"
        plot_visit_heatmaps(visits, grid, out / f"heatmap_{args.map}.png", subtitles)

    if "unseen" in what:
        dqn = load_dqn(paths["dqn_random"] / "model.pt", "python -m experiments.run_deep --random-maps 8 8 0.15 --timesteps 500000")
        rooms = generate_map_set(8, 8, 0.15, 3, seed=UNSEEN_SEED)
        room_frames = []
        for i, room in enumerate(rooms):
            lists = []
            for name, agent in (("DQN (phòng lạ)", dqn), ("greedy_bfs", GreedyBFSAgent())):
                frames, s = rollout(agent, VacuumEnv(room), f"phòng {i + 1}: {name}", args.seed, None, 32)
                lists.append(frames)
                rows.append({"demo": f"unseen_room{i + 1}", **s})
            room_frames.append(lists)
        n = max(len(f) for lists in room_frames for f in lists)
        indices = range(n) if n <= MAX_GIF_FRAMES else sorted(set(np.linspace(0, n - 1, MAX_GIF_FRAMES).round().astype(int)))
        frames = [compose_vertical([compose_side_by_side([f[min(t, len(f) - 1)] for f in lists])
                                    for lists in room_frames]) for t in indices]
        save_gif(frames, out / "unseen_rooms.gif", fps=FPS, last_frame_hold=FPS * 2)

    table = pd.DataFrame(rows)
    if not table.empty:
        write_text("# Số liệu demo\n\n" + df_to_markdown_table(table), out / "demo_summary.md")
        print(df_to_markdown_table(table))
    for path in sorted(out.glob("*.gif")) + sorted(out.glob("*.png")):
        print(f"{path.name}: {path.stat().st_size / 1e6:.2f} MB")
    return table


if __name__ == "__main__":
    main()
