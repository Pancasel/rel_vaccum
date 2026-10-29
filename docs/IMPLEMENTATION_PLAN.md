# IMPLEMENTATION_PLAN.md — Detailed implementation spec for the coding agent

> **Lưu ý:** tên file/hàm/thư mục trong tài liệu này là tên **tại thời điểm lập kế hoạch/ghi nhật ký** (vd. `env/`, `VacuumEnv`, `run_deep`). Dự án đã đổi sang tên tường minh; xem bảng đổi tên ở [TU_DIEN_KY_HIEU.md](TU_DIEN_KY_HIEU.md) mục 7.

> This file **expands** `PLAN_rel.md` (Parts B, C, D, E) into a spec precise enough that a coding agent (Claude Code) can build the whole project **without guessing**.
>
> - `PLAN_rel.md` (Vietnamese) = **what and why**: motivation, scope, MDP, roadmap, submission.
> - `IMPLEMENTATION_PLAN.md` (this file) = **how**: signatures, formulas, build order, concrete tests, acceptance commands, expected values.
> - **On technical details, this file wins** (it is the refinement). On scope or submission requirements, `PLAN_rel.md` wins.

---

## Table of contents

0. [Agent operating rules (READ FIRST)](#0)
1. [Locked technical decisions](#1)
2. [Development environment and dependencies](#2)
3. [Repository layout](#3)
4. [Module specifications](#4)
5. [Task cards, in execution order](#5)
6. [Experiment specifications E1–E7](#6)
7. [Result file and plotting conventions](#7)
8. [RL debugging checklist](#8)
9. [Report template](#9)
10. [Appendix: hand-verified reference values](#10)

---

<a id="0"></a>
## 0. Agent operating rules (READ FIRST)

### 0.1 Workspace boundary — never act outside this folder

The project root is:

```
D:\DoQuocAn\ki8\REL\project_vaccum
```

(referred to below as `ROOT`). These rules are absolute:

1. **Read, create, modify, and delete files only inside `ROOT`.** Never touch any other path: not the user's home directory, Desktop, Downloads, other projects, system folders, or global config files.
2. **No global installs.** All Python packages go into a virtual environment at `ROOT/.venv`. Never run `pip install` against the system Python, never use `--user`, never modify `PATH` or system environment variables permanently.
3. **No global configuration changes.** Do not run `git config --global`, do not edit `~/.gitconfig`, `~/.bashrc`, PowerShell profiles, VS Code user settings, or registry keys. If git needs a user identity, use repo-local `git config user.name/user.email` only after asking the user.
4. **Keep caches and temp files inside `ROOT`:**
   - `pytest.ini` sets `--basetemp=.pytest_tmp` so `tmp_path` lives under `ROOT`.
   - `tests/conftest.py` and every script entry point set `os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))` before importing matplotlib.
   - Use `pip install --no-cache-dir` (or `PIP_CACHE_DIR=ROOT/.cache/pip`).
   - All outputs go to `ROOT/results/`.
5. **Network:** only to install packages from PyPI and the PyTorch wheel index. No other downloads, no uploads, no `git push`, no remote creation, no emails, no Google Drive actions. Submission (Drive, email) is done by the humans.
6. **Destructive commands** (`rm -rf`, `git reset --hard`, `git clean`, deleting `results/`) are allowed only on paths inside `ROOT` that the agent itself created in this project, and never on `PLAN_rel.md` or `IMPLEMENTATION_PLAN.md`.
7. **Do not edit `PLAN_rel.md`.** When a design change must be logged in its table B8, write the proposed row in the task report (section 9) and in `CHANGELOG_DESIGN.md` (see 0.5); the user copies it over.

If a task seems to require breaking any of these rules, **stop and ask the user**.

### 0.2 Execution mode

- Default mode is **sequential autonomous**: work through the **core** task cards in section 5 in order. After a card passes its acceptance criteria, commit, write the short report (section 9), and continue to the next card.
- **Optional cards (O1, O2) are never started automatically.** After the last core card (T8.4), stop and ask which optional cards to do.
- Stop and wait for the user only when:
  - a card is listed as a **checkpoint** (marked 🛑 in section 5), or
  - a decision is not covered by this file and would change behavior, interfaces, or experiment conclusions, or
  - an acceptance criterion still fails after the fix procedure in 0.4, or
  - the user asked for a single card only ("do T2.1 only").
- If the user names a specific card, do that card (after verifying its dependencies pass) and stop.

### 0.3 Code quality bar — "polished, complete flow"

Every piece of code delivered must satisfy all of the following. A card is **not done** until it does.

1. **No stubs.** No `pass` placeholders, `TODO`, `NotImplementedError`, fake return values, or commented-out code in delivered files. If something is out of scope for the current card, it does not exist yet; it is not stubbed.
2. **End-to-end wiring.** Each feature is connected through its whole path: config/CLI → environment → agent → training loop → CSV log → evaluation → plot/GIF. A script that "trains but doesn't save", or "saves but no plotting function reads it", is incomplete.
3. **Every script runs from a clean checkout** with one command, `python -m experiments.<name> ...`, has `--help`, a `--seed`/`--seeds` argument, a `--quick` mode that finishes in under ~2 minutes, and writes `config.json` next to its outputs.
4. **Input validation with clear errors.** Public functions validate arguments and raise `ValueError`/`KeyError` with a message explaining what was wrong and what is allowed.
5. **Determinism.** Same seed → identical CSVs (for tabular and baselines; for torch, identical on the same machine with `torch.use_deterministic_algorithms` where practical).
6. **Tests accompany code** in the same card. Unit tests for logic, plus a pipeline smoke test (`tests/test_pipeline.py`) that runs every finished script in `--quick` mode end to end.
7. **Readable code.** Type hints on every public function, docstrings on every public function/class, functions preferably under ~50 lines, no magic numbers (put them in config dataclasses or UPPER_CASE module constants with a comment).
8. **No warnings left unexplained.** Running tests must not print warnings from our code. Unavoidable third-party warnings get a comment explaining why they are harmless, or a targeted `filterwarnings` in `pytest.ini`.
9. **Lint clean.** `python -m compileall -q .` succeeds; no unused imports or variables (check manually or with `python -m pyflakes` if available in the venv; do not add new dependencies just for linting).

### 0.4 Fix-it-now policy

When something is wrong, the agent fixes it immediately instead of leaving it for later:

1. **A test fails or a script crashes** → find the root cause, fix the **code** (not the test), rerun the full fast test suite. Never weaken a test, skip it, loosen a tolerance, or add `xfail` to make it pass.
2. **A test itself is wrong** (contradicts this spec) → fix the test, and state in the report exactly why it was wrong.
3. **This spec is wrong** (e.g. an expected value is mathematically incorrect, or two sections contradict) → do the minimal correct thing, document it in `CHANGELOG_DESIGN.md` and in the report with evidence (computation, output). If the correction changes an interface, a default, or an experiment's meaning, stop and ask first.
4. **Bug found in earlier code** while working on a later card → fix it right away in the same session, add a regression test that would have caught it, rerun all tests, mention it in the report under "Fixes to earlier work".
5. **A learning result is poor** (agent doesn't learn) → this is not automatically a bug. Run the checklist in section 8 first. Fix genuine bugs immediately; if no bug is found, report the result honestly with evidence and propose next steps. Do not tune hyperparameters on the test set and do not cherry-pick seeds.
6. **Fix loop limit:** if the same failure survives three serious fix attempts, stop, report what was tried and what was observed, and ask the user.
7. After any fix: `python -m pytest -m "not slow"` must be fully green before committing.

### 0.5 Per-card workflow

1. Re-read the card and the module spec it references.
2. Verify dependencies: run the tests of the cards it depends on.
3. Implement code + tests together, in small steps; run the relevant tests after each step.
4. Run the card's acceptance commands. Fix until green (0.4).
5. Run the full fast suite: `python -m pytest -m "not slow"`.
6. Self-review the diff against the quality bar (0.3): no stubs, flows wired, docstrings present, no debug prints.
7. `git add` only the files inside `ROOT` that belong to this card; commit with the message given in the card (Conventional Commits style, English).
8. Write the report (section 9). If design changed, append to `CHANGELOG_DESIGN.md`:
   ```
   ## <date> — <card>
   - Change: ...
   - Reason: ...
   - Evidence/result: ...
   - Proposed B8 row: | vN | <date> | ... | ... | ... |
   ```

### 0.6 Coding conventions

| Item | Convention |
|---|---|
| Identifiers | English. `snake_case` functions/variables, `PascalCase` classes, `UPPER_CASE` constants |
| **Comments and docstrings** | **Vietnamese** (required by `PLAN_rel.md` C2.5 — the students must explain the code in an oral exam). Explain **why**, not only what |
| Docstring format | Google style: keep the keywords `Args:`, `Returns:`, `Raises:` in English, write the content in Vietnamese |
| CLI help text, log column names, file names, commit messages | English |
| Plot titles/axis labels | Vietnamese (for the slides) |
| Textbook references | In comments, e.g. `# Sutton & Barto 2018, mục 6.5 (Q-learning): "Q(S,A) <- Q(S,A) + α[R + γ max_a Q(S',a) − Q(S,A)]"` |
| Imports | Absolute from repo root: `from env.vacuum_env import VacuumEnv`. Never `import env` and then name a variable `env` |
| Running scripts | Always as modules from `ROOT`: `python -m experiments.run_baselines --seeds 0 1` |
| Randomness | Each agent owns `np.random.default_rng(seed)`; envs use `self.np_random`. No reliance on global `np.random` state |
| Algorithm fidelity | Implement Sutton & Barto pseudocode exactly. Any variant must carry a `# BIẾN THỂ: ... vì ...` comment and be listed in section 1 or approved by the user |
| Dependencies | Only those in section 2. Ask before adding anything |
| Committed files | Never commit `.venv/`, caches, `*.pt`, `*.zip`, or any file > 5 MB |

---

<a id="1"></a>
## 1. Locked technical decisions

These settle every ambiguity in `PLAN_rel.md`. Implement them exactly.

| ID | Topic | Decision | Why |
|---|---|---|---|
| D-01 | Map representation | `np.ndarray` shape `(H, W)`, dtype `np.int8`, `0 = free`, `1 = obstacle`. Fixed maps are written as strings: `.` free, `#` obstacle | Easy to read, draw, test |
| D-02 | Action directions | `0=up (-1,0)`, `1=down (+1,0)`, `2=left (0,-1)`, `3=right (0,+1)` as `(drow, dcol)`; row 0 is the **top** | Matches B2 |
| D-03 | Reward composition | `reward = step + event_reward + completion (if terminated) + shaping (if enabled) + naive_bonus (if enabled)`. `event_reward` is **exactly one** of `{new_cell, revisit, collision}` | No double-counted events |
| D-04 | Bumping a wall/obstacle | Robot stays, `event = "collision"`, **still counts as one step** (step counter increments, `step` penalty applied), `info["collided"] = True` | Collisions waste real time |
| D-05 | Start cell | Marked cleaned at `reset`, earns no reward | Matches B2 |
| D-06 | Degenerate maps | `|F| < 2` → `ValueError` in `__init__` | Nothing to learn |
| D-07 | `max_steps` | Default `4 * |F|`. Counts collision steps. `truncated = steps >= max_steps and not terminated` | Never both True |
| D-08 | `start` | `None` → uniform over free cells via `self.np_random`; tuple → fixed (must be free, else `ValueError`). Overridable per episode via `reset(options={"start": (r, c)})` | Lets VI/tests pick any start |
| D-09 | Coverage | `coverage = n_cleaned / |F|`, **including** the start cell | Robot starts at 1/|F| |
| D-10 | Redundancy | `redundancy = n_revisit / steps` (collision steps excluded from numerator, included in denominator); `0.0` if `steps == 0` | Kept separate from `collisions` |
| D-11 | Transition model | One **pure** function `transition(...)` in `env/vacuum_env.py` is the single source of truth. `VacuumEnv.step` **and** Value Iteration both call it | Prevents env/VI reward mismatch (a classic silent bug) |
| D-12 | Tabular state key | `(row, col, mask)` with `mask` an `int` bitmask over **free-cell indices** in row-major order; bit `i` = 1 if free cell `i` is cleaned. Tabular supports `|F| <= 62` | Hashable, compact, shared by VI and TD agents |
| D-13 | E1 "no cleaned map" | Tabular key `(row, col)` via `state_mode="position_only"`. Deep version: wrapper zeroes channel 1 | Exactly the E1 hypothesis |
| D-14 | Potential-based shaping (E2c) | `Φ(s) = -d(s) / (H + W)` where `d(s)` = BFS distance (around obstacles) from robot to nearest uncleaned free cell. `F = shaping_coef * (γ·Φ(s') − Φ(s))`, with `Φ(s') = 0` if `s'` is terminal | Ng, Harada & Russell (1999): potential-based shaping preserves the optimal policy |
| D-15 | "Hackable" reward (E2 demo) | `naive_progress_bonus`: add `+bonus` when `d(s') < d(s)`, **no penalty** when `d` increases. Default `0.0` | Creates an exploitable "step away, step back" loop to demonstrate reward hacking |
| D-16 | Determinism | The environment is fully deterministic; there is **no** reward noise option (it existed only for the dropped E5) | Simpler env; VI is exact |
| D-17 | Random maps during training | Subclass `RandomMapVacuumEnv(VacuumEnv)` regenerates the map on each `reset` (same `H, W`), or samples from a fixed pool of `K` maps | Keeps the C3 interface of `VacuumEnv` intact |
| D-18 | Network input size | One DQN model is trained per grid size (`H × W` fixed per model). No padding wrapper (curriculum was dropped) | Simplest correct design |
| D-19 | Zigzag with obstacles | Snake order (even rows left→right, odd rows right→left). Target = **first uncleaned** cell in snake order; move along a BFS shortest path to it | Always completes on connected maps; deterministic; explainable |
| D-20 | Tie-breaking | Deterministic `argmax` picks the **lowest action index** among values equal within `1e-9`. ε-greedy exploitation during training breaks ties **randomly** with the agent's rng | Deterministic tests; unbiased learning |
| D-21 | Value Iteration form | **Synchronous, vectorized** (two-array) updates. S&B section 4.1 confirms the two-array version converges too. Mark with `# BIẾN THỂ` | Hundreds of times faster than Python loops |
| D-22 | True minimum steps | Computed separately by **BFS on the state graph** (unweighted) → `min_steps_to_cover`. This is the exact "fewest steps" answer | The VI policy is optimal for discounted return, not necessarily for step count |
| D-23 | Evaluation seeds | Episode `i` of `evaluate(..., seed=s)` uses `env.reset(seed=s * 10_000 + i)` | No seed overlap across runs |
| D-24 | Stateful agents | `Agent` protocol requires `act`; if an agent also has `reset()`, `evaluate` calls it at the start of each episode | Compatible with C3, still supports memory |
| D-25 | Python version | Support **3.11–3.13** (the machine has 3.13.9). Use the project venv | Team machines differ |
| D-26 | Demo rendering | Frames drawn with **Pillow `ImageDraw`** (robot = circle, faint trail of the last path, text overlay for step/coverage). No new dependency (Pillow ships with matplotlib) | Presentation-quality visuals without pygame |
| D-27 | Model scope | See 1.1. The project is built around **Q-learning**. DQN is "Q-learning with a neural network". Only two optional cards exist (SARSA, interactive demo) | The user chose Q-learning; everything else is picked for being the easiest to understand and the best fit |

### 1.1 Model scope — what is actually used

The project does **not** try to implement every algorithm in Sutton & Barto. The user chose **Q-learning** as the main model. Every other component was picked by two criteria: **easiest to understand/explain in the oral exam** and **best fit for the coverage problem**.

| Role | Component | Book / source | Why this one |
|---|---|---|---|
| **Main model** | **Q-learning** (tabular) | S&B 6.5 | Chosen by the user. Simple update rule, off-policy, learns the optimal policy on small rooms (≤ 4×4); every line maps to the book's pseudocode |
| **Scale-up of the main model** | **DQN = Deep Q-learning** | S&B 9 (function approximation), 11.3 (deadly triad), 16.5 (DQN); Mnih et al. 2015 | It is **the same Q-learning target** `R + γ max_a Q(S',a)` with the table replaced by a small CNN. Nothing new to learn conceptually except "replay buffer" and "target network", each explainable in one sentence. Needed because an 8×8 room has ~10¹⁷ states and because a network can handle **unseen rooms** |
| Answer key (not a learner) | Value Iteration | S&B 4.4 | Computes the **exact optimum** on small rooms, so we can prove Q-learning is correct |
| Baselines (not learners) | random_walk, zigzag, greedy_bfs | — | The three most intuitive non-RL strategies (random, lawn-mower, nearest-dirty-cell). Show whether RL is worth it |

**Code-level consequence for DQN:** `deep/dqn.py` must read like a sibling of `tabular/q_learning.py`: same ε-schedule shape, same variable names (`state`, `action`, `reward`, `next_state`, `terminated`), and the target line carries the same S&B 6.5 comment, followed by a comment explaining that the max is taken over the **target network** (Mnih 2015) instead of the table. The README and slides explain DQN as "Q-learning + network + replay buffer + target network".

**The story the code must support** (and the presentation tells):

> Small room → **Q-learning** reaches the optimum (verified by Value Iteration) → on larger rooms the Q-table explodes (E7) and position-only states fail (E1) → **same algorithm with a network (DQN)** → compare against zigzag / greedy, including unseen rooms (E3, E4) → **demo** (T8.4).

**Optional extensions** (section 5, "Optional cards"; built only after the core is finished and the user says go):

| Card | Component | Why it is the only optional learner kept |
|---|---|---|
| O1 | SARSA (S&B 6.4) | Differs from Q-learning in **one line** (`Q(S',A')` instead of `max_a Q(S',a)`), so it is the easiest possible comparison and a classic exam question (on-policy vs off-policy) |
| O2 | Interactive demo window | Live click-to-place-furniture demo for the presentation |

**Dropped on purpose** (do not implement; propose B8 rows for them — see `CHANGELOG_DESIGN.md` first entry in T1.2):

| Dropped | Reason |
|---|---|
| Double Q-learning (S&B 6.7) and experiment E5 | Maximization bias is hard to show in a deterministic environment and adds a concept not needed for the story |
| Double DQN | A refinement, not needed to answer the project question |
| PPO (Stable-Baselines3) | A different family (policy gradient) that is hard to explain alongside Q-learning; debugging is covered by the VI answer key and the 3×3 → 4×4 → 8×8 ladder instead |
| Curriculum learning | Training trick; not needed |

Rule for the agent: **do not create files, CLI choices, config fields, or tests for optional components while working on core cards**, and never implement dropped components. No placeholders (rule 0.3.1).

---

<a id="2"></a>
## 2. Development environment and dependencies

### 2.1 `requirements.txt`

```
numpy>=1.26,<3
pandas>=2.1
matplotlib>=3.8
gymnasium>=1.0,<2
torch>=2.2
pytest>=8.0
```

- `stable-baselines3` is **not used** (PPO was dropped, see 1.1).

- No `pygame`, no `imageio`: GIFs are written with **Pillow** (already a matplotlib dependency).
- No tensorboard; all logs go to CSV.
- After a successful install and green tests: `pip freeze > requirements-lock.txt`, commit it.

### 2.2 Setup (Windows PowerShell, run from `ROOT`)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
pip install --no-cache-dir -r requirements.txt
python -m pytest -m "not slow"
```

All later commands assume the venv is active (or call `.\.venv\Scripts\python.exe` explicitly). On Colab: `pip install -r requirements.txt`.

### 2.3 `pytest.ini`

```ini
[pytest]
testpaths = tests
addopts = -q --basetemp=.pytest_tmp
markers =
    slow: long test (> 10 s); skip with -m "not slow"
```

Time budget: `python -m pytest -m "not slow"` **< 60 s** on a laptop; all `slow` tests together < 5 min.

### 2.4 `.gitignore`

```
.venv/
.cache/
.pytest_tmp/
.pytest_cache/
__pycache__/
*.pyc
results/**/*.pt
results/**/*.zip
results/**/*.mp4
results/raw/
.ipynb_checkpoints/
```

---

<a id="3"></a>
## 3. Repository layout

The repo root is `ROOT` (= `vacuum-rl/` in B9). **Bold** entries are additions to B9, justified in section 1. Entries marked *(optional, Ox)* are created **only** by that optional card — they must not exist after the core cards.

```
project_vaccum/
├── PLAN_rel.md                 # master plan (what/why) — read-only for the agent
├── IMPLEMENTATION_PLAN.md      # this file (how)
├── CHANGELOG_DESIGN.md         # **added**: design changes + proposed B8 rows
├── README.md
├── requirements.txt
├── requirements-lock.txt       # generated after a successful install
├── pytest.ini                  # **added**
├── .gitignore                  # **added**
├── env/
│   ├── __init__.py             # exports VacuumEnv, RandomMapVacuumEnv, RewardConfig, get_map
│   ├── maps.py                 # fixed maps, parsing, random generation, connectivity check
│   ├── vacuum_env.py           # RewardConfig, transition(), VacuumEnv, RandomMapVacuumEnv
│   ├── grid_utils.py           # **added**: BFS helpers, obs decoding, valid_actions, StateEncoder
│   ├── wrappers.py             # **added**: HideCleanedChannel (E1 deep)
│   └── render.py               # numpy frame rendering, GIF export, episode recording
├── utils/                      # **added**
│   ├── __init__.py
│   ├── paths.py                # ROOT constant, cache setup (MPLCONFIGDIR)
│   ├── seeding.py              # set_global_seed
│   └── io.py                   # ensure_dir, save_csv, save_config_json
├── baselines/
│   ├── __init__.py             # exports agents + make_baseline(name, seed)
│   ├── agent.py                # **added**: Agent Protocol
│   ├── random_walk.py
│   ├── zigzag.py
│   └── greedy_bfs.py
├── tabular/
│   ├── __init__.py
│   ├── common.py               # **added**: TabularConfig, TabularAgent base, ε schedule
│   ├── state_space.py          # **added**: reachable-state BFS, min_steps_to_cover
│   ├── value_iteration.py
│   ├── q_learning.py
│   └── sarsa.py                # (optional, O1)
├── deep/
│   ├── __init__.py
│   ├── networks.py
│   ├── replay_buffer.py
│   └── dqn.py                  # Deep Q-learning
├── demo/                       # (optional, O2)
│   ├── __init__.py
│   └── play.py                 # interactive window
├── experiments/
│   ├── __init__.py
│   ├── evaluate.py
│   ├── run_baselines.py
│   ├── run_value_iteration.py  # **added**
│   ├── run_tabular.py
│   ├── run_state_explosion.py  # **added** (E7 / task 5.1)
│   ├── run_deep.py
│   ├── run_ablation.py
│   ├── run_generalization.py   # **added** (E3, E4)
│   ├── make_demo.py            # **added** (T8.4): comparison video, learning progression, heatmaps
│   ├── reproduce_all.py        # **added**: runs every experiment (quick or full)
│   └── plots.py
├── tests/
│   ├── conftest.py             # fixtures, MPLCONFIGDIR setup
│   ├── test_smoke.py           # **added**
│   ├── test_maps.py            # **added**
│   ├── test_env.py
│   ├── test_grid_utils.py      # **added**
│   ├── test_render.py          # **added**
│   ├── test_baselines.py
│   ├── test_evaluate.py        # **added**
│   ├── test_value_iteration.py # **added**
│   ├── test_tabular.py
│   ├── test_deep.py
│   └── test_pipeline.py        # **added**: every script in --quick mode, end to end
├── results/
│   └── .gitkeep
└── slides/
    └── .gitkeep
```

Every package directory has an `__init__.py`. `tests/test_demo.py` is added by T8.4.

---

<a id="4"></a>
## 4. Module specifications

### 4.1 `env/maps.py`

**Constants**

```python
FREE: int = 0
OBSTACLE: int = 1
```

**Fixed maps** (`FIXED_MAPS: dict[str, list[str]]`). Copy character by character:

```python
FIXED_MAPS = {
    # 3x3 empty: |F| = 9. VI verification, fast tabular tests.
    "empty_3x3": [
        "...",
        "...",
        "...",
    ],
    # 4x4 empty: |F| = 16. E7, medium-size VI check.
    "empty_4x4": [
        "....",
        "....",
        "....",
        "....",
    ],
    # 4x4 with 3 obstacles: |F| = 13. Main tabular map (E1, E2, E6).
    "obstacle_4x4": [
        "....",
        ".#..",
        "...#",
        "#...",
    ],
    # 5x5 and 6x6 empty: only for task 5.3 (where tabular breaks down).
    "empty_5x5": ["....."] * 5,
    "empty_6x6": ["......"] * 6,
    # 6x6 room with furniture: |F| = 29. Intermediate DQN map.
    "room_6x6": [
        "......",
        ".##...",
        ".##.#.",
        "....#.",
        "#.....",
        "......",
    ],
    # 8x8 with furniture (2x2 table, cabinet, chair legs): |F| = 52. Main deep RL map.
    "furniture_8x8": [
        "........",
        ".##.....",
        ".##..##.",
        "........",
        "...#....",
        ".#.#..#.",
        ".#....#.",
        "........",
    ],
}
```

(Comments in the real file are written in Vietnamese per 0.6.)

**Public functions**

```python
def parse_map(lines: list[str]) -> np.ndarray:
    """(H, W) int8 array. ValueError on ragged rows, unknown characters, or no free cell."""

def get_map(name: str) -> np.ndarray:
    """Return a COPY of a fixed map. KeyError listing valid names."""

def free_cells(grid: np.ndarray) -> list[tuple[int, int]]:
    """Free cells in row-major order. This order defines the D-12 bitmask; never change it."""

def is_connected(grid: np.ndarray) -> bool:
    """True if all free cells are 4-connected (BFS from the first free cell)."""

def generate_random_map(height: int, width: int, obstacle_density: float,
                        rng: np.random.Generator, max_tries: int = 1000) -> np.ndarray:
    """Exactly round(density*H*W) obstacles, always connected."""

def generate_map_set(height: int, width: int, obstacle_density: float,
                     n_maps: int, seed: int) -> list[np.ndarray]:
    """n_maps pairwise-distinct maps from one seed. Used for E3/E4 train/test sets."""
```

**`generate_random_map` algorithm** (required, guarantees connectivity and exact obstacle count):

1. `n_obs = round(density * H * W)`; require `0 <= n_obs <= H*W - 2`, else `ValueError`.
2. `grid = zeros`; take a random permutation of all cells (`rng.permutation`).
3. For each cell in the permutation: tentatively place an obstacle; keep it if `is_connected(grid)`, otherwise remove it. Stop at `n_obs`.
4. If the permutation runs out before `n_obs`, retry with a new permutation, up to `max_tries`, then `RuntimeError`.

**Tests (`tests/test_maps.py`)**

- `parse_map` returns correct shape/dtype/values; unknown character → `ValueError`; ragged rows → `ValueError`.
- All `FIXED_MAPS` are connected; `get_map` returns a copy (mutating it doesn't change the original).
- Free counts: `empty_3x3 == 9`, `obstacle_4x4 == 13`, `room_6x6 == 29`, `furniture_8x8 == 52`.
- `generate_random_map` over 200 seeds × densities `{0, 0.1, 0.2, 0.3}` on 8×8: always connected, obstacle count exactly `round(d*64)`.
- Same seed → same map; different seeds → (almost always) different maps.
- `generate_map_set` returns `n_maps` pairwise-distinct maps.

---

### 4.2 `env/vacuum_env.py`

#### 4.2.1 `RewardConfig`

```python
@dataclass(frozen=True)
class RewardConfig:
    new_cell: float = 1.0               # entering an uncleaned cell
    revisit: float = -0.1               # entering a cleaned cell
    collision: float = -0.5             # bumping wall/obstacle (stay in place)
    step: float = -0.01                 # every time step
    completion: float = 10.0            # bonus when 100% cleaned
    shaping_coef: float = 0.0           # potential-based shaping (D-14); 0 = off
    naive_progress_bonus: float = 0.0   # hackable reward (D-15); 0 = off
    gamma: float = 0.99                 # used only by potential shaping

    @classmethod
    def dense(cls) -> "RewardConfig": ...                     # B2 default (v1)
    @classmethod
    def sparse(cls) -> "RewardConfig": ...                    # completion=10, everything else 0 (E2a)
    @classmethod
    def shaped(cls, coef: float = 1.0) -> "RewardConfig": ... # dense + shaping (E2c)
    @classmethod
    def hackable(cls, bonus: float = 0.5) -> "RewardConfig": ...
        # new_cell=1, revisit=0, collision=0, step=0, completion=10, naive_progress_bonus=bonus
    @classmethod
    def from_name(cls, name: str, **overrides) -> "RewardConfig": ...  # "dense"|"sparse"|"shaped"|"hackable"
```

`frozen=True` so configs cannot be mutated mid-ablation.

#### 4.2.2 Pure transition function (D-11)

```python
ACTION_DELTAS: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
ACTION_NAMES: tuple[str, ...] = ("up", "down", "left", "right")

@dataclass(frozen=True)
class TransitionResult:
    next_pos: tuple[int, int]
    next_cleaned: np.ndarray      # NEW array (copy), shape (H, W), dtype bool
    reward: float
    terminated: bool
    event: str                    # "new_cell" | "revisit" | "collision"

def transition(grid: np.ndarray, pos: tuple[int, int], cleaned: np.ndarray,
               action: int, cfg: RewardConfig) -> TransitionResult:
    """Deterministic MDP model p(s', r | s, a). No global state, no randomness,
    never mutates the input `cleaned`."""
```

Algorithm:

```
dr, dc = ACTION_DELTAS[action]
nr, nc = pos + (dr, dc)
if (nr, nc) out of bounds or grid[nr, nc] == OBSTACLE:
    next_pos = pos; event = "collision"; event_r = cfg.collision
elif cleaned[nr, nc]:
    next_pos = (nr, nc); event = "revisit"; event_r = cfg.revisit
else:
    next_pos = (nr, nc); event = "new_cell"; event_r = cfg.new_cell
next_cleaned = cleaned.copy(); next_cleaned[next_pos] = True
terminated = next_cleaned[grid == FREE].all()
reward = cfg.step + event_r + (cfg.completion if terminated else 0)
if cfg.shaping_coef != 0 or cfg.naive_progress_bonus != 0:
    d_s  = bfs_distance_to_nearest_uncleaned(grid, pos, cleaned)
    d_s2 = 0 if terminated else bfs_distance_to_nearest_uncleaned(grid, next_pos, next_cleaned)
if cfg.shaping_coef != 0:
    phi_s  = -d_s / (H + W)
    phi_s2 = 0 if terminated else -d_s2 / (H + W)
    reward += cfg.shaping_coef * (cfg.gamma * phi_s2 - phi_s)
if cfg.naive_progress_bonus != 0 and d_s2 < d_s:
    reward += cfg.naive_progress_bonus
```

`bfs_distance_to_nearest_uncleaned` lives in `env/grid_utils.py` (4.3).

#### 4.2.3 `VacuumEnv`

```python
class VacuumEnv(gymnasium.Env):
    metadata = {"render_modes": ["rgb_array", "ansi"], "render_fps": 4}

    def __init__(self, grid: np.ndarray | list[str] | str,
                 reward_config: RewardConfig | None = None,
                 max_steps: int | None = None,
                 start: tuple[int, int] | None = None,
                 render_mode: str | None = None): ...
```

- `grid`: array, list of strings, or a fixed-map **name** (`str` → `get_map`). Always store a copy.
- Validate: connected (else `ValueError`), `|F| >= 2` (D-06), valid `render_mode`, `max_steps > 0`.
- `self.action_space = spaces.Discrete(4)`
- `self.observation_space = spaces.Box(0.0, 1.0, shape=(3, H, W), dtype=np.float32)`
- Read-only properties: `grid` (copy), `height`, `width`, `n_free`, `position`, `cleaned` (copy), `steps`, `reward_config`.

**`reset(self, *, seed=None, options=None) -> tuple[np.ndarray, dict]`**

1. `super().reset(seed=seed)` (required so `self.np_random` is seeded the Gymnasium way).
2. Start: `options["start"]` if given → else `self._start` if not None → else uniform over `free_cells` using `self.np_random.integers`.
3. `cleaned = zeros(bool)`, mark start. Reset counters: `steps, n_revisit, n_collisions`.
4. Return `(self._get_obs(), self._get_info(collided=False, new_cell=False, event="reset"))`.

**`step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]`**

1. Accept `int` or `np.integer`; invalid → `ValueError`. Calling `step` before `reset` → `RuntimeError`.
2. Call `transition(...)`, update position, `cleaned`, counters.
3. `self.steps += 1`; `truncated = self.steps >= self.max_steps and not terminated`.
4. Return reward as a Python `float`.

**`_get_obs()`**: build a **new** `(3, H, W)` float32 array every call: channel 0 one-hot robot, channel 1 cleaned, channel 2 obstacles. Never return a reference to internal arrays (C2.11 state-leak bug).

**`info`** (every key always present, also at `reset`):

| Key | Type | Meaning |
|---|---|---|
| `coverage` | float | D-09 |
| `collided` | bool | this step was a collision |
| `new_cell` | bool | this step entered a new cell |
| `event` | str | `"reset" \| "new_cell" \| "revisit" \| "collision"` |
| `steps` | int | steps taken |
| `n_cleaned` | int | cells cleaned |
| `n_free` | int | `|F|` |
| `position` | tuple[int, int] | position after the step |
| `n_revisit` | int | cumulative |
| `n_collisions` | int | cumulative |
| `action_mask` | np.ndarray int8 (4,) | 1 if the action does not bump |

**`render()`**: `rgb_array` → `render.render_rgb(...)`; `ansi` → multi-line string: `#` obstacle, `.` uncleaned, `o` cleaned, `R` robot; `render_mode=None` → `None`.

**`state_key(self) -> tuple`**: optional fast path; if implemented, must equal `StateEncoder(grid).encode_obs(obs)` (tested).

#### 4.2.4 `RandomMapVacuumEnv`

```python
class RandomMapVacuumEnv(VacuumEnv):
    def __init__(self, height: int, width: int, obstacle_density: float,
                 map_pool: list[np.ndarray] | None = None,
                 reward_config: RewardConfig | None = None,
                 max_steps_factor: int = 4, render_mode: str | None = None): ...
```

- On each `reset`: if `map_pool` → choose one uniformly with `self.np_random`; else `generate_random_map(..., rng=self.np_random)`.
- `max_steps = max_steps_factor * |F|` recomputed per map. Start always random.
- `observation_space` unchanged.
- `reset(options={"grid": arr})` forces a specific map (for fixed test sets).

#### 4.2.5 Tests (`tests/test_env.py`), minimum set

| Test | Setup | Expected |
|---|---|---|
| `test_reset_shapes` | `empty_3x3`, start (0,0) | obs `(3,3,3)` float32, `obs[0].sum()==1`, `obs[1][0,0]==1`, `info["coverage"]==1/9` |
| `test_collision_wall` | start (0,0), action 0 | position unchanged, `reward == -0.51`, `collided`, `steps==1` |
| `test_collision_obstacle` | `obstacle_4x4`, start (0,1), action 1 (into `#` at (1,1)) | stays, reward `-0.51` |
| `test_new_cell` | start (0,0), action 3 | pos (0,1), reward `0.99`, `new_cell` |
| `test_revisit` | start (0,0), actions 3 then 2 | 2nd reward `-0.11` |
| `test_terminated_on_full` | `empty_3x3` start (0,0), snake `[3,3,1,2,2,1,3,3]` | last step `terminated=True`, `truncated=False`, last reward `10.99`, total return `17.92` (tol 1e-6) |
| `test_truncated` | `max_steps=3`, bump wall 3× | step 3 `truncated=True`, `terminated=False` |
| `test_never_both` | 500 random steps, many episodes | never `terminated and truncated` |
| `test_step_before_reset` | new env, `step(0)` | `RuntimeError` |
| `test_seed_reproducible` | start=None, `reset(seed=42)` twice | same start, same obs |
| `test_obs_not_aliased` | keep obs, step, compare | old obs unchanged |
| `test_transition_pure` | call `transition` twice with same input | same result, input `cleaned` unchanged |
| `test_invalid_start` | start on obstacle | `ValueError` |
| `test_disconnected_map` | isolated free cell | `ValueError` |
| `test_sparse_reward` | `RewardConfig.sparse()`, snake path | all rewards 0 except last = 10 |
| `test_potential_shaping_telescopes` | `shaped(coef=1)` with `gamma=1.0` in config, any episode to termination | sum of shaping terms = `-Φ(s0)` |
| `test_hackable_loop` | `hackable(0.5)`, manual "away then back" moves next to the frontier | the "back" step earns `+0.5`, the "away" step earns `0` |
| `test_action_mask` | corner (0,0) | `[0,1,0,1]` |
| `test_check_env` | `gymnasium.utils.env_checker.check_env(VacuumEnv("empty_3x3"))` | no exception |
| `test_random_map_env` | `RandomMapVacuumEnv(6,6,0.2)`, two seeds | same obs shape, different maps |

---

### 4.3 `env/grid_utils.py`

```python
def bfs_distances(grid: np.ndarray, source: tuple[int, int]) -> np.ndarray:
    """(H, W) int32 4-neighbour distances from source; -1 for obstacles/unreachable."""

def bfs_distance_to_nearest_uncleaned(grid, pos, cleaned) -> int:
    """Distance to nearest uncleaned free cell; 0 if everything is cleaned."""

def bfs_first_action_towards(grid, pos, targets: np.ndarray) -> int | None:
    """First action on a shortest path from pos to the nearest cell where targets is True.
    Tie-break: expand neighbours in action order 0,1,2,3 (D-20) → deterministic.
    None if no target is reachable."""

def decode_obs(obs: np.ndarray) -> tuple[tuple[int, int], np.ndarray, np.ndarray]:
    """obs (3,H,W) → (position, cleaned bool (H,W), obstacles int8 (H,W)).
    ValueError if channel 0 is not exactly one-hot."""

def valid_actions(obs: np.ndarray) -> list[int]:
    """Actions that don't bump a wall/obstacle."""

class StateEncoder:
    """Tabular state encoding per D-12."""
    def __init__(self, grid: np.ndarray, state_mode: str = "full"): ...
        # state_mode: "full" -> (r, c, mask); "position_only" -> (r, c)  (E1)
    free_index: dict[tuple[int, int], int]
    full_mask: int                   # (1 << |F|) - 1
    def encode(self, pos, cleaned) -> tuple: ...
    def encode_obs(self, obs) -> tuple: ...
    def decode_mask(self, mask: int) -> np.ndarray: ...
```

Fast mask: precompute `free_flat_idx` and `weights = 1 << arange(|F|)` (int64); `mask = int((cleaned.ravel()[free_flat_idx].astype(np.int64) * weights).sum())`. `ValueError` if `|F| > 62`.

**Tests (`tests/test_grid_utils.py`)**: BFS distances match hand-computed values on a map with obstacles; `encode` → `decode_mask` round-trips; `encode_obs(obs) == encode(pos, cleaned)` on 100 random states; `position_only` yields 2-tuples; `bfs_first_action_towards` deterministic; `decode_obs` rejects a bad channel 0.

---

### 4.4 `env/wrappers.py`

```python
class HideCleanedChannel(gymnasium.ObservationWrapper):
    """Zero channel 1 (E1, deep version): agent sees only position + obstacles."""
```

Tests: channel 1 is all zeros after reset and step, other channels unchanged, `check_env` passes on the wrapped env.

---

### 4.5 `env/render.py`

```python
CELL_PX: int = 32
COLORS = {  # RGB uint8
    "free": (235, 235, 235), "cleaned": (160, 210, 255),
    "obstacle": (60, 60, 60), "robot": (220, 50, 50), "grid_line": (200, 200, 200),
}

def render_rgb(grid, pos, cleaned, cell_px: int = CELL_PX,
               trail: list[tuple[int, int]] | None = None) -> np.ndarray:
    """(H*cell_px, W*cell_px, 3) uint8, drawn with Pillow ImageDraw (D-26):
    floor cells, cleaned cells, obstacles as dark blocks, thin grid lines,
    optional trail (polyline through the last ≤ 30 positions, semi-transparent),
    robot as a filled circle (20% margin) with a small dot showing its last move direction."""

def annotate_frame(frame: np.ndarray, title: str, lines: list[str]) -> np.ndarray:
    """Add a header band (agent name) and a footer band (e.g. 'bước 52 · 100% · lặp 3')
    using Pillow's default font. Returns a new, taller image."""

def compose_side_by_side(frames: list[np.ndarray], gap_px: int = 8) -> np.ndarray:
    """Place equally-sized frames horizontally with a white gap."""

def render_ansi(grid, pos, cleaned) -> str: ...

def save_gif(frames: list[np.ndarray], path: str | Path, fps: int = 4) -> None:
    """Pillow: Image.fromarray(...).save(save_all=True, append_images=..., duration=1000//fps, loop=0).
    Creates parent directories. Path must be inside ROOT."""

def record_episode(agent, env, path: str | Path, seed: int = 0,
                   max_frames: int = 500, fps: int = 4) -> dict:
    """Run one episode (env must have render_mode='rgb_array'), save GIF, return metrics."""

def demo_random_gif(map_name: str, path: str | Path, seed: int = 0) -> Path:
    """Convenience demo: random actions on a fixed map → GIF."""

def visit_heatmap(visits: np.ndarray, grid: np.ndarray, ax=None):
    """matplotlib imshow of visit counts per cell; obstacles grey."""
```

Tests (`tests/test_render.py`): image shape; pixel at robot cell centre has robot colour; `annotate_frame` output is taller than input and same width; `compose_side_by_side` width = sum of widths + gaps; `save_gif` writes a non-empty file under `tmp_path`.

---

### 4.6 `utils/`

```python
# utils/paths.py
ROOT: Path = Path(__file__).resolve().parents[1]
def setup_local_caches() -> None:
    """os.environ.setdefault('MPLCONFIGDIR', ROOT/.cache/matplotlib); create the dir.
    Called at the top of every experiments/* script and in tests/conftest.py (rule 0.1.4)."""
def assert_inside_root(path: Path) -> Path:
    """Resolve path; raise ValueError if it is outside ROOT. Used by every writer in utils/io.py."""

# utils/seeding.py
def set_global_seed(seed: int) -> None:
    """Seed random, numpy global (for third-party code), and torch if importable."""

# utils/io.py
def ensure_dir(path) -> Path: ...
def save_csv(df: pd.DataFrame, path) -> Path: ...            # parent dirs, index=False, inside ROOT
def save_config_json(config: dict | Any, path) -> Path: ...  # dataclasses -> asdict; adds timestamp, git commit hash (if any), python/torch versions
```

Every experiment writes `config.json` next to its CSVs (honest hyperparameter reporting, Part E rule 4).

---

### 4.7 Baselines

#### `baselines/agent.py`

```python
class Agent(Protocol):
    def act(self, obs: np.ndarray, info: dict | None = None) -> int: ...
```

#### `baselines/random_walk.py`

```python
class RandomWalkAgent:
    def __init__(self, seed: int = 0, valid_only: bool = True): ...
    def act(self, obs, info=None) -> int:
        # valid_only: uniform over valid_actions(obs); otherwise uniform over 0..3
    def reset(self) -> None: ...   # does NOT reseed (episodes should differ)
```

#### `baselines/zigzag.py`

```python
class ZigzagAgent:
    """D-19. Deterministic, stateless (recomputed from obs every step)."""
    def act(self, obs, info=None) -> int:
        pos, cleaned, obstacles = decode_obs(obs)
        order = snake_order(obstacles)            # cached by obstacles.tobytes()
        target = first cell in order not cleaned
        return bfs_first_action_towards(obstacles, pos, mask_with_only(target))
```

Termination argument (put in the docstring): the target only changes after it is cleaned; BFS distance to the target drops by 1 each step; finitely many targets.

#### `baselines/greedy_bfs.py`

```python
class GreedyBFSAgent:
    """Go to the NEAREST uncleaned cell (BFS), repeat. Deterministic."""
    def act(self, obs, info=None) -> int:
        pos, cleaned, obstacles = decode_obs(obs)
        targets = (obstacles == 0) & (~cleaned)
        return bfs_first_action_towards(obstacles, pos, targets)
```

Termination: distance to the nearest uncleaned cell drops by ≥ 1 each step until 0.

If `act` is called on an already fully-cleaned observation (should not happen inside `evaluate`), return `0` and document it.

#### `baselines/__init__.py`

```python
BASELINES = {"random_walk": RandomWalkAgent, "zigzag": ZigzagAgent, "greedy_bfs": GreedyBFSAgent}
def make_baseline(name: str, seed: int = 0) -> Agent: ...   # KeyError listing valid names
```

#### Tests (`tests/test_baselines.py`)

- Every baseline runs on `empty_3x3`, `obstacle_4x4`, `furniture_8x8`, all starts: no exception, actions in `0..3`.
- `zigzag` and `greedy_bfs`: coverage 1.0 on **every** fixed map with `max_steps = 4|F|`, and on 50 random 8×8 maps at density 0.3.
- `zigzag` on `empty_3x3` from (0,0): exactly 8 steps, no collisions.
- `greedy_bfs` never collides.
- `random_walk` with `valid_only=True`: `collisions == 0`.
- Determinism: two runs with the same seed → identical trajectories.

---

### 4.8 `experiments/evaluate.py`

```python
METRIC_COLUMNS = ["episode", "env_seed", "start_row", "start_col",
                  "return_undiscounted", "return_discounted", "coverage", "steps",
                  "steps_to_full", "redundancy", "collisions", "n_revisit",
                  "terminated", "truncated"]

def run_episode(agent, env, seed: int, gamma: float = 0.99,
                record_visits: bool = False, record_frames: bool = False) -> dict:
    """One episode. Returns METRIC_COLUMNS (minus 'episode'), plus 'visits' (H,W)
    if record_visits and 'frames' if record_frames."""

def evaluate(agent, env_factory: Callable[[], gymnasium.Env], n_episodes: int,
             seed: int, gamma: float = 0.99) -> pd.DataFrame:
    """C3. Create the env ONCE via env_factory(); episode i uses reset(seed=seed*10_000+i) (D-23).
    Calls agent.reset() at each episode start if present. Returns a METRIC_COLUMNS DataFrame."""

def summarize(df: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    """Mean and std (ddof=1) of numeric metrics, plus n and success_rate (fraction terminated).
    Columns like 'coverage_mean', 'coverage_std'."""
```

Definitions:
- `return_discounted = Σ_t γ^t r_t` (t from 0).
- `steps_to_full = steps` if `terminated`, else `NaN`.
- `gamma` is a parameter (default 0.99), not read from the env, so every reward config is measured with the same yardstick.
- **When comparing different reward configs (E2), compare `coverage`, `steps`, `redundancy` — never `return`**, because reward scales differ.

Tests (`tests/test_evaluate.py`): `evaluate(GreedyBFSAgent(), lambda: VacuumEnv("empty_3x3", start=(0,0)), 3, seed=0)` → 3 rows, correct columns, coverage 1, steps 8; `summarize` gives correct mean/std on a synthetic DataFrame; `agent.reset` is called once per episode (mock agent).

---

### 4.9 State space and Value Iteration

#### `tabular/state_space.py`

```python
@dataclass
class StateGraph:
    encoder: StateEncoder
    states: list[tuple[int, int, int]]          # index -> key (r, c, mask)
    index: dict[tuple[int, int, int], int]      # key -> index
    next_state: np.ndarray    # (S, 4) int64
    reward: np.ndarray        # (S, 4) float64
    terminal_next: np.ndarray # (S, 4) bool: s' is terminal
    is_terminal: np.ndarray   # (S,) bool

def build_state_graph(grid, cfg: RewardConfig, starts: list[tuple[int, int]] | None = None,
                      max_states: int = 5_000_000) -> StateGraph:
    """BFS from every initial state (pos, mask={pos}) for pos in starts (default: all free cells)
    → enumerate all REACHABLE states. Every edge uses transition() (D-11).
    Terminal states (mask == full) are absorbing: their rows point to themselves with reward 0.
    Exceeding max_states → RuntimeError ("grid too large for tabular" — the Phase 5 motivation)."""

def count_reachable_states(grid, max_states: int = 50_000_000,
                           time_limit_s: float = 600) -> int | None:
    """Count only (no rewards stored), for E7. None if a limit is exceeded."""

def min_steps_to_cover(graph: StateGraph, start: tuple[int, int]) -> int:
    """Unweighted BFS from (start, mask={start}) to the nearest terminal state.
    Exact minimum number of steps to clean everything (D-22)."""
```

#### `tabular/value_iteration.py`

```python
@dataclass
class VIResult:
    graph: StateGraph
    V: np.ndarray            # (S,)
    Q: np.ndarray            # (S, 4)
    policy: np.ndarray       # (S,) int, deterministic argmax (D-20)
    n_iterations: int
    delta_history: list[float]

def value_iteration(graph: StateGraph, gamma: float = 0.99, theta: float = 1e-10,
                    max_iterations: int = 100_000) -> VIResult: ...

def greedy_rollout(result: VIResult, start: tuple[int, int], max_steps: int) -> dict:
    """Follow π* from start; return steps, return_discounted, coverage, trajectory."""

class VIAgent:
    """Wraps π* as an Agent (act(obs)) so it runs through evaluate()."""

def save_vi_result(result: VIResult, path) -> None: ...
def load_vi_result(path) -> VIResult: ...
```

Pseudocode (comments must map each line to S&B section 4.4):

```
# S&B 2018, 4.4 "Value Iteration, for estimating π ≈ π*"
# "Initialize V(s) arbitrarily for all s, except V(terminal) = 0"
V = zeros(S)
loop:
    # BIẾN THỂ (D-21): synchronous vectorized update instead of in-place sweep;
    # section 4.1 confirms the two-array version also converges.
    # "V(s) <- max_a Σ p(s',r|s,a)[r + γV(s')]"  (deterministic MDP: single term)
    Q = reward + gamma * where(terminal_next, 0, V[next_state])
    V_new = Q.max(axis=1); V_new[is_terminal] = 0
    # "Δ <- max(Δ, |v - V(s)|)" ... "until Δ < θ"
    delta = abs(V_new - V).max(); V = V_new
    if delta < theta: break
# "Output a deterministic policy π ≈ π*, such that π(s) = argmax_a ..."
policy = argmax with lowest-index tie-break
```

`max_iterations` reached without convergence → `RuntimeError`.

#### Tests (`tests/test_value_iteration.py`)

| Test | Expected |
|---|---|
| State count `empty_3x3` | > 0; every key's mask contains the robot's own bit |
| `V*` on `empty_3x3` from (0,0) | `16.968928730705` (tol 1e-6; appendix 10) |
| `min_steps_to_cover` on `empty_3x3` | (0,0) → 8; (1,1) → 8; (0,1) → 9 (checkerboard argument, appendix) |
| `greedy_rollout` from (0,0) | 8 steps, coverage 1.0 |
| Optimal ≥ baselines | For every start on `empty_3x3` and `obstacle_4x4`: `V*(s0) >= greedy_bfs return_discounted - 1e-9`, `min_steps <= greedy_bfs steps`, `min_steps <= zigzag steps` |
| Bellman residual | `max |V - max_a(r + γ V')|` < 1e-8 |
| `obstacle_4x4` runtime | < 30 s; log `n_states`, `n_iterations` |
| Large grid guard | `build_state_graph(get_map("empty_6x6"), ..., max_states=10_000)` → `RuntimeError` |
| Save/load | round-trip gives identical `V` and `policy` |

---

### 4.10 Tabular RL: Q-learning (main model); SARSA (optional O1)

The base class in `common.py` is written so that SARSA can later subclass it, but **no SARSA code, CLI choice, or test exists until O1** (1.1). The SARSA spec below is reference material for that card.

#### `tabular/common.py`

```python
@dataclass
class TabularConfig:
    n_episodes: int = 20_000
    alpha: float = 0.5                   # deterministic env → a large α is fine (S&B 6.5 allows any α in (0,1])
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_fraction: float = 0.8  # linear decay over the first 80% of episodes
    q_init: float = 0.0
    state_mode: str = "full"             # "full" | "position_only" (E1)
    eval_every: int = 100                # greedy evaluation every N episodes
    eval_episodes: int = 5               # 1 is enough with fixed start + deterministic env
    time_budget_min: float | None = None # stop early when exceeded (task 5.3)
    checkpoint_episodes: tuple[int, ...] = ()  # save Q snapshots at these episodes (for the
                                               # "robot learns" demo in T8.4), e.g. (0, 1000, 10000)
    seed: int = 0

def linear_epsilon(episode: int, cfg: TabularConfig) -> float: ...

class TabularAgent:
    """Base class: holds Q (defaultdict -> np.full(4, q_init)), encoder, rng.
    Subclasses implement only _run_episode(env, epsilon) following the pseudocode exactly."""
    def __init__(self, grid: np.ndarray, cfg: TabularConfig): ...
    def q_values(self, state_key) -> np.ndarray: ...
    def epsilon_greedy(self, state_key, epsilon: float) -> int: ...  # random tie-break (D-20)
    def greedy_action(self, state_key) -> int: ...                   # lowest-index tie-break
    def act(self, obs, info=None) -> int: ...                         # greedy, for evaluate()
    def train(self, env_factory, log_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Returns (train_log, eval_log); writes CSVs if log_dir is given."""
    def save(self, path) -> None: ...      # pickle the Q dict (small file)
    @classmethod
    def load(cls, path) -> "TabularAgent": ...
```

`train_log` (one row per episode): `episode, epsilon, return_undiscounted, coverage, steps, terminated, truncated, n_states_visited, q_start_max, wall_time_s`. `q_start_max = max_a Q(s0, a)` for the fixed start state (plotted against `V*(s0)` from VI to show Q-learning's estimate converging to the true optimal value; `NaN` when the start is random).

`eval_log`: every `eval_every` episodes, run the greedy policy through `evaluate(..., n_episodes=eval_episodes)` on a **separate env instance** and record `episode` + `summarize` columns.

**`terminated` vs `truncated`** (C2.11):
- `terminated` → TD target = `R` (Q(terminal,·) = 0).
- `truncated` → **still bootstrap** from `S'`, then end the episode.

#### `tabular/q_learning.py` (S&B 6.5)

```
# "Initialize Q(s,a) arbitrarily, except Q(terminal,·) = 0"
# "Loop for each episode:"  "Initialize S"
S = encode(reset())
# "Loop for each step of episode:"
#   "Choose A from S using policy derived from Q (e.g., ε-greedy)"
A = epsilon_greedy(S, ε)
#   "Take action A, observe R, S'"
obs', R, term, trunc = step(A); S' = encode(obs')
#   "Q(S,A) <- Q(S,A) + α[R + γ max_a Q(S',a) − Q(S,A)]"
target = R if term else R + γ * max(Q[S'])
Q[S][A] += α * (target - Q[S][A])
#   "S <- S'"   "until S is terminal"   (+ stop on truncated)
```

Snapshots: at each episode in `checkpoint_episodes` (and at the end), `save()` to `log_dir/checkpoints/ep{episode}.pkl`. Episode `0` means "before any update" (all-`q_init` table → effectively the untrained agent).

#### `tabular/sarsa.py` (S&B 6.4) — optional card O1

```
S = encode(reset()); A = epsilon_greedy(S, ε)          # "Choose A from S ..."
loop:
    obs', R, term, trunc = step(A); S' = encode(obs')  # "Take action A, observe R, S'"
    if term: target = R
    else:
        A' = epsilon_greedy(S', ε)                     # "Choose A' from S' ..."
        target = R + γ * Q[S'][A']
    Q[S][A] += α (target − Q[S][A])                    # "Q(S,A) <- Q(S,A) + α[R + γQ(S',A') − Q(S,A)]"
    if term or trunc: break
    S, A = S', A'                                      # "S <- S'; A <- A'"
```

ε is fixed within an episode (set from the episode index). The docstring of `SarsaAgent` must state the one-line difference from Q-learning and why it matters (SARSA learns the value of the ε-greedy policy it actually follows, so it is more "careful"; Q-learning learns the greedy optimum).

#### Tests (`tests/test_tabular.py`)

Core (T4.1):

| Test | Setup | Expected |
|---|---|---|
| `test_q_update_single_step` | hand-made Q, one fake transition | value matches the formula (hand-computed) |
| `test_terminal_no_bootstrap` | terminated transition, Q[S'] ≠ 0 | target = R |
| `test_truncated_bootstraps` | `max_steps=1`, Q[S'] ≠ 0 | target = R + γ max Q[S'] |
| `test_q_learning_matches_vi_3x3` (slow if > 10 s) | `empty_3x3` start (0,0), 3000 episodes, seeds 0..2 | greedy policy: coverage 1.0, steps == 8 (= `min_steps`); `|max Q(s0) − V*(s0)| < 0.5` |
| `test_position_only_key` | `state_mode="position_only"` | every Q key has length 2 |
| `test_reproducible` | same seed twice | identical `train_log` (ignoring `wall_time_s`) |
| `test_save_load` | save then load | identical greedy actions on 50 states |
| `test_checkpoints` | `checkpoint_episodes=(0, 10)`, 20 episodes | files `ep0.pkl`, `ep10.pkl`, final exist; `ep0` table is all `q_init` |

Added by optional cards:

| Test | Card | Expected |
|---|---|---|
| `test_sarsa_update_uses_next_action` | O1 | uses Q[S'][A'], not max |
| `test_sarsa_matches_vi_3x3` (`epsilon_end=0.01`) | O1 | steps == 8 |

If a convergence test fails with these budgets: apply 0.4 — first look for a real bug (compare a few updates by hand, check terminal handling). If the code is correct and only the budget is too small, report the learning curve and the proposed new budget, and ask before changing the test.

---

### 4.11 Deep RL

#### `deep/networks.py`

```python
class QNetwork(nn.Module):
    """Small CNN for obs (3, H, W). Keeps spatial size (padding=1); no pooling,
    because per-cell position information matters on tiny grids."""
    def __init__(self, height: int, width: int, n_actions: int = 4,
                 channels: tuple[int, int] = (32, 64), hidden: int = 256): ...
    # Conv2d(3,32,3,p=1) -> ReLU -> Conv2d(32,64,3,p=1) -> ReLU -> Flatten
    # -> Linear(64*H*W, 256) -> ReLU -> Linear(256, n_actions)
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...  # (B,3,H,W) -> (B,4)

def count_parameters(model: nn.Module) -> int: ...
```

Tests: output shape `(B, 4)` for `H,W ∈ {4, 8, 10}`; gradient flows (loss.backward gives non-None grads).

#### `deep/replay_buffer.py`

```python
class ReplayBuffer:
    """Circular buffer with preallocated numpy arrays."""
    def __init__(self, capacity: int, obs_shape: tuple[int, ...], seed: int = 0): ...
    def add(self, obs, action: int, reward: float, next_obs, terminated: bool) -> None: ...
        # store COPIES; store only terminated, NEVER truncated
    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        # keys: obs (B,*obs_shape) float32, actions (B,) int64, rewards (B,) float32,
        #       next_obs (B,*obs_shape) float32, terminated (B,) float32
    def __len__(self) -> int: ...
```

Store obs as `uint8` (values are 0/1), convert to float32 in `sample`. `sample` with `batch_size > len` → `ValueError`.

Tests: `len` correct below capacity; above capacity `len == capacity` and the oldest entry is overwritten; `sample` shapes/dtypes; same seed → same batch; mutating the original obs after `add` doesn't change stored data.

#### `deep/dqn.py` — Deep Q-learning (S&B 6.5 target + S&B 9/16.5; Mnih et al. 2015)

**Explain it as Q-learning with a network** (1.1). The module docstring (Vietnamese) must contain this mapping table so a student can explain it in the oral exam:

| Tabular Q-learning (`tabular/q_learning.py`) | DQN (`deep/dqn.py`) | Why the change |
|---|---|---|
| `Q[S][A]` looked up in a dict | `QNetwork(obs)[A]` computed by a CNN | Too many states for a table; the network generalizes to unseen states/rooms (S&B 9.1) |
| Target `R + γ max_a Q(S',a)` | Same target, with `max` over a **frozen copy** of the network (target network) | A target that moves every step makes learning unstable (S&B 11.3 deadly triad; Mnih 2015) |
| Update right after each step | Store the step in a **replay buffer**, learn from random mini-batches | Consecutive steps are highly correlated; random batches decorrelate them (Mnih 2015) |
| `Q[S][A] += α (target − Q[S][A])` | gradient step on Huber loss `(Q(S,A) − target)` | Same "move estimate toward target" idea; α becomes the learning rate |
| ε-greedy, linear decay | ε-greedy, linear decay | Identical |
| `terminated` → no bootstrap | `(1 − terminated)` mask | Identical (C2.11) |

```python
@dataclass
class DQNConfig:
    total_timesteps: int = 200_000
    learning_rate: float = 5e-4
    gamma: float = 0.99
    buffer_size: int = 50_000
    batch_size: int = 64
    learning_starts: int = 2_000          # random-action warm-up steps
    train_freq: int = 4                   # one gradient step every 4 env steps (Mnih 2015)
    target_update_interval: int = 1_000   # hard target update, in env steps
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_fraction: float = 0.3   # linear decay over the first 30% of steps
    max_grad_norm: float = 10.0
    loss: str = "huber"                   # "huber" | "mse"
    reward_scale: float = 1.0             # multiply rewards before storing (only if needed for stability)
    checkpoint_steps: tuple[int, ...] = ()  # save model snapshots (demo "robot learns", T8.4)
    eval_every_steps: int = 5_000
    eval_episodes: int = 20
    log_every_steps: int = 500
    device: str = "cpu"
    seed: int = 0

def compute_targets(batch: dict, target: nn.Module, gamma: float) -> torch.Tensor:
    """y = r + γ (1 − terminated) max_a Q_target(s', a), under torch.no_grad().
    Separate pure function so it can be unit-tested; it only receives the target network,
    so it is impossible to accidentally use the online network here."""

class DQNAgent:
    def __init__(self, obs_shape: tuple[int, int, int], cfg: DQNConfig): ...
    def act(self, obs, info=None, epsilon: float = 0.0) -> int: ...   # greedy for evaluation
    def train(self, env_factory, log_dir: Path) -> None: ...
    def save(self, path) -> None: ...
    @classmethod
    def load(cls, path, obs_shape) -> "DQNAgent": ...
```

Training loop (comments map to Algorithm 1 of Mnih et al. 2015):

```
set_global_seed(cfg.seed)
env = env_factory(); obs, _ = env.reset(seed=cfg.seed)
for t in 1..total_timesteps:
    ε = linear schedule(t)
    a = random if t <= learning_starts or rand < ε else argmax Q_online(obs)
    obs2, r, term, trunc, info = env.step(a)
    buffer.add(obs, a, r * reward_scale, obs2, term)      # only term (C2.11)
    obs = obs2
    if term or trunc: log episode; obs, _ = env.reset()
    if t > learning_starts and t % train_freq == 0:
        batch = buffer.sample(batch_size)
        # S&B 6.5: "R + γ max_a Q(S',a)" — the max uses the target network (Mnih 2015)
        y = compute_targets(batch, target, gamma)
        loss = huber(Q_online(s)[a], y)       # "move Q(S,A) toward the target", like Q[S][A] += α(...)
        optimizer.zero_grad(); loss.backward(); clip_grad_norm_; optimizer.step()
    if t % target_update_interval == 0: target.load_state_dict(online.state_dict())
    if t % eval_every_steps == 0: evaluate(greedy agent, env_factory, eval_episodes, seed=10_000+t) → eval CSV
    if t in checkpoint_steps: save(log_dir / "checkpoints" / f"step{t}.pt")
save step0.pt before the loop if 0 in checkpoint_steps; save final model + final evaluation
```

CSV logs:
- `train_episodes.csv`: `global_step, episode, return_undiscounted, coverage, steps, terminated, truncated, epsilon`
- `train_losses.csv` (every `log_every_steps`): `global_step, loss, mean_q, mean_target, grad_norm`
- `eval.csv`: `global_step` + `summarize` columns
- `config.json`, `model.pt` (gitignored)

Tests (`tests/test_deep.py`):
- Smoke: `DQNConfig(total_timesteps=500, learning_starts=100, eval_every_steps=250)` on `empty_3x3` runs and creates all three CSVs + `config.json`.
- `compute_targets` with a fake target network of known output: `y` equals the hand-computed `r + γ max`.
- `terminated=1` → `y == r`.
- Save/load gives identical greedy actions.
- `checkpoint_steps=(0, 300)` in the smoke run → `step0.pt`, `step300.pt` exist.
- `slow`: DQN on `empty_3x3`, 30k steps → greedy policy coverage 1.0.

---

<a id="5"></a>
## 5. Task cards, in execution order

Card numbers match the tasks in Part D of `PLAN_rel.md`. Lettered sub-cards (`T2.1a`) are smaller loops. 🛑 = checkpoint: stop and wait for the user after reporting. Cards `O1`, `O2` are optional (1.1) and listed at the end of this section. Mapping to Part D of `PLAN_rel.md`: task 4.2 (SARSA) became O1; tasks 4.3 (Double Q), 6.3 (PPO), 6.4 (Double DQN) are **dropped** (1.1).

### Dependency overview (core)

```
T1.2 ─► T2.2 ─► T2.1 ─► T2.3 ─► T2.4
                  │
                  ├─► T3.1 ─► T3.2
                  │             │
                  └─► T3.3 ◄────┘
                        │
                        ├─► T4.1 ─► T4.4
                        └─► T5.1 ; T5.2 (needs T4.1) ; T5.3 (needs T4.1)
T2.x ─► T6.1 ─► T6.2
T4.4 + T6.2 ─► T7.1 .. T7.4 ─► T8.2 ─► T8.4
```

Core order to execute: T1.2, T2.2, T2.1a–c, T2.3, T2.4, T3.1, T3.2, T3.3, T4.1, T4.4, T5.1, T5.2, T5.3, T6.1, T6.2, T7.1, T7.2, T7.3, T7.4, T8.2, T8.4.

Minimum submission (if time runs out before deep RL): T1.2 → T4.4 + T5.1 + T5.2 + T7.1 (tabular part) + T7.4 + T8.2 + T8.4 (with Q-learning in place of DQN).

Every card implicitly includes: update `tests/test_pipeline.py` if the card adds a script, run the full fast suite, commit, report.

---

### T1.2 Repository bootstrap 🛑

- **Depends on:** nothing.
- **Creates:** the full layout of section 3 with empty `__init__.py` files, `requirements.txt`, `pytest.ini`, `.gitignore`, `README.md` (skeleton), `CHANGELOG_DESIGN.md`, `results/.gitkeep`, `slides/.gitkeep`, `utils/paths.py`, `tests/conftest.py`, `tests/test_smoke.py`.
- **Steps:**
  1. `git init` inside `ROOT` (local only, no remote).
  2. Create `.venv` inside `ROOT` and install per section 2.2.
  3. `tests/test_smoke.py`: imports `numpy, pandas, matplotlib, PIL, gymnasium, torch` succeed; `utils.paths.ROOT` equals the repo root.
  4. `README.md` skeleton: title, 3-line description, "Setup" (2.2), "Run tests", empty "Results" and "Reproduce experiments" sections.
  5. `CHANGELOG_DESIGN.md` first entry: the model-scope decision of 1.1 (Q-learning main, DQN as its scale-up, SARSA optional; Double Q, E5, Double DQN, PPO, curriculum dropped) with proposed B8 rows.
  6. `pip freeze > requirements-lock.txt`.
- **Acceptance:** install has no errors; `python -m pytest` green; `git status` clean after the first commit; nothing created outside `ROOT`.
- **Commit:** `chore: bootstrap repository structure`
- 🛑 Checkpoint: report the installed versions and ask the user to confirm git identity if needed.

### T2.2 Maps (before T2.1 because the env depends on it)

- **Depends on:** T1.2.
- **Files:** `env/maps.py`, `env/grid_utils.py` (only `bfs_distances` and helpers maps needs), `tests/test_maps.py`.
- **Spec:** 4.1.
- **Acceptance:** `python -m pytest tests/test_maps.py` green; print (ANSI) all 7 fixed maps in the report.
- **Commit:** `feat(env): fixed maps and connected random map generation`

### T2.1 Environment (three loops)

- **T2.1a** `RewardConfig` + `transition()` + `bfs_distance_to_nearest_uncleaned` + tests for `transition` (collision, new, revisit, terminated, shaping, hackable, purity).
- **T2.1b** `VacuumEnv` (`__init__`, `reset`, `step`, `_get_obs`, `_get_info`, ANSI `render`) + the 4.2.5 tests.
- **T2.1c** `RandomMapVacuumEnv` and the rest of `grid_utils` (`decode_obs`, `valid_actions`, `bfs_first_action_towards`, `StateEncoder`) + `tests/test_grid_utils.py`.
- **Depends on:** T2.2.
- **Files:** `env/vacuum_env.py`, `env/grid_utils.py`, `env/__init__.py`, `tests/test_env.py`, `tests/test_grid_utils.py`.
- **Acceptance:** `python -m pytest tests/test_env.py tests/test_grid_utils.py` green, including `test_check_env`.
- **Commits:** one per loop, `feat(env): ...`.

### T2.3 Rendering, GIFs, wrappers

- **Depends on:** T2.1.
- **Files:** `env/render.py`, `env/wrappers.py`, `tests/test_render.py`, wrapper tests in `tests/test_env.py`.
- **Demo command** (document in README):
  `python -c "from env.render import demo_random_gif; demo_random_gif('furniture_8x8', 'results/gifs/random_8x8.gif')"`
- **Acceptance:** tests green; `results/gifs/random_8x8.gif` exists, shows the robot moving and cells changing colour when cleaned (describe the first frames in the report).

### T2.4 Gymnasium compliance

- **Depends on:** T2.1, T2.3.
- **Work:** `check_env` passes for `VacuumEnv`, `RandomMapVacuumEnv`, `HideCleanedChannel(VacuumEnv)` with `render_mode` `None`, `"rgb_array"`, `"ansi"`.
- **Acceptance:** `test_check_env_*` green, no checker warnings (or a comment explaining any unavoidable one).

### T3.1 Baselines

- **Depends on:** T2.1.
- **Files:** `baselines/*.py`, `tests/test_baselines.py`.
- **Spec:** 4.7.
- **Acceptance:** `python -m pytest tests/test_baselines.py` green; GIFs `results/gifs/zigzag_8x8.gif`, `results/gifs/greedy_8x8.gif`.

### T3.2 Evaluation and baseline runs

- **Depends on:** T3.1.
- **Files:** `utils/seeding.py`, `utils/io.py`, `experiments/evaluate.py`, `experiments/run_baselines.py`, `tests/test_evaluate.py`, `tests/test_pipeline.py` (first entry).
- **CLI `run_baselines.py`:**
  ```
  --maps empty_3x3 obstacle_4x4 furniture_8x8     (default: all 3)
  --agents random_walk zigzag greedy_bfs          (default: all 3)
  --seeds 0 1 2 3 4
  --episodes 20                                   (per seed)
  --start random|fixed                            (fixed = (0,0))
  --out results/baselines
  --quick                                         (1 seed, 3 episodes)
  ```
- **Outputs:** `results/baselines/{map}/{agent}/seed{s}.csv`, `results/baselines/summary.csv` (map, agent + `summarize` columns), `config.json`; print a markdown summary table.
- **Acceptance:** default run < 2 min; `summary.csv` has 9 rows; zigzag and greedy `coverage_mean == 1.0`.

### T3.3 Value Iteration 🛑

- **Depends on:** T2.1 (T3.1 for comparison tests).
- **Files:** `tabular/state_space.py`, `tabular/value_iteration.py`, `experiments/run_value_iteration.py`, `tests/test_value_iteration.py`.
- **CLI:** `--maps empty_3x3 obstacle_4x4 empty_4x4 --gamma 0.99 --out results/vi [--quick]`.
- **Outputs:** `results/vi/{map}_summary.csv`, one row per start: `start_row, start_col, v_star, min_steps, vi_policy_steps, greedy_bfs_steps, zigzag_steps, greedy_bfs_return_disc`; `results/vi/{map}_meta.json`: `n_states, n_iterations, runtime_s`; `results/vi/{map}_vi.pkl`.
- **Acceptance:** tests green; `min_steps <= greedy_bfs_steps` for every start; values match appendix 10.
- 🛑 Checkpoint: this is the ground truth for everything after; show the summary tables to the user.

### T4.1 Q-learning

- **Depends on:** T3.3.
- **Files:** `tabular/common.py`, `tabular/q_learning.py`, Q-learning part of `tests/test_tabular.py`.
- **Acceptance:** `python -m pytest tests/test_tabular.py -k q_learning` green (including slow).

### T4.4 Tabular runs 🛑

- **Depends on:** T4.1.
- **Files:** `experiments/run_tabular.py`, `experiments/plots.py` (learning curves).
- **CLI** (`--algos` accepts only `q_learning` in the core; O1 adds `sarsa`):
  ```
  --algos q_learning
  --map obstacle_4x4
  --start 0 0                    (or --start random)
  --seeds 0 1 2 3 4
  --episodes 50000
  --alpha 0.5 --gamma 0.99 --epsilon-end 0.05
  --state-mode full|position_only
  --reward dense|sparse|shaped|hackable
  --time-budget-min (optional)
  --checkpoints 0 1000 10000     (Q snapshots for the T8.4 demo)
  --out results/tabular
  --quick                        (episodes / 10, 2 seeds)
  ```
- **Outputs:** `results/tabular/{reward}/{state_mode}/{map}/{algo}/seed{s}_train.csv`, `seed{s}_eval.csv`, `config.json`, `q_table.pkl`; `results/tabular/summary_{map}.csv` (final steps, coverage, gap to optimum).
- **Figure:** `results/figures/tabular_learning_curve_{map}.png`: x = episode; two panels (coverage, steps of greedy eval); mean ± 1 std over seeds; horizontal line at `min_steps` from VI; horizontal reference lines for greedy_bfs and zigzag steps (from T3.2/T3.3).
- **Second figure:** `results/figures/q_convergence_{map}.png` via `plot_q_convergence`: `max_a Q(s0,a)` vs episode (mean ± std) against the horizontal line `V*(s0)` — visual proof that Q-learning learns the true optimal value.
- **Acceptance:** run `--quick` first, then the full run on `empty_3x3` and `obstacle_4x4`; both figures and the summary table produced.
- 🛑 Checkpoint: minimum-submission core is complete; show figure and table.

### T5.1 State explosion (E7)

- **Depends on:** T3.3. **Files:** `experiments/run_state_explosion.py`.
- **Work:** empty grids `(n,m) ∈ {(2,2),(2,3),(3,3),(3,4),(4,4),(4,5)}`: count reachable states (`count_reachable_states` with time limit); for `(5,5)` … `(10,10)` only the upper bound `|F|·2^|F|`. Record runtime and estimated Q-table memory (`n_states × 4 × 8 bytes`).
- **Figure:** `results/figures/state_explosion.png`, log y-axis, two series ("reachable, measured" and "upper bound"), shaded region where the Q-table exceeds 8 GB.
- **Acceptance:** figure + `results/state_explosion.csv`.

### T5.2 Experiment E1

- **Depends on:** T4.1. See section 6, E1.

### T5.3 Tabular on 5×5, 6×6

- **Depends on:** T4.1.
- **Work:** Q-learning on `empty_5x5` and `empty_6x6` with `--time-budget-min 15`; log states visited over time, greedy coverage, approximate Q-table memory.
- **Acceptance:** `results/tabular/scaling.csv` + figure "states visited vs episode" for 3×3, 4×4, 5×5, 6×6.

### T6.1 Network and replay buffer

- **Depends on:** T2.1. **Files:** `deep/networks.py`, `deep/replay_buffer.py`, matching tests.

### T6.2 DQN 🛑

- **Depends on:** T6.1.
- **Files:** `deep/dqn.py`, `experiments/run_deep.py`, tests.
- **CLI `run_deep.py`:**
  ```
  --map obstacle_4x4 | furniture_8x8 | ...
  --random-maps H W DENSITY       (instead of --map: RandomMapVacuumEnv)
  --n-train-maps K                (0 = new map every episode)
  --hide-cleaned                  (E1 deep version)
  --reward dense|sparse|shaped|hackable
  --timesteps 200000
  --seeds 0 1 2 3 4
  --checkpoints 0 50000 200000    (model snapshots for the T8.4 demo)
  --out results/deep
  --quick
  ```
- **Mandatory acceptance ladder (no skipping):**
  1. `empty_3x3`, fixed start, 30k steps → coverage 1.0, steps ≈ 8.
  2. `obstacle_4x4`, random start, 200k steps → compare with mean `min_steps` over starts (from T3.3). Target: coverage ≥ 0.95, mean step gap ≤ 30% over optimum.
  3. `furniture_8x8`, random start, 500k steps → compare with greedy_bfs and zigzag.
  - If rung 1 fails: run section 8, fix bugs (0.4); if still failing, stop and report.
  - Also train the DQN used for the demo and for E3: `RandomMapVacuumEnv(8, 8, 0.15)`, 500k steps, `--checkpoints 0 50000 500000`. This is the model that must handle unseen rooms in T8.4.
- 🛑 Checkpoint after the ladder.

### T7.1–T7.3

See section 6: T7.1 = E2, T7.2 = E4, T7.3 = E3 and E6. Files: `experiments/run_ablation.py`, `experiments/run_generalization.py`, `experiments/plots.py`.

### T7.4 Design log

- Draft B8 rows (version, change, reason, result with numbers) from `CHANGELOG_DESIGN.md`, `config.json` files and summaries; put them in the report for the user to paste into `PLAN_rel.md`.

### T8.2 README and final polish 🛑

- README: setup, tests, **one command per experiment** (table: experiment → command → output files → estimated runtime), main results table, Limitations, "How the team used AI tools" (F2) — the human-facing prose sections in Vietnamese.
- `experiments/reproduce_all.py --quick|--full`: runs every experiment in order, stops on the first failure with a clear message.
- Full review against 0.3 over the whole repo: no stubs, no dead code, no debug prints, all docstrings present, every script has `--help` and `--quick`.
- **Acceptance:** fresh clone into a subfolder of `ROOT` (e.g. `ROOT/.cache/clean_check`), new venv there, `pip install`, `python -m pytest`, `python -m experiments.reproduce_all --quick` all green; then delete that check folder.

### T8.4 Demo — showing the result in simulation 🛑

Goal: presentation material that shows the vacuum robot working, built from the trained models (no retraining here).

- **Depends on:** T8.2 (models from T4.4 and T6.2 exist in `results/`). If DQN is unavailable (minimum submission), every demo uses Q-learning on `obstacle_4x4` instead and says so in the title.
- **Files:** `experiments/make_demo.py`, `tests/test_demo.py`, small additions to `env/render.py` if needed.
- **CLI:**
  ```
  python -m experiments.make_demo --what all|compare|progress|heatmap|unseen
                                  --map furniture_8x8 --start 0 0 --seed 0
                                  --out results/demo [--quick]
  ```
- **Outputs (all in `results/demo/`):**

| File | Content |
|---|---|
| `compare_{map}.gif` (+ `.png` of the final frame) | **Side-by-side race**: random_walk, zigzag, greedy_bfs, DQN (and Q-learning on 4×4) on the same room, same start, frames synchronized by step. Each panel has a header (agent name) and a footer (`bước N · X% · lặp R · đụng C`). An agent that finishes freezes on its last frame with a ✓ in the footer |
| `compare_obstacle_4x4.gif` | Same race on 4×4 including Q-learning and the **VI optimal** path as a fifth panel ("tối ưu") |
| `progress_qlearning.gif`, `progress_dqn.gif` | **"The robot learns"**: the same room played by checkpoints (ep 0 / 1k / 10k / final; step 0 / 50k / final) side by side |
| `heatmap_{map}.png` | Visit-count heatmaps, one subplot per agent, same colour scale; title shows redundancy |
| `unseen_rooms.gif` | DQN (trained on random maps) cleaning 3 random rooms it never saw (`generate_map_set(..., seed=777)`), next to greedy_bfs |
| `demo_summary.md` | Table of the numbers shown in the GIFs (steps, coverage, redundancy) — ready to paste into slides |

- **Also:** add a "Demo" section to README (command + embedded `compare_furniture_8x8.gif`), and add `make_demo --quick` to `reproduce_all.py`.
- **Rules:** GIFs < 5 MB each (subsample frames or lower `cell_px` if needed); fps 6; the deterministic greedy policy is used for all learned agents; if an agent loops until `max_steps`, keep it in the GIF (honest) and note it in `demo_summary.md`.
- **Tests (`tests/test_demo.py`):** `--quick` with tiny freshly-trained models (a few hundred steps/episodes, created inside `tmp_path`) produces every output file; frame synchronization: the composed GIF has `max(steps_i) + 1` frames (after subsampling rule applied consistently).
- **Acceptance:** all files in the table exist; the user opens `compare_furniture_8x8.gif` and `unseen_rooms.gif`. 🛑 Checkpoint: end of core work — ask whether to do O1 (SARSA) and/or O2 (interactive demo).

---

### Optional cards (only after T8.4, and only when the user says go)

Each optional card follows the same workflow (0.5) and quality bar (0.3), extends CLIs/configs/tests in full, and adds its outputs to `reproduce_all.py`, `make_demo.py` (where relevant) and the README.

#### O1 SARSA (= PLAN_rel task 4.2)

- **Files:** `tabular/sarsa.py`, tests (4.10 "added by optional cards"), `sarsa` choice in `run_tabular.py --algos`.
- **Acceptance:** tests green; learning curve figure with Q-learning vs SARSA on `obstacle_4x4` (5 seeds); a SARSA column in the E6 table; 2–3 sentence analysis (on-policy vs off-policy, S&B Example 6.6 "Cliff walking" analogy).

#### O2 Interactive demo window

- **Files:** `demo/play.py`, `tests/test_demo.py` additions (non-GUI logic only).
- **Behavior:** `python -m demo.play --agent dqn --model <path> --size 8 8` opens a matplotlib window (TkAgg backend) showing the room top-down. Left-click toggles furniture on a cell (refused, with a message, if it would disconnect the room); right-click sets the start; buttons: **Play/Pause**, **Step**, **Reset**, **Random room**, agent selector (DQN / greedy_bfs / zigzag / random). A side panel shows step, coverage, revisits.
- **Design:** separate a pure `DemoState` class (map edits, connectivity check, stepping the agent) from the matplotlib view, so `DemoState` is unit-tested without a GUI.
- **Acceptance:** tests for `DemoState` green; the user runs the window and cleans a room they drew themselves.

---

<a id="6"></a>
## 6. Experiment specifications E1–E7

Shared rules: 5 seeds (0..4), report mean ± std, write `config.json`. Fair comparison: same maps, same `max_steps`, same number of evaluation episodes, same training budget (episodes for tabular, env steps for deep). Never pick the best seed.

### E1. State with / without the cleaned map

| Item | Content |
|---|---|
| Hypothesis | Without the cleaned channel the agent can't tell progress apart → Markov property broken → no stable full-coverage policy |
| Tabular | Q-learning on `empty_3x3` and `obstacle_4x4`, fixed start (0,0), `--state-mode full` vs `position_only`, 20k / 50k episodes |
| Deep (optional) | DQN on `obstacle_4x4`, with and without `--hide-cleaned`, 200k steps |
| Command | `python -m experiments.run_tabular --algos q_learning --map obstacle_4x4 --state-mode position_only --seeds 0 1 2 3 4` (and `full`) |
| Figure | `results/figures/e1_markov.png`: greedy coverage vs episode, 2 lines |
| Required analysis | With `position_only`, a deterministic policy is one direction per cell → the robot follows a fixed cycle; show a concrete cycle (GIF `e1_position_only.gif`) |

### E2. Reward design

| Item | Content |
|---|---|
| Configs | (a) `sparse`, (b) `dense`, (c) `shaped` (coef 1.0), (d) `hackable` (bonus 0.5) as a demo |
| Agent | Q-learning on `obstacle_4x4` (required); DQN on `room_6x6` (if time) |
| Command | `python -m experiments.run_ablation --exp reward --map obstacle_4x4 --algo q_learning --seeds 0 1 2 3 4` |
| Metrics | coverage, steps, redundancy of the greedy policy (NOT return, see 4.8) |
| Outputs | `e2_reward_curves.png` (coverage vs episode, 4 lines), `e2_reward_summary.csv` |
| Reward hacking | With `hackable`: expect an away/back loop until `max_steps`; GIF `e2_reward_hacking.gif`; count loop cycles. If it doesn't appear, report honestly and try bonus 1.0 |
| Theory check | VI on `obstacle_4x4` with `dense` and `shaped` (γ = 0.99): count states where `π*` differs; expect ≈ 0 (only ties). Put the count in the table |

### E3. RL vs baselines across obstacle density

| Item | Content |
|---|---|
| Setup | 8×8 (6×6 if compute is short — state it), densities {0, 0.1, 0.2, 0.3}. Per density: test set of 20 maps (`generate_map_set(..., seed=1000+int(100*d))`), identical for all agents |
| Agents | random_walk, zigzag, greedy_bfs, DQN trained on `RandomMapVacuumEnv` at that density, train maps disjoint from test maps |
| Command | `python -m experiments.run_generalization --exp density --size 8 8 --densities 0 0.1 0.2 0.3 --algo dqn --timesteps 500000 --seeds 0 1 2` (3 seeds allowed for deep if time is short; state it) |
| Figure | `e3_density.png`: x = density, y = mean steps (completed episodes) and coverage, one line per agent with error bars |
| Analysis | Include cases where RL loses (Part E, rule 3) |

### E4. Generalization

| Item | Content |
|---|---|
| Setup | 6×6, density 0.15. Train DQN on K ∈ {1, 10, 100, ∞} maps (`--n-train-maps`, ∞ = 0) with the same step budget. Test on 50 unseen maps |
| Metrics | coverage/steps on train set and test set, train–test gap |
| Outputs | `e4_generalization.csv`, `e4_generalization.png` |

### E5. Dropped

E5 (Q-learning vs SARSA vs Double Q, maximization bias) from `PLAN_rel.md` is dropped (1.1). The Q-learning vs SARSA comparison lives in optional card O1. Numbering E6/E7 is kept to stay aligned with `PLAN_rel.md`.

### E6. Gap to optimum

| Item | Content |
|---|---|
| Table | Rows: map × start (3×3: 3 representative starts; 4×4: 5 starts). Columns: `min_steps`, VI, Q-learning, DQN, greedy_bfs, zigzag (+ SARSA if O1 is done). Cells: steps (mean ± std over seeds) and `gap% = (steps − min_steps) / min_steps` |
| Outputs | `e6_optimality_gap.csv` + markdown version for the slides |

### E7. State explosion

Task T5.1.

---

<a id="7"></a>
## 7. Result file and plotting conventions

### 7.1 Results tree

```
results/
├── baselines/{map}/{agent}/seed{s}.csv, summary.csv, config.json
├── vi/{map}_summary.csv, {map}_meta.json, {map}_vi.pkl
├── tabular/{reward}/{state_mode}/{map}/{algo}/seed{s}_train.csv, seed{s}_eval.csv, config.json, q_table.pkl
├── deep/{algo}/{env_tag}/seed{s}/train_episodes.csv, train_losses.csv, eval.csv, config.json, model.pt
├── ablation/..., generalization/...
├── figures/*.png      (dpi 150, 8×4.5 in, font ≥ 11 so it reads on slides)
└── gifs/*.gif         (each < 5 MB; drop frames if needed)
```

`env_tag` examples: `furniture_8x8`, `rand6x6_d0.15_K10`, `rand8x8_d0.2_K0`.

### 7.2 Plotting API (`experiments/plots.py`)

```python
AGENT_COLORS: dict[str, str]   # fixed colour per agent across all figures

def load_runs(pattern: str) -> pd.DataFrame: ...   # glob seeds, add 'seed' column
def plot_learning_curve(df, x: str, y: str, hue: str, out_path, smooth: int = 1,
                        hline: float | None = None, hline_label: str = "") -> None:
    """Mean ± 1 std over seeds (fill_between alpha 0.2); smooth = moving-average window."""
def plot_bar_with_error(summary_df, ...): ...
def plot_state_explosion(df, out_path): ...
def plot_q_convergence(train_logs, v_star, out_path): ...  # q_start_max vs episode, line at V*(s0)
def plot_visit_heatmap(visits, grid, out_path): ...
def df_to_markdown_table(df, float_fmt=".2f") -> str: ...
```

Every figure: Vietnamese title, axis labels with units, legend with algorithm names, subtitle with seed count (e.g. "trung bình ± std, 5 seed"). Use the `Agg` backend (no GUI windows).

---

<a id="8"></a>
## 8. RL debugging checklist

When an agent doesn't learn (flat curve), check in this order (most likely first). Diagnose before changing hyperparameters:

1. **Environment:** does `greedy_bfs` reach coverage 1.0 on the same env? Are the first 10 rewards exactly as in B2?
2. **Terminal mask:** targets use `terminated`, not `terminated or truncated`; the buffer stores the right flag.
3. **Observation aliasing:** are stored obs copies? (`test_obs_not_aliased`, buffer mutation test)
4. **Target network:** targets computed with `target_net` under `torch.no_grad()`; `target_update_interval` neither tiny (< 100) nor huge.
5. **ε schedule:** does ε actually decay? Log `epsilon`. `learning_starts` < total steps.
6. **Reward scale:** `+10` completion vs ±1 step rewards; try `reward_scale=0.1` or confirm Huber loss.
7. **Grid too big for the budget:** go back to 3×3; if 3×3 fails too, it's a code bug, not a hard problem.
8. **Compare with tabular Q-learning on the same 3×3/4×4 room:** both use the same target; if tabular Q-learning reaches `min_steps` and DQN doesn't, the bug is in the DQN-specific parts (network input, buffer, target network, loss), not in the environment or reward.
9. **Greedy loops:** a deterministic greedy DQN can cycle forever; `steps == max_steps` is the symptom. Report it honestly; optionally add an ε = 0.05 evaluation as a secondary number.

Bugs found here are fixed immediately (0.4) with a regression test.

---

<a id="9"></a>
## 9. Report template

```markdown
### Report T<x.y>: <title>

**Done**
- <file>: <one-line summary>

**Checks run**
- `python -m pytest -m "not slow"` → <N passed, time>
- `<script command>` → <key result, output path>

**Key numbers** (if any)
| ... |

**Textbook mapping**
- <algorithm> follows Sutton & Barto section <x.y>; variants (if any): <description + reason>

**Fixes to earlier work** (if any)
- <bug> in <file>: <root cause> → <fix>, regression test <name>

**Known limitations / not done**
- ...

**Questions for the user** (if any)
- ...

**Proposed B8 row** (if the design changed)
| vN | <date> | <change> | <reason> | <result> |
```

### Example prompts the user can give

> "Read `PLAN_rel.md` and `IMPLEMENTATION_PLAN.md`. Start at T1.2 and continue in order until the next 🛑 checkpoint."

> "Do card **T2.1a** only. Report using section 9."

> "DQN failed rung 1 of T6.2. Go through section 8 and report each check's result; fix any bug you confirm."

---

<a id="10"></a>
## 10. Appendix: hand-verified reference values

Use these as ground truth in tests. Reward `dense`, γ = 0.99.

**10.1 Empty 3×3, start (0,0)**

- An 8-step Hamiltonian path exists (e.g. snake `[3,3,1,2,2,1,3,3]`), each step entering a new cell.
- Each step: `1 − 0.01 = 0.99`; the last step adds `10`.
- Undiscounted return: `8 × 0.99 + 10 = 17.92`.
- Discounted return: `Σ_{t=0}^{7} 0.99^t × 0.99 + 0.99^7 × 10 = 16.968928730705...`
- This is optimal (`V*`): no step can earn more than 0.99, and the bonus cannot arrive before step 8 (8 new cells need ≥ 8 steps). Every Hamiltonian path achieves exactly this.

**10.2 Minimum steps on 3×3 (checkerboard argument)**

- Colour `(r,c)` "even" if `r+c` is even: 5 even cells (4 corners + centre), 4 odd. Every move flips colour.
- A 9-cell alternating path must start and end on the colour with 5 cells, i.e. even.
- Start at a corner (0,0) or the centre (1,1): even → **8 steps**.
- Start at an edge middle, e.g. (0,1): odd → no Hamiltonian path → **9 steps** (one revisit).

**10.3 State-space size (upper bound `|F| · 2^|F|`)**

| Map | `|F|` | Upper bound |
|---|---|---|
| empty_3x3 | 9 | 4,608 |
| obstacle_4x4 | 13 | 106,496 |
| empty_4x4 | 16 | 1,048,576 |
| empty_5x5 | 25 | ≈ 8.4 × 10⁸ |
| room_6x6 | 29 | ≈ 1.6 × 10¹⁰ |
| empty_6x6 | 36 | ≈ 2.5 × 10¹² |
| furniture_8x8 | 52 | ≈ 2.3 × 10¹⁷ |

The number of **reachable** states is smaller (the robot always sits inside the cleaned set, which is connected); T5.1 measures it.

---

*End. Every spec change made while implementing must be recorded in `CHANGELOG_DESIGN.md` (and proposed as a B8 row).*
