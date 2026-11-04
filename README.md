# Dual-System VLA on LIBERO

A Dual-System policy for the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) benchmark, built on [LeRobot](https://github.com/huggingface/lerobot). It answers the brief in [CHALLENGE.md](CHALLENGE.md).

- **System 2** (slow, semantic): a deterministic supervisor. It plans atomic commands once and decides when to switch to the next one. It uses robot signals (gripper width, hand height) and a frozen Qwen3-VL 2B for subtasks no signal shows.
- **System 1** (fast, reactive): a frozen pi0.5 (`lerobot/pi05_libero_finetuned_v044`). It turns the current command and the observation into a 7-D action at every simulation step.
- **Interface:** one English sentence, written into System 1's prompt. No latent embedding, so every step is readable and easy to ablate.

**Main finding:** System 2 switches commands about as well as an oracle and detects every forced drop, but it does **not** improve System 1. Execution under atomic commands is the bottleneck, not reasoning. Details below.

## Deliverables

| Requirement | Where |
|---|---|
| Codebase (Dual-System, training loop, LeRobot evaluation) | [src/](src/): [dual_system/](src/dual_system/) (policy, evaluation, finetuning), [infra/](src/infra/) (Colab) |
| README setup guide | [Setup](#setup), [Data](#data), [Running the evaluation](#running-the-evaluation), [Training loop](#training-loop-finetuning) |
| Rollout videos with System 2 output | [Rollout videos](#rollout-videos) |
| Research report, max 3 pages | [report.pdf](report.pdf) (3 pages). Source: [docs/report.tex](docs/report.tex), build in `docs/` with `pdflatex report.tex`, twice |
| Ablation study | [Ablation results](#ablation-results), full log in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) |

## Challenge questions

| Question | Answer | Evidence |
|---|---|---|
| **Counterfactual:** System 2 removed, frozen or naive? | Removing it does not hurt: 9/20 without System 2 against 4/20 with it on `libero_10`. Freezing it hurts on `libero_goal` (1/4). No difference is significant (smallest p = 0.18) | runs 13, 17 |
| **Error recovery:** does System 2 help after a perturbation? | It detects all 8 forced drops within 1 to 3 steps of the simulator. Success is the same with and without recovery (6/8 each). pi0.5 re-picks the bowl by itself | run 16 |
| **Bottleneck:** reasoning or execution? | Execution. 13 of fused's 16 `libero_10` failures also fail under oracle switching, so no better progress check would have saved them | run 17 |

Counterfactual conditions: `passthrough` (System 2 removed, whole instruction), `frozen` (first atomic command for the whole episode), `fused` (final System 2), `oracle` (the simulator decides when a phase is done, an upper bound for any progress check). Not run: shuffled commands and a gibberish System 2 inside the dual system for pi0.5. Gibberish was tested for System 1 alone (run 10).

## Architecture

```text
instruction ──► System 2 ──────────────────────────────────────────────┐
                 decompose once: plan of atomic commands                │ one English
                 per step: gripper width, hand height ──► phase done?  │ sentence
                 act phases only: Qwen3-VL 2B, P("done") ≥ 0.8 twice    │
                 timeouts, empty-grasp and drop recovery                ▼
camera + proprioception ───────────────────────────────► System 1: pi0.5 ──► 7-D action, 20 Hz ──► LIBERO
```

- **Plan:** atomic commands come from the subtask segmentation of the LIBERO demos ([finetune/segment.py](src/dual_system/finetune/segment.py)). Qwen's own decomposition is logged next to it but not used.
- **Pick done:** the gripper holds something for 8 steps and the hand rose 3 cm.
- **Place done:** the gripper reopens after a hold. With drop detection on, the hand must also have come down at least 2 cm while holding.
- **Act done** (open, close, turn on): Qwen3-VL 2B is asked a yes/no question every 10 steps. Its probability of "yes" must reach 0.8 twice in a row.
- **Safety:** every phase times out after 150 steps, an empty grasp triggers "open the gripper" and a retry, and System 2 never ends an episode.
- **Queue:** when the command changes, System 1's queued actions are discarded.

Code: [fused.py](src/dual_system/system2/fused.py) (final System 2), [eval/final.py](src/dual_system/eval/final.py) (ablation runner), [system1/base.py](src/dual_system/system1/base.py) (System 1 interface, so another policy can be swapped in).

**How the design developed.** The generic design (frozen VLM and VLA joined by text) stayed fixed. Only when System 2 runs and who judges progress changed:

| Version | System 2 runs | Progress judged by | Outcome |
|---|---|---|---|
| v1: SmolVLA | Every 20 steps plus environment triggers | Qwen3-VL 2B: running, done or failed. "Success" ends the episode | 19/36 alone, 4/36 dual |
| v2: pi0.5 | Same | Same | 8/12 alone, 3/12 dual |
| v3: phase-aware prompt | Triggers, 150-step timeout | VLM told phase and "done" criteria | Never declared a step done in 76 checks |
| **v4: final** | Every step on robot signals, VLM every 10 steps for act phases | Gripper width and hand height, VLM only for "turn on", "open", "close" | Matches oracle switching |

The earlier VLM-judge conditions (`full_free_atomic`, `full_phase_atomic`, `full`, `full_scripted`) are still in [configs/conditions/](src/dual_system/configs/conditions/) and run through [eval/grid.py](src/dual_system/eval/grid.py).

## Setup

All results were produced on **Google Colab with a T4 or L4 GPU**, driven from the laptop through the [Colab MCP server](https://github.com/googlecolab/colab-mcp). There is no local GPU. LIBERO needs Linux, so a local run needs WSL2 or Linux.

Set the runtime to a GPU, then run in notebook cells:

```bash
!git clone https://github.com/tkpf/dual-system-architecture-vla.git /content/repo
!pip install -q -e /content/repo
!python -m infra start install "python -m infra install"   # background job, about 5 minutes
!python -m infra log install                               # finished when this shows the lerobot-eval path
!python -m infra egl                                       # NVIDIA EGL libraries for rendering
!python -m infra check                                     # prints the GPU name
```

pi0 and pi0.5 use a gated tokenizer. Accept the licence at [google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224), add a read token as the Colab secret `HF_TOKEN`, and set it before starting jobs:

```python
import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
```

- **Versions:** LeRobot 0.6.1, transformers 5.5.4, torch 2.11.0, robosuite 1.4.0, MuJoCo 3.8.1. The installer does not pin them. Each run writes its versions and git commit to `run_info_w<N>.json`.
- **Memory:** pi0.5 plus Qwen3-VL 2B needs about 14 GB GPU memory and fits a T4. A standard T4 runtime has room for one worker.
- **Local tests** (WSL2, Ubuntu 22.04, LeRobot in a conda env `lerobot-eval`): `wsl -d Ubuntu-22.04 -- bash scripts/wsl_test.sh python -m pytest -q tests`

## Data

No dataset download is needed to evaluate. The simulator and the LIBERO tasks come from LeRobot's LIBERO pipeline and download on first use. The evaluation slice is:

| Suite | Tasks used | Why |
|---|---|---|
| `libero_goal` | 1 (bowl on stove), 4 (bowl on cabinet) | Easy scenes: sanity check, and the drop test |
| `libero_10` | task 2 (stove, then moka pot) in the small run; all 10 in the main ablation | Long-horizon, two-step tasks |

The three "seen" scenes are the default of `eval.final --tasks 0 1 2` (indices into `atomic.SEEN`). `--suite-tasks libero_10:<id>` selects any other task. Older SmolVLA grids also used `libero_goal` task 0 and `libero_10` tasks 0 and 3.

Start states: `A` is the stored LIBERO start. `C` and `B` add a joint offset that moves the gripper about 10 cm and 19 cm, to test robustness. A pair of episodes shares task, state and seed, so it differs only in the condition.

Datasets are used only for the subtask labels and the finetuning experiments:

- [`lerobot/libero`](https://huggingface.co/datasets/lerobot/libero): demos for finetuning and labelling. Only `data/*` and `meta/*` are fetched by [segment.py](src/dual_system/finetune/segment.py).
- [`yifengzhu-hf/LIBERO-datasets`](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets): the raw demos. They are matched to `lerobot/libero` by actions and replayed to find where each subtask ends.
- **Dataset slice** `boundaries.jsonl` ([finetune/boundaries.jsonl](src/dual_system/finetune/boundaries.jsonl)): 1,492 of 1,693 episodes with subtask boundaries (pick, place, act). It is committed, so the slice does not have to be rebuilt. To rebuild: `python -m dual_system.finetune.segment run --out <res>/boundaries.jsonl`.

## Running the evaluation

Models and tasks download on first use. The commands assume the Colab setup. Each run gets its own folder:

```bash
RUN = !python -m infra newrun
RUN = RUN[0]
```

**Reproduce the ablation** (conditions `passthrough`, `frozen`, `fused`, `oracle`, same scene, start and seed). About 40 s per episode on an L4:

```bash
# the 3 seen scenes (2 libero_goal, 1 libero_10), starts C and B (run 13)
!python -m infra start final "python -m dual_system.eval.final run --res {RUN}/final --seeds 2 --states C B"

# all 10 libero_10 scenes, start B, 2 seeds (run 17, 80 episodes)
!python -m infra start l10 "python -m dual_system.eval.final run --res {RUN}/libero10 --seeds 2 --states B --suite-tasks libero_10:0 libero_10:1 libero_10:2 libero_10:3 libero_10:4 libero_10:5 libero_10:6 libero_10:7 libero_10:8 libero_10:9"

# error recovery: forced drop mid-carry (run 16)
!python -m infra start drop "python -m dual_system.eval.final run --res {RUN}/drop --tasks 0 1 --states C --seeds 4 --conditions passthrough fused fused_norec oracle --perturb drop"

# tables and 2x2 videos (passthrough, frozen / fused, oracle)
!python -m dual_system.eval.final report {RUN}/libero10
!python -m dual_system.eval.final video {RUN}/libero10      # writes side_by_side/
!python -m infra status                                    # jobs, finished episodes, GPU and RAM
```

Finished cells are skipped when a folder is reused after a restart. Downloaded results land in `data/YYYYMMDD-HHMM` (ignored by git). Each episode writes its outcome, every command and how it ended, every System 2 call with its raw answer, and an `episode.mp4`.

**Other experiments** (all in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)):

```bash
# does System 1 follow a command? No System 2, compared with a gibberish control (run 2)
!python -m dual_system.eval.command_probe --res {RUN}/probe
# atomic commands at fixed steps, two models side by side (run 3)
!python -m dual_system.eval.command_probe --type pi05 --path lerobot/pi05_libero_finetuned_v044 --kinds schedule --only-first --seeds 2 --states C B --video --res {RUN}/pi05
!python -m dual_system.eval.command_probe --type smolvla --path lerobot/smolvla_libero_plus --kinds schedule --only-first --seeds 2 --states C B --video --res {RUN}/smolvla
!python -m dual_system.eval.compare_probe {RUN}/pi05 {RUN}/smolvla --out {RUN}/side_by_side
# older VLM-judge System 2 against System 1 alone
!python -m dual_system.eval.grid pi05_atomic_test --res {RUN} --stop-after-min 40
# can a VLM see what System 2 has to judge? (runs 6, 11, 12)
!python -m dual_system.eval.perception collect --res {RUN}
!python -m dual_system.eval.perception run qwen2b --res {RUN}
```

## Training loop (finetuning)

The final system is frozen, so it needs no training. The training code is the finetuning that was tried and did not help (runs 9 and 10). It lives on the `finetune` branch and in [src/dual_system/finetune/](src/dual_system/finetune/):

```bash
# System 1: train the action expert on subtask-labelled demos (SmolVLA or pi0.5), then export a loadable folder
!python -m dual_system.finetune.system1 train --res {RUN} --boundaries src/dual_system/finetune/boundaries.jsonl --type pi05 --steps 400
!python -m dual_system.finetune.system1 export --res {RUN} --step 400 --out {RUN}/export --type pi05

# System 2: LoRA SmolVLM as a judge of finished subtasks (written and unit-tested, not run)
!python -m dual_system.finetune.collect_judge --path <checkpoint> --res {RUN}
!python -m dual_system.finetune.system2 train --data {RUN} --res {RUN}
```

Training resumes from the newest checkpoint. Held-out tasks and 10% of demos per task are never trained on. Result: validation loss fell, but command following did not improve (pi0.5: 17/28 goals against 21/28 for stock).

## Rollout videos

2×2 panels: passthrough and frozen on top, fused and oracle below. Each panel shows the command System 2 sends. In the drop video the frozen panel is empty because frozen was not run.

- [libero_goal_bowl_on_stove_C.mp4](docs/videos/libero_goal_bowl_on_stove_C.mp4): an easy scene. Frozen holds the bowl and never places it.
- [libero_10_stove_moka_B.mp4](docs/videos/libero_10_stove_moka_B.mp4): turn on the stove, then the moka pot. Atomic commands succeed where the whole instruction fails.
- [drop_bowl_on_stove_C.mp4](docs/videos/drop_bowl_on_stove_C.mp4): the bowl is forced out of the gripper mid-carry. Fused notices it and commands a new pick, and oracle recovers after three re-picks.

All episodes of the two ablation runs (run 17 on `libero_10`, run 13 on `libero_goal`) are in [docs/videos/INDEX.md](docs/videos/INDEX.md), with the outcome of every condition. New videos: `eval.final video <res>`.

## Ablation results

**All 10 `libero_10` scenes, start B, two seeds (run 17, 80 episodes):**

| Condition | Success |
|---|---|
| passthrough (System 2 removed) | 9/20 |
| frozen (first atomic command only) | 7/20 |
| fused System 2 | 4/20 |
| oracle switching | 5/20 |

- **Atomic commands cost success with pi0.5,** even when the simulator switches them. pi0.5 was trained on whole instructions.
- **Fused is about as good as oracle:** the same outcome in 15 of 20 pairs.
- **Atomic commands make outcomes less repeatable:** both seeds agree in 9/10 scenes with the whole instruction, in 5 to 6 of 10 with atomic commands.
- **No pairwise difference is significant** (passthrough against fused: 7 against 2 pairs, p = 0.18).

**`libero_goal`, 4 scenes (run 13):** passthrough 4/4, frozen 1/4, fused 4/4, oracle 4/4. These scenes are easy for pi0.5 alone, so they cannot show System 2 helping.

**Error recovery, forced drop, 2 scenes × 4 seeds (run 16):**

| passthrough | fused | fused without recovery | oracle |
|---|---|---|---|
| 5/8 | 6/8 | 6/8 | 5/8 |

System 2 detects all 8 drops, but outcomes are identical with and without recovery in every episode.

**Bottleneck (run 17):** 13 of fused's 16 failures also fail under oracle switching (System 1 execution or the atomic interface). 3 succeed under oracle (System 2 or noise). Remaining System 2 weaknesses: the gripper signal shows that something is held, not what, and Qwen's "close" checks are lenient.

## Limitations

- **Small samples:** 20 scene-seed pairs, 8 drop episodes, one LIBERO init state. Single-scene differences are mostly noise.
- **Tuning on test starts:** the gripper thresholds were set after a failed smoke test on start C.
- **Not run:** shuffled commands and a gibberish System 2 for pi0.5 in the dual system. The prompt that keeps the whole instruction and appends the current step (`oracle_ctx`, `fused_ctx` in `eval.final`) is implemented but was not run. It is the most direct test of whether System 2's output can help pi0.5 at all.
- **Object identity:** the gripper signal cannot tell which object is held. A wrist-camera check is the next step.
- **Not comparable with published LIBERO numbers:** the protocol differs (shifted starts, step budgets, subset of tasks).

## Repository layout

```text
src/dual_system/
  system1/      System 1 interface and LeRobot wrapper (SmolVLA, pi0, pi0.5)
  system2/      fused (final), supervisor, phase supervisor, reasoners (Qwen, scripted), passthrough
  knowledge/    command profiles: free_atomic, closed_set, phase_atomic
  configs/      experiment grids and conditions
  eval/         final ablation runner, grid runner, report, side-by-side videos, command probe, perception benchmark, demo judge, subtask evaluation
  finetune/     subtask labels from demos, splits, System 1 training, judge data and SmolVLM LoRA training
  env.py, rollout.py, triggers.py, robot_states.py, tracing.py, types.py
src/infra/      Colab install, EGL rendering, background jobs, result sync
scripts/        WSL test runner, robot start-state tuning, local unpacking of finetuning downloads
tests/          supervisor, trigger, rollout, perception and finetuning tests
docs/           report, experiment log, research notes, videos
```

Documents:

- [docs/report.tex](docs/report.tex): the submitted report (max 3 pages).
- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md): every run, what it showed and what it does not show.
- [docs/REPORT.md](docs/REPORT.md): the longer first draft of the report.
- [docs/RESEARCH.md](docs/RESEARCH.md): literature behind the design (Hi Robot is the role model).
- [docs/PREREQUISITES.md](docs/PREREQUISITES.md): hardware, time and experience constraints.
