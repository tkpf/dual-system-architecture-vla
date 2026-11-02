# Dual-System VLA on LIBERO

A two-level robot policy for the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) manipulation benchmark, built on [LeRobot](https://github.com/huggingface/lerobot). The challenge brief is in [CHALLENGE.md](CHALLENGE.md).

- **System 2** (slow): a frozen Qwen3-VL-2B vision-language model inside a rule-based supervisor. It splits the instruction into atomic commands and judges progress from camera images.
- **System 1** (fast): a frozen pi0.5 policy (`lerobot/pi05_libero_finetuned_v044`) that turns the current command and the observation into robot actions. SmolVLA is still supported and was the first System 1.

Both models are used as pretrained. Finetuning is in progress on the `finetune` branch (`src/dual_system/finetune/`): SmolVLA on subtasks cut from LIBERO demos, and SmolVLM as a LoRA judge of finished subtasks. A first SmolVLA run did not make it follow language better (run 9 in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)). Evaluation runs live in the simulator.

> **Status: work in progress.** The final System 2 (`fused`, below) matched oracle command switching in a first 16-episode test. It still has not beaten System 1 alone, because pi0.5 alone also succeeded on those scenes. Earlier VLM-only System 2 variants performed **worse** than System 1 alone. The ablation study is incomplete. See [Results so far](#results-so-far).

## Architecture

```text
 instruction ──► System 2 (runs when a trigger fires)
                 ├─ gate       is the instruction sensible?
                 ├─ decompose  split it into atomic commands
                 ├─ check      running / done / failed?  (2 agreeing answers needed)
                 ├─ recover    "open the gripper", "move back to the home position"
                 └─ judge      end the episode with a success or failure verdict
                      │
                      │ one English sentence
                      ▼
 camera + state ─► System 1: pi0.5 ──► 7-D action at every sim step ──► LIBERO
```

**Interface.** System 2 passes plain text, written into the prompt that System 1 reads. When the command changes, System 1's queued actions are discarded. Text is readable in videos and easy to freeze, swap or corrupt for an ablation.

**When System 2 runs.** Triggers call it every 20 steps, and also on a gripper toggle, when the robot stops moving, on an empty grasp, and after a time limit on one command (300 steps, or 150 in the phase-aware condition). They are set per condition, for example in [full_free_atomic.yaml](src/dual_system/configs/conditions/full_free_atomic.yaml).

**Swappable parts.** The supervisor logic in [supervisor.py](src/dual_system/system2/supervisor.py) is deterministic. All perception goes through a `Reasoner` ([reasoners.py](src/dual_system/system2/reasoners.py)), so a different VLM changes the reasoning quality and nothing else. System 1 sits behind a small interface ([base.py](src/dual_system/system1/base.py)).

| Condition | What it does |
|---|---|
| `passthrough` | No System 2: the full instruction goes to System 1 unchanged. This is the baseline |
| `full_free_atomic` | System 2 writes its own atomic commands and answers a one-letter progress check |
| `full_phase_atomic` | System 2 knows all phases, its position and what a finished phase looks like, reasons briefly, and ends with the current subtask |
| `full` | System 2 chooses commands from the candidate list in the `closed_set` profile |
| `full_scripted` | Fixed answers instead of a VLM, for pipeline checks |

**The final System 2** ([fused.py](src/dual_system/system2/fused.py), run by [eval/final.py](src/dual_system/eval/final.py)) replaces the VLM progress check, which runs 11 and 12 showed to be unreliable. It switches atomic commands on signals the robot has: a pick is done when the gripper holds something and the hand rises, a place when the gripper reopens after a hold. Qwen 2B is asked only about subtasks no robot signal shows, such as "turn on the stove", and needs two confident answers in a row. Every phase has a timeout, an empty grasp triggers "open the gripper" and a retry, and System 2 never ends an episode. `eval.final` compares it with `passthrough`, `frozen` (the first atomic command for the whole episode) and `oracle` (the simulator decides when a phase is done) on the same scene, start and seed.

## Setup

### Google Colab (T4 GPU), used for all results below

Set the runtime to T4 GPU, then run these in notebook cells:

```bash
!git clone https://github.com/tkpf/dual-system-architecture-vla.git /content/repo
!pip install -q -e /content/repo
!python -m infra start install "python -m infra install"   # background job, about 5 minutes
!python -m infra log install                               # the install is finished when this shows the lerobot-eval path
!python -m infra egl                                       # NVIDIA EGL libraries for rendering
!python -m infra check                                     # must print "Tesla T4"
```

pi0 and pi0.5 use a gated tokenizer. Accept the licence at [google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224), add a read token as the Colab secret `HF_TOKEN`, and set it in the notebook before starting jobs:

```python
import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
```

Results were produced with LeRobot 0.6.1, transformers 5.5.4, torch 2.11.0, robosuite 1.4.0 and MuJoCo 3.8.1. The installer does not pin these versions, so a newer LeRobot may behave differently. Each run writes its exact versions and git commit to `run_info_w<N>.json`.

A standard T4 runtime (2 vCPUs, 12 GB RAM) has room for one worker only.

### Local tests (WSL2, Ubuntu 22.04)

LIBERO needs Linux. With LeRobot installed in a conda env named `lerobot-eval`:

```bash
wsl -d Ubuntu-22.04 -- bash scripts/wsl_test.sh python -m pytest -q tests
```

## Running the evaluation

Models and tasks download on first use. All commands below assume the Colab setup.

```bash
# Every run gets its own folder, /content/runs/YYYYMMDD-HHMM (UTC), so the name that sorts last is the most recent.
RUN = !python -m infra newrun
RUN = RUN[0]

# Start the grid in the background. Finished cells are skipped when the same folder is reused after a restart.
!python -m infra start w0 "python -m dual_system.eval.grid pi05_atomic_test --res {RUN} --stop-after-min 40"
!python -m infra status          # one line of JSON: jobs, finished episodes, GPU and RAM

# Final comparison: passthrough, frozen, fused System 2 and oracle; 3 tasks x starts C,B x 2 seeds = 48 episodes
!python -m infra start final "python -m dual_system.eval.final run --res {RUN}/final --seeds 2 --states C B"
!python -m dual_system.eval.final report {RUN}/final
!python -m dual_system.eval.final video {RUN}/final      # 2x2 videos in {RUN}/final/side_by_side

# Tables, sub-goal CSV and side-by-side videos
!python -m dual_system.eval.report {RUN}
!python -m dual_system.eval.lookup {RUN} --task 0 --state B
!python -m dual_system.eval.side_by_side {RUN}             # writes {RUN}/videos

# Does System 1 follow a command? No System 2, compared with a gibberish control
!python -m infra start probe "python -m dual_system.eval.command_probe --res {RUN}/probe"

# Atomic commands at fixed steps, recorded, for two models, stitched side by side
!python -m dual_system.eval.command_probe --type pi05 --path lerobot/pi05_libero_finetuned_v044 --kinds schedule --only-first --seeds 2 --states C B --video --res {RUN}/pi05
!python -m dual_system.eval.command_probe --type smolvla --path lerobot/smolvla_libero_plus --kinds schedule --only-first --seeds 2 --states C B --video --res {RUN}/smolvla
!python -m dual_system.eval.compare_probe {RUN}/pi05 {RUN}/smolvla --out {RUN}/side_by_side

# Can a VLM see what System 2 has to judge? Yes/no questions scored against simulator truth, no robot control
!python -m dual_system.eval.perception collect --res {RUN}
!python -m dual_system.eval.perception run qwen2b --res {RUN}   # also smolvlm, qwen4b, qwen8b (8b needs bitsandbytes)
!python -m dual_system.eval.perception report --res {RUN}
```

Downloaded results land in `data/YYYYMMDD-HHMM`, using the local download time, so the name that sorts last is the most recent.

| Config | Purpose |
|---|---|
| `pi05_atomic_test` | pi0.5, `passthrough` vs `full_free_atomic`, 3 tasks, shifted starts C and B, 2 seeds: 24 episodes |
| `pi05_phase_test` | pi0.5, `passthrough` vs `full_phase_atomic`, 3 tasks, state A, 2 seeds, 300-step limit: 12 episodes |
| `free_atomic_test` | SmolVLA, `passthrough` vs `full_free_atomic`, 6 tasks, 3 robot start states, 2 seeds: 72 episodes |
| `profiles_test` | Compares the `closed_set` and `free_atomic` command profiles |
| `libero_plus_checkpoint` | SmolVLA alone, with the checkpoint trained on `lerobot/libero_plus` |
| `first_test` | The first dual-system run, kept for reference |
| `smoke` | Short check with real SmolVLA and no VLM. Runs on CPU |

Robot start states: `A` is the stored LIBERO start. `C` and `B` add a joint offset that moves the gripper about 10 cm and 19 cm. A pair of episodes shares its task, state and seed, so the pair differs only in the condition.

Each cell folder holds `episodes.jsonl` (outcome), `subgoals.jsonl` (each command and how it ended), `trace.jsonl` (every System 2 call with its raw answer), `steps.jsonl`, keyframes and `episode.mp4`.

## Results so far

Full run details are in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

**Final dual system (run 13, 16 episodes):** the 3 tasks from start C and bowl-on-stove from start B, stock pi0.5 as System 1.

| Condition | Success | Median steps |
|---|---|---|
| passthrough (System 1 alone) | 4/4 | 99 |
| frozen (first atomic command only) | 1/4 | 218 |
| fused System 2 | 4/4 | 101 |
| oracle switching | 4/4 | 102 |

Fused switches came 0 to 13 steps after simulator truth, and none was wrong. Atomic commands need a working switcher (frozen 1/4), but these scenes are easy for pi0.5 alone, so they cannot show System 2 helping. Side-by-side videos are made with `python -m dual_system.eval.final video <res>`.

**On `libero_10` scenes where pi0.5 alone failed (run 14):** 4 of the 5 scenes that passthrough failed from start B, out of 10 screened.

| Condition | Success |
|---|---|
| passthrough | 0/4 (chosen for failing) |
| frozen | 2/4 |
| fused System 2 | 2/4 |
| oracle switching | 2/4 |

All three System 2 conditions rescued the same two scenes, frozen included, so progress tracking is not what helped. The other two scenes failed even with oracle switching: the bottleneck there is System 1 execution.

Earlier comparisons:

| Comparison | System 1 alone, or first model | Dual, or second model |
|---|---|---|
| pi0.5 vs SmolVLA on a fixed atomic schedule, shifted starts | pi0.5 5/12 | SmolVLA 0/12 |
| pi0.5 alone vs dual (`full_free_atomic`), shifted starts | 8/12 | 3/12 |
| pi0.5 alone vs dual (`full_phase_atomic`), stored start, 300 steps | 6/6 | 2/6 |
| SmolVLA alone vs dual (`full_free_atomic`), 3 start states | 19/36 | 4/36 |
| SmolVLA finetuned 600 steps on subtasks, seen scenes, real vs gibberish commands (1 episode each) | 2/3 goals | 1/3 goals |

pi0.5 is the stronger System 1, but no dual variant has beaten System 1 alone overall. The phase-aware System 2 removed false successes but never declared a step finished, in 76 checks. Comparing its answers with camera frames shows why: Qwen 2B calls a glowing red burner dark and a held bowl on the table. A partial offline benchmark scored Qwen 2B at 0.72 balanced accuracy on such yes/no questions, where 0.5 is chance.

Finetuning SmolVLA's action expert on subtasks, with its VLM frozen, fitted the data but made it react less to the command than before (run 9). On the seen scenes it finished more subtasks than stock SmolVLA, but about as many with gibberish.

### SmolVLA in detail

From the full `free_atomic_test` grid at commit `c7452e6`: 72 episodes over `libero_goal` tasks 0, 1 and 4 and `libero_10` tasks 0, 2 and 3, with 2 seeds per cell.

| Start state | SmolVLA alone | Dual system |
|---|---|---|
| A (stored start) | 12/12 | 2/12 |
| C (about 10 cm shift) | 5/12 | 2/12 |
| B (about 19 cm shift) | 2/12 | 0/12 |
| **Total** | **19/36** | **4/36** |

Over `libero_goal`, SmolVLA alone reached 12/18 and the dual system 4/18. Over `libero_10`, the figures are 7/18 and 0/18. Of the 36 paired episodes, 15 succeeded only with SmolVLA alone and none only with the dual system.

The dual system reduces success on these tasks. Nothing here shows that System 2 helps. Two separate causes show up:

**System 1 follows language only weakly.** In the command probe, with no System 2 involved:

| Scene | Command | Result (2 seeds, 200 steps) |
|---|---|---|
| `libero_goal` task 1 | "put the bowl on the stove" | Goal reached, 2/2 |
| `libero_goal` task 1 | "pick up the bowl" | Bowl lifted and held, 2/2. This is the correct behaviour |
| `libero_goal` task 1 | gibberish | Bowl also lifted, 2/2 |
| `libero_10` task 2 | "turn on the stove" | Stove not turned on, 0/2 |
| `libero_10` task 2 | "pick up the moka pot" | Stove turned on instead, pot not lifted, 2/2 |
| `libero_10` task 2 | gibberish | Stove also turned on, 2/2 |

A gibberish prompt can produce the same behaviour as a real command, so the scene drives much of what SmolVLA does. "turn on the stove" did succeed in an earlier single run in the scene where it is a benchmark task (`libero_goal` task 7), but it fails in the `libero_10` scene. This matches LIBERO-Plus and LIBERO-PRO, which report that VLAs trained on LIBERO largely ignore language.

**System 2's progress check is unreliable, in both directions.**

- *It misses finished steps.* "pick up the bowl" and "turn on the stove" were never judged done, so those episodes never reached their second command. In the "pick up the bowl" episodes, the gripper often closes, but the check keeps answering "still running". The 300-step timeout then coincides with the 300-step episode limit of `libero_goal`.
- *It accepts unfinished steps.* 7 of System 2's 11 success verdicts came while the LIBERO goal was false, and each one ended the episode. All six `libero_10` task 3 episodes ran "open the drawer", "place the bowl in the drawer" and "close the drawer" to "done", and none of them succeeded.

Only 14 commands after the first one ran in all 36 dual episodes, 12 of them in that one task. So this run does not isolate System 1's handling of atomic commands. Running the same atomic commands on a fixed schedule, with no VLM, separates the two (`--kinds schedule` in the command probe).

## Challenge questions

| Question | Current answer |
|---|---|
| **Counterfactual** | Partial. Run 13 compares removed (`passthrough`), frozen, fused and oracle System 2 on 16 episodes: frozen 1/4, the others 4/4. Shuffled commands are not implemented, and harder scenes are needed to see System 2 help |
| **Error recovery** | Not answered. Offset start states exist, but mid-episode perturbations are not implemented. Recovery commands ("open the gripper", "move back to the home position") did not noticeably change the robot's behaviour in the probe |
| **Bottleneck** | With the final System 2: System 1 execution. On `libero_10` from start B, 2 of 4 scenes fail even with oracle switching (run 14). The earlier VLM-only System 2 failed on perception (runs 5, 11 and 12) |

## Deliverables still open

- Oracle, frozen and shuffled conditions, and mid-episode perturbations
- A System 2 that can perceive progress. The perception benchmark decides whether a larger VLM does it
- A gibberish control for pi0.5, to show whether it follows commands
- A System 1 that follows subtask commands: SmolVLA with its top language layers trained, or pi0.5
- The SmolVLM judge: data collection and LoRA training are written and unit-tested, not yet run
- Rollout videos in the repo. They are generated by `side_by_side` but not committed
- The research report (PDF, at most 3 pages)

## Repository layout

```text
src/dual_system/
  system1/      System 1 interface and LeRobot wrapper (SmolVLA, pi0, pi0.5)
  system2/      supervisor, phase supervisor, reasoners (Qwen, scripted), passthrough
  knowledge/    command profiles: free_atomic, closed_set, phase_atomic
  configs/      experiment grids and conditions
  eval/         grid runner, report, lookup, side-by-side videos, command probe, model comparison videos, perception benchmark, subtask evaluation (atomic)
  finetune/     subtask labels from demos, splits, SmolVLA training, judge data and SmolVLM LoRA training, checkpoints
  env.py, rollout.py, triggers.py, robot_states.py, tracing.py, types.py
src/infra/      Colab install, EGL rendering, background jobs, result sync
scripts/        WSL test runner, robot start-state tuning, local unpacking of finetuning downloads
tests/          supervisor, trigger, rollout, perception and finetuning tests
docs/           experiment log and research notes (PDFs of the papers are not committed)
```

Start with [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for what was measured, [docs/RESEARCH.md](docs/RESEARCH.md) for the literature behind the design and [docs/PREREQUISITES.md](docs/PREREQUISITES.md) for the hardware constraints.
