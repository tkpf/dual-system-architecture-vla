# Dual-System VLA on LIBERO

A two-level robot policy for the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) manipulation benchmark, built on [LeRobot](https://github.com/huggingface/lerobot). The challenge brief is in [CHALLENGE.md](CHALLENGE.md).

- **System 2** (slow): a frozen Qwen3-VL-2B vision-language model inside a rule-based supervisor. It splits the instruction into atomic commands and judges progress from camera images.
- **System 1** (fast): a frozen SmolVLA policy that turns the current command and the observation into robot actions.

Both models are pretrained and neither is trained here, so the repo has no training loop and no dataset to download. Evaluation runs live in the simulator.

> **Status: work in progress.** The pipeline runs end to end, but so far the dual system performs **worse** than SmolVLA alone. The ablation study is incomplete. See [Results so far](#results-so-far).

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
 camera + state ─► System 1: SmolVLA ──► 7-D action at every sim step ──► LIBERO
```

**Interface.** System 2 passes plain text, written into the prompt that SmolVLA reads. When the command changes, SmolVLA's queued actions are discarded. Text is readable in videos and easy to freeze, swap or corrupt for an ablation.

**When System 2 runs.** Triggers call it every 20 steps, and also on a gripper toggle, when the robot stops moving, on an empty grasp, and after 300 steps on one command. They are set in [full_free_atomic.yaml](src/dual_system/configs/conditions/full_free_atomic.yaml).

**Swappable parts.** The supervisor logic in [supervisor.py](src/dual_system/system2/supervisor.py) is deterministic. All perception goes through a `Reasoner` ([reasoners.py](src/dual_system/system2/reasoners.py)), so a different VLM changes the reasoning quality and nothing else. System 1 sits behind a small interface ([base.py](src/dual_system/system1/base.py)).

| Condition | What it does |
|---|---|
| `passthrough` | No System 2: the full instruction goes to SmolVLA unchanged. This is the baseline |
| `full_free_atomic` | System 2 writes its own atomic commands. Default dual condition |
| `full` | System 2 chooses commands from the candidate list in the `closed_set` profile |
| `full_scripted` | Fixed answers instead of a VLM, for pipeline checks |

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
# Start the grid in the background. Finished cells are skipped on restart.
!python -m infra start w0 "python -m dual_system.eval.grid free_atomic_test --res /content/res --stop-after-min 40"
!python -m infra status          # one line of JSON: jobs, finished episodes, GPU and RAM

# Tables, sub-goal CSV and side-by-side videos
!python -m dual_system.eval.report /content/res
!python -m dual_system.eval.lookup /content/res --task 0 --state B
!python -m dual_system.eval.side_by_side /content/res      # writes /content/res/videos

# Does System 1 follow a command? No System 2, compared with a gibberish control
!python -m infra start probe "python -m dual_system.eval.command_probe --res /content/probe"
```

| Config | Purpose |
|---|---|
| `free_atomic_test` | Default. `passthrough` vs `full_free_atomic`, 6 tasks, 3 robot start states, 2 seeds: 72 episodes |
| `profiles_test` | Compares the `closed_set` and `free_atomic` command profiles |
| `libero_plus_checkpoint` | SmolVLA alone, with the checkpoint trained on `lerobot/libero_plus` |
| `first_test` | The first dual-system run, kept for reference |
| `smoke` | Short check with real SmolVLA and no VLM. Runs on CPU |

Robot start states: `A` is the stored LIBERO start. `C` and `B` add a joint offset that moves the gripper about 10 cm and 19 cm. A pair of episodes shares its task, state and seed, so the pair differs only in the condition.

Each cell folder holds `episodes.jsonl` (outcome), `subgoals.jsonl` (each command and how it ended), `trace.jsonl` (every System 2 call with its raw answer), `steps.jsonl`, keyframes and `episode.mp4`.

## Results so far

From `free_atomic_test` at commit `c7452e6`. The run covered **27 of the 72 planned episodes** before it stopped, so these numbers are a pilot. `libero_goal` used tasks 0, 1 and 4; `libero_10` used tasks 0 and 2. All episodes come from the first of the two planned seeds.

| Start state | SmolVLA alone | Dual system |
|---|---|---|
| A (stored start) | 5/5 | 1/5 |
| C (about 10 cm shift) | 3/5 | 1/4 |
| B (about 19 cm shift) | 0/4 | 0/4 |
| **Total** | **8/14** | **2/13** |

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

**System 2 rarely advances.** Across the 13 dual episodes, only 2 commands were judged done, and both were the single-step "open the middle drawer". **No second atomic command was ever issued**, so a "place" step never ran. In the "pick up the bowl" episodes, the gripper stays closed on the object from around step 150 in some cases, but the check keeps answering "still running". The 300-step timeout then coincides with the 300-step episode limit of `libero_goal`.

The second cause means that this run does not isolate System 1's handling of atomic commands. Running the same atomic commands on a fixed schedule, with no VLM, would separate the two.

## Challenge questions

| Question | Current answer |
|---|---|
| **Counterfactual** | Partial. Only `passthrough` vs `full*` has been run, and the dual system is worse. Frozen, shuffled and oracle conditions are not implemented |
| **Error recovery** | Not answered. Offset start states exist, but mid-episode perturbations are not implemented. Recovery commands ("open the gripper", "move back to the home position") did not noticeably change the robot's behaviour in the probe |
| **Bottleneck** | Preliminary: both systems. System 1 follows language weakly, and System 2's progress check rarely closes a step. An oracle condition is needed to measure each one's share |

## Deliverables still open

- Oracle, frozen and shuffled conditions, and mid-episode perturbations
- A System 1 that follows atomic commands. SmolVLA is expected to be replaced, without retraining
- Full 72-episode grid, with confidence intervals
- Rollout videos in the repo. They are generated by `side_by_side` but not committed
- The research report (PDF, at most 3 pages)

## Repository layout

```text
src/dual_system/
  system1/      System 1 interface and SmolVLA wrapper
  system2/      supervisor, reasoners (Qwen, scripted), passthrough
  knowledge/    command profiles: free_atomic, closed_set
  configs/      experiment grids and conditions
  eval/         grid runner, report, lookup, side-by-side videos, command probe
  env.py, rollout.py, triggers.py, robot_states.py, tracing.py, types.py
src/infra/      Colab install, EGL rendering, background jobs, result sync
scripts/        WSL test runner, robot start-state tuning
tests/          supervisor, trigger and rollout tests
docs/           research notes (PDFs of the papers are not committed)
```

Start with [docs/RESEARCH.md](docs/RESEARCH.md) for the literature behind the design and [docs/PREREQUISITES.md](docs/PREREQUISITES.md) for the hardware constraints.
