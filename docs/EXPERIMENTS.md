# Experiment Log

What was run, what it showed, and what it does not show. Everything here was measured on a Colab T4 with LeRobot 0.6.1 unless noted. Episode counts are small, so treat single-digit differences as rough. Results are not comparable with published LIBERO numbers, because the protocol differs.

The `data/` folder is not committed. Folder names below refer to the local copy.

## Summary

1. **pi0.5 is the better System 1.** On the same atomic schedule from shifted starts, it reached the goal in 5/12 episodes and SmolVLA in 0/12.
2. **No dual configuration has beaten System 1 alone overall.** Every dual variant so far has a lower total than its baseline.
3. **The whole instruction beats atomic commands on `libero_goal`.** pi0.5 alone scored 12/12 across shifted and stored starts, and only 3/8 on scheduled atomic commands from shifted starts (different seeds).
4. **System 2's main failure is perception.** Qwen 2B says a glowing red burner is dark and a held bowl is on the table, so better prompts alone did not help.
5. **pi0.5 often does not obey command boundaries.** It sometimes finishes the task under a command that means something else.

## Runs

### 1. SmolVLA, single vs dual (`free_atomic_test`)

Folder `2026-11-01_20-36_dual`, commit `c7452e6`. Checkpoint `lerobot/smolvla_libero_plus`, 6 tasks, 3 start states, 2 seeds: 72 episodes.

| Start state | SmolVLA alone | Dual |
|---|---|---|
| A (stored) | 12/12 | 2/12 |
| C (about 10 cm) | 5/12 | 2/12 |
| B (about 19 cm) | 2/12 | 0/12 |
| **Total** | **19/36** | **4/36** |

15 pairs succeeded only alone, none only dual. 7 of 11 System 2 success verdicts came while the goal was false.

### 2. SmolVLA command probe

Folder `2026-11-01_19-58_dual/command_probe`. No System 2, state A, 2 seeds, 200 steps for task commands. Gibberish produced the same behaviour as real commands in several cases: it lifted the bowl and turned on the stove. "pick up the moka pot" turned on the stove instead. This matches LIBERO-Plus and LIBERO-PRO.

### 3. pi0.5 vs SmolVLA on a fixed atomic schedule

Folder `20261101-2138`, commit `1421b93`. Commands switched at fixed steps, with no VLM. 3 scenes, states C and B, 2 seeds: 12 episodes per model. There was no gibberish control.

| | pi0.5 | SmolVLA (`libero_plus`) |
|---|---|---|
| Goal reached | 5/12 | 0/12 |
| Lifted the commanded bowl | 8/8 | 0/8 |
| Lifted the moka pot | 4/4 | 0/4 |
| Turned on the stove | 2/4 | 1/4 |
| Lifted the wrong object (cream cheese) | 1/8 | 3/8 |

Fisher's exact test on goal reached: p about 0.04. In 3 of pi0.5's 5 successes, the goal was reached before "place" was issued.

### 4. pi0.5, single vs dual, shifted starts (`pi05_atomic_test`)

Folder `20261101-2229`, commit `cb728e9`. Condition `full_free_atomic`.

| Scene | pi0.5 alone | Dual |
|---|---|---|
| `libero_goal` task 1 | 4/4 | 1/4 |
| `libero_goal` task 4 | 4/4 | 0/4 |
| `libero_10` task 2 | 0/4 | 2/4 |
| **Total** | **8/12** | **3/12** |

- 6 false successes ended episodes, all on `libero_goal`. One episode was wrongly refused by the gate ("stove not visible").
- The 2 dual wins came on "turn on the stove" alone, past the step the goal was reached. They show that a shorter prompt can help, not that progress tracking does.
- One inspected episode put the bowl on the plate under "pick up the bowl", when the target was the cabinet.

### 5. pi0.5, single vs phase-aware dual, stored start (`pi05_phase_test`)

Folder `20261101-2302`, commit `6785078`. Condition `full_phase_atomic`, state A, 2 seeds, 300-step limit.

| Scene | pi0.5 alone | Dual |
|---|---|---|
| `libero_goal` task 1 | 2/2 | 0/2 |
| `libero_goal` task 4 | 2/2 | 0/2 |
| `libero_10` task 2 | 2/2 | 2/2 |
| **Total** | **6/6** | **2/6** |

- Qwen answered "running" 75 times out of 76, gave 1 unusable answer, and never said "done". No second command was issued.
- There were no false successes, but the check is now too strict in practice, because it can't see the condition.
- Both dual wins reached the goal at steps 223 and 231, while the active command was "move back to the home position". The command was not driving the behaviour.
- One episode was refused at step 0 because "the bowl is not on the stove", which treats the goal as a precondition.

Qwen's answers against the saved frames:

| Episode, step | Frames show | Qwen said |
|---|---|---|
| `libero_10` task 2, steps 100 to 140 | Burner glows red in both cameras | "The stove is off because the burner ring is dark, not glowing red" |
| `libero_goal` task 1, steps 60 to 100 | Bowl lifted and held, large in the wrist view | "The bowl is on the table… with its gripper open" |

### 6. Offline perception benchmark (results pending)

Run `/content/runs/20261101-2211` on Colab. It compares Qwen 2B, 4B, 8B (4-bit) and SmolVLM2-2.2B on yes/no questions about pi0.5 rollout frames, scored against simulator truth. See [the skill](../.claude/skills/dual-system-experiments/SKILL.md) for how it works. No result exists yet.

## Practical measurements

| Measure | Value |
|---|---|
| pi0.5 alone | About 0.35 s per sim step on the T4 |
| pi0.5 GPU memory | About 10.3 GB while running; 14.9 GB seen during the compile attempt |
| pi0.5 + Qwen 2B | About 14 GB, fits a T4 |
| Dual episode, 300 steps, reasoning prompt | About 140 to 160 s |
| Phase check with reasoning | About 3 s per call |
| Qwen 2B one-letter check | About 0.14 s per call |

Hub checkpoint sizes, for planning: Qwen 2B 4.3 GB, 4B 8.9 GB, 8B 17.5 GB, SmolVLM2-2.2B 9.0 GB. pi0.5 with Qwen 4B is expected not to fit on a T4 together (not measured).

## Open questions

- Does pi0.5 follow commands at all? The schedule probe had no gibberish control. Run one before claiming it does.
- Do the actions actually change at a command switch? The queue clearing is unit-tested, but actions were not compared at the switch.
- Which VLM, if any, can judge progress? The perception benchmark answers this.
