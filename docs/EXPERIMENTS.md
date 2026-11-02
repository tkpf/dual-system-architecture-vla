# Experiment Log

What was run, what it showed, and what it does not show. Everything here was measured on a Colab T4 with LeRobot 0.6.1 unless noted. Episode counts are small, so treat single-digit differences as rough. Results are not comparable with published LIBERO numbers, because the protocol differs.

The `data/` folder is not committed. Folder names below refer to the local copy.

## Summary

1. **pi0.5 is the better System 1.** On the same atomic schedule from shifted starts, it reached the goal in 5/12 episodes and SmolVLA in 0/12.
2. **No dual configuration has beaten System 1 alone overall.** Every dual variant so far has a lower total than its baseline.
3. **The whole instruction beats atomic commands on `libero_goal`.** pi0.5 alone scored 12/12 across shifted and stored starts, and only 3/8 on scheduled atomic commands from shifted starts (different seeds).
4. **System 2's main failure is perception.** Qwen 2B says a glowing red burner is dark and a held bowl is on the table, so better prompts alone did not help.
5. **pi0.5 often does not obey command boundaries.** It sometimes finishes the task under a command that means something else.
6. **Training only SmolVLA's action expert on subtasks did not make it follow language.** After 600 steps, validation loss fell from 0.53 to 0.32, but actions differed less between a real command and gibberish than with the stock model (0.172 vs 0.209). Training was stopped there (run 9).

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

### 6. Offline perception benchmark (partial, files lost)

Run `/content/runs/20261101-2211` on Colab. It compares Qwen 2B, 4B, 8B (4-bit) and SmolVLM2-2.2B on yes/no questions about pi0.5 rollout frames, scored against simulator truth. See [the skill](../.claude/skills/dual-system-experiments/SKILL.md) for how it works.

The runtime was replaced before anything was downloaded, so these numbers were read from the live job and cannot be rechecked. Scores are balanced accuracy, where 0.5 is chance.

| Model | Answers | both cameras | scene only | with definition |
|---|---|---|---|---|
| Qwen 2B | 1,344, complete | 0.72 | 0.66 | 0.70 |
| SmolVLM2-2.2B | 160, partial | 0.50 | 0.50 | 0.50 |
| Qwen 4B | 85, partial | not scored | | |

SmolVLM answered "no" to every question with both cameras or the scene camera, and "yes" to every question with the definition. Its run used image splitting (17 tiles per image), unlike the later finetuning code. Stage 5 of the finetuning plan scores untrained SmolVLM and Qwen 2B again on new frames.

### 7. Stock SmolVLA on atomic commands (`smolvla_libero`)

Folder `data/20261101-2311/smolvla_stock`. `command_probe --kinds schedule`, states A and C, 2 seeds: 12 episodes per prompt kind, commands switched at fixed steps.

| Prompt | Goal reached |
|---|---|
| Atomic commands | 0/12 |
| Whole instruction | 9/12 |
| Gibberish | 1/12 |

The checkpoint that later finetuning starts from does the whole task but not the same task given as subtasks.

### 8. Subtask labels for LIBERO demos

Folder `data/20261102-0626`, commit `76e679c`. `lerobot/libero` has no simulator state, so each episode was matched to its raw LIBERO demo by actions, and the demo's stored states were replayed without rendering to find when each subtask finished.

- 1,492 of 1,693 episodes kept. Dropped: no raw demo within tolerance 170, goal not reached 22, object never held 9.
- Median phase length in frames (10 fps): pick 64, place 57, act 95.
- Contact sheets caught two labelling errors that unit tests had passed. A pick that ended at the last grasp showed the moka pot already over the stove, because finger contact flickers during the carry; it now ends at the first firm grasp. A place that ended when the goal predicate turned true cut off the release; it now ends when the fingers let go. At that point the gripper command is already "open" in 1,257 of 1,318 final place phases.
- LIBERO's `On` predicate turns true while the object is still in the gripper. Rollout evaluation (`eval.atomic`) therefore also requires the fingers to have let go before a place counts as done.

### 9. SmolVLA finetuned on subtasks, pilot (stopped at step 600)

Folder `data/20261102-0626/s1` (checkpoints 200, 400, 600) and `pilot`, training at commit `76e679c`. Start `lerobot/smolvla_libero`. Trained: the action expert and its input and output projections, 99.9M of 450M parameters. Frozen: the whole VLM (vision encoder, connector, language model). Batch 16, learning rate 1e-5, chunks held still past each subtask's end. Unseen tasks and 10% of demos per task are held out.

Same validation frames for every row. "Language response" is the mean absolute difference between predicted actions for the real command and for gibberish, with the same noise.

| | Validation loss | Language response |
|---|---|---|
| Stock | 1.11 | 0.209 |
| Step 200 | 0.53 | 0.177 |
| Step 400 | 0.39 | 0.174 |
| Step 600 | 0.32 | 0.172 |

The plan's gate needed language response to grow by about step 500. It fell, so training stopped.

Rollouts with `eval.atomic`: the next command is issued when the simulator says the subtask is done, 150 steps per subtask. Init state 1, state A, 1 seed: 3 seen scenes and 4 unseen tasks, one episode each.

| | Step 600, real | Step 600, gibberish | Stock, real | Stock, gibberish |
|---|---|---|---|---|
| Seen: goal | 2/3 | 1/3 | 0/3 | 0/3 |
| Seen: subtasks done | 4/7 | 5/7 | 0/7 | 2/7 |
| Unseen: goal | 0/4 | 1/4 | 1/4 | 0/4 |
| Unseen: subtasks done | 1/10 | 2/10 | 3/10 | 0/10 |
| Wrong object lifted | 2 | 1 | 0 | 1 |

- On seen scenes, step 600 completes subtasks that stock does not, but it does as well with gibberish. Gibberish completed all 3 moka pot subtasks, where the real commands failed the first one ("turn on the stove").
- On unseen tasks, step 600 is not better than stock, and lifted the wrong object twice.
- Neither model ran on into the next subtask when kept on the first command (0/3 and 0/2).
- One episode per cell: these are signs, not results. The likely reading is that the expert learned to infer the phase from the images, and the frozen VLM gives it no way to make commands easier to tell apart.

## Practical measurements

| Measure | Value |
|---|---|
| pi0.5 alone | About 0.35 s per sim step on the T4 |
| pi0.5 GPU memory | About 10.3 GB while running; 14.9 GB seen during the compile attempt |
| pi0.5 + Qwen 2B | About 14 GB, fits a T4 |
| Dual episode, 300 steps, reasoning prompt | About 140 to 160 s |
| Phase check with reasoning | About 3 s per call |
| Qwen 2B one-letter check | About 0.14 s per call |
| SmolVLA training, action expert only, batch 16 | 0.22 steps/s over 600 steps, 4.7 GB GPU: about 15 minutes per 200 steps |
| `eval.atomic`, 7 tasks with real, gibberish and stop runs | About 10 minutes per model |

Hub checkpoint sizes, for planning: Qwen 2B 4.3 GB, 4B 8.9 GB, 8B 17.5 GB, SmolVLM2-2.2B 9.0 GB. pi0.5 with Qwen 4B is expected not to fit on a T4 together (not measured).

## Open questions

- Does pi0.5 follow commands at all? The schedule probe had no gibberish control. Run one before claiming it does. `eval.atomic` with its gibberish and stop runs would answer this on the same scenes as run 9, once it accepts `--type pi05`.
- Does SmolVLA learn to follow subtask commands if the top layers of its language model also train, or if gibberish frames are trained to stay still? Run 9 shows that training the expert alone does not.
- Do the actions actually change at a command switch? The queue clearing is unit-tested, but actions were not compared at the switch.
- Which VLM, if any, can judge progress? The perception benchmark answers this.
