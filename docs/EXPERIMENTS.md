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
7. **Stock pi0.5 already follows subtask commands; finetuning it on relabelled demos made it slightly worse.** With oracle switching it did 16/17 subtasks with real commands and 3/17 with gibberish. After 400 steps of expert-only training, shifted starts reached 17/28 goals against 21/28 for stock (run 10).
8. **No untrained VLM up to Qwen 8B judges reliably whether a subtask is finished.** On labelled demo frames, "Is this subtask finished?" scored at chance for every model. Asking about the visible state was better but still weak: Qwen 8B 0.64, with 30% false "done" (run 11).
9. **Showing the frame from the start of the subtask helps Qwen 4B and 8B most among nine prompt styles, but not enough.** AUROC rose from 0.64–0.66 to 0.71–0.72; still about 32% false "done" at the best threshold. One camera instead of two, describing first and multiple choice did not beat it (run 12).

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

### 10. pi0.5 on subtasks: stock, and finetuned for 400 steps

Folder `data/20261102-1704` (`pi05_eval`, `pi05_shift`, `pi05/checkpoints/step_0000400`), L4 GPU, commits `717fdc7` to `8feba82`. Start `lerobot/pi05_libero_finetuned_v044`, `train_expert_only=True` as LeRobot defines it: 693M trainable parameters, PaliGemma frozen. Batch 8 with gradient checkpointing, 0.23 steps/s, 17 GB. Same data, splits and held chunks as run 9.

Validation loss went from 0.60 (step 20) to 0.14 (step 400). Language response fell from 0.176 to 0.162, but stock pi0.5 already follows commands in rollouts at 0.176, so for pi0.5 this measure does not track command following. Rollouts decide.

`eval.atomic`, oracle switching, same scenes as run 9. Stored start (state A, init state 1, one episode each):

| | Stock | Step 400 |
|---|---|---|
| Real: subtasks / goals | 16/17, 6/7 | 16/17, 6/7 |
| Gibberish: subtasks / goals | 3/17, 1/7 | 5/17, 2/7 |
| Ran into the next subtask | 1/7 | 1/7 |

Shifted starts (states C and B, init states 1 and 2, paired seeds):

| | Stock | Step 400 |
|---|---|---|
| Real: goals | 21/28 | 17/28 |
| Real: subtasks | 49/68 | 42/68 |
| State B (about 19 cm): goals | 8/14 | 5/14 |
| Gibberish: goals / subtasks | 4/14, 11/34 | 2/14, 6/34 |
| Ran into the next subtask | 1/11 | 1/9 |

All 4 real-command pairs that differed favoured stock (sign test p = 0.12). Training was stopped at step 400. The demos are the ones the checkpoint was trained on, relabelled, so they hold nothing new for a model that already links subtask text to motion.

### 11. Untrained VLMs judging finished subtasks on demo frames

Folder `data/20261102-1704/vlm`, commit `88bf39e`, `eval.demo_judge`. 324 questions from 40 demo episodes (one per task), 138 finished. Labels from the subtask boundaries; frames within 4 steps of a boundary are left out. No rollouts and no failures, so this is the easy case.

Balanced accuracy averaged over pick, place and act, interval from resampling episodes:

| Model | finished? (both cameras) | + finish condition | state question | False "done", state | Peak GPU | Time per answer |
|---|---|---|---|---|---|---|
| Qwen3-VL 2B | 0.50 | 0.51 | 0.60 [0.52, 0.68] | 0.49 | 4.0 GB | 0.13 s |
| Qwen3-VL 4B | 0.53 | 0.54 | 0.58 [0.53, 0.68] | 0.26 | 8.4 GB | 0.15 s |
| Qwen3-VL 8B, bf16 | 0.52 | 0.52 | 0.64 [0.53, 0.71] | 0.30 | 16.4 GB | 0.22 s |
| SmolVLM2 2.2B | 0.50 | 0.49 | 0.57 [0.51, 0.63] | 0.08 | 4.3 GB | 0.30 s |

- Asked whether a subtask is finished, every model mostly says "no", including on frames where the wrist camera shows the bowl gripped and lifted (checked by eye). The written finish condition does not help.
- Asking about the visible state ("Is the bowl held between the gripper fingers and lifted above the table?") helps every model. Qwen 8B is best on act subtasks (0.78), but no model reaches a usable level: Qwen's false "done" rates of 26–49% would skip subtasks.
- Larger is not clearly better: the 8B interval overlaps the 2B and 4B ones.

### 12. Prompt styles for the VLM judge

Folder `data/20261102-1704/vlm` (`prompt_answers_*.jsonl`, `prompt_report.txt`), commits `1806317` to `3d4c6d6`, `eval.judge_prompts`, L4. Same 324 questions as run 11. All nine styles for Qwen 2B and 4B; the first five for Qwen 8B and SmolVLM.

Each style asks about the same frame differently (prompts in `judge_prompts.py`):
- **finished**: "Is this subtask finished?" with its finish condition (run 11's prompt)
- **state**: a question about the visible result, e.g. "Is the top drawer open?", both cameras
- **negated**: the opposite question, "Is the top drawer still closed?"
- **camera**: says which camera shows the result and what it looks like there
- **choice**: lettered states, the finished one never first
- **describe**: one sentence on what is visible, then yes or no
- **scene_only**: the state question, scene camera only
- **one_camera**: the wrist camera for a pick, the scene camera otherwise
- **before_after**: the scene at the start of the subtask, the scene now and the wrist now, asked to compare

Besides the answer, the probability of the finished answer is read from the first answer token. AUROC uses it without any threshold; "threshold acc." picks a threshold on half the episodes and scores the other half; false "done" is at that threshold.

| Model | Style | Answer acc. | Says done | AUROC [95%] | Threshold acc. | False done | s/answer |
|---|---|---|---|---|---|---|---|
| Qwen 2B | finished | 0.52 | 0.09 | 0.59 [0.55, 0.63] | 0.54 | 0.39 | 0.22 |
| | state | 0.56 | 0.54 | 0.61 [0.57, 0.66] | 0.58 | 0.61 | 0.21 |
| | negated | 0.50 | 0.00 | 0.52 [0.48, 0.56] | 0.51 | 0.79 | 0.21 |
| | camera | 0.55 | 0.89 | **0.66** [0.61, 0.70] | 0.62 | 0.32 | 0.21 |
| | choice | 0.53 | 0.95 | 0.62 [0.58, 0.67] | 0.56 | 0.22 | 0.21 |
| | describe | 0.53 | 0.64 | - | - | - | 1.00 |
| | scene_only | 0.57 | 0.42 | 0.59 [0.55, 0.63] | 0.56 | 0.44 | 0.20 |
| | one_camera | 0.58 | 0.80 | 0.61 [0.56, 0.65] | 0.61 | 0.63 | 0.21 |
| | before_after | 0.57 | 0.49 | 0.61 [0.55, 0.66] | 0.57 | 0.46 | 0.22 |
| Qwen 4B | finished | 0.56 | 0.12 | 0.64 [0.60, 0.69] | 0.60 | 0.28 | 0.27 |
| | state | 0.56 | 0.32 | 0.64 [0.59, 0.69] | 0.59 | 0.47 | 0.26 |
| | negated | 0.60 | 0.46 | 0.60 [0.55, 0.65] | 0.58 | 0.45 | 0.26 |
| | camera | 0.60 | 0.21 | 0.66 [0.62, 0.70] | 0.61 | 0.26 | 0.26 |
| | choice | 0.58 | 0.69 | 0.60 [0.56, 0.66] | 0.57 | 0.62 | 0.26 |
| | describe | 0.49 | 0.44 | - | - | - | 1.56 |
| | scene_only | 0.57 | 0.22 | 0.64 [0.60, 0.69] | 0.58 | 0.44 | 0.25 |
| | one_camera | 0.61 | 0.37 | 0.64 [0.59, 0.70] | 0.61 | 0.26 | 0.24 |
| | before_after | 0.64 | 0.73 | **0.71** [0.68, 0.76] | **0.66** | 0.32 | 0.29 |
| Qwen 8B | state | 0.59 | 0.37 | 0.66 [0.60, 0.71] | 0.59 | 0.38 | 0.37 |
| | before_after | 0.63 | 0.60 | 0.72 [0.67, 0.76] | 0.64 | 0.32 | 0.45 |
| SmolVLM | state | 0.54 | 0.11 | 0.60 [0.56, 0.64] | 0.57 | 0.31 | 0.56 |
| | before_after | 0.51 | 0.34 | 0.53 [0.51, 0.57] | 0.52 | 0.53 | 0.78 |

Qwen 8B and SmolVLM on the other first-pass styles are in `prompt_report.txt`; they were not better than these rows.

- **Best per model:** Qwen 4B with before_after; Qwen 2B with camera (where to look and what a held object looks like). Before_after does nothing for Qwen 2B and hurts SmolVLM, which seems to handle three images poorly.
- **One image** (scene_only, one_camera) ranks frames no better than both cameras. one_camera lowers false "done" for Qwen 4B (0.47 to 0.26) at equal AUROC.
- **Describing first** costs 4–6 times the time and is not better. The sentences are often wrong: Qwen 2B described the gripper "holding a plate" in frames where it held nothing.
- **Negated questions** do not cancel the lean towards one answer: Qwen 2B answered "no" to every negated question, so it says "not finished" either way.
- **The old "finished" prompt ranks above chance** (AUROC 0.59–0.64); its low accuracy in run 11 was mostly the "no" lean, which a threshold corrects. Every style still leaves about a third of unfinished frames called "done" or misses many finished ones.

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

- Answered: pi0.5 does follow subtask commands (run 10).
- Can a System 2 judge progress without a trained VLM? Run 11 says not from images alone. Simulator-free alternatives are signals the robot has anyway (gripper width, contact, lift), or a judge trained on rollout frames.
- Does SmolVLA learn to follow subtask commands if the top layers of its language model also train, or if gibberish frames are trained to stay still? Run 9 shows that training the expert alone does not.
- Do the actions actually change at a command switch? The queue clearing is unit-tested, but actions were not compared at the switch.
- Which VLM, if any, can judge progress? The perception benchmark answers this.
