---
name: dual-system-experiments
description: Run and read this project's experiments on Colab: pi0.5 as System 1, single vs dual grids, the fixed-schedule command probe, side-by-side videos, the phase-aware System 2 and the offline VLM perception benchmark. Use when choosing or running an experiment, comparing System 1 or System 2 variants, or interpreting results. For a fresh runtime, use colab-startup first.
---

# Dual-system experiments

Results so far are in `docs/EXPERIMENTS.md`. Add every new run there, with its folder, commit and episode counts.

Training models (SmolVLA on subtasks, the SmolVLM judge), subtask evaluation with `eval.atomic` and downloading checkpoints are in the `finetuning` skill.

## Before a run
1. Commit, push the working branch, and verify with `git ls-remote origin` that it matches local `HEAD`.
2. On Colab: `git pull`, then `RUN=$(python -m infra newrun)`. Every run gets a new `YYYYMMDD-HHMM` folder.
3. pi0 and pi0.5 need the gated tokenizer: the user must have accepted the licence at `huggingface.co/google/paligemma-3b-pt-224` and set the Colab secret `HF_TOKEN`. Set it in the kernel without printing it (`os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")`). Jobs started from the notebook inherit it.
4. Tell the user when an action is expensive, and do not run more episodes than asked.

## Choosing an experiment

| Question | Tool |
|---|---|
| Single vs dual on full episodes | `python -m dual_system.eval.grid <config> --res $RUN` |
| Does a System 1 follow given commands, without a VLM? | `command_probe --kinds schedule`, with controls |
| Compare two System 1 models on the same scenes | `command_probe` twice, then `compare_probe` |
| Can a VLM see what System 2 must judge? | `perception`, no robot control involved |

Configs: `pi05_atomic_test` (pi0.5, shifted starts C and B, 12 episodes per condition), `pi05_phase_test` (pi0.5, state A, 6 per condition, 300-step limit), `free_atomic_test` (SmolVLA).

## Command probe
```bash
python -m dual_system.eval.command_probe --type pi05 --path lerobot/pi05_libero_finetuned_v044 \
  --kinds schedule --only-first --seeds 2 --states C B --video --res $RUN/pi05
python -m dual_system.eval.command_probe --type smolvla --path lerobot/smolvla_libero_plus \
  --kinds schedule --only-first --seeds 2 --states C B --video --res $RUN/smolvla
python -m dual_system.eval.compare_probe $RUN/pi05 $RUN/smolvla --out $RUN/side_by_side
```
- `--only-first` runs only the atomic schedule, 12 episodes here. Without it, each scene also runs the whole instruction and gibberish: 36 episodes. Those controls are what show whether a model follows language.
- Switches come at fixed steps (`SCHEDULE_CASES`: 120 per command on `libero_goal`, 130 on `libero_10`), not when a step is finished.
- Each row records `start_shift_cm`, `start_contacts` and `first_success_step`. Use `first_success_step`, because `env_success_at_end` can go false again after success.
- Finished runs are skipped when the same folder is reused.

## Videos
- Probe runs: `compare_probe` writes one video per pair plus `all_episodes.mp4`.
- Grids: `python -m dual_system.eval.side_by_side $RUN` writes `$RUN/videos`, with no combined file. Join them with ffmpeg's concat demuxer and `-c copy`.
- Extract a few frames and look at them before calling videos ready.

## System 2 variants
- `full_free_atomic`: writes its own commands, one-letter progress check.
- `full_phase_atomic`: `PhaseSupervisor` plus the `phase_atomic` profile. The prompt holds the phase list, position and finish condition. Qwen reasons briefly and ends with `CURRENT SUBTASK: <n>. <text>` or `DONE`. A later phase moves on by one, an earlier one goes back, each after 2 agreeing answers. Sub-goal timeout is 150 steps, because a timeout equal to the episode limit blocks recovery.
- Finish conditions are in `knowledge/profiles/phase_atomic.yaml`, matched by the first keyword found in the phase text.

## Perception benchmark
```bash
python -m dual_system.eval.perception collect --res $RUN        # about 20 min: 12 pi0.5 episodes, 16 frames each
pip install -q bitsandbytes                                     # only for qwen8b; not part of infra install
python -m dual_system.eval.perception run qwen2b --res $RUN     # also smolvlm, qwen4b, qwen8b; resumable
python -m dual_system.eval.perception report --res $RUN
```
- Labels are simulator truth. Held means risen more than 3 cm, gripper width below 7 cm and gripper within 15 cm of the object. Goals come from the LIBERO predicates.
- Modes: `both` cameras, `scene` only, and `definition` (both plus the finish condition, to see whether definitions steer the answer).
- The score is balanced accuracy averaged over questions, so always answering the same word gets 0.5. The interval resamples whole episodes, since frames in one episode are correlated.
- One model at a time, because they are separate processes. Qwen 8B is 17.5 GB, so it loads in 4 bits on a T4. That makes it a quantized result, so report it as one.

## Pitfalls seen
- **pi checkpoints enable `compile_model` with `max-autotune`.** It stalled the first episode for over 10 minutes. `LeRobotSystem1` turns it off. Stock `lerobot-eval` needs `--policy.compile_model=false`.
- **pi0 and pi0.5 read `_action_queue`, not `_queues[action]`.** Clearing the wrong one silently keeps up to 10 old actions after a command change. `clear_action_queue` handles this.
- **A notebook kernel keeps old imports.** After `git pull`, run `importlib.reload(infra.bootstrap)` and `importlib.reload(infra.sync)` before calling `to_browser`, or restart.
- **`system1.reset()` clears the command.** Call `set_command` after it.
- **A 27-episode partial pull gave a wrong conclusion once.** Check `SYNC_INFO.json` and the `data/` folders for a more complete pull before stating results.

## Reading results
- Score against `success` and `first_success_step`, never against System 2's verdict.
- When System 2 judges, env success does not end the episode, so compare `first_success_step` too.
- Check which command was active when a goal was reached. Reaching it under an unrelated command means the command did not cause it.
- Compare Qwen's raw answers with saved keyframes before blaming prompts or System 1.
- Only compare models on the same scenes and start states, and say when seeds differ. Grid and probe seeds differ.
- State the sample size. Use Fisher's exact test or a sign test on pairs, and do not claim a difference within noise.
