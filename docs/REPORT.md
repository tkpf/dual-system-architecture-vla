# Dual-System VLA on LIBERO: what a System 2 adds to pi0.5, and what it does not

*Draft, 2026-11-03. Experiment details and folders are in [EXPERIMENTS.md](EXPERIMENTS.md); run numbers below refer to it.*

## 1. Summary

We built a two-level policy for LIBERO. A frozen pi0.5 policy (System 1) produces every action. A slow System 2 splits the instruction into atomic commands ("pick up the bowl", "place the bowl on the stove") and decides when to move to the next one. The final System 2 judges progress mainly from signals the robot has anyway, the gripper opening and the hand height. It uses a small VLM (Qwen3-VL 2B) only for subtasks no robot signal shows, such as "turn on the stove".

The ablation does **not** show that System 2 improves System 1:

- **Easy scenes:** pi0.5 alone and the dual system both succeed (4/4).
- **Ten `libero_10` scenes from a shifted start:** the whole instruction beats atomic commands, even atomic commands switched by the simulator itself (5/10 against 3/10 with oracle switching).
- **Bottleneck:** most failures remain under oracle switching, so they come from System 1 under atomic commands, not from System 2's reasoning.
- **Error recovery:** System 2 detects a dropped object as reliably as the simulator does, but re-commanding the pick does not change the outcome.

## 2. Architecture

```
instruction ──► System 2 ──────────────────────────────────────────────┐
                 decompose once: plan of atomic commands                │ one English
                 per step: gripper width, hand height ──► phase done?  │ sentence
                 act phases only: Qwen3-VL 2B, P("done") ≥ 0.8 twice    │
                 timeouts, empty-grasp and drop recovery                ▼
camera + proprioception ───────────────────────────────► System 1: pi0.5 ──► 7-D action, 20 Hz ──► LIBERO
```

**System 1** is `lerobot/pi05_libero_finetuned_v044`, used frozen. It predicts action chunks and runs 10 actions per inference. When the command changes, the queued actions are dropped, so the new command acts at once. pi0.5 follows subtask commands: with simulator-timed switching it finished 16/17 subtasks with real commands and 3/17 with gibberish (run 10). The first System 1, SmolVLA, did not: gibberish produced the same behaviour as real commands (runs 2 and 7).

**The interface is text.** System 2 writes one sentence into pi0.5's prompt. Text is readable in videos, matches what pi0.5 was trained on, and is easy to freeze or replace for an ablation.

**System 2** is deterministic logic with one perception call. It never ends an episode, because early verdicts were the main way earlier versions failed (section 3).

- **Plan:** the atomic commands come from the subtask segmentation of the LIBERO demos. Qwen's own decomposition is logged next to it but not used. It matched the reference in 3 of 4 `libero_goal` episodes and 4 of 10 `libero_10` episodes (often a harmless paraphrase, sometimes a missing step).
- **Pick done:** the gripper holds something (finger width between empty-closed and open) for 8 steps, and the hand has risen 3 cm.
- **Place done:** the gripper reopens after a hold. With drop detection on, the hand must also have come down at least 2 cm while holding, otherwise the release counts as a drop and the pick is commanded again.
- **Act done** (open, close, turn on): Qwen3-VL 2B is asked a camera-specific yes/no question every 10 steps. Its probability of "yes" must be at least 0.8 twice in a row.
- **Timeouts and recovery:** a phase times out after 150 steps. A gripper that closes on nothing triggers "open the gripper" and a retry.

## 3. Why System 2 does not rely on the VLM

The first System 2 asked the VLM after every trigger whether the command was running, done or failed, and ended the episode on a success verdict. With SmolVLA as System 1 it reduced success from 19/36 to 4/36. 7 of its 11 success verdicts came while the goal was false (run 1). With pi0.5 it reduced success from 8/12 to 3/12 (run 4). A phase-aware prompt removed false successes but never declared a step done in 76 checks (run 5).

Offline, on labelled demo frames, no untrained VLM up to Qwen3-VL 8B judged reliably whether a subtask was finished. Asking "is this subtask finished?" was at chance. The best of nine prompt styles reached an AUROC of 0.71 and still called about a third of unfinished frames done (runs 11 and 12). Finetuning System 1 on relabelled subtask demos did not help either (runs 9 and 10). This is why the final System 2 moves pick and place detection to the gripper and keeps the VLM for the subtasks no robot signal shows.

## 4. Ablation

**Protocol.** Every condition runs on the same scene, robot start and seed. Shifted starts move the gripper about 10 cm (start C) or 19 cm (start B) from the stored LIBERO pose. Each phase gets 150 steps (300 to 600 per task), and every episode ends at the environment's success. Conditions:

- **passthrough:** System 2 removed; the whole instruction is the prompt.
- **frozen:** System 2 frozen after its first output; the first atomic command for the whole episode.
- **fused:** the final System 2.
- **oracle:** the simulator decides when a phase is done. This is the best any progress check could do.

### 4.1 Counterfactual

| Condition | `libero_goal`, 4 scenes (run 13) | `libero_10`, 10 scenes, start B (run 15) |
|---|---|---|
| passthrough | 4/4 (median 99 steps) | 5/10 (349) |
| frozen | 1/4 (218) | 4/10 (376) |
| fused | 4/4 (101) | 3/10 (501) |
| oracle | 4/4 (102) | 3/10 (195) |

- **Commands must change for atomic commands to work.** On `libero_goal`, frozen picked the bowl up and held it.
- **Fused switching is about as good as oracle switching.** On the single-object scenes, fused switched 0 to 2 steps from the simulator's own pick check. The stove check (VLM) switched 13 steps after the burner came on. Fused and oracle reached the same totals, with equal steps on `libero_goal`.
- **Atomic commands themselves cost success on `libero_10`.** With oracle switching, success fell from 5/10 to 3/10. Passthrough succeeded and oracle failed on 4 scenes, and the reverse happened on 2. pi0.5 was trained on whole `libero_10` instructions, so the atomic commands are a distribution shift for it. A naive baseline, the frozen first command, does as well as the full System 2.

### 4.2 Error recovery

Once the gripper has held the bowl for 20 steps, it is forced open for 10 steps, so the bowl falls during the carry (run 16; both bowl scenes, start C, 2 seeds).

| passthrough | fused | fused, no recovery | oracle |
|---|---|---|---|
| 2/4 | 2/4 | 2/4 | 3/4 |

- **System 2 detected every drop** within 1 to 2 steps of the simulator.
- **Detection changed no outcome:** fused with and without recovery failed on the same episodes. On the cabinet scene, pi0.5 picks the bowl up again under any command. On the stove scene, re-commanding the pick led to a new grasp but not to success.
- **Oracle's one extra success** needed three re-picks.

So System 2 notices errors, but recovery is limited by execution.

### 4.3 Bottleneck: reasoning or execution?

We attribute each failed fused episode by whether oracle switching failed on the same scene, start and seed.

- **Execution (System 1):** 6 of 7 fused failures on `libero_10` also failed under oracle switching. Either System 1 cannot carry out the atomic commands, or the atomic-command interface itself is the problem. In the remaining failure, fused and oracle switched 1 step apart and the rollouts diverged, which is noise.
- **Reasoning (System 2):**
  - *Gripper signals:* the gripper signal says that something is held, not what. In two-object scenes, 15 of 19 signal switches came while the simulator said the named object was not picked or placed. Mostly pi0.5 had taken the other object, or the objects in another order. This cost little, because pi0.5 largely ignores which object the command names.
  - *VLM checks:* the stove check worked in all three runs where it applied. The drawer and microwave close checks were too lenient: Qwen gave 0.82 to 0.84 from the first look.
- **How we know:** the oracle condition gives an upper bound for any progress check on the same rollout seeds. Fused reaches that bound. Every remaining gap to success is a gap that perfect reasoning does not close.

## 5. Limitations

- **Small samples:** one or two seeds per scene and a single LIBERO init state. Single-scene differences between conditions are mostly noise: a one-step difference in a switch can flip an outcome (run 15, book scene).
- **Tuning on test starts:** the gripper thresholds were set from pi0.5 command-probe traces after a failed smoke test on start C, so the start C results are not fully independent of tuning.
- **Not run:** shuffled commands and a gibberish System 2 were not run for pi0.5 in the dual system. Gibberish was tested for System 1 alone (run 10).
- **Object identity:** the gripper signal cannot tell which object is held. A wrist-camera check of the held object is the obvious next step.

## 6. Reproducing

```bash
python -m dual_system.eval.final run --res <res> --seeds 2 --states C B                          # libero_goal scenes
python -m dual_system.eval.final run --res <res> --suite-tasks libero_10:0 ... libero_10:9 --states B
python -m dual_system.eval.final run --res <res> --tasks 0 1 --states C --seeds 2 \
    --conditions passthrough fused fused_norec oracle --perturb drop                               # error recovery
python -m dual_system.eval.final report <res>; python -m dual_system.eval.final video <res>       # tables, 2x2 videos
```
