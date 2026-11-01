# Robustness Research and Ablation Plan

Back to [RESEARCH.md](../RESEARCH.md).

## What the robustness papers say

**[LIBERO-Plus](../papers/2510.13626-libero-plus.pdf)** perturbs LIBERO along seven dimensions: object layout, camera viewpoint, robot initial state, language instruction, lighting, background texture, sensor noise. Findings from the abstract:
- Success can drop from about 95% to below 30% under modest perturbations; camera viewpoint and robot initial state hurt most.
- Models are largely **insensitive to language variations**, and may ignore the instruction altogether.
- LeRobot has a LIBERO-Plus page in its docs (`huggingface.co/docs/lerobot/libero_plus`), so the perturbed environments may be usable directly. Verify.

**[LIBERO-PRO](../papers/2510.03827-libero-pro.pdf)** perturbs manipulated objects, initial states, instructions and environments. Models above 90% on standard LIBERO fall to 0% under its generalized setting, which suggests memorization of action sequences and layouts.

**Implications for us**
1. A System 1 that ignores its conditioning will make System 2 look useless. The training setup has to make the sub-goal necessary: for example, a scene where the initial image alone does not determine the action, or sub-goal-labelled phases of long-horizon tasks.
2. Perturbations from these papers are ready-made material for the error-recovery question. Cheap ones to implement ourselves: random initial robot joint offsets, object position shifts, mid-episode pushes of an object, camera noise, and scripted "drop the object" or "reset gripper" events.
3. Report results with confidence intervals. With small episode counts, a difference of a few points is noise.

## Out-of-distribution start-state tests (idea)

Idea: perturb the initial state more strongly than LIBERO's own variation and use the success drop as an out-of-distribution (OOD) test. It fits the finding above that robot initial state is one of the most damaging perturbations (LIBERO-Plus), and it is cheap because it only changes the reset.

**What the default variation is** (measured on libero_spatial task 0, 50 stored initial states, our own run with LeRobot 0.6.1):
- Robot joints: std about 1 degree per joint, range 4 to 6 degrees per joint.
- Gripper position: range about 4 to 5 cm per axis, still 4 to 5 cm after the 10 settle steps.
- Objects: std about 3 cm.
- Robot start poses therefore look the same in videos. LeRobot also uses stored state `i` for env `i`, so one episode per task always means state 0 unless the index is randomised (see the Colab skill's random-state launcher).

**Proposed test**
1. Reset to a stored initial state as usual, then add a seeded offset to the robot joint angles before the settle steps. Severity levels, for example 0 (baseline), 5, 10, 20 and 30 degrees maximum per joint. These magnitudes are a starting guess, not tested. Check what LIBERO-Plus uses for its robot-initial-state dimension and align the levels if that is documented.
2. Optionally also shift object positions by a seeded offset (cm levels), as a second axis.
3. Reject or clip offsets that start in collision with the table or objects, and log the rejection rate per level.
4. Use the same seeds and offsets across all conditions (Full, Static, Frozen, ...) so differences come from the System 2 interface and not from the draws.
5. Report success versus severity as a curve with confidence intervals, plus the baseline at level 0.

**Why it is useful for the ablation**
- Large offsets make memorised action sequences fail, so a System 2 that re-plans from the current image has room to help (links to Q2 below).
- If success falls to near zero for the frozen or static condition but not for Full, that is evidence for System 2. If all conditions fall equally, the bottleneck is System 1 execution (Q3).

**Caveats**
- This is not the standard benchmark protocol. Do not compare these numbers with published LIBERO success rates, and report the baseline run in the same setup.
- Episode budget: SmolVLA took about 70 s per episode on Colab and 107 to 164 s on the laptop CPU, and one run of 10 episodes per suite gave 70, 80, 100 and 70 % (spatial, object, goal, libero_10). With that few episodes only large effects are visible, so use few severity levels and more episodes per level.
- Implementation point: the patch must run in the process that creates the env (sync envs, or a launcher that patches the env class). The Colab skill shows how this was done for the random initial state.

## Ablation plan

Maps to the three required questions in [CHALLENGE.md](../../CHALLENGE.md).

### Q1. Counterfactual: is System 2 responsible?

Same System 1 weights wherever possible; change only what System 2 provides.

| Condition | System 2 output given to System 1 |
|---|---|
| Full | Predicted sub-goal, updated as the episode progresses |
| No System 2 (naive) | Static embedding of the initial task instruction |
| Frozen | Sub-goal from the first call, never updated |
| Shuffled | Sub-goal from a different episode or task (tests whether System 1 reads it) |
| Corrupted | Random or wrong-phase sub-goal |
| Oracle | Ground-truth sub-goal from demonstration segmentation (upper bound on System 2) |
| Lower update rate | Same as Full but System 2 called every N steps, N varied |

Metrics: success rate, steps to success, task progress (fraction of sub-goals completed), with per-task breakdown.

Reading the results: Full much greater than Static means System 2 matters. Shuffled or Corrupted much worse than Full means System 1 actually uses the interface. If Shuffled is close to Full, System 1 ignores System 2, and the effect is not proven.

### Q2. Error recovery

Inject perturbations at a fixed time (e.g. step 30% into the episode), identically across conditions:
- Object displaced or knocked away.
- Grasped object dropped.
- Robot pushed to an off-distribution pose.

Compare recovery rate (success after perturbation) for Full vs Static vs Frozen. The mechanism to look for: System 2 re-plans (e.g. goes back to "pick up the bowl") and System 1 follows. A System 2 that sees the new state each call can do this, and a frozen or static one cannot. Report also the time to recovery and how often System 2's output changed after the perturbation.

### Q3. Bottleneck analysis

Separate reasoning failures from execution failures:
- **System 2 accuracy.** Compare predicted sub-goal to ground truth (from segmentation) on held-out demonstration frames and during rollouts (instruction accuracy, as in Hi Robot).
- **System 1 given the oracle.** Run Oracle-sub-goal rollouts. If success is still low, System 1 is the bottleneck.
- **System 2 given perfect execution is not possible in sim without a scripted policy**, but a proxy is sub-goal completion rate: for each sub-goal, did System 1 reach it within K steps given the correct sub-goal?
- **Failure taxonomy.** For each failed episode, label it: wrong sub-goal (reasoning), correct sub-goal but not reached (execution), correct completion of all sub-goals but task not marked successful (interface or evaluation mismatch). Report counts.

Decision rule: compare success of Full, Oracle, and System 1 sub-goal reach rate. Gap Full to Oracle is the cost of System 2 errors. Gap Oracle to 100% is the cost of System 1 errors.

## Pitfalls to avoid

- Differences in training data or training steps between conditions. Train once, vary the interface at inference where possible, and note where retraining was needed.
- Evaluating on the same initial states used in training; LIBERO-PRO suggests this inflates results. Use held-out initial states or seeds where possible.
- Too few episodes. Decide the budget after measuring simulation speed and report uncertainty.
