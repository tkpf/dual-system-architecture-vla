# LIBERO and LeRobot

Back to [RESEARCH.md](../RESEARCH.md).

## Benchmark

Paper: [LIBERO](../papers/2306.03310-libero.pdf). Docs: <https://huggingface.co/docs/lerobot/libero>. Upstream: <https://github.com/Lifelong-Robot-Learning/LIBERO>.

| Suite | LeRobot name | Tasks | Focus |
|---|---|---|---|
| LIBERO-Spatial | `libero_spatial` | 10 | Same objects, different spatial relations |
| LIBERO-Object | `libero_object` | 10 | Different objects |
| LIBERO-Goal | `libero_goal` | 10 | Same scene, different goals. **Language really matters here** |
| LIBERO-Long | `libero_10` | 10 | Long-horizon, multi-step tasks. **Best fit for dual-system** |
| LIBERO-90 | `libero_90` | 90 | Short-horizon pretraining set |

Suggested slice: `libero_10` for the long-horizon story, optionally `libero_goal` as a second suite where the instruction decides the behavior. Pick a handful of tasks per suite, not all 10, given CPU simulation speed.

## Datasets on Hugging Face

From the LeRobot docs and my search (check the dataset cards for the exact features and format):

- `lerobot/libero`: dataset recommended in the LeRobot docs.
- `HuggingFaceVLA/libero`: preprocessed copy used in the SmolVLA and GR00T examples.
- `physical-intelligence/libero`: the original Physical Intelligence conversion.
- The challenge text also mentions per-suite datasets such as `lerobot/libero_10` and `lerobot/libero_spatial`. These names were not confirmed in my search, so check that they exist.

## Install and evaluate (from LeRobot docs)

```bash
# inside WSL2 Ubuntu; LIBERO requires Linux
pip install -e ".[libero]"          # from a lerobot checkout
export MUJOCO_GL=egl                # headless; try osmesa if EGL fails without a GPU

lerobot-eval \
  --policy.path="<policy-id>" \
  --env.type=libero \
  --env.task=libero_10 \
  --env.task_ids='[0,1,2]' \
  --eval.batch_size=1 \
  --eval.n_episodes=10
```

- `--env.task` picks the suite (comma-separated for several). `--env.task_ids` restricts task indices. `--eval.n_episodes` is episodes per task.
- `--env.control_mode=relative|absolute`: **must match how the policy was trained**.
- With fixed initial states you can use soft resets (`--env.init_states=true --env.hard_reset=false`) to avoid rebuilding the simulator every episode. This speeds up evaluation noticeably *(verify flags against the installed version)*.
- The docs recommend 10 episodes per task over the four standard suites for the default protocol.

## Reference numbers

The LeRobot docs report a finetuned pi0.5 checkpoint around 92 to 96% on LIBERO-10, with the two doc pages disagreeing slightly. Use these only as an upper-bound anchor, not as a target.

## Practical notes

- The challenge says to use a custom **System 2 inside the eval loop**. `lerobot-eval` evaluates a single `policy`, so the dual system should be wrapped as one LeRobot policy class (System 2 invoked inside `select_action`), or use a custom evaluation script built on the LeRobot LIBERO env. Look at LeRobot's "Adding a Policy" docs.
- Log per-episode: success, step count, System 2 outputs, and System 2 call timestamps. The ablation needs step counts and the System 2 trace for the bottleneck analysis.
- Render videos with the environment's `rgb_array`, and overlay the current sub-goal text on the frames for the deliverable.
- Budget: measure seconds per episode first, then decide episodes per condition. The ablation needs at least 5 to 6 conditions, multiplied by tasks and seeds.
