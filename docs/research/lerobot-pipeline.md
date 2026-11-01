# LeRobot evaluation pipeline and dual-system integration

Back to [RESEARCH.md](../RESEARCH.md). Related: [libero-and-lerobot.md](libero-and-lerobot.md) (benchmark, commands), [dual-system-architectures.md](dual-system-architectures.md) (papers).

Source read: the installed packages in the conda env `lerobot-eval` (lerobot 0.6.1, robosuite 1.4.0, LIBERO from pip), in WSL under `~/miniconda3/envs/lerobot-eval/lib/python3.12/site-packages/`. The checkout `~/prjcts/lerobot` (detached HEAD, August 2026 commit) has identical files for the ones compared (`envs/libero.py`, `scripts/lerobot_eval.py`, `policies/smolvla/modeling_smolvla.py`, `processor/env_processor.py`, `envs/configs.py`), but it belongs to the older hackathon setup, not this project. Statements about robosuite internals marked *(memory)* come from my knowledge and were not found in the code that was read.

## 1. The evaluation loop

`rollout()` in `lerobot/scripts/lerobot_eval.py` is the whole pipeline. One iteration per environment step:

```
env obs (numpy, nested dict)
  → preprocess_observation        generic: numpy → torch, LeRobot key names
  → observation["task"] = env.call("task_description")      language instruction
  → env_preprocessor              LIBERO: LiberoProcessorStep
  → preprocessor                  policy: rename, batch dim, tokenize, device, normalize
  → policy.select_action(obs)     ONE action (B, action_dim)
  → postprocessor                 policy: unnormalize, to CPU
  → env_postprocessor             LIBERO: empty pipeline (identity)
  → env.step(action_numpy)
```

`policy.reset()` and `env.reset()` run once at the start of each rollout (episode). The loop runs until every env in the batch is done or `max_steps` is reached.

## 2. What moves the joints

The policy never outputs joint commands. `LiberoEnv.step` passes the 7-D action straight to a robosuite environment (`self._env.step(action)`): 6-D end-effector pose delta plus 1 gripper dimension. `control_freq=20` is set in `LiberoEnv.__init__`.

**Controller: OSC_POSE** (Operational Space Controller, Khatib 1987), `robosuite/controllers/osc.py`.
- Python/NumPy inside the simulation process. **Not ROS2**, no `ros2_control`, no hardware interface.
- Open source (robosuite, ARISE-Initiative). License: I believe MIT; the package metadata only says "UNKNOWN", so check the repo before citing.
- **Not Panda-specific.** It uses MuJoCo's Jacobian and mass matrix and works for any arm. Panda enters only through `default_panda.json` and the actuator ranges in the Panda `robot.xml`.

Per control step (20 Hz):
1. Action clipped to [-1, 1], scaled to ±0.05 m and ±0.5 rad (`osc_pose.json`: `output_max/min`).
2. `set_goal` turns it into a target pose. In relative mode (`use_delta=True`) the target is the *current* pose plus the delta *(memory, `set_goal` not re-read)*.

Per physics step (2 ms, `SIMULATION_TIMESTEP = 0.002`, so 0.05 / 0.002 = 25 steps per control step, loop in `robosuite/environments/base.py`). The goal stays fixed during the 25 sub-steps; torques are recomputed each time:
1. `F = kp·pos_err + kd·(−vel)`, same form for orientation (`kp=150`, `damping_ratio=1`).
2. Mapped through the task-space inertia `Λ = (J M⁻¹ Jᵀ)⁻¹`.
3. `τ = Jᵀ Λ F` + gravity/Coriolis compensation (`torque_compensation`).
4. Plus a nullspace term pulling toward the initial joint posture.
5. Torques go to MuJoCo.

**Limits that exist** (weak compared with a real robot):

| Limit | Value |
|---|---|
| Input clipping | goal moves at most 5 cm / 0.5 rad per step |
| Gain bounds | `kp_limits` [0, 300], `damping_ratio_limits` [0, 10] |
| Torque saturation | Panda motors `ctrlrange` ±80 Nm (joints 1 to 5), ±12 Nm (joints 6, 7) |
| Joint position limits | MuJoCo contact constraints |
| Not set | `position_limits`, `orientation_limits`, `interpolation` are all `null` |

There is **no velocity, acceleration or jerk limiting** and no trajectory generator (no Ruckig). The Franka's real torque-rate limits are not modelled. A policy that works in LIBERO needs smoothing before it runs on a real arm. Real-world counterpart: `libfranka` torque interface with a Cartesian-impedance controller such as the one in `franka_ros2` *(memory)*.

`control_mode` (`relative` / `absolute`) toggles `robot.controller.use_delta` in `LiberoEnv.reset()`. It must match how the policy was trained.

## 3. Time: simulated vs wall clock, and asynchrony

- **In `lerobot-eval` there is no real-time requirement.** The loop blocks: observe, infer, step. "20 Hz" means 0.05 s of *simulated* time per step. If a render plus inference takes 0.2 s of wall time, the episode just runs slower; the policy still sees one frame per 0.05 s of sim time. The sim does not advance while the policy computes.
- **`use_async_envs` is not latency hiding.** It switches `SyncVectorEnv` to `AsyncVectorEnv` (one subprocess per env), and only for `n_envs > 1`. With `eval.batch_size=1` it does nothing.
- **Real-time asynchrony** exists only in `lerobot/async_inference/` (`policy_server.py`, `robot_client.py`), meant for real robots. It overlaps inference with execution via a client-side action queue (section 5).
- **Action queue inside the policy.** `SmolVLAPolicy.select_action` calls the model only when its action queue is empty and pops one action per step. Per-step cost is therefore very uneven: most steps are almost free, every `n_action_steps`-th step runs the full VLM.

Measured on this machine (from the `wsl-lerobot-libero-eval` skill): with `n_action_steps=10`, about 70 % of an episode is `select_action` (CPU inference), about 12 % is `LiberoEnv.step` (sim plus two camera renders). Rendering is not the bottleneck.

To model a slow System 2 in a synchronous eval, simulate latency in steps: call System 2 every `k` steps while System 1 keeps stepping on a cached sub-goal, or apply a new sub-goal only `latency × 20` steps after it was requested.

## 4. Policy API: `select_action` vs `predict_action_chunk`

For SmolVLA:
- **`predict_action_chunk(batch)`** returns a fresh chunk `(B, chunk_size, action_dim)`. Does not touch the action queue, does not truncate to `n_action_steps`. Used for training-time evaluation, the async server and RTC. Not called in the `lerobot-eval` path (matches the profile).
- **`select_action(batch)`** is the stateful wrapper, returns one action `(B, action_dim)`: (1) if the action queue is empty, call `_get_action_chunk` and keep the first `n_action_steps` actions in a `deque`; (2) pop one action. Refill and pop happen in the same call, so the queue is never empty at the pop.

**Observation queues: SmolVLA has none.** `SmolVLAPolicy.reset()` creates `_queues = {ACTION: deque(...)}` only. Both methods call `populate_queues(..., exclude_keys=[ACTION])`, which skips every key that is not already in `_queues`. So the call does nothing for SmolVLA, and the policy sees only the current frame. (Diffusion Policy, by contrast, creates `deque(maxlen=n_obs_steps)` queues for state and images and stacks the history.) If System 1 needs history, create the queues yourself in `reset()`.

## 5. Async client: aggregation and empty queue

Only relevant for real-time deployment (`async_inference`), not for `lerobot-eval`.

`aggregate_fn_name` (registry in `async_inference/configs.py`), `old` = already queued action, `new` = incoming:

| Name | Formula |
|---|---|
| `weighted_average` | `0.3·old + 0.7·new` |
| `latest_only` | `new` |
| `average` | `0.5·old + 0.5·new` |
| `conservative` | `0.7·old + 0.3·new` |

`_aggregate_action_queues` merges per timestep index: incoming actions at or before the last executed step are dropped; timesteps not yet queued are appended; timesteps in both are blended element-wise. The queue is **rebuilt from the incoming chunk only**, so queued actions the new chunk does not cover are discarded. The blend is per index (no time warping) and also averages the **gripper** dimension, which can produce an intermediate gripper command between "open" and "close".

Empty queue: `control_loop` sends an action only `if self.actions_available()`. When empty, no command is sent and the robot holds its last setpoint (position-controlled arm: stands still, does not drift). A new request goes out when `qsize / action_chunk_size <= chunk_size_threshold` (default 0.5); `must_go` forces processing when the queue is empty. In `lerobot-eval` the queue cannot run out.

## 6. LIBERO-specific pre- and post-processing

`get_env_processors()` in `envs/configs.py` returns `Pipeline([LiberoProcessorStep()])` as pre and `Pipeline([])` as post (identity).

`LiberoProcessorStep` (`processor/env_processor.py`):
- **Images:** `torch.flip(img, dims=[2, 3])` (flip H and W, a 180° rotation) to match the `HuggingFaceVLA/libero` dataset orientation.
- **State:** builds `observation.state` as 8-D float32 = eef position (3) + quaternion converted to axis-angle (3) + gripper joint positions (2). Joint positions/velocities, rotation matrix and gripper velocity are dropped.

Also LIBERO-specific, inside `LiberoEnv`: 10 no-op settle steps after reset (`num_steps_wait`), the `use_delta` toggle, camera name mapping, stored initial states. The `--rename_map` (`image`→`camera1`, `image2`→`camera2`) is a policy-side concern for SmolVLA.

## 7. Parallelism: `SyncVectorEnv` vs `AsyncVectorEnv`

- **`SyncVectorEnv`:** no parallelism. `step` is a Python `for` loop over the sub-envs in one process. Each sub-env has its own full MuJoCo state (model, data, renderer), run one after another.
- **What is batched** is the policy forward pass: B observations are stacked and the network runs once.
- **`AsyncVectorEnv`:** one subprocess per env, each holding its own full sim state; only actions and observations cross the process boundary. Parallelises `env.step` including rendering across cores. Policy inference stays in the main process.
- Autoreset mode is `NEXT_STEP`; finished envs are masked by the cumulative `done`.
- With `batch_size=1` (this laptop) neither matters.

## 8. Policy contract and how to plug in a dual system

Four pieces, built by `policies/factory.py`:

| Piece | Contract |
|---|---|
| Config | `PreTrainedConfig` subclass registered with `@PreTrainedConfig.register_subclass("name")`; declares features, `chunk_size`, `n_action_steps` |
| Policy | `PreTrainedPolicy` (an `nn.Module`): `reset()`, `forward(batch) → (loss, info)`, `predict_action_chunk(batch)`, `select_action(batch)` |
| Preprocessor | `PolicyProcessorPipeline`: list of `ProcessorStep`s, dict → dict. SmolVLA: rename, add batch dim, newline task, tokenizer, to device, normalize |
| Postprocessor | `PolicyProcessorPipeline` over the action tensor. SmolVLA: unnormalize, to CPU |

Env-specific conversions go in `env_preprocessor` / `env_postprocessor`; keep model logic in the policy pipelines.

**Suggested design: one policy class wrapping System 2 and System 1**, so `lerobot-eval` stays unchanged:

```python
@PreTrainedConfig.register_subclass("dual_system")
@dataclass
class DualSystemConfig(PreTrainedConfig):
    system2_every: int = 40
    system1_ckpt: str = "lerobot/smolvla_libero"

class DualSystemPolicy(PreTrainedPolicy):
    config_class = DualSystemConfig
    name = "dual_system"

    def reset(self):
        self.system1.reset()
        self._t = 0
        self._subgoal = None
        self.trace = []                  # [(step, sub-goal text)] per episode

    @torch.no_grad()
    def select_action(self, batch, **kw):
        if self._t % self.config.system2_every == 0 or self._subgoal is None:
            self._subgoal = self.system2(batch["observation.images.image"], batch["task"])
            self.trace.append((self._t, self._subgoal))
        batch = {**batch, "task": self._subgoal}
        self._t += 1
        return self.system1.select_action(batch)
```

Points to handle:
- **Tokenization order.** The tokenizer step runs in the preprocessor, *before* `select_action`, so `batch["task"]` is already tokenized when System 1 receives it. Either re-tokenize the new sub-goal inside the policy, or use a preprocessor without the tokenizer step and tokenize inside.
- **`trace`.** Not LeRobot API; it is the per-episode log of System 2 calls the ablation and bottleneck analysis need. It is reset in `reset()` because `rollout()` calls `policy.reset()` at the start of every episode (same for `_t` and `_subgoal`). Nothing collects it automatically: write it out after each episode (callback or a small custom eval script).
- **Batch size.** The sketch assumes `batch_size=1`; for B>1 keep one sub-goal and counter per env.
- **History.** Create observation queues yourself in `reset()` if needed (section 4).
- **Ablations.** The text interface makes the sub-goal swap the single place to freeze, shuffle, corrupt or oracle-replace the sub-goal.
- **Training System 1.** Train on sub-goal-conditioned data, otherwise it ignores the text and the ablation shows no effect (see takeaway 4 in [RESEARCH.md](../RESEARCH.md)).
- **Loading.** Registration plus `from_pretrained` / `save_pretrained` lets `--policy.path` load the class. Alternative: a thin custom script that calls `rollout()` directly with your policy object.
- **Edit the installed package carefully.** This project uses the pip-installed copy in `lerobot-eval`. Do not edit the `~/prjcts/lerobot` checkout, which the old env `lerobot` (0.1.0, editable) points to. Put your own policy code in a separate package instead of patching site-packages.

## Not verified

- `PreTrainedPolicy` abstract methods beyond those found by grep; check `policies/pretrained.py` before writing the class.
- `robosuite` `set_goal` (relative-mode target) and the license text.
- Behaviour of zero-delta actions in LIBERO (expected: hold pose with small steady-state error).
- Whether `lerobot/smolvla_libero` is the right System 1 checkpoint for a sub-goal-conditioned variant.
- The default value of `aggregate_fn_name` in `PolicyServerConfig`/`RobotClientConfig`.
