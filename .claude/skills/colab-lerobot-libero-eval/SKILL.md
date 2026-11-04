---
name: colab-lerobot-libero-eval
description: Evaluate pretrained LIBERO policies (SmolVLA, pi0, pi0.5) with LeRobot's lerobot-eval on a Google Colab T4 driven through the colab-mcp server. Use when setting up or debugging this evaluation, or when lerobot-eval fails on Colab (egl_probe build, OOM exit 137, gated PaliGemma tokenizer, async env hangs).
---

# LeRobot LIBERO evaluation on Colab (T4) via colab-mcp

Status of what was actually run (2026-10-31, LeRobot 0.6.1, Colab Python 3.13, Tesla T4 15 GB, 12 GB host RAM, 2 CPU cores):

| Policy | Checkpoint | Status |
|---|---|---|
| SmolVLA | `lerobot/smolvla_libero` | Works. Smoke test libero_spatial task 0, 2 episodes: 2/2 success, about 71 s per episode |
| pi0.5 | `lerobot/pi05_libero_finetuned_v044` | Works through `dual_system` since 2026-11-01, with the HF token set: about 0.35 s per sim step, about 10.3 GB GPU. Results are in `docs/EXPERIMENTS.md`. Not run through stock `lerobot-eval` |
| pi0 | `lerobot/pi0_libero_finetuned_v044` | Loads on the GPU with the launcher below. Not run |

SmolVLA, 1 episode per task from a random stored initial state (section 5b), 10 tasks per suite, `n_action_steps=10`, `--seed=1000`, `INIT_SEED=0`:

| Suite | Success | Time per episode |
|---|---|---|
| libero_spatial | 7/10 (70 %) | 104 s |
| libero_object | 8/10 (80 %) | 69 s |
| libero_goal | 10/10 (100 %) | 67 s |
| libero_10 | 7/10 (70 %) | 147 s |

Only 10 episodes per suite, one random state per task, so differences below about 30 percentage points are within sampling noise. These are not the official 50-state protocol and are not comparable with published numbers. The times come from the eval's `eval_s` and include the env setup.

Not done yet: a working batch_size above 1 (the batch_size=5 run hung or ran out of RAM, section 7), pi0 episodes.

Other LIBERO repos seen on the Hub (not tested): `lerobot/pi05_libero_finetuned_quantiles_v044`, `lerobot/pi05-libero`, `lerobot/pi0_libero_base`, `lerobot/pi05_libero_base`, `lerobot/smolvla_libero_plus`. The `lerobot/pi05_libero_finetuned` name in `docs/research/policies-and-encoders.md` was not in the Hub listing; use the `_v044` names.

## 1. Driving Colab with the MCP server

1. The user opens a Colab notebook in the browser and sets Runtime > Change runtime type > GPU (T4). The MCP server cannot change the runtime. Changing it may restart the session.
2. Load the deferred tool schema (`ToolSearch select:mcp__colab-mcp__open_colab_browser_connection`), then call it. It returns `true` and unlocks `get_cells`, `add_code_cell`, `update_cell`, `run_code_cell`, `delete_cell`, `move_cell`, `add_text_cell`. These also need loading with ToolSearch (query `colab`).
3. `add_code_cell` returns a cell id. `run_code_cell` blocks and returns the output. Reuse a cell with `update_cell` plus `run_code_cell` instead of adding many cells.
4. Verify the GPU first: `!nvidia-smi`. If it is missing, the runtime is still CPU.
5. Long jobs: start with `nohup ... > log 2>&1 &` and poll the log in short cells. A blocking `sleep` cell can be interrupted by the user and loses the output. Never print the progress bar (tqdm writes thousands of `\r` lines); read `tail -c` of the log with `\r` replaced by newlines, or filter out "Running rollout".
6. `%%bash` cells raise `CalledProcessError` with a huge traceback on failure. Prefer `tail` or `grep` on the command output, and print the exit code (`echo "exit code: $?"`). Plain Python cells with `subprocess.run(..., capture_output=True)` give cleaner errors and are fine too.
7. When the browser tab disconnects, call `open_colab_browser_connection` again. In one observed case (runtime up about 2 h) `/content`, the installed packages and the background queue all survived, and the queue and a profile run that was waiting behind it finished normally. The notebook cells are kept as well. This is one observation, not a guarantee.
8. Colab files are not persistent. Copy results (`/content/eval_*`) to Drive or download them before the session ends.

## 2. Install (cell 1)

```bash
%%bash
apt-get -qq update > /dev/null 2>&1
apt-get -qq install -y cmake build-essential libegl1-mesa-dev libgl1-mesa-dev libglu1-mesa-dev libosmesa6-dev libglfw3 ffmpeg > /dev/null 2>&1
export CMAKE_POLICY_VERSION_MINIMUM=3.5
pip install -q setuptools wheel
pip install -q --no-build-isolation hf-egl-probe egl_probe
pip install -q "lerobot[libero,pi,smolvla]"
which lerobot-eval
```

Why the extra steps:
- `pip install "lerobot[libero,...]"` fails with `Failed building wheel for egl_probe / hf-egl-probe`. The build runs `cmake --version` inside pip's isolated build env and that call returns non-zero. Installing the two packages first with `--no-build-isolation` (system cmake 3.31 on PATH) works.
- Do not pass `--no-build-isolation` to the whole lerobot install.

## 3. LIBERO first-import prompt

`import libero.libero` asks "Do you want to specify a custom path for the dataset folder? (Y/N)" on first use. Under `lerobot-eval` this raises `EOFError`. Run once:

```bash
echo N | python -c "import libero.libero"
```

This writes the LIBERO config; later runs do not prompt.

## 4. Environment

```bash
export MUJOCO_GL=egl
```

Only EGL was tried. Whether it actually uses hardware rendering on Colab is unverified (see section 8).

## 5. SmolVLA command (verified)

The checkpoint expects `camera1` / `camera2`, while the LIBERO env emits `image` / `image2`, so pass a rename map. pi0 and pi0.5 do not need it (their inputs are `image`, `image2`).

```bash
lerobot-eval \
  --policy.path=lerobot/smolvla_libero \
  --policy.n_action_steps=10 \
  --env.type=libero --env.task=libero_spatial --env.task_ids='[0]' \
  --eval.batch_size=1 --eval.n_episodes=2 \
  --rename_map='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}' \
  --output_dir=/content/eval_smoke
```

- `--env.task` is the suite (`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`). `--env.task_ids` selects tasks inside it.
- `n_action_steps=10` follows `docs/research/policies-and-encoders.md` (said to match OpenPI). The checkpoints default to 50, so always pass it.
- The result is printed at the end as `Overall Aggregated Metrics` with `pc_success`, plus per-task and per-group blocks. Videos go to `<output_dir>/videos/`.
- The eval process holds about 1 GB of GPU memory. GPU utilisation reads about 0 % most of the time: the policy runs once per 10 env steps and the rest is CPU-bound simulation and rendering.

## 5b. Random initial state per task (launcher)

By default LeRobot gives env `i` the stored LIBERO initial state `i` (`init_state_id = episode_index`, then `+ n_envs` per reset). One episode per task therefore always uses state 0. The 50 stored states per task vary the robot pose and the objects. Plain `--env.init_states=false` varies only the objects: the robot joints were identical across seeds (std 0.0), objects varied (std about 1.1 cm). The stored states vary both: joints std about 1 degree (range 4 to 6 degrees), gripper range about 4 to 5 cm, objects std about 3 cm (measured on libero_spatial task 0). So the robot start pose varies only a little.

Launcher `/content/run_eval_random.py`: same CLI as `lerobot-eval`, but each `LiberoEnv` takes a seeded random stored state and logs the chosen index.

```python
import json, os, sys, zlib
import numpy as np
from lerobot.envs import libero as L

SEED = int(os.environ.get("INIT_SEED", "0"))
LOG = os.environ.get("INIT_LOG", "/content/init_ids.jsonl")
_orig = L.LiberoEnv.__init__

def _init(self, *a, **k):
    _orig(self, *a, **k)
    if self._init_states is not None:
        key = zlib.crc32(f"{self.task}-{self.episode_index}-{SEED}".encode())
        self.init_state_id = int(np.random.default_rng(key).integers(len(self._init_states)))
        with open(LOG, "a") as f:
            f.write(json.dumps({"task": self.task, "init_state_id": self.init_state_id, "n_states": len(self._init_states)}) + "\n")
L.LiberoEnv.__init__ = _init

from lerobot.scripts.lerobot_eval import main
sys.exit(main())
```

Run it with `--eval.batch_size=1 --eval.n_episodes=1 --eval.use_async_envs=false --seed=1000`. Sync envs are required: with the default async envs the env is built in a subprocess and the patch does not apply. Change `INIT_SEED` for a different draw. For a multi-suite queue, loop over the suites in a bash script started with `subprocess.Popen(..., start_new_session=True)`, one output dir and log per suite, and append `"$SUITE exit $?"` to a queue log. Videos are written at 80 fps (the env's `render_fps`) while control runs at 20 Hz, so playback is 4x too fast. That is cosmetic.

Episodes end on success, so a successful episode has far fewer than the 280 allowed steps (the smoke-test videos had 78 to 83 frames). With `n_action_steps=10` that is about 8 policy calls per episode. The env also runs 10 no-op settle steps after each reset that are not in the video.

## 6. pi0 / pi0.5

Save every run under its own `YYYYMMDD-HHMM` folder (`python -m infra newrun`, see the `colab-startup` skill).

The checkpoints set `compile_model: true` with `max-autotune`, which stalled the first episode for over 10 minutes on the T4. `LeRobotSystem1` turns compilation off; a stock `lerobot-eval` run needs `--policy.compile_model=false`.

### 6a. Host OOM (exit code 137) and the launcher

Stock `lerobot-eval` is killed with exit code 137 and no traceback right after "Making policy". The pi0/pi0.5 backbone is built in fp32 on the CPU, then a 7.0 to 7.5 GB state dict is loaded, which exceeds the 12 GB host RAM. Building the backbone in bf16 on the CPU was not enough. Building it directly on the GPU worked: host RAM stayed at about 3 GB. Use this launcher (same CLI as `lerobot-eval`):

```python
%%writefile /content/run_eval.py
import functools, sys
import torch
from lerobot.policies.pi0 import modeling_pi0
from lerobot.policies.pi05 import modeling_pi05

def gpu_bf16_init(cls):
    orig = cls.__init__
    @functools.wraps(orig)
    def init(self, *a, **k):
        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.device("cuda"):
                orig(self, *a, **k)
        finally:
            torch.set_default_dtype(prev)
    cls.__init__ = init

gpu_bf16_init(modeling_pi0.PaliGemmaWithExpertModel)
gpu_bf16_init(modeling_pi05.PaliGemmaWithExpertModel)

from lerobot.scripts.lerobot_eval import main
sys.exit(main())
```

Run it as `python /content/run_eval.py --policy.path=lerobot/pi05_libero_finetuned_v044 --policy.n_action_steps=10 --env.type=libero ...` (no rename map).

Caveat: during that run `nvidia-smi` showed 14.5 GB of 15 GB GPU memory used. The bf16 default-dtype patch may not have taken effect for every submodule (the backbone may be fp32 on the GPU). This is unverified and leaves little headroom on the T4. If inference runs out of GPU memory, check the parameter dtypes after construction.

### 6b. Gated tokenizer (needs the user once)

The pi0 / pi0.5 processors load the tokenizer from `google/paligemma-3b-pt-224`, a gated Hugging Face repo. Without a token the run fails with `ValueError: Failed to instantiate processor step 'tokenizer_processor' ... gated repo`. The user must:
1. Accept the licence at https://huggingface.co/google/paligemma-3b-pt-224 with their account.
2. Create a read token and add it in Colab (key icon, left sidebar) as secret `HF_TOKEN` with Notebook access on.

Then in a cell, without printing the token:

```python
from google.colab import userdata
import os
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
```

Never paste the token into a cell or the chat. The same token also removes the "unauthenticated requests" warning. No ungated mirror of the tokenizer was tried.

## 7. Compute budget

- SmolVLA: about 71 s per episode at batch_size=1 (2 episodes took 143 s).
- Full sweep (4 suites x 10 tasks) at 5 episodes is 200 episodes, which is hours. Scope it: one suite, a task subset, or few episodes per task.
- `--eval.batch_size` is the number of parallel envs, and with async envs (default, `async=True` in the log) it is also the number of worker processes, each loading its own MuJoCo/robosuite copy.
- `batch_size=5` with 10 tasks and 5 episodes was launched in the background and looked unhealthy: host RAM at 12.1 / 12.9 GB, no rollout progress after 80 s, and a gymnasium warning "Calling `close` while waiting for a pending call to `reset`". It had not finished or been confirmed dead when last checked. Do not rely on batch_size=5.
- Untested options: `batch_size=2`, or `--eval.use_async_envs=false` (all envs step one after another in the main process: slowest, lowest memory). Colab has only 2 cores, so more than 2 workers likely does not help.
- Check for a stale job before relaunching: `pgrep -af lerobot-eval`, and kill it if needed.

## 7b. Where the time goes on Colab (profiled)

One successful libero_spatial task 0 episode, default init state 0, sync env, `batch_size=1`, `n_action_steps=10`, T4. The profile wraps the hot calls with timers (CUDA-synchronised) via a launcher with the same CLI as `lerobot-eval`. Total process wall 92.5 s (includes startup and model loading, not timed separately). The per-call table:

| Call | Total | Calls | Mean |
|---|---|---|---|
| `LiberoEnv.step` (simulation plus two camera renders) | 38.8 s | 78 | 498 ms |
| robosuite `ControlEnv.step` (inside the above, incl. the 10 settle steps) | 43.8 s | 88 | 498 ms |b
| `SmolVLAPolicy.select_action` (all calls) | 5.4 s | 78 | 69 ms |
| `SmolVLAPolicy._get_action_chunk` (a new chunk) | 5.2 s | 8 | 651 ms |
| `VLAFlowMatching.embed_prefix` (vision and language) | 2.5 s | 8 | 315 ms |
| `VLAFlowMatching.denoise_step` (10 per chunk) | 2.3 s | 80 | 28 ms |
| `SmolVLAPolicy.prepare_images` | 0.0 s | 8 | 1.6 ms |

- **On Colab the env step dominates, not the policy.** Env steps are 39 s against 5.4 s of policy time. The T4 computes a 50-step chunk in about 0.65 s.
- **Rendering is the cost, and it is done on the CPU.** With camera observations off, a step takes 30.5 ms (physics only). With the two cameras it takes about 400 ms at 64x64 and still 411, 434 and 490 ms at 128, 256 and 360 pixels. So about 370 ms per step is a nearly fixed per-frame cost, almost independent of resolution. The reason for the fixed cost was not found.
- **By default the GL renderer is llvmpipe (Mesa software, LLVM 20.1.2), not the T4.** The container has `NVIDIA_DRIVER_CAPABILITIES=compute,utility` (no graphics), the only EGL vendor file is `50_mesa.json`, and no NVIDIA EGL/GL libraries (`libEGL_nvidia` etc.) are present. Installing them fixed it, see section 7c. The runtime has 2 vCPUs (Intel Xeon 2.0 GHz); torch reported 1 thread.
- **Comparison with the laptop (WSL, CPU only):** env step 166 ms (3x faster than Colab, also on software rendering with 4 physical cores) but the policy needs about 9 s per chunk on the CPU (about 14x slower). The WSL skill has that profile.
- **Reset** took 20.8 s with llvmpipe (6.6 s with the T4 renderer). Measured parts of `LiberoEnv.reset` with llvmpipe: env construction 1.9 s, hard reset 3.1 s (done **twice** on the first reset: once inside `_ensure_env`, once in `LiberoEnv.reset`), `set_init_state` 0.5 s, and the 10 settle steps 4.8 s (each is a full camera render). That is about 13 s of the 20.8 s; the remaining 7 s is unexplained, probably first-time initialisation (not verified). A later reset needs one hard reset, so about 8 s. `--env.hard_reset=false` (soft reset, requires `init_states=true`) took 0.5 s per reset in a direct test, but LeRobot warns that the settle steps can make observations differ from a hard reset, so it was not used in the evaluations.
- **batch_size above 1** did not help with llvmpipe (rendering was CPU-bound on 2 vCPUs). Not re-tested with the T4 renderer.

Profile recipe: wrap `LiberoEnv.step/reset`, `libero.libero.envs.env_wrapper.ControlEnv.step`, `SmolVLAPolicy.select_action/_get_action_chunk/prepare_images` and `VLAFlowMatching.embed_prefix/sample_actions/denoise_step` with `torch.cuda.synchronize()` before and after each call, print a table at exit, and use `--eval.use_async_envs=false` so the patched classes run in the timed process. `predict_action_chunk` exists but is not called in the eval path. To isolate rendering, time `OffScreenRenderEnv(..., use_camera_obs=False)` steps against the default.

## 7c. Hardware rendering on the T4 (works, about 12x faster steps)

With the user's OK, the NVIDIA EGL userspace libraries were installed into a private folder. Nothing in the system directories and no kernel module is touched. The libraries must match the loaded driver exactly (`nvidia-smi` showed 580.82.07), so apt's `libnvidia-gl-580` (580.196.07) was not used. The exact installer is on NVIDIA's download server (396 MB). `/content` is not persistent, so redo this after a runtime reset (and check the driver version first, it may differ on another runtime):

```python
import subprocess
V = subprocess.run("nvidia-smi --query-gpu=driver_version --format=csv,noheader", shell=True, capture_output=True, text=True).stdout.strip()
sh = lambda c: print(subprocess.run(c, shell=True, capture_output=True, text=True).stdout[-1500:])
sh(f"mkdir -p /content/nvidia && cd /content/nvidia && curl -s -o installer.run https://download.nvidia.com/XFree86/Linux-x86_64/{V}/NVIDIA-Linux-x86_64-{V}.run && ls -l installer.run")
sh("cd /content/nvidia && sh installer.run --extract-only --target /content/nvidia/ex > extract.log 2>&1; echo extract exit $?")
sh(f"""mkdir -p /content/nvidia/gl && cd /content/nvidia/gl &&
for l in libEGL_nvidia libnvidia-eglcore libnvidia-glcore libnvidia-glsi libnvidia-gpucomp libnvidia-tls; do cp -f ../ex/$l.so.{V} . ; done &&
ln -sf libEGL_nvidia.so.{V} libEGL_nvidia.so.0 && cp ../ex/10_nvidia.json .""")
```

Start every process that renders (the eval, the profiler, launchers, queue scripts) with these environment variables. They must be set when the process starts:

```bash
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export LD_LIBRARY_PATH=/content/nvidia/gl:$LD_LIBRARY_PATH
export __EGL_VENDOR_LIBRARY_FILENAMES=/content/nvidia/gl/10_nvidia.json
```

Check: `glGetString(GL_RENDERER)` must print `Tesla T4/PCIe/SSE2` (vendor NVIDIA Corporation). Without the variables it prints `llvmpipe`. At exit a harmless `TypeError: 'NoneType' object is not callable` from MuJoCo's renderer cleanup appears.

Measured (libero_spatial task 0, default init state 0, sync env, batch_size 1, same profiler as 7b):

| | llvmpipe (default) | T4 EGL |
|---|---|---|
| Env step, 64x64 cameras | 394 ms | 33 ms |
| Env step, 360x360 cameras | 482 ms | 33 ms |
| `LiberoEnv.step` in a real episode | 498 ms | 36 ms |
| `LiberoEnv.reset` | 20.8 s | 6.6 s |
| `_get_action_chunk` | 651 ms | 556 ms |
| Episode time (`eval_s`) | not recorded for this run | 15.5 s (79 steps, success) |
| Process wall | 92.5 s | 40.1 s |

- The policy is now the larger cost per step: 8 chunk computations of about 0.55 s (4.4 s of 15.5 s), plus the reset.
- Frames from the two renderers looked equivalent (video mean pixel 117.1 against 117.7, std 52.2 against 52.1, both episodes succeeded). The renderers are different implementations, though, so images are not bit-identical. **All success rates in the table at the top of this file were produced with llvmpipe.** Re-run them with the T4 renderer before comparing against later results, and note which renderer each run used.
- The single timing run has one episode per configuration, so treat the ratios as indicative.

## 8. Open questions

- Why the camera render cost is almost independent of image size on llvmpipe.
- The unexplained 7 s of the first reset with llvmpipe.
- Whether `batch_size` above 1 helps now that rendering is on the GPU, and what the RAM limit allows.
- Whether the policy's success rates change with the T4 renderer (different rasterisation). Only one episode was compared.
- Whether the pi0 / pi0.5 numbers match the OpenPI / LeRobot docs (pi0.5 about 97 %). Those figures in `docs/research/policies-and-encoders.md` are from the docs, not from our runs.
- Success rate of pi0 and pi0.5 under correct, wrong and empty prompts (the planned first experiment) has not been run. Only atomic schedules ran on pi0.5.
