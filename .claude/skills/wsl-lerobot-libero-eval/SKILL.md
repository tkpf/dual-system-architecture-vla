---
name: wsl-lerobot-libero-eval
description: Run LeRobot's lerobot-eval on LIBERO with the pretrained SmolVLA policy locally in WSL2 Ubuntu 22.04 (CPU only, no NVIDIA GPU), in a dedicated conda env. Use when setting up or debugging the local evaluation pipeline, when an episode is slow, or when MuJoCo rendering, conda, or the WSL command line misbehave.
---

Save every run under its own `YYYYMMDD-HHMM` folder in `data/`, for example `data/20261101-2036`, so the name that sorts last is the most recent.

# LeRobot LIBERO evaluation in WSL (CPU)

Machine (checked 2026-10-31): Surface Laptop Studio (i5-11300H, 4 cores / 8 threads, Intel Iris Xe iGPU, **no NVIDIA GPU**, see `docs/PREREQUISITES.md`), WSL2 Ubuntu 22.04.5, about 7.8 GB RAM visible to WSL (default limit; 16 GB on Windows), no `sudo` without a password. For the Colab T4 variant see the `colab-lerobot-libero-eval` skill.

## Results

| Policy | Device | Result |
|---|---|---|
| SmolVLA `lerobot/smolvla_libero` | CPU | libero_spatial task 0, 1 episode each, all successful: 164 s (first env), 117 s (clean env, async), 107 s (clean env, sync, profiled) per episode. One run each, so the spread is not explained |

Only these single episodes were run in WSL. pi0 / pi0.5 were not tried here: about 3B parameters, a gated tokenizer and about 7.8 GB RAM make them unrealistic (an expectation, not a measurement). Use Colab for them.

## Where the time goes (profiled, one 76-step episode, sync env)

| Part | Time | Share of 107 s |
|---|---|---|
| `SmolVLAPolicy.select_action` (76 calls, mean 978 ms) | 74.3 s | about 70 % |
| `LiberoEnv.reset` (1 call, includes the 10 settle steps and first-time setup) | 18.2 s | about 17 % |
| `LiberoEnv.step` (76 calls, mean 166 ms: simulation plus two camera renders) | 12.6 s | about 12 % |
| Video frame render (`LiberoEnv.render`, 77 calls) | 0.0 s | about 0 % |

- **Policy inference on the CPU dominates.** With `n_action_steps=10` the policy computes a new chunk only about every 10th step, so the 74 s is probably spread over about 8 heavy calls of roughly 9 s each. That split is inferred from the call counts, not timed per call. torch used 4 threads.
- **Rendering is not the bottleneck.** Env step including both camera renders is only about 12 % of the episode, even with software EGL. Fixing GPU rendering (see below) can gain at most that.
- Per task the 18 s reset cost may be paid again for each new task (not verified for later tasks).

### Ways to speed it up (not tried, each changes something)

- `--policy.n_action_steps=50` (the checkpoint default) would cut the number of chunk computations about 5x, but runs longer open-loop and may change the success rate. The sweeps so far used 10, so do not mix settings in a comparison.
- Fewer flow-matching denoising steps (`--policy.num_steps`, checkpoint value 10) would cut inference time, but changes the policy output.
- More torch threads (8 instead of 4) may not help: the CPU has 4 physical cores.
- Use Colab (T4) for sweeps: about 70 s per episode there in the single-env smoke test, and policy inference runs on the GPU.

## Keep file I/O inside WSL (rule)

Files under `/mnt/c/...` go through the Windows filesystem bridge, which is much slower than WSL's own ext4 disk (`/home/till`). Rules for every WSL script:
- Write all outputs, logs, videos (`--output_dir`), caches and temporary files under `~` (for example `~/eval_runs/<run>/`), never under `/mnt/c`.
- Keep the conda envs, the Hugging Face cache (`~/.cache/huggingface`, the default) and the LIBERO assets inside WSL. Do not point `HF_HOME`, `--output_dir` or the LIBERO paths at `/mnt/c`.
- Do not run heavy reads or writes over `/mnt/c` (no cloning repos or unpacking datasets there).
- Reading a small script from `/mnt/c` once at start is fine. All runs so far read the `.sh` script from the Windows side, while logs, model cache and output videos went to `~`. Copying scripts into `~` first is cleaner, but the one copy attempt made silently did nothing, so a reliable copy method is not verified yet.
- Fetch results to Windows only at the end, as one small copy, not by writing to `/mnt/c` during the run. The Windows view of WSL files is `\\wsl$\Ubuntu-22.04\home\till\...`.
- This rule was set by the user for speed. The speed gain itself has not been measured here.

## Ask before installing (rule)

If something needs installing and cannot be installed without sudo (or is otherwise unavailable), stop and ask the user. Do not substitute a workaround. This is also in the user-level `C:\Users\TillK\.claude\CLAUDE.md`.

## Running WSL commands from the Windows side

- `conda` is only initialised in interactive shells. In scripts start with `source ~/miniconda3/etc/profile.d/conda.sh; conda activate <env>`.
- From PowerShell, nested quotes with `$` and `|` returned nothing. Put the commands in a `.sh` file and run `wsl -d Ubuntu-22.04 -- bash /mnt/c/<path>/script.sh`.
- From Git Bash (the Bash tool) set `MSYS_NO_PATHCONV=1`, otherwise `/mnt/...` is rewritten to `C:/dev/Git/mnt/...`.
- A `cp` plus `nohup ... &` inside a short `wsl -- bash -c` call silently did nothing (cause unknown: neither the copy nor the log appeared). Run the whole script as a long-lived foreground call instead (for example as a background task of the calling tool) and write the log in `~/`.
- WSL prints `wsl: ... .wslconfig` parse errors on every call. The Windows file `%USERPROFILE%\.wslconfig` holds personal notes, not WSL settings, so no memory limit is configured. Leave it untouched and do not print it. Filter the lines with `grep -av "wsl:"`.

## Environments

`~/miniconda3` (conda 25.3.1). Existing envs, not to be modified: `lerobot` (Python 3.10, lerobot 0.1.0 editable from `~/prjcts/lerobot`), `llm`, `vla-hackathon`. LeRobot 0.6.x needs a newer Python, so separate envs were created:
- `lerobot-eval-test`: **the clean, verified recipe below** (3.4 GB, torch `2.11.0+cpu`).
- `lerobot-eval`: first attempt with unneeded workarounds (7.7 GB, torch `2.11.0+cu130` CUDA build, conda ffmpeg and cmake). It works but is redundant. The user decides which one to keep.

## Setup (verified clean recipe)

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create -y -n lerobot-eval -c conda-forge python=3.12
conda activate lerobot-eval
pip install "lerobot[libero,smolvla]" --extra-index-url https://download.pytorch.org/whl/cpu
```

Verified on WSL in `lerobot-eval-test` (same recipe, other env name), no workarounds:
- The install succeeds in about 2 min 17 s with the system `cmake 3.22.1`. **No** `--no-build-isolation`, **no** `CMAKE_POLICY_VERSION_MINIMUM`, **no** conda `cmake`. (Those were Colab fixes, not needed here.)
- `--extra-index-url .../whl/cpu` gives `torch 2.11.0+cpu` (`cuda False`). The default index installed the CUDA build `2.11.0+cu130` in the first env.
- **No ffmpeg** (conda or system) is needed: the eval ran with no ffmpeg on PATH and still wrote the mp4.

## One-time LIBERO prompt (needed)

On first import, LIBERO looks for `~/.libero/config.yaml`. If it is missing it asks on stdin: "Do you want to specify a custom path for the dataset folder? (Y/N)". `N` writes a config with default paths inside the installed package. `Y` asks for a folder, creates a `datasets` folder there, and asks for confirmation. Without a terminal `lerobot-eval` fails with `EOFError` (reproduced on WSL, and no config gets written). Answer once:

```bash
echo N | python -c "import libero.libero"
```

The config is shared by all conda envs and its paths point into the env that created it. If that env is deleted, `lerobot-eval` in the other envs breaks until the config is regenerated. Back it up before experiments (`mv ~/.libero ~/.libero.bak`) and restore it afterwards.

## MuJoCo rendering

```bash
export MUJOCO_GL=egl
```

- **EGL works, in software.** The log shows `libEGL warning: failed to open /dev/dri/renderD128: Permission denied` and `falling back to kms_swrast`, so rendering runs on the CPU (Mesa software).
- **Why:** `/dev/dri/renderD128` is `root:render` mode 660 and the user `till` is in `video` but **not** in `render`. The Intel Iris Xe is exposed through `/dev/dxg`, and Mesa 23.2.1 ships the `d3d12` driver and the WSL D3D12 libraries are present. So GPU rendering on the iGPU is probably possible after `sudo usermod -aG render till` plus `wsl --shutdown`. **Not done and not tested.** The profile shows it could gain at most about 12 % of an episode, so it is low priority.
- At exit MuJoCo prints `Exception ignored in Renderer.__del__ ... EGLError`. Harmless: the eval completed with exit code 0.
- `MUJOCO_GL=osmesa` fails (`'NoneType' object has no attribute 'glGetError'`) because `libosmesa6` is missing and installing it needs sudo. Use `egl`.
- With `MUJOCO_GL` unset, a bare `mujoco.Renderer` test scene rendered fine. The eval itself was only run with `egl`, so keep setting it.

## Run command (verified in the clean env)

```bash
conda activate lerobot-eval; export MUJOCO_GL=egl
lerobot-eval \
  --policy.path=lerobot/smolvla_libero --policy.n_action_steps=10 \
  --env.type=libero --env.task=libero_spatial --env.task_ids='[0]' \
  --eval.batch_size=1 --eval.n_episodes=1 \
  --rename_map='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}' \
  --output_dir=$HOME/eval_runs/smoke
```

- **`--policy.device=cpu` is not needed.** LeRobot logs `Device 'cuda' is not available. Switching to 'cpu'`.
- **`--eval.use_async_envs=false` is not needed.** Async (the default) peaked at 3.3 GB host RAM out of 7.8 GB with `batch_size=1`. Use `false` only when a launcher patches env classes in the main process, as the profiler did.
- **The rename map is required** for SmolVLA (the checkpoint's camera inputs are `camera1` / `camera2`; the LIBERO env emits `image` / `image2`). Known from the checkpoint config and the Colab run; not re-tested by removing it.
- Videos land in `<output_dir>/videos/`. Results print at the end under `Overall Aggregated Metrics`.

## Profiling recipe

Wrap the entry point to time `SmolVLAPolicy.select_action`, `LiberoEnv.step/reset/render` and `libero.libero.envs.env_wrapper.ControlEnv.step`, with a sync env (`--eval.use_async_envs=false`, otherwise the patched classes run in a subprocess and are not timed). Print the table at exit. The WSL profile timed `predict_action_chunk`, which exists on `SmolVLAPolicy` but is not called in the eval path (it never showed up in the table), so per-chunk time was not measured there. `select_action` calls `_get_action_chunk`; time that, plus `VLAFlowMatching.embed_prefix`, `sample_actions` and `denoise_step` (see the Colab skill's profile launcher for the method names). Not yet re-run in WSL.

## Planning

- About 107 to 164 s per episode in WSL: 40 episodes (4 suites x 10 tasks x 1) is about 1.2 to 1.8 hours, dominated by CPU inference. Use WSL for small checks and reproducibility runs, Colab for sweeps.
- Initial states: by default LeRobot uses stored LIBERO init state `i` for env `i`, so one episode per task always uses state 0 (the 50 stored states vary robot pose and objects). See the Colab skill for the random-state launcher.

## Not verified

- Larger `batch_size` in WSL (RAM about 7.8 GB).
- GPU rendering via the `render` group and Mesa `d3d12`.
- Whether the per-task 18 s reset cost repeats for every task.
- Any of the speed-up options above.
- pi0 / pi0.5 in WSL.
