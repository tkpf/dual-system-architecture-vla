---
name: colab-startup
description: Start a fresh Google Colab T4 session for this project and run the dual-system / System-1 probing experiments from the local repo (git clone, pip install -e, lerobot install, EGL rendering, queue, results download). Use when a Colab runtime was (re)started, when moving local code to Colab, or when running notebooks/colab_run.ipynb.
---

# Colab startup for the dual-system project

## Dual-system pipeline (`infra` + `dual_system`, use this first)
Status 2026-11-01: run on a standard T4 with these steps. Two grids of 24 new episodes each finished without errors, about 40 s per dual episode; setup took under 10 minutes.
1. Clone or pull, then `pip install -q -e /content/repo`.
2. Run `python -m infra start install "python -m infra install"` (about 5 min; a job, so a cell timeout cannot kill it), then `!python -m infra egl` and `!python -m infra check` (must print `Tesla T4`).
3. Run `!python -m infra status` for vCPUs, GPU memory and jobs. It prints one JSON line and is safe to poll.
4. Default System 1 is pi0.5, which needs `HF_TOKEN` set in the kernel (see the `dual-system-experiments` skill). Default configs: `pi05_atomic_test` and `pi05_phase_test`, unless the user asks for another. `free_atomic_test`, `first_test` and `profiles_test` document earlier SmolVLA runs. For choosing and reading experiments, use `dual-system-experiments`.
5. **Every run gets a new folder named `YYYYMMDD-HHMM`:** `RUN=$(python -m infra newrun)` creates `/content/runs/YYYYMMDD-HHMM` (UTC). Never reuse `/content/res` or an old run folder for a new run. Resuming the same run after a restart uses the same folder. Queue: `python -m infra start w0 "python -m dual_system.eval.grid free_atomic_test --res $RUN --stop-after-min 40"`. On the standard T4 (2 vCPUs, 12 GB) use one worker; a second (`--worker 0 --workers 2`, plus `w1` with `--worker 1`) only with at least 4 vCPUs. High RAM was not available on a T4 here.
6. Sync: `!python -m infra drive` (mount Drive in a cell first) and, in a cell, `from infra import sync; sync.to_browser()`. The notebook kernel started before the editable install, so add `/content/repo/src` to `sys.path` first, and after a `git pull` reload the module (`importlib.reload(infra.sync)`), because the kernel keeps the old code. Then locally `powershell -File scripts/pull_results.ps1 -Run dual`.
7. **Where data lands:** every pull creates a new folder `data/YYYYMMDD-HHMM`, e.g. `data/20261101-2036`, using the local download time, so the name that sorts last is always the most recent download. There is no suffix; the run name is in `RUN.txt` inside the folder. Do not use `-latest` names. Existing folders are never overwritten. Folders such as `data/2026-11-01_20-36_dual` and `data/dual-latest` are from before this rule. To pull under another run name, zip with `sync.to_browser(prefix="<run>_results")` and pull with `-Run <run>`; `to_browser` names the folder inside the zip `res` itself.
8. `to_browser()`, `infra drive` and `infra status` use the newest run folder unless given another. Results: `python -m dual_system.eval.report <res>`, `python -m dual_system.eval.lookup <res> --task 0 --state B`, `python -m dual_system.eval.side_by_side <res>` (videos in `<res>/videos`, rendered only on request of this command, so run it before the final sync).
9. Whenever System 2 said success while the LIBERO condition is false, tell the user: the report prints these and writes `false_success.csv`. System 2 still ends those episodes. The user inspects the frames.
10. **Pushing:** the user allows Claude Code to push the working branch. If the permission check blocks it or the SSH key's passphrase stops it, tell the user the exact command and do not work around it. Verify with `git ls-remote origin` before starting a run.

Local checks without Colab: `wsl -d Ubuntu-22.04 -- bash scripts/wsl_test.sh python -m pytest -q tests`, and the `smoke` config. `scripts/tune_robot_state.py` checks a joint offset on all tasks.

## System-1 probing (`single_system`)

Companion to `colab-lerobot-libero-eval` (installation details, timings, pi0/pi0.5, profiling). This skill is the short path for a **new runtime**: get the repo onto Colab, set up lerobot and rendering, run the queue, bring results back. Status: the steps below were run by hand on 2026-11-01; **the packaged version (`single_system.colab`, `single_system.runner`) has not been run on Colab yet**, so verify with one small cell first (see TODO.md).

## Layout (local workspace = the repo)
```
pyproject.toml            package `dual-system` (src layout), no heavy deps; lerobot is installed separately
src/single_system/
  config.py               ProbeConfig (env-var transport to the launcher)
  prompts.py              prompt variants (orig, other, unrelated, empty, gibberish, typo, chain, paraphrase, fixed:<text>)
  perturb.py              synthetic image perturbations + real-photo replacement
  probe_env.py            monkeypatches lerobot's LiberoEnv (prompt, images, init state, step log)
  launch.py               `python -m single_system.launch <lerobot-eval args>`
  runner.py               Cell, resumable queue, prompt->video manifest.csv
  experiments.py          cell grids for probes A (prompts), B1/B2 (visual OOD), C (atomic commands)
  real_clips.py           builds real_clips.npz from lerobot/droid_100
  colab.py                install / egl / check / zip helpers
  analysis.py             manifest loading, Wilson CIs, eef displacement
notebooks/colab_run.ipynb thin entry point that imports the package
TODO.md                   what is still missing
```

## Moving code to Colab: git clone (chosen approach)
1. Local: commit and push the workspace to a GitHub repo (ask the user before any push; `data/`, `papers/`, `.venv` are in `.gitignore`). Set `REPO_URL` in the notebook once. Private repo: add a GitHub token as Colab secret `GH_TOKEN` and clone with `https://<token>@github.com/...` read from `userdata.get`, never printed or pasted into chat.
2. Colab: `git clone` (or `git pull` if `/content/repo` exists), `pip install -q -e /content/repo`. Editable, so a later `git pull` takes effect after restarting the Python processes (background jobs import fresh each time; a notebook kernel needs `importlib.reload` or restart).
3. Edit loop: change locally, push, `git pull` in Colab. Do not edit in Colab and forget to copy back.
4. Fallback without GitHub: build a zip locally and push it into a Colab cell (base64 in chunks through the MCP). Slow, only for small trees.

## Driving it (with the colab-mcp tools)
1. The user opens a notebook, sets Runtime > T4 GPU. Load tools with ToolSearch (`colab`), call `open_colab_browser_connection`. Either upload `notebooks/colab_run.ipynb` themselves or run the same code in cells you add.
2. **GPU check pitfall:** the first `subprocess.run("nvidia-smi; ...", capture_output=True)` printed nothing even though the T4 was attached (output lost with the combined command). Re-run `nvidia-smi` alone before telling the user the runtime has no GPU. Expected: Tesla T4 15360 MiB, driver 580.82.07, Python 3.13, 2 vCPUs, 12 GB RAM.
3. Order: clone + `pip install -e` -> `python -m single_system.colab install` in the background (about 5 min, log `/content/install.log`, ends with the `lerobot-eval` path) -> `colab.extract_egl()` -> `colab.check_renderer()` must print `Tesla T4/PCIe/SSE2` (not llvmpipe, which is ~12x slower) -> `build_droid_clips()` (0.5 GB download) -> queue.
4. Long jobs run as `nohup ... &` with `start_new_session=True`; poll logs in short cells. A 540 s `sleep` cell was interrupted by the user and lost; keep waits at or below about 2 to 4 minutes. Never print tqdm output.
5. Only one queue at a time: the box has 2 vCPUs and 12 GB RAM, parallel sims slow each other down and risk OOM. To chain queues, wait for the previous `pgrep -f run_queue` / `single_system.runner` to finish.
6. If a cell tool call is rejected or interrupted, the background job may still be running. Check `pgrep -af single_system` and `cat /content/res/queue.log` before relaunching.
7. `update_cell` then `run_code_cell` reuses a cell; check that the cell you run is the one you just updated (a poll cell can silently run stale content).

## Running experiments
- **Default: 1 eval episode per task (`eps=1`) unless the user says otherwise.** The stored initial states differ little (robot joints about 1 degree), so extra episodes per task add little and cost time. The first Experiment A run used 3 per task (a pilot, 2026-11-01); everything after that, and the defaults in `experiments.py`, use 1. Only raise it when the user asks, and prefer more tasks over more episodes of the same task.
- Cells are defined in `single_system/experiments.py` (`queue_A/B1/B2/C`), saved with `save_queue`, run with `python -m single_system.runner queue.json /content/res`. Finished cells have a `DONE` marker, so after a runtime restart only missing cells run.
- One cell = one launcher process = one condition on one suite with several tasks, so the model loads once. Measured on the T4 with EGL: about 16 s per episode in libero_goal (cell of 5 tasks x 3 episodes: 239 s), baseline correct prompt 14/15.
- `--eval.max_episodes_rendered` is NOT a CLI option (it is a function argument); videos are written anyway to `<output_dir>/videos/<suite>_<task>/eval_episode_<k>.mp4`. The manifest maps each episode to its video and prompt (`<res>/manifest.csv`: cell, suite, task_id, episode_in_task, init_state_id, prompt_mode, img_mode, true_instruction, prompt, success, n_steps, video).
- The launcher needs `--eval.use_async_envs=false` and `--eval.batch_size=1` (patches only apply in the process that builds the env). The SmolVLA checkpoint needs the camera `--rename_map`; both are in `runner.build_cmd`.
- Prompt override works by setting `LiberoEnv.task_description`; image perturbations by patching `LiberoEnv._format_raw_obs`; `render()` is kept clean so videos show the real sim. For real photos the frames are pre-rotated 180 deg because the LIBERO processor flips raw frames. Hook names are from lerobot 0.6.1.
- Installed lerobot lives in `/usr/local/lib/python3.13/dist-packages/lerobot/` (read `envs/libero.py`, `scripts/lerobot_eval.py` there when a hook breaks).

## Getting results back: periodic sync every 15 minutes (browser download + local mirror)
Why: on 2026-11-01 the Colab usage limit recycled the runtime and every result in `/content` was lost (9 of 18 A cells). Colab cannot write to the local disk, so results are pulled out regularly. **Partial data counts**: a pipeline does not have to be finished, running cells are included.

What is synced: all of `/content/res` (manifest.csv, per-cell episodes.jsonl step logs, run.log, videos, eval_info.json) plus `SYNC_INFO.json` (time, episode count, cell counts). `episodes.jsonl` is rewritten after every finished episode (`probe_env.Recorder.write`), and `manifest.csv` includes unfinished cells (`cell_done=False`), so a running cell shows up after its first episode completes. Videos are written by lerobot when an episode ends.

**One sync = two steps**
1. Colab: run `from single_system import colab; colab.sync_to_browser('/content/res')` (notebook cell 7). It zips to `/content/system1_results_<timestamp>.zip` and calls `files.download`. It refuses (prints "nothing to sync") when no episodes exist, so a fresh or recycled runtime never produces a zip.
2. Local, about 30 to 60 s later (browser must finish the download): `powershell -File scripts/pull_results.ps1`. It takes the newest `system1_results_*.zip` from `~/Downloads`, **deletes `data/system1-latest/` and replaces it with the zip content (mirror)**, and removes the processed zips from Downloads. Exit codes: 1 no zip, 2 download still running (retry in a minute), 3 refused.
   - Safety: it refuses to replace the mirror with a zip that has fewer episodes than the current mirror (typical after a runtime reset). Only pass `-Force` if the user confirms the old local data is obsolete. Because the mirror is replaced fully, copy anything you want to keep out of `data/system1-latest/` (or use another folder) before analysing; put analysis outputs elsewhere (e.g. `data/analysis/`).

**Scheduling (Claude Code side).** Create a recurring job with `CronCreate` (`7,22,37,52 * * * *`, off-minute on purpose) once the queue is running. Prompt for the job, in this form: "Colab sync: run the sync cell in the Colab notebook via colab-mcp (colab.sync_to_browser('/content/res')), wait 45 s, run scripts/pull_results.ps1, report episodes/cells done in 2 lines. Do not touch the queue. If the cell returns stale output (old timestamp) or hangs, say the runtime is unreachable and stop retrying." Cron jobs fire only while the Claude session is idle, are session-only and expire after 7 days. Delete the job with `CronDelete` when the queue is finished or the runtime is gone.

**Known risks (not yet tested end to end)**
- `files.download` only works while the Colab tab is open; background tabs may delay it, and Chrome asks once whether to allow multiple automatic downloads (the user must accept).
- A hung or stale cell result means the kernel is disconnected (usage limit or idle timeout). A call can block for 30 minutes before the MCP aborts it, so do not retry in a loop; after reconnecting check the date/uptime in the output, a repeated old timestamp means the cell did not run.
- Everything since the last successful sync is lost on a recycle, so a shorter interval trades quota for safety. Run a sync before ending a session or when the user says the limit is near.
- Zip size grows with the videos (not measured yet); if it gets large, exclude `*/run.log` or sync only new cells.

The notebook writes the queue to `/content/res/queue.json`, so every sync also carries the exact grid that produced the data.

## Open points
- Packaged code unverified on Colab (first run may expose import or quoting bugs in `runner.build_cmd`, which passes values via `json.dumps` inside a bash command line).
- Whether `pip install -e` of the repo conflicts with Colab's preinstalled packages: not tested.
- Private-repo token flow not tested.
