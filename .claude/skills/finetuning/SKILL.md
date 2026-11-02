---
name: finetuning
description: Finetune this project's System 1 (SmolVLA on subtasks) and System 2 (SmolVLM judge) on Colab. Covers subtask segmentation of LIBERO demos, the training loop, checkpoints that survive a dying runtime, downloading them locally, and how to read the overfitting checks. Use when training, resuming, evaluating or debugging a finetuned model. For a fresh runtime use colab-startup first; for reading experiments use dual-system-experiments.
---

# Finetuning

Only what was measured on real runs is here. Update it after each stage.

## Measured (T4, 2 vCPUs, 12 GB RAM, LeRobot 0.6.1, transformers 5.5.4)

| Measure | Value |
|---|---|
| SmolVLA, expert only, batch 16 | 0.24 steps/s in a short test, 0.22 over 600 steps (about 15 min per 200), 4.7 GB GPU, data loading not limiting |
| SmolVLA, batch 32, 0 workers | 0.11 steps/s, 7.9 GB GPU |
| SmolVLA, batch 32, 2 workers | Data-loader worker crash, cause not looked into |
| SmolVLA, batch 64 | GPU out of memory |
| Trainable parameters | 99.9M of 450M |
| Sizes | Full model 907 MB, trainable weights 206 MB, optimizer 413 MB |
| Demo replay, no rendering | About 0.15 s per demo; a full `segment run` took about 25 min with demos cached |
| `eval.atomic`, 7 tasks, real + gibberish + stop | About 10 min per model (failed phases end episodes early) |
| Checkpoint download | 206 MB as 2 parts, arrives within a minute |

Use batch 16.

## Data facts
- `lerobot/libero`: 1,693 episodes, 273,465 frames, 40 tasks, listed at 10 fps, AV1 video, no sim state. 1.9 GB.
- Raw demos in `yifengzhu-hf/LIBERO-datasets` hold 47-dimensional sim states per step. The four suites are about 34 GB.
- Episodes match raw demos by actions, with a difference around 3e-9 when it is the same demo. Some episodes have no demo within tolerance, and nearest-demo matching gives false pairs, so `segment.match_episode` uses a tolerance.
- LIBERO's goal predicates only cover the final goal. "open the top drawer and put the bowl inside" has no open predicate, so `subtasks.yaml` adds one with `predicate:`.
- The old `perception.is_held` stays true after release. `LiberoWorld.gripped` checks that both fingers touch the object.
- LIBERO's `On` predicate turns true while the object is still in the gripper. Place phases end when the fingers let go, and `eval.atomic.phase_done` requires that too. The demos already command "open" at that moment in 1,257 of 1,318 final place phases, and stop about 10 frames later.
- Segmentation result (`data/20261102-0626/boundaries.jsonl`): 1,492 of 1,693 kept; no matching demo 170, goal not reached 22, never held 9. `boundaries_goal_end.jsonl` there is the older version, with places ending at the goal predicate.

## What the first SmolVLA run showed
Training only the action expert (`train_expert_only=True`, VLM fully frozen) fitted the subtask data: validation loss went 0.53, 0.39, 0.32 at steps 200, 400, 600. Language response, the action difference between the real command and gibberish, fell instead: 0.177, 0.174, 0.172, against 0.209 for the stock model. In rollouts gibberish did about as well as the real commands. So watch language response, not loss: low loss here means the expert reads the phase from the images. The next try should train the top language layers too, or add gibberish frames whose target is to stay still. Details: run 9 in `docs/EXPERIMENTS.md`.

## Not yet run on a GPU
`collect_judge.py`, `system2.py` and the reasoner's `yes_no` phase check with an adapter are only unit-tested. Expect first-run fixes, for example in how `images_kwargs` reaches the SmolVLM processor in a batch.

## Downloads
1. Colab, in a cell, with the tab open: `from infra import sync; sync.to_browser_new("<run>")`. It sends only what changed since the last call, never `optim.pt`, never an unfinished checkpoint.
2. Local: `python scripts/pull_finetune.py`. It checks each zip, unpacks into `data/<run>/` and writes `CONFIRMED.json`. A bad zip stays in Downloads and counts as not saved.
3. Disk low on Colab: `sync.remove_confirmed(res, confirmed, protect)` deletes only confirmed, unchanged files, and logs them to `DELETED.jsonl`. Protect the newest checkpoint, the best one, and anything a job still reads.

Google Drive is not used for finetuning.

`files.download` delivered 40 MB and 120 MB, but a 421 MB zip never arrived, with no error, after three tries. So `sync.download` sends anything above 100 MB as parts `<zip>.pNNofMM`. The browser writes empty placeholders and `*.part` files while downloading, so `pull_finetune.py` joins parts only when all are complete, and deletes them only once the joined zip passes its check. A sent zip stays on the runtime: `sync.download(<path>)` sends it again.

## Fresh runtime, in one job
```bash
cd /content && git clone -q -b finetune https://github.com/tkpf/dual-system-architecture-vla.git repo
pip install -q -e /content/repo && cd /content/repo
python -m infra start setup "python -m infra install && python -m infra egl && python -m infra check"
```
About 3 minutes. Runtimes were replaced four times in one session, each time losing `/content`, and one replacement came back without a GPU (check `nvidia-smi` first). Boundaries take about 25 minutes to redo once demos are cached, longer on a fresh runtime because the raw demos download again, so save `boundaries.jsonl` locally and restore it rather than recomputing.

A notebook kernel that started before `pip install -e` cannot import `infra`: add `/content/repo/src` to `sys.path` in the cell.

## Pitfalls
- `pkill -f '<pattern>'` from a notebook cell also kills the cell's own shell, because its command line contains the pattern. Kill by pid, or start the next step as a job from a script file.
- A job piped through `grep` writes its log only when the buffer fills, so an empty log can mean a healthy job. Check result files instead.
- A notebook runtime can be replaced without warning, and the old one's `/content` is gone. Download before relying on anything there.
- SmolVLM's processor cuts each image into 17 tiles, 2,909 tokens for two cameras, and runs out of memory at batch 2. Pass `images_kwargs={"do_image_splitting": False}`: 245 tokens, batch 8 at 1.7 s per step and 9.9 GB, 0.5 s per answer.
- Look at the contact sheets before training. Taking the last held stretch as the end of a pick looked fine in tests, but the sheets showed the moka pot already over the stove, because contact flickers during the carry. The first firm grasp is right.
- Stock SmolVLA's loss on the same validation frames: whole instruction 0.08, atomic command 0.71, atomic command with chunks cut at the boundary 0.49, with holds 1.51. A high first loss is expected, not a bug.
