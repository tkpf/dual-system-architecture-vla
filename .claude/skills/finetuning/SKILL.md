---
name: finetuning
description: Finetune this project's System 1 (SmolVLA on subtasks) and System 2 (SmolVLM judge) on Colab. Covers subtask segmentation of LIBERO demos, the training loop, checkpoints that survive a dying runtime, downloading them locally, and how to read the overfitting checks. Use when training, resuming, evaluating or debugging a finetuned model. For a fresh runtime use colab-startup first; for reading experiments use dual-system-experiments.
---

# Finetuning

Only what was measured on real runs is here. Update it after each stage.

## Measured (T4, 2 vCPUs, 12 GB RAM, LeRobot 0.6.1, transformers 5.5.4)

| Measure | Value |
|---|---|
| SmolVLA, expert only, batch 16 | 0.24 steps/s, 4.7 GB GPU, data loading not limiting |
| SmolVLA, batch 32, 0 workers | 0.11 steps/s, 7.9 GB GPU |
| SmolVLA, batch 32, 2 workers | Data-loader worker crash, cause not looked into |
| SmolVLA, batch 64 | GPU out of memory |
| Trainable parameters | 99.9M of 450M |
| Sizes | Full model 907 MB, trainable weights 206 MB, optimizer 413 MB |
| Demo replay, no rendering | About 0.15 s per demo |

Use batch 16.

## Data facts
- `lerobot/libero`: 1,693 episodes, 273,465 frames, 40 tasks, listed at 10 fps, AV1 video, no sim state. 1.9 GB.
- Raw demos in `yifengzhu-hf/LIBERO-datasets` hold 47-dimensional sim states per step. The four suites are about 34 GB.
- Episodes match raw demos by actions, with a difference around 3e-9 when it is the same demo. Some episodes have no demo within tolerance, and nearest-demo matching gives false pairs, so `segment.match_episode` uses a tolerance.
- LIBERO's goal predicates only cover the final goal. "open the top drawer and put the bowl inside" has no open predicate, so `subtasks.yaml` adds one with `predicate:`.
- The old `perception.is_held` stays true after release. `LiberoWorld.gripped` checks that both fingers touch the object.

## Downloads
1. Colab, in a cell, with the tab open: `from infra import sync; sync.to_browser_new("<run>")`. It sends only what changed since the last call, never `optim.pt`, never an unfinished checkpoint.
2. Local: `python scripts/pull_finetune.py`. It checks each zip, unpacks into `data/<run>/` and writes `CONFIRMED.json`. A bad zip stays in Downloads and counts as not saved.
3. Disk low on Colab: `sync.remove_confirmed(res, confirmed, protect)` deletes only confirmed, unchanged files, and logs them to `DELETED.jsonl`. Protect the newest checkpoint, the best one, and anything a job still reads.

Google Drive is not used for finetuning.

## Fresh runtime, in one job
```bash
cd /content && git clone -q -b finetune https://github.com/tkpf/dual-system-architecture-vla.git repo
pip install -q -e /content/repo && cd /content/repo
python -m infra start setup "python -m infra install && python -m infra egl && python -m infra check"
```
About 3 minutes. Runtimes were replaced three times in one session, each time losing `/content`. Boundaries take about 15 minutes to redo once demos are cached, longer on a fresh runtime because the raw demos download again, so save `boundaries.jsonl` locally and restore it rather than recomputing.

## Pitfalls
- A job piped through `grep` writes its log only when the buffer fills, so an empty log can mean a healthy job. Check result files instead.
- A notebook runtime can be replaced without warning, and the old one's `/content` is gone. Download before relying on anything there.
- SmolVLM's processor cuts each image into 17 tiles, 2,909 tokens for two cameras, and runs out of memory at batch 2. Pass `images_kwargs={"do_image_splitting": False}`: 245 tokens, batch 8 at 1.7 s per step and 9.9 GB, 0.5 s per answer.
- Look at the contact sheets before training. Taking the last held stretch as the end of a pick looked fine in tests, but the sheets showed the moka pot already over the stove, because contact flickers during the carry. The first firm grasp is right.
- Stock SmolVLA's loss on the same validation frames: whole instruction 0.08, atomic command 0.71, atomic command with chunks cut at the boundary 0.49, with holds 1.51. A high first loss is expected, not a bug.
