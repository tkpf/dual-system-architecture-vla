"""Finetune SmolVLA on subtasks. Only the action expert and projections train; vision and the VLM stay frozen.

    python -m dual_system.finetune.system1 train --res <res> --boundaries <file> [--steps 5000] [--stop-after-min 60]
    python -m dual_system.finetune.system1 export --res <res> --step 2000 --out <dir>   # a folder LeRobotSystem1 can load

Restarting train with the same --res resumes from the newest checkpoint. metrics.jsonl has train and validation loss, and how
far predicted actions move between the real command and gibberish on the same frames: if that does not grow, the model is
not learning language. Loss alone does not pick the best checkpoint; eval.atomic does that with rollouts.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import checkpoint
from .dataset import SubtaskDataset, frame_index, load_boundaries, load_splits, summary

BASE = "lerobot/smolvla_libero"
GIBBERISH = "qxv lorp zindle wug flarn"  # same as eval.command_probe
FPS = 10


def rising(losses: list[float], patience: int = 3) -> bool:
    """Validation loss went up on each of the last patience evaluations."""
    tail = losses[-patience - 1:]
    return len(tail) == patience + 1 and all(b > a for a, b in zip(tail, tail[1:]))


def lr_factor(step: int, total: int, warmup: int) -> float:
    """Warmup, then cosine down to 10%. A plain float: a NumPy one cannot be read back from a checkpoint with weights_only."""
    return float(min(1.0, (step + 1) / warmup) * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * min(step, total) / total))))


def milestone_steps(total: int, every: int = 1000) -> set[int]:
    return set(range(every, total + 1, every))


def jitter(batch: dict, rng: torch.Generator) -> None:
    """Small brightness and contrast change per image, no flips: flipping would break left and right in the commands."""
    for key in batch:
        if key.startswith("observation.images"):
            x = batch[key]
            n = len(x)
            bright = 1 + 0.15 * (2 * torch.rand(n, 1, 1, 1, generator=rng) - 1)
            contrast = 1 + 0.15 * (2 * torch.rand(n, 1, 1, 1, generator=rng) - 1)
            mean = x.mean(dim=(2, 3), keepdim=True)
            batch[key] = ((x - mean) * contrast + mean).mul(bright).clamp(0, 1)


def build(load_path: str = BASE):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.policies import make_policy, make_pre_post_processors

    from ..system1.lerobot_policy import RENAME_MAP

    cfg = PreTrainedConfig.from_pretrained(load_path)
    cfg.pretrained_path, cfg.device = load_path, "cuda"
    cfg.train_expert_only, cfg.freeze_vision_encoder = True, True
    rename = RENAME_MAP if set(RENAME_MAP.values()) <= set(cfg.input_features) else {}
    policy = make_policy(cfg=cfg, env_cfg=LiberoEnvConfig(), rename_map=rename)
    pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=load_path, preprocessor_overrides={
        "device_processor": {"device": "cuda"}, "rename_observations_processor": {"rename_map": rename}})
    return cfg, policy, pre, post


def trainable(policy) -> dict:
    return {n: p for n, p in policy.named_parameters() if p.requires_grad}


@torch.no_grad()
def evaluate(policy, pre, loader, batches: int) -> dict:
    """Validation loss and language response, with fixed noise so that evaluations can be compared."""
    policy.eval()
    losses, moved = [], []
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(0)
        for i, raw in enumerate(loader):
            if i >= batches:
                break
            b = pre(dict(raw))
            losses.append(float(policy.forward(b)[0]))
            state = torch.get_rng_state(), torch.cuda.get_rng_state()
            real = policy.predict_action_chunk(b)
            torch.set_rng_state(state[0]), torch.cuda.set_rng_state(state[1])
            control = policy.predict_action_chunk(pre(dict(raw, task=[GIBBERISH] * len(raw["task"]))))
            moved.append(float((real - control).abs().mean()))
    policy.train()
    return dict(val_loss=float(np.mean(losses)), language_response=float(np.mean(moved)))


def train(a) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    res = Path(a.res)
    root = res / "checkpoints"
    res.mkdir(parents=True, exist_ok=True)
    boundaries, splits = load_boundaries(a.boundaries), load_splits()
    print(json.dumps(summary(boundaries, splits)), flush=True)
    cfg, policy, pre, _ = build()
    policy.train()
    params = trainable(policy)
    print("trainable M", sum(p.numel() for p in params.values()) / 1e6, flush=True)
    optim = torch.optim.AdamW(params.values(), lr=a.lr, weight_decay=a.weight_decay)
    warmup = 100
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: lr_factor(s, a.steps, warmup))
    step, val_losses = 0, []
    last = checkpoint.latest(root)
    if last:
        restored = checkpoint.load(last, optim, sched)
        missing = policy.load_state_dict(restored["weights"], strict=False)
        assert not missing.unexpected_keys
        step, val_losses = restored["state"]["step"], restored["state"]["extra"].get("val_losses", [])
        print("resumed", last.name, "optimizer restored", restored["state"]["optimizer_restored"], flush=True)

    base = LeRobotDataset("lerobot/libero", delta_timestamps={"action": [i / FPS for i in range(cfg.chunk_size)]}, video_backend="torchcodec")
    train_ds = SubtaskDataset(base, frame_index(boundaries, splits, "train"))
    val_frames = frame_index(boundaries, splits, "val")
    val_frames = [val_frames[i] for i in np.random.default_rng(0).permutation(len(val_frames))]
    val_loader = torch.utils.data.DataLoader(SubtaskDataset(base, val_frames), batch_size=a.batch_size, num_workers=0)
    generator = torch.Generator().manual_seed(a.seed + step)  # a resumed run draws new batches, not the same ones again
    loader = torch.utils.data.DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, drop_last=True, num_workers=a.workers,
                                         persistent_workers=a.workers > 0, generator=generator)
    colour = torch.Generator().manual_seed(a.seed + step + 1)
    metrics = open(res / "metrics.jsonl", "a", encoding="utf-8")
    t0, last_time, keep = time.time(), time.time(), milestone_steps(a.steps)

    def save_point():
        path = checkpoint.save(root, step, {n: p.detach().cpu() for n, p in params.items()}, optim, sched, dict(val_losses=val_losses))
        checkpoint.prune(root, keep_last=2, keep=keep)
        return path

    stop = None
    while stop is None:
        for raw in loader:
            jitter(raw, colour)
            loss = policy.forward(pre(raw))[0]
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params.values(), 10.0)
            optim.step()
            sched.step()
            step += 1
            if step % a.log_every == 0:
                now = time.time()
                row = dict(step=step, loss=round(float(loss), 5), lr=sched.get_last_lr()[0], steps_per_s=round(a.log_every / (now - last_time), 3),
                           gpu_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))
                last_time = now
                metrics.write(json.dumps(row) + "\n")
                metrics.flush()
                print(json.dumps(row), flush=True)
            if step % a.save_every == 0 or step >= a.steps:
                row = dict(step=step, **evaluate(policy, pre, val_loader, a.val_batches))
                val_losses.append(row["val_loss"])
                metrics.write(json.dumps(row) + "\n")
                metrics.flush()
                print(json.dumps(row), flush=True)
                save_point()
                last_time = time.time()
                if rising(val_losses):
                    stop = "validation loss rose 3 times"
            if step >= a.steps:
                stop = stop or "reached steps"
            if a.stop_after_min and time.time() - t0 > a.stop_after_min * 60:
                save_point()
                stop = stop or "time limit"
            if stop:
                break
    print("stopped:", stop, "at step", step, flush=True)


def export(a) -> None:
    """Stock checkpoint plus the trained weights of one step, saved with its processors."""
    cfg, policy, pre, post = build()
    path = Path(a.res) / "checkpoints" / f"step_{a.step:07d}"
    weights = torch.load(path / "weights.pt", map_location="cpu")
    missing = policy.load_state_dict(weights, strict=False)
    assert not missing.unexpected_keys
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg.pretrained_path = str(out)
    policy.save_pretrained(out)
    pre.save_pretrained(out)
    post.save_pretrained(out)
    print("exported", out, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["train", "export"])
    p.add_argument("--res", required=True)
    p.add_argument("--boundaries")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--val-batches", type=int, default=10)
    p.add_argument("--stop-after-min", type=float)
    p.add_argument("--step", type=int)
    p.add_argument("--out")
    a = p.parse_args()
    {"train": train, "export": export}[a.command](a)
