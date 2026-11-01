"""Resume points that survive a dying runtime: written to a temp folder and renamed, so a crash never leaves half a checkpoint.

A checkpoint is a folder step_XXXXXXX with weights.pt (trainable weights only, about 200 MB for SmolVLA), state.pt (step, RNG,
scheduler) and optim.pt (about 400 MB). Only weights and state are downloaded, so optim.pt can be missing after a restore:
training then goes on with fresh optimizer moments, which is harmless at a low learning rate.
"""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch


def _name(step: int) -> str:
    return f"step_{step:07d}"


def _rng() -> dict:
    state = dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate())
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict) -> None:
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save(root: Path, step: int, weights: dict, optimizer=None, scheduler=None, extra: dict | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    final, tmp = root / _name(step), root / f".tmp_{_name(step)}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    torch.save(weights, tmp / "weights.pt")
    torch.save(dict(step=step, rng=_rng(), scheduler=scheduler.state_dict() if scheduler else None, extra=extra or {}), tmp / "state.pt")
    if optimizer is not None:
        torch.save(optimizer.state_dict(), tmp / "optim.pt")
    (tmp / "COMPLETE").write_text(json.dumps(dict(step=step)))
    shutil.rmtree(final, ignore_errors=True)
    tmp.rename(final)
    return final


def latest(root: Path) -> Path | None:
    done = sorted(p for p in root.glob("step_*") if (p / "COMPLETE").exists()) if root.exists() else []
    return done[-1] if done else None


def load(path: Path, optimizer=None, scheduler=None) -> dict:
    """Returns weights and state. Restores RNG, and the optimizer and scheduler when they were saved."""
    state = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
    weights = torch.load(path / "weights.pt", map_location="cpu")
    _restore_rng(state["rng"])
    if scheduler is not None and state["scheduler"] is not None:
        scheduler.load_state_dict(state["scheduler"])
    state["optimizer_restored"] = False
    if optimizer is not None and (path / "optim.pt").exists():
        optimizer.load_state_dict(torch.load(path / "optim.pt", map_location="cpu"))
        state["optimizer_restored"] = True
    return dict(weights=weights, state=state)


def prune(root: Path, keep_last: int = 2, keep: set[int] | None = None) -> list[Path]:
    """Remove all but the newest checkpoints and those named in keep. Kept older ones lose optim.pt, which only the newest needs."""
    keep = keep or set()
    done = sorted(p for p in root.glob("step_*") if (p / "COMPLETE").exists())
    removed = []
    for p in done[:-keep_last] if keep_last else done:
        if int(p.name.split("_")[1]) in keep:
            (p / "optim.pt").unlink(missing_ok=True)
        else:
            shutil.rmtree(p)
            removed.append(p)
    return removed
