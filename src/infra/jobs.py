"""Detached background jobs that survive a cell ending: start, check, stop. State lives in plain files."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

from .bootstrap import EGL_ENV

JOB_DIR = Path("/content/jobs")


def start(name: str, cmd: str, egl: bool = True, job_dir: Path = JOB_DIR) -> int:
    """Refuses to start a job whose previous run is still alive."""
    job_dir.mkdir(parents=True, exist_ok=True)
    if is_alive(name, job_dir):
        raise RuntimeError(f"job {name} is still running")
    log = open(job_dir / f"{name}.log", "a")
    full = f"{EGL_ENV}; {cmd}" if egl else cmd
    p = subprocess.Popen(full, shell=True, executable="/bin/bash", stdout=log, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    (job_dir / f"{name}.json").write_text(json.dumps(dict(pid=p.pid, cmd=cmd, started=time.strftime("%H:%M:%S"))))
    return p.pid


def is_alive(name: str, job_dir: Path = JOB_DIR) -> bool:
    f = job_dir / f"{name}.json"
    if not f.exists():
        return False
    try:
        os.kill(json.loads(f.read_text())["pid"], 0)
    except (ProcessLookupError, PermissionError):
        return False
    # An exited child of this process stays as a zombie until reaped, and kill(pid, 0) still succeeds on it.
    stat = Path(f"/proc/{json.loads(f.read_text())['pid']}/stat")
    return stat.exists() and stat.read_text().rsplit(")", 1)[1].split()[0] != "Z"


def stop(name: str, job_dir: Path = JOB_DIR) -> bool:
    if not is_alive(name, job_dir):
        return False
    os.killpg(json.loads((job_dir / f"{name}.json").read_text())["pid"], signal.SIGTERM)
    return True


def list_jobs(job_dir: Path = JOB_DIR) -> dict:
    if not job_dir.exists():
        return {}
    return {f.stem: dict(alive=is_alive(f.stem, job_dir), **json.loads(f.read_text())) for f in sorted(job_dir.glob("*.json"))}


def log_tail(name: str, chars: int = 1500, job_dir: Path = JOB_DIR) -> str:
    """tqdm bars are carriage-return lines; they are removed so the tail stays readable."""
    f = job_dir / f"{name}.log"
    if not f.exists():
        return ""
    data = f.read_bytes()[-chars * 20:].decode("utf-8", "replace").replace("\r", "\n")
    lines = [l for l in data.splitlines() if l.strip() and "it/s]" not in l and "s/it]" not in l and "Running rollout" not in l]
    return "\n".join(lines)[-chars:]
