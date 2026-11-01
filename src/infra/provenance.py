"""What produced a result: code version, packages, hardware."""
from __future__ import annotations

import json
import subprocess
import time
from importlib import metadata
from pathlib import Path

PACKAGES = ("lerobot", "transformers", "torch", "robosuite", "mujoco", "hf-libero")


def _out(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except subprocess.TimeoutExpired:
        return ""


def collect() -> dict:
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            versions[p] = None
    return dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), git_sha=_out(f"git -C {Path(__file__).parent} rev-parse HEAD"),
                git_dirty=bool(_out(f"git -C {Path(__file__).parent} status --porcelain")), versions=versions,
                gpu=_out("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"),
                vcpus=_out("nproc"), ram=_out("free -g | awk '/Mem:/{print $2}'"))


def write_run_info(path: str | Path) -> dict:
    info = collect()
    Path(path).write_text(json.dumps(info, indent=1))
    return info
