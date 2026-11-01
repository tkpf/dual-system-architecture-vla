"""Get results off the runtime: Google Drive as durable copy, browser zip for the local mirror (scripts/pull_results.ps1).

A recycled runtime has already cost results once, so both refuse to export when no episodes exist: an empty runtime can
never overwrite good data.
"""
from __future__ import annotations

import glob
import json
import os
import time

from .bootstrap import sh

DRIVE_DEFAULT = "/content/drive/MyDrive/sereact_results"


def counts(res_dir: str) -> dict:
    files = glob.glob(f"{res_dir}/*/episodes.jsonl")
    cells = [os.path.dirname(f) for f in files]
    return dict(n_episodes=sum(sum(1 for _ in open(f)) for f in files), n_cells=len(cells),
                n_cells_done=sum(os.path.exists(c + "/DONE") for c in cells))


def write_sync_info(res_dir: str) -> dict | None:
    info = counts(res_dir)
    if info["n_episodes"] == 0:
        print("no episodes under", res_dir, "- nothing to sync")
        return None
    info["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(info, open(f"{res_dir}/SYNC_INFO.json", "w"))
    return info


def to_drive(res_dir: str = "/content/res", run: str = "dual", drive_dir: str = DRIVE_DEFAULT) -> dict | None:
    """Additive copy, nothing is deleted on Drive. Mount first in a notebook cell (needs your click once)."""
    if not os.path.isdir("/content/drive/MyDrive"):
        print("Drive is not mounted: from google.colab import drive; drive.mount('/content/drive')")
        return None
    info = write_sync_info(res_dir)
    if info:
        dest = f"{drive_dir}/{run}"
        r = sh(f"mkdir -p {dest} && cp -ru {res_dir}/. {dest}/ && du -sh {dest}")
        print(info, r.stdout.strip(), r.stderr[-300:])
    return info


def to_browser(res_dir: str = "/content/res", prefix: str = "dual_results", exclude_keyframes: bool = False) -> str | None:
    """Zip everything so far (running cells included) and download it. Must run in a notebook cell, with the tab open."""
    info = write_sync_info(res_dir)
    if info is None:
        return None
    out = f"/content/{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.zip"
    skip = " -x '*/keyframes/*'" if exclude_keyframes else ""
    sh(f"rm -f /content/{prefix}_*.zip; cd {os.path.dirname(res_dir)} && zip -qr {out} {os.path.basename(res_dir)}{skip}")
    print(info, sh(f"ls -lh {out}").stdout.strip())
    from google.colab import files

    files.download(out)
    return out
