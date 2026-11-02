"""Get results off the runtime: Google Drive as durable copy, browser zip for the local mirror (scripts/pull_results.ps1).

A recycled runtime has already cost results once, so both refuse to export when no episodes exist: an empty runtime can
never overwrite good data. Every run gets its own folder, /content/runs/YYYYMMDD-HHMM (UTC on Colab), so names sort by time.
"""
from __future__ import annotations

import glob
import json
import os
import time

from .bootstrap import sh

DRIVE_DEFAULT = "/content/drive/MyDrive/sereact_results"
RUNS_DIR = "/content/runs"


def new_run_dir(runs_dir: str = RUNS_DIR) -> str:
    path = f"{runs_dir}/{time.strftime('%Y%m%d-%H%M')}"
    os.makedirs(path, exist_ok=True)
    return path


def latest_run_dir(runs_dir: str = RUNS_DIR) -> str:
    runs = sorted(glob.glob(f"{runs_dir}/[0-9]*-[0-9]*"))
    return runs[-1] if runs else "/content/res"


def counts(res_dir: str) -> dict:
    files = glob.glob(f"{res_dir}/*/episodes.jsonl")
    cells = [os.path.dirname(f) for f in files]
    probes = glob.glob(f"{res_dir}/*/probe.jsonl") + glob.glob(f"{res_dir}/probe.jsonl")
    return dict(n_episodes=sum(sum(1 for _ in open(f)) for f in files + probes), n_cells=len(cells),
                n_cells_done=sum(os.path.exists(c + "/DONE") for c in cells),
                n_checkpoints=len(glob.glob(f"{res_dir}/**/step_*/COMPLETE", recursive=True)),
                n_labels=sum(sum(1 for _ in open(f)) for name in ("boundaries.jsonl", "judge_labels.jsonl") for f in glob.glob(f"{res_dir}/{name}")))


def write_sync_info(res_dir: str) -> dict | None:
    info = counts(res_dir)
    if info["n_episodes"] == 0 and info["n_checkpoints"] == 0 and info["n_labels"] == 0:
        print("no episodes, checkpoints or labels under", res_dir, "- nothing to sync")
        return None
    info["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(info, open(f"{res_dir}/SYNC_INFO.json", "w"))
    return info


def to_drive(res_dir: str | None = None, drive_dir: str = DRIVE_DEFAULT) -> dict | None:
    """Additive copy, nothing is deleted on Drive. Mount first in a notebook cell (needs your click once)."""
    res_dir = res_dir or latest_run_dir()
    if not os.path.isdir("/content/drive/MyDrive"):
        print("Drive is not mounted: from google.colab import drive; drive.mount('/content/drive')")
        return None
    info = write_sync_info(res_dir)
    if info:
        dest = f"{drive_dir}/{os.path.basename(res_dir)}"
        r = sh(f"mkdir -p {dest} && cp -ru {res_dir}/. {dest}/ && du -sh {dest}")
        print(info, r.stdout.strip(), r.stderr[-300:])
    return info


def to_browser(res_dir: str | None = None, prefix: str = "dual_results", exclude_keyframes: bool = False) -> str | None:
    """Zip everything so far (running cells included) and download it. Must run in a notebook cell, with the tab open.
    The folder inside the zip is always named res, which is what the local pull script expects."""
    res_dir = res_dir or latest_run_dir()
    info = write_sync_info(res_dir)
    if info is None:
        return None
    out = f"/content/{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.zip"
    skip = " -x '*/keyframes/*'" if exclude_keyframes else ""
    sh(f"rm -f /content/{prefix}_*.zip; rm -rf /tmp/zip_root; mkdir /tmp/zip_root; ln -s {res_dir} /tmp/zip_root/res; "
       f"cd /tmp/zip_root && zip -qr {out} res{skip}")
    print(info, sh(f"ls -lh {out}").stdout.strip())
    from google.colab import files

    files.download(out)
    return out


# ---------------------------------------------------------------- finetuning: small, incremental downloads
SENT_NAME, DELETED_NAME = "SENT.json", "DELETED.jsonl"
NEVER_ZIP = ("optim.pt", ".tmp_", SENT_NAME)  # the optimizer is 400 MB and only useful on the runtime


def _signature(path: str) -> list:
    stat = os.stat(path)
    return [stat.st_size, int(stat.st_mtime)]


def _new_files(res_dir: str, sent: dict) -> dict:
    """Files not sent yet, or changed since. Checkpoints only count when complete, so a half-written one is never sent."""
    out = {}
    for folder, _, names in os.walk(res_dir):
        rel_folder = os.path.relpath(folder, res_dir)
        if ".tmp_" in rel_folder or ("step_" in rel_folder and "COMPLETE" not in names):
            continue
        for name in names:
            rel = os.path.normpath(os.path.join(rel_folder, name)).replace(os.sep, "/")
            if any(mark in rel for mark in NEVER_ZIP):
                continue
            if sent.get(rel) != _signature(os.path.join(folder, name)):
                out[rel] = _signature(os.path.join(folder, name))
    return out


def to_browser_new(res_dir: str | None = None, prefix: str = "finetune_results") -> str | None:
    """Zip only what changed since the last call, with a manifest, and download it. Must run in a notebook cell, with the tab open.
    The files count as sent only after the zip is written, and as safe only when scripts/pull_finetune.py has checked them."""
    import zipfile

    res_dir = res_dir or latest_run_dir()
    info = write_sync_info(res_dir)
    if info is None:
        return None
    sent_path = f"{res_dir}/{SENT_NAME}"
    sent = json.load(open(sent_path)) if os.path.exists(sent_path) else {}
    new = _new_files(res_dir, sent)
    if not new:
        print("nothing new", info)
        return None
    manifest = dict(run=os.path.basename(res_dir), files=new, info=info)
    out = f"/content/{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:  # weights and frames do not compress much
        for rel in new:
            z.write(f"{res_dir}/{rel}", f"res/{rel}")
        z.writestr("res/MANIFEST.json", json.dumps(manifest))
    json.dump({**sent, **new}, open(sent_path, "w"))
    print(len(new), "files", round(os.path.getsize(out) / 1e6), "MB", info)
    from google.colab import files

    files.download(out)
    return out


def remove_confirmed(res_dir: str, confirmed: dict, protect: list[str], low_gb: float = 10.0, shutil_module=None) -> list[str]:
    """Free disk: delete only files the local check confirmed (same size and time) and no protected path is part of.
    confirmed is the CONFIRMED.json written by scripts/pull_finetune.py. Does nothing while enough disk is free."""
    import shutil

    if (shutil_module or shutil).disk_usage(res_dir).free / 1e9 >= low_gb:
        return []
    removed = []
    for rel, signature in confirmed.items():
        path = f"{res_dir}/{rel}"
        if any(mark in rel for mark in protect) or not os.path.exists(path) or _signature(path) != signature:
            continue
        os.remove(path)
        removed.append(rel)
    with open(f"{res_dir}/{DELETED_NAME}", "a") as log:
        for rel in removed:
            log.write(json.dumps(dict(file=rel, time=time.strftime("%Y-%m-%d %H:%M:%S"))) + "\n")
    print("removed", len(removed), "confirmed files")
    return removed
