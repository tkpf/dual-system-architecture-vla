"""Unpack finetune_results_*.zip from Downloads into data/<run>/, additively, after checking each zip. Run on the local machine.

    python scripts/pull_finetune.py [--downloads <dir>] [--data <dir>] [--keep-last 3]

A zip that fails its check stays in Downloads and nothing from it counts as saved. data/<run>/CONFIRMED.json lists the files
that passed, and only those may be removed from the runtime (infra.sync.remove_confirmed). Older checkpoints are pruned to the
last few, plus every step_XXX000 milestone and any step named in BEST.txt.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import zipfile
from pathlib import Path


def check(zip_path: Path) -> dict | None:
    """The manifest if the zip is intact and has exactly the listed files with the listed sizes."""
    try:
        with zipfile.ZipFile(zip_path) as z:
            if z.testzip() is not None:
                return None
            manifest = json.loads(z.read("res/MANIFEST.json"))
            for rel, (size, _) in manifest["files"].items():
                if z.getinfo(f"res/{rel}").file_size != size:
                    return None
            return manifest
    except (zipfile.BadZipFile, KeyError, json.JSONDecodeError):
        return None


def unpack(zip_path: Path, data: Path, manifest: dict) -> Path:
    target = data / manifest["run"]
    with zipfile.ZipFile(zip_path) as z:
        for rel in manifest["files"]:
            dest = target / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".part")
            with z.open(f"res/{rel}") as src, open(tmp, "wb") as out:
                shutil.copyfileobj(src, out)
            os.replace(tmp, dest)
    confirmed_path = target / "CONFIRMED.json"
    confirmed = json.loads(confirmed_path.read_text()) if confirmed_path.exists() else {}
    confirmed_path.write_text(json.dumps({**confirmed, **manifest["files"]}))
    return target


def prune_local(root: Path, keep_last: int) -> list[str]:
    best_file = root.parent / "BEST.txt"
    best = set(best_file.read_text().split()) if best_file.exists() else set()
    done = sorted(p for p in root.glob("step_*") if (p / "COMPLETE").exists())
    removed = []
    for p in done[:-keep_last]:
        if int(p.name.split("_")[1]) % 1000 and p.name not in best:
            shutil.rmtree(p)
            removed.append(p.name)
    return removed


def join_parts(downloads: Path) -> list[str]:
    """Join <zip>.pNNofMM parts into <zip> once all of them are there and the browser has finished writing them (it keeps
    empty placeholders next to *.part files while downloading). Parts are removed only when the joined zip passes its check."""
    groups: dict[str, list[Path]] = {}
    for part in downloads.glob("finetune_results_*.zip.p*of*"):
        if re.search(r"\.p\d+of\d+$", part.name):
            groups.setdefault(part.name.rsplit(".p", 1)[0], []).append(part)
    report = []
    for name, parts in sorted(groups.items()):
        total = int(parts[0].name.rsplit("of", 1)[1])
        busy = any(downloads.glob(name.removesuffix(".zip") + ".*.part")) or any(p.stat().st_size == 0 for p in parts)
        if len(parts) != total or busy:
            report.append(f"waiting for parts of {name}: {len(parts)}/{total}{', still downloading' if busy else ''}")
            continue
        tmp = downloads / (name + ".joining")
        with open(tmp, "wb") as out:
            for part in sorted(parts):
                out.write(part.read_bytes())
        os.replace(tmp, downloads / name)
        if check(downloads / name) is None:
            report.append(f"joined {name} fails its check, parts kept")
            (downloads / name).unlink()
            continue
        for part in parts:
            part.unlink()
    return report


def pull(downloads: Path, data: Path, keep_last: int = 3) -> list[str]:
    report = join_parts(downloads)
    for zip_path in sorted(downloads.glob("finetune_results_*.zip")):
        manifest = check(zip_path)
        if manifest is None:
            report.append(f"BAD ZIP, kept: {zip_path.name}")
            continue
        target = unpack(zip_path, data, manifest)
        removed = []
        for root in target.rglob("checkpoints"):
            removed += prune_local(root, keep_last)
        zip_path.unlink()
        report.append(f"saved {zip_path.name} -> {target}: {len(manifest['files'])} files, {manifest['info']}, pruned {removed}")
    return report


if __name__ == "__main__":
    here = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser()
    p.add_argument("--downloads", default=str(Path.home() / "Downloads"))
    p.add_argument("--data", default=str(here / "data"))
    p.add_argument("--keep-last", type=int, default=3)
    a = p.parse_args()
    lines = pull(Path(a.downloads), Path(a.data), a.keep_last)
    print("\n".join(lines) if lines else "no finetune_results zip in downloads")
