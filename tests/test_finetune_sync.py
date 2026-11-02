import importlib.util
import json
import os
import zipfile
from collections import namedtuple
from pathlib import Path

from infra import sync

spec = importlib.util.spec_from_file_location("pull_finetune", Path(__file__).parent.parent / "scripts" / "pull_finetune.py")
pull_finetune = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pull_finetune)


def _run(tmp_path):
    res = tmp_path / "20261102-1000"
    ck = res / "s1" / "checkpoints" / "step_0000200"
    ck.mkdir(parents=True)
    (ck / "weights.pt").write_bytes(b"w" * 10)
    (ck / "optim.pt").write_bytes(b"o" * 10)
    (ck / "COMPLETE").write_text("{}")
    (res / "s1" / "checkpoints" / ".tmp_step_0000400").mkdir()
    (res / "s1" / "checkpoints" / ".tmp_step_0000400" / "weights.pt").write_bytes(b"x")
    half = res / "s1" / "checkpoints" / "step_0000600"
    half.mkdir()
    (half / "weights.pt").write_bytes(b"h")
    (res / "metrics.jsonl").write_text("{}\n")
    return res


def test_new_files_skip_optimizer_and_unfinished_checkpoints(tmp_path):
    new = sync._new_files(str(_run(tmp_path)), {})
    assert sorted(new) == ["metrics.jsonl", "s1/checkpoints/step_0000200/COMPLETE", "s1/checkpoints/step_0000200/weights.pt"]


def test_new_files_only_returns_changes(tmp_path):
    res = _run(tmp_path)
    sent = sync._new_files(str(res), {})
    assert sync._new_files(str(res), sent) == {}
    (res / "metrics.jsonl").write_text("{}\n{}\n")
    assert list(sync._new_files(str(res), sent)) == ["metrics.jsonl"]


def test_training_run_can_be_synced_without_episodes(tmp_path):
    assert sync.write_sync_info(str(_run(tmp_path)))["n_checkpoints"] == 1


def test_empty_runtime_syncs_nothing(tmp_path):
    (tmp_path / "metrics.jsonl").write_text("{}")
    assert sync.write_sync_info(str(tmp_path)) is None


def _zip(downloads, run="r1", name="finetune_results_1.zip", content=b"weights", claimed=None):
    downloads.mkdir(exist_ok=True)
    rel = "s1/checkpoints/step_0000200/weights.pt"
    with zipfile.ZipFile(downloads / name, "w") as z:
        z.writestr(f"res/{rel}", content)
        z.writestr("res/MANIFEST.json", json.dumps(dict(run=run, files={rel: [claimed or len(content), 100]}, info={})))
    return downloads / name, rel


def test_good_zip_is_unpacked_confirmed_and_removed(tmp_path):
    zip_path, rel = _zip(tmp_path / "dl")
    pull_finetune.pull(tmp_path / "dl", tmp_path / "data")
    assert (tmp_path / "data" / "r1" / rel).read_bytes() == b"weights"
    assert json.loads((tmp_path / "data" / "r1" / "CONFIRMED.json").read_text())[rel] == [7, 100]
    assert not zip_path.exists()


def test_incomplete_zip_is_kept_and_not_confirmed(tmp_path):
    zip_path, _ = _zip(tmp_path / "dl", claimed=999)
    report = pull_finetune.pull(tmp_path / "dl", tmp_path / "data")
    assert "BAD ZIP" in report[0] and zip_path.exists()
    assert not (tmp_path / "data" / "r1").exists()


def test_truncated_download_is_rejected(tmp_path):
    zip_path, _ = _zip(tmp_path / "dl")
    data = zip_path.read_bytes()
    zip_path.write_bytes(data[:len(data) // 2])
    assert pull_finetune.check(zip_path) is None


def test_local_prune_keeps_last_milestones_and_best(tmp_path):
    root = tmp_path / "checkpoints"
    for s in (200, 400, 600, 1000, 1200, 1400, 1600, 1800):
        (root / f"step_{s:07d}").mkdir(parents=True)
        (root / f"step_{s:07d}" / "COMPLETE").write_text("{}")
    (tmp_path / "BEST.txt").write_text("step_0000400")
    pull_finetune.prune_local(root, 3)
    assert sorted(int(p.name[5:]) for p in root.iterdir()) == [400, 1000, 1400, 1600, 1800]


Usage = namedtuple("Usage", "free")


class Disk:
    def __init__(self, gb):
        self.gb = gb

    def disk_usage(self, _):
        return Usage(self.gb * 1e9)


def _confirmed(tmp_path):
    (tmp_path / "keep.pt").write_bytes(b"k")
    (tmp_path / "old.pt").write_bytes(b"o")
    (tmp_path / "changed.pt").write_bytes(b"c")
    (tmp_path / "unknown.pt").write_bytes(b"u")
    confirmed = {n: sync._signature(str(tmp_path / n)) for n in ("keep.pt", "old.pt")}
    confirmed["changed.pt"] = [999, 1]  # the file on the runtime differs from what was saved locally
    return confirmed


def test_removes_only_confirmed_unchanged_unprotected_files_when_disk_is_low(tmp_path):
    removed = sync.remove_confirmed(str(tmp_path), _confirmed(tmp_path), ["keep"], shutil_module=Disk(1))
    assert removed == ["old.pt"]
    assert sorted(p for p in os.listdir(tmp_path) if p.endswith(".pt")) == ["changed.pt", "keep.pt", "unknown.pt"]
    assert "old.pt" in (tmp_path / sync.DELETED_NAME).read_text()


def test_nothing_is_removed_while_disk_is_free(tmp_path):
    assert sync.remove_confirmed(str(tmp_path), _confirmed(tmp_path), [], shutil_module=Disk(50)) == []
    assert (tmp_path / "old.pt").exists()


def test_a_run_with_only_labels_can_be_synced_but_an_empty_one_cannot(tmp_path):
    assert sync.write_sync_info(str(tmp_path)) is None
    (tmp_path / "boundaries.jsonl").write_text("{}\n{}\n")
    assert sync.write_sync_info(str(tmp_path))["n_labels"] == 2
