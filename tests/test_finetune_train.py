import numpy as np
import torch

from dual_system.finetune import checkpoint
from dual_system.finetune.dataset import frame_index, hold_past_end, load_splits, split_of
from dual_system.finetune.segment import load_subtasks
from dual_system.finetune.system1 import lr_factor, rising


def test_chunk_holds_after_phase_end():
    action = torch.arange(1, 36, dtype=torch.float32).reshape(5, 7)
    action[:, 6] = torch.tensor([1.0, 1.0, -1.0, -1.0, -1.0])
    pad = torch.tensor([False, False, False, True, True])
    a, p = hold_past_end(action, pad, frames_left=1)
    assert torch.equal(a[:2], action[:2])  # inside the phase, unchanged
    assert (a[2:, :6] == 0).all() and (a[2:, 6] == 1.0).all()  # no next-phase motion, gripper as at the end
    assert not p.any()  # holds are real targets, also past the episode end
    assert torch.equal(action[2, :6], torch.arange(15, 21, dtype=torch.float32))  # the input is not changed


def test_chunk_inside_phase_is_unchanged():
    action, pad = torch.ones(5, 7), torch.zeros(5, dtype=torch.bool)
    a, p = hold_past_end(action, pad, frames_left=10)
    assert a is action and p is pad


def test_splits_do_not_overlap_and_cover_a_unseen_task_per_suite():
    splits = load_splits()
    tasks = load_subtasks()
    assert len(splits["unseen_tasks"]) == 4 and set(splits["unseen_tasks"]) <= set(tasks)
    assert not set(splits["init_states"]["select"]) & set(splits["init_states"]["test"])
    for ablation in ("put the bowl on the stove", "put the bowl on top of the cabinet", "turn on the stove and put the moka pot on it"):
        assert ablation not in splits["unseen_tasks"]
    pairs = [(t, e) for t in tasks for e in range(20)]
    rows = [dict(task=t, episode=e, first_index=i * 100, phases=[dict(command="c", start=0, end=3)]) for i, (t, e) in enumerate(pairs)]
    seen = [frame_index(rows, splits, s) for s in ("train", "val", "unseen_task")]
    assert all(len(s) for s in seen)
    assert split_of(splits["unseen_tasks"][0], 0, splits) == "unseen_task"
    assert sum(len(s) for s in seen) == 3 * len(rows)  # every frame is in exactly one split
    train, val = ({row[0] for row in s} for s in seen[:2])
    assert not train & val


def test_frames_left_counts_down_to_phase_end():
    rows = [dict(task="t", episode=1, first_index=100, phases=[dict(command="a", start=0, end=3), dict(command="b", start=3, end=5)])]
    splits = dict(unseen_tasks=[], val_percent=0)
    assert frame_index(rows, splits, "train") == [(100, 2, "a"), (101, 1, "a"), (102, 0, "a"), (103, 1, "b"), (104, 0, "b")]


def _weights(x):
    return {"w": torch.full((2,), float(x))}


def test_resume_restores_step_rng_and_optimizer(tmp_path):
    p = torch.nn.Parameter(torch.ones(2))
    opt = torch.optim.AdamW([p], lr=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 ** s)
    p.sum().backward()
    opt.step()
    sched.step()
    torch.manual_seed(5)
    path = checkpoint.save(tmp_path, 200, _weights(3), opt, sched, dict(val_losses=[1.0]))
    expected = torch.rand(3)
    opt2 = torch.optim.AdamW([torch.nn.Parameter(torch.ones(2))], lr=0.1)
    sched2 = torch.optim.lr_scheduler.LambdaLR(opt2, lambda s: 0.5 ** s)
    got = checkpoint.load(path, opt2, sched2)
    assert got["state"]["step"] == 200 and got["state"]["extra"]["val_losses"] == [1.0] and got["state"]["optimizer_restored"]
    assert torch.equal(torch.rand(3), expected)
    assert sched2.state_dict()["last_epoch"] == 1
    assert opt2.state_dict()["state"][0]["step"] == 1


def test_scheduled_optimizer_reloads_with_safe_loading(tmp_path):
    p = torch.nn.Parameter(torch.ones(2))
    opt = torch.optim.AdamW([p], lr=1e-5)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, 100, 10))
    p.sum().backward()
    opt.step()
    sched.step()
    path = checkpoint.save(tmp_path, 1, _weights(1), opt, sched)
    opt2 = torch.optim.AdamW([torch.nn.Parameter(torch.ones(2))], lr=1e-5)
    assert checkpoint.load(path, opt2)["state"]["optimizer_restored"]


def test_lr_factor_warms_up_and_decays_to_a_tenth():
    assert lr_factor(0, 100, 10) < 0.2
    assert lr_factor(10, 100, 10) > 0.9
    assert abs(lr_factor(100, 100, 10) - 0.1) < 1e-9
    assert isinstance(lr_factor(5, 100, 10), float)


def test_resume_works_without_optimizer_after_a_restore(tmp_path):
    path = checkpoint.save(tmp_path, 10, _weights(1))
    assert not (path / "optim.pt").exists()
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.ones(2))])
    assert checkpoint.load(path, opt)["state"]["optimizer_restored"] is False


def test_unfinished_checkpoint_is_ignored(tmp_path):
    checkpoint.save(tmp_path, 10, _weights(1))
    (tmp_path / "step_0000020").mkdir()  # crashed before COMPLETE
    (tmp_path / ".tmp_step_0000030").mkdir()
    assert checkpoint.latest(tmp_path).name == "step_0000010"


def test_prune_keeps_last_and_milestones(tmp_path):
    for s in (200, 400, 600, 800, 1000, 1200, 1400):
        checkpoint.save(tmp_path, s, _weights(s), torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.1))
    checkpoint.prune(tmp_path, keep_last=2, keep={400, 1000})
    assert sorted(p.name for p in tmp_path.glob("step_*")) == [f"step_{s:07d}" for s in (400, 1000, 1200, 1400)]
    assert not (tmp_path / "step_0000400" / "optim.pt").exists()  # kept only as weights
    assert (tmp_path / "step_0001400" / "optim.pt").exists()


def test_rising_needs_three_rises_in_a_row():
    assert rising([1.0, 0.9, 0.95, 1.0, 1.1])
    assert not rising([1.0, 0.9, 0.95, 0.9, 1.1])
    assert not rising(list(np.array([1.0, 1.1])))
