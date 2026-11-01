"""Run a grid of episodes; one cell = one (suite, task, robot state, condition) episode.

    python -m dual_system.eval.grid first_test --res /content/res [--worker 0 --workers 2] [--stop-after-min 40]

Models load once per process. Finished cells have a DONE marker and are skipped on restart. Cells are assigned to
workers by index, so workers need no coordination. Seeds depend on suite, task and state only, so a pair differs in
its condition alone.
"""
from __future__ import annotations

import argparse
import shutil
import time
import traceback
import zlib
from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).parent.parent / "configs"


@dataclass
class Cell:
    suite_key: str
    suite: str
    task_id: int
    robot_state: str
    condition: str
    seed: int
    is_libero_plus: bool = False

    @property
    def name(self) -> str:
        return f"{self.suite_key}__t{self.task_id}__{self.robot_state}__{self.condition}"


def load_config(name: str) -> dict:
    path = Path(name) if name.endswith(".yaml") else CONFIG_DIR / f"{name}.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["condition_cfgs"] = {c: yaml.safe_load((CONFIG_DIR / "conditions" / f"{c}.yaml").read_text(encoding="utf-8"))
                             for c in cfg["conditions"]}
    return cfg


def build_cells(cfg: dict, only_suites: list[str] | None = None) -> list[Cell]:
    """Task by task, so every task gets its complete set of pairs before the next one starts."""
    cells = []
    for s in cfg["suites"]:
        if only_suites and s["name"] not in only_suites:
            continue
        for task in s["tasks"]:
            for state in cfg["robot_states"]:
                seed = cfg["base_seed"] + zlib.crc32(f"{s.get('suite', s['name'])}-{task}-{state}".encode()) % 100000
                for cond in cfg["conditions"]:
                    cells.append(Cell(s["name"], s.get("suite", s["name"]), task, state, cond, seed,
                                      s.get("is_libero_plus", False)))
    return cells


def run_cell(cell: Cell, cfg: dict, res: Path, system1, shared: dict, run_info: dict) -> None:
    from ..env import LiberoWorld
    from ..interface import build_interface
    from ..robot_states import build_robot_states
    from ..rollout import RolloutRunner
    from ..system2 import build_system2
    from ..tracing import TraceWriter
    from ..triggers import build_triggers
    from ..types import RunSpec

    d = res / cell.name
    if d.exists():  # a crashed attempt must not mix into the rerun
        shutil.rmtree(d)
    d.mkdir(parents=True)
    ccfg = cfg["condition_cfgs"][cell.condition]
    world = LiberoWorld(cell.suite, cell.task_id, cell.is_libero_plus, cfg.get("max_steps"))
    try:
        spec = RunSpec(suite=cell.suite_key, task_id=cell.task_id, robot_state=cell.robot_state, condition=cell.condition,
                       seed=cell.seed, instruction=world.instruction, max_steps=world.max_steps, run=cfg["run"],
                       git_sha=run_info.get("git_sha", ""), cell=cell.name)
        tracer = TraceWriter(d, spec)
        try:
            RolloutRunner(world, system1, build_system2(ccfg["system2"], shared), build_interface(ccfg.get("interface")),
                          build_triggers(ccfg.get("triggers")), tracer).run(spec, build_robot_states(cfg["robot_states"])[cell.robot_state])
        finally:
            tracer.close()
    finally:
        world.close()
    (d / "DONE").write_text("ok")


def run_grid(cfg: dict, res: Path, worker: int = 0, workers: int = 1, only_suites: list[str] | None = None,
             stop_after_min: float | None = None, max_cells: int | None = None) -> None:
    from infra.provenance import write_run_info

    from ..system1 import build_system1

    res.mkdir(parents=True, exist_ok=True)
    run_info = write_run_info(res / f"run_info_w{worker}.json")
    cells = [c for i, c in enumerate(build_cells(cfg, only_suites)) if i % workers == worker]
    todo = [c for c in cells if not (res / c.name / "DONE").exists()]
    log = open(res / f"grid_w{worker}.log", "a", buffering=1)
    log.write(f"{time.strftime('%H:%M:%S')} worker {worker}/{workers}: {len(todo)} of {len(cells)} cells to run\n")
    if not todo:
        return
    system1, shared = build_system1(cfg["system1"]), {}
    t0 = time.time()
    for n, cell in enumerate(todo):
        if stop_after_min is not None and (time.time() - t0) / 60 > stop_after_min:
            log.write("time box reached, stopping\n")
            break
        if max_cells is not None and n >= max_cells:
            break
        t1 = time.time()
        try:
            run_cell(cell, cfg, res, system1, shared, run_info)
            log.write(f"{time.strftime('%H:%M:%S')} {cell.name} ok {time.time() - t1:.0f}s\n")
        except Exception:
            log.write(f"{time.strftime('%H:%M:%S')} {cell.name} FAILED\n{traceback.format_exc()}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--res", default="/content/res")
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--only-suites", nargs="*")
    p.add_argument("--stop-after-min", type=float)
    p.add_argument("--max-cells", type=int)
    a = p.parse_args()
    run_grid(load_config(a.config), Path(a.res), a.worker, a.workers, a.only_suites, a.stop_after_min, a.max_cells)
