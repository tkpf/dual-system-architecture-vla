"""python -m infra install | egl | check | status [res] | start <name> <cmd...> | stop <name> | log <name> | drive [res] [run]

status prints compact JSON and never prints progress bars, so polling from a cell stays cheap.
"""
from __future__ import annotations

import json
import sys

from . import bootstrap, jobs, sync


def status(res_dir: str = "/content/res") -> dict:
    mem = bootstrap.sh("nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader").stdout.strip()
    ram = bootstrap.sh("free -g | awk '/Mem:/{print $3\"/\"$2\" GB\"}'").stdout.strip()
    return dict(jobs={k: v["alive"] for k, v in jobs.list_jobs().items()}, results=sync.counts(res_dir), gpu=mem, ram=ram,
                vcpus=bootstrap.sh("nproc").stdout.strip())


def main(argv: list[str]) -> None:
    cmd = argv[0] if argv else ""
    if cmd == "install":
        bootstrap.install()
    elif cmd == "egl":
        bootstrap.extract_egl()
    elif cmd == "check":
        bootstrap.check_renderer()
    elif cmd == "status":
        print(json.dumps(status(*argv[1:])))
    elif cmd == "start":
        print("pid", jobs.start(argv[1], " ".join(argv[2:])))
    elif cmd == "stop":
        print("stopped" if jobs.stop(argv[1]) else "not running")
    elif cmd == "log":
        print(jobs.log_tail(argv[1]))
    elif cmd == "drive":
        sync.to_drive(*argv[1:])
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
