#!/bin/bash
# Run a command from the repo in the WSL lerobot-eval env, e.g.  wsl -d Ubuntu-22.04 -- bash scripts/wsl_test.sh pytest -q tests
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot-eval
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src" PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl
"$@"
