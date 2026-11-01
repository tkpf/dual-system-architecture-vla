"""Colab session setup on a T4 runtime (Linux, root). Steps verified in .claude/skills/colab-lerobot-libero-eval."""
from __future__ import annotations

import subprocess
import sys

EGL_ENV = ("export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LD_LIBRARY_PATH=/content/nvidia/gl:$LD_LIBRARY_PATH "
           "__EGL_VENDOR_LIBRARY_FILENAMES=/content/nvidia/gl/10_nvidia.json")


def sh(cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, shell=True, executable="/bin/bash", capture_output=True, text=True)


def install() -> None:
    script = """
apt-get -qq update > /dev/null 2>&1
apt-get -qq install -y cmake build-essential libegl1-mesa-dev libgl1-mesa-dev libglu1-mesa-dev libosmesa6-dev libglfw3 ffmpeg > /dev/null 2>&1
export CMAKE_POLICY_VERSION_MINIMUM=3.5
pip install -q setuptools wheel
pip install -q --no-build-isolation hf-egl-probe egl_probe
pip install -q "lerobot[libero,smolvla]"
which lerobot-eval
echo N | python -c "import libero.libero"
"""
    r = sh(script)
    print(r.stdout[-1500:], r.stderr[-500:])


def extract_egl(dest: str = "/content/nvidia") -> None:
    """Installer version must equal the loaded driver version exactly."""
    v = sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader").stdout.strip()
    if not v:
        sys.exit("nvidia-smi returned nothing: is the runtime a GPU runtime?")
    steps = [
        f"mkdir -p {dest} && cd {dest} && curl -s -o installer.run https://download.nvidia.com/XFree86/Linux-x86_64/{v}/NVIDIA-Linux-x86_64-{v}.run && ls -l installer.run",
        f"cd {dest} && sh installer.run --extract-only --target {dest}/ex > extract.log 2>&1; echo extract exit $?",
        f"mkdir -p {dest}/gl && cd {dest}/gl && for l in libEGL_nvidia libnvidia-eglcore libnvidia-glcore libnvidia-glsi libnvidia-gpucomp libnvidia-tls; do cp -f ../ex/$l.so.{v} . ; done && ln -sf libEGL_nvidia.so.{v} libEGL_nvidia.so.0 && cp ../ex/10_nvidia.json . && ls",
    ]
    for s in steps:
        print(sh(s).stdout[-600:])


def check_renderer() -> str:
    code = ("import os; os.environ['MUJOCO_GL']='egl'\nfrom mujoco import gl_context\nc=gl_context.GLContext(64,64); c.make_current()\n"
            "from OpenGL import GL\nprint(GL.glGetString(GL.GL_RENDERER))")
    r = sh(f"{EGL_ENV}; python - <<'E'\n{code}\nE")
    out = r.stdout.strip()
    print(out, "(harmless EGL cleanup error at exit is expected)" if r.stderr else "")
    if "Tesla" not in out:
        print("WARNING: not the T4 renderer (llvmpipe is ~12x slower). Check /content/nvidia/gl and the env exports.")
    return out
