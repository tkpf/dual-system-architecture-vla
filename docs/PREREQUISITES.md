# Prerequisites

This document records the hardware, software, and experience constraints that shape the approach to the challenge described in [CHALLENGE.md](../CHALLENGE.md).

## 1. Machine Specs

| Component | Details |
|---|---|
| Device | Microsoft Surface Laptop Studio (x64) |
| CPU | Intel Core i5-11300H @ 3.10 GHz (Tiger Lake), 4 cores / 8 threads, max 3.11 GHz |
| RAM | 16 GB total (about 15.8 GB usable) |
| GPU (integrated) | Intel Iris Xe Graphics, driver 32.0.101.7085 |
| GPU (discrete) | None. No NVIDIA GPU, and `nvidia-smi` is not available. |
| Storage | Samsung SSD 970 EVO 1 TB, about 320 GB free on C: |
| OS | Windows 11 Home, build 22621 |
| Python | 3.10 (`C:\Program Files\Python310\python.exe`) |
| Linux environment | WSL with Ubuntu 22.04, available when needed |
| BIOS | 33.104.143 |

## 2. Compute Constraints

- **No CUDA.** There is no NVIDIA GPU, so PyTorch CUDA training is not available. Training runs on the CPU, or on the Intel Iris Xe iGPU if it can be used through a backend such as DirectML or OpenVINO. Neither route is a reliable base for training.
- **Limited memory.** 16 GB of system RAM is shared with the OS and with the simulator. The integrated GPU has no dedicated VRAM.
- **Heavy training is out of scope.** The design has to reduce the need for training, not depend on it.

### Consequences for the approach

- Prefer **frozen pretrained models** (e.g., a SigLIP/CLIP encoder or a small open VLM) for perception and reasoning, and run them for inference only.
- Keep the **trainable part small**: a lightweight policy head (MLP or small ACT/diffusion head) on top of frozen features, trained on a small LIBERO slice (e.g., one task suite or a few tasks).
- Use **low-dimensional, cached features**. Precompute embeddings once and train on the cache, instead of running the encoders in every training step.
- Keep System 2 **infrequent**. It runs at a low rate (e.g., once per sub-goal) and its output is reused by System 1, which is the cheap part.
- Expect **slow simulation and evaluation** on CPU. Plan evaluation budgets (number of episodes, number of seeds) around this.
- Use a **subset of LIBERO tasks** and a small number of episodes per ablation condition.

## 3. Experience Constraints

- **Limited hands-on experience with VLMs and VLAs.** The architecture and the data pipeline will need to be learned while building, not assumed.
- Plan for a **learning phase** early in the week: load a small pretrained VLM or SigLIP encoder, run it on LIBERO frames, and confirm the feature shapes and latency before designing the interface between System 2 and System 1.
- Favor the **simplest System 2 representation that can be ablated cleanly**, such as a sub-goal text string or a static instruction embedding, over more complex options (visual waypoints, topological graphs). The ablation study matters more than architectural complexity.

## 4. Optional Remote Compute: Google Colab via MCP

Google's open-source Colab MCP server lets a local agent such as Claude Code drive a Colab notebook that is open in the browser. Sources: the [announcement](https://developers.googleblog.com/announcing-the-colab-mcp-server-connect-any-ai-agent-to-google-colab/) (17 March 2026) and the [repo](https://github.com/googlecolab/colab-mcp) (Apache-2.0).

**What is confirmed**
- The server bridges a local agent to a Colab session in the browser. The client must run locally and support `notifications/tools/list_changed`; Claude Code is named as meeting this.
- The announcement describes adding, moving and running cells, creating `.ipynb` files, and installing packages with `!pip install`.
- Requirements: Python, git, and `uv`. The documented config entry is:

```json
"mcpServers": {
  "colab-mcp": {
    "command": "uvx",
    "args": ["git+https://github.com/googlecolab/colab-mcp"],
    "timeout": 30000
  }
}
```

  For Claude Code this should translate to `claude mcp add colab-mcp -- uvx git+https://github.com/googlecolab/colab-mcp`. That command is my inference, not from the README, so check it against Claude Code's docs.

**What is not documented** (neither the announcement nor the README says): GPU/TPU support or runtime selection, the tool names, how authentication works, whether a notebook tab must stay open, Windows support, and quotas or limits. The README has an "Internal - For Colab Developers" section, accepts no external contributions, and asks users to use Discussions instead of Issues.

**Feasibility for this challenge**
- Plausible, and worth a short test before relying on it. Colab notebooks run on Linux, which is what LIBERO requires, so the WSL step is not needed there. Only `uvx` runs on the laptop.
- A GPU would be chosen in Colab's runtime settings in the browser (general Colab behavior, not stated by the MCP docs). Whether a GPU is available depends on the account tier and current availability.
- If it works, it relaxes the "no CUDA" constraint above for **training** a small System 1 head and for running a small VLM for System 2. It does not remove the need to keep the design small.
- Risks:
  - Colab sessions time out and the disk is not persistent, so save checkpoints and cached features to Google Drive or download them.
  - MuJoCo rendering (`MUJOCO_GL=egl`) has to be set up in the notebook, and I have not tested that on Colab.
  - Data and code leave the laptop, so use only the public LIBERO data and no API keys in cells.
  - The server is labeled internal, so it may change or break.

**Suggested first test** (about 30 minutes): add the server, open a blank Colab notebook with a GPU runtime, ask the agent to run `nvidia-smi`, then install `lerobot[libero]` and run one episode. If this fails, fall back to local CPU as described in sections 2 and 3.

## 5. Summary

| Constraint | Impact |
|---|---|
| No NVIDIA GPU | No CUDA training locally; keep trainable parameters and training steps small. Colab via MCP may offer a GPU (untested, see section 4) |
| 16 GB RAM, iGPU only | Use frozen encoders, cached features, and small batches |
| Limited VLM/VLA experience | Budget time for learning and start with a simple System 2 interface |
| One-week deadline | Scope to a small LIBERO subset and a minimal, fully ablated pipeline |
