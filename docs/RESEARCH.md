# Research Knowledge Base

Index and synthesis of the literature relevant to [CHALLENGE.md](../CHALLENGE.md) (Dual-System VLA on LIBERO, with an ablation study). Constraints from [PREREQUISITES.md](PREREQUISITES.md) apply: CPU only, 16 GB RAM, one week, little VLA experience.

All PDFs are in [papers/](papers/), named `<arxiv-id>-<short-name>.pdf`. Summaries below are based on the abstracts and on what I know of the papers. Details marked *(verify)* were not checked against the paper text, so confirm them before citing in the report.

## Index

| Page | Contents |
|---|---|
| [research/dual-system-architectures.md](research/dual-system-architectures.md) | Hi Robot, GR00T N1, RoboDual, HiRT, DP-VLA, OpenHelix, Fast-in-Slow, Hume, Libra-VLA: how each splits System 2 and System 1 and what interface it uses |
| [research/language-interfaces.md](research/language-interfaces.md) | Language-as-interface work: RT-H, Embodied CoT. Closest to a text sub-goal System 2 |
| [research/policies-and-encoders.md](research/policies-and-encoders.md) | System 1 candidates and frozen encoders: ACT, Diffusion Policy, flow matching, SmolVLA, OpenVLA, pi0, pi0.5, SigLIP |
| [research/libero-and-lerobot.md](research/libero-and-lerobot.md) | LIBERO suites, LeRobot datasets, install and eval commands, practical notes for this machine |
| [research/lerobot-pipeline.md](research/lerobot-pipeline.md) | How `lerobot-eval` works: rollout loop, OSC controller, action queues, async, LIBERO pre/post steps, how to plug in a dual-system policy |
| [research/robustness-and-ablation.md](research/robustness-and-ablation.md) | LIBERO-Plus and LIBERO-PRO, plus a concrete ablation plan answering the three challenge questions |

## Paper list

| arXiv | Paper | Role for us | PDF |
|---|---|---|---|
| 2502.19417 | Hi Robot (starting paper) | Template: VLM System 2 emits text sub-commands for a VLA | [pdf](papers/2502.19417-hi-robot.pdf) |
| 2503.14734 | GR00T N1 | Dual-system with latent hand-off (VLM tokens into a diffusion transformer) | [pdf](papers/2503.14734-gr00t-n1.pdf) |
| 2410.08001 | RoboDual | Generalist (slow) plus specialist (fast) policy | [pdf](papers/2410.08001-robodual.pdf) |
| 2410.05273 | HiRT | VLM at low frequency, small fast policy conditioned on cached latents | [pdf](papers/2410.05273-hirt.pdf) |
| 2410.15549 | DP-VLA | Large System 2 at low frequency, small System 1 for control | [pdf](papers/2410.15549-dp-vla.pdf) |
| 2505.03912 | OpenHelix | Survey and empirical study of dual-system design choices | [pdf](papers/2505.03912-openhelix.pdf) |
| 2506.01953 | Fast-in-Slow (FiS-VLA) | System 1 embedded in the VLM's final layers | [pdf](papers/2506.01953-fast-in-slow.pdf) |
| 2505.21432 | Hume | System 2 value-guided thinking, evaluated on LIBERO | [pdf](papers/2505.21432-hume.pdf) |
| 2604.24921 | Libra-VLA | Coarse-to-fine dual system, evaluated on LIBERO and LIBERO-Plus | [pdf](papers/2604.24921-libra-vla.pdf) |
| 2403.01823 | RT-H | Language motions as intermediate action layer | [pdf](papers/2403.01823-rt-h.pdf) |
| 2407.08693 | Embodied Chain-of-Thought | Reasoning (plan, sub-task, move, gripper) before acting | [pdf](papers/2407.08693-ecot.pdf) |
| 2304.13705 | ACT | Cheap action-chunking System 1 | [pdf](papers/2304.13705-act.pdf) |
| 2303.04137 | Diffusion Policy | Strong System 1 baseline, expensive to train on CPU | [pdf](papers/2303.04137-diffusion-policy.pdf) |
| 2210.02747 | Flow Matching for Generative Modeling | Basis of flow-matching action heads (pi0, SmolVLA, GR00T) | [pdf](papers/2210.02747-flow-matching.pdf) |
| 2209.03003 | Rectified Flow | Straight-path flow formulation, closely related | [pdf](papers/2209.03003-rectified-flow.pdf) |
| 2006.11239 | DDPM | Diffusion reference for comparison | [pdf](papers/2006.11239-ddpm.pdf) |
| 2506.01844 | SmolVLA | Small VLA with first-class LeRobot and LIBERO support | [pdf](papers/2506.01844-smolvla.pdf) |
| 2406.09246 | OpenVLA | Reference VLA; flat baseline, too heavy for us | [pdf](papers/2406.09246-openvla.pdf) |
| 2410.24164 | pi0 | Flow-matching VLA; backbone of Hi Robot's System 1 | [pdf](papers/2410.24164-pi0.pdf) |
| 2504.16054 | pi0.5 | Co-trained VLA with high-level subtask prediction; LeRobot has a LIBERO checkpoint | [pdf](papers/2504.16054-pi05.pdf) |
| 2303.15343 | SigLIP | Frozen image-text encoder for cached features | [pdf](papers/2303.15343-siglip.pdf) |
| 2306.03310 | LIBERO | The benchmark | [pdf](papers/2306.03310-libero.pdf) |
| 2510.13626 | LIBERO-Plus | Perturbation robustness (7 dimensions); source of OOD ideas | [pdf](papers/2510.13626-libero-plus.pdf) |
| 2510.03827 | LIBERO-PRO | Shows models memorize action sequences; informs ablation design | [pdf](papers/2510.03827-libero-pro.pdf) |

## Key takeaways for this challenge

1. **The common pattern.** System 2 is a VLM that runs rarely. System 1 is a small, fast policy that runs every step, conditioned on System 2's output. Papers differ mainly in the interface: text (Hi Robot, RT-H, ECoT), latent embeddings (GR00T N1, RoboDual, HiRT, OpenHelix), or shared weights (Fast-in-Slow).
2. **Text sub-goals are the easiest to ablate.** With a text interface you can freeze, shuffle, corrupt or oracle-replace the sub-goal string, and each change is a clean counterfactual. This matches the advice in PREREQUISITES.md.
3. **Latent interfaces are criticized.** Libra-VLA argues that static latent hand-offs lag behind the changing scene and are hard to interpret *(verify in paper)*. A text or discrete interface gives interpretability and makes the bottleneck analysis easier.
4. **LIBERO is partly memorized.** LIBERO-PRO and LIBERO-Plus report that models score above 90% on standard LIBERO but collapse under perturbations, and that models largely ignore the language instruction. This is both a risk and an opportunity: if System 1 ignores language, a System 2 ablation will show no effect. The design must force System 1 to depend on the sub-goal, for example with sub-goal-conditioned data and the multi-task `libero_10` / `libero_goal` suites, where the same scene has different goals.
5. **Compute fit.** The CPU-friendly stack is: frozen SigLIP or a small VLM for System 2, cached features, and a small ACT-style or MLP System 1. Fine-tuning SmolVLA (about 450M parameters *(verify)*) is probably too heavy on this laptop. Using the released LeRobot checkpoints for inference or for a baseline is feasible, though slow.
6. **LIBERO needs Linux.** LeRobot's LIBERO extra only installs on Linux, so run everything in WSL2 Ubuntu 22.04 with `MUJOCO_GL=egl` (or `osmesa` if EGL fails, since there is no discrete GPU).

## Suggested next steps

1. Set up WSL, install `lerobot[libero]`, and run one random-policy or pretrained episode to measure sim speed. See [research/libero-and-lerobot.md](research/libero-and-lerobot.md).
2. Inspect `lerobot/libero` (or `libero_10`) frames and language annotations. Decide how to derive sub-goals (segment demos by gripper events, or have a VLM label them).
3. Fix the System 2 to System 1 interface (see the ablation plan in [research/robustness-and-ablation.md](research/robustness-and-ablation.md)).
4. Write the ablation harness first, so every model variant plugs into the same evaluation.
