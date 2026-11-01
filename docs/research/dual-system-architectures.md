# Dual-System Architectures

Back to [RESEARCH.md](../RESEARCH.md). Papers are in [../papers/](../papers/).

Terminology: **System 2** is the slow VLM or reasoner. **System 1** is the fast action policy. The "interface" is what System 2 passes to System 1.

## Comparison

| Paper | System 2 | System 1 | Interface | Notes |
|---|---|---|---|---|
| [Hi Robot](../papers/2502.19417-hi-robot.pdf) (2502.19417) | VLM that reasons over prompts and user feedback | pi0-style VLA | Text: next atomic command (e.g. "pick up the cup") | Trained with synthetic prompts and interjections. Beats flat VLAs and a GPT-4o high level on instruction accuracy and task progress (per abstract). Our starting point |
| [GR00T N1](../papers/2503.14734-gr00t-n1.pdf) (2503.14734) | VLM module | Diffusion transformer | Latent VLM tokens through cross-attention | Jointly trained end to end; System 1 about 10 ms latency *(verify)* |
| [RoboDual](../papers/2410.08001-robodual.pdf) (2410.08001) | Generalist VLA | Diffusion-transformer specialist | Latent plus discretized actions from the generalist | Specialist is cheap to train; reduces generalist call rate |
| [HiRT](../papers/2410.05273-hirt.pdf) (2410.05273) | VLM at low frequency | Small high-frequency vision policy | Slowly updated latent features | Explicit frequency vs performance trade-off. Good model for "System 2 once per sub-goal" |
| [DP-VLA](../papers/2410.15549-dp-vla.pdf) (2410.15549) | Large System 2 model, low frequency | Small System 1 model | Intention and reasoning output | Evaluated on RoboCasa. Early, simple dual-process framing |
| [OpenHelix](../papers/2505.03912-openhelix.pdf) (2505.03912) | MLLM | Policy head | Compares prompt tokens, latent tokens, and other design choices | Survey plus empirical study. **Read first** for design-choice evidence; it is the closest thing to a published ablation of the interface |
| [Fast-in-Slow](../papers/2506.01953-fast-in-slow.pdf) (2506.01953) | Full VLM | Last VLM blocks reused as System 1 | Shared parameters | Joint diffusion plus autoregressive objective. Needs a large model, so not feasible here |
| [Hume](../papers/2505.21432-hume.pdf) (2505.21432) | VLA backbone with a value-query head | Cascaded action denoising | Value-guided candidate selection | Reports LIBERO results; checkpoints on Hugging Face (`Hume-vla`). System 2 acts by sampling candidates and scoring them, not by emitting sub-goals |
| [Libra-VLA](../papers/2604.24921-libra-vla.pdf) (2604.24921) | Semantic planner for coarse, discrete macro-direction | Diffusion refiner for continuous fine alignment | Discrete coarse action | Evaluated on LIBERO and LIBERO-Plus. Argues against static or opaque latents *(verify)* |

## Observations relevant to us

- **Update rate.** Every paper runs System 2 slower than System 1. For us System 2 can run once per sub-goal, or every N steps, and re-plan on failure. Re-planning frequency is itself an ablation axis.
- **Ablation precedent.** Look for these in OpenHelix, HiRT, Libra-VLA and Hume: with and without System 2, different interface forms, System 2 update frequency. Reuse their framing in the report.
- **Hi Robot's recipe, scaled down.**
  1. Segment demonstrations into sub-skills and label them with short text commands.
  2. Train System 1 on (observation, command) to action.
  3. System 2 maps (observation, task) to the next command.
  Hi Robot used a large pi0 backbone, so the small-model version should use a cached-feature policy instead.
- **Value of text.** A text interface gives readable System 2 output for the rollout videos, which the challenge asks for.
