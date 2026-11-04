# System 1 Candidates and Frozen Encoders

Back to [RESEARCH.md](../RESEARCH.md).

**Decision so far:** System 1 is pi0.5 (`lerobot/pi05_libero_finetuned_v044`), which replaced SmolVLA after a direct comparison (see [EXPERIMENTS.md](../EXPERIMENTS.md)). System 2 is Qwen3-VL-2B for now, but it misreads the scene, and a perception benchmark is comparing alternatives. Background on System 2 is in [System 2: frozen VLM that emits atomic sentences](#system-2-frozen-vlm-that-emits-atomic-sentences) at the end of this page.

## Trainable low-level policies

| Model | Paper | Fit for this laptop | Notes |
|---|---|---|---|
| ACT | [pdf](../papers/2304.13705-act.pdf) | Best | Transformer predicting action chunks. Native in LeRobot. Chunking reduces compounding error. Smallest realistic choice for training on CPU, especially with a lightweight visual backbone or cached features |
| Diffusion Policy | [pdf](../papers/2303.04137-diffusion-policy.pdf) | Poor to medium | Strong multimodal action distributions, but many denoising steps at inference and slow training. Likely too slow for repeated evaluation on CPU |
| MLP head on frozen features | n/a | Best for speed | Fastest to train and iterate; likely lower success on precise tasks. Useful as a minimal baseline |
| SmolVLA | [pdf](../papers/2506.01844-smolvla.pdf) | Inference only | Small VLA built for LeRobot; documented on LIBERO. Fine-tuning needs a GPU. Candidate as an **external baseline** or as System 1 if a checkpoint is available |
| OpenVLA | [pdf](../papers/2406.09246-openvla.pdf) | No | About 7B parameters. Reference for flat VLA numbers only |
| pi0 / pi0.5 | [pi0](../papers/2410.24164-pi0.pdf), [pi0.5](../papers/2504.16054-pi05.pdf) | No for training | LeRobot hosts a finetuned pi0.5 LIBERO checkpoint (`lerobot/pi05_libero_finetuned_v044`). pi0 is Hi Robot's System 1. See [Pretrained LIBERO VLAs as a frozen System 1](#pretrained-libero-vlas-as-a-frozen-system-1) |

## Pretrained LIBERO VLAs as a frozen System 1

Option: use an already LIBERO-finetuned VLA with frozen weights as System 1, and let System 2 supply the language prompt (the Hi Robot interface). LIBERO simulates a Franka Panda, so these checkpoints are native to the robot. They are **not zero-shot**: they were finetuned on LIBERO demonstrations, so all benchmark tasks are in-distribution.

Success rates below: pi0.5 rows are from the LeRobot LIBERO docs and the OpenPI README; the others are from memory and must be verified before citing.

| Model | Architecture | LIBERO (Spatial / Object / Goal / 10, avg) | Notes |
|---|---|---|---|
| pi0.5 (OpenPI `pi05_libero`) | PaliGemma 3B plus flow-matching action expert; co-training and discrete tokens in training | 98.8 / 98.2 / 98.0 / 92.4, avg 96.85 | Authors' numbers. JAX checkpoint, so run in openpi or convert to PyTorch (conversion unverified) |
| pi0.5 (`lerobot/pi05_libero_finetuned_v044`) | Same, plus 6k extra finetune steps by LeRobot | 97 / 99 / 98 / 96, avg 97.5 (10 episodes per suite) | Small sample, differences vs OpenPI are within noise. Reason for the extra steps not found. The name without `_v044` was not on the Hub |
| pi0 (`pi0_libero`) | PaliGemma 3B plus flow-matching expert, 50-step chunks | about 96.8 / 98.8 / 95.8 / 85.2, avg about 94 (memory) | Hi Robot's actual System 1. Weaker on LIBERO-10, so more headroom for a System 2 ablation. LeRobot repo name unverified |
| pi0-FAST | Same VLM, autoregressive DCT-compressed action tokens | avg about 85 (memory) | Lower. Not a candidate |
| OpenVLA-OFT | 7B Llama-2, fused SigLIP and DINOv2, parallel decoding, action chunks, L1 head | avg about 97 (memory) | Heavy, GPU only |
| SmolVLA | about 450M, SmolVLM2 with skipped layers plus flow-matching expert | avg about 87 (memory) | Realistic on a modest GPU, native in LeRobot |
| GR00T N1.5 | Frozen Eagle VLM plus DiT flow-matching head | not checked | Same VLM plus action head split as this challenge |
| Octo | 93M transformer, diffusion head | poor without finetuning | Not a frozen option |

pi0.5-DROID targets a Franka but uses real-world DROID cameras and a joint-space action interface, so it is not a drop-in zero-shot option for LIBERO sim.

**Interface.** The LIBERO checkpoints expect end-effector delta actions (6 pose dims plus gripper), two cameras and an 8-dim state. Any perturbation wrapper must preserve this format. Evaluate with `n_action_steps=10`, which LeRobot says matches OpenPI.

**Risks for the ablation.**
1. The base policy is already near ceiling on the plain instruction (about 97%), so a "static initial instruction" baseline may match any System 2 that only rewords or splits it. Headroom needs harder settings: chained long-horizon tasks, perturbations, unseen instructions.
2. My recollection (verify: LIBERO-PRO and LIBERO-Plus papers) is that LIBERO-finetuned VLAs often ignore the instruction and replay memorised visual trajectories. If so, a frozen System 1 may not respond to sub-goal text at all.
3. Compute: about 3B parameters, so CPU inference takes seconds per chunk. A GPU is needed at least for inference.

**First experiment.** Run the candidate on a few LIBERO-10 tasks with correct, wrong and empty prompts. The success drop measures how much System 1 uses language and decides whether it is usable as a frozen System 1. Test pi0 and pi0.5 and pick the one whose success responds to the prompt.

**What was found (2026-11-01).** On scheduled atomic commands from shifted starts, pi0.5 reached the goal in 5/12 episodes and SmolVLA in 0/12. Risk 2 above still applies: pi0.5 sometimes finished a task under a command that meant something else, and no gibberish control has been run on it. The first-experiment comparison of pi0 was not run. Details are in [EXPERIMENTS.md](../EXPERIMENTS.md).

**Fallbacks.** On CPU only: SmolVLA (frozen backbone, trained action expert) or ACT on cached features.

## Flow matching as the action head

Flow matching is the generative action head used by several of the VLAs above (pi0 and pi0.5, and per my understanding SmolVLA and GR00T N1; verify in each paper). It is a close relative of diffusion and is worth understanding before choosing System 1.

Papers: [Flow Matching for Generative Modeling](../papers/2210.02747-flow-matching.pdf) (2210.02747), [Rectified Flow](../papers/2209.03003-rectified-flow.pdf) (2209.03003), and [DDPM](../papers/2006.11239-ddpm.pdf) (2006.11239) as the diffusion reference.

**How it works.** Instead of learning to denoise step by step, the network learns a velocity field that moves a noise sample toward a data sample. For action chunks:
- Training: take a real action chunk `a`, sample noise `e`, sample a time `t` in [0, 1], form the interpolation `x_t = (1 - t) * e + t * a`, and regress the network output (conditioned on observation and sub-goal) to the target velocity `a - e`. This is a plain regression loss with no noise schedule to tune. (Exact conventions for `t` vary between papers; pi0 flips the direction.)
- Inference: start from noise and integrate the velocity field with a few Euler steps (around 10 in pi0, per my recollection; verify).

**Compared to diffusion policy.**

| | Diffusion Policy | Flow matching |
|---|---|---|
| Training target | Predict noise (or sample) at a random noise level | Predict velocity along a straight path |
| Inference steps | Often tens to 100 | Typically about 10, sometimes fewer |
| Hyperparameters | Noise schedule, sampler | Few: number of Euler steps |
| Multimodal actions | Yes | Yes |

**Fit for this challenge.**
- Cheaper inference than diffusion is useful on CPU, and the loss is as simple as an MLP regression, so a small flow-matching head on cached features is realistic to train. It is a candidate for System 1 alongside ACT.
- It keeps multimodal action distributions, which ACT's L1 regression (without a latent variable) does not. This matters if demonstrations of the same sub-goal differ.
- Stochastic sampling adds variance between rollouts. For ablations, fix the noise seed or average over several seeds, otherwise condition differences can be confused with sampling noise.
- Risk: more moving parts than ACT (time conditioning, integration loop). If time is short, ACT or an MLP is the safer first System 1. Flow matching can be a second variant.
- Hi Robot's low level is pi0, a flow-matching VLA, so a flow-matching System 1 is also the closest small-scale analogue of the starting paper.

## Frozen encoders

- **SigLIP** ([pdf](../papers/2303.15343-siglip.pdf)): sigmoid-loss image-text model. Use for image features and for text embeddings of sub-goals. Small variants run on CPU; precompute features once per frame and per sub-goal string.
- **CLIP**: allowed by the challenge. Interchangeable with SigLIP for this purpose.
- **Small open VLMs** for System 2 (candidates to evaluate for CPU latency; check Hugging Face for current versions): SmolVLM family, Qwen-VL small variants, PaliGemma. System 2 runs rarely, so seconds per call may be acceptable. Measure it early.
- **API VLM** (allowed by the challenge): fastest path to a strong System 2, with no local compute. Costs money and adds latency and non-determinism, which complicates reproducibility. If used, cache outputs by (task, state-hash) and log every call.

## Practical rule

Cache everything the frozen parts produce (image embeddings per frame, text embeddings per sub-goal). Training System 1 then only touches the cached features. See [../PREREQUISITES.md](../PREREQUISITES.md).

## System 2: frozen VLM that emits atomic sentences

**Interface (decided).** System 2 receives the long task instruction plus the current camera image(s) and returns one atomic English sentence, for example "pick up the alphabet soup". SmolVLA gets that sentence as its language prompt. When System 2 judges the sub-task done, it emits the next sentence. This is the Hi Robot / RT-H interface (see [language-interfaces.md](language-interfaces.md)) and gives the cleanest ablations (freeze, shuffle, corrupt, oracle).

Sizes, benchmark numbers and release details below come from web search snippets and my memory. Verify each on its Hugging Face model card before citing.

### What System 2 has to do

1. **Decompose** the instruction into atomic steps (once per episode, or re-planned).
2. **Track progress**: decide from the current frame whether the current step is finished. This is the harder part. A plan-once VLM with a fixed step budget is a fallback.
3. **Stay in System 1's vocabulary.** SmolVLA was finetuned on LIBERO instructions, so sentences far from that phrasing may be ignored or misread. Which LIBERO data the SmolVLA checkpoint was trained on, and whether atomic sub-sentences such as "put the bowl on the plate" occur in it, is *not yet checked*. Check this first, because it decides how much of the VLM's freedom is usable.

**Measured limitation.** On 2026-11-01, Qwen3-VL-2B misread frames even with explicit finish conditions: it called a glowing red burner dark and a held bowl on the table. See [EXPERIMENTS.md](../EXPERIMENTS.md).

### Candidates

| Model | Size | Why consider it | Concerns |
|---|---|---|---|
| **Qwen3-VL** ([tech report](https://arxiv.org/pdf/2511.21631)) | 2B, 4B, 8B (Instruct and Thinking variants) | Third-party guides rank Qwen3-VL as the strongest open family below about 35B ([Labellerr](https://www.labellerr.com/blog/top-open-source-vision-language-models/), [tinyweights](https://tinyweights.dev/posts/best-local-vision-language-models-2026/)). Small sizes exist, good instruction following and structured output. **Recommended default** | No embodied-planning scores found for the 2B. On ERQA, Qwen3-VL-8B is reported at 41.8 vs 49.5 for the 235B ([ERQA-Plus page](https://flageval-baai.github.io/ERQA-Plus-page)), so small models are clearly weaker at embodied reasoning |
| **Gemini Robotics-ER 1.5** ([blog](https://developers.googleblog.com/building-the-next-generation-of-physical-agents-with-gemini-robotics-er-15/), [paper](https://arxiv.org/pdf/2510.03342v3)) | API only (`gemini-robotics-er-1.5-preview`) | Built for task planning, spatial reasoning and progress estimation. Used as the orchestrator over a VLA in Google's own system. Best fit by design | Public preview, API cost and non-determinism, cannot be frozen in the weight sense (the version can change). Allowed by the challenge, but cache every call. Good as an upper-bound or "strong System 2" variant |
| **Cosmos-Reason1-7B** ([HF](https://huggingface.co/nvidia/cosmos-reason1-7b), [paper](https://arxiv.org/pdf/2503.15558)) | 7B, Qwen2.5-VL-7B base | Post-trained for physical common sense and embodied reasoning; positioned as a planner for embodied agents | Long chain-of-thought outputs, so slow. Built for video input. Needs a GPU |
| **RoboBrain 2.0** ([paper](https://arxiv.org/abs/2507.02029)) | 7B and 32B | Perception, reasoning and long-horizon planning for embodied tasks; code and weights public | 7B needs a GPU. Prompt format for planning not checked |
| **SmolVLM2** | 256M, 500M, 2.2B | Same family as SmolVLA's backbone *(verify)*, tiny, runs on CPU. Natural "small System 2" | Weak reasoning at this size; likely to hallucinate decompositions. Useful as a weak-System-2 baseline, not as the main choice |
| **InternVL3.5-4B** | 4B | Alternative family; used for local robot-image tasks in one paper ([arXiv 2608.12515](https://arxiv.org/pdf/2608.12515)) | No planning evidence found |
| **Gemma 3 4B** | 4B | Easiest runtime support (e.g. Ollama) per [tinyweights](https://tinyweights.dev/posts/best-local-vision-language-models-2026/) | Reported below Qwen3-VL-4B on vision benchmarks |
| Qwen2.5-VL-3B | 3B | Older, widely supported, base of several robotics VLMs | Superseded by Qwen3-VL |
| PaliGemma | 3B | Backbone of pi0 / pi0.5 | Not instruction-tuned for chat-style planning; skip |

Not found in my search (follow up if wanted): Embodied-R1, Magma, and any head-to-head of small VLMs on LIBERO-style decomposition.

### Recommendation

1. **Main System 2: Qwen3-VL-4B-Instruct** (8B if it fits). With a Colab GPU (assumed T4, 16 GB; check with `nvidia-smi`), 4B in fp16 should take about 8 to 9 GB and leave room for SmolVLA in the same session. T4 has no native bf16, so use fp16. 8B in fp16 is too tight; use 8-bit or 4-bit. Frozen, prompted with few-shot examples in LIBERO phrasing.
   - **Latency on GPU (estimate, not measured):** a bare call that returns only the sentence or a yes/no should take well under 1 s for 2B to 4B with plain transformers. **This excludes reasoning.** Chain-of-thought (decomposing the task, locating the robot, noticing a missed grasp) is 100 to 500+ tokens, and Thinking variants far more. Decoding then dominates; at an assumed 20 to 40 tokens/s for 4B fp16 on a T4, 200 tokens is about 5 to 10 s. Measure it.
   - **Split calls by purpose.** Plan (once per episode, or on replan): full reasoning is acceptable. Progress check (repeating): short structured output such as `{"gripper_holds_object": bool, "step_done": bool}`, capped near 30 tokens. Replan: only when the check reports a failure such as a dropped object. Prefer Instruct over Thinking for repeating calls.
   - **Measured smoke test (2026-11-01, Colab T4, Qwen3-VL-2B-Instruct fp16, transformers 5.5.4, `generate` greedy, 256x256 `lerobot/libero_10` frames, median of 8 frames, 4.3 GB GPU):** prefill is about 0.06 to 0.1 s for 88 to 161 input tokens, and decode is about 50 ms per token (about 20 tokens/s) regardless of output length. Total time is therefore set by output length.

     | Prompt | Output tokens | Total |
     |---|---|---|
     | A. Plan, text only, JSON list | 27 | 1.4 s |
     | B. Plan with image, `scene` + `steps` JSON | 89 | 4.5 s |
     | C. Done-check JSON (two booleans) | 20 | 1.0 s |
     | D. Free-form chain-of-thought, capped at 400 | 342 | 16.8 s |

     Output was only eyeballed on a few frames (plans were sensible), not scored. About 8 of the 20 tokens in C are the markdown code fence and JSON syntax; asking for a bare `true`/`false` or scoring a closed set would cut them. A 2B model at 50 ms per token is probably overhead-bound, not compute-bound, so a 4B model may not be proportionally slower; not measured. The script is not saved in the repo yet.
   - **Reasoning on/off is an ablation:** bare sentence vs. short structured reasoning vs. long chain-of-thought. Only the reasoning variants can notice a missed grasp, so this feeds the error-recovery question.
   - **Low-latency small models (Qwen3-VL-2B, SmolVLM2-2.2B)** become the *weak reference* in the ablation, not the main choice. Local CPU (Q4 GGUF in llama.cpp, roughly 1 to 5 s per call, unmeasured) remains a fallback if Colab is unavailable.
2. **Strong reference: Gemini Robotics-ER 1.5** through the API, with a cache. It tells you how much a better System 2 helps, which feeds the bottleneck analysis.
3. **Weak reference: SmolVLM2-2.2B** or Qwen3-VL-2B. If success tracks System 2 quality (weak < main < strong), that is direct evidence that System 2 matters.
4. Skip Cosmos-Reason1 and RoboBrain 2.0 unless time remains; they are heavier and the gain over a prompted Qwen3-VL for simple LIBERO decomposition is unproven.

### Making "frozen" work: no training, so constrain the output

- **Few-shot prompt** with 3 to 5 LIBERO instructions and their atomic decompositions.
- **Constrained generation**: ask for JSON (`{"subtask": "...", "done": true/false}`) or restrict the sentence to a verb list ("pick up", "put ... on/in", "open", "close", "turn on") and the object names present in the scene.
- **Best-of closed set**: give the VLM the list of sub-task sentences System 1 is known to handle and have it choose. This removes vocabulary mismatch and makes System 2 errors easy to classify (wrong choice vs. System 1 failing the right choice).
- **Low rate**: call System 2 only when the current sub-task is judged done, or every N steps for the done check. Cache outputs by (episode, step).
- **Temperature 0** for reproducibility.

### Ablation hooks this enables

| Variant | Meaning |
|---|---|
| Full instruction only | No System 2 (naive baseline from CHALLENGE.md) |
| Oracle sub-goals | Hand-written or demo-derived sentences; upper bound for reasoning |
| Shuffled or wrong sub-goal | Shows whether SmolVLA reads the sentence at all |
| Frozen first sub-goal | Never advances; tests the progress-tracking part |
| Weak / main / strong VLM | Dose-response of System 2 quality |
| Perturbation (object moved mid-episode) | Does re-planning from the new frame recover? |

Bottleneck analysis: if oracle sub-goals fix most failures, System 2 is the bottleneck. If they do not, System 1 execution is.

### First experiments (before building anything)

1. Load Qwen3-VL-4B, feed it a LIBERO frame and a `libero_10` instruction, and look at the decomposition and latency.
2. Run SmolVLA on a few tasks with correct, atomic, wrong and empty prompts (same test as the pi0 note above). If SmolVLA ignores or fails on atomic sentences, the whole interface needs adjusting before System 2 matters.
