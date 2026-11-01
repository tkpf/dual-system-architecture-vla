# Coding Challenge: Dual-System Architecture for Lifelong Robot Learning

## Background

Current Vision-Language-Action (VLA) models often struggle with long-horizon tasks and compounding execution errors because they conflate high-level semantic reasoning with low-level high-frequency motor control.

Your task is to solve this by designing and evaluating a Dual-System architecture on the LIBERO robot manipulation benchmark.

- **System 2 (High-Level):** Slow, deliberate, and semantic. This module should reason about the task, the scene, and the long-horizon goal.
- **System 1 (Low-Level):** Fast, reactive, and precise. This module should handle the high-frequency continuous control and immediate obstacle avoidance, conditioned on the output of System 2.

## Setup & Infrastructure

To save you from boilerplate data-loading and environment setup, you will use the Hugging Face LeRobot ecosystem.

- **Dataset:** You may use any of the LIBERO datasets hosted on Hugging Face (e.g., `lerobot/libero_10`, `lerobot/libero_spatial`, etc.).
- **Environment:** Use the official `huggingface/lerobot-libero` pipeline for simulation and evaluation.

## Rules & Freedom

You have one week to complete this challenge. You have complete architectural freedom.

- **System 2 Representation:** We do not care how System 2 passes information to System 1. For example, it could be intermediate text generation (chain-of-thought), generated visual waypoints, sub-goal latent embeddings, a topological graph, etc.
- **Model Choice:** You may use pretrained VLMs, Diffusion Policies, Action Chunking (ACT), MLPs, or anything else.
- **Pre-training:** You may leverage external frozen models (e.g., CLIP, SigLIP, or open-source LLMs) to handle the reasoning or APIs.

## The Core Deliverable: The Ablation Study

Building a complex system is easy; proving why it works is hard. Your primary evaluation will be based on your ablation study. You must achieve a respectable success rate on a subset of LIBERO tasks, but more importantly, you must empirically prove that System 2 is actively responsible for improving System 1's performance.

In your write-up, you must answer and demonstrate the following:

1. **The Counterfactual:** What happens to the success rate and step-count if System 2 is removed, frozen, or replaced with a naive baseline (e.g., a static embedding of the initial instruction)?
2. **Error Recovery:** Does System 2 actually help System 1 recover from out-of-distribution perturbations or compounding errors? Show the data.
3. **Bottleneck Analysis:** Is the primary bottleneck in your final system a failure of reasoning (System 2 hallucinating/planning poorly) or a failure of execution (System 1 failing to reach the sub-goal)? How do you know?

## Submission Requirements

Please submit a GitHub repository containing:

1. **The Codebase:** Clean, reproducible code for your Dual-System VLA, including the training loop and the LeRobot evaluation scripts.
2. **A `README.md` Setup Guide:** Instructions on how to run your code, download your specific dataset slice, and execute the evaluation.
3. **Recorded Rollout Video:** Video/GIFs of the policy working inside the benchmark simulation, as well as the output of System 2 if possible.
4. **The Research Report (Max 3 pages):** A PDF detailing your architectural choices, your interface between System 2 and System 1, and your ablation study.
