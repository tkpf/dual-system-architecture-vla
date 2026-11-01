# Language as the System 2 to System 1 Interface

Back to [RESEARCH.md](../RESEARCH.md).

## RT-H: Action Hierarchies Using Language ([pdf](../papers/2403.01823-rt-h.pdf))

- Idea: between a task ("pick coke can") and actions, insert **language motions** such as "move arm forward". The policy first predicts the language motion, then the action conditioned on it.
- Motivation: tasks that are semantically far apart share little data, but their low-level motions are shared.
- Humans can correct the robot in language motions, and corrections can be learned from.
- Relevance: a model for **fine-grained** sub-goals. A System 2 that emits motion-level or phase-level strings gives System 1 a richer signal than the raw task string.

## Embodied Chain-of-Thought ([pdf](../papers/2407.08693-ecot.pdf))

- A VLA is trained to emit reasoning before the action: task plan, sub-task, movement primitive, object bounding boxes and gripper position, then the action.
- The abstract notes that purely semantic reasoning about sub-tasks is not enough; grounded reasoning (bounding boxes, gripper position) matters *(verify details)*.
- Relevance: intermediate text or grounded output is cheap to read and to corrupt for ablations. Also a caution: a long chain of thought slows inference a lot, which hurts on CPU.

## Hi Robot ([pdf](../papers/2502.19417-hi-robot.pdf))

See [dual-system-architectures.md](dual-system-architectures.md). The relevant design lessons:

- The high level runs at a lower rate and issues atomic commands.
- Training data for the high level is derived from the same robot demonstrations, with commands labeled per segment.
- Evaluation measures **instruction accuracy** and **task progress**, which are good metrics to adopt: instruction accuracy for System 2 quality, task progress for System 1 execution.

## Design options for our interface

| Option | Pros | Cons |
|---|---|---|
| Sub-goal text string, encoded with a frozen text encoder (e.g. SigLIP text tower) | Easy to ablate (freeze, shuffle, corrupt, oracle), readable in videos | Needs sub-goal labels |
| Phase or skill index (discrete) | Trivial System 1 conditioning, easy to measure accuracy | Fixed vocabulary, less general |
| Visual sub-goal (future frame or waypoint) | Rich | Needs a generative model; heavy on CPU |
| Latent embedding (GR00T-style) | Often best on benchmarks | Hard to interpret; bottleneck analysis harder |

Recommendation (consistent with PREREQUISITES.md): text sub-goal, encoded once per sub-goal by a frozen encoder and cached.
