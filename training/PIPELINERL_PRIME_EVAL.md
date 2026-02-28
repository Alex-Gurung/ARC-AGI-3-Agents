# PipelineRL / PRIME-RL Fit Check

This note evaluates whether `PipelineRL` or `PRIME-RL` should replace the
current ARC training paths (`training/openrlhf`, `training/verl`).

## Current Constraints

- Single-GPU first (`1x` GPU baseline runs must remain viable).
- Multi-turn, environment-coupled rollouts (LoopAgent boundaries, memory updates).
- On-policy or near-on-policy GRPO-style updates preferred.
- Need practical implementation speed, not just theoretical throughput.

## PipelineRL

Reference:

- https://github.com/microsoft/rsl-pipelinerl
- https://arxiv.org/abs/2509.19128

What it provides:

- Distributed pipeline-style RL with overlap of generation and updates.
- Throughput-oriented design for RL post-training.

Fit for this repo:

- Strong for large-scale multi-GPU throughput.
- Weak fit for current single-GPU + fast-iteration phase:
  - higher integration cost than OpenRLHF path already in-tree
  - less immediate benefit without multi-node/multi-GPU overlap
  - would still require custom LoopAgent environment glue and boundary logging

Verdict now:

- Not the first backend to operationalize here.
- Revisit when scaling to multi-node training where pipeline overlap matters.

## PRIME-RL

Reference:

- https://github.com/PRIME-RL/PRIME
- https://docs.prime-rl.org/
- https://www.arxiv.org/abs/2509.20113

What it provides:

- Full async RL stack (orchestrator + trainer + inference manager + verifier envs).
- Native support for GRPO/GSPO and async updates.
- Explicit online RL entrypoints and verifier environment abstractions.

Fit for this repo:

- Closer conceptual fit than PipelineRL for environment-coupled online RL.
- Still non-trivial integration:
  - LoopAgent-style boundary rollouts need verifier wrappers
  - memory/state schemas and grouped candidate traces need adapters
  - async policy lag controls need tuning to stay near on-policy

Verdict now:

- Promising as next backend after OpenRLHF/veRL stabilizes.
- Best treated as a dedicated integration project, not a quick swap.

## Recommendation

1. Keep OpenRLHF as the immediate trainable path for full staged LoopAgent rollouts.
2. Keep veRL path for explicit grouped branching + semantic surprise experiments.
3. Add PRIME-RL as next integration target (higher ROI than PipelineRL for ARC loop semantics).
4. Defer PipelineRL unless/ until multi-GPU throughput becomes the bottleneck.
5. Reuse the shared boundary reward event contract:
   - `training/BOUNDARY_REWARD_SCHEMA.md`
   - `training/boundary_event.schema.json`

## TODO (Follow-up)

- Build a minimal PRIME verifier wrapper around ARC `ls20` with boundary-level events.
- Port one reward path first (`debiased_nll` or heuristic surprise), then expand.
- Add policy-lag diagnostics when running async PRIME updates.
- Compare sample-efficiency and wall-clock vs OpenRLHF staged path on the same seed budget.
