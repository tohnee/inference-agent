---
name: "cuda-optimized-skill"
description: "Use when optimizing custom CUDA, CUTLASS, or Triton operators with a correctness-benchmark-NCU-strategy-memory loop."
---

# CUDA Optimized Skill

This skill integrates the upstream `cuda-optimized-skill` toolkit into the current project under MIT license.

It is the operator-level companion to the serving-level skills in this repository.

Use it when:

- the bottleneck has already been narrowed to a kernel or custom operator
- you need reproducible correctness + benchmark + NCU evidence
- you want multi-round optimization with strategy memory
- you are working on CUDA, CUTLASS, or Triton operator implementations

## Main Components

- [`kernel-benchmark`](kernel-benchmark/SKILL.md)
- [`ncu-rep-analyze`](ncu-rep-analyze/SKILL.md)
- [`operator-optimize-loop`](operator-optimize-loop/SKILL.md)
- [`reference`](reference/)

## Integrated Workflow

1. run preflight checks
2. validate correctness against a reference when available
3. benchmark the operator
4. collect targeted NCU
5. collect full NCU
6. write an optimization proposal with strategy tags
7. generate the next version
8. record positive, negative, or rejected strategy memory

## Use With

- [`custom-kernel-workflow-skill`](../custom-kernel-workflow-skill/SKILL.md) to decide whether custom kernel work is justified
- [`profile-triage-skill`](../profile-triage-skill/SKILL.md) to connect serving evidence to operator evidence
- [`serving-correctness-skill`](../serving-correctness-skill/SKILL.md) to preserve parity gates

## Attribution

- upstream source integrated from `KernelFlow-ops/cuda-optimized-skill`
- local copy includes `README.md`, `LICENSE`, scripts, strategy-memory seed file, and reference docs
