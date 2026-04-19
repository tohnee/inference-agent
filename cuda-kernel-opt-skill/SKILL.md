---
name: "cuda-kernel-opt-skill"
description: "Use when optimizing CUDA, CUTLASS, or Triton kernels and operator-level bottlenecks with correctness-first benchmark and Nsight Compute evidence."
---

# CUDA Kernel Optimization System

## Overview

This module is dedicated to kernel-level optimization.

It is the right module when the bottleneck has already been narrowed to:

- a CUDA kernel
- a CUTLASS operator
- a Triton kernel
- a fused custom op
- an operator-level memory or scheduler issue that requires Nsight Compute evidence

This module complements:

- [e2e-inference-opt-skill](file:///Users/tc/Downloads/推理优化skills/e2e-inference-opt-skill/SKILL.md) for end-to-end non-LLM chains
- [llm-serving-opt-skill](file:///Users/tc/Downloads/推理优化skills/llm-serving-opt-skill/SKILL.md) for LLM serving systems
- [auto-profiling](file:///Users/tc/Downloads/推理优化skills/auto-profiling/SKILL.md) for bounded experiment orchestration

## Scope

Use this module for:

- correctness + benchmark + NCU loops
- kernel crash triage
- operator bottleneck analysis
- backend selection at the kernel/operator level
- prefix-cache, scheduling, or correctness issues after the problem has already been reduced to operator behavior

## Skills

- [`cuda-crash-debug-skill`](skills/cuda-crash-debug-skill/SKILL.md)
- [`profile-triage-skill`](skills/profile-triage-skill/SKILL.md)
- [`backend-selection-skill`](skills/backend-selection-skill/SKILL.md)
- [`kv-cache-prefix-cache-skill`](skills/kv-cache-prefix-cache-skill/SKILL.md)
- [`scheduler-batching-skill`](skills/scheduler-batching-skill/SKILL.md)
- [`serving-correctness-skill`](skills/serving-correctness-skill/SKILL.md)
- [`remote-gpu-validation-skill`](skills/remote-gpu-validation-skill/SKILL.md)
- [`custom-kernel-workflow-skill`](skills/custom-kernel-workflow-skill/SKILL.md)
- [`operator-backend-synthesis-skill`](skills/operator-backend-synthesis-skill/SKILL.md)
- [`cuda-optimized-skill`](skills/cuda-optimized-skill/SKILL.md)

## Auto-Profiling Link

When `auto-profiling` is used for kernel-level optimization, this module is the primary knowledge source.

Recommended aim template:

- [aim.cuda-kernel.md](file:///Users/tc/Downloads/推理优化skills/auto-profiling/aim.cuda-kernel.md)
