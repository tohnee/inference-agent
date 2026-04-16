---
name: "llm-serving-opt-skill"
description: "Use when optimizing large-language-model serving with exactness-first rules across benchmark, profiling, CUDA crash triage, KV cache, scheduler, backend, and deployment decisions."
---

# LLM Serving Optimization System

## Overview

This skill is a dedicated operating system for LLM inference and serving optimization.

It is designed for workloads such as:

- decoder-only LLM online serving
- prefill/decode split pipelines
- long-context serving with KV cache pressure
- continuous batching and high-concurrency request scheduling
- production servers built with SGLang, vLLM, TensorRT-LLM, Triton, or custom PyTorch serving paths

This skill complements the sibling modules:

- [e2e-inference-opt-skill](file:///Users/tc/Downloads/推理优化skills/e2e-inference-opt-skill/SKILL.md) for end-to-end inference chains
- [cuda-kernel-opt-skill](file:///Users/tc/Downloads/推理优化skills/cuda-kernel-opt-skill/SKILL.md) for kernel and operator optimization
- [auto-profiling](file:///Users/tc/Downloads/推理优化skills/auto-profiling/SKILL.md) as the bounded orchestration entrypoint

This package is the router for service-level LLM inference skills:

- [`sglang-benchmark-skill`](skills/sglang-benchmark-skill/SKILL.md)
- [`serving-benchmark-skill`](skills/serving-benchmark-skill/SKILL.md)
- [`serving-deployment-skill`](skills/serving-deployment-skill/SKILL.md)

All narrow LLM-serving skills now live under `llm-serving-opt-skill/skills`.
This keeps the repository physically organized under one LLM-serving capability root while preserving a router-plus-subskill architecture.

## First Principle

The default mode is **exact-parity first**.

That means:

- no speedup is accepted if deterministic decode outputs regress
- no speedup is accepted if cache semantics become stale or order-dependent
- no speedup is accepted if scheduler or batching changes break request isolation
- precision, backend, quantization, speculative decoding, and kernel swaps are all gated by explicit parity validation

Only explicit human override should allow bounded numeric drift.

## Core Metrics

Always track the serving metrics that matter for LLMs:

- TTFT: time to first token
- TPOT: time per output token
- ITL: inter-token latency
- end-to-end throughput: requests/s and tokens/s
- p50, p95, and p99 latency under realistic concurrency
- GPU memory footprint and KV cache growth
- scheduler fairness, queueing delay, and cancellation behavior

## Directly Reusable Upstream Patterns

This skill intentionally absorbs operational patterns that are already proven useful upstream.

### From FlashInfer

Use these upstream skill ideas directly when they fit your task:

- `debug-cuda-crash`: API-boundary logging before a CUDA crash
- `benchmark-kernel`: reproducible kernel benchmarking with correctness checks and CUPTI-or-event timing fallback
- `add-cuda-kernel`: source-backed checklist for introducing new CUDA kernels cleanly

FlashInfer is also directly relevant as a serving backend because it focuses on:

- paged and ragged KV cache kernels
- decode, prefill, and append kernels
- cascade attention for shared prefixes
- fused sampling kernels
- CUDAGraph and `torch.compile` compatibility

### From SGLang

Use SGLang's serving workflow as the primary benchmark and deployment reference:

- `bench_serving` for realistic online serving measurements
- `bench_one_batch_server` for single-batch end-to-end latency
- `bench_offline_throughput` for in-process throughput ceilings
- `bench_one_batch` for low-level static-batch profiling
- torch-profiler collection on live serving systems
- deployment cookbook patterns for local, docker, and service-style serving

## When To Use

Use this skill when any of the following is true:

- TTFT is too high for the product SLA
- TPOT or ITL dominates user-visible latency
- throughput collapses when concurrency rises
- decode is slow even though the model fits in memory
- prefill and decode interfere with each other
- KV cache grows too quickly or fragments badly
- FlashInfer, FlashAttention, cuDNN, Triton, or TensorRT-LLM choices need evidence
- a serving stack such as SGLang needs benchmark, profile, triage, or deployment structure

## Quick Route

1. establish serving metrics and exactness contract -> [01_serving_baseline.md](references/01_serving_baseline.md)
2. choose the right benchmark workflow -> [02_benchmark_workflows.md](references/02_benchmark_workflows.md)
3. collect and interpret profiles -> [03_profile_analysis.md](references/03_profile_analysis.md)
4. triage CUDA crashes and bad tensors -> [04_cuda_crash_triage.md](references/04_cuda_crash_triage.md)
5. choose kernels, backends, and compiler lanes -> [05_kernel_backend_playbook.md](references/05_kernel_backend_playbook.md)
6. optimize KV cache and the scheduler -> [06_kv_cache_scheduler.md](references/06_kv_cache_scheduler.md)
7. package and deploy cleanly -> [07_deployment_cookbook.md](references/07_deployment_cookbook.md)
8. convert findings into guarded automation -> [08_agentic_optimization.md](references/08_agentic_optimization.md)

## Multi-Skill Route

Use the narrower sub-skills when the task is already clear:

| Task | Preferred skill |
| --- | --- |
| benchmark an SGLang server | [`sglang-benchmark-skill`](skills/sglang-benchmark-skill/SKILL.md) |
| benchmark across SGLang, vLLM, TensorRT-LLM, Triton, or PyTorch | [`serving-benchmark-skill`](skills/serving-benchmark-skill/SKILL.md) |
| launch and harden service deployments | [`serving-deployment-skill`](skills/serving-deployment-skill/SKILL.md) |
| need kernel, NCU, cache, or correctness deep dives | [cuda-kernel-opt-skill](file:///Users/tc/Downloads/推理优化skills/cuda-kernel-opt-skill/SKILL.md) |
| need E2E small-model or multi-stage chain optimization | [e2e-inference-opt-skill](file:///Users/tc/Downloads/推理优化skills/e2e-inference-opt-skill/SKILL.md) |

## Module Boundary

Stay in this module when the bottleneck is still at the serving-system layer:

- online serving benchmark design
- TTFT / TPOT / ITL diagnosis
- service launch and rollout
- LLM-serving scheduler and cache reasoning before kernel dive

Escalate to `cuda-kernel-opt-skill` after serving evidence shows the real bottleneck is inside an operator or kernel.

## Problem Routing Table

| Need | Start here |
| --- | --- |
| define TTFT/TPOT/ITL baseline | [01_serving_baseline.md](references/01_serving_baseline.md) |
| benchmark a running server | [02_benchmark_workflows.md](references/02_benchmark_workflows.md) |
| inspect kernel overlap or fuse opportunities | [03_profile_analysis.md](references/03_profile_analysis.md) |
| debug illegal memory access or NaN/Inf | [04_cuda_crash_triage.md](references/04_cuda_crash_triage.md) |
| choose FlashInfer vs other backend lanes | [05_kernel_backend_playbook.md](references/05_kernel_backend_playbook.md) |
| reduce KV cache pressure or scheduler stalls | [06_kv_cache_scheduler.md](references/06_kv_cache_scheduler.md) |
| launch or harden SGLang-style deployment | [07_deployment_cookbook.md](references/07_deployment_cookbook.md) |
| turn manual iteration into bounded automation | [08_agentic_optimization.md](references/08_agentic_optimization.md) |

## Standard Working Contract

For each task, hand back:

- workload class: online serving, offline throughput, or kernel triage
- exactness contract
- baseline table with TTFT, TPOT, ITL, throughput, concurrency, and memory
- profiler evidence or crash evidence
- one dominant bottleneck statement
- ranked optimizations with trade-offs
- a minimal safe implementation sequence
- a verification plan

## Guardrails

- Benchmark realistic serving first; do not jump straight to kernel microbenchmarks.
- Use `bench_serving` by default for product-facing conclusions.
- Separate prefill and decode evidence before proposing optimization.
- Never accept a backend or precision switch without parity evidence.
- Treat KV cache semantics and request isolation as correctness, not as performance details.
- Keep deployment changes behind reproducible launch commands and smoke tests.

## Expected Output

When using this skill, Claude should return:

- a concise serving problem statement
- the right benchmark path
- the right profile or crash-triage path
- 3 to 5 evidence-backed optimizations
- a safe experiment order
- a deployment or validation checklist when relevant
