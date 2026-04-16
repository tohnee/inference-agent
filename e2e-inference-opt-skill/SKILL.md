---
name: "e2e-inference-opt-skill"
description: "Use when optimizing end-to-end inference pipelines for small or non-LLM workloads while preserving exact inference results across preprocess, model forward, postprocess, batching, caching, and deployment."
---

# E2E Inference Optimization System

## Overview

This skill is a system-level playbook for end-to-end inference optimization in production-like environments.

- Hardware: CPU, A40, L20, A100, H100, and virtualized variants
- Model families: classical ML, deep learning, Transformer, ViT, diffusion, retrieval and embedding models
- Codebase style: Python + PyTorch first, with ONNX, TensorRT, Triton, Torch-TensorRT, custom CUDA/Triton kernels as downstream options

Core principle:

1. preserve exact inference results first
2. build a trustworthy baseline
3. profile the real workload
4. use roofline thinking to classify the bottleneck
5. optimize one dominant bottleneck at a time
6. verify latency, throughput, memory, exactness, and stability after every change

This skill is intentionally generic for non-LLM or small-model pipelines.

Use it for:

- CV, NLP, ranking, retrieval, rerank, tabular, classical ML, and mixed multi-stage inference paths
- preprocess -> infer -> postprocess -> store/serve style pipelines
- CPU/GPU co-design and E2E latency decomposition

Do not use it as the primary module for:

- autoregressive LLM serving optimization
- kernel-level CUDA/CUTLASS/Triton operator work

Those belong to:

- [llm-serving-opt-skill](file:///Users/tc/Downloads/推理优化skills/llm-serving-opt-skill/SKILL.md)
- [cuda-kernel-opt-skill](file:///Users/tc/Downloads/推理优化skills/cuda-kernel-opt-skill/SKILL.md)

## First Principle

The default operating mode of this skill is **exact-parity mode**.

That means:

- no optimization is acceptable if it changes model results
- no optimization is acceptable if it introduces stale cache outputs or request-order bugs
- no optimization is acceptable if it changes numerical precision unless the deployed baseline already uses that precision
- if exact equality cannot be proven, the change is rejected by default

Only explicit human override should allow non-zero output drift.

## When To Use

Use this skill when any of the following is true:

- p50 or p95 latency is above SLA
- throughput does not scale with batch or concurrency
- GPU utilization is low but latency is still high
- CPU is hot, worker count grows, but requests do not get faster
- model is OOM, KV cache grows too fast, or fragmentation is severe
- a team wants to evaluate `torch.compile`, FlashAttention, FlashInfer, quantization, ONNX, TensorRT, or Triton
- there is disagreement about whether the bottleneck is IO, memory, compute, launch overhead, synchronization, or scheduling

Do not use this skill as the primary tool for:

- training-time optimization
- model quality research unrelated to inference
- distributed training, ZeRO, or NCCL tuning

## Default Working Contract

For every optimization task, produce these deliverables before recommending large changes:

- exactness contract for the workload
- hardware and software fingerprint
- workload definition and business metric definition
- baseline table with mean, p50, p95, p99, throughput, memory, exactness guardrail
- profiling evidence
- bottleneck classification
- ranked optimization backlog
- verification result after each experiment

Never recommend an optimization solely because it is fashionable. The burden of proof is benchmark plus profiler evidence.

## Golden Principles

- preserve exactness before pursuing speed
- measure first, optimize second
- separate cold start, warm path, and burst path
- optimize the dominant bottleneck, not the most interesting bottleneck
- validate end-to-end business metrics, not only kernel microbenchmarks
- treat vGPU, MIG, and noisy-neighbor effects as real system constraints
- require both performance evidence and exactness evidence for every change

## Supported Environments

- hardware: CPU-only, A40, L20, A100, H100, and virtualized GPU slices
- models: Transformer, decoder-only LLM, ViT, diffusion, CNN, retrieval encoder, tabular or classical ML
- runtime: Python, PyTorch, Torch-TensorRT, ONNX Runtime, TensorRT, Triton, specialized LLM servers

## Quick Start

### Step 0: classify the workload

- **Single-shot latency path**: online search, vision retrieval, classification, rerank
- **Steady-state throughput path**: offline batches, streaming pipelines, large queue backlogs
- **Autoregressive path**: LLM decode, KV cache, continuous batching, prefill/decode split
- **Multi-stage path**: preprocess → infer → postprocess → store/serve

### Step 0.5: define the exactness contract

- exact tensor equality when deterministic execution is feasible
- exact decoded result equality for deterministic generation
- exact ranking or label equality only if tensor equality is not the product contract
- identical cache semantics, preprocessing semantics, and request isolation

### Step 1: establish baseline

Go to [01_baseline.md](references/01_baseline.md).

### Step 2: collect profiling evidence

Go to [02_profiling.md](references/02_profiling.md).

### Step 3: classify compute-vs-bandwidth limit

Go to [03_roofline.md](references/03_roofline.md).

### Step 4: route to the main bottleneck

- kernel / matmul / attention / compiler issue → [04_compute_opt.md](references/04_compute_opt.md)
- H2D, D2H, format conversion, cache miss, allocator issue → [05_memory_io.md](references/05_memory_io.md)
- batching, concurrency, process model, multi-GPU layout → [06_parallelism.md](references/06_parallelism.md)
- stage overlap, async pipeline, two-level cache, decode serving path → [07_pipeline_cache.md](references/07_pipeline_cache.md)
- export, engine build, serving stack, rollout → [08_deployment.md](references/08_deployment.md)

## Problem Routing Table

| Need | Start here |
| --- | --- |
| do not know how to begin | [01_baseline.md](references/01_baseline.md) |
| need latency, throughput, or concurrency numbers | [01_baseline.md](references/01_baseline.md) |
| need hotspot evidence | [02_profiling.md](references/02_profiling.md) |
| unsure whether the limit is compute or bandwidth | [03_roofline.md](references/03_roofline.md) |
| need attention, precision, compile, or fused-kernel choices | [04_compute_opt.md](references/04_compute_opt.md) |
| need H2D, D2H, pinned memory, allocator, or KV memory guidance | [05_memory_io.md](references/05_memory_io.md) |
| need batching, concurrency, multi-process, or multi-GPU guidance | [06_parallelism.md](references/06_parallelism.md) |
| need pipeline overlap or cache design | [07_pipeline_cache.md](references/07_pipeline_cache.md) |
| need ONNX, TensorRT, Triton, or production rollout guidance | [08_deployment.md](references/08_deployment.md) |

## Symptom Routing Map

| Symptom | First stop | Likely next stop |
| --- | --- | --- |
| p95 latency unstable | 01 baseline | 02 profiling, 07 pipeline |
| GPU util < 40% with high latency | 02 profiling | 03 roofline, 05 memory IO |
| GPU util > 90% and latency still high | 03 roofline | 04 compute |
| OOM or memory spikes | 01 baseline | 05 memory IO, 07 pipeline |
| batch size grows but throughput plateaus | 02 profiling | 03 roofline, 06 parallelism |
| LLM serving tail latency too high | 01 baseline | 06 parallelism, 07 pipeline, 08 deployment |
| PyTorch path tops out early | 04 compute | 08 deployment |

## Standard Optimization Loop

Run this loop in order:

1. define the exact serving scenario
2. record cold and warm baselines
3. profile the slow path
4. identify the single highest-value bottleneck
5. design one experiment with one changed variable that preserves exactness
6. verify latency, throughput, memory, cost, and exactness
7. keep or revert based on evidence

## Optimization Priority Heuristics

Prioritize in this order unless evidence shows otherwise:

1. fake work elimination  
   repeated preprocessing, repeated embedding build, CPU postprocess in hot path, unnecessary copies
2. data movement reduction  
   host-device copies, dtype inflation, layout conversion, KV cache traffic
3. batching and scheduling  
   microbatching, continuous batching, stage overlap, worker/process topology
4. kernel and graph optimization  
   fused attention, compile, cuda graph, fused preprocessing, custom op
5. engine-level deployment  
   ONNX, TensorRT, Triton only after PyTorch path and profiler justify the migration cost

In exact-parity mode, prefer optimizations that preserve numerical behavior:

1. eliminate fake work
2. reduce copies without changing math
3. improve scheduling and overlap without changing math
4. compile or fuse only after parity is proven
5. change runtime or backend only after full golden-set verification

## Guardrails

- Always separate cold start, warm steady-state, and burst scenarios
- Always define what exact equality means for this workload before optimizing
- Always reject lower precision, quantization, or approximate kernels unless the human explicitly allows non-zero drift
- Always record p95 or p99, not just mean latency
- Always note virtualization and MIG / vGPU limits before drawing hardware conclusions
- Never compare runs with different input distributions without labeling them
- Never mix data loading, preprocessing, model forward, and postprocessing into one undifferentiated number
- Never ship a faster path that fails golden-output verification

## Reference Index

- [01_baseline.md](references/01_baseline.md): fingerprint, measurement design, baseline schema
- [02_profiling.md](references/02_profiling.md): profiler and trace workflow
- [03_roofline.md](references/03_roofline.md): arithmetic intensity, ridge point, bottleneck classification
- [04_compute_opt.md](references/04_compute_opt.md): attention, compile, quantization, fused kernels, custom ops
- [05_memory_io.md](references/05_memory_io.md): copies, pinned memory, streams, allocator, KV cache memory
- [06_parallelism.md](references/06_parallelism.md): batching, multi-process, multi-GPU, concurrency
- [07_pipeline_cache.md](references/07_pipeline_cache.md): overlap, queueing, cache design, prefill/decode path
- [08_deployment.md](references/08_deployment.md): ONNX, TensorRT, Triton, rollout checklist

## Expected Output Template

When using this skill, future Claude should hand back:

- a concise problem statement
- workload classification and SLA framing
- exactness contract and verification method
- baseline table
- evidence-backed bottleneck diagnosis
- 3 to 5 ranked optimizations with trade-offs
- a validation plan
- a minimal safe implementation sequence
