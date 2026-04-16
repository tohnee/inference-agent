---
name: "serving-benchmark-skill"
description: "Use when designing or reviewing serving benchmarks across SGLang, vLLM, TensorRT-LLM, Triton, or PyTorch paths."
---

# Serving Benchmark Skill

## Use When

- comparing serving frameworks
- reviewing benchmark methodology
- designing reproducible TTFT/TPOT/ITL experiments
- preventing invalid comparisons across SGLang, vLLM, TensorRT-LLM, Triton, and PyTorch services

## First Principle

Benchmark the serving system you actually intend to ship.

Do not compare:

- a live server against an offline engine loop
- cold start against warm steady-state
- different prompt distributions without labeling them
- different cache or scheduler settings as if they were identical

## Framework Notes

- **SGLang**: prefer `bench_serving` for realistic online serving
- **vLLM**: use its serving benchmark path and keep random vs ShareGPT-like datasets explicit
- **TensorRT-LLM**: preserve build, serve, and benchmark steps separately
- **Triton Inference Server**: record model config, backend, instance groups, and client tool
- **PyTorch custom server**: ensure the client path and request shape match production behavior

## Required Metrics

- TTFT
- TPOT
- ITL
- req/s
- input tok/s
- output tok/s
- p95 or p99 latency
- peak memory
- parity status

## Required Report Shape

1. workload definition
2. server launch command
3. benchmark command
4. hardware and software fingerprint
5. metric table
6. exactness result
7. caveats and next lane

## Common Failures

- infinite-rate load on one framework vs bounded-rate load on another
- no concurrency sweep
- no saved commands
- no parity check
- over-reading synthetic data results as product truth

## Evidence Requirements

A usable benchmark conclusion needs:

- identical workload shape across candidates
- explicit cache and scheduler settings
- cold-vs-warm labeling
- exactness or bounded-tolerance result

## Cross-Framework Cautions

- vLLM and SGLang may expose different scheduler behaviors under the same concurrency target
- TensorRT-LLM comparisons are only fair when engine build assumptions are recorded
- Triton comparisons are invalid if model config and instance-group settings are omitted
