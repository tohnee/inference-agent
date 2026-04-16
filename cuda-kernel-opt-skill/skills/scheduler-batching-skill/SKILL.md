---
name: "scheduler-batching-skill"
description: "Use when optimizing continuous batching, fairness, queueing delay, prefill/decode interference, or scheduler policy."
---

# Scheduler Batching Skill

## Use When

- throughput collapses as concurrency rises
- short requests are starved by long ones
- TTFT or TPOT tails are unstable
- prefill interferes with decode

## Core Measurements

- queueing delay
- active batch size
- short-vs-long request fairness
- prefill-to-decode handoff delay
- cancellation and timeout behavior

## High-Value Lanes

- continuous batching tuning
- chunked prefill tuning
- disaggregated prefill/decode evaluation
- dual-batch overlap or microbatch thresholds

## Guardrails

- do not optimize average throughput by silently ruining tail latency
- do not assume scheduler issues if memory fragmentation is the real limit
- preserve request isolation and deterministic behavior where promised

## Deliverables

- scheduler symptom summary
- fairness and queueing observations
- one dominant scheduler bottleneck
- next safe tuning steps

## Tuning Order

1. confirm the bottleneck is scheduling rather than memory
2. compare low and moderate concurrency behavior
3. inspect short-request starvation
4. only then adjust continuous batching or chunked prefill thresholds
