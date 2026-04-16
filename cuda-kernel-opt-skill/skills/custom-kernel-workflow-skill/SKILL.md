---
name: "custom-kernel-workflow-skill"
description: "Use when considering a fused op, Triton kernel, or custom CUDA kernel for LLM inference after higher-level lanes have been exhausted."
---

# Custom Kernel Workflow Skill

## Use When

- a hotspot remains after serving-level fixes
- existing backends do not cover the needed pattern
- a fused op or custom kernel might materially improve TTFT or TPOT

## Do Not Use Yet If

- scheduler or queueing still dominates
- no realistic serving benchmark exists
- no profile points to a repeated kernel family
- correctness gates are still weak

## Preferred Order

1. enable existing fast path
2. compare existing backends
3. try framework-native fusion or compile
4. only then consider Triton or custom CUDA

## Upstream Influence

Follow the discipline seen in FlashInfer:

- benchmark before claiming a win
- keep a clean add-kernel workflow
- keep crash-debug hooks ready

## Deliverables

- evidence that custom kernel work is justified
- minimal target operator or pattern
- parity risks
- benchmark plan

## Before Writing A Custom Kernel

- prove a repeated hotspot exists
- show that an existing backend cannot cover it
- define the exact operator pattern
- define the parity gate and benchmark recipe first
