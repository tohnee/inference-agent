---
name: "backend-selection-skill"
description: "Use when choosing or comparing FlashInfer, FlashAttention, cuDNN, Triton, TensorRT-LLM, Triton Inference Server, vLLM, SGLang, or PyTorch serving lanes."
---

# Backend Selection Skill

## Use When

- selecting a serving framework or kernel backend
- evaluating migration cost vs expected gain
- deciding whether to stay in PyTorch, move to SGLang or vLLM, or adopt TensorRT-LLM or Triton-serving paths

## Selection Order

1. verify the current path is configured correctly
2. benchmark the current path realistically
3. profile the dominant hotspot
4. compare candidate backends under the same workload
5. choose the smallest migration that fixes the dominant bottleneck

## Typical Lanes

- **PyTorch**: easiest iteration, weakest for large-scale serving without extra work
- **SGLang**: strong serving workflow, benchmark and profile tooling, RadixAttention-style serving patterns
- **vLLM**: strong prefix caching, scheduler, and online serving ergonomics
- **TensorRT-LLM**: strongest when engine build cost is justified and throughput or latency at scale matters
- **Triton Inference Server**: service packaging and production integration strength
- **Triton kernels / custom CUDA**: only when a real kernel-level gap remains

## Required Comparison Axes

- parity
- TTFT
- TPOT
- ITL
- throughput
- memory
- deployment complexity
- debuggability

## Deliverables

- backend comparison table
- migration recommendation
- risk and rollback notes

## Selection Heuristic

- stay in PyTorch when the problem is still above the kernel line
- prefer SGLang or vLLM when scheduler, prefix cache, and serving ergonomics dominate
- prefer TensorRT-LLM when deployment accepts build complexity for repeated high-volume gain
- prefer Triton Inference Server when service packaging and fleet integration matter most
