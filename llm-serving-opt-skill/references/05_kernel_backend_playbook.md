# 05 Kernel and Backend Playbook

## Goal

Choose the right optimization lane after evidence says the serving bottleneck is kernel or backend related.

## Safe Order

Prefer this order:

1. verify the fast path is actually enabled
2. compare existing backends under a fixed workload
3. use a proven upstream kernel library when possible
4. introduce new fused or custom kernels only when evidence remains strong

## Backend Choices

### FlashInfer

Prefer when you need:

- optimized decode, prefill, and append kernels
- paged or ragged KV cache support
- shared-prefix and cascade patterns
- fused sampling operators
- strong compatibility with low-latency serving paths

### Other lanes

- **FlashAttention / cuDNN**: when framework integration is already stable and evidence is close enough
- **CUTLASS / Triton**: when a missing fused kernel or custom variant is truly the bottleneck
- **TensorRT-LLM**: when engine build complexity is justified by repeated production gain
- **PyTorch eager / compile**: when the current bottleneck is still above the framework line

## Evidence Checklist Before a Backend Switch

- exact same model and prompt distribution
- same concurrency target
- same KV-cache policy
- same output-parity gate
- benchmark result from serving path, not only from microbench
- profiler evidence that points to the backend-sensitive kernel family

## Directly Reusable Upstream Skills

When the work becomes kernel-implementation work, route to the upstream FlashInfer playbooks directly:

- `benchmark-kernel` for backend comparison and reproducible microbench
- `add-cuda-kernel` for a clean implementation checklist
- `debug-cuda-crash` for post-change triage

## Risk Table

| Change | Typical upside | Typical risk |
| --- | --- | --- |
| enable existing optimized backend | medium | low |
| switch attention backend | medium to high | medium |
| enable `torch.compile` or CUDA graph | medium | medium |
| add fused kernel | high | high |
| custom CUDA kernel | high | very high |
| quantization or low precision change | high | exactness risk |

## When To Stop Going Lower

Do not write custom kernels yet if:

- `bench_serving` still shows scheduler or queueing dominance
- CPU preprocessing or transport overhead dominates
- TTFT is dominated by load or initialization issues
- parity is not yet stabilized on the current path
