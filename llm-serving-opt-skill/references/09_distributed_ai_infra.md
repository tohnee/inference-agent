# 09 Distributed AI-Infra Deep Optimization

This reference focuses on distributed-system and AI-infra practices for production LLM serving.

## Scope

- multi-node prefill/decode separation
- model-parallel + pipeline-parallel deployment trade-offs
- interconnect-aware request routing (NVLink / PCIe / RDMA)
- queueing and admission control for tail-latency stability

## Backend-specific notes

- SGLang:
  - validate scheduler fairness and cancellation behavior under burst traffic
  - profile TTFT/TPOT with realistic context-length mix
- vLLM:
  - tune paged KV behavior and chunked prefill for long-context load
  - monitor allocator pressure and fragmentation
- TensorRT-LLM:
  - keep engine build matrix aligned with workload shape distribution
  - verify engine fallback path and deployment rollout policy

## Profiling checklist

- per-stage latency: ingress, prefill, decode, egress
- GPU utilization + memory headroom per rank
- inter-node communication time and queue wait time
- p95/p99 at target concurrency

## Optimization loop

1. establish baseline under production-like request mix
2. identify dominant distributed bottleneck
3. run one bounded infra change (routing, scheduler, parallelism policy)
4. re-validate exactness + TTFT/TPOT/ITL + throughput
5. keep or revert by evidence
