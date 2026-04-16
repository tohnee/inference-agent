# 06 Parallelism

## Goal

Choose the right batching and parallelism strategy for the actual service pattern instead of copying a training-era design into inference.

Parallelism is only valid if it preserves exact per-request semantics and does not introduce result drift, stale state, or cross-request contamination.

## Exactness Rule

Before adopting a batching or parallelism strategy, verify:

- each request receives the same output as in the single-request reference path
- request ordering does not corrupt cache or response mapping
- deterministic workloads remain deterministic
- shared state does not leak across requests, workers, or GPUs

## Decision Order

1. fix single-request inefficiencies first
2. characterize request mix
3. choose batching policy
4. choose process or thread topology
5. add multi-GPU strategy only when one device is proven insufficient

## Batching Modes

| Mode | Best for | Watch out for |
| --- | --- | --- |
| static batching | offline or stable-size vision jobs | padding waste, latency inflation |
| microbatching | online systems with small burst windows | queueing delay |
| dynamic batching | mixed online traffic | tail latency if window too wide |
| continuous batching | autoregressive LLM serving | scheduler complexity, fairness |

In exact-parity mode, prefer batching strategies that do not alter padding semantics, masking semantics, or decode state behavior.

## Continuous Batching

Strong candidate when:

- model is decoder-heavy
- requests have variable generation lengths
- throughput matters without giving up latency too aggressively

Key controls:

- max running sequences
- batching window
- prefill/decode scheduling
- cache eviction and backpressure

Additional parity checks:

- deterministic decode settings per request
- isolated RNG streams if sampling exists
- correct mapping from request to output sequence

Do not import continuous batching into vision or simple encoder services unless the request pattern actually benefits.

## Process Model

### Threads

Prefer threads when:

- most time is in native kernels
- shared model memory matters
- Python overhead is not dominant

### Processes

Prefer processes when:

- CPU preprocessing is heavy
- GIL contention is visible
- isolation matters more than shared memory efficiency

Cost of processes:

- duplicated host memory unless carefully shared
- more IPC
- more complex cache ownership
- more ways to serve stale or cross-process inconsistent state

## Multi-GPU Routing

| Strategy | Good for | Risk |
| --- | --- | --- |
| request-level sharding | independent requests | load imbalance |
| data parallel replicas | standard online inference scale-out | memory duplication |
| tensor parallel | single model too large or too slow on one GPU | communication overhead |
| pipeline parallel | very large sequential models | bubble overhead |

Inference usually prefers simpler routing first:

- replicate whole model across GPUs
- route requests intelligently
- add TP or PP only when one GPU cannot satisfy memory or latency needs

In exact-parity mode, introduce communication-heavy strategies only after proving the simpler topology preserves outputs and operational semantics.

## CPU Serving

For CPU-bound inference:

- use NUMA-aware pinning when relevant
- favor operator libraries already optimized by the runtime
- benchmark threads and processes separately
- keep batch sizes moderate to avoid cache thrash

On CPU, parallelism often fails because memory bandwidth saturates before compute does.

## vGPU, MIG, And Virtualization

Treat virtualized GPU resources as first-class constraints:

- available memory may be the main limiter
- effective bandwidth may differ sharply from bare metal
- co-tenancy can destabilize p95

Always measure on the actual quota shape, not on the nearest physical SKU.

## Queueing Questions

Before changing parallelism, answer:

1. what is the request arrival pattern?
2. what is the maximum tolerated queue wait?
3. which stage can batch without violating SLA?
4. does batching improve compute reuse enough to offset added wait?
5. does batching preserve exact outputs for each request

## Signs You Need Better Scheduling Instead Of Better Kernels

- batch size 1 dominates traffic
- GPU idle periods exist between requests
- throughput rises sharply with only modest batching
- end-to-end latency is mostly queue and preprocess, not model forward

## Common Mistakes

- enabling TP before proving a single GPU path is healthy
- maximizing throughput while ignoring p95
- using too many workers and overwhelming CPU caches or PCIe
- treating DataLoader tuning as an inference strategy
- ignoring fairness when mixing short and long LLM requests
- accepting scheduler bugs because aggregate throughput improved

## Exit Criteria

You should be able to state:

- which batching mode is selected and why
- which process model is selected and why
- whether one GPU, many replicas, or model-parallel inference is appropriate
- how exact per-request parity is preserved under the chosen topology
