# 01 Serving Baseline

## Goal

Build a trustworthy exactness-first baseline for LLM serving before changing kernels, caches, schedulers, or deployment topology.

## First Questions

Answer these before measuring anything:

- Is this online serving, batch generation, or offline throughput?
- Is the user SLA dominated by TTFT, TPOT, ITL, or p95 end-to-end latency?
- Is the workload deterministic enough for exact token-by-token comparison?
- Are we evaluating cold start, warm steady state, or burst traffic?

## Exactness Contract

For LLM serving, exactness should usually include:

- identical generated tokens for deterministic decode
- identical prompt preprocessing and truncation rules
- identical request isolation when multiple sessions share a scheduler
- identical cache semantics for cache hit and cache miss paths
- explicit tolerance policy only when the human allows cross-device or precision drift

Minimum parity record:

```text
- decode mode: greedy / temperature 0 / fixed seed
- prompt set: golden prompts with stable lengths
- expected outputs: token ids or exact text
- cache rules: prefix cache on/off, reuse semantics, eviction behavior
- isolation rules: concurrent requests must not perturb outputs
```

## Baseline Metrics

Always capture:

- TTFT
- TPOT
- ITL
- tokens/s
- requests/s
- p50, p95, p99 latency
- peak GPU memory
- KV cache allocated bytes and effective utilization
- failure rate and timeout rate

If using a scheduler, also capture:

- queueing delay
- active concurrency
- prompt length distribution
- output length distribution
- cancellation or timeout behavior

## Workload Matrix

Build at least one matrix that reflects product reality:

| Dimension | Example values |
| --- | --- |
| max concurrency | 1, 4, 8, 16, 32 |
| prompt length | 128, 512, 2k, 8k |
| output length | 16, 128, 512 |
| traffic mix | short-chat, long-chat, summarization |
| cache state | cold, warm, shared-prefix |

Do not compare runs with different distributions without labeling them clearly.

## Fingerprint

Record the environment used for every baseline:

- model name and revision
- serving framework and version
- GPU model, count, and partition mode if MIG or vGPU is used
- CUDA version, driver version, PyTorch version
- attention backend and sampling backend
- launch command and benchmark command
- batch, chunked-prefill, and memory-related flags

## Baseline Table Template

| Scenario | Concurrency | Input/Output | TTFT ms | TPOT ms | ITL ms | tok/s | req/s | peak mem GB | parity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| short-chat | 8 | 256/64 |  |  |  |  |  |  | pass/fail |
| long-chat | 8 | 4096/128 |  |  |  |  |  |  | pass/fail |
| burst | 32 | mixed |  |  |  |  |  |  | pass/fail |

## Practical Baseline Sequence

1. run one deterministic smoke case
2. capture exact outputs on a golden prompt set
3. run warm steady-state benchmark
4. run at least one concurrency sweep
5. record memory growth and KV-cache behavior
6. freeze this baseline before any optimization

## Common Mistakes

- using mean latency only
- using a single prompt or single concurrency point
- mixing server launch changes with benchmark changes
- skipping parity checks because throughput improved
- comparing a warm cache run to a cold cache run without saying so
