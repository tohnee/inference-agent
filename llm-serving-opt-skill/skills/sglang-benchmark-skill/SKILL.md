---
name: "sglang-benchmark-skill"
description: "Use when benchmarking SGLang serving with the right tool, command shape, concurrency sweep, and profiling hooks."
---

# SGLang Benchmark Skill

## Use When

- benchmarking a running SGLang server
- choosing between `bench_serving`, `bench_one_batch_server`, `bench_offline_throughput`, and `bench_one_batch`
- collecting TTFT, TPOT, ITL, throughput, and concurrency evidence
- preparing a profile-friendly benchmark run

## Default Rule

Use `bench_serving` by default for product-facing serving conclusions.

## Tool Routing

| Question | Tool |
| --- | --- |
| realistic online serving | `bench_serving` |
| one batch through HTTP server | `bench_one_batch_server` |
| in-process throughput ceiling | `bench_offline_throughput` |
| static-batch low-level profiling | `bench_one_batch` |

## Standard Workflow

1. save the exact `launch_server` command
2. define prompt and output length distribution
3. run a deterministic smoke prompt
4. run `bench_serving` with a concurrency sweep
5. capture TTFT, TPOT, ITL, req/s, tok/s, and p95/p99
6. rerun with `--profile` only for a small profiling window
7. record parity result alongside performance result

## Good Defaults

- keep `num-prompts >= 5 * max-concurrency`
- benchmark warm steady-state separately from cold start
- vary concurrency and prompt length independently
- keep launch flags and benchmark flags in the same report

## Example

```bash
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct
python3 -m sglang.bench_serving \
  --backend sglang \
  --max-concurrency 16 \
  --num-prompts 80 \
  --random-input-len 256 \
  --random-output-len 32 \
  --dataset-name random
```

## Profiling Hook

When collecting a torch profile:

- set `SGLANG_TORCH_PROFILER_DIR` on both server and client
- keep the benchmark short
- separate prefill and decode when PD disaggregation is used

## Deliverables

- launch command
- benchmark command
- metrics table
- concurrency sweep summary
- parity result
- next bottleneck hypothesis

## Inputs To Ask For

- server launch command
- benchmark goal: TTFT, TPOT, ITL, throughput, or tail latency
- prompt and output length distribution
- target concurrency
- whether profiling is required

## Common Mistakes

- profiling a long unstable run instead of a short controlled run
- declaring one-batch latency as serving truth
- not recording the exact launch flags
