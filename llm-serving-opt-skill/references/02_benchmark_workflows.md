# 02 Benchmark Workflows

## Goal

Choose the right benchmark tool for the question you are trying to answer.

## Recommended Default

Use **SGLang `bench_serving` by default** for product-facing conclusions.

Why:

- it exercises a running server
- it measures realistic online serving metrics
- it captures TTFT, TPOT, ITL, and throughput
- it includes scheduler behavior instead of bypassing it

A useful rule is to keep `num-prompts >= 5 * max-concurrency` so the system reaches a more stable state.

## Tool Selection

| Tool | Use it for | Avoid using it for |
| --- | --- | --- |
| `bench_serving` | realistic online serving | micro-kernel conclusions |
| `bench_one_batch_server` | single-batch end-to-end latency | steady-state serving claims |
| `bench_offline_throughput` | engine throughput ceiling without HTTP overhead | product SLA conclusions |
| `bench_one_batch` | low-level static batch profiling | scheduler realism |

## Example Commands

### Realistic online serving

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --max-concurrency 16 \
  --num-prompts 80 \
  --random-input-len 256 \
  --random-output-len 32 \
  --dataset-name random
```

### One batch through the server

```bash
python3 -m sglang.bench_one_batch_server \
  --base-url http://127.0.0.1:30000 \
  --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --batch-size 32 \
  --input-len 256 \
  --output-len 32
```

### Offline throughput without HTTP

```bash
python3 -m sglang.bench_offline_throughput \
  --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --num-prompts 10
```

### Static-batch profiling path

```bash
python3 -m sglang.bench_one_batch \
  --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --batch-size 32 \
  --input-len 256 \
  --output-len 32
```

## Benchmark Design

For each benchmark campaign:

- vary concurrency intentionally
- vary prompt length intentionally
- vary output length intentionally
- keep the dataset source stable
- separate warmup from measurement
- save both launch and benchmark commands

## When to Add Kernel Microbenchmarks

After the serving benchmark points to a likely kernel bottleneck, drop down to FlashInfer-style kernel benchmarking or a framework-specific microbench path.

Kernel benchmarks are best for:

- comparing backends such as FA2, cuDNN, CUTLASS, or TensorRT-LLM
- validating a fused operator win
- comparing prefill kernels or decode kernels in isolation

Kernel benchmarks are not enough for:

- scheduler or queueing problems
- prefix cache wins
- transport or HTTP overhead
- noisy multi-tenant server behavior

## Benchmark Output Checklist

Every report should include:

- server launch command
- benchmark command
- dataset or random workload definition
- concurrency sweep
- TTFT, TPOT, ITL, throughput
- p95 or p99 latency
- memory result
- parity result

## Common Mistakes

- trusting `bench_one_batch` as a production proxy
- using tiny `num-prompts` and declaring a steady-state result
- changing both concurrency and prompt length in one unexplained jump
- not preserving benchmark commands for reproduction
