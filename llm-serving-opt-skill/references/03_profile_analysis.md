# 03 Profile Analysis

## Goal

Turn traces into actionable serving decisions instead of screenshot collections.

## Collection Rules

For SGLang torch-profiler collection:

- set `SGLANG_TORCH_PROFILER_DIR` on both server and client
- keep benchmark commands alongside the trace
- use smaller profiling windows for prefill when traces become too large
- separate prefill and decode profiling in PD disaggregation mode

Example environment setup:

```bash
export SGLANG_TORCH_PROFILER_DIR=/root/sglang/profile_log
```

Example server and benchmark flow:

```bash
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct
python -m sglang.bench_serving --backend sglang --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 10 --sharegpt-output-len 100 --profile
```

## Triage Tables

Inspired by SGLang profile-analysis practice, produce three tables from every profile.

### 1. Kernel table

| Stage | Kernel family | GPU time | Share | Launches | Python site | CPU op |
| --- | --- | --- | --- | --- | --- | --- |

Purpose:

- find the dominant kernel families
- verify whether the fast path is active
- detect over-fragmentation or too many short launches

### 2. Overlap opportunity table

| Stage | Candidate | Hidden time | Risk | Why it matters |
| --- | --- | --- | --- | --- |

Purpose:

- identify CPU/GPU overlap gaps
- identify communication/compute overlap gaps
- spot prefill/decode or copy/compute serialization

### 3. Fuse-pattern table

| Pattern | Evidence | Likely site | Confidence | Next action |
| --- | --- | --- | --- | --- |

Purpose:

- look for known fuse opportunities
- map them back to real Python or framework sites
- avoid fuzzy hand-waving

## Interpretation Order

Read traces in this order:

1. determine whether the slowdown is prefill or decode dominated
2. identify top kernel families by time share
3. inspect launch count and fragmentation
4. inspect copy, sync, and idle gaps
5. propose only one dominant bottleneck statement

## What To Look For

### Prefill-heavy symptoms

- large GEMM or attention kernels dominate
- CPU preprocessing delays server saturation
- prompt chunking creates serialization

### Decode-heavy symptoms

- too many tiny kernels
- scheduler overhead or launch overhead dominates
- cache access and memory bandwidth dominate
- TPOT rises faster than concurrency

### Copy or host-bound symptoms

- long H2D or D2H gaps
- tokenizer or postprocess hot on CPU
- HTTP or request marshaling visible in critical path

## Common Mistakes

- reading total trace time without stage split
- skipping kernel category grouping
- proposing fusion before confirming a repeated pattern exists
- using profile output without the exact benchmark command that produced it
