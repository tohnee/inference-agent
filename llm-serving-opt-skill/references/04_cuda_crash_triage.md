# 04 CUDA Crash Triage

## Goal

Make CUDA failures inspectable before the process dies.

## Core Pattern

Borrow the most useful idea from FlashInfer's `debug-cuda-crash` workflow:

- log API inputs before the CUDA call happens
- record tensor shape, dtype, device, and contiguity
- add statistics such as min/max/mean/nan/inf when debugging numerical issues
- log to files when a process may crash hard

## FlashInfer Logging Controls

Useful environment variables:

| Variable | Value | Meaning |
| --- | --- | --- |
| `FLASHINFER_LOGLEVEL` | `0` | disable logging |
| `FLASHINFER_LOGLEVEL` | `1` | function names only |
| `FLASHINFER_LOGLEVEL` | `3` | input/output metadata |
| `FLASHINFER_LOGLEVEL` | `5` | metadata plus tensor statistics |
| `FLASHINFER_LOGDEST` | `stdout` / `stderr` / path | choose the output target |
| `FLASHINFER_LOGDEST` | `log_%i.txt` | split logs per process |

Typical workflow:

```bash
export FLASHINFER_LOGLEVEL=3
export FLASHINFER_LOGDEST=debug.log
python my_script.py
```

Use level 5 when chasing NaN/Inf issues.

## What To Inspect First

For illegal memory access or shape mismatch:

- tensor shape
- head dimension alignment
- dtype consistency
- device placement
- contiguity and stride assumptions

For NaN/Inf:

- `nan_count`
- `inf_count`
- suspicious min/max ranges
- unstable upstream normalization or sampling path

For OOM:

- sequence length explosion
- unexpected batch expansion
- KV-cache allocation spikes

## Multi-Process Triage

Use per-rank logs when debugging serving systems with multiple workers:

```bash
export FLASHINFER_LOGLEVEL=3
export FLASHINFER_LOGDEST=debug_rank_%i.txt
torchrun --nproc_per_node=4 my_script.py
```

## Complementary Tools

Combine logging with:

- `compute-sanitizer --tool memcheck`
- `cuda-gdb`
- targeted `printf()` inside kernels
- a minimal deterministic reproducer

## Serving-Specific Crash Questions

Always ask:

- does the crash happen in prefill or decode?
- does it reproduce only with prefix cache or only without it?
- does concurrency matter?
- does the failure disappear when switching backend or disabling CUDA graph?
- is there a single prompt shape that always breaks?

## Common Mistakes

- trying to debug a hard crash without pre-call logging
- logging only stack traces and not input tensors
- mixing rank logs together in multi-process setups
- assuming every crash is a kernel bug when shape/dtype bugs are more common
