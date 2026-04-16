---
name: "cuda-crash-debug-skill"
description: "Use when a serving stack crashes with CUDA errors, NaN/Inf, bad tensors, or backend-specific failures."
---

# CUDA Crash Debug Skill

## Use When

- illegal memory access appears
- a serving worker exits during prefill or decode
- NaN or Inf propagates through the pipeline
- a backend switch introduces hard crashes

## Preferred Debug Order

1. reproduce with a minimal deterministic case
2. log inputs before the failing CUDA boundary
3. inspect shape, dtype, device, contiguity, and statistics
4. isolate whether the crash is prefill, decode, cache, or scheduler related
5. escalate to `compute-sanitizer` or `cuda-gdb` if needed

## FlashInfer-Inspired Logging Pattern

Useful settings:

```bash
export FLASHINFER_LOGLEVEL=3
export FLASHINFER_LOGDEST=debug.log
```

Use level 5 for NaN/Inf or value-range issues.

## Questions To Answer

- does it fail in prefill or decode?
- does it require concurrency to reproduce?
- does it require prefix cache or speculative decode?
- does it disappear on another backend?
- does one prompt length or head dimension trigger it?

## Deliverables

- minimal reproducer
- last good input signature
- first bad input signature
- likely failure class
- next debugging step

## Escalation Path

If API-boundary logging is not enough:

1. disable optional fast paths such as CUDA graph or speculative decode
2. switch backend to isolate the failing lane
3. run `compute-sanitizer` on the smallest reproducer
4. capture per-rank logs for multi-process serving
