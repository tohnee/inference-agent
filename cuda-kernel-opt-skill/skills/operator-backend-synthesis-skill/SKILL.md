---
name: "operator-backend-synthesis-skill"
description: "Generate CPU reference baseline and selectable CUDA/Triton operator scaffolds, then route into benchmark/profile optimization loops."
---

# Operator Backend Synthesis Skill

Use this skill when you have an operator logic description and want to quickly bootstrap:

1. CPU reference implementation (exactness baseline)
2. backend-selectable implementation scaffold (`cuda` or `triton`)
3. benchmark + exactness harness compatible with optimization workflow

## Why this exists

This skill packages practical patterns used in production operator engineering:

- **CPU reference first** for deterministic correctness baseline.
- **Backend decision as an explicit step** instead of hard-coding CUDA-only.
- **Benchmark + profiler artifacts** before declaring any optimization as valid.

These principles are aligned with open-source production operator practices such as Tencent `hpc-ops`:

- production-first high-performance kernels
- strong focus on attention/GEMM/MoE bottlenecks
- integration friendliness for serving stacks
- benchmark/profiling-driven iteration

## Entry Command

```bash
python cuda-kernel-opt-skill/skills/cuda-optimized-skill/operator-optimize-loop/scripts/operator_backend_synth.py \
  --name=<op_name> \
  --logic="<operator logic>" \
  --op-type=matmul|elementwise_add|layernorm \
  --backend=auto|cuda|triton \
  --m=<M> --n=<N> --k=<K> \
  --output-dir=<workspace>
```

## Generated Outputs

For each operator it generates a folder `<output-dir>/<op_name>/` with:

- `cpu_reference.py`: CPU baseline implementation (`op_cpu`)
- `kernel_cuda.cu` or `kernel_triton.py`: selected backend scaffold
- `benchmark_harness.py`: emits `metric.json` + `exactness.json`
- `manifest.json`: backend decision + file manifest

## Backend Selection Rule (auto mode)

- `matmul`: prefer CUDA for larger GEMM-like shapes (`M*N*K` large and `K` aligned), otherwise Triton
- `layernorm` / `elementwise_add`: prefer Triton for faster iteration

You can always override with `--backend=cuda` or `--backend=triton`.

## Recommended Next Step

After synthesis:

1. Fill the generated backend kernel implementation.
2. Run harness to validate exactness against CPU baseline.
3. Enter iterative optimization loop:

```bash
python cuda-kernel-opt-skill/skills/cuda-optimized-skill/operator-optimize-loop/scripts/optimize_loop.py <kernel_file> --backend=<backend> --max-iterations=<N> --ref=<cpu_reference.py>
```
