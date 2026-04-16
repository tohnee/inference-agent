# 04 Compute Optimization

## Goal

Improve arithmetic efficiency only after measurement proves compute or launch efficiency is the limiter.

Default rule: no compute optimization is valid if it changes inference results.

## Exact-Parity Rule

This file contains both parity-safe and parity-risky ideas.

In default mode:

- keep the deployed baseline precision unchanged
- reject quantization, approximate kernels, and precision lowering unless explicitly approved
- treat backend, compiler, and fused-kernel changes as safe only after golden-set parity passes

## Quick Classification

| Category | Default status in exact-parity mode |
| --- | --- |
| better scheduling with same math | allowed after verification |
| kernel fusion with same math | allowed after verification |
| `torch.compile` with same dtype | allowed after verification |
| CUDA Graphs with same math | allowed after verification |
| TF32, FP16, BF16, FP8 precision changes | blocked by default unless baseline already uses them |
| INT8, GPTQ, AWQ, INT4, approximate attention | blocked by default |

## Main Levers

- attention backend selection
- graph capture and compiler optimization
- lower precision and quantization
- fused kernels and custom operators
- engine-level lowering such as TensorRT

## Attention Path

### Preferred decision order

1. validate sequence length and batch shape distribution
2. benchmark available SDPA backends on the target GPU
3. evaluate FlashAttention or FlashInfer if attention is confirmed hot
4. verify accuracy, numerical stability, and memory headroom

Then add one more gate:

5. verify exact equality on a golden set before keeping the new backend

### Heuristics

- long sequence, attention-heavy Transformer → FlashAttention is often high value
- decode serving with paged KV cache → FlashInfer can be attractive
- short sequence encoder or ViT → backend choice must be benchmarked, not assumed

## Compiler Path

### `torch.compile`

Use when:

- the hot path is stable
- shapes are not excessively dynamic
- profiler shows many small ops or Python overhead

Check first:

- graph break count
- compilation time impact
- shape specialization behavior
- memory growth during compiled runs

Recommended mindset:

- stable shapes, long-lived service → be aggressive
- highly dynamic inputs, bursty short jobs → compilation cost may outweigh benefit

### CUDA Graphs

Use when:

- shapes, addresses, and execution path are stable
- launch overhead is visible
- service handles repetitive inference patterns

Avoid when:

- shapes are highly variable
- per-request control flow diverges
- output parity is not yet proven for the captured path

## Precision Path

### Precision order of evaluation

- baseline deployed precision first
- alternative precisions only with explicit human override from exact-parity mode

### Quick precision guide

| Precision | Good first use | Main concern |
| --- | --- | --- |
| FP32 | debugging, parity baseline | slow and memory-heavy |
| TF32 | A100-class matmul-heavy workloads | only applies to supported GPU paths |
| FP16 | mainstream GPU inference | overflow or reduced numerical headroom |
| BF16 | modern datacenter inference | sometimes slightly lower peak speed than FP16 |
| FP8 | Hopper-class optimized stacks | tooling maturity and validation cost |
| INT8 | deployment-focused compression | calibration and unsupported-op issues |
| INT4 or weight-only 4-bit | memory-constrained LLM serving | quality drift and runtime compatibility |

### Rules

- if the trusted baseline is FP32, stay on FP32 in default mode
- if the trusted baseline is BF16 or FP16, preserve that deployed precision in optimized paths
- TF32, FP16, BF16, FP8, INT8, and weight-only formats are all precision changes if they differ from the baseline
- quantization without representative data is a common failure mode
- any precision change requires an explicit exception from the human because it breaks exact-parity mode

## Quantization Routing

This section is out of default scope. Keep it only as a reference for cases where the human explicitly allows non-zero drift.

| Model class | Typical good first try |
| --- | --- |
| ViT / encoder | FP16 or BF16, then PTQ INT8 if accuracy allows |
| diffusion | FP16 or BF16 first, selective quantization only if validated |
| decoder-only LLM | BF16 or FP16 baseline, then AWQ or GPTQ, or TensorRT-LLM style paths |
| classical MLP / tabular | ONNX Runtime or TensorRT INT8 can be worthwhile if kernels are supported |

## Kernel Fusion

Target fusion when profiler shows:

- normalization + matmul + activation chains
- repeated reshape, permute, cast overhead
- preprocessing ops dominating latency
- many tiny elementwise kernels between large kernels

Typical high-value examples:

- GELU or activation chains
- normalization plus residual plus cast chains
- preprocessing resize, convert, normalize chains

Candidate implementations:

- PyTorch eager fusion opportunities
- `torch.compile`
- Triton kernels
- custom C++ or CUDA extension
- TensorRT plugins if export/runtime path requires them

For every fused implementation, require:

- exact equality against the unfused reference
- stable behavior on edge-case shapes
- deterministic handling if the workload requires deterministic outputs

## Triton Or Custom CUDA

Choose Triton first when:

- the pattern is tensor-centric
- iteration speed matters
- integration should stay Python-friendly

Choose custom CUDA or C++ op when:

- ultimate control is needed
- Triton limitations block the desired schedule
- deployment path already relies on native extensions

## Model-Family Playbook

### Transformer / LLM

- benchmark SDPA backend
- inspect prefill versus decode separately
- optimize KV cache access before chasing exotic math kernels
- consider continuous batching before custom kernels if utilization is low
- keep deterministic decode settings and request isolation if exact output replay is required

### ViT

- verify whether preprocessing dominates end-to-end latency
- test memory-efficient SDPA and fused preprocessing
- compile and batch shaping can matter more than exotic quantization

### Diffusion

- separate text encoder, U-Net, VAE, scheduler
- compile or engine-lower the dominant blocks
- test memory format and precision carefully
- validate output quality with fixed seeds and prompt sets

## Compute Decision Matrix

| Situation | Likely next move | Main risk |
| --- | --- | --- |
| FP32 inference on tensor-core GPU | first try scheduling, copy, or compile optimizations that keep FP32 | backend parity risk |
| short-sequence ViT attention | benchmark SDPA backends before FlashAttention assumptions | backend mismatch |
| long-context LLM attention | evaluate FlashAttention or FlashInfer | stack compatibility |
| many tiny kernels | test `torch.compile` or CUDA Graphs | graph break and warmup cost |
| strict memory budget | first remove fake allocations and shrink cache footprint | architecture pressure |
| repeated elementwise hotspots | fuse with Triton or custom op | maintenance cost |

## Common Mistakes

- enabling every optimization at once
- assuming FlashAttention is always optimal
- shipping quantization based only on mean latency improvement
- ignoring compile warmup and cache behavior
- optimizing model forward when preprocessing or memcpy dominates
- treating precision change as a default optimization instead of an explicit exception path

## Keep / Revert Rules

Keep a compute optimization only if it improves the intended metric and does not violate:

- exactness guardrail
- exact golden-set parity
- memory budget
- startup budget
- deployment maintainability threshold

If a change speeds up one microbenchmark but worsens p95 end-to-end latency, revert it.
