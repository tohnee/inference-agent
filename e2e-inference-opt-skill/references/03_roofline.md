# 03 Roofline

## Goal

Use roofline analysis to decide whether the next improvement should target compute efficiency or memory traffic.

This prevents random optimization work.

Roofline is a prioritization tool, not permission to break exactness.

## Exactness Boundary

In default mode, roofline may guide only among optimizations that preserve exact outputs.

That means:

- copy reduction is allowed if it preserves the same values
- scheduling and overlap changes are allowed if they preserve request semantics
- compiler or backend changes require parity proof
- lower precision, quantization, or approximate math are outside default scope unless explicitly approved

## Core Model

Runtime is bounded by the lower of:

- peak compute capability
- peak memory bandwidth

Arithmetic intensity:

```text
AI = total FLOPs / total bytes moved from main memory
```

Ridge point:

```text
ridge_point = peak_compute / peak_bandwidth
```

Interpretation:

- `AI < ridge_point` → memory-bandwidth-limited
- `AI > ridge_point` → compute-limited

## Practical Use

Apply roofline at two levels:

1. **whole model or stage level**
2. **dominant kernel level**

Stage level answers which family of optimization to try.
Kernel level answers how to tune the hot op.

## Hardware Template

Do not hardcode one table for every environment. A40, L20, A100, H100, MIG, and vGPU slices can differ materially by clock, interconnect, and exposed bandwidth.

Build a local table like this:

| Device | Precision mode | Peak compute | Peak bandwidth | Ridge point |
| --- | --- | --- | --- | --- |
| A40 | FP16 tensor core | vendor spec | vendor spec | compute / bandwidth |
| L20 | FP16 or BF16 | vendor spec | vendor spec | compute / bandwidth |
| A100 | FP16 or BF16 | vendor spec | vendor spec | compute / bandwidth |
| H100 | BF16, FP16, FP8 | vendor spec | vendor spec | compute / bandwidth |
| vGPU | effective spec | measured or quota-aware | measured or quota-aware | compute / bandwidth |

For virtualized environments, prefer effective measured bandwidth over marketing peak values.

## Indicative Ridge-Point Thinking

Use indicative ridge points only as a quick classification aid:

- A40 class hardware often rewards bandwidth-conscious optimization earlier than raw compute tuning
- L20 and A100 class hardware sit in a middle zone where both bandwidth and compute tuning can matter depending on sequence length and batch size
- H100 class hardware raises the ridge point substantially, so many kernels that looked compute-heavy on older cards are still memory-sensitive there

The same kernel can change category across GPUs. Never blindly transfer the optimization recipe from one SKU to another.

## Estimation Workflow

### Step 1: estimate FLOPs

- matmul or conv heavy models: derive analytically when feasible
- Transformer blocks: separate attention, MLP, embedding, KV update
- diffusion: separate U-Net, VAE, text encoder, scheduler overhead

### Step 2: estimate bytes moved

Track:

- input read
- weight read
- activation read/write
- KV cache read/write
- host-device traffic if it is on the critical path

### Step 3: compare with achieved metrics

From profiler or kernel tools:

- achieved FLOPs
- achieved bandwidth
- SM busy
- DRAM throughput

## Quick Heuristics

| Observation | Likely class |
| --- | --- |
| high DRAM throughput, modest SM utilization | memory-bound |
| low DRAM throughput, high tensor core activity | compute-bound |
| low DRAM and low SM utilization | scheduling, launch, or dependency issue |
| large KV cache traffic in decode | memory-bound |
| ViT or short-seq encoder with tiny kernels | launch-bound or memory-bound |
| long-context attention with large sequence | IO-aware attention likely matters |

## Decision Tree

```text
estimate or measure arithmetic intensity
        |
        v
is AI below the ridge point
        |
   yes / no
    /     \
memory     compute
bound      bound
 |          |
reduce      improve tensor-core use
traffic     keep baseline precision
fuse IO     compile or fuse kernels
improve     increase effective batch
reuse       reduce algorithmic FLOPs
```

## Model Family Hints

### Transformer encoder / ViT

- short sequence cases often suffer from launch and overhead before raw FLOPs become the issue
- attention backend selection and batch shaping matter
- preprocessing can dominate end-to-end latency for vision workloads

### Autoregressive Transformer

- prefill tends to be compute-heavier
- decode tends to be KV-cache and memory-traffic-sensitive
- continuous batching shifts the operating point and can improve weight reuse

### Diffusion

- U-Net usually dominates
- scheduler and VAE can matter for low-step or low-resolution settings
- memory layout and compile effectiveness are often material

## Optimization Mapping

If memory-bound, prefer:

- fewer copies
- better layout
- fused read/write paths
- paged KV cache improvements
- FlashAttention or FlashInfer when the bottleneck is attention IO
- lower precision only if the human explicitly permits leaving exact-parity mode

If compute-bound, prefer:

- tensor-core-friendly dtype and layout
- larger effective batch or continuous batching
- `torch.compile`, CUDA Graphs, kernel fusion
- TensorRT or custom fused kernels

In exact-parity mode, keep the baseline dtype and numerical contract unless the baseline itself already runs in that dtype.

If neither is clearly saturated, prefer:

- pipeline overlap
- batch shaping
- launch reduction
- lock and queue cleanup

## Common Operator Intuition

These are directional heuristics, not guaranteed classifications:

- LayerNorm and GELU are commonly memory-sensitive
- naive attention is commonly memory-sensitive
- IO-aware attention raises effective data reuse and can move the kernel closer to compute saturation
- large GEMMs and large convolutions are often compute-friendlier than surrounding elementwise chains

## Simple Arithmetic Intensity Helper

```python
def ridge_point(peak_tflops, peak_gbps):
    return (peak_tflops * 1e12) / (peak_gbps * 1e9)

def arithmetic_intensity(flops, bytes_moved):
    return flops / bytes_moved
```

## Decision Questions

Before choosing an optimization, answer:

1. Is the current stage below or above the ridge point?
2. Is the measured bottleneck consistent with profiler evidence?
3. Is the bottleneck at model level or kernel level?
4. Would a better schedule hide the bottleneck better than a faster kernel?

## Common Mistakes

- using vendor peak specs as if achieved metrics must approach them
- forgetting PCIe limits in end-to-end inference
- ignoring vGPU or MIG bandwidth constraints
- treating the whole model as compute-bound because one kernel is compute-heavy
- skipping stage-level analysis in multi-stage pipelines
- using roofline to justify math-changing optimizations before exactness review

## Exit Criteria

Proceed to optimization only when you can classify the dominant path as one of:

- compute-limited
- bandwidth-limited
- launch-limited
- schedule-limited
