# 05 Memory And IO

## Goal

Eliminate unnecessary movement of bytes before trying to optimize math.

In many production systems, the biggest wins come from removing copies, dtype inflation, synchronization, and cache-miss work.

These optimizations are attractive because many of them can preserve exact outputs when implemented carefully.

## Exactness Rule

Memory and IO changes are acceptable only if they preserve:

- identical input values reaching the model
- identical preprocessing semantics
- identical cache semantics and invalidation behavior
- identical per-request outputs

## Primary Targets

- host-to-device copies
- device-to-host copies
- CPU preprocessing on the critical path
- layout conversion and dtype conversion
- GPU allocator churn and fragmentation
- KV cache footprint and traffic

## Hot-Path Rules

- avoid `.cpu()`, `.numpy()`, and serialization in the hot path unless strictly required
- use pinned host memory for staged GPU input
- use `non_blocking=True` when the source is compatible
- move resize, normalize, and other tensor-friendly preprocessing to GPU when it reduces wall time
- do not convert `uint8` to `float32` on CPU only to copy a 4x larger tensor
- do not change interpolation mode, normalization formula, rounding rule, or layout semantics without parity verification

## Copy Reduction Checklist

| Symptom | Likely fix |
| --- | --- |
| H2D time is material | pin memory, use smaller dtype, stage fewer tensors |
| repeated casts or permutes | standardize layout once upstream |
| D2H appears inside request path | keep postprocess on GPU longer |
| many small memcpy events | batch and coalesce transfers |

## Pinned Memory Pattern

Pinned memory is often one of the cheapest transfer optimizations on GPU inference paths.

```python
import torch

def build_loader(dataset, batch_size, workers=4):
    kwargs = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = 2
    return torch.utils.data.DataLoader(dataset, **kwargs)
```

Use it when:

- the hot path repeatedly stages CPU tensors to GPU
- H2D is visible in the timeline
- host memory headroom is sufficient

Pinned memory should not change results. If results change after this change, the issue is elsewhere and the rollout should stop.

## Zero-Copy And Near-Zero-Copy Ideas

- DLPack for framework handoff
- memory-mapped files for CPU-side reuse
- shared memory or IPC for multi-process CPU stages
- page-locked host buffers for repeat transfers
- GPU resident caches for reusable embeddings or intermediate states

Use zero-copy only when ownership, lifetime, and mutability are clear. Hidden aliasing bugs are expensive.

## GPU Preprocessing Heuristics

Move preprocessing to GPU when:

- image resize or normalize is large relative to forward time
- batches are moderate or large
- CPU is already saturated

Keep some preprocessing on CPU when:

- the operation is branch-heavy and tiny
- request rate is low and GPU would sit idle
- device transfer cost outweighs compute savings

## UInt8 Transfer Pattern

A common vision optimization is to transfer compact `uint8` inputs first and normalize on GPU instead of expanding to `float32` on CPU.

```python
import torch

def uint8_h2d(image_uint8_hwc, device):
    x = torch.from_numpy(image_uint8_hwc).permute(2, 0, 1).contiguous()
    x = x.to(device, non_blocking=True)
    x = x.float().div_(255.0)
    return x
```

This is attractive when:

- images are moderate or large
- CPU preprocessing is visible
- PCIe transfer time is non-trivial

Keep this optimization only if GPU-side resize, normalization, dtype conversion, and layout handling are proven equivalent to the reference path.

## KV Cache

For autoregressive inference, always model KV cache as a first-class budget item.

Track:

- bytes per token
- total active sequences
- max context length
- page size or block size
- fragmentation and reuse

Questions:

- is decode limited by KV cache bandwidth?
- is eviction or reuse strategy stable?
- does long context starve new short requests?
- can the cache return stale or cross-request data under concurrency

## Allocator And Fragmentation

Watch for:

- reserved memory climbing faster than active memory
- repeated OOM after varying shapes
- sudden latency spikes due to allocator activity

Mitigations:

- constrain shapes or bucket them
- reuse buffers where possible
- isolate large and small requests
- keep long-lived caches explicit instead of accidental

## Stream And Overlap Basics

Use streams to overlap:

- H2D with compute
- preprocessing with previous batch inference
- postprocess with next request staging

Do not claim overlap until the system timeline proves it.

If overlap does not appear in the system trace, suspect:

- hidden synchronization
- pageable rather than pinned host memory
- all work accidentally issued on one stream

## Example H2D Staging Pattern

```python
import torch

def stage_to_gpu(x_cpu, device):
    if not x_cpu.is_pinned():
        x_cpu = x_cpu.pin_memory()
    return x_cpu.to(device, non_blocking=True)
```

## Example Preprocess Routing

| Workload | Better default |
| --- | --- |
| image inference, large batches | GPU resize and normalize |
| small online tabular model | CPU preprocessing |
| LLM tokenization | CPU tokenizer, then pinned transfer |
| diffusion input preparation | benchmark both, often mixed path |

## Two-Level Cache Pattern

Applicable to:

- embedding galleries
- retrieval feature banks
- reusable vision encoder outputs
- decoder KV reuse in constrained serving paths

Pattern:

- L1 in-process memory cache
- L2 shared disk or object-backed cache
- explicit invalidation key from content hash or version hash

In exact-parity mode, also require:

- preprocessing version in the cache key
- model version in the cache key
- deterministic serialization and deserialization
- protection against partial or stale writes

## Common Mistakes

- optimizing kernels while memcpy dominates
- keeping cache in GPU memory when CPU memory is sufficient
- measuring H2D using unrealistic synthetic inputs
- treating pinned memory as free
- forgetting that vGPU bandwidth may be much worse than bare metal
- treating stale-cache bugs as acceptable because the numerical path is unchanged

## Exit Criteria

Leave this section only when you know:

- how many bytes move per request
- which transfers are unavoidable
- which transfers can be hidden
- which transfers can be eliminated
- why the chosen memory or cache optimization preserves exact outputs
