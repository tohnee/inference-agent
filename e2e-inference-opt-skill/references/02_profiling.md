# 02 Profiling

## Goal

Profiling is not about finding many slow places. It is about identifying the one constraint that dominates the current operating point.

Profiling never overrules exactness. A hotspot is actionable only if the candidate fix preserves the output contract.

## Correctness-First Rule

Before acting on profiler evidence:

- freeze the trusted reference path
- keep a golden input and golden output set ready
- treat any output mismatch after an optimization as a hard failure
- prefer scheduling, overlap, and copy reductions before math-changing changes

## Tool Selection

| Tool | Best for | What it reveals | Cost |
| --- | --- | --- | --- |
| `torch.profiler` | PyTorch op timeline | operator hotspots, CPU/GPU overlap, memory | medium |
| Nsight Systems `nsys` | system timeline | kernels, memcpy, stream overlap, launch gaps | medium |
| Nsight Compute `ncu` | kernel deep dive | occupancy, memory throughput, tensor core usage | high |
| `py-spy` | Python runtime | GIL, blocking, CPU hotspots, worker imbalance | low |
| memory profiler / tracemalloc | CPU memory growth | leaks, allocator churn | low |
| `nvidia-smi dmon` | coarse live telemetry | util, mem, power, encoder/decoder | low |

## Profiling Stack

Think in layers:

| Layer | Primary question | Common tool |
| --- | --- | --- |
| process or Python layer | where does host time go | `py-spy`, tracing, logs |
| framework op layer | which PyTorch ops dominate | `torch.profiler` |
| system timeline layer | are copy and compute overlapping | `nsys` |
| kernel layer | why is this kernel inefficient | `ncu` |
| memory layer | where does peak memory come from | PyTorch memory stats, snapshots |

## Profiling Order

1. coarse telemetry
2. framework timeline
3. system timeline
4. kernel deep dive

Jumping directly to kernel analysis is usually wasteful if the real bottleneck is Python scheduling, IO, or copies.

## Triage Questions

Ask these in order:

1. Is the time inside model forward, outside model forward, or both?
2. Is the bottleneck CPU-side, GPU-side, or transfer-side?
3. Is GPU busy but slow, or GPU mostly idle?
4. Is the system latency dominated by launch overhead, memory movement, or arithmetic?
5. Is there hidden serialization from synchronization, locks, or the GIL?
6. Which candidate fixes preserve exact outputs, and which would change math or semantics?

## Torch Profiler Pattern

```python
import torch
from torch.profiler import profile, ProfilerActivity

def profile_step(fn, wait=1, warmup=1, active=3):
    schedule = torch.profiler.schedule(wait=wait, warmup=warmup, active=active)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(wait + warmup + active):
            fn()
            prof.step()
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
```

Use it to answer:

- which ops dominate self CUDA time
- whether CPU preprocessing or Python code outweighs GPU work
- whether shape polymorphism causes unstable execution

For deeper analysis, also export a trace and compare total CPU self time against total CUDA self time. A large CPU-to-GPU skew often means the next optimization belongs outside the model kernel path.

## NVTX And System Timeline Pattern

Use explicit stage markers when the service has multiple phases.

```python
import torch.cuda.nvtx as nvtx

def annotated_step(preprocess, h2d, forward, postprocess):
    nvtx.range_push("preprocess")
    x = preprocess()
    nvtx.range_pop()

    nvtx.range_push("h2d")
    x = h2d(x)
    nvtx.range_pop()

    nvtx.range_push("forward")
    y = forward(x)
    nvtx.range_pop()

    nvtx.range_push("postprocess")
    out = postprocess(y)
    nvtx.range_pop()
    return out
```

System timeline checklist:

- does H2D overlap with compute
- are there long gaps between kernels
- are there unexpected synchronizations
- is copy work on the critical path
- do overlap changes preserve request isolation and cache correctness

## Nsight Systems Pattern

Use when you suspect:

- H2D or D2H copies are too large or too frequent
- kernels are separated by gaps
- streams are not overlapping
- worker threads are serialized

Typical questions for the timeline:

- Are memcpy calls on the critical path?
- Are kernels short and numerous, suggesting launch overhead?
- Are CPU stages waiting for GPU completion too often?
- Does preprocessing overlap with model forward?

## Nsight Compute Pattern

Use only after a kernel is confirmed hot.

Key metrics to inspect:

- achieved occupancy
- tensor core utilization
- SM busy
- memory throughput relative to peak
- L2 hit rate
- register pressure
- warp stalls by reason

Interpretation rule:

- high bandwidth, low arithmetic progress → memory-bound
- high stall on dependency or low occupancy → scheduling or register issue
- low tensor core use for matmul-heavy path → dtype/layout/backend mismatch

In exact-parity mode, a dtype or backend mismatch is actionable only if the fix does not change the numerical contract.

## Python-Side Profiling

Use `py-spy` or similar when:

- GPU utilization is low
- CPU is pegged
- increasing DataLoader workers changes nothing
- request handling stack includes JSON, PIL, OpenCV, tokenization, or custom business logic

Watch for:

- PIL decode and resize
- repeated tokenizer setup
- repeated model or processor lookup
- per-request logging or serialization
- lock contention in cache or batching queues

## Memory Profiling

Separate:

- GPU active memory
- GPU reserved memory
- CPU resident set size
- pinned host memory
- KV cache growth

If peak reserved memory grows while active memory does not, suspect fragmentation, allocator churn, or varying shapes.

If the path is close to OOM, capture a memory snapshot or allocator history and inspect whether the problem is:

- true model size
- activation growth
- fragmentation from variable shapes
- cache retention that never shrinks

Do not solve memory pressure by changing precision in default mode. First eliminate fake allocations, reuse buffers, and control shapes.

## Bottleneck Signatures

| Signature | Likely cause | Next file |
| --- | --- | --- |
| GPU idle gaps between tiny kernels | launch overhead, no graph capture, tiny batch | 04 compute, 06 parallelism |
| memcpy dominates timeline | dtype inflation, no pinning, bad staging | 05 memory IO |
| CPU preprocess > model forward | image/token prep or Python stack | 05 memory IO, 07 pipeline |
| one attention kernel dominates | attention backend or KV path | 04 compute |
| throughput drops at higher concurrency | queue design or process model | 06 parallelism |
| memory climbs with request length | KV cache or cache eviction failure | 05 memory IO, 07 pipeline |

## Interpretation Order

Use this sequence after collecting profiler evidence:

1. compare CPU time versus GPU time
2. determine whether the dominant cost is compute, memory traffic, or scheduling
3. check for overlap opportunities between host work, copies, and kernels
4. confirm whether the chosen batch size and concurrency are realistic for production
5. rank only those candidate optimizations that preserve exactness

This prevents misclassifying a pipeline problem as a kernel problem.

## Profiling Deliverable

Always summarize profiling into:

- top 3 hotspots
- percent of end-to-end time per hotspot
- whether each hotspot is CPU, GPU, or transfer
- evidence screenshot or exported table
- recommended next experiment
- parity-safe candidates versus parity-risky candidates

## Common Profiling Mistakes

- profiling only one tiny synthetic input
- collecting traces before warmup is complete
- interpreting profiler overhead as real latency
- chasing secondary kernels before fixing copy or scheduling bottlenecks
- trusting operator names without validating real call frequency
- jumping from hotspot identification directly to lower precision or approximate kernels

## Exit Criteria

Move on only when you can state, in one sentence:

- the dominant bottleneck
- why it dominates
- what one experiment should be run next
- why that experiment preserves exactness
