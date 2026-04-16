# 01 Baseline

## Goal

Build a baseline that is stable enough to support engineering decisions across CPU and GPU inference.

If the baseline is weak, every downstream optimization conclusion is weak.

## Exactness Contract

Baseline collection starts with the correctness contract, not with timing.

Default rule for this skill:

- optimized path must produce exactly the same result as the trusted baseline path
- if the workload is deterministic, require exact tensor equality or exact decoded output equality
- if the product contract is label or ranking based, still record tensor equality where feasible and treat any output divergence as a failure unless explicitly approved

Before collecting latency numbers, freeze:

- reference model weights and checkpoint hash
- reference preprocessing and postprocessing implementation
- golden input set
- golden output set
- deterministic decoding settings if generation is involved

## What To Record First

### Hardware fingerprint

- CPU model, sockets, cores, threads
- NUMA topology if relevant
- GPU SKU, memory size, PCIe or SXM, MIG or vGPU status
- driver version, CUDA runtime, cuDNN, NCCL, TensorRT, PyTorch version
- clock policy, power limit, thermal constraints

Useful optional fields when available:

- theoretical memory bandwidth
- theoretical FP32, TF32, FP16, BF16, or FP8 peak
- PCIe generation or NVLink topology
- virtualization quota or tenant limit

### Runtime fingerprint

- Python version
- PyTorch build and CUDA availability
- `torch.backends` precision flags
- container image tag or host environment hash
- model name, checkpoint hash, tokenizer or processor version
- input shape distribution and representative sample source

### Business metrics

At minimum capture:

- mean latency
- p50, p95, p99 latency
- throughput
- max stable concurrency
- peak memory
- accuracy or quality guardrail
- cost proxy such as requests per GPU or tokens per second per GPU

In exact-parity mode also capture:

- golden-set pass rate
- exact mismatch count
- first mismatching sample identifier

## Scenario Matrix

Never report only one number. Measure at least these scenarios:

| Scenario | Why it matters |
| --- | --- |
| cold start | load cost, graph compile, cache miss, first request penalty |
| warm steady-state | primary SLA path for long-running services |
| burst | queue buildup and scheduler resilience |
| max safe batch | throughput ceiling and memory pressure |
| concurrency sweep | service operating point |

For autoregressive models add:

- prefill-only
- decode-only
- mixed request traffic
- long-context path

## Timing Rules

- CPU sections use `time.perf_counter()`
- GPU kernel timing uses `torch.cuda.Event(enable_timing=True)`
- synchronization is allowed only outside the measured hot path
- warm up before recording measurements
- use fixed seeds where feasible
- pin input distributions for A/B experiments

Timing data is invalid for decision-making if the optimized path has not already passed the exactness contract.

## Determinism Controls

Use deterministic settings when the workload and operator set support them.

```python
import torch

def enable_deterministic_mode():
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
```

## Indicative Hardware Template

Treat this as a starting point, not ground truth. Always prefer measured effective behavior on the target environment.

| Device | Typical precision focus | Typical memory bandwidth class | Notes |
| --- | --- | --- | --- |
| A40 | FP16, BF16 | mid-to-high | PCIe deployment card, good general-purpose baseline |
| L20 | FP16, BF16 | high | often used for inference-heavy deployments |
| A100 | BF16, FP16, TF32 | very high | strong balance for both training and inference |
| H100 | BF16, FP16, FP8 | extremely high | Hopper-specific gains depend on stack maturity |
| vGPU or MIG slice | quota-dependent | quota-dependent | effective peak may differ materially from bare metal |

## Minimal Fingerprint Script

```python
import os
import platform
import subprocess
import torch

def runtime_fingerprint():
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            {
                "gpu_name": props.name,
                "gpu_mem_gb": round(props.total_memory / 1024**3, 2),
                "sm_count": props.multi_processor_count,
                "cc": f"{props.major}.{props.minor}",
            }
        )
    return info
```

## Benchmark Suite Pattern

For reusable benchmarking, structure the harness around three modes:

- latency
- throughput
- concurrency

This keeps online and offline inference comparable under one measurement contract.

```python
import time
import numpy as np
import threading
import torch

class BenchmarkSuite:
    def __init__(self, warmup=10, repeat=100):
        self.warmup = warmup
        self.repeat = repeat

    def measure_latency(self, fn):
        for _ in range(self.warmup):
            fn()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(self.repeat)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(self.repeat)]
        for i in range(self.repeat):
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize()
        samples = np.array([s.elapsed_time(e) for s, e in zip(starts, ends)])
        return {
            "mean_ms": float(samples.mean()),
            "p50_ms": float(np.percentile(samples, 50)),
            "p95_ms": float(np.percentile(samples, 95)),
            "p99_ms": float(np.percentile(samples, 99)),
        }

    def measure_throughput(self, fn, batch_size, duration_sec=10.0):
        for _ in range(self.warmup):
            fn()
        torch.cuda.synchronize()
        count = 0
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < duration_sec:
            fn()
            count += batch_size
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        return {
            "samples_per_sec": count / elapsed,
            "batches_per_sec": count / max(batch_size, 1) / elapsed,
        }

    def measure_concurrency(self, fn, workers=4, requests=100):
        latencies = []
        lock = threading.Lock()

        def worker():
            for _ in range(requests // workers):
                t0 = time.perf_counter()
                fn()
                torch.cuda.synchronize()
                with lock:
                    latencies.append((time.perf_counter() - t0) * 1000)

        threads = [threading.Thread(target=worker) for _ in range(workers)]
        start = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.perf_counter() - start
        arr = np.array(latencies)
        return {
            "qps": len(latencies) / elapsed,
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
        }
```

## Exactness Harness Pattern

Run parity verification before trusting any benchmark result.

```python
import torch

def assert_exact_outputs(reference_fn, candidate_fn, golden_inputs):
    for idx, sample in enumerate(golden_inputs):
        ref = reference_fn(sample)
        got = candidate_fn(sample)
        if isinstance(ref, torch.Tensor):
            if not torch.equal(ref, got):
                raise AssertionError(f"exact mismatch at sample {idx}")
        else:
            if ref != got:
                raise AssertionError(f"exact mismatch at sample {idx}")
```

## Minimal Latency Harness

```python
import time
import numpy as np
import torch

def benchmark_gpu_step(fn, warmup=20, repeat=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    for i in range(repeat):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()

    samples = np.array([s.elapsed_time(e) for s, e in zip(starts, ends)])
    return {
        "mean_ms": float(samples.mean()),
        "p50_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
        "p99_ms": float(np.percentile(samples, 99)),
        "min_ms": float(samples.min()),
        "max_ms": float(samples.max()),
    }
```

## Throughput Harness

For online services:

- measure end-to-end requests per second
- include queue wait separately from model service time
- sweep concurrency: 1, 2, 4, 8, 16, 32

For offline jobs:

- sweep batch size until memory limit or throughput plateau
- track both items per second and cost per item

## Standard Baseline Collection Order

Run baseline collection in this order:

1. fingerprint hardware and runtime
2. validate one representative input path
3. record cold-start latency
4. record warm steady-state latency
5. sweep batch size
6. sweep concurrency
7. sweep precision only after correctness guardrail is defined
8. save results in a machine-readable file for later comparisons

## Accuracy Guardrail

Every optimization experiment must define a no-regression rule.

In this skill, the default no-regression rule is exact equality, not approximate equality.

Approximate metrics are supporting evidence only and do not override exact mismatch failures.

Secondary metrics may still be useful:

- classifier or ranker: top-1, top-k, recall, AUC, cosine delta
- generator: output similarity, token-level parity where possible, quality eval sample
- diffusion: CLIP score, seed-fixed visual diff, prompt set pass rate

Precision changes without an exactness exception from the human are not allowed in default mode.

## Baseline Table Template

| Case | Batch | Concurrency | Mean | P95 | Throughput | Peak Mem | Accuracy Delta | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cold | 1 | 1 |  |  |  |  |  |  |
| warm | 1 | 1 |  |  |  |  |  |  |
| warm | 8 | 1 |  |  |  |  |  |  |
| warm | 8 | 8 |  |  |  |  |  |  |

Add a parity table beside it:

| Case | Golden Samples | Exact Pass | Mismatch Count | First Failure | Notes |
| --- | --- | --- | --- | --- | --- |
| baseline reference |  |  |  |  |  |
| optimized candidate |  |  |  |  |  |

## Common Baseline Mistakes

- mixing preprocessing, network, and postprocess into one opaque timer
- measuring only average latency
- comparing two runs with different input shapes
- using synthetic inputs only, then generalizing to production
- benchmarking under thermal throttling or noisy neighbors without noting it
- treating vGPU results as if they were bare-metal GPU results
- accepting a speedup before running golden-set verification
- using tolerance-based equality by habit when the business contract requires exactness

## Exit Criteria

Do not move to optimization until all are true:

- baseline table is complete
- exactness contract is written down
- golden-set verification passes for the current trusted path
- hardware/runtime fingerprint is recorded
- at least one warm steady-state run is repeatable
- exactness guardrail is defined
- concurrency and batch sweep data exist for the main scenario
