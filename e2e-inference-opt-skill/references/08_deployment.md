# 08 Deployment

## Goal

Choose the lightest deployment stack that achieves the target latency, throughput, and operability.

Do not migrate to a new runtime just because it is faster on a microbenchmark.

Default deployment rule: do not ship a faster runtime if exact outputs are not preserved.

## Exactness Gate

Every deployment candidate must pass all of these before rollout:

- golden-set exact output parity against the trusted reference path
- identical preprocessing and postprocessing semantics
- deterministic behavior for deterministic workloads
- no stale-cache or request-routing correctness failures under concurrency

If any gate fails, the deployment path is rejected by default.

## Deployment Ladder

Default escalation path:

1. clean PyTorch eager path
2. optimized PyTorch path with better batching, memory, and compiler settings
3. export or engine path such as ONNX, Torch-TensorRT, TensorRT
4. production serving stack such as Triton or a specialized LLM server

In exact-parity mode, stay on the lowest rung that both meets performance goals and preserves exact outputs.

Stay on the lowest rung that meets requirements.

## When To Consider ONNX

Use ONNX when:

- model graph is export-friendly
- cross-runtime portability matters
- ONNX Runtime or TensorRT downstream is likely

Watch for:

- unsupported operators
- dynamic shape complexity
- preprocessing and postprocessing drift outside the graph
- subtle numerical or semantic divergence versus the reference path

## When To Consider TensorRT

Use TensorRT when:

- the model is GPU-served and relatively stable
- profiler shows the PyTorch path is leaving clear performance on the table
- precision lowering and kernel selection matter materially
- the team can manage engine build, validation, and fallback

In exact-parity mode, TensorRT is acceptable only if the built engine matches the trusted path exactly on the golden set. If exact parity fails, do not deploy it by default.

Good candidates:

- stable-shape vision models
- mature Transformer serving paths
- throughput-sensitive GPU deployment

Higher-risk candidates:

- highly dynamic graphs
- research code with frequent architectural changes
- models needing many unsupported custom ops

## When To Consider Triton

Use Triton when:

- multiple models or stages must be served in one platform
- dynamic batching is needed
- observability and versioned deployment matter
- model ensemble or mixed backends are useful

In exact-parity mode, platform gains do not justify any change in output semantics.

Do not adopt Triton if:

- one lightweight service already meets needs
- operational overhead would exceed the performance gain

## LLM Serving Stacks

Evaluate specialized servers when:

- continuous batching is required
- paged KV cache matters
- token streaming and scheduler behavior dominate performance

Keep them only if scheduler behavior, cache behavior, and output mapping remain exactly correct for each request.

Common reasons to leave a custom PyTorch service:

- queueing and batch scheduling complexity
- KV cache memory efficiency
- prefill/decode routing

## Production Validation Matrix

Before rollout, validate:

- cold start and warm steady-state
- long context and short context
- low and high concurrency
- memory headroom after hours of traffic
- rollback path to previous runtime
- golden-set exact parity on the candidate runtime
- hit-path and miss-path parity if caching is involved

## Engine Build Rules

- version all exported artifacts
- pin build environment
- record input shape assumptions
- keep golden inputs and golden outputs
- test unsupported-op fallback behavior explicitly
- record the exactness verification method and final pass result

## Rollout Checklist

| Check | Required |
| --- | --- |
| baseline compared against current production | yes |
| exactness guardrail passes | yes |
| p95 and p99 meet SLA | yes |
| memory headroom is acceptable | yes |
| fallback or rollback path exists | yes |
| observability dashboards updated | yes |

## Canary Strategy

Prefer:

- low traffic slice
- representative request mix
- explicit error and latency abort thresholds
- side-by-side metric comparison

Never canary only the easy path and then generalize to full production traffic.

## Common Mistakes

- exporting too early before single-runtime optimization is exhausted
- claiming TensorRT win without end-to-end service comparison
- forgetting preprocessing and postprocessing when comparing runtimes
- building engines on one environment and deploying on an incompatible one
- omitting rollback drills
- accepting runtime-specific output drift because aggregate metrics look good

## Exit Criteria

A deployment path is ready only when:

- performance gain is proven end-to-end
- correctness is validated
- operational ownership is clear
- rollback is tested
- exact output parity is proven or an explicit human exception exists
