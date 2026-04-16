# 07 Pipeline And Cache

## Goal

Improve end-to-end inference by overlapping stages and reusing expensive work instead of repeatedly accelerating the same work from scratch.

No overlap or cache optimization is acceptable if it serves stale outputs, mixes request state, or changes the effective computation.

## Exactness Rule

Before keeping a pipeline or cache optimization, verify:

- the same request yields the same output as the reference path
- cache hit and cache miss paths are equivalent
- background work cannot expose partially built artifacts
- overlap does not reorder or corrupt request ownership

## Typical Multi-Stage Pipeline

```text
ingest -> decode/load -> preprocess -> H2D -> model forward -> postprocess -> store/respond
```

For many production systems, the highest return comes from:

- overlapping adjacent stages
- caching immutable or slow-changing artifacts
- making queue ownership explicit

## Overlap Strategy

### What can often overlap

- CPU load/decode with previous request GPU forward
- next batch H2D with current batch compute
- postprocess with next request staging
- embedding build in background while serving cached results

### What often cannot overlap safely

- hidden synchronization inserted by debug code
- cache rebuild that competes on the same GPU without admission control
- shared mutable buffers without lifecycle discipline
- any path that exposes partially written cache entries

## Two-Level Cache Pattern

Use this when reference data changes slowly but queries are frequent.

- L1: in-process memory cache for fastest reuse
- L2: disk, object store, or distributed cache for restart resilience

Good fits:

- embedding gallery
- retrieval corpus features
- reusable prompt encodings or prefix states
- model artifacts after deterministic preprocessing

## Cache Design Rules

- cache key must be content-derived or version-derived
- invalidation must be explicit
- cache value should match the access pattern
- keep GPU memory cache only when reuse rate justifies it
- prefer CPU-side or disk-side cache when GPU memory is precious
- include model version and preprocessing version in cache identity
- make writes atomic so readers never observe partial state

## LLM-Specific Reuse

Relevant patterns:

- paged KV cache
- shared prefix reuse
- prefix tree or prefix-hash lookup
- prefill and decode separation for uneven traffic

Questions:

- is cache locality improving throughput or only increasing complexity?
- do long prompts evict the short prompts that matter most?
- is cache reuse worth the extra bookkeeping and memory?
- can any cache path return stale data or mismatched request state

## Pipeline Capacity Planning

Measure every stage:

- service time
- queue wait
- max concurrency
- memory footprint

The slowest stage sets throughput. The burstiest stage often sets tail latency.

## A Simple Backpressure Model

If stage B is slower than stage A:

- stage A must slow down
- queue must grow
- or work must be dropped or degraded

Optimization options:

- accelerate stage B
- batch stage B more effectively
- move some B work earlier or later
- add resource isolation

Choose the option that preserves exact semantics first, even if it is not the most aggressive speedup.

## Vision And Retrieval Pattern

Common high-value path:

- cache gallery or reference features
- keep query preprocessing cheap
- compute similarity on GPU
- only materialize small result objects back to CPU

## Diffusion Pattern

High-value reuse candidates:

- text encoder outputs for repeated prompts or templates
- scheduler state where framework permits
- preallocated latent and workspace buffers

## Failure Modes

- cache key ignores preprocessing version and serves stale outputs
- overlap attempt actually serializes due to hidden sync
- background rebuild steals capacity from foreground traffic
- queue grows without admission control and destroys p95
- hit path and miss path return different outputs
- request A can ever observe artifacts computed for request B

## Observability

Track:

- cache hit rate by tier
- queue length by stage
- overlap effectiveness from timeline evidence
- rebuild time and rebuild frequency
- p95 with and without cache hits
- exact parity on hit path versus miss path

## Exit Criteria

Proceed only when you can describe:

- which work is reused
- which stages truly overlap
- how invalidation works
- what protects the system from stale data and runaway queues
- why cache-hit and overlap paths preserve exact outputs
