---
name: "kv-cache-prefix-cache-skill"
description: "Use when optimizing KV cache layout, prefix caching, cache reuse, or cache isolation across serving frameworks."
---

# KV Cache Prefix Cache Skill

## Use When

- memory footprint is too high
- prefix caching or cache reuse is underperforming
- cache correctness or isolation is in doubt
- fragmentation limits concurrency

## Primary Questions

- what cache layout is in use?
- what is the hit/miss semantics?
- how is cache uniqueness defined?
- can multi-tenant or multi-LoRA requests collide incorrectly?
- what are the eviction and reuse rules?

## Cross-Framework Notes

- **vLLM**: prefix caching is hash-based and cache key design matters for determinism and isolation
- **SGLang**: shared-prefix and scheduler behavior must be observed under real traffic
- **TensorRT-LLM**: KV memory fraction and reuse strategy strongly affect throughput and latency

## Validation Rules

- exact outputs on cache hit and miss paths
- no stale results after prompt changes
- no incorrect cross-request reuse
- stable memory growth under load

## Deliverables

- cache semantics summary
- memory and hit-rate observations
- correctness risks
- 3 ranked cache experiments

## Evidence To Collect

- cache hit and miss latency
- hit-rate under realistic prompt reuse
- peak memory and fragmentation trend
- correctness comparison for reused vs non-reused paths
