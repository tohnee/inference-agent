# 08 Agentic Optimization

## Goal

Convert LLM serving optimization into a guarded agent workflow that can later be executed by `auto-profiling`.

## Division of Responsibility

Use the current repository layers like this:

- `llm-serving-opt-skill` chooses the right optimization lane
- `e2e-inference-opt-skill` provides generic inference principles when the issue is broader than LLM serving
- `auto-profiling` runs bounded experiments and keep-or-revert decisions

## Lane Mapping

| Observed problem | Skill lane | Future experiment lane |
| --- | --- | --- |
| TTFT too high | baseline + benchmark + deployment | prefill path, tokenizer, chunked prefill |
| TPOT too high | benchmark + profile + backend | decode kernel, launch overhead, scheduler |
| crash during serving | crash triage | minimal reproducer, backend toggle |
| memory spikes | baseline + KV cache | cache layout, eviction, batch limits |
| concurrency collapse | benchmark + scheduler | continuous batching, fairness, overlap |

## Recommended `aim.md` Extensions for LLM Serving

When adapting `auto-profiling`, consider adding fields such as:

- `serving_framework`
- `server_launch_command`
- `benchmark_command`
- `golden_prompt_set_path`
- `ttft_metric_name`
- `tpot_metric_name`
- `itl_metric_name`
- `kv_cache_mode`
- `scheduler_mode`
- `prefill_profile_path`
- `decode_profile_path`

## Experiment Rules

For agent-driven serving optimization:

1. one variable per experiment
2. parity gate before performance gate
3. serving benchmark before kernel benchmark for product conclusions
4. preserve launch and benchmark commands as artifacts
5. record rejected experiments, not only successful ones

## Minimal Safe Loop

1. run deterministic smoke prompts
2. run `bench_serving`
3. collect one profile or crash artifact
4. choose one lane
5. implement one bounded change
6. rerun parity
7. rerun benchmark
8. keep or revert

## What Not To Automate Blindly

Avoid fully automatic mutation of these until parity and observability are mature:

- quantization mode switches
- speculative decoding changes
- scheduler rewrites
- cache invalidation logic
- custom CUDA kernel generation

## Exit Criteria

A candidate optimization is ready for promotion only if:

- parity passes
- TTFT, TPOT, ITL, or throughput improves on the declared target
- no new memory or stability regression appears
- the launch and benchmark recipe is reproducible
