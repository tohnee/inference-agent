# 06 KV Cache and Scheduler

## Goal

Optimize the parts of LLM serving that most often dominate real-world latency: KV cache behavior and request scheduling.

## KV Cache Questions

Always answer:

- is the cache paged, ragged, contiguous, or hybrid?
- what is the true cache hit path for shared prefixes?
- what is the peak memory footprint under the target concurrency?
- does fragmentation grow over time?
- what happens on eviction?

## High-Value Lanes

### Prefix and shared-prefix caching

Useful when:

- many requests share a long system prompt
- workloads have repeated context windows
- prompt reuse is common across sessions

Verify:

- exact token equality on both hit and miss paths
- no stale reuse after prefix changes
- correct invalidation after cache updates

### Paged or ragged KV cache

Useful when:

- sequences vary heavily in length
- fragmentation is limiting concurrency
- decode is bandwidth sensitive

### Cascade and hierarchical cache patterns

Useful when:

- shared prefixes are long
- many branches decode from the same prompt root

## Scheduler Questions

Always measure:

- queueing delay
- active batch size over time
- fairness across short and long requests
- prefill-to-decode handoff delay
- cancellation and timeout handling

## Continuous Batching

Continuous batching is often the first major serving win, but it is only safe if:

- request isolation is preserved
- scheduling does not starve short requests
- decode latency tails stay under SLA
- benchmark traffic resembles production traffic

## Prefill and Decode Split

Treat prefill and decode as different systems.

Common patterns:

- prefill is compute-dense and larger-kernel heavy
- decode is memory-bandwidth and launch-overhead sensitive
- combining them naively can raise tail latency

If using PD disaggregation, benchmark and profile them separately before drawing conclusions.

## Common Mistakes

- treating cache hit rate as a speed metric without correctness validation
- optimizing average concurrency but ignoring tail latency
- using a scheduler change to fix what is actually a memory-fragmentation problem
- assuming prefix cache wins apply equally to all prompt distributions
