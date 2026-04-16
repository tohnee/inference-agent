# 03 Mutation Lanes

## Goal

Give the agent a disciplined menu of optimization moves so it does not jump randomly across the system.

## Lane Priority In Exact-Parity Mode

Prefer mutation lanes in this order:

1. instrumentation and benchmark cleanup
2. copy reduction and memory movement cleanup
3. scheduling, overlap, and queue cleanup
4. cache correctness and cache reuse
5. compile or backend changes
6. precision or approximate techniques only if the human explicitly allows them

## Lane 1: Instrumentation

Examples:

- improve timing boundaries
- separate cold and warm runs
- export clearer profiler traces
- record queue wait separately from service time

Why it is safe:

- it should not change math
- it improves decision quality for later iterations

## Lane 2: Memory And IO

Examples:

- pinned-memory staging
- non-blocking transfers
- compact transfer dtype that preserves the exact logical input representation
- keeping postprocess on GPU longer without changing values

Use when:

- profiler shows memcpy or copy gaps
- CPU-side staging dominates wall time

## Lane 3: Scheduling And Overlap

Examples:

- overlap H2D with compute
- restructure request queues
- reduce launch gaps
- improve worker topology

Use when:

- GPU is underutilized
- queue wait is high
- kernels are separated by visible idle gaps

## Lane 4: Cache And Reuse

Examples:

- add L1 and L2 cache for reusable features
- cache deterministic preprocess outputs
- add prefix reuse for safe serving paths

Use when:

- repeated work dominates
- cold and warm paths differ sharply

Gate:

- hit path and miss path must be exactly equivalent

## Lane 5: Compiler And Backend

Examples:

- `torch.compile`
- CUDA Graphs
- backend selection within the same numerical contract
- export/runtime evaluation

Use when:

- the hot path is stable
- profiler shows many small ops or launch overhead
- previous safer lanes are exhausted or insufficient

Gate:

- exactness proof is mandatory before any performance claim

## Lane 6: Explicit Exception Lanes

These are outside default mode:

- TF32, FP16, BF16, FP8 changes versus baseline
- quantization
- approximate kernels
- algorithmic output changes

Use only when `aim.md` explicitly allows leaving exact-parity mode.

## Lane Selection Questions

Before choosing a lane, answer:

1. what is the current dominant bottleneck
2. which lane best targets it
3. can this lane preserve exact outputs
4. can the lane be tested in one bounded experiment

## Common Mistakes

- picking the most glamorous lane instead of the safest high-value lane
- skipping instrumentation and moving straight to compiler or engine work
- mixing cache changes and backend changes in one iteration
- treating precision reduction as a default performance lane
