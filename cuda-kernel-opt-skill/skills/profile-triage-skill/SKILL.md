---
name: "profile-triage-skill"
description: "Use when interpreting serving profiles to rank bottlenecks, overlap headroom, and fuse opportunities."
---

# Profile Triage Skill

## Use When

- you already have a torch profiler, nsys, or similar trace
- you need to explain why TTFT or TPOT is bad
- you need a ranked bottleneck statement instead of raw screenshots

## Output Format

Always produce three tables:

1. kernel table
2. overlap-opportunity table
3. fuse-pattern table

## Interpretation Order

1. split prefill vs decode
2. find top kernel families by time share
3. inspect launch count and fragmentation
4. inspect copy, sync, and idle gaps
5. state one dominant bottleneck

## Typical Findings

- too many tiny decode kernels
- prefill dominated by GEMM or attention
- CPU preprocessing on the critical path
- cache or copy traffic dominating memory bandwidth
- a fused path not actually active

## Guardrails

- do not recommend fusion without a repeated pattern
- do not recommend backend migration before serving evidence exists
- do not confuse profile size with bottleneck size

## Deliverables

- three triage tables
- one dominant bottleneck statement
- 3 ranked next experiments

## Inputs To Ask For

- trace source: torch profiler, nsys, or framework-native trace
- exact benchmark command
- whether the trace covers prefill, decode, or both
- target symptom: TTFT, TPOT, ITL, throughput, or memory
