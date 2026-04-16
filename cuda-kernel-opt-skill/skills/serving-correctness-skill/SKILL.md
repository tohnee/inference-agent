---
name: "serving-correctness-skill"
description: "Use when defining parity rules, golden prompts, regression gates, and safe acceptance criteria for LLM serving changes."
---

# Serving Correctness Skill

## Use When

- any serving optimization may affect outputs
- backend, precision, cache, or scheduler changes are proposed
- you need a golden prompt and parity strategy

## Default Rule

Speed never outranks correctness.

## Correctness Checklist

- deterministic decode configuration when possible
- stable golden prompt set
- exact token or text comparison for deterministic paths
- explicit tolerance only when the human allows it
- cache hit and miss equivalence
- concurrent-request isolation

## Changes That Need Extra Scrutiny

- quantization
- speculative decoding
- prefix caching
- scheduler rewrites
- backend migration
- custom kernels

## Deliverables

- exactness contract
- golden prompt set definition
- regression gate checklist
- keep or reject recommendation

## Evidence Before Accepting A Change

- golden prompt outputs
- cache hit/miss equivalence
- concurrency isolation check
- regression result against baseline
- explicit human approval if drift is tolerated
