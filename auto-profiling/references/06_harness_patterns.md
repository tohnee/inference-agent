# 06 Harness Patterns

## Goal

Keep long-running auto-profiling sessions coherent, skeptical, and restart-friendly.

## Core Harness Idea

The harness should not rely on one endlessly growing session.

Instead it should rely on:

- bounded experiments
- structured handoff artifacts
- explicit planner and evaluator roles
- resumable state on disk

## Planner, Generator, Evaluator

Map the roles to auto-profiling like this:

- **planner**: define the next bounded experiment contract
- **generator**: apply one change inside the allowed mutation surface
- **evaluator**: independently decide keep or revert using exactness, metric, scope, and reproducibility criteria

Even if the same Claude session performs all three roles, the artifacts should keep the reasoning separated.

## Structured Handoff Files

Use these files as persistent state:

- `current_contract.md`
- `evaluator_report.md`
- `next_handoff.md`
- `session_state.json`
- `experiment_log.jsonl`

These reduce drift when the session is long or must be resumed later.

## Contract Before Change

Before modifying code, define:

- current bottleneck
- selected mutation lane
- exactness mode
- success metric
- keep or revert rule

If the contract is unclear, the run is not ready for mutation.

## Evaluator Criteria

The evaluator should judge at least these dimensions:

1. exactness contract
2. target metric improvement
3. scope compliance
4. operational stability
5. reproducibility

This makes the evaluator skeptical in concrete ways instead of vaguely positive.

## Exactness Modes

### Exact-Parity

Use when:

- same device class
- same precision contract
- same algorithm and logic path

Rule:

- any mismatch fails the experiment

### Bounded-Tolerance

Use only when all are true:

- logic remains equivalent
- algorithm remains equivalent
- the difference comes from declared device or precision transitions
- `abs_tolerance` and `rel_tolerance` are explicitly declared in `aim.md`

Rule:

- exceed either tolerance and fail the experiment

## Long-Running Session Discipline

To prevent drift:

- keep one changed variable per experiment
- write evaluator and handoff artifacts after every iteration
- prefer fresh-session resume from artifacts over relying on long conversational memory
- keep the state machine explicit

## Simplification Principle

Every harness component should justify itself.

If a new model or simpler workflow can replace a scaffold without losing reliability, remove that scaffold. Keep only load-bearing complexity.

## Common Failure Modes

- the agent grades its own work too generously
- the session tries to optimize too many things at once
- no handoff artifact exists, so a resumed session loses intent
- tolerance mode is used without proving logic and algorithm equivalence
- the evaluator checks performance but not scope or reproducibility
