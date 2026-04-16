# 04 Exactness Gate

## Goal

Make exact output preservation the hard gate of the autonomous loop.

## Default Policy

The default policy is binary:

- pass exactness → experiment may continue
- fail exactness → revert immediately

There is no partial credit in exact-parity mode.

## Supported Modes

### Exact-Parity

Use when the logical path, algorithm, device class, and precision contract are effectively unchanged.

Rule:

- any mismatch fails

### Bounded-Tolerance

Use only when:

- logic remains equivalent
- algorithm remains equivalent
- the difference comes from declared device changes or declared precision transitions
- `abs_tolerance` and `rel_tolerance` are explicit in `aim.md`

Rule:

- exceed either tolerance and fail
- if logic or algorithm equivalence fails, tolerance mode is not allowed to rescue the run

## What Must Match

Depending on the workload, define one or more of:

- exact tensor equality
- exact decoded output equality
- exact label equality
- exact ranking equality
- exact cache-hit versus cache-miss equality
- exact per-request isolation under concurrency

## Golden Set

The agent should keep a golden set with:

- representative small inputs
- edge-case inputs
- warm-path and cold-path cases if relevant
- concurrency-sensitive cases if the service is concurrent

The golden set should be cheap enough to rerun every iteration.

## Gate Order

Run exactness checks before performance checks.

```text
candidate built
  -> exactness gate
     -> fail => revert
     -> pass => benchmark and profile
```

## Recommended Checks

### Functional parity

- reference output equals candidate output
- or, in declared tolerance mode, numeric error stays within both declared thresholds

### Cache parity

- miss path equals reference
- hit path equals miss path

### Concurrency parity

- request A never receives request B output
- repeated concurrent runs remain stable

### Determinism parity

- deterministic workloads remain deterministic across runs

## Failure Handling

On failure:

- stop collecting performance claims for that candidate
- record the first mismatch and reproduction command
- revert to last known-good state
- classify failure type

Suggested failure classes:

- numeric mismatch
- stale cache
- request routing bug
- nondeterministic behavior
- baseline harness error
- tolerance contract violated

## Exactness Log Template

| Experiment | Check Type | Result | First Failure | Reproduction |
| --- | --- | --- | --- | --- |
| exp-001 | tensor equality | pass |  |  |
| exp-002 | cache parity | fail | sample 4 hit path | command |

## Common Mistakes

- using loose tolerances by habit
- using tolerance mode for logic or algorithm changes
- testing only the happy path
- verifying only cache misses
- verifying only sequential execution when concurrency is enabled
- accepting mismatch because latency improved
