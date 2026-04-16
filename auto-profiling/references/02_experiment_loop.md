# 02 Experiment Loop

## Goal

Define a repeatable loop for autonomous profiling and performance improvement.

## Standard Loop

```text
read aim.md
  -> validate environment
  -> run trusted baseline
  -> collect exactness evidence
  -> collect performance evidence
  -> choose one safe mutation lane
  -> apply one bounded change
  -> rerun exactness gate
  -> rerun benchmark/profile
  -> keep or revert
  -> log the experiment
  -> repeat until budget is exhausted
```

## Stage 1: Read The Contract

Extract from `aim.md`:

- metric to optimize
- exactness requirements
- allowed mutation surfaces
- blocked mutation types
- runtime budget

If any of these are missing, the agent should fill nothing in by guesswork. It should operate conservatively and stay within the clearly declared scope.

## Stage 2: Establish The Baseline

Collect:

- command success or failure
- environment fingerprint
- baseline metric
- baseline profiler output
- baseline exactness result

Store a baseline snapshot before the first mutation.

## Stage 3: Choose The Next Experiment

Pick exactly one lane per iteration:

- measurement cleanup
- profiling visibility
- copy or memory reduction
- scheduling or overlap
- cache implementation
- compile or backend path

Do not combine lanes unless the first lane is purely instrumentation.

## Stage 4: Run The Gate Order

Always run in this order:

1. syntax or build sanity if needed
2. exactness gate
3. benchmark gate
4. profiling gate if relevant

If exactness fails, stop immediately and revert. Performance numbers from a failed exactness run do not count.

## Stage 5: Keep Or Revert

### Keep

Keep only when:

- exactness passes
- target metric improves
- no unacceptable regression appears in memory, startup, or stability

### Revert

Revert when:

- exactness fails
- metric does not improve meaningfully
- the change escapes the declared mutation scope
- the candidate introduces operational risk larger than the gain

## Stage 6: Log The Experiment

Each log entry should record:

- experiment id
- hypothesis
- changed files
- commands run
- exactness result
- baseline metric
- candidate metric
- keep or revert decision
- next recommendation

## Budgeting

Good default budgeting:

- small number of bounded iterations per session
- fixed per-experiment runtime budget
- immediate stop on repeated exactness failures

This keeps autonomous iteration from turning into undirected churn.

## Common Mistakes

- profiling before baseline exactness is known
- letting the candidate survive after a failed parity check
- changing code and benchmark harness in the same experiment without separation
- optimizing for a metric that is not the one declared in `aim.md`
