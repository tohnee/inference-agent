# 01 Operating Model

## Goal

Make autonomous profiling predictable, bounded, and reviewable.

The human should configure intent once in `aim.md`. The agent should then iterate without requiring repeated human steering.

## Core Design

Auto-profiling follows a deliberately constrained model:

- one human objective file
- one trusted baseline path
- one experiment at a time
- one keep-or-revert decision per experiment
- one append-only experiment history

This keeps diffs small and experiment causality understandable.

The minimal runtime assumes:

- `uv` is used to run the local commands
- git is available for revision tracking
- `runner.py` is the contract executor
- `scorer.py` is the keep-or-revert decision engine
- planner, generator, and evaluator responsibilities are separated by artifacts even if one Claude session performs them all

## Human Versus Agent Responsibilities

### Human responsibilities

- provide runnable baseline code
- provide environment commands
- define exactness contract
- define mutation boundaries
- define success metric

### Agent responsibilities

- validate that baseline actually runs
- collect baseline evidence
- choose the highest-value safe optimization lane
- run exactly one bounded experiment
- verify parity first, then performance
- keep or revert
- log the result

## Baseline-First Rule

No autonomous iteration starts before these pass:

- baseline setup command succeeds
- baseline run command succeeds
- baseline exactness check succeeds
- baseline profiling command or equivalent evidence path succeeds

If any of these fail, the session is in setup mode, not optimization mode.

## Fixed Experiment Envelope

Each experiment should have:

- a short name
- one changed variable
- one exactness result
- one metric result
- one keep-or-revert decision

This is the profiling equivalent of a fixed-time research run: every iteration should be comparable, bounded, and logged.

## Recommended Workspace Artifacts

The agent should create or update these in the target repository, not in the skill folder:

- `.auto-profiling/experiment_log.md`
- `.auto-profiling/experiment_log.jsonl`
- `.auto-profiling/best_result.json`
- `.auto-profiling/baseline_snapshot.json`
- `.auto-profiling/task_plan.md`
- `.auto-profiling/findings.md`
- `.auto-profiling/progress.md`
- `.auto-profiling/worklog.md`
- `.auto-profiling/session_state.json`
- `.auto-profiling/current_contract.md`
- `.auto-profiling/evaluator_report.md`
- `.auto-profiling/next_handoff.md`

## Session States

| State | Meaning | Next action |
| --- | --- | --- |
| setup | baseline or environment not ready | fix execution path |
| baseline | trusted path established | collect baseline evidence |
| contract | bounded experiment defined | perform one safe mutation |
| profiling | bottleneck identification active | choose one lane |
| experiment | candidate optimization active | run parity then measurement |
| revert | candidate failed | restore previous best |
| promote | candidate improved and passed parity | mark as new best |
| handoff | session is safe to reset or resume | start from handoff artifact |

## Common Failure Modes

- baseline command exists but is not reproducible
- exactness command is missing or underspecified
- mutation scope is too wide
- agent changes multiple files and variables in one iteration
- failed runs are not reverted cleanly

## Exit Criteria

The operating model is healthy only when:

- the current best version is known
- every experiment is attributable
- exactness status is visible
- the next experiment can be chosen from evidence, not guesswork
