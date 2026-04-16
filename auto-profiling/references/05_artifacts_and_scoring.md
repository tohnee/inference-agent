# 05 Artifacts And Scoring

## Goal

Define what the agent records and how experiments are scored so the system can run repeatedly without becoming opaque.

## Required Artifacts

The target repository should maintain:

- `.auto-profiling/experiment_log.md`
- `.auto-profiling/experiment_log.jsonl`
- `.auto-profiling/baseline_snapshot.json`
- `.auto-profiling/best_result.json`
- `.auto-profiling/task_plan.md`
- `.auto-profiling/findings.md`
- `.auto-profiling/progress.md`
- `.auto-profiling/worklog.md`
- `.auto-profiling/session_state.json`
- `.auto-profiling/current_contract.md`
- `.auto-profiling/evaluator_report.md`
- `.auto-profiling/next_handoff.md`
- `.auto-profiling/failures/` for saved failure summaries when needed

The runtime shape is defined in [log_schema.json](file:///Users/tc/Downloads/推理优化skills/auto-profiling/log_schema.json).

## Baseline Snapshot Schema

Suggested fields:

- timestamp
- git revision if available
- environment fingerprint
- baseline metric
- baseline exactness result
- baseline command set
- target files allowed to change

## Best Result Schema

Suggested fields:

- experiment_id
- metric_name
- baseline_value
- best_value
- relative_improvement
- exactness_status
- changed_files
- keep_reason

## Experiment Log Entry

Each entry should include:

- id
- hypothesis
- lane
- changed files
- exactness result
- metric before and after
- revert or keep
- why the decision was made

The markdown files are human-readable modules. The JSONL file is the machine-readable stream for automation and later analysis.

## Scoring Rule

Score experiments lexicographically:

1. exactness pass beats exactness fail
2. among passes, better target metric wins
3. among similar metrics, lower operational risk wins
4. among similar risk, smaller change wins

In bounded-tolerance mode, an experiment is still an exactness failure if:

- logic equivalence fails
- algorithm equivalence fails
- absolute error exceeds `abs_tolerance`
- relative error exceeds `rel_tolerance`

This prevents the system from preferring a flashy but fragile change over a cleaner equivalent gain.

## Rejection Reasons

Common explicit rejection reasons:

- failed exactness gate
- no meaningful metric gain
- regression in memory or startup
- mutation exceeded declared scope
- result not reproducible
- tolerance contract violated

## Promotion Rule

Promote a candidate to best-known result only when:

- exactness passes
- the metric is strictly better than current best under the same measurement contract
- the run is reproducible

## Suggested Summary Output

At the end of a session, the agent should produce:

- current best candidate
- metric delta versus baseline
- exactness status
- experiments attempted
- experiments rejected and why
- most promising next lane

## Common Mistakes

- keeping results only in prose and not in structured artifacts
- overwriting baseline evidence
- comparing experiments measured under different contracts
- claiming a best result without recording exactness status
