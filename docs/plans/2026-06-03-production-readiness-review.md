# 2026-06-03 Production Readiness Review

## Review Goal

This review treats `inference-agent` as a production engineering system rather than a demo harness. The target product posture is:

- one stable user contract (`auto-profiling/aim.md`);
- deterministic runtime state in `<target_repo>/.auto-profiling/`;
- exactness-first promotion rules;
- bounded command execution so a bad benchmark, profiler, or exactness script cannot hang an automation session forever;
- auditable handoff artifacts that let a human or a later agent resume safely.

## What Was Reviewed

- Runtime entry point and CLI: `auto-profiling/runner.py`.
- Score/evaluator logic: `auto-profiling/scorer.py`.
- Aim API and routing: `auto-profiling/aim_schema.json`, `auto-profiling/skill_routes.json`, and `auto-profiling/bootstrap_aim.py`.
- Regression coverage: repository-level `tests/` and `auto-profiling/tests/`.
- User-facing onboarding docs: `README.md`, `README.en.md`, and `auto-profiling/README.md`.

## Findings

### P0 / Production Blockers

1. **External commands had no timeout guard.**
   - Risk: `baseline_run_command`, profiler commands, exactness checks, or install/warmup scripts could hang the full session indefinitely.
   - Fix: command execution now accepts a timeout sourced from `command_timeout_seconds` or, by default, `max_runtime_per_experiment` in `aim.md`.
   - Behavior: timed-out commands return `exit_code: 124`, `timed_out: true`, `timeout_seconds`, captured partial output, and a timeout-specific `RuntimeError` when required commands fail.

### P1 / Must-Have Hardening Before Wider Rollout

1. **Make artifact schemas stricter.**
   - Current metric/exactness loaders accept flexible JSON shapes, which is helpful for prototypes but can hide malformed results.
   - Recommendation: add explicit metric/exactness JSON schema validation and persist schema version in every runtime artifact.

2. **Separate candidate generation from candidate evaluation.**
   - Current `run_contract` reuses baseline command fields for candidate runs, which is simple but ambiguous for fully autonomous mutation loops.
   - Recommendation: add optional `candidate_run_command`, `candidate_profile_command`, and `candidate_exactness_command` fields, with fallback to baseline fields for backward compatibility.

3. **Introduce workspace locking.**
   - Concurrent CLI invocations can write the same `.auto-profiling/` files.
   - Recommendation: add a lock file around mutating subcommands (`baseline`, `candidate`, `evaluate`, `loop`, `autopilot`).

4. **Record command retry attempts individually.**
   - Current retry only returns the final successful result or raises the final error.
   - Recommendation: persist failed attempts in `experiment_log.jsonl` for flake triage.

### P2 / Operability Improvements

1. **Add CI entrypoints.**
   - Recommended checks: `python -m pytest -q`, `python -m unittest discover -s auto-profiling/tests`, and packaging smoke tests for `auto-profiling`.

2. **Add redaction policy.**
   - Runtime logs capture command stdout/stderr; production usage may expose API keys, model paths, tenant IDs, or private prompts.
   - Recommendation: add redact patterns configurable from `aim.md`.

3. **Add a dry-run command.**
   - A `validate` or `doctor` subcommand should validate aim fields, route resolution, target repo status, output paths, and command timeout/retry policy without running benchmarks.

## Implemented in This Change

- Added timeout-aware shell execution in `auto-profiling/runner.py`.
- Added aim-driven command timeout resolution with `command_timeout_seconds` overriding `max_runtime_per_experiment`.
- Added regression tests proving timeout metadata is reported and required commands raise timeout-specific errors.
- Updated README guidance so users know how to bound long-running benchmark/profile/exactness commands.

## Recommended Production Acceptance Gate

Before calling the system production-ready for customer repositories, require:

1. `aim.md` validates against schema and declares `max_runtime_per_experiment`.
2. `init` and `collect-env` complete without unhandled exceptions.
3. `baseline` produces metric + exactness artifacts and exactness passes.
4. At least one `candidate` or `autopilot` run rejects a known-bad candidate and reverts when `require_revert_on_failure: true`.
5. `status` and `handoff` are sufficient for a new operator to resume without reading terminal scrollback.
