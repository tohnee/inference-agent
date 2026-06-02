# .auto-profiling Template

This directory is the initialization template copied into the target repository root.

Expected runtime artifacts:

- `experiment_log.md`
- `experiment_log.jsonl`
- `baseline_snapshot.json`
- `best_result.json`
- `session_state.json`
- `task_plan.md`
- `findings.md`
- `progress.md`
- `worklog.md`
- `current_contract.md`
- `evaluator_report.md`
- `next_handoff.md`
- `next_candidate.md`
- `next_candidate.json`

This directory is source template state only. Runtime state is generated under `<target_repo>/.auto-profiling/`.
The runner creates missing files automatically and keeps them git-friendly.
