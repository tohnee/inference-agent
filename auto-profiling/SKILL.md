---
name: "auto-profiling"
description: "Use when a user wants to drive inference optimization from one aim.md while preserving exact outputs."
---

# Auto-Profiling

Auto-Profiling is the orchestration skill for aim-driven inference optimization.

## Operating model

The human edits one file: `aim.md`.

The agent/runtime must:

1. validate `aim.md` against the required contract,
2. establish a trusted baseline,
3. collect metric, exactness, and profile evidence,
4. route the work to the correct skill lane,
5. plan exactly one bounded candidate experiment,
6. evaluate exactness before performance,
7. keep only candidates with correctness and metric evidence,
8. write resumable state and handoff artifacts.

## Underlying skill libraries

Route by `scenario`:

- `e2e-inference` -> `../e2e-inference-opt-skill/SKILL.md`
- `llm-serving` -> `../llm-serving-opt-skill/SKILL.md`
- `cuda-kernel` -> `../cuda-kernel-opt-skill/SKILL.md`
- `operator-kernel` -> `../cuda-kernel-opt-skill/SKILL.md` plus operator synthesis subskills

The route table is data-driven in `skill_routes.json`.

## Required aim fields

The stable aim API is documented by `aim_schema.json`. The key required fields are:

- `scenario`
- `target_repo_path`
- `baseline_run_command`
- `baseline_profile_command`
- `metric_output_path`
- `exactness_output_path`
- `exactness_check_command`
- `target_metric_name`
- `target_metric_direction`
- `allowed_mutations`
- `blocked_by_default`

## Runtime artifacts

The source templates live in `templates/state/`.
The target project receives generated runtime state under `<target_repo>/.auto-profiling/`, including:

- `current_contract.md`
- `next_candidate.md`
- `next_candidate.json`
- `evaluator_report.md`
- `next_handoff.md`
- `experiment_log.md`
- `experiment_log.jsonl`
- `session_state.json`

## Exactness rule

Default to `exact-parity`. Reject speedups that change output correctness, cache semantics, request isolation, or declared algorithm equivalence.
