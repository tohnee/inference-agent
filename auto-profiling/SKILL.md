---
name: "auto-profiling"
description: "Use when given an environment and baseline inference code and asked to autonomously iterate on performance from a single aim.md while preserving exact outputs."
---

# Auto-Profiling

## Overview

This skill turns inference optimization into an autonomous experiment loop.

The human edits one aim file.
The agent reads the target environment and baseline code, proposes one experiment at a time, verifies exact output parity, measures performance, keeps or rejects the change, and repeats.

Supported scenarios:

- end-to-end small-model or non-LLM inference chains
- LLM serving optimization
- CUDA / CUTLASS / Triton kernel-level optimization

This is the profiling and inference-optimization counterpart of a lightweight autonomous research program:

- one human-controlled objective file
- one fixed experiment loop
- one trusted baseline
- one explicit keep-or-revert rule

The project now includes a minimal runtime:

- [runner.py](file:///Users/tc/Downloads/推理优化skills/auto-profiling/runner.py) executes the `aim.md` contract
- [scorer.py](file:///Users/tc/Downloads/推理优化skills/auto-profiling/scorer.py) compares baseline and candidate outcomes
- [log_schema.json](file:///Users/tc/Downloads/推理优化skills/auto-profiling/log_schema.json) defines the structured log entry shape
- [pyproject.toml](file:///Users/tc/Downloads/推理优化skills/auto-profiling/pyproject.toml) enables `uv`-based execution
- [.auto-profiling](file:///Users/tc/Downloads/推理优化skills/auto-profiling/.auto-profiling) provides git-friendly initialization templates

The runtime also behaves like a long-running harness:

- planner-style contract files keep each bounded experiment explicit
- evaluator-style reports independently judge keep or revert
- handoff files make context resets and fresh-session resumes safer
- `loop` gives one resume command for bounded autonomous iteration
- `collect-env` performs stronger environment collection before optimization begins

## First Principle

Auto-profiling runs in **exact-parity mode** by default.

That means:

- no experiment is successful if outputs change
- no experiment is successful if cache semantics change incorrectly
- no experiment is successful if request routing or concurrency introduces wrong results
- performance gains are secondary to exact output preservation

Only an explicit human override in `aim.md` can allow non-zero drift.

There is one bounded exception path:

- if logic and algorithm remain equivalent
- and the difference comes from declared cross-device execution or declared precision transitions
- then `bounded-tolerance` mode may be used with explicit `abs_tolerance` and `rel_tolerance`

This is still a hard contract, not a soft suggestion.

## What The Human Edits

The only required human-edited file is one selected aim file:

- [aim.e2e.md](file:///Users/tc/Downloads/推理优化skills/auto-profiling/aim.e2e.md)
- [aim.llm-serving.md](file:///Users/tc/Downloads/推理优化skills/auto-profiling/aim.llm-serving.md)
- [aim.cuda-kernel.md](file:///Users/tc/Downloads/推理优化skills/auto-profiling/aim.cuda-kernel.md)
- or the generic [aim.md](file:///Users/tc/Downloads/推理优化skills/auto-profiling/aim.md) if you want to author the contract from scratch

It defines:

- target repository or code path
- environment and runtime commands
- trusted baseline command
- exactness contract
- success metric
- allowed mutation surface
- forbidden changes
- experiment budget
- artifact paths and progress-document paths

## What The Agent Does

For each iteration, the agent must:

1. read `aim.md`
2. validate environment and baseline command
3. run trusted baseline
4. collect parity evidence and performance evidence
5. choose one optimization lane
6. make exactly one bounded change
7. rerun parity gate first
8. rerun measurement
9. keep or revert
10. append experiment results to the log

## Required Background

This skill is the orchestration layer.

Execution knowledge comes from:

- [e2e-inference-opt-skill](file:///Users/tc/Downloads/推理优化skills/e2e-inference-opt-skill/SKILL.md)
- [llm-serving-opt-skill](file:///Users/tc/Downloads/推理优化skills/llm-serving-opt-skill/SKILL.md)
- [cuda-kernel-opt-skill](file:///Users/tc/Downloads/推理优化skills/cuda-kernel-opt-skill/SKILL.md)
- [01_operating_model.md](file:///Users/tc/Downloads/推理优化skills/auto-profiling/references/01_operating_model.md)
- [02_experiment_loop.md](file:///Users/tc/Downloads/推理优化skills/auto-profiling/references/02_experiment_loop.md)
- [03_mutation_lanes.md](file:///Users/tc/Downloads/推理优化skills/auto-profiling/references/03_mutation_lanes.md)
- [04_exactness_gate.md](file:///Users/tc/Downloads/推理优化skills/auto-profiling/references/04_exactness_gate.md)
- [05_artifacts_and_scoring.md](file:///Users/tc/Downloads/推理优化skills/auto-profiling/references/05_artifacts_and_scoring.md)

## When To Use

Use when:

- the user has baseline inference code and wants autonomous iteration
- the optimization goal is performance, not model quality research
- the user wants to drive the system from one human-authored objective file
- the user wants repeatable profiling, mutation, verification, and experiment logging

Do not use when:

- the main goal is training research
- output drift is acceptable but not yet specified
- there is no runnable baseline command
- the environment cannot execute verification and profiling commands

## Quick Start

### Human step

Pick one aim file and edit it:

- [aim.e2e.md](file:///Users/tc/Downloads/推理优化skills/auto-profiling/aim.e2e.md)
- [aim.llm-serving.md](file:///Users/tc/Downloads/推理优化skills/auto-profiling/aim.llm-serving.md)
- [aim.cuda-kernel.md](file:///Users/tc/Downloads/推理优化skills/auto-profiling/aim.cuda-kernel.md)

Then fill:

- repository path
- setup and run commands
- golden-set and exactness rules
- metric to optimize
- allowed mutation surfaces
- metric and exactness output paths

Initialize and run with `uv`:

- `uv run runner.py init --aim aim.llm-serving.md`
- `uv run runner.py collect-env --aim aim.llm-serving.md`
- `uv run runner.py baseline --aim aim.llm-serving.md`
- `uv run runner.py candidate --aim aim.llm-serving.md --label exp-001`
- `uv run runner.py evaluate --aim aim.llm-serving.md --label eval-001`
- `uv run runner.py handoff --aim aim.llm-serving.md`
- `uv run runner.py loop --aim aim.llm-serving.md --label loop-001`
- `uv run runner.py status --aim aim.llm-serving.md`

### Agent step

Follow this route:

- scenario `e2e-inference` -> [e2e-inference-opt-skill](file:///Users/tc/Downloads/推理优化skills/e2e-inference-opt-skill/SKILL.md)
- scenario `llm-serving` -> [llm-serving-opt-skill](file:///Users/tc/Downloads/推理优化skills/llm-serving-opt-skill/SKILL.md)
- scenario `cuda-kernel` -> [cuda-kernel-opt-skill](file:///Users/tc/Downloads/推理优化skills/cuda-kernel-opt-skill/SKILL.md)
- operating model → [01_operating_model.md](file:///Users/tc/Downloads/推理优化skills/auto-profiling/references/01_operating_model.md)
- experiment loop → [02_experiment_loop.md](file:///Users/tc/Downloads/推理优化skills/auto-profiling/references/02_experiment_loop.md)
- mutation choice → [03_mutation_lanes.md](file:///Users/tc/Downloads/推理优化skills/auto-profiling/references/03_mutation_lanes.md)
- exactness gate → [04_exactness_gate.md](file:///Users/tc/Downloads/推理优化skills/auto-profiling/references/04_exactness_gate.md)
- scoring and artifacts → [05_artifacts_and_scoring.md](file:///Users/tc/Downloads/推理优化skills/auto-profiling/references/05_artifacts_and_scoring.md)
- long-running harness patterns → [06_harness_patterns.md](file:///Users/tc/Downloads/推理优化skills/auto-profiling/references/06_harness_patterns.md)

## Default Keep Rule

Keep an experiment only if all are true:

- exactness gate passes
- target metric improves
- no new correctness, stability, or memory regression appears
- the change stays within the allowed mutation surface in `aim.md`

Otherwise revert it.

## Runtime Artifacts

The runtime maintains:

- `.auto-profiling/experiment_log.md`
- `.auto-profiling/experiment_log.jsonl`
- `.auto-profiling/baseline_snapshot.json`
- `.auto-profiling/best_result.json`
- `.auto-profiling/session_state.json`
- `.auto-profiling/task_plan.md`
- `.auto-profiling/findings.md`
- `.auto-profiling/progress.md`
- `.auto-profiling/worklog.md`
- `.auto-profiling/current_contract.md`
- `.auto-profiling/evaluator_report.md`
- `.auto-profiling/next_handoff.md`

The runtime now writes lane-aware metadata into key artifacts:

- `status` exposes `scenario`, `target_module`, and `recommended_skill_route`
- `current_contract.md` records the active lane and recommended module route
- `next_handoff.md` preserves the same lane context for resumed sessions

## Common Mistakes

- editing code before validating the baseline command
- changing multiple variables in one experiment
- profiling before defining exactness
- accepting a speedup without rerunning parity checks
- allowing mutation outside the declared scope

## Expected Output

When using this skill, the agent should hand back:

- current best experiment
- exactness status
- metric trend versus baseline
- rejected experiments and why they failed
- recommended next experiment
