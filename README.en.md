# Inference Agent

`inference-agent` is organized around one product flow: **the user submits one `aim.md`, `auto-profiling` drives the optimization loop, and the runtime routes profile evidence to one of three underlying skill libraries.**

## User entry point

The recommended user-facing API is:

- `auto-profiling/aim.md`

The aim file declares the target repository, baseline/profile/exactness commands, metric direction, allowed mutation surface, and blocked changes. The runtime validates the aim before execution.

## Architecture

```text
inference-agent/
├── auto-profiling/              # aim -> baseline -> profile -> candidate plan -> evaluate -> handoff
├── e2e-inference-opt-skill/     # end-to-end inference optimization knowledge
├── llm-serving-opt-skill/       # LLM serving optimization knowledge
├── cuda-kernel-opt-skill/       # CUDA/operator/kernel optimization knowledge
├── docs/                        # architecture notes and historical archive
└── tests/                       # runtime, schema, catalog, and tool tests
```

`auto-profiling` is the optimization driver. The three `*-opt-skill` directories are the underlying knowledge libraries.

## Quick start

Generate or edit the single aim file:

```bash
python3 auto-profiling/bootstrap_aim.py \
  --mode llm-serving \
  --profile vllm \
  --project-name demo-vllm \
  --target-repo-path /path/to/target-repo \
  --output auto-profiling/aim.md
```

CUDA/kernel optimization uses the same entry point:

```bash
python3 auto-profiling/bootstrap_aim.py \
  --mode cuda-kernel \
  --profile triton \
  --project-name demo-kernel \
  --target-repo-path /path/to/target-repo \
  --output auto-profiling/aim.md
```

Run the loop:

```bash
cd auto-profiling
python3 runner.py init --aim aim.md
python3 runner.py doctor --aim aim.md
python3 runner.py collect-env --aim aim.md
python3 runner.py baseline --aim aim.md
python3 runner.py autopilot --aim aim.md --iterations 3
python3 runner.py status --aim aim.md
```

## User-facing preflight

Before running expensive benchmark/profile commands, run:

```bash
cd auto-profiling
python3 runner.py doctor --aim aim.md
```

`doctor` writes `<target_repo>/.auto-profiling/doctor_report.md` and `doctor_report.json`. It checks aim completeness, target repository access, git/revert safety, required commands, metric/exactness contracts, output artifact paths, and mutation scope. It returns 0 by default for interactive review; add `--strict` in CI when failing checks should produce a non-zero exit.

## Stable aim API

`auto-profiling/aim_schema.json` defines the required aim fields. Scenario routing lives in `auto-profiling/skill_routes.json`, so new optimization lanes can be added without hard-coding the runtime.

## Runtime state boundary

- `auto-profiling/templates/state/` is the source template shipped by this repository.
- `<target_repo>/.auto-profiling/` is generated runtime state for a target project.
- `docs/archive/` contains historical notes and is not part of the normal user workflow.

## Production execution guardrails

- `max_runtime_per_experiment` is used as the default timeout for install, warmup, baseline, exactness, and profile commands so external tools cannot hang a session forever.
- Add `command_timeout_seconds` to `aim.md` when a stricter per-command timeout is needed; it takes precedence over `max_runtime_per_experiment`.
- Timed-out commands are reported with `exit_code: 124`, `timed_out: true`, and a timeout-specific required-command error.
- See `docs/plans/2026-06-03-production-readiness-review.md` for the current hardening review and follow-up checklist.
