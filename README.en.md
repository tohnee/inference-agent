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
python3 runner.py collect-env --aim aim.md
python3 runner.py baseline --aim aim.md
python3 runner.py autopilot --aim aim.md --iterations 3
python3 runner.py status --aim aim.md
```

## Stable aim API

`auto-profiling/aim_schema.json` defines the required aim fields. Scenario routing lives in `auto-profiling/skill_routes.json`, so new optimization lanes can be added without hard-coding the runtime.

## Runtime state boundary

- `auto-profiling/templates/state/` is the source template shipped by this repository.
- `<target_repo>/.auto-profiling/` is generated runtime state for a target project.
- `docs/archive/` contains historical notes and is not part of the normal user workflow.
