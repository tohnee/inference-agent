# Inference Agent

`inference-agent` is a repository for exactness-first inference optimization.

It combines reusable optimization knowledge with a runnable optimization harness, so you can move from "ideas and playbooks" to "baseline, evidence, keep-or-revert decisions, and resumable iterations" inside one repository.

This repository is built for:

- end-to-end optimization of small-model or non-LLM inference pipelines
- LLM serving optimization across SGLang, vLLM, TensorRT-LLM, Triton, and custom PyTorch stacks
- CUDA / CUTLASS / Triton kernel-level investigation once the bottleneck is narrowed down
- teams that want a repeatable, auditable, exactness-first optimization workflow

## Core Principles

The default operating model is:

- exactness first, performance second
- establish a trusted baseline before optimizing
- run one bounded experiment at a time
- keep changes only with exactness and metric evidence
- make optimization resumable and handoff-friendly

If your workflow is "change a lot of things and hope the system gets faster," this repository is intentionally not designed for that.

## Repository Layout

The repository has four main parts:

- `e2e-inference-opt-skill/`
  - knowledge base for end-to-end inference pipelines
- `llm-serving-opt-skill/`
  - knowledge base for service-level LLM inference optimization
- `cuda-kernel-opt-skill/`
  - knowledge base for kernel, operator, and low-level CUDA optimization
- `auto-profiling/`
  - minimal runtime for executing baselines, candidate experiments, evaluation, and handoff

In short:

- the three `*-opt-skill` modules tell you what to optimize and which lane to use
- `auto-profiling` turns that lane into a runnable optimization loop

## Three Scenarios

### 1. End-to-End Inference

Use this when:

- the workload is not primarily an LLM serving system
- the bottleneck spans preprocess, forward, postprocess, IO, caching, or deployment
- you need system-level optimization before kernel-level work

Start here:

- [e2e-inference-opt-skill](e2e-inference-opt-skill/SKILL.md)
- [aim.e2e.md](auto-profiling/aim.e2e.md)

### 2. LLM Serving

Use this when:

- you care about TTFT, TPOT, ITL, req/s, or tok/s
- you are optimizing SGLang, vLLM, TensorRT-LLM, Triton, or a custom serving stack
- you need scheduler, KV cache, prefix cache, benchmark, or deployment guidance

Start here:

- [llm-serving-opt-skill](llm-serving-opt-skill/SKILL.md)
- [aim.llm-serving.md](auto-profiling/aim.llm-serving.md)

### 3. CUDA / Kernel Optimization

Use this when:

- profiling already shows the bottleneck is inside an operator or kernel
- you need NCU, kernel microbenchmarking, custom kernel workflow, or CUDA crash triage
- you are drilling down from serving-level symptoms into kernel-level root causes

Start here:

- [cuda-kernel-opt-skill](cuda-kernel-opt-skill/SKILL.md)
- [aim.cuda-kernel.md](auto-profiling/aim.cuda-kernel.md)

## Quick Start

The recommended first-use flow is:

1. choose the scenario: `e2e-inference`, `llm-serving`, or `cuda-kernel`
2. read the matching top-level `SKILL.md`
3. open the corresponding `aim.*.md` under `auto-profiling/`
4. fill in your target repository, baseline command, exactness command, and metric output paths
5. run `init`, `collect-env`, and `baseline`
6. inspect the generated contract, evaluator report, and handoff artifacts
7. run one bounded candidate experiment

### 5-Minute Example

```bash
cd auto-profiling

uv run runner.py init --aim aim.llm-serving.md
uv run runner.py collect-env --aim aim.llm-serving.md
uv run runner.py baseline --aim aim.llm-serving.md
uv run runner.py status --aim aim.llm-serving.md
uv run runner.py candidate --aim aim.llm-serving.md --label exp-001
```

If `uv` is not available, run the same commands with `python3 runner.py ...`.

## What `auto-profiling` Does

`auto-profiling/` is the executable part of the repository.

Key files:

- [runner.py](auto-profiling/runner.py)
- [scorer.py](auto-profiling/scorer.py)
- [README.md](auto-profiling/README.md)
- [aim.md](auto-profiling/aim.md)

The runtime writes `.auto-profiling/` artifacts into the target project so work can be resumed and audited:

- `current_contract.md`
- `evaluator_report.md`
- `next_handoff.md`
- `session_state.json`
- `experiment_log.md`
- `experiment_log.jsonl`

This gives you:

- resumable optimization sessions
- explicit keep-or-revert decisions
- disk-backed handoff instead of relying on chat context alone

## Module Responsibilities

### `e2e-inference-opt-skill`

Focuses on:

- baseline
- profiling
- roofline
- memory and IO
- parallelism
- pipeline overlap
- deployment

### `llm-serving-opt-skill`

Focuses on:

- serving baselines
- benchmark workflow
- profile analysis
- KV cache and scheduler behavior
- service deployment

### `cuda-kernel-opt-skill`

Focuses on:

- CUDA crash debugging
- profile triage
- backend selection
- custom kernel workflow
- vendored `cuda-optimized-skill`

## Environment Strategy

The current runtime defaults are:

- prefer `uv` for project installation
- fall back to `pip` when `uv` is unavailable
- allow running inside an existing `conda` environment via the active `python3`
- allow explicit environment activation through `python_env_command` in the aim file

The safest pattern is usually:

1. activate the Python environment you actually want to use
2. then run `runner.py` from that environment

## Verification

From the repository root:

```bash
python3 -m unittest tests/test_skill_catalog.py -v
```

From `auto-profiling/`:

```bash
cd auto-profiling
python3 -m unittest tests/test_runtime.py -v
```

## Common Mistakes

- optimizing before establishing a trusted baseline
- treating output drift as a normal cost of speedup
- changing multiple variables in a single experiment
- failing to produce structured metric and exactness JSON
- mixing up end-to-end, serving, and kernel-level problem scopes

## Read More

- [Chinese root README](README.md)
- [auto-profiling/README.md](auto-profiling/README.md)
- [e2e-inference-opt-skill/SKILL.md](e2e-inference-opt-skill/SKILL.md)
- [llm-serving-opt-skill/SKILL.md](llm-serving-opt-skill/SKILL.md)
- [cuda-kernel-opt-skill/SKILL.md](cuda-kernel-opt-skill/SKILL.md)

If you only remember one sentence:

> Choose the right scenario, write the right `aim`, and let `auto-profiling` run a bounded optimization loop under an exactness contract.
