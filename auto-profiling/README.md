# Auto-Profiling

Auto-Profiling is a long-running inference optimization harness built on top of the local skills in this workspace.

It is designed for the following workflow:

- the human provides environment, baseline code, and verification commands
- the human edits only `aim.md`
- the runtime initializes a git-friendly workspace under `.auto-profiling/`
- the agent runs bounded experiments, evaluates them, keeps or rejects them, and writes handoff artifacts for the next iteration

The default philosophy is:

- exact-parity first
- performance second
- one bounded experiment at a time
- skeptical evaluation instead of self-congratulation

## What This Project Contains

### Runtime

- `runner.py`: executes the `aim.md` contract
- `scorer.py`: compares reference and candidate results
- `pyproject.toml`: enables `uv` execution
- `log_schema.json`: structured log schema

The runtime also performs environment detection:

- prefers `zsh`
- falls back to `bash` when `zsh` is not available
- falls back to the system shell if neither is available
- prefers `uv` for project installs when it exists
- falls back to `pip` when `uv` is unavailable
- collects richer runtime metadata such as Python version, platform, machine, tool paths, package versions, and optional `vllm collect-env` evidence

### Human Control Surface

- `aim.md`: the only required human-edited runtime contract
- `aim.zh-CN.md`: Chinese version of the same template
- `aim.e2e.md`: template for small-model and non-LLM end-to-end optimization
- `aim.llm-serving.md`: template for LLM serving optimization
- `aim.cuda-kernel.md`: template for CUDA/CUTLASS/Triton kernel optimization

### Harness State

The runtime creates and updates `.auto-profiling/` in the target repository:

- `experiment_log.md`
- `experiment_log.jsonl`
- `baseline_snapshot.json`
- `best_result.json`
- `session_state.json`
- `current_contract.md`
- `evaluator_report.md`
- `next_handoff.md`
- `task_plan.md`
- `findings.md`
- `progress.md`
- `worklog.md`

### Reference Guides

- `references/01_operating_model.md`
- `references/02_experiment_loop.md`
- `references/03_mutation_lanes.md`
- `references/04_exactness_gate.md`
- `references/05_artifacts_and_scoring.md`
- `references/06_harness_patterns.md`

## Core Architecture

Auto-Profiling uses a planner-generator-evaluator style harness, but keeps the runtime minimal.

### Planner Role

The planner role is represented by:

- `aim.md`
- `.auto-profiling/current_contract.md`

This is where the current objective, exactness mode, selected lane, and experiment boundary are written down.

The runtime now makes the lane explicit in both machine-readable and human-readable artifacts:

- current `scenario`
- target optimization module
- recommended skill route
- recommended aim template

### Generator Role

The generator role is the actual bounded optimization attempt:

- run baseline
- apply one candidate change
- rerun exactness checks
- rerun measurement

The runtime itself does not hardcode a mutation engine. It provides the loop, state, and artifacts so Claude can mutate code while staying bounded.

## Three Scenarios

Auto-Profiling now acts as the common entrypoint for three optimization families:

- `e2e-inference`: use [e2e-inference-opt-skill](file:///Users/tc/Downloads/推理优化skills/e2e-inference-opt-skill/SKILL.md) for small-model and non-LLM end-to-end chains
- `llm-serving`: use [llm-serving-opt-skill](file:///Users/tc/Downloads/推理优化skills/llm-serving-opt-skill/SKILL.md) for serving systems such as SGLang, vLLM, TensorRT-LLM, Triton, and custom PyTorch servers
- `cuda-kernel`: use [cuda-kernel-opt-skill](file:///Users/tc/Downloads/推理优化skills/cuda-kernel-opt-skill/SKILL.md) after the bottleneck has been narrowed to an operator or kernel

### Evaluator Role

The evaluator role is separated from generation through:

- `scorer.py`
- `.auto-profiling/evaluator_report.md`

The evaluator decides whether a candidate should be kept based on:

- exactness
- metric improvement
- scope compliance
- stability
- reproducibility

This prevents the system from blindly praising its own changes.

### Handoff Role

Long-running work is stabilized through:

- `.auto-profiling/session_state.json`
- `.auto-profiling/next_handoff.md`

These files make it possible to resume in a fresh session without depending on a huge conversation history.

## Exactness Model

### 1. Exact-Parity Mode

This is the default mode.

Use it when:

- logic is unchanged
- algorithm is unchanged
- device class is effectively the same
- numerical contract is unchanged

Rule:

- any output mismatch fails the experiment

### 2. Bounded-Tolerance Mode

Use it only when all are true:

- logic remains equivalent
- algorithm remains equivalent
- the difference comes from declared cross-device execution or declared precision transitions
- `abs_tolerance` and `rel_tolerance` are explicitly set in `aim.md`

Example:

- CPU reference vs GPU candidate
- FP32 reference vs BF16 candidate

Rule:

- exceed either tolerance and the experiment fails
- if logic or algorithm equivalence fails, tolerance mode does not save the run

## Runtime Commands

Run from this directory with `uv` when available.

If `uv` is not installed, use the Python entrypoint directly:

```bash
python3 runner.py status --aim aim.md
```

For target project dependency installation, the runtime now auto-detects:

- `uv sync` when `uv` exists and the target project uses `pyproject.toml`
- `python -m pip install -r requirements.txt` when the target project uses `requirements.txt`
- `python -m pip install -e .` when `uv` is unavailable but the target project is installable as a package

### Initialize

```bash
uv run runner.py init --aim aim.md
```

Creates or refreshes the runtime workspace and handoff scaffolding.

### Capture Baseline

```bash
uv run runner.py baseline --aim aim.md
```

Runs the trusted path, records exactness and performance, and stores the baseline snapshot.

### Evaluate A Candidate

```bash
uv run runner.py candidate --aim aim.md --label exp-001
```

Runs a bounded candidate experiment and promotes it if it beats the current reference under the exactness contract.

### Evaluate Without Promotion

```bash
uv run runner.py evaluate --aim aim.md --label eval-001
```

Runs the same evaluation path but does not promote the candidate to best-known result.

### Long-Running Loop Step

```bash
uv run runner.py loop --aim aim.md --label loop-001
```

Runs one resumable harness step:

- if no baseline exists, it creates one
- otherwise it runs a candidate step against the current best reference

### Read Current Status

```bash
uv run runner.py status --aim aim.md
```

Shows:

- baseline availability
- best result availability
- current session state
- current scenario, target module, and recommended skill route
- detected shell and package manager
- workspace paths

### Collect Environment

```bash
uv run runner.py collect-env --aim aim.llm-serving.md
```

Collects a stronger runtime fingerprint before optimization begins, including:

- shell and package manager fallback results
- Python, platform, and machine metadata
- detected tool paths such as `git`, `nvidia-smi`, `nvcc`, `uv`, `pip`, and `vllm`
- lightweight command diagnostics
- installed package versions for common serving and kernel stacks
- optional `vllm collect-env` summary when `vllm` is available

### Produce A Handoff

```bash
uv run runner.py handoff --aim aim.md
```

Writes and prints the next handoff summary for a resumed session.

## How To Fill `aim.md`

The minimum required fields are:

- `target_repo_path`
- `baseline_run_command`
- `baseline_profile_command`
- `metric_output_path`
- `exactness_output_path`
- `exactness_check_command`
- `target_metric_name`
- `target_metric_direction`

Strongly recommended fields:

- `allowed_mutations`
- `blocked_by_default`
- `known_bottlenecks`
- `suspected_safe_lanes`
- `max_iterations_per_session`
- `max_runtime_per_experiment`

Environment-related notes:

- leave `install_command` empty to let the runtime auto-detect `uv` or `pip`
- set `install_command: auto` if you want to be explicit but still use auto-detection
- use `python_env_command` only when you need to activate a custom environment manually

## Expected Output Files

Your commands should write machine-readable JSON for:

### Metric payload

Example:

```json
{
  "metrics": {
    "p95_ms": 8.4,
    "throughput": 120.0
  }
}
```

### Exactness payload

Exact-parity example:

```json
{
  "passed": true,
  "mismatch_count": 0
}
```

Bounded-tolerance example:

```json
{
  "passed": false,
  "logic_equivalent": true,
  "algorithm_equivalent": true,
  "mismatch_count": 3,
  "max_abs_error": 0.000004,
  "max_rel_error": 0.000003
}
```

The runtime computes the final exactness decision based on the mode declared in `aim.md`.

## Relationship To The Three Modules

`auto-profiling` is the execution entrypoint, while the three sibling skill modules provide optimization knowledge.

### `e2e-inference-opt-skill`

Path:

- [e2e-inference-opt-skill](file:///Users/tc/Downloads/推理优化skills/e2e-inference-opt-skill)

Role:

- the optimization knowledge base for small-model and non-LLM end-to-end chains
- the routing system for preprocess, forward, postprocess, cache, overlap, and deployment

### `llm-serving-opt-skill`

Path:

- [llm-serving-opt-skill](file:///Users/tc/Downloads/推理优化skills/llm-serving-opt-skill)

Role:

- the optimization knowledge base for LLM serving systems
- the routing system for TTFT, TPOT, ITL, serving benchmark, service deployment, and serving-specific bottlenecks

### `cuda-kernel-opt-skill`

Path:

- [cuda-kernel-opt-skill](file:///Users/tc/Downloads/推理优化skills/cuda-kernel-opt-skill)

Role:

- the optimization knowledge base for CUDA/CUTLASS/Triton kernel work
- the routing system for correctness, benchmark, Nsight Compute, custom kernel workflow, and strategy-memory loops

### `auto-profiling`

Path:

- [auto-profiling](file:///Users/tc/Downloads/推理优化skills/auto-profiling)

Role:

- the orchestration harness
- the runtime loop
- the artifact manager
- the keep-or-revert execution engine

It tells you **how to run repeated optimization iterations safely and continuously**.

### Practical Relationship

Use them together like this:

1. choose the scenario: `e2e-inference`, `llm-serving`, or `cuda-kernel`
2. use the matching module to understand the bottleneck and select a safe lane
3. encode the project contract in the matching aim file
4. run `auto-profiling` to execute bounded experiments and preserve state
5. use the matching module references when choosing the next mutation lane

In short:

- `e2e-inference-opt-skill` / `llm-serving-opt-skill` / `cuda-kernel-opt-skill` = optimization brains
- `auto-profiling` = optimization harness

## Recommended Workflow

1. prepare a runnable baseline repository
2. fill one of `aim.e2e.md`, `aim.llm-serving.md`, or `aim.cuda-kernel.md`
3. run `uv run runner.py init --aim aim.llm-serving.md`
4. run `uv run runner.py baseline --aim aim.llm-serving.md`
5. inspect `.auto-profiling/current_contract.md`
6. run `uv run runner.py collect-env --aim aim.llm-serving.md`
7. run one bounded candidate step
8. read `.auto-profiling/evaluator_report.md`
9. use `.auto-profiling/next_handoff.md` to continue the next session

## Common Mistakes

- treating tolerance mode as a shortcut for logic changes
- changing multiple variables in one experiment
- failing to provide JSON metric and exactness outputs
- skipping baseline and trying to optimize immediately
- forgetting that `auto-profiling` still needs the matching scenario module for optimization strategy guidance

## Suggested Next Improvements

If you continue evolving this project, the most natural next steps are:

- automatic multi-round loop execution
- automatic contract generation from profiler evidence
- richer evaluator rubric
- lane-specific helpers that directly import the logic of the three scenario modules
