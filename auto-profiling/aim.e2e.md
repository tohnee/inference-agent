# Auto-Profiling Aim

Edit this file only for **small-model or non-LLM end-to-end inference optimization**.

## 1. Mission

- scenario: e2e-inference
- project_name:
- primary_goal:
- optimize_for: latency | throughput | cost_per_request | memory | startup_time
- target_metric_name:
- target_metric_direction: lower_is_better | higher_is_better
- target_sla:

## 2. Scope

- target_repo_path:
- target_entrypoints:
- baseline_files_allowed_to_change:
- files_never_touch:

## 3. Environment

- os:
- hardware:
- accelerator:
- python_env_command:
- git_required: true
- install_command:
- warmup_command:

## 4. Baseline Execution

- baseline_setup_command:
- baseline_run_command:
- baseline_profile_command:
- metric_output_path:
- exactness_output_path:

## 5. Exactness Contract

- exactness_mode: exact-parity | bounded-tolerance
- abs_tolerance: 0.0
- rel_tolerance: 0.0
- require_logic_equivalence: true
- require_algorithm_equivalence: true
- reference_path_description:
- golden_input_location:
- golden_output_location:
- exactness_check_command:

## 6. Allowed Mutation Surface

- allowed_mutations:
  - preprocess optimization
  - postprocess optimization
  - copy reduction
  - batching and worker model
  - pipeline overlap
  - deployment configuration
- blocked_by_default:
  - output-changing algorithm updates
  - precision drift without approval
  - approximate kernels

## 7. Notes

- recommended module: `e2e-inference-opt-skill`
- use this template for CV, ranking, retrieval, rerank, tabular, and multi-stage inference chains
