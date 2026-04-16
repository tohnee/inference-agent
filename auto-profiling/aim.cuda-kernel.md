# Auto-Profiling Aim

Edit this file only for **CUDA, CUTLASS, or Triton kernel-level optimization**.

## 1. Mission

- scenario: cuda-kernel
- project_name:
- primary_goal:
- optimize_for: latency | throughput | memory
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
  - kernel benchmark harness
  - tile shape and launch configuration
  - memory movement optimization
  - NCU-guided operator tuning
  - CUDA / CUTLASS / Triton implementation details
- blocked_by_default:
  - output-changing algorithm updates
  - approximate math without approval
  - silent precision drift

## 7. Notes

- recommended module: `cuda-kernel-opt-skill`
- use this template after the bottleneck has already been narrowed to an operator or kernel
