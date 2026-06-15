# Auto-Profiling Aim

Edit this file only. The agent treats it as the operating contract.

## 1. Mission

- scenario: e2e-inference | llm-serving | cuda-kernel
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

Notes:

- `scenario` selects the primary skill family and recommended aim lane
- runtime status and handoff artifacts will echo `scenario`, `target_module`, and `recommended_skill_route`
- leave `install_command` empty or set it to `auto` to let the runtime choose `uv` or `pip`
- shell execution prefers `zsh`, falls back to `bash`, then falls back to the system shell

## 4. Baseline Execution

- baseline_setup_command:
- baseline_run_command:
- baseline_profile_command:
- profile_output_path:
- metric_output_path:
- exactness_output_path:

## 5. Exactness Contract

- exactness_mode: exact-parity | bounded-tolerance
- abs_tolerance: 0.0
- rel_tolerance: 0.0
- require_logic_equivalence: true
- require_algorithm_equivalence: true
- allowed_precision_transitions:
- allowed_environment_differences:
- reference_path_description:
- golden_input_location:
- golden_output_location:
- exactness_check_command:
- deterministic_requirements:
- cache_semantics_requirements:
- request_isolation_requirements:

Notes:

- `exact-parity` is the default red line
- `bounded-tolerance` is only for declared cross-device or precision transitions with equivalent logic and algorithm

## 6. Reasoning Quality Contract

- reasoning_task_type: chat_reasoning | code_reasoning | rag_synthesis | tool_agent | long_context | batch_eval
- reasoning_quality_metric: exact_match | semantic_rubric | judge_score | tool_trace_parity | citation_fidelity
- reasoning_quality_threshold:
- golden_dataset_path:
- golden_trace_path:
- judge_command:
- required_output_properties:
  - preserves final answer correctness
  - preserves required citations or tool-call arguments when applicable

Notes:

- Fill this section for reasoning-heavy workloads where exact byte-for-byte parity is not enough to describe quality.
- `doctor` fails `reasoning-task` aims without a complete reasoning-quality contract.

## 7. Allowed Mutation Surface

- allowed_mutations:
  - profiling instrumentation
  - benchmark harness improvements
  - copy reduction
  - scheduling and overlap
  - cache implementation
  - compile or backend changes only after parity proof
- blocked_by_default:
  - precision changes versus baseline
  - quantization
  - approximate kernels
  - algorithm changes that alter outputs
  - data or model changes

## 8. Experiment Budget

- max_iterations_per_session:
- max_runtime_per_experiment:
- stop_after_consecutive_failures:
- require_revert_on_failure: true

## 9. Logging

- experiment_log_path:
- best_result_path:
- progress_doc_path:
- worklog_doc_path:
- evaluator_report_path:
- handoff_doc_path:
- save_failed_runs: true

## 10. Human Override

- allow_non_zero_drift: false
- override_reason:

## 11. Notes

- additional_constraints:
- business_context:
- known_bottlenecks:
- suspected_safe_lanes:
