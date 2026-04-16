# Auto-Profiling 目标文件

你只需要编辑这个文件。agent 会把它当作运行契约。

## 1. 任务目标

- scenario: e2e-inference | llm-serving | cuda-kernel
- project_name:
- primary_goal:
- optimize_for: latency | throughput | cost_per_request | memory | startup_time
- target_metric_name:
- target_metric_direction: lower_is_better | higher_is_better
- target_sla:

## 2. 作用范围

- target_repo_path:
- target_entrypoints:
- baseline_files_allowed_to_change:
- files_never_touch:

## 3. 环境信息

- os:
- hardware:
- accelerator:
- python_env_command:
- git_required: true
- install_command:
- warmup_command:

说明：

- `scenario` 用于选择主技能模块和推荐的 aim 路线
- runtime 的 `status`、`current_contract` 和 `handoff` 会显式回写 `scenario`、`target_module` 和 `recommended_skill_route`
- `install_command` 留空或写成 `auto` 时，runtime 会自动在 `uv` 和 `pip` 之间选择
- shell 执行优先使用 `zsh`，没有 `zsh` 时回退到 `bash`，再不行则回退到系统默认 shell

## 4. Baseline 执行

- baseline_setup_command:
- baseline_run_command:
- baseline_profile_command:
- metric_output_path:
- exactness_output_path:

## 5. Exactness 契约

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

说明：

- `exact-parity`：默认模式。只要输出不一致就失败。
- `bounded-tolerance`：仅适用于逻辑和算法保持一致，但因 CPU/GPU 差异或 precision 切换引入微小数值误差的场景。
- 若使用 `bounded-tolerance`，必须明确填写 `abs_tolerance` 和 `rel_tolerance`。

## 6. 允许的修改边界

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

## 7. 实验预算

- max_iterations_per_session:
- max_runtime_per_experiment:
- stop_after_consecutive_failures:
- require_revert_on_failure: true

## 8. 日志与工件

- experiment_log_path:
- best_result_path:
- progress_doc_path:
- worklog_doc_path:
- evaluator_report_path:
- handoff_doc_path:
- save_failed_runs: true

## 9. 人类 Override

- allow_non_zero_drift: false
- override_reason:

说明：

- 默认应保持 `false`
- 只有你明确允许时，系统才可以接受非零 drift

## 10. 备注

- additional_constraints:
- business_context:
- known_bottlenecks:
- suspected_safe_lanes:

## 填写建议

最低要求：

- `target_repo_path`
- `baseline_run_command`
- `baseline_profile_command`
- `metric_output_path`
- `exactness_output_path`
- `exactness_check_command`
- `target_metric_name`
- `target_metric_direction`

强烈建议补充：

- `allowed_mutations`
- `blocked_by_default`
- `known_bottlenecks`
- `suspected_safe_lanes`
- `max_iterations_per_session`
- `max_runtime_per_experiment`

## Metric 输出示例

```json
{
  "metrics": {
    "p95_ms": 8.4,
    "throughput": 120.0
  }
}
```

## Exactness 输出示例

Exact-parity：

```json
{
  "passed": true,
  "mismatch_count": 0
}
```

Bounded-tolerance：

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
