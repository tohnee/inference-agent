#!/usr/bin/env python3
"""Generate single-aim contracts for inference optimization scenarios."""

from __future__ import annotations

import argparse
from pathlib import Path


E2E_PRESETS = {
    "small-model": {
        "optimize_for": "latency",
        "target_metric_name": "p95_ms",
        "target_metric_direction": "lower_is_better",
        "baseline_profile_command": "python3 tools/profile_e2e.py --trace --stages preprocess,infer,postprocess",
        "known_bottlenecks": "python overhead, copies, stage imbalance",
        "suspected_safe_lanes": "batching, torch.compile, io overlap",
    },
    "diffusion": {
        "optimize_for": "latency",
        "target_metric_name": "steps_per_second",
        "target_metric_direction": "higher_is_better",
        "baseline_profile_command": "python3 tools/profile_diffusion.py --trace --components unet,vae,text_encoder",
        "known_bottlenecks": "unet attention, scheduler step cost, host-device sync",
        "suspected_safe_lanes": "attention kernel, scheduler tuning, graph capture",
    },
    "dl": {
        "optimize_for": "throughput",
        "target_metric_name": "samples_per_second",
        "target_metric_direction": "higher_is_better",
        "baseline_profile_command": "python3 tools/profile_e2e.py --trace --components dataloader,forward,postprocess",
        "known_bottlenecks": "dataloader and pre/post imbalance",
        "suspected_safe_lanes": "pipeline overlap, async io, copy reduction",
    },
    "transformer": {
        "optimize_for": "latency",
        "target_metric_name": "p95_ms",
        "target_metric_direction": "lower_is_better",
        "baseline_profile_command": "python3 tools/profile_transformer.py --trace --ops attention,mlp",
        "known_bottlenecks": "attention and kv movement",
        "suspected_safe_lanes": "flash-attn, compile, kv/cache layout",
    },
    "sam": {
        "optimize_for": "latency",
        "target_metric_name": "p95_ms",
        "target_metric_direction": "lower_is_better",
        "baseline_profile_command": "python3 tools/profile_sam.py --trace --components image_encoder,prompt_encoder,mask_decoder",
        "known_bottlenecks": "image encoder heavy compute",
        "suspected_safe_lanes": "encoder graph capture, pre/post parallel",
    },
    "vit": {
        "optimize_for": "throughput",
        "target_metric_name": "samples_per_second",
        "target_metric_direction": "higher_is_better",
        "baseline_profile_command": "python3 tools/profile_vit.py --trace --ops patch_embed,attention,mlp",
        "known_bottlenecks": "attention and memory format conversion",
        "suspected_safe_lanes": "channels-last, fused kernels, compile",
    },
    "tree": {
        "optimize_for": "latency",
        "target_metric_name": "p95_ms",
        "target_metric_direction": "lower_is_better",
        "baseline_profile_command": "python3 tools/profile_tree_model.py --trace --components feature_fetch,predict",
        "known_bottlenecks": "feature engineering and cpu cache miss",
        "suspected_safe_lanes": "vectorization, feature cache, thread pinning",
    },
}

LLM_BACKEND_PRESETS = {
    "sglang": {
        "target_metric_name": "ttft_ms",
        "target_metric_direction": "lower_is_better",
        "baseline_run_command": "python3 benchmarks/bench_serving.py --backend sglang --dataset realistic --output .auto-profiling/metric.json",
        "baseline_profile_command": "python3 benchmarks/profile_serving.py --backend sglang --trace --output .auto-profiling/profile.json",
        "known_bottlenecks": "prefill/decode overlap and scheduler fairness",
        "suspected_safe_lanes": "continuous batching, prefix cache, cuda graph",
    },
    "vllm": {
        "target_metric_name": "tpot_ms",
        "target_metric_direction": "lower_is_better",
        "baseline_run_command": "python3 benchmarks/bench_serving.py --backend vllm --dataset realistic --output .auto-profiling/metric.json",
        "baseline_profile_command": "python3 benchmarks/profile_serving.py --backend vllm --trace --output .auto-profiling/profile.json",
        "known_bottlenecks": "kv cache paging and decode kernel tail latency",
        "suspected_safe_lanes": "scheduler policy, paged kv tuning, chunked prefill",
    },
    "trtllm": {
        "target_metric_name": "tokens_per_second",
        "target_metric_direction": "higher_is_better",
        "baseline_run_command": "python3 benchmarks/bench_serving.py --backend trtllm --dataset realistic --output .auto-profiling/metric.json",
        "baseline_profile_command": "python3 benchmarks/profile_serving.py --backend trtllm --trace --output .auto-profiling/profile.json",
        "known_bottlenecks": "engine shape coverage and queueing tail",
        "suspected_safe_lanes": "engine rebuild policy, dynamic batching, inter-node topology",
    },
}

CUDA_KERNEL_PRESETS = {
    "cuda": {
        "optimize_for": "latency",
        "target_metric_name": "latency_ms",
        "target_metric_direction": "lower_is_better",
        "baseline_run_command": "python3 benchmarks/bench_kernel.py --backend cuda --output .auto-profiling/metric.json",
        "baseline_profile_command": "ncu --set full --target-processes all -o .auto-profiling/ncu_cuda python3 benchmarks/bench_kernel.py --backend cuda",
        "known_bottlenecks": "global memory bandwidth, occupancy limits, warp divergence, launch overhead",
        "suspected_safe_lanes": "memory coalescing, block sizing, shared-memory tiling, launch configuration",
    },
    "triton": {
        "optimize_for": "latency",
        "target_metric_name": "latency_ms",
        "target_metric_direction": "lower_is_better",
        "baseline_run_command": "python3 benchmarks/bench_kernel.py --backend triton --output .auto-profiling/metric.json",
        "baseline_profile_command": "python3 benchmarks/profile_triton.py --trace --output .auto-profiling/profile.json",
        "known_bottlenecks": "program tiling, memory bandwidth, register pressure, autotune coverage",
        "suspected_safe_lanes": "tile shape, num_warps, num_stages, vectorized loads",
    },
    "cutlass": {
        "optimize_for": "throughput",
        "target_metric_name": "throughput_gbps",
        "target_metric_direction": "higher_is_better",
        "baseline_run_command": "python3 benchmarks/bench_kernel.py --backend cutlass --output .auto-profiling/metric.json",
        "baseline_profile_command": "ncu --set full -o .auto-profiling/ncu_cutlass python3 benchmarks/bench_kernel.py --backend cutlass",
        "known_bottlenecks": "threadblock shape coverage, epilogue fusion, tensor core utilization",
        "suspected_safe_lanes": "CUTLASS kernel selection, epilogue fusion, alignment and layout",
    },
    "operator": {
        "optimize_for": "latency",
        "target_metric_name": "latency_ms",
        "target_metric_direction": "lower_is_better",
        "baseline_run_command": "python3 benchmarks/bench_operator.py --output .auto-profiling/metric.json",
        "baseline_profile_command": "python3 benchmarks/profile_operator.py --trace --output .auto-profiling/profile.json",
        "known_bottlenecks": "operator backend choice, memory traffic, occupancy, launch overhead",
        "suspected_safe_lanes": "backend synthesis, CPU reference parity, Triton/CUDA scaffold, microbenchmark loop",
    },
}

REASONING_PRESETS = {
    "rag": {
        "scenario": "reasoning-task",
        "optimize_for": "latency",
        "target_metric_name": "p95_ms",
        "target_metric_direction": "lower_is_better",
        "baseline_run_command": "python3 evals/run_rag_eval.py --output .auto-profiling/metric.json --quality-output .auto-profiling/reasoning_quality.json",
        "baseline_profile_command": "python3 evals/profile_rag.py --trace --output .auto-profiling/profile.json",
        "reasoning_task_type": "rag_synthesis",
        "reasoning_quality_metric": "citation_fidelity",
        "reasoning_quality_threshold": "no citation-support regression and rubric_score >= baseline",
        "golden_dataset_path": "evals/golden/rag_questions.jsonl",
        "golden_trace_path": "",
        "judge_command": "python3 evals/check_rag_quality.py --output .auto-profiling/reasoning_quality.json",
        "required_output_properties": [
            "preserves final answer correctness",
            "preserves required citations",
            "preserves source support for grounded claims",
        ],
        "known_bottlenecks": "retrieval fanout, context packing, prefill latency, synthesis latency",
        "suspected_safe_lanes": "retrieval batching, context deduplication, prefix cache, prompt packing gated by citation checks",
    },
    "tool-agent": {
        "scenario": "reasoning-task",
        "optimize_for": "latency",
        "target_metric_name": "p95_ms",
        "target_metric_direction": "lower_is_better",
        "baseline_run_command": "python3 evals/run_agent_eval.py --output .auto-profiling/metric.json --quality-output .auto-profiling/reasoning_quality.json",
        "baseline_profile_command": "python3 evals/profile_agent.py --trace --output .auto-profiling/profile.json",
        "reasoning_task_type": "tool_agent",
        "reasoning_quality_metric": "tool_trace_parity",
        "reasoning_quality_threshold": "required tool calls valid and task_success_rate >= baseline",
        "golden_dataset_path": "evals/golden/agent_tasks.jsonl",
        "golden_trace_path": "evals/golden/tool_traces.jsonl",
        "judge_command": "python3 evals/check_agent_quality.py --output .auto-profiling/reasoning_quality.json",
        "required_output_properties": [
            "preserves final answer correctness",
            "preserves valid tool-call arguments",
            "preserves required tool ordering constraints",
        ],
        "known_bottlenecks": "tool latency, retry loops, planner serialization, request orchestration",
        "suspected_safe_lanes": "tool concurrency, scheduler tuning, trace caching, retry budget tightening gated by trace validity",
    },
    "code-reasoning": {
        "scenario": "reasoning-task",
        "optimize_for": "cost_per_request",
        "target_metric_name": "tokens_per_task",
        "target_metric_direction": "lower_is_better",
        "baseline_run_command": "python3 evals/run_code_eval.py --output .auto-profiling/metric.json --quality-output .auto-profiling/reasoning_quality.json",
        "baseline_profile_command": "python3 evals/profile_code_agent.py --trace --output .auto-profiling/profile.json",
        "reasoning_task_type": "code_reasoning",
        "reasoning_quality_metric": "semantic_rubric",
        "reasoning_quality_threshold": "tests_passed >= baseline and rubric_score >= baseline",
        "golden_dataset_path": "evals/golden/code_tasks.jsonl",
        "golden_trace_path": "",
        "judge_command": "python3 evals/check_code_quality.py --output .auto-profiling/reasoning_quality.json",
        "required_output_properties": [
            "preserves test pass rate",
            "preserves edit correctness",
            "preserves required explanation constraints",
        ],
        "known_bottlenecks": "long prompts, repeated repository context, slow test loop, excessive retries",
        "suspected_safe_lanes": "context cache, test selection, prompt packing, batching gated by pass-rate checks",
    },
    "long-context": {
        "scenario": "reasoning-task",
        "optimize_for": "memory",
        "target_metric_name": "peak_memory_mb",
        "target_metric_direction": "lower_is_better",
        "baseline_run_command": "python3 evals/run_long_context_eval.py --output .auto-profiling/metric.json --quality-output .auto-profiling/reasoning_quality.json",
        "baseline_profile_command": "python3 evals/profile_long_context.py --trace --output .auto-profiling/profile.json",
        "reasoning_task_type": "long_context",
        "reasoning_quality_metric": "semantic_rubric",
        "reasoning_quality_threshold": "answer_completeness >= baseline and no critical fact loss",
        "golden_dataset_path": "evals/golden/long_context_tasks.jsonl",
        "golden_trace_path": "",
        "judge_command": "python3 evals/check_long_context_quality.py --output .auto-profiling/reasoning_quality.json",
        "required_output_properties": [
            "preserves final answer correctness",
            "preserves required evidence coverage",
            "preserves long-context recall requirements",
        ],
        "known_bottlenecks": "prefill latency, context-window pressure, memory footprint",
        "suspected_safe_lanes": "prefix cache, context compaction, chunking, attention backend tuning gated by completeness checks",
    },
    "chat-reasoning": {
        "scenario": "reasoning-task",
        "optimize_for": "latency",
        "target_metric_name": "p95_ms",
        "target_metric_direction": "lower_is_better",
        "baseline_run_command": "python3 evals/run_chat_eval.py --output .auto-profiling/metric.json --quality-output .auto-profiling/reasoning_quality.json",
        "baseline_profile_command": "python3 evals/profile_chat.py --trace --output .auto-profiling/profile.json",
        "reasoning_task_type": "chat_reasoning",
        "reasoning_quality_metric": "judge_score",
        "reasoning_quality_threshold": "judge_score >= baseline and no refusal/instruction-following regression",
        "golden_dataset_path": "evals/golden/chat_tasks.jsonl",
        "golden_trace_path": "",
        "judge_command": "python3 evals/check_chat_quality.py --output .auto-profiling/reasoning_quality.json",
        "required_output_properties": [
            "preserves final answer correctness",
            "preserves instruction-following behavior",
            "preserves refusal and safety-policy behavior",
        ],
        "known_bottlenecks": "decode latency, prompt length, repeated system/context tokens",
        "suspected_safe_lanes": "prompt packing, prefix cache, batching, serving scheduler tuning gated by rubric checks",
    },
}


def render_template(scenario: str, project_name: str, repo_path: str, preset: dict[str, str]) -> str:
    return "\n".join(
        [
            "# Auto-Profiling Aim",
            "",
            "## 1. Mission",
            "",
            f"- scenario: {scenario}",
            f"- project_name: {project_name}",
            "- primary_goal: optimize inference",
            f"- optimize_for: {preset['optimize_for']}",
            f"- target_metric_name: {preset['target_metric_name']}",
            f"- target_metric_direction: {preset['target_metric_direction']}",
            "",
            "## 2. Scope",
            "",
            f"- target_repo_path: {repo_path}",
            "- target_entrypoints: service.py",
            "- baseline_files_allowed_to_change: ",
            "- files_never_touch: ",
            "",
            "## 3. Environment",
            "",
            "- os: linux",
            "- hardware: gpu",
            "- accelerator: nvidia",
            "- python_env_command: ",
            "- git_required: true",
            "- install_command: auto",
            "- warmup_command: ",
            "",
            "## 4. Baseline Execution",
            "",
            "- baseline_setup_command: ",
            f"- baseline_run_command: {preset['baseline_run_command']}",
            f"- baseline_profile_command: {preset['baseline_profile_command']}",
            "- profile_output_path: .auto-profiling/profile.json",
            "- metric_output_path: .auto-profiling/metric.json",
            "- exactness_output_path: .auto-profiling/exactness.json",
            "",
            "## 5. Exactness Contract",
            "",
            "- exactness_mode: exact-parity",
            "- reference_path_description: cpu or trusted serving baseline",
            "- golden_input_location: .auto-profiling/golden_input.json",
            "- golden_output_location: .auto-profiling/golden_output.json",
            "- exactness_check_command: python3 tools/check_exactness.py --output .auto-profiling/exactness.json",
            "- deterministic_requirements: fixed seed + deterministic decode",
            "- cache_semantics_requirements: exact cache behavior",
            "- request_isolation_requirements: exact request isolation",
            "",
            "## 6. Reasoning Quality Contract",
            "",
            f"- reasoning_task_type: {preset.get('reasoning_task_type', '')}",
            f"- reasoning_quality_metric: {preset.get('reasoning_quality_metric', '')}",
            f"- reasoning_quality_threshold: {preset.get('reasoning_quality_threshold', '')}",
            f"- golden_dataset_path: {preset.get('golden_dataset_path', '')}",
            f"- golden_trace_path: {preset.get('golden_trace_path', '')}",
            f"- judge_command: {preset.get('judge_command', '')}",
            "- required_output_properties:",
            *[f"  - {item}" for item in preset.get('required_output_properties', [])],
            "",
            "## 7. Allowed Mutation Surface",
            "",
            "- allowed_mutations:",
            "  - runtime and scheduler tuning",
            "  - graph/compile/fusion optimization",
            "  - memory layout and copy reduction",
            "- blocked_by_default:",
            "  - algorithmic behavior change",
            "  - unsafe precision drift",
            "",
            "## 8. Experiment Budget",
            "",
            "- max_iterations_per_session: 8",
            "- max_runtime_per_experiment: 1800",
            "- stop_after_consecutive_failures: 3",
            "- require_revert_on_failure: true",
            "",
            "## 9. Logging",
            "",
            "- experiment_log_path: ",
            "- best_result_path: ",
            "- progress_doc_path: ",
            "- worklog_doc_path: ",
            "- save_failed_runs: true",
            "",
            "## 10. Human Override",
            "",
            "- allow_non_zero_drift: false",
            "- override_reason: ",
            "",
            "## 11. Notes",
            "",
            "- additional_constraints: distributed serving and ai-infra constraints must be recorded",
            "- business_context: ",
            f"- known_bottlenecks: {preset['known_bottlenecks']}",
            f"- suspected_safe_lanes: {preset['suspected_safe_lanes']}",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate scenario-specific aim template")
    parser.add_argument("--mode", choices=["e2e", "llm-serving", "cuda-kernel", "reasoning"], required=True)
    parser.add_argument("--profile", required=True, help="E2E model family, serving backend, or CUDA/kernel backend")
    parser.add_argument("--project-name", required=True)
    parser.add_argument("--target-repo-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.mode == "e2e":
        if args.profile not in E2E_PRESETS:
            raise SystemExit(f"unsupported e2e profile: {args.profile}")
        preset = {
            **E2E_PRESETS[args.profile],
            "baseline_run_command": "python3 tools/run_e2e_infer.py --output .auto-profiling/metric.json",
        }
        scenario = "e2e-inference"
    elif args.mode == "llm-serving":
        if args.profile not in LLM_BACKEND_PRESETS:
            raise SystemExit(f"unsupported llm-serving profile: {args.profile}")
        preset = {**LLM_BACKEND_PRESETS[args.profile], "optimize_for": "latency"}
        scenario = "llm-serving"
    elif args.mode == "cuda-kernel":
        if args.profile not in CUDA_KERNEL_PRESETS:
            raise SystemExit(f"unsupported cuda-kernel profile: {args.profile}")
        preset = CUDA_KERNEL_PRESETS[args.profile]
        scenario = "operator-kernel" if args.profile == "operator" else "cuda-kernel"
    else:
        if args.profile not in REASONING_PRESETS:
            raise SystemExit(f"unsupported reasoning profile: {args.profile}")
        preset = REASONING_PRESETS[args.profile]
        scenario = preset["scenario"]

    content = render_template(
        scenario=scenario,
        project_name=args.project_name,
        repo_path=args.target_repo_path,
        preset=preset,
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
