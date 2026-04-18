#!/usr/bin/env python3
"""Generate scenario-specific aim files for E2E and LLM serving optimization."""

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
            "## 6. Allowed Mutation Surface",
            "",
            "- allowed_mutations:",
            "  - runtime and scheduler tuning",
            "  - graph/compile/fusion optimization",
            "  - memory layout and copy reduction",
            "- blocked_by_default:",
            "  - algorithmic behavior change",
            "  - unsafe precision drift",
            "",
            "## 7. Experiment Budget",
            "",
            "- max_iterations_per_session: 8",
            "- max_runtime_per_experiment: 1800",
            "- stop_after_consecutive_failures: 3",
            "- require_revert_on_failure: true",
            "",
            "## 8. Logging",
            "",
            "- experiment_log_path: ",
            "- best_result_path: ",
            "- progress_doc_path: ",
            "- worklog_doc_path: ",
            "- save_failed_runs: true",
            "",
            "## 9. Human Override",
            "",
            "- allow_non_zero_drift: false",
            "- override_reason: ",
            "",
            "## 10. Notes",
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
    parser.add_argument("--mode", choices=["e2e", "llm-serving"], required=True)
    parser.add_argument("--profile", required=True, help="E2E model family or serving backend")
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
    else:
        if args.profile not in LLM_BACKEND_PRESETS:
            raise SystemExit(f"unsupported llm-serving profile: {args.profile}")
        preset = {**LLM_BACKEND_PRESETS[args.profile], "optimize_for": "latency"}
        scenario = "llm-serving"

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
