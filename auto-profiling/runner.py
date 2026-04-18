from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scorer import compare_runs, load_json, write_json


DEFAULT_SCENARIO = "e2e-inference"
SCENARIO_LANES = {
    "e2e-inference": {
        "target_module": "e2e-inference-opt-skill",
        "recommended_skill_route": ["auto-profiling", "e2e-inference-opt-skill"],
        "recommended_aim_template": "aim.e2e.md",
    },
    "llm-serving": {
        "target_module": "llm-serving-opt-skill",
        "recommended_skill_route": [
            "auto-profiling",
            "llm-serving-opt-skill",
            "serving-benchmark-skill",
        ],
        "recommended_aim_template": "aim.llm-serving.md",
    },
    "cuda-kernel": {
        "target_module": "cuda-kernel-opt-skill",
        "recommended_skill_route": [
            "auto-profiling",
            "cuda-kernel-opt-skill",
            "cuda-optimized-skill",
        ],
        "recommended_aim_template": "aim.cuda-kernel.md",
    },
    "operator-kernel": {
        "target_module": "cuda-kernel-opt-skill",
        "recommended_skill_route": [
            "auto-profiling",
            "cuda-kernel-opt-skill",
            "operator-backend-synthesis-skill",
            "cuda-optimized-skill",
        ],
        "recommended_aim_template": "aim.cuda-kernel.md",
    },
}


def parse_scalar(value: str) -> Any:
    stripped = value.strip()
    lowered = stripped.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    if stripped == "":
        return ""
    try:
        if "." in stripped:
            return float(stripped)
        return int(stripped)
    except ValueError:
        return stripped


def parse_aim_markdown(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    current_list_key: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if line.startswith("#"):
            current_list_key = None
            continue
        if line.startswith("- "):
            current_list_key = None
            body = line[2:]
            if ":" not in body:
                continue
            key, value = body.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value == "":
                data[key] = []
                current_list_key = key
            else:
                data[key] = parse_scalar(value)
            continue
        if line.startswith("  - ") and current_list_key:
            data[current_list_key].append(parse_scalar(line[4:].strip()))
    return data


def read_aim(path: Path) -> dict[str, Any]:
    data = parse_aim_markdown(path.read_text(encoding="utf-8"))
    data["_aim_path"] = str(path)
    return data


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def repo_root_from_aim(aim: dict[str, Any], aim_path: Path) -> Path:
    target = aim.get("target_repo_path")
    if target:
        return Path(target).expanduser().resolve()
    return aim_path.parent.resolve()


def template_root() -> Path:
    return Path(__file__).resolve().parent / ".auto-profiling"


def detect_preferred_shell() -> dict[str, Any]:
    for name in ("zsh", "bash", "sh"):
        path = shutil.which(name)
        if path:
            return {"name": name, "path": path}
    return {"name": "system-default", "path": None}


def detect_package_manager() -> dict[str, Any]:
    uv_path = shutil.which("uv")
    if uv_path:
        return {"name": "uv", "path": uv_path}
    pip_path = shutil.which("pip") or shutil.which("pip3")
    if pip_path:
        return {"name": "pip", "path": pip_path}
    return {"name": "pip", "path": None}


def auto_install_command(project_root: Path) -> str | None:
    package_manager = detect_package_manager()["name"]
    has_pyproject = (project_root / "pyproject.toml").exists()
    has_requirements = (project_root / "requirements.txt").exists()
    has_setup = (project_root / "setup.py").exists() or (project_root / "setup.cfg").exists()

    if package_manager == "uv" and has_pyproject:
        return "uv sync"
    if has_requirements:
        return f'"{sys.executable}" -m pip install -r requirements.txt'
    if has_pyproject or has_setup:
        return f'"{sys.executable}" -m pip install -e .'
    return None


def resolve_scenario_lane(aim: dict[str, Any]) -> dict[str, Any]:
    raw_scenario = str(aim.get("scenario") or DEFAULT_SCENARIO).strip()
    scenario = raw_scenario if raw_scenario in SCENARIO_LANES else DEFAULT_SCENARIO
    lane = dict(SCENARIO_LANES[scenario])
    lane["scenario"] = scenario
    lane["requested_scenario"] = raw_scenario
    lane["recommended_skill_route_text"] = " -> ".join(lane["recommended_skill_route"])
    return lane


def preview_text(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:240]
    return None


def detect_installed_packages() -> dict[str, str]:
    packages: dict[str, str] = {}
    for name in ("torch", "vllm", "sglang", "triton", "flashinfer", "transformers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return packages


def probe_tool(command: list[str], timeout_seconds: int = 15) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "command": " ".join(command),
            "exit_code": None,
            "stdout_preview": None,
            "stderr_preview": str(exc),
        }
    return {
        "command": " ".join(command),
        "exit_code": completed.returncode,
        "stdout_preview": preview_text(completed.stdout),
        "stderr_preview": preview_text(completed.stderr),
    }


def collect_optional_vllm_env() -> dict[str, Any]:
    command: list[str] | None = None
    vllm_binary = shutil.which("vllm")
    if vllm_binary:
        command = [vllm_binary, "collect-env"]
    elif "vllm" in detect_installed_packages():
        command = [sys.executable, "-m", "vllm", "collect-env"]
    if not command:
        return {"available": False, "invoked": False, "command": None, "summary": None}
    result = probe_tool(command, timeout_seconds=30)
    return {
        "available": True,
        "invoked": result["exit_code"] == 0,
        "command": result["command"],
        "summary": result["stdout_preview"] or result["stderr_preview"],
        "exit_code": result["exit_code"],
    }


def detect_runtime_environment(project_root: Path) -> dict[str, Any]:
    shell = detect_preferred_shell()
    package_manager = detect_package_manager()
    tool_paths = {
        name: path
        for name in ("git", "python3", "nvidia-smi", "nvcc", "uv", "pip", "vllm")
        if (path := shutil.which(name))
    }
    diagnostics = {
        "git": probe_tool(["git", "--version"]) if tool_paths.get("git") else None,
        "python": probe_tool([sys.executable, "--version"]),
        "package_manager": (
            probe_tool(["uv", "--version"])
            if package_manager["name"] == "uv"
            else probe_tool([sys.executable, "-m", "pip", "--version"])
        ),
        "nvidia_smi": (
            probe_tool(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
            if tool_paths.get("nvidia-smi")
            else None
        ),
        "nvcc": probe_tool(["nvcc", "--version"]) if tool_paths.get("nvcc") else None,
    }
    return {
        "shell": shell,
        "package_manager": package_manager,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.system().lower(),
        "platform_release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "project_root": str(project_root),
        "auto_install_command": auto_install_command(project_root),
        "virtual_env": os.environ.get("VIRTUAL_ENV"),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tool_paths": tool_paths,
        "tool_diagnostics": diagnostics,
        "detected_packages": detect_installed_packages(),
        "vllm_collect_env": collect_optional_vllm_env(),
    }


def initialize_workspace(project_root: Path) -> dict[str, str]:
    state_dir = project_root / ".auto-profiling"
    state_dir.mkdir(parents=True, exist_ok=True)
    failures_dir = state_dir / "failures"
    failures_dir.mkdir(exist_ok=True)

    files = {
        "experiment_log_md": state_dir / "experiment_log.md",
        "experiment_log_jsonl": state_dir / "experiment_log.jsonl",
        "baseline_snapshot_json": state_dir / "baseline_snapshot.json",
        "best_result_json": state_dir / "best_result.json",
        "session_state_json": state_dir / "session_state.json",
        "task_plan_md": state_dir / "task_plan.md",
        "findings_md": state_dir / "findings.md",
        "progress_md": state_dir / "progress.md",
        "worklog_md": state_dir / "worklog.md",
        "current_contract_md": state_dir / "current_contract.md",
        "evaluator_report_md": state_dir / "evaluator_report.md",
        "next_handoff_md": state_dir / "next_handoff.md",
        "skill_route_plan_md": state_dir / "skill_route_plan.md",
    }

    template_dir = template_root()
    if template_dir.exists():
        for key, target in files.items():
            source = template_dir / target.name
            if source.exists() and not target.exists():
                shutil.copyfile(source, target)

    defaults = {
        "experiment_log_md": "# Experiment Log\n\n| Experiment | Lane | Exactness | Metric | Decision |\n| --- | --- | --- | --- | --- |\n",
        "experiment_log_jsonl": "",
        "baseline_snapshot_json": "{}\n",
        "best_result_json": "{}\n",
        "session_state_json": "{\n  \"status\": \"initialized\",\n  \"iterations_completed\": 0,\n  \"consecutive_failures\": 0,\n  \"last_experiment\": null,\n  \"best_experiment\": null,\n  \"next_action\": \"run baseline\"\n}\n",
        "task_plan_md": "# Task Plan\n\n## Goal\n\n- establish baseline\n- iterate one experiment at a time\n\n## Current Phase\n\n- setup\n",
        "findings_md": "# Findings\n\n## Discoveries\n\n",
        "progress_md": "# Progress\n\n## Session Log\n\n",
        "worklog_md": "# Worklog\n\n## Overview\n\n",
        "current_contract_md": "# Current Contract\n\n## Objective\n\n- fill from aim.md before each experiment\n",
        "evaluator_report_md": "# Evaluator Report\n\n## Latest Decision\n\n- no evaluation yet\n",
        "next_handoff_md": "# Next Handoff\n\n## Resume Here\n\n- initialize baseline and capture trusted metrics\n",
        "skill_route_plan_md": "# Skill Route Plan\n\n## Current Scenario\n\n- not resolved yet\n",
    }

    for key, target in files.items():
        if not target.exists():
            target.write_text(defaults[key], encoding="utf-8")

    return {"state_dir": str(state_dir), **{key: str(value) for key, value in files.items()}}


def shell_result(command: str, cwd: Path, prefix: str | None = None) -> dict[str, Any]:
    shell = detect_preferred_shell()
    if not command:
        return {
            "command": command,
            "cwd": str(cwd),
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
            "started_at": utc_now(),
            "finished_at": utc_now(),
            "shell": shell,
        }
    full_command = command if not prefix else f"{prefix} && {command}"
    started_at = utc_now()
    run_kwargs = {
        "args": full_command,
        "cwd": str(cwd),
        "shell": True,
        "text": True,
        "capture_output": True,
    }
    if shell["path"]:
        run_kwargs["executable"] = shell["path"]
    completed = subprocess.run(**run_kwargs)
    finished_at = utc_now()
    return {
        "command": full_command,
        "cwd": str(cwd),
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "started_at": started_at,
        "finished_at": finished_at,
        "shell": shell,
    }



def run_required(command: str, cwd: Path, prefix: str | None = None) -> dict[str, Any]:
    result = shell_result(command, cwd=cwd, prefix=prefix)
    if result["exit_code"] != 0:
        raise RuntimeError(
            f"command failed: {result['command']}\n{result['stderr'] or result['stdout']}"
        )
    return result


def run_required_with_retry(
    command: str,
    cwd: Path,
    prefix: str | None = None,
    retry_count: int = 1,
) -> dict[str, Any]:
    attempts = max(1, retry_count)
    last_error: RuntimeError | None = None
    for attempt in range(1, attempts + 1):
        try:
            return run_required(command, cwd=cwd, prefix=prefix)
        except RuntimeError as exc:
            last_error = exc
            if attempt == attempts:
                break
    assert last_error is not None
    raise last_error


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def append_markdown(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def load_metric_payload(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if "metrics" in payload:
        return payload
    return {"metrics": payload}


def load_exactness_payload(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if "exactness" in payload:
        return payload["exactness"]
    return payload


def collect_candidate_payload(aim: dict[str, Any], project_root: Path) -> dict[str, Any]:
    metric_output_path = Path(str(aim["metric_output_path"]))
    exactness_output_path = Path(str(aim["exactness_output_path"]))
    if not metric_output_path.is_absolute():
        metric_output_path = project_root / metric_output_path
    if not exactness_output_path.is_absolute():
        exactness_output_path = project_root / exactness_output_path
    metric_payload = load_metric_payload(metric_output_path)
    exactness_payload = load_exactness_payload(exactness_output_path)
    return {"metrics": metric_payload["metrics"], "exactness": exactness_payload}


def exactness_policy_from_aim(aim: dict[str, Any]) -> dict[str, Any]:
    mode = str(aim.get("exactness_mode", "exact-parity"))
    policy = {
        "mode": mode,
        "abs_tolerance": float(aim.get("abs_tolerance", 0.0) or 0.0),
        "rel_tolerance": float(aim.get("rel_tolerance", 0.0) or 0.0),
        "require_logic_equivalence": bool(aim.get("require_logic_equivalence", True)),
        "require_algorithm_equivalence": bool(aim.get("require_algorithm_equivalence", True)),
    }
    return policy


def git_revision(project_root: Path) -> str | None:
    result = shell_result("git rev-parse HEAD", cwd=project_root)
    if result["exit_code"] == 0:
        return result["stdout"].strip() or None
    return None


def load_state(path: str | Path) -> dict[str, Any]:
    payload = load_json(path)
    if payload:
        return payload
    return {
        "status": "initialized",
        "iterations_completed": 0,
        "consecutive_failures": 0,
        "last_experiment": None,
        "best_experiment": None,
        "next_action": "run baseline",
    }


def ensure_git_repo(project_root: Path, required: bool) -> None:
    if not required:
        return
    result = shell_result("git rev-parse --is-inside-work-tree", cwd=project_root)
    if result["exit_code"] != 0 or result["stdout"].strip().lower() != "true":
        raise RuntimeError("git repository required by aim.md but not available at target_repo_path")


def apply_workspace_overrides(
    workspace: dict[str, str],
    aim: dict[str, Any],
    project_root: Path,
) -> dict[str, str]:
    mapping = {
        "experiment_log_path": "experiment_log_md",
        "best_result_path": "best_result_json",
        "progress_doc_path": "progress_md",
        "worklog_doc_path": "worklog_md",
        "evaluator_report_path": "evaluator_report_md",
        "handoff_doc_path": "next_handoff_md",
    }
    updated = dict(workspace)
    for aim_key, workspace_key in mapping.items():
        value = aim.get(aim_key)
        if not value:
            continue
        path = Path(str(value))
        if not path.is_absolute():
            path = project_root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            shutil.copyfile(Path(updated[workspace_key]), path)
        updated[workspace_key] = str(path)
    return updated


def resolve_install_command(aim: dict[str, Any], project_root: Path) -> str | None:
    explicit = aim.get("install_command")
    if explicit and str(explicit).strip().lower() != "auto":
        return str(explicit)
    return auto_install_command(project_root)


def command_retry_count_from_aim(aim: dict[str, Any]) -> int:
    raw = aim.get("command_retry_count", 1)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 1
    return max(1, value)


def write_skill_route_plan(workspace: dict[str, str], lane: dict[str, Any]) -> None:
    lines = [
        "# Skill Route Plan",
        "",
        "## Current Scenario",
        "",
        f"- scenario: {lane['scenario']}",
        f"- target_module: {lane['target_module']}",
        f"- recommended_skill_route: {lane['recommended_skill_route_text']}",
        "",
        "## Independent Skill Entry Points (for Codex/agent sessions)",
        "",
        "- e2e-inference: e2e-inference-opt-skill/SKILL.md",
        "- llm-serving: llm-serving-opt-skill/SKILL.md",
        "- cuda-kernel/operator: cuda-kernel-opt-skill/SKILL.md",
        "",
        "## Operator Synthesis Shortcut",
        "",
        "- python cuda-kernel-opt-skill/skills/cuda-optimized-skill/operator-optimize-loop/scripts/operator_backend_synth.py "
        "--name=<op> --logic='<logic>' --op-type=matmul --backend=auto --output-dir=<dir>",
    ]
    Path(workspace["skill_route_plan_md"]).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_contract(
    aim: dict[str, Any],
    project_root: Path,
    phase: str,
    label: str,
) -> dict[str, Any]:
    prefix = aim.get("python_env_command") or None
    environment = detect_runtime_environment(project_root)
    lane = resolve_scenario_lane(aim)
    retry_count = command_retry_count_from_aim(aim)
    commands = []
    command_specs = [
        ("install_command", resolve_install_command(aim, project_root)),
        ("warmup_command", aim.get("warmup_command")),
        ("baseline_setup_command", aim.get("baseline_setup_command")),
        ("baseline_run_command", aim.get("baseline_run_command")),
        ("exactness_check_command", aim.get("exactness_check_command")),
        ("baseline_profile_command", aim.get("baseline_profile_command")),
    ]
    for key, command in command_specs:
        if command:
            commands.append(
                {
                    "name": key,
                    **run_required_with_retry(
                        str(command),
                        cwd=project_root,
                        prefix=prefix,
                        retry_count=retry_count,
                    ),
                }
            )
    payload = collect_candidate_payload(aim, project_root)
    return {
        "label": label,
        "phase": phase,
        "lane": lane,
        "timestamp": utc_now(),
        "git_revision": git_revision(project_root),
        "commands": commands,
        "environment": environment,
        "metrics": payload["metrics"],
        "exactness": payload["exactness"],
    }


def log_session_artifacts(
    workspace: dict[str, str],
    record: dict[str, Any],
    decision: dict[str, Any] | None = None,
) -> None:
    experiment_log_md = Path(workspace["experiment_log_md"])
    experiment_log_jsonl = Path(workspace["experiment_log_jsonl"])
    progress_md = Path(workspace["progress_md"])
    worklog_md = Path(workspace["worklog_md"])

    metric_name = None
    metric_value = None
    if decision:
        metric_name = decision["metric_name"]
        metric_value = decision["candidate_value"]
        exactness_text = "pass" if decision["candidate_exactness"]["passed"] else "fail"
        decision_text = "keep" if decision["keep"] else f"revert:{decision['rejection_reason']}"
    else:
        metrics = record.get("metrics", {})
        if metrics:
            metric_name, metric_value = next(iter(metrics.items()))
        exactness_payload = record.get("exactness", {})
        exactness_text = "pass" if exactness_payload.get("passed", False) else "fail"
        decision_text = record["phase"]

    append_markdown(
        experiment_log_md,
        f"| {record['label']} | {record.get('lane', {}).get('scenario', record['phase'])} | {exactness_text} | {metric_name}={metric_value} | {decision_text} |\n",
    )
    append_jsonl(
        experiment_log_jsonl,
        {
            "experiment_id": record["label"],
            "phase": record["phase"],
            "timestamp": record["timestamp"],
            "git_revision": record["git_revision"],
            "lane": record.get("lane"),
            "metrics": record["metrics"],
            "exactness": record["exactness"],
            "decision": decision,
        },
    )
    append_markdown(
        progress_md,
        f"- {record['timestamp']} {record['label']} {record['phase']} exactness={exactness_text}\n",
    )
    append_markdown(
        worklog_md,
        f"## {record['label']}\n\n- phase: {record['phase']}\n- timestamp: {record['timestamp']}\n- git_revision: {record['git_revision']}\n- metrics: {json.dumps(record['metrics'], ensure_ascii=False)}\n- exactness: {json.dumps(record['exactness'], ensure_ascii=False)}\n- decision: {json.dumps(decision, ensure_ascii=False) if decision else 'baseline_snapshot'}\n\n",
    )


def write_contract_doc(workspace: dict[str, str], aim: dict[str, Any], label: str, phase: str) -> None:
    environment = detect_runtime_environment(Path(workspace["state_dir"]).parent)
    lane = resolve_scenario_lane(aim)
    lines = [
        "# Current Contract",
        "",
        f"## {label}",
        "",
        f"- phase: {phase}",
        f"- scenario: {lane['scenario']}",
        f"- target_module: {lane['target_module']}",
        f"- recommended_skill_route: {lane['recommended_skill_route_text']}",
        f"- recommended_aim_template: {lane['recommended_aim_template']}",
        f"- primary_goal: {aim.get('primary_goal', '')}",
        f"- optimize_for: {aim.get('optimize_for', '')}",
        f"- metric: {aim.get('target_metric_name', '')} ({aim.get('target_metric_direction', '')})",
        f"- exactness_mode: {aim.get('exactness_mode', 'exact-parity')}",
        f"- shell: {environment['shell']['name']}",
        f"- package_manager: {environment['package_manager']['name']}",
        f"- auto_install_command: {environment['auto_install_command']}",
        f"- known_bottlenecks: {aim.get('known_bottlenecks', '')}",
        f"- suspected_safe_lanes: {aim.get('suspected_safe_lanes', '')}",
        "",
        "## Allowed Mutations",
        "",
    ]
    for item in aim.get("allowed_mutations", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Blocked By Default", ""])
    for item in aim.get("blocked_by_default", []):
        lines.append(f"- {item}")
    Path(workspace["current_contract_md"]).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_evaluator_report(
    workspace: dict[str, str],
    reference: dict[str, Any],
    record: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    keep_text = "keep" if decision["keep"] else f"revert ({decision['rejection_reason']})"
    lines = [
        "# Evaluator Report",
        "",
        f"## {record['label']}",
        "",
        f"- reference_experiment: {reference.get('label', 'baseline')}",
        f"- candidate_experiment: {record['label']}",
        f"- exactness_mode: {decision['candidate_exactness'].get('mode', 'exact-parity')}",
        f"- exactness_passed: {decision['candidate_exactness']['passed']}",
        f"- metric_name: {decision['metric_name']}",
        f"- baseline_value: {decision['baseline_value']}",
        f"- candidate_value: {decision['candidate_value']}",
        f"- improvement: {decision['improvement']}",
        f"- decision: {keep_text}",
        "",
        "## Criteria",
        "",
        "- exactness",
        "- metric improvement",
        "- scope compliance",
        "- stability and reproducibility",
        "",
        "## Notes",
        "",
        f"- candidate_exactness: {json.dumps(decision['candidate_exactness'], ensure_ascii=False)}",
        f"- baseline_exactness: {json.dumps(decision['baseline_exactness'], ensure_ascii=False)}",
    ]
    Path(workspace["evaluator_report_md"]).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_handoff(
    workspace: dict[str, str],
    state: dict[str, Any],
    best: dict[str, Any],
    lane: dict[str, Any],
    last_decision: dict[str, Any] | None = None,
) -> None:
    next_action = state.get("next_action", "resume from status")
    lines = [
        "# Next Handoff",
        "",
        "## Resume Here",
        "",
        f"- status: {state.get('status')}",
        f"- iterations_completed: {state.get('iterations_completed')}",
        f"- consecutive_failures: {state.get('consecutive_failures')}",
        f"- last_experiment: {state.get('last_experiment')}",
        f"- best_experiment: {state.get('best_experiment')}",
        f"- next_action: {next_action}",
        "",
        "## Lane",
        "",
        f"- scenario: {lane['scenario']}",
        f"- target_module: {lane['target_module']}",
        f"- recommended_skill_route: {lane['recommended_skill_route_text']}",
        f"- recommended_aim_template: {lane['recommended_aim_template']}",
        "",
        "## Environment",
        "",
        f"- shell: {best.get('environment', {}).get('shell', {}).get('name') if best else None}",
        f"- package_manager: {best.get('environment', {}).get('package_manager', {}).get('name') if best else None}",
        f"- auto_install_command: {best.get('environment', {}).get('auto_install_command') if best else None}",
        "",
        "## Current Best",
        "",
        f"- label: {best.get('label') if best else None}",
        f"- metrics: {json.dumps(best.get('metrics', {}), ensure_ascii=False) if best else '{}'}",
        "",
        "## Last Decision",
        "",
        f"- decision: {json.dumps(last_decision, ensure_ascii=False) if last_decision else 'none'}",
    ]
    Path(workspace["next_handoff_md"]).write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_state(
    workspace: dict[str, str],
    *,
    status: str,
    last_experiment: str | None,
    best_experiment: str | None,
    keep: bool | None,
    next_action: str,
) -> dict[str, Any]:
    state = load_state(workspace["session_state_json"])
    if last_experiment:
        state["last_experiment"] = last_experiment
    if best_experiment is not None:
        state["best_experiment"] = best_experiment
    if keep is not None:
        state["iterations_completed"] = int(state.get("iterations_completed", 0)) + 1
        if keep:
            state["consecutive_failures"] = 0
        else:
            state["consecutive_failures"] = int(state.get("consecutive_failures", 0)) + 1
    state["status"] = status
    state["next_action"] = next_action
    write_json(workspace["session_state_json"], state)
    return state


def prepare_context(aim_file: str) -> tuple[dict[str, Any], Path, dict[str, str]]:
    aim_path = Path(aim_file).resolve()
    aim = read_aim(aim_path)
    project_root = repo_root_from_aim(aim, aim_path)
    ensure_git_repo(project_root, bool(aim.get("git_required", False)))
    workspace = apply_workspace_overrides(initialize_workspace(project_root), aim, project_root)
    return aim, project_root, workspace


def select_reference(workspace: dict[str, str]) -> dict[str, Any]:
    best = load_json(workspace["best_result_json"])
    if best:
        return best
    baseline = load_json(workspace["baseline_snapshot_json"])
    if baseline:
        return baseline
    raise RuntimeError("baseline snapshot is missing; run baseline first")


def execute_candidate(
    aim: dict[str, Any],
    project_root: Path,
    workspace: dict[str, str],
    label: str,
    promote: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    reference = select_reference(workspace)
    write_contract_doc(workspace, aim, label, "candidate")
    record = run_contract(aim, project_root, phase="candidate", label=label)
    decision = compare_runs(
        reference,
        record,
        metric_name=str(aim["target_metric_name"]),
        metric_direction=str(aim["target_metric_direction"]),
        exactness_policy=exactness_policy_from_aim(aim),
    )
    log_session_artifacts(workspace, record, decision)
    if promote and decision["keep"]:
        promoted = {**record, "decision": decision}
        write_json(workspace["best_result_json"], promoted)
    write_evaluator_report(workspace, reference, record, decision)
    best = load_json(workspace["best_result_json"])
    next_action = "propose next bounded experiment" if decision["keep"] else "revert or adjust candidate and rerun"
    state = update_state(
        workspace,
        status="candidate_kept" if decision["keep"] else "candidate_rejected",
        last_experiment=record["label"],
        best_experiment=(best.get("label") if best else reference.get("label")),
        keep=decision["keep"],
        next_action=next_action,
    )
    write_handoff(workspace, state, best or reference, resolve_scenario_lane(aim), decision)
    return record, decision, state


def handle_init(args: argparse.Namespace) -> int:
    aim, project_root, workspace = prepare_context(args.aim)
    environment = detect_runtime_environment(project_root)
    lane = resolve_scenario_lane(aim)
    write_skill_route_plan(workspace, lane)
    write_contract_doc(workspace, aim, "init", "init")
    payload = {
        "label": "init",
        "phase": "init",
        "lane": lane,
        "timestamp": utc_now(),
        "git_revision": git_revision(project_root),
        "metrics": {},
        "exactness": {"passed": False, "mismatch_count": 0},
    }
    log_session_artifacts(workspace, payload)
    state = update_state(
        workspace,
        status="initialized",
        last_experiment="init",
        best_experiment=None,
        keep=None,
        next_action="run baseline",
    )
    write_handoff(workspace, state, {}, lane)
    print(
        json.dumps(
            {
                "workspace": workspace,
                "project_root": str(project_root),
                "lane": lane,
                "environment": environment,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def handle_baseline(args: argparse.Namespace) -> int:
    aim, project_root, workspace = prepare_context(args.aim)
    lane = resolve_scenario_lane(aim)
    write_skill_route_plan(workspace, lane)
    write_contract_doc(workspace, aim, args.label or "baseline", "baseline")
    record = run_contract(aim, project_root, phase="baseline", label=args.label or "baseline")
    policy = exactness_policy_from_aim(aim)
    baseline_decision = {
        "metric_name": str(aim["target_metric_name"]),
        "metric_direction": str(aim["target_metric_direction"]),
        "baseline_value": record["metrics"].get(str(aim["target_metric_name"])),
        "candidate_value": record["metrics"].get(str(aim["target_metric_name"])),
        "improvement": 0.0,
        "relative_improvement": 0.0,
        "baseline_exactness": compare_runs(
            record,
            record,
            metric_name=str(aim["target_metric_name"]),
            metric_direction=str(aim["target_metric_direction"]),
            exactness_policy=policy,
        )["baseline_exactness"],
        "candidate_exactness": compare_runs(
            record,
            record,
            metric_name=str(aim["target_metric_name"]),
            metric_direction=str(aim["target_metric_direction"]),
            exactness_policy=policy,
        )["candidate_exactness"],
        "keep": True,
        "rejection_reason": None,
    }
    if not baseline_decision["baseline_exactness"].get("passed", False):
        raise RuntimeError("baseline exactness check failed; baseline cannot be trusted")
    write_json(workspace["baseline_snapshot_json"], record)
    write_json(workspace["best_result_json"], record)
    log_session_artifacts(workspace, record)
    write_evaluator_report(workspace, record, record, baseline_decision)
    state = update_state(
        workspace,
        status="baseline_ready",
        last_experiment=record["label"],
        best_experiment=record["label"],
        keep=None,
        next_action="run loop or candidate",
    )
    write_handoff(workspace, state, record, lane, baseline_decision)
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


def handle_candidate(args: argparse.Namespace) -> int:
    aim, project_root, workspace = prepare_context(args.aim)
    label = args.label or f"candidate-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    record, decision, _state = execute_candidate(aim, project_root, workspace, label, promote=True)
    print(json.dumps({"record": record, "decision": decision}, ensure_ascii=False, indent=2))
    return 0


def handle_status(args: argparse.Namespace) -> int:
    aim, project_root, workspace = prepare_context(args.aim)
    baseline = load_json(workspace["baseline_snapshot_json"])
    best = load_json(workspace["best_result_json"])
    state = load_state(workspace["session_state_json"])
    environment = detect_runtime_environment(project_root)
    lane = resolve_scenario_lane(aim)
    write_skill_route_plan(workspace, lane)
    payload = {
        "project_root": str(project_root),
        "scenario": lane["scenario"],
        "target_module": lane["target_module"],
        "recommended_skill_route": lane["recommended_skill_route"],
        "recommended_aim_template": lane["recommended_aim_template"],
        "lane": lane,
        "environment": environment,
        "baseline_available": bool(baseline),
        "best_available": bool(best),
        "baseline": baseline,
        "best": best,
        "state": state,
        "workspace": workspace,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def handle_evaluate(args: argparse.Namespace) -> int:
    aim, project_root, workspace = prepare_context(args.aim)
    label = args.label or f"evaluate-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    record, decision, state = execute_candidate(aim, project_root, workspace, label, promote=False)
    write_handoff(
        workspace,
        state,
        load_json(workspace["best_result_json"]) or select_reference(workspace),
        resolve_scenario_lane(aim),
        decision,
    )
    print(json.dumps({"record": record, "decision": decision}, ensure_ascii=False, indent=2))
    return 0


def handle_collect_env(args: argparse.Namespace) -> int:
    aim, project_root, workspace = prepare_context(args.aim)
    lane = resolve_scenario_lane(aim)
    payload = {
        "project_root": str(project_root),
        "scenario": lane["scenario"],
        "target_module": lane["target_module"],
        "recommended_skill_route": lane["recommended_skill_route"],
        "recommended_aim_template": lane["recommended_aim_template"],
        "lane": lane,
        "environment": detect_runtime_environment(project_root),
        "workspace": workspace,
    }
    if args.output:
        write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def handle_handoff(args: argparse.Namespace) -> int:
    aim, _project_root, workspace = prepare_context(args.aim)
    state = load_state(workspace["session_state_json"])
    best = load_json(workspace["best_result_json"])
    last_line = None
    experiment_log = Path(workspace["experiment_log_jsonl"])
    if experiment_log.exists():
        lines = [line for line in experiment_log.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            last_line = json.loads(lines[-1]).get("decision")
    write_handoff(workspace, state, best, resolve_scenario_lane(aim), last_line)
    print(Path(workspace["next_handoff_md"]).read_text(encoding="utf-8"))
    return 0


def handle_loop(args: argparse.Namespace) -> int:
    aim, project_root, workspace = prepare_context(args.aim)
    baseline = load_json(workspace["baseline_snapshot_json"])
    if not baseline:
        return handle_baseline(args)
    label = args.label or f"loop-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    record, decision, _state = execute_candidate(aim, project_root, workspace, label, promote=True)
    print(json.dumps({"record": record, "decision": decision}, ensure_ascii=False, indent=2))
    return 0


def handle_autopilot(args: argparse.Namespace) -> int:
    aim, project_root, workspace = prepare_context(args.aim)
    lane = resolve_scenario_lane(aim)
    write_skill_route_plan(workspace, lane)

    baseline = load_json(workspace["baseline_snapshot_json"])
    actions: list[dict[str, Any]] = []
    if not baseline:
        baseline_label = args.baseline_label or "baseline"
        baseline_record = run_contract(aim, project_root, phase="baseline", label=baseline_label)
        write_json(workspace["baseline_snapshot_json"], baseline_record)
        write_json(workspace["best_result_json"], baseline_record)
        log_session_artifacts(workspace, baseline_record)
        state = update_state(
            workspace,
            status="baseline_ready",
            last_experiment=baseline_record["label"],
            best_experiment=baseline_record["label"],
            keep=None,
            next_action="run autopilot candidate iterations",
        )
        write_handoff(workspace, state, baseline_record, lane)
        actions.append({"phase": "baseline", "label": baseline_record["label"]})

    iterations = max(1, int(args.iterations))
    decisions: list[dict[str, Any]] = []
    for idx in range(iterations):
        label = f"{args.label_prefix}-{idx + 1:03d}"
        _record, decision, _state = execute_candidate(aim, project_root, workspace, label, promote=True)
        decisions.append({"label": label, "keep": decision["keep"], "improvement": decision["improvement"]})

    payload = {
        "lane": lane,
        "actions": actions,
        "iterations": iterations,
        "decisions": decisions,
        "workspace": workspace,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="auto-profiling")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--aim", default="aim.md")
    init_parser.set_defaults(handler=handle_init)

    baseline_parser = subparsers.add_parser("baseline")
    baseline_parser.add_argument("--aim", default="aim.md")
    baseline_parser.add_argument("--label")
    baseline_parser.set_defaults(handler=handle_baseline)

    candidate_parser = subparsers.add_parser("candidate")
    candidate_parser.add_argument("--aim", default="aim.md")
    candidate_parser.add_argument("--label")
    candidate_parser.set_defaults(handler=handle_candidate)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--aim", default="aim.md")
    evaluate_parser.add_argument("--label")
    evaluate_parser.set_defaults(handler=handle_evaluate)

    handoff_parser = subparsers.add_parser("handoff")
    handoff_parser.add_argument("--aim", default="aim.md")
    handoff_parser.set_defaults(handler=handle_handoff)

    loop_parser = subparsers.add_parser("loop")
    loop_parser.add_argument("--aim", default="aim.md")
    loop_parser.add_argument("--label")
    loop_parser.set_defaults(handler=handle_loop)

    autopilot_parser = subparsers.add_parser("autopilot")
    autopilot_parser.add_argument("--aim", default="aim.md")
    autopilot_parser.add_argument("--iterations", type=int, default=1)
    autopilot_parser.add_argument("--label-prefix", default="auto")
    autopilot_parser.add_argument("--baseline-label")
    autopilot_parser.set_defaults(handler=handle_autopilot)

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--aim", default="aim.md")
    status_parser.set_defaults(handler=handle_status)

    collect_env_parser = subparsers.add_parser("collect-env")
    collect_env_parser.add_argument("--aim", default="aim.md")
    collect_env_parser.add_argument("--output")
    collect_env_parser.set_defaults(handler=handle_collect_env)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
