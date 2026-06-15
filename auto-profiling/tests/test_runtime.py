import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

RUNTIME_ROOT = Path(__file__).resolve().parents[1]
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

from runner import (
    auto_install_command,
    build_parser,
    detect_runtime_environment,
    detect_package_manager,
    detect_preferred_shell,
    git_commit_current_state,
    initialize_workspace,
    load_skill_routes,
    parse_aim_markdown,
    run_required,
    shell_result,
    validate_aim,
    build_doctor_report,
)
from scorer import compare_runs


class AimParserTests(unittest.TestCase):
    def test_parse_scalars_and_lists(self):
        text = """
## 1. Mission

- project_name: demo
- target_metric_name: p95_ms
- target_metric_direction: lower_is_better

## 6. Allowed Mutation Surface

- allowed_mutations:
  - profiling instrumentation
  - copy reduction
- blocked_by_default:
  - quantization
"""
        data = parse_aim_markdown(text)
        self.assertEqual(data["project_name"], "demo")
        self.assertEqual(data["target_metric_name"], "p95_ms")
        self.assertEqual(
            data["allowed_mutations"],
            ["profiling instrumentation", "copy reduction"],
        )
        self.assertEqual(data["blocked_by_default"], ["quantization"])

    def test_blank_scalar_values_stay_unset_instead_of_empty_list(self):
        text = """
- require_logic_equivalence:
- git_required:
- project_name: demo
"""
        data = parse_aim_markdown(text)
        self.assertNotIn("require_logic_equivalence", data)
        self.assertNotIn("git_required", data)
        # Defaults must apply instead of bool([]) == False.
        self.assertTrue(bool(data.get("require_logic_equivalence", True)))
        self.assertEqual(data["project_name"], "demo")

    def test_key_seen_as_scalar_then_list_does_not_crash(self):
        # A malformed document that repeats a key first as a scalar then as a
        # list header must not raise AttributeError on str.append.
        text = "- foo: scalarval\n- foo:\n  - item1\n  - item2\n"
        data = parse_aim_markdown(text)
        self.assertEqual(data["foo"], ["item1", "item2"])


class AimSchemaTests(unittest.TestCase):
    def test_validate_aim_rejects_missing_required_fields(self):
        with self.assertRaises(ValueError) as ctx:
            validate_aim({"scenario": "llm-serving"})
        self.assertIn("baseline_run_command", str(ctx.exception))

    def test_skill_routes_loaded_from_config(self):
        routes = load_skill_routes()
        self.assertIn("llm-serving", routes["scenarios"])
        self.assertEqual(
            routes["scenarios"]["cuda-kernel"]["target_module"],
            "cuda-kernel-opt-skill",
        )


class WorkspaceInitTests(unittest.TestCase):
    def test_initialize_workspace_creates_runtime_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = initialize_workspace(root)
            expected = {
                "state_dir",
                "experiment_log_md",
                "experiment_log_jsonl",
                "baseline_snapshot_json",
                "best_result_json",
                "session_state_json",
                "task_plan_md",
                "findings_md",
                "progress_md",
                "worklog_md",
                "current_contract_md",
                "evaluator_report_md",
                "next_handoff_md",
                "next_candidate_md",
                "next_candidate_json",
                "doctor_report_md",
                "doctor_report_json",
            }
            self.assertTrue(expected.issubset(paths.keys()))
            for key in expected:
                self.assertTrue(Path(paths[key]).exists(), key)

    def test_runner_baseline_and_status_end_to_end(self):
        runtime_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            subprocess.run(
                ["git", "init"],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            )
            aim_path = Path(tmp) / "aim.md"
            baseline_cmd = (
                'python3 -c "from pathlib import Path; import json; '
                "p=Path('.auto-profiling/metric.json'); "
                "p.parent.mkdir(parents=True, exist_ok=True); "
                "p.write_text(json.dumps({'metrics': {'p95_ms': 10.0}}), encoding='utf-8')\""
            )
            exactness_cmd = (
                'python3 -c "from pathlib import Path; import json; '
                "p=Path('.auto-profiling/exact.json'); "
                "p.parent.mkdir(parents=True, exist_ok=True); "
                "p.write_text(json.dumps({'passed': True, 'mismatch_count': 0}), encoding='utf-8')\""
            )
            aim_path.write_text(
                "\n".join(
                    [
                        "# Auto-Profiling Aim",
                        "",
                        "## 1. Mission",
                        "",
                        "- scenario: llm-serving",
                        "- project_name: demo",
                        "- primary_goal: baseline",
                        "- optimize_for: latency",
                        "- target_metric_name: p95_ms",
                        "- target_metric_direction: lower_is_better",
                        "- target_sla: 10",
                        "",
                        "## 2. Scope",
                        "",
                        f"- target_repo_path: {repo}",
                        "- target_entrypoints: main.py",
                        "- baseline_files_allowed_to_change: runner.py",
                        "- files_never_touch: model.bin",
                        "",
                        "## 3. Environment",
                        "",
                        "- os: macos",
                        "- hardware: cpu",
                        "- accelerator: none",
                        "- python_env_command:",
                        "- git_required: true",
                        "- install_command:",
                        "- warmup_command:",
                        "",
                        "## 4. Baseline Execution",
                        "",
                        "- baseline_setup_command:",
                        f"- baseline_run_command: {baseline_cmd}",
                        '- baseline_profile_command: python3 -c "print(\'profile-ok\')"',
                        "- metric_output_path: .auto-profiling/metric.json",
                        "- exactness_output_path: .auto-profiling/exact.json",
                        "",
                        "## 5. Exactness Contract",
                        "",
                        "- exactness_mode: exact-parity",
                        "- reference_path_description: reference",
                        "- golden_input_location: inputs.json",
                        "- golden_output_location: outputs.json",
                        f"- exactness_check_command: {exactness_cmd}",
                        "- deterministic_requirements: fixed",
                        "- cache_semantics_requirements: exact",
                        "- request_isolation_requirements: exact",
                        "",
                        "## 6. Allowed Mutation Surface",
                        "",
                        "- allowed_mutations:",
                        "  - profiling instrumentation",
                        "- blocked_by_default:",
                        "  - quantization",
                        "",
                        "## 7. Experiment Budget",
                        "",
                        "- max_iterations_per_session: 3",
                        "- max_runtime_per_experiment: 60",
                        "- stop_after_consecutive_failures: 2",
                        "- require_revert_on_failure: true",
                        "",
                        "## 8. Logging",
                        "",
                        "- experiment_log_path:",
                        "- best_result_path:",
                        "- progress_doc_path:",
                        "- worklog_doc_path:",
                        "- save_failed_runs: true",
                        "",
                        "## 9. Human Override",
                        "",
                        "- allow_non_zero_drift: false",
                        "- override_reason:",
                        "",
                        "## 10. Notes",
                        "",
                        "- additional_constraints:",
                        "- business_context:",
                        "- known_bottlenecks:",
                        "- suspected_safe_lanes:",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            baseline = subprocess.run(
                [sys.executable, str(runtime_root / "runner.py"), "baseline", "--aim", str(aim_path)],
                cwd=runtime_root,
                check=True,
                capture_output=True,
                text=True,
            )
            status = subprocess.run(
                [sys.executable, str(runtime_root / "runner.py"), "status", "--aim", str(aim_path)],
                cwd=runtime_root,
                check=True,
                capture_output=True,
                text=True,
            )
            baseline_payload = json.loads(baseline.stdout)
            status_payload = json.loads(status.stdout)
            self.assertTrue(baseline_payload["exactness"]["passed"])
            self.assertEqual(baseline_payload["metrics"]["p95_ms"], 10.0)
            self.assertTrue(status_payload["baseline_available"])
            self.assertEqual(status_payload["scenario"], "llm-serving")
            self.assertEqual(status_payload["target_module"], "llm-serving-opt-skill")
            self.assertEqual(
                status_payload["recommended_skill_route"],
                [
                    "auto-profiling",
                    "llm-serving-opt-skill",
                    "serving-benchmark-skill",
                ],
            )
            self.assertTrue((repo / ".auto-profiling" / "experiment_log.md").exists())
            self.assertTrue((repo / ".auto-profiling" / "worklog.md").exists())
            current_contract = (repo / ".auto-profiling" / "current_contract.md").read_text(encoding="utf-8")
            self.assertIn("- scenario: llm-serving", current_contract)
            self.assertIn("- target_module: llm-serving-opt-skill", current_contract)
            self.assertIn(
                "- recommended_skill_route: auto-profiling -> llm-serving-opt-skill -> serving-benchmark-skill",
                current_contract,
            )
            next_handoff = (repo / ".auto-profiling" / "next_handoff.md").read_text(encoding="utf-8")
            self.assertIn("- scenario: llm-serving", next_handoff)
            self.assertIn("- target_module: llm-serving-opt-skill", next_handoff)
            self.assertIn(
                "- recommended_skill_route: auto-profiling -> llm-serving-opt-skill -> serving-benchmark-skill",
                next_handoff,
            )
            subprocess.run(
                [sys.executable, str(runtime_root / "runner.py"), "candidate", "--aim", str(aim_path), "--label", "exp-001"],
                cwd=runtime_root,
                check=True,
                capture_output=True,
                text=True,
            )
            # After a rejected candidate is reverted to the best commit,
            # next_candidate.md reflects the baseline state.
            self.assertTrue((repo / ".auto-profiling" / "next_candidate.md").exists())


class EnvironmentDetectionTests(unittest.TestCase):
    @patch("runner.shutil.which")
    def test_shell_falls_back_to_bash_when_zsh_missing(self, mock_which):
        mapping = {"zsh": None, "bash": "/bin/bash"}
        mock_which.side_effect = lambda name: mapping.get(name)
        shell = detect_preferred_shell()
        self.assertEqual(shell["name"], "bash")
        self.assertEqual(shell["path"], "/bin/bash")

    @patch("runner.shutil.which")
    def test_detect_package_manager_prefers_pip_when_uv_missing(self, mock_which):
        mapping = {"uv": None}
        mock_which.side_effect = lambda name: mapping.get(name)
        package_manager = detect_package_manager()
        self.assertEqual(package_manager["name"], "pip")

    @patch("runner.shutil.which")
    def test_auto_install_command_uses_pip_when_uv_missing(self, mock_which):
        mock_which.side_effect = lambda name: None if name == "uv" else "/usr/bin/python3"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0.1.0'\n", encoding="utf-8")
            command = auto_install_command(root)
        self.assertIn("-m pip install -e .", command)

    def test_detect_runtime_environment_collects_richer_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            environment = detect_runtime_environment(root)
        self.assertIn("python_version", environment)
        self.assertIn("platform", environment)
        self.assertIn("machine", environment)
        self.assertIn("tool_paths", environment)


class ScoringTests(unittest.TestCase):
    def test_exactness_failure_rejects_even_when_metric_improves(self):
        baseline = {
            "metrics": {"p95_ms": 10.0},
            "exactness": {"passed": True, "mismatch_count": 0},
        }
        candidate = {
            "metrics": {"p95_ms": 7.0},
            "exactness": {"passed": False, "mismatch_count": 2},
        }
        result = compare_runs(
            baseline,
            candidate,
            metric_name="p95_ms",
            metric_direction="lower_is_better",
        )
        self.assertFalse(result["keep"])
        self.assertEqual(result["rejection_reason"], "exactness_failed")

    def test_lower_is_better_metric_keeps_better_exact_candidate(self):
        baseline = {
            "metrics": {"p95_ms": 10.0},
            "exactness": {"passed": True, "mismatch_count": 0},
        }
        candidate = {
            "metrics": {"p95_ms": 8.5},
            "exactness": {"passed": True, "mismatch_count": 0},
        }
        result = compare_runs(
            baseline,
            candidate,
            metric_name="p95_ms",
            metric_direction="lower_is_better",
        )
        self.assertTrue(result["keep"])
        self.assertAlmostEqual(result["improvement"], 1.5)

    def test_higher_is_better_metric(self):
        baseline = {
            "metrics": {"throughput": 100.0},
            "exactness": {"passed": True, "mismatch_count": 0},
        }
        candidate = {
            "metrics": {"throughput": 120.0},
            "exactness": {"passed": True, "mismatch_count": 0},
        }
        result = compare_runs(
            baseline,
            candidate,
            metric_name="throughput",
            metric_direction="higher_is_better",
        )
        self.assertTrue(result["keep"])
        self.assertAlmostEqual(result["improvement"], 20.0)

    def test_tolerance_mode_accepts_small_numeric_error(self):
        baseline = {
            "metrics": {"p95_ms": 100.0},
            "exactness": {"passed": True, "mismatch_count": 0},
        }
        candidate = {
            "metrics": {"p95_ms": 95.0},
            "exactness": {
                "passed": False,
                "logic_equivalent": True,
                "algorithm_equivalent": True,
                "mismatch_count": 1,
                "max_abs_error": 1e-6,
                "max_rel_error": 1e-6,
            },
        }
        result = compare_runs(
            baseline,
            candidate,
            metric_name="p95_ms",
            metric_direction="lower_is_better",
            exactness_policy={
                "mode": "bounded-tolerance",
                "abs_tolerance": 1e-5,
                "rel_tolerance": 1e-5,
            },
        )
        self.assertTrue(result["keep"])
        self.assertEqual(result["candidate_exactness"]["mode"], "bounded-tolerance")

    def test_tolerance_mode_rejects_mismatch_without_error_bounds(self):
        baseline = {
            "metrics": {"p95_ms": 100.0},
            "exactness": {"passed": True, "mismatch_count": 0},
        }
        candidate = {
            "metrics": {"p95_ms": 95.0},
            "exactness": {"passed": False, "mismatch_count": 5},
        }
        result = compare_runs(
            baseline,
            candidate,
            metric_name="p95_ms",
            metric_direction="lower_is_better",
            exactness_policy={
                "mode": "bounded-tolerance",
                "abs_tolerance": 1e-5,
                "rel_tolerance": 1e-5,
            },
        )
        self.assertFalse(result["keep"])
        self.assertEqual(result["rejection_reason"], "exactness_failed")


class GitCommitTests(unittest.TestCase):
    def _init_repo(self, repo: Path) -> None:
        for cmd in (
            ["git", "init"],
            ["git", "config", "user.email", "a@b.c"],
            ["git", "config", "user.name", "tester"],
            ["git", "config", "commit.gpgsign", "false"],
        ):
            subprocess.run(cmd, cwd=repo, check=True, capture_output=True, text=True)

    def test_commit_message_with_shell_metachars_is_not_injected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            self._init_repo(repo)
            (repo / "f.txt").write_text("x", encoding="utf-8")
            sentinel = repo / "PWNED"
            evil = 'exp"; touch PWNED; echo "'
            result = git_commit_current_state(repo, f"keep {evil}")
            self.assertEqual(result["exit_code"], 0, result)
            # Injected `touch PWNED` must never have executed.
            self.assertFalse(sentinel.exists(), "shell injection executed")
            subject = subprocess.run(
                ["git", "log", "-1", "--pretty=%s"],
                cwd=repo,
                capture_output=True,
                text=True,
            ).stdout.strip()
            self.assertIn(evil, subject)


class CommandGuardTests(unittest.TestCase):
    def test_shell_result_reports_timeout_without_hanging(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = shell_result(
                f"{sys.executable} -c \"import time; time.sleep(2)\"",
                cwd=Path(tmp),
                timeout_seconds=1,
            )
        self.assertEqual(result["exit_code"], 124)
        self.assertTrue(result["timed_out"])
        self.assertIn("timed out", result["stderr"])

    @unittest.skipUnless(hasattr(os, "killpg"), "requires POSIX process groups")
    def test_shell_result_timeout_kills_child_process_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "orphan_marker"
            result = shell_result(
                f"(sleep 2 && touch {marker}) & sleep 5",
                cwd=Path(tmp),
                timeout_seconds=1,
            )
            self.assertTrue(result["timed_out"])
            time.sleep(2.5)
            self.assertFalse(
                marker.exists(),
                "backgrounded child survived the timeout and wrote its marker",
            )

    def test_run_required_raises_timeout_specific_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError) as ctx:
                run_required(
                    f"{sys.executable} -c \"import time; time.sleep(2)\"",
                    cwd=Path(tmp),
                    timeout_seconds=1,
                )
        self.assertIn("command timed out", str(ctx.exception))



class DoctorReportTests(unittest.TestCase):
    def test_doctor_report_surfaces_user_facing_warnings(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            workspace = initialize_workspace(repo)
            aim = {
                "scenario": "llm-serving",
                "target_repo_path": str(repo),
                "git_required": False,
                "baseline_run_command": "python3 bench.py",
                "baseline_profile_command": "python3 profile.py",
                "exactness_check_command": "python3 check.py",
                "metric_output_path": ".auto-profiling/metric.json",
                "exactness_output_path": ".auto-profiling/exactness.json",
                "target_metric_name": "p95_ms",
                "target_metric_direction": "lower_is_better",
                "exactness_mode": "exact-parity",
                "allowed_mutations": ["scheduler tuning"],
                "blocked_by_default": ["algorithmic behavior change"],
            }
            report = build_doctor_report(aim, repo, workspace)
        self.assertEqual(report["status"], "warn")
        self.assertTrue(any(check["name"] == "git_repo" and check["status"] == "warn" for check in report["checks"]))
        self.assertEqual(report["lane"]["target_module"], "llm-serving-opt-skill")

    def test_doctor_report_requires_reasoning_quality_contract_for_reasoning_scenario(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            workspace = initialize_workspace(repo)
            aim = {
                "scenario": "reasoning-task",
                "target_repo_path": str(repo),
                "git_required": False,
                "baseline_run_command": "python3 bench.py",
                "baseline_profile_command": "python3 profile.py",
                "exactness_check_command": "python3 check.py",
                "metric_output_path": ".auto-profiling/metric.json",
                "exactness_output_path": ".auto-profiling/exactness.json",
                "target_metric_name": "p95_ms",
                "target_metric_direction": "lower_is_better",
                "exactness_mode": "exact-parity",
                "allowed_mutations": ["prompt packing"],
                "blocked_by_default": ["answer quality regression"],
            }
            report = build_doctor_report(aim, repo, workspace)
        reasoning_check = next(check for check in report["checks"] if check["name"] == "reasoning_quality_contract")
        self.assertEqual(reasoning_check["status"], "fail")
        self.assertEqual(report["lane"]["scenario"], "reasoning-task")

    def test_doctor_report_accepts_complete_reasoning_quality_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            workspace = initialize_workspace(repo)
            aim = {
                "scenario": "reasoning-task",
                "target_repo_path": str(repo),
                "git_required": False,
                "baseline_run_command": "python3 bench.py",
                "baseline_profile_command": "python3 profile.py",
                "exactness_check_command": "python3 check.py",
                "metric_output_path": ".auto-profiling/metric.json",
                "exactness_output_path": ".auto-profiling/exactness.json",
                "target_metric_name": "p95_ms",
                "target_metric_direction": "lower_is_better",
                "exactness_mode": "exact-parity",
                "reasoning_task_type": "tool_agent",
                "reasoning_quality_metric": "tool_trace_parity",
                "reasoning_quality_threshold": "task_success_rate >= baseline",
                "golden_trace_path": "evals/golden/tool_traces.jsonl",
                "required_output_properties": ["preserves valid tool-call arguments"],
                "allowed_mutations": ["tool concurrency"],
                "blocked_by_default": ["invalid tool calls"],
            }
            report = build_doctor_report(aim, repo, workspace)
        reasoning_check = next(check for check in report["checks"] if check["name"] == "reasoning_quality_contract")
        self.assertEqual(reasoning_check["status"], "pass")

class HarnessCommandTests(unittest.TestCase):
    def test_parser_exposes_loop_evaluate_and_handoff(self):
        parser = build_parser()
        choices = parser._subparsers._group_actions[0].choices
        self.assertIn("loop", choices)
        self.assertIn("evaluate", choices)
        self.assertIn("handoff", choices)
        self.assertIn("collect-env", choices)
        self.assertIn("doctor", choices)


if __name__ == "__main__":
    unittest.main()
