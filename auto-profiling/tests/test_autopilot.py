import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class AutopilotRuntimeTests(unittest.TestCase):
    def test_autopilot_bootstraps_baseline_and_runs_iterations(self):
        runtime_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)

            metric_cmd = (
                'python3 -c "from pathlib import Path; import json; '
                "p=Path('.auto-profiling/metric.json'); p.parent.mkdir(parents=True, exist_ok=True); "
                "p.write_text(json.dumps({'metrics': {'p95_ms': 10.0}}), encoding='utf-8')\""
            )
            exact_cmd = (
                'python3 -c "from pathlib import Path; import json; '
                "p=Path('.auto-profiling/exactness.json'); p.parent.mkdir(parents=True, exist_ok=True); "
                "p.write_text(json.dumps({'exactness': {'passed': True, 'mismatch_count': 0}}), encoding='utf-8')\""
            )

            aim = Path(tmp) / "aim.md"
            aim.write_text(
                "\n".join(
                    [
                        "# Auto-Profiling Aim",
                        "",
                        "## 1. Mission",
                        "",
                        "- scenario: llm-serving",
                        "- project_name: autopilot-test",
                        "- primary_goal: iterate automatically",
                        "- optimize_for: latency",
                        "- target_metric_name: p95_ms",
                        "- target_metric_direction: lower_is_better",
                        "",
                        "## 2. Scope",
                        "",
                        f"- target_repo_path: {repo}",
                        "- target_entrypoints: service.py",
                        "",
                        "## 3. Environment",
                        "",
                        "- git_required: true",
                        "- install_command:",
                        "",
                        "## 4. Baseline Execution",
                        "",
                        f"- baseline_run_command: {metric_cmd}",
                        '- baseline_profile_command: python3 -c "print(\'profile\')"',
                        "- metric_output_path: .auto-profiling/metric.json",
                        "- exactness_output_path: .auto-profiling/exactness.json",
                        "",
                        "## 5. Exactness Contract",
                        "",
                        f"- exactness_check_command: {exact_cmd}",
                        "- exactness_mode: exact-parity",
                        "",
                        "## 6. Allowed Mutation Surface",
                        "",
                        "- allowed_mutations:",
                        "  - scheduler tuning",
                        "- blocked_by_default:",
                        "  - unsafe precision drift",
                        "",
                        "## 7. Experiment Budget",
                        "",
                        "- max_iterations_per_session: 1",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(runtime_root / "runner.py"),
                    "autopilot",
                    "--aim",
                    str(aim),
                    "--iterations",
                    "2",
                    "--label-prefix",
                    "auto",
                ],
                cwd=runtime_root,
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(result.stdout)
            self.assertEqual(payload["iterations"], 1)
            self.assertEqual(len(payload["decisions"]), 1)
            self.assertTrue((repo / ".auto-profiling" / "skill_route_plan.md").exists())

    def test_autopilot_stops_after_configured_consecutive_failures(self):
        runtime_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)

            driver = Path(tmp) / "driver.py"
            driver.write_text(
                "\n".join(
                    [
                        "import argparse",
                        "import json",
                        "from pathlib import Path",
                        "",
                        "parser = argparse.ArgumentParser()",
                        "parser.add_argument('--repo', required=True)",
                        "parser.add_argument('--mode', choices=['run', 'exact'], required=True)",
                        "args = parser.parse_args()",
                        "",
                        "repo = Path(args.repo)",
                        "state = repo / '.auto-profiling' / 'phase-count.json'",
                        "state.parent.mkdir(parents=True, exist_ok=True)",
                        "if state.exists():",
                        "    payload = json.loads(state.read_text(encoding='utf-8'))",
                        "else:",
                        "    payload = {'count': 0}",
                        "",
                        "if args.mode == 'run':",
                        "    payload['count'] += 1",
                        "    state.write_text(json.dumps(payload), encoding='utf-8')",
                        "    metric = repo / '.auto-profiling' / 'metric.json'",
                        "    metric.write_text(json.dumps({'metrics': {'p95_ms': 10.0 + payload['count']}}), encoding='utf-8')",
                        "else:",
                        "    passed = payload['count'] == 1",
                        "    exactness = repo / '.auto-profiling' / 'exactness.json'",
                        "    exactness.write_text(json.dumps({'exactness': {'passed': passed, 'mismatch_count': 0 if passed else 1}}), encoding='utf-8')",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            aim = Path(tmp) / "aim.md"
            run_cmd = f'{sys.executable} {driver} --repo {repo} --mode run'
            exact_cmd = f'{sys.executable} {driver} --repo {repo} --mode exact'
            aim.write_text(
                "\n".join(
                    [
                        "# Auto-Profiling Aim",
                        "",
                        "## 1. Mission",
                        "",
                        "- scenario: llm-serving",
                        "- project_name: autopilot-stop-test",
                        "- primary_goal: iterate automatically",
                        "- optimize_for: latency",
                        "- target_metric_name: p95_ms",
                        "- target_metric_direction: lower_is_better",
                        "",
                        "## 2. Scope",
                        "",
                        f"- target_repo_path: {repo}",
                        "- target_entrypoints: service.py",
                        "",
                        "## 3. Environment",
                        "",
                        "- git_required: true",
                        "- install_command:",
                        "",
                        "## 4. Baseline Execution",
                        "",
                        f"- baseline_run_command: {run_cmd}",
                        '- baseline_profile_command: python3 -c "print(\'profile\')"',
                        "- metric_output_path: .auto-profiling/metric.json",
                        "- exactness_output_path: .auto-profiling/exactness.json",
                        "",
                        "## 5. Exactness Contract",
                        "",
                        f"- exactness_check_command: {exact_cmd}",
                        "- exactness_mode: exact-parity",
                        "",
                        "## 6. Allowed Mutation Surface",
                        "",
                        "- allowed_mutations:",
                        "  - scheduler tuning",
                        "- blocked_by_default:",
                        "  - unsafe precision drift",
                        "",
                        "## 7. Experiment Budget",
                        "",
                        "- max_iterations_per_session: 5",
                        "- stop_after_consecutive_failures: 1",
                        "- require_revert_on_failure: true",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(runtime_root / "runner.py"),
                    "autopilot",
                    "--aim",
                    str(aim),
                    "--iterations",
                    "3",
                    "--label-prefix",
                    "auto",
                ],
                cwd=runtime_root,
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(result.stdout)
            self.assertEqual(payload["iterations"], 1)
            self.assertEqual(len(payload["decisions"]), 1)
            self.assertFalse(payload["decisions"][0]["keep"])

    def test_autopilot_reverts_failed_candidate_when_required(self):
        runtime_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True, capture_output=True, text=True)
            subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=repo, check=True, capture_output=True, text=True)
            tracked = repo / "tracked.txt"
            tracked.write_text("baseline\n", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True, capture_output=True, text=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True, text=True)

            driver = Path(tmp) / "driver.py"
            driver.write_text(
                "\n".join(
                    [
                        "import argparse",
                        "import json",
                        "from pathlib import Path",
                        "",
                        "parser = argparse.ArgumentParser()",
                        "parser.add_argument('--repo', required=True)",
                        "parser.add_argument('--mode', choices=['run', 'exact'], required=True)",
                        "args = parser.parse_args()",
                        "",
                        "repo = Path(args.repo)",
                        "state = repo / '.auto-profiling' / 'phase-count.json'",
                        "state.parent.mkdir(parents=True, exist_ok=True)",
                        "if state.exists():",
                        "    payload = json.loads(state.read_text(encoding='utf-8'))",
                        "else:",
                        "    payload = {'count': 0}",
                        "",
                        "if args.mode == 'run':",
                        "    payload['count'] += 1",
                        "    state.write_text(json.dumps(payload), encoding='utf-8')",
                        "    (repo / 'tracked.txt').write_text(f'candidate-{payload[\"count\"]}\\n', encoding='utf-8')",
                        "    metric = repo / '.auto-profiling' / 'metric.json'",
                        "    metric.write_text(json.dumps({'metrics': {'p95_ms': 10.0 + payload['count']}}), encoding='utf-8')",
                        "else:",
                        "    passed = payload['count'] == 1",
                        "    exactness = repo / '.auto-profiling' / 'exactness.json'",
                        "    exactness.write_text(json.dumps({'exactness': {'passed': passed, 'mismatch_count': 0 if passed else 1}}), encoding='utf-8')",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            aim = Path(tmp) / "aim.md"
            run_cmd = f'{sys.executable} {driver} --repo {repo} --mode run'
            exact_cmd = f'{sys.executable} {driver} --repo {repo} --mode exact'
            aim.write_text(
                "\n".join(
                    [
                        "# Auto-Profiling Aim",
                        "",
                        "## 1. Mission",
                        "",
                        "- scenario: llm-serving",
                        "- project_name: autopilot-revert-test",
                        "- primary_goal: iterate automatically",
                        "- optimize_for: latency",
                        "- target_metric_name: p95_ms",
                        "- target_metric_direction: lower_is_better",
                        "",
                        "## 2. Scope",
                        "",
                        f"- target_repo_path: {repo}",
                        "- target_entrypoints: service.py",
                        "",
                        "## 3. Environment",
                        "",
                        "- git_required: true",
                        "- install_command:",
                        "",
                        "## 4. Baseline Execution",
                        "",
                        f"- baseline_run_command: {run_cmd}",
                        '- baseline_profile_command: python3 -c "print(\'profile\')"',
                        "- metric_output_path: .auto-profiling/metric.json",
                        "- exactness_output_path: .auto-profiling/exactness.json",
                        "",
                        "## 5. Exactness Contract",
                        "",
                        f"- exactness_check_command: {exact_cmd}",
                        "- exactness_mode: exact-parity",
                        "",
                        "## 6. Allowed Mutation Surface",
                        "",
                        "- allowed_mutations:",
                        "  - scheduler tuning",
                        "- blocked_by_default:",
                        "  - unsafe precision drift",
                        "",
                        "## 7. Experiment Budget",
                        "",
                        "- max_iterations_per_session: 3",
                        "- stop_after_consecutive_failures: 1",
                        "- require_revert_on_failure: true",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    str(runtime_root / "runner.py"),
                    "autopilot",
                    "--aim",
                    str(aim),
                    "--iterations",
                    "1",
                    "--label-prefix",
                    "auto",
                ],
                cwd=runtime_root,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(tracked.read_text(encoding="utf-8"), "candidate-1\n")


if __name__ == "__main__":
    unittest.main()
