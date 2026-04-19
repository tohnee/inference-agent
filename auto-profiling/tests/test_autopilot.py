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


if __name__ == "__main__":
    unittest.main()
