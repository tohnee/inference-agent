import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "auto-profiling" / "bootstrap_aim.py"


class BootstrapAimTests(unittest.TestCase):
    def test_generate_e2e_diffusion_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "aim.diffusion.md"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--mode",
                    "e2e",
                    "--profile",
                    "diffusion",
                    "--project-name",
                    "demo-diff",
                    "--target-repo-path",
                    "/tmp/target",
                    "--output",
                    str(out),
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            text = out.read_text(encoding="utf-8")
            self.assertIn("- scenario: e2e-inference", text)
            self.assertIn("- target_metric_name: steps_per_second", text)
            self.assertIn("profile_diffusion.py", text)

    def test_generate_llm_vllm_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "aim.vllm.md"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--mode",
                    "llm-serving",
                    "--profile",
                    "vllm",
                    "--project-name",
                    "demo-vllm",
                    "--target-repo-path",
                    "/tmp/serving",
                    "--output",
                    str(out),
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            text = out.read_text(encoding="utf-8")
            self.assertIn("- scenario: llm-serving", text)
            self.assertIn("--backend vllm", text)
            self.assertIn("- target_metric_name: tpot_ms", text)


if __name__ == "__main__":
    unittest.main()
