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

    def test_generate_cuda_triton_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "aim.triton.md"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--mode",
                    "cuda-kernel",
                    "--profile",
                    "triton",
                    "--project-name",
                    "demo-kernel",
                    "--target-repo-path",
                    "/tmp/kernel",
                    "--output",
                    str(out),
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            text = out.read_text(encoding="utf-8")
            self.assertIn("- scenario: cuda-kernel", text)
            self.assertIn("--backend triton", text)
            self.assertIn("- target_metric_name: latency_ms", text)
            self.assertIn("memory bandwidth", text)

    def test_generate_reasoning_rag_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "aim.reasoning.md"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--mode",
                    "reasoning",
                    "--profile",
                    "rag",
                    "--project-name",
                    "demo-rag",
                    "--target-repo-path",
                    "/tmp/rag-agent",
                    "--output",
                    str(out),
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            text = out.read_text(encoding="utf-8")
            self.assertIn("- scenario: reasoning-task", text)
            self.assertIn("- reasoning_task_type: rag_synthesis", text)
            self.assertIn("- reasoning_quality_metric: citation_fidelity", text)
            self.assertIn("preserves required citations", text)


if __name__ == "__main__":
    unittest.main()
