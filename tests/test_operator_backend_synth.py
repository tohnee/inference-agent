import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "cuda-kernel-opt-skill"
    / "skills"
    / "cuda-optimized-skill"
    / "operator-optimize-loop"
    / "scripts"
    / "operator_backend_synth.py"
)


class OperatorBackendSynthTests(unittest.TestCase):
    def test_generate_triton_scaffold_with_cpu_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--name",
                    "demo_add",
                    "--logic",
                    "elementwise add",
                    "--op-type",
                    "elementwise_add",
                    "--backend",
                    "auto",
                    "--output-dir",
                    str(out),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            payload = json.loads(result.stdout)
            op_dir = out / "demo_add"
            self.assertEqual(payload["backend"], "triton")
            self.assertTrue((op_dir / "cpu_reference.py").exists())
            self.assertTrue((op_dir / "kernel_triton.py").exists())
            self.assertTrue((op_dir / "benchmark_harness.py").exists())
            self.assertTrue((op_dir / "manifest.json").exists())

    def test_generate_cuda_scaffold_for_large_matmul(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--name",
                    "demo_gemm",
                    "--logic",
                    "matmul",
                    "--op-type",
                    "matmul",
                    "--backend",
                    "auto",
                    "--m",
                    "1024",
                    "--n",
                    "1024",
                    "--k",
                    "1024",
                    "--output-dir",
                    str(out),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            payload = json.loads(result.stdout)
            op_dir = out / "demo_gemm"
            self.assertEqual(payload["backend"], "cuda")
            self.assertTrue((op_dir / "cpu_reference.py").exists())
            self.assertTrue((op_dir / "kernel_cuda.cu").exists())
            self.assertTrue((op_dir / "kernel_cuda.py").exists())
            self.assertTrue((op_dir / "benchmark_harness.py").exists())


if __name__ == "__main__":
    unittest.main()
