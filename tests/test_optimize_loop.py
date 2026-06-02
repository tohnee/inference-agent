import importlib.util
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
    / "optimize_loop.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("optimize_loop_module", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class OptimizeLoopPathTests(unittest.TestCase):
    def test_support_paths_resolve_to_existing_assets(self):
        module = load_module()
        benchmark_script, global_memory_file = module.resolve_support_paths()
        self.assertTrue(benchmark_script.exists(), benchmark_script)
        self.assertTrue(global_memory_file.exists(), global_memory_file)


if __name__ == "__main__":
    unittest.main()
