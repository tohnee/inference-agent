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


class OptimizeLoopLogicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_pick_iteration_index_skips_gaps(self):
        manifest = {"iterations": [{"iteration": 0}, {"iteration": 2}]}
        self.assertEqual(self.module.pick_iteration_index(manifest, -1), 3)
        self.assertEqual(self.module.pick_iteration_index({"iterations": []}, -1), 0)
        self.assertEqual(self.module.pick_iteration_index(manifest, 5), 5)

    def test_choose_best_iteration_skips_failed_benchmarks(self):
        good = {
            "iteration": 0,
            "benchmark_rc": 0,
            "benchmark_result": {"kernel": {"median_ms": 2.0, "average_ms": 2.0}},
        }
        failed_but_faster = {
            "iteration": 1,
            "benchmark_rc": 1,
            "benchmark_result": {"kernel": {"median_ms": 1.0, "average_ms": 1.0}},
        }
        best = self.module.choose_best_iteration([good, failed_but_faster])
        self.assertIsNotNone(best)
        self.assertEqual(best["iteration"], 0)

    def test_read_json_returns_default_for_corrupt_file(self):
        import tempfile
        from pathlib import Path as P

        with tempfile.TemporaryDirectory() as tmp:
            path = P(tmp) / "broken.json"
            path.write_text('{"truncated": ', encoding="utf-8")
            self.assertEqual(self.module.read_json(path, {"fallback": True}), {"fallback": True})

    def test_classify_requires_iteration_zero_for_baseline_seed(self):
        record = {"iteration": 3, "benchmark_rc": 0, "benchmark_result": {}}
        outcome, reason = self.module.classify_strategy_outcome(record, None)
        self.assertEqual(outcome, "rejected")
        self.assertEqual(reason, "no_previous_record")

        baseline = {"iteration": 0, "benchmark_rc": 0, "benchmark_result": {}}
        outcome, reason = self.module.classify_strategy_outcome(baseline, None)
        self.assertEqual(outcome, "positive")
        self.assertEqual(reason, "baseline_seed")


if __name__ == "__main__":
    unittest.main()
