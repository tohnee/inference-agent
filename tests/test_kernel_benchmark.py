import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "cuda-kernel-opt-skill"
    / "skills"
    / "cuda-optimized-skill"
    / "kernel-benchmark"
    / "scripts"
    / "benchmark.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("kernel_benchmark_module", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    fake_torch = types.SimpleNamespace(
        float32="float32",
        float64="float64",
        int32="int32",
        int64="int64",
        int16="int16",
        int8="int8",
        uint8="uint8",
        uint16="uint16",
        uint32="uint32",
        Tensor=object,
        cuda=types.SimpleNamespace(),
    )
    original_torch = sys.modules.get("torch")
    sys.modules["torch"] = fake_torch
    try:
        spec.loader.exec_module(module)
    finally:
        if original_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = original_torch
    return module


class _FakeCuda:
    synchronize = staticmethod(lambda: None)
    _durations = [1.0, 2.0, 3.0]
    _pair_index = 0
    _record_phase = 0

    class Event:
        def __init__(self, enable_timing=True):
            self.pair_index = None

        def record(self):
            if self.pair_index is None and _FakeCuda._record_phase == 0:
                self.pair_index = _FakeCuda._pair_index
                _FakeCuda._record_phase = 1
            elif self.pair_index is None:
                self.pair_index = _FakeCuda._pair_index
                _FakeCuda._pair_index += 1
                _FakeCuda._record_phase = 0

        def elapsed_time(self, _other):
            return _FakeCuda._durations[self.pair_index]


class KernelBenchmarkTimingTests(unittest.TestCase):
    def test_measure_iteration_times_preserves_per_iteration_samples(self):
        module = load_module()
        _FakeCuda._pair_index = 0
        _FakeCuda._record_phase = 0
        with patch.object(module.torch, "cuda", _FakeCuda):
            samples = module.measure_iteration_times(lambda: None, warmup=1, repeat=3)
        self.assertEqual(samples, [1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
