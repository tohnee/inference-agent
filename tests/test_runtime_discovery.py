import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RuntimeDiscoveryTests(unittest.TestCase):
    def test_root_can_discover_auto_profiling_tests(self):
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", str(ROOT / "auto-profiling" / "tests")],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout + "\n" + result.stderr)


if __name__ == "__main__":
    unittest.main()
