import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LLM_ROOT = ROOT / "llm-serving-opt-skill"
LLM_SKILLS_ROOT = LLM_ROOT / "skills"
E2E_ROOT = ROOT / "e2e-inference-opt-skill"
CUDA_ROOT = ROOT / "cuda-kernel-opt-skill"
CUDA_SKILLS_ROOT = CUDA_ROOT / "skills"
LLM_SUBSKILLS = [
    "sglang-benchmark-skill",
    "serving-benchmark-skill",
    "serving-deployment-skill",
]
CUDA_SUBSKILLS = [
    "cuda-crash-debug-skill",
    "profile-triage-skill",
    "backend-selection-skill",
    "kv-cache-prefix-cache-skill",
    "scheduler-batching-skill",
    "serving-correctness-skill",
    "remote-gpu-validation-skill",
    "custom-kernel-workflow-skill",
    "cuda-optimized-skill",
]


class LlmServingSkillCatalogTests(unittest.TestCase):
    def test_three_top_level_modules_exist(self):
        self.assertTrue(LLM_ROOT.exists())
        self.assertTrue((LLM_ROOT / "SKILL.md").exists())
        self.assertTrue(E2E_ROOT.exists())
        self.assertTrue((E2E_ROOT / "SKILL.md").exists())
        self.assertTrue(CUDA_ROOT.exists())
        self.assertTrue((CUDA_ROOT / "SKILL.md").exists())
        self.assertFalse((ROOT / "inference-opt-skill").exists())

    def test_llm_serving_references_exist(self):
        expected = [
            "01_serving_baseline.md",
            "02_benchmark_workflows.md",
            "03_profile_analysis.md",
            "04_cuda_crash_triage.md",
            "05_kernel_backend_playbook.md",
            "06_kv_cache_scheduler.md",
            "07_deployment_cookbook.md",
            "08_agentic_optimization.md",
        ]
        for name in expected:
            self.assertTrue((LLM_ROOT / "references" / name).exists(), name)

    def test_e2e_references_exist(self):
        expected = [
            "01_baseline.md",
            "02_profiling.md",
            "03_roofline.md",
            "04_compute_opt.md",
            "05_memory_io.md",
            "06_parallelism.md",
            "07_pipeline_cache.md",
            "08_deployment.md",
        ]
        for name in expected:
            self.assertTrue((E2E_ROOT / "references" / name).exists(), name)

    def test_skill_mentions_sglang_and_flashinfer(self):
        text = (LLM_ROOT / "SKILL.md").read_text(encoding="utf-8")
        self.assertIn("SGLang", text)
        self.assertIn("FlashInfer", text)
        self.assertIn("TTFT", text)
        self.assertIn("KV cache", text)

    def test_llm_subskill_packages_exist(self):
        for name in LLM_SUBSKILLS:
            skill_dir = LLM_SKILLS_ROOT / name
            self.assertTrue(skill_dir.exists(), name)
            self.assertTrue((skill_dir / "SKILL.md").exists(), name)

    def test_cuda_subskill_packages_exist(self):
        for name in CUDA_SUBSKILLS:
            skill_dir = CUDA_SKILLS_ROOT / name
            self.assertTrue(skill_dir.exists(), name)
            self.assertTrue((skill_dir / "SKILL.md").exists(), name)

    def test_top_level_subskill_packages_removed(self):
        for name in LLM_SUBSKILLS + CUDA_SUBSKILLS:
            self.assertFalse((ROOT / name).exists(), name)

    def test_llm_router_mentions_three_module_links(self):
        text = (LLM_ROOT / "SKILL.md").read_text(encoding="utf-8")
        self.assertIn("sglang-benchmark-skill", text)
        self.assertIn("cuda-kernel-opt-skill", text)
        self.assertIn("e2e-inference-opt-skill", text)
        self.assertIn("vLLM", text)
        self.assertIn("TensorRT-LLM", text)
        self.assertIn("Triton", text)
        self.assertIn("PyTorch", text)
        self.assertIn("llm-serving-opt-skill/skills", text)

    def test_cuda_optimized_skill_vendors_upstream_assets(self):
        root = CUDA_SKILLS_ROOT / "cuda-optimized-skill"
        self.assertTrue((root / "SKILL.md").exists())
        self.assertTrue((root / "README.md").exists())
        self.assertTrue((root / "LICENSE").exists())
        self.assertTrue((root / "kernel-benchmark" / "scripts" / "benchmark.py").exists())
        self.assertTrue((root / "ncu-rep-analyze" / "SKILL.md").exists())
        self.assertTrue((root / "operator-optimize-loop" / "scripts" / "optimize_loop.py").exists())
        self.assertTrue((root / "operator-optimize-loop" / "strategy-memory" / "global_strategy_memory.json").exists())
        self.assertTrue((root / "reference" / "cuda" / "optim.md").exists())
        self.assertTrue((root / "reference" / "cutlass" / "cutlass-optim.md").exists())
        self.assertTrue((root / "reference" / "triton" / "triton-optim.md").exists())

    def test_auto_profiling_has_three_scenario_aims(self):
        auto_root = ROOT / "auto-profiling"
        self.assertTrue((auto_root / "aim.e2e.md").exists())
        self.assertTrue((auto_root / "aim.llm-serving.md").exists())
        self.assertTrue((auto_root / "aim.cuda-kernel.md").exists())


if __name__ == "__main__":
    unittest.main()
