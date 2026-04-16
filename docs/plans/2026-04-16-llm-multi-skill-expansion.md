# LLM Multi-Skill Expansion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expand the repository from one LLM serving skill into a multi-skill system with 8-12 focused sub-skills inspired by FlashInfer, SGLang, vLLM, TensorRT-LLM, Triton, and PyTorch practices.

**Architecture:** Keep `llm-serving-opt-skill` as the router, then add narrow operational sub-skills for benchmark, crash triage, profiling, backend selection, KV cache, scheduler, correctness, deployment, remote validation, and custom-kernel workflows. Add structural tests first, then write each skill as an operational playbook with cross-framework guidance.

**Tech Stack:** Markdown skill packages, Python `unittest`, existing repository planning files.

---

### Task 1: Expand structural tests for multi-skill packaging

**Files:**
- Modify: `/Users/tc/Downloads/推理优化skills/tests/test_skill_catalog.py`

**Step 1: Write the failing test**

Add assertions for:
- existence of 10 new skill directories with `SKILL.md`
- router skill mentions selected sub-skills
- a few representative framework names such as `vLLM`, `TensorRT-LLM`, `Triton`, `PyTorch`

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests/test_skill_catalog.py`
Expected: FAIL because the new sub-skill directories do not exist yet.

**Step 3: Write minimal implementation**

No production content yet. Only update the test file.

**Step 4: Run test to verify it fails**

Run: `python3 -m unittest tests/test_skill_catalog.py`
Expected: FAIL with missing-skill assertions.

### Task 2: Update the router skill

**Files:**
- Modify: `/Users/tc/Downloads/推理优化skills/llm-serving-opt-skill/SKILL.md`

**Step 1: Write the failing test**

Covered by Task 1 router-content assertions.

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests/test_skill_catalog.py`
Expected: FAIL due to missing mentions and missing skill directories.

**Step 3: Write minimal implementation**

Update the router to:
- explain the multi-skill architecture
- route tasks to specific sub-skills
- emphasize cross-framework support

**Step 4: Run test to verify partial progress**

Run: `python3 -m unittest tests/test_skill_catalog.py`
Expected: still FAIL until all sub-skills exist.

### Task 3: Add benchmark-focused skills

**Files:**
- Create: `/Users/tc/Downloads/推理优化skills/sglang-benchmark-skill/SKILL.md`
- Create: `/Users/tc/Downloads/推理优化skills/serving-benchmark-skill/SKILL.md`

**Step 1: Write the failing test**

Covered by Task 1 missing-directory assertions.

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests/test_skill_catalog.py`
Expected: FAIL for missing benchmark skill directories.

**Step 3: Write minimal implementation**

Create:
- `sglang-benchmark-skill` for SGLang command selection and trace collection
- `serving-benchmark-skill` for cross-framework benchmark methodology across SGLang, vLLM, TRT-LLM, Triton, and PyTorch

**Step 4: Run test**

Run: `python3 -m unittest tests/test_skill_catalog.py`
Expected: benchmark-skill related assertions pass.

### Task 4: Add crash and profile skills

**Files:**
- Create: `/Users/tc/Downloads/推理优化skills/cuda-crash-debug-skill/SKILL.md`
- Create: `/Users/tc/Downloads/推理优化skills/profile-triage-skill/SKILL.md`

**Step 1: Write the failing test**

Covered by Task 1.

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests/test_skill_catalog.py`
Expected: FAIL until these directories exist.

**Step 3: Write minimal implementation**

Create operational guides for:
- pre-crash logging and sanitizers
- triage tables for kernel, overlap, and fuse opportunities

**Step 4: Run test**

Run: `python3 -m unittest tests/test_skill_catalog.py`
Expected: these assertions pass.

### Task 5: Add serving-core optimization skills

**Files:**
- Create: `/Users/tc/Downloads/推理优化skills/backend-selection-skill/SKILL.md`
- Create: `/Users/tc/Downloads/推理优化skills/kv-cache-prefix-cache-skill/SKILL.md`
- Create: `/Users/tc/Downloads/推理优化skills/scheduler-batching-skill/SKILL.md`
- Create: `/Users/tc/Downloads/推理优化skills/serving-correctness-skill/SKILL.md`

**Step 1: Write the failing test**

Covered by Task 1.

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests/test_skill_catalog.py`
Expected: FAIL for missing directories.

**Step 3: Write minimal implementation**

Create focused skills for:
- backend selection across frameworks
- KV cache and prefix cache correctness/performance
- continuous batching, fairness, and prefill/decode scheduling
- exactness and serving regression gates

**Step 4: Run test**

Run: `python3 -m unittest tests/test_skill_catalog.py`
Expected: these assertions pass.

### Task 6: Add deployment and workflow skills

**Files:**
- Create: `/Users/tc/Downloads/推理优化skills/serving-deployment-skill/SKILL.md`
- Create: `/Users/tc/Downloads/推理优化skills/remote-gpu-validation-skill/SKILL.md`
- Create: `/Users/tc/Downloads/推理优化skills/custom-kernel-workflow-skill/SKILL.md`

**Step 1: Write the failing test**

Covered by Task 1.

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests/test_skill_catalog.py`
Expected: FAIL for missing directories.

**Step 3: Write minimal implementation**

Create focused skills for:
- deployment and rollout across SGLang, TRT-LLM, Triton, and PyTorch services
- remote GPU machine validation and safe environment checks
- custom kernel workflow with an emphasis on reuse before custom code

**Step 4: Run test**

Run: `python3 -m unittest tests/test_skill_catalog.py`
Expected: all structural assertions pass.

### Task 7: Verify the full skill catalog

**Files:**
- Verify existing and new markdown skills

**Step 1: Run targeted test**

Run: `python3 -m unittest tests/test_skill_catalog.py`
Expected: PASS

**Step 2: Run full discovered tests**

Run: `python3 -m unittest discover -s tests`
Expected: PASS

**Step 3: Run diagnostics**

Check diagnostics for the repository to ensure no new issues were introduced.

**Step 4: Update planning files**

Record:
- new skill inventory
- research findings
- verification results

