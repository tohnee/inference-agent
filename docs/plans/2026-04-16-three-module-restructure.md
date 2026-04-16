# Three Module Restructure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restructure the repository into three top-level optimization modules plus `auto-profiling`: `e2e-inference-opt-skill`, `llm-serving-opt-skill`, and `cuda-kernel-opt-skill`, and make `auto-profiling` route to all three with scenario-specific aim templates.

**Architecture:** Rename the current generic inference module into an E2E module, extract kernel/CUDA-focused skills into a new sibling module, keep LLM serving in its own module, then update `auto-profiling` to treat these modules as scenario lanes. Use structural tests first, then perform file moves and documentation updates, and finally add scenario-specific aim templates.

**Tech Stack:** Markdown skill packages, Python `unittest`, existing `auto-profiling` runtime/docs.

---

### Task 1: Write failing tests for the new three-module structure

**Files:**
- Modify: `/Users/tc/Downloads/推理优化skills/tests/test_skill_catalog.py`

**Step 1: Write the failing test**

Add assertions for:
- `/Users/tc/Downloads/推理优化skills/e2e-inference-opt-skill/SKILL.md`
- `/Users/tc/Downloads/推理优化skills/llm-serving-opt-skill/SKILL.md`
- `/Users/tc/Downloads/推理优化skills/cuda-kernel-opt-skill/SKILL.md`
- expected nested sub-skills under the LLM-serving root and CUDA-kernel root
- absence of the old `/Users/tc/Downloads/推理优化skills/inference-opt-skill`
- existence of scenario aim templates under `auto-profiling`

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests/test_skill_catalog.py`
Expected: FAIL because the new module layout and aim templates do not exist yet.

### Task 2: Rename the E2E module

**Files:**
- Move: `/Users/tc/Downloads/推理优化skills/inference-opt-skill` -> `/Users/tc/Downloads/推理优化skills/e2e-inference-opt-skill`
- Modify references pointing to the old name

**Step 1: Implement the rename**

Move the directory and update route text so the module clearly means end-to-end inference chain optimization.

**Step 2: Run targeted test**

Run: `python3 -m unittest tests/test_skill_catalog.py`
Expected: E2E assertions pass while CUDA/auto-profiling assertions still fail.

### Task 3: Extract the CUDA/kernel module

**Files:**
- Create: `/Users/tc/Downloads/推理优化skills/cuda-kernel-opt-skill/SKILL.md`
- Create/move: `/Users/tc/Downloads/推理优化skills/cuda-kernel-opt-skill/skills/...`
- Move CUDA-related sub-skills out of `llm-serving-opt-skill`

**Step 1: Implement the extraction**

Move kernel-focused skills into the new CUDA module and add a top-level router skill.

**Step 2: Run targeted test**

Run: `python3 -m unittest tests/test_skill_catalog.py`
Expected: CUDA module assertions pass while some auto-profiling assertions may still fail.

### Task 4: Refresh the LLM serving module

**Files:**
- Modify: `/Users/tc/Downloads/推理优化skills/llm-serving-opt-skill/SKILL.md`
- Modify: nested `skills/README.md`

**Step 1: Update routing**

Make the LLM-serving root clearly service-focused and remove CUDA-kernel responsibilities that now belong to the CUDA module.

**Step 2: Run targeted test**

Run: `python3 -m unittest tests/test_skill_catalog.py`
Expected: LLM module assertions pass.

### Task 5: Connect `auto-profiling` to the three scenarios

**Files:**
- Modify: `/Users/tc/Downloads/推理优化skills/auto-profiling/SKILL.md`
- Modify: `/Users/tc/Downloads/推理优化skills/auto-profiling/aim.md`
- Create: `/Users/tc/Downloads/推理优化skills/auto-profiling/aim.e2e.md`
- Create: `/Users/tc/Downloads/推理优化skills/auto-profiling/aim.llm-serving.md`
- Create: `/Users/tc/Downloads/推理优化skills/auto-profiling/aim.cuda-kernel.md`
- Update related README documents as needed

**Step 1: Implement scenario routing**

Document and template three scenarios:
- small-model end-to-end optimization
- LLM serving optimization
- CUDA/kernel-level optimization

**Step 2: Run targeted test**

Run: `python3 -m unittest tests/test_skill_catalog.py`
Expected: scenario aim assertions pass.

### Task 6: Verify the repository

**Files:**
- Verify modified and moved files

**Step 1: Run targeted test**

Run: `python3 -m unittest tests/test_skill_catalog.py`
Expected: PASS

**Step 2: Run full discovered tests**

Run: `python3 -m unittest discover -s tests`
Expected: PASS

**Step 3: Run diagnostics**

Check diagnostics for the repository.

**Step 4: Update planning files**

Record the final module layout, scenario routing, and verification results.
