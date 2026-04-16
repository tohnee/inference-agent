# Auto-Profiling Runtime Lane Awareness Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `auto-profiling/runner.py` understand `scenario` at runtime, surface lane metadata in runtime artifacts, and add stronger environment collection before optimization begins.

**Architecture:** Extend the existing `aim.md` parsing path with a scenario-to-lane mapping layer, include that lane metadata in `status`, `current_contract.md`, and `next_handoff.md`, and enrich runtime environment detection with Python, platform, package, and optional `vllm collect-env` evidence. Drive the work through `unittest` updates first, then implement the minimum code to satisfy those tests, then refresh docs.

**Tech Stack:** Python stdlib runtime, Python `unittest`, Markdown docs.

---

### Task 1: Write failing runtime tests

**Files:**
- Modify: `/Users/tc/Downloads/推理优化skills/auto-profiling/tests/test_runtime.py`

**Step 1: Write the failing test**

Add focused tests for:
- scenario `llm-serving` mapping to the correct target module and recommended skill route
- `status` JSON including `scenario`, `target_module`, and `recommended_skill_route`
- `.auto-profiling/current_contract.md` including the same lane metadata
- `.auto-profiling/next_handoff.md` including the same lane metadata
- stronger environment collection returning richer runtime metadata and optional external tool evidence

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests/test_runtime.py -v`
Expected: FAIL because the runtime does not yet expose lane metadata or enriched environment collection.

### Task 2: Implement scenario-aware lane metadata

**Files:**
- Modify: `/Users/tc/Downloads/推理优化skills/auto-profiling/runner.py`

**Step 1: Write minimal implementation**

Add helpers that:
- normalize and validate `scenario`
- map each scenario to a target module and recommended skill route
- attach lane metadata to status payloads and generated markdown artifacts

**Step 2: Run targeted test**

Run: `python3 -m unittest tests/test_runtime.py -v`
Expected: lane metadata tests pass while enriched environment tests may still fail.

### Task 3: Implement stronger environment collection

**Files:**
- Modify: `/Users/tc/Downloads/推理优化skills/auto-profiling/runner.py`

**Step 1: Write minimal implementation**

Extend environment detection to collect:
- Python version and executable
- platform and machine details
- shell and package manager fallback results
- common accelerator tooling paths when present
- optional `vllm collect-env` style evidence when `vllm` is installed

**Step 2: Run targeted test**

Run: `python3 -m unittest tests/test_runtime.py -v`
Expected: runtime tests pass.

### Task 4: Refresh docs

**Files:**
- Modify: `/Users/tc/Downloads/推理优化skills/auto-profiling/README.md`
- Modify: `/Users/tc/Downloads/推理优化skills/auto-profiling/README.zh-CN.md`
- Modify: `/Users/tc/Downloads/推理优化skills/auto-profiling/SKILL.md`

**Step 1: Update documentation**

Describe the runtime lane metadata, recommended skill routing, and stronger environment collection behavior.

**Step 2: Run targeted verification**

Run: `python3 -m unittest tests/test_runtime.py -v`
Expected: PASS.

### Task 5: Final verification

**Files:**
- Verify modified files

**Step 1: Run tests**

Run: `python3 -m unittest tests/test_runtime.py -v`
Expected: PASS.

**Step 2: Run diagnostics**

Check diagnostics for edited files and fix any introduced issues.
