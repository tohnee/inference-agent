# Task Plan

## Goal

Build a project-level LLM inference optimization skill system on top of the existing repository:

- define three sibling optimization modules:
  - `e2e-inference-opt-skill`
  - `llm-serving-opt-skill`
  - `cuda-kernel-opt-skill`
- make `auto-profiling` the common entrypoint for the three scenarios
- provide scenario-specific aim templates for E2E, LLM serving, and CUDA/kernel optimization
- keep exactness and safe optimization gates aligned with existing project principles

## Phases

| Phase | Status | Notes |
| --- | --- | --- |
| inspect current repository | complete | confirmed existing split between `e2e-inference-opt-skill` and `auto-profiling` |
| collect external references | complete | extracted FlashInfer and SGLang reusable skill patterns plus SGLang benchmark/profiling guidance |
| define module split | complete | settled on E2E, LLM-serving, and CUDA-kernel as three sibling modules |
| write RED validation | complete | updated tests to require the new three-module layout and scenario aim templates |
| implement module and aim changes | complete | renamed `inference-opt-skill`, extracted CUDA-related skills into a new sibling module, and added scenario-based auto-profiling aims |
| verify | complete | targeted tests, full tests, `py_compile`, and diagnostics all passed |

## Constraints

- exact-parity remains the default red line
- skills should be directly usable by Claude Code as structured guidance
- open-source material should be adapted into reusable workflows, not copied blindly
- prioritize LLM serving metrics: TTFT, TPOT, ITL, throughput, concurrency, KV-cache behavior
- keep the new layer complementary to existing `e2e-inference-opt-skill` and `auto-profiling`

## Errors Encountered

| Error | Attempt | Resolution |
| --- | --- | --- |
| `session-catchup.py` path under `~/.claude` not found | 1 | continue with manual planning files in project root and record the environment mismatch |
| large single-shot file-generation script hit shell parsing failure | 1 | switched to one-file-at-a-time patches to keep the content stable and auditable |
| long zsh here-doc for batch skill updates broke parsing | 1 | wrote a temporary Python script file, ran it, then deleted it |
| `git clone` from GitHub timed out | 1 | switched to downloading the repository zip from `codeload.github.com` and unpacked it locally |
