# Findings

## Repository Findings

- The current repository already has a two-layer architecture:
  - `e2e-inference-opt-skill` as the optimization knowledge router
  - `auto-profiling` as the safe experiment harness
- There is no dedicated LLM-serving skill package yet.
- The top-level repository had no planning files for this new task before this session.

## External Reference Findings

- FlashInfer demonstrates that high-value skills are narrow, operational, and source-backed:
  - CUDA crash debugging
  - kernel benchmarking
  - add-kernel implementation workflow
- FlashInfer's library strengths are directly relevant to LLM inference:
  - paged and ragged KV-cache
  - decode, prefill, append kernels
  - cascade attention for shared prefixes
  - fused sampling kernels
  - CUDAGraph and `torch.compile` compatibility
- SGLang's benchmark stack is especially useful for skill design:
  - `bench_serving` for realistic online serving metrics
  - `bench_one_batch_server` for single-batch end-to-end latency
  - `bench_offline_throughput` for in-process engine throughput
  - `bench_one_batch` for low-level static-batch profiling
- SGLang benchmark guidance strongly suggests using `bench_serving` by default for realistic TTFT, TPOT, ITL, and throughput measurement.
- SGLang profiling guidance adds a practical contract:
  - set `SGLANG_TORCH_PROFILER_DIR` on both server and client
  - separate prefill and decode profiling in PD disaggregation mode
- The most reusable open-source patterns for this repository are workflow patterns and command recipes, not wholesale runtime import.

## Design Direction

- Recommended structure: add a dedicated `llm-serving-opt-skill` package instead of overloading `e2e-inference-opt-skill`.
- Recommended content model:
  - one routing `SKILL.md`
  - focused references for benchmark, profiling, CUDA crash triage, kernel/backend choice, KV-cache and scheduler, deployment, and auto-driven iteration
- Recommended integration model:
  - `llm-serving-opt-skill` tells the agent how to think and operate
  - `auto-profiling` remains the execution harness for guarded experiments

## Implementation Findings

- The new package should stay operational rather than encyclopedic; each reference needs a specific workflow and not just concept bullets.
- FlashInfer material is most valuable for:
  - pre-crash API logging
  - kernel benchmark discipline
  - source-backed kernel-integration workflow
- SGLang material is most valuable for:
  - benchmark tool selection
  - serving-profile triage
  - launch-command and deployment reproducibility
- A small structural test is worthwhile even for markdown-heavy skill work because it prevents silent regressions in skill packaging.
- The cleanest repository layering is now:
  - `e2e-inference-opt-skill` = generic inference reasoning
  - `llm-serving-opt-skill` = LLM-serving specialization
  - `auto-profiling` = bounded experiment execution

## Multi-Skill Findings

- A single specialized router skill is still useful, but it should route into narrower operational skills rather than trying to hold every workflow itself.
- The best sub-skill split for this repository is a hybrid of framework-specific and generic skills:
  - framework-anchored where the command surface matters, such as `sglang-benchmark-skill`
  - generic where the method should transfer across frameworks, such as crash triage, profile triage, backend selection, correctness, and deployment
- vLLM contributes especially useful practice around:
  - prefix-caching semantics and isolation
  - serve-vs-offline distinction
  - profiling controls and trace-handling caveats
- TensorRT-LLM contributes especially useful practice around:
  - build/serve/benchmark separation
  - throughput-at-latency thinking
  - disaggregated serving trade-offs
  - benchmark hygiene such as GPU persistence, power, and clock settings for reproducibility
- Triton Inference Server contributes the service-packaging and rollout discipline more than low-level kernel methodology.
- PyTorch remains important as the baseline and fallback lane because it offers the shortest path for validation and debugging even when it is not the final serving path.

## Consolidation Findings

- Physically consolidating all LLM-serving skills under `llm-serving-opt-skill/skills` is cleaner than keeping them at repository top level because:
  - the router and sub-skills become one coherent package
  - future references can live next to their owning skill
  - the repository root stays focused on major systems rather than many sibling skill folders
- A strict migration test is important: the suite should verify both the presence of nested sub-skills and the absence of the old top-level directories.
- The best lightweight deepening pass for each skill is to add:
  - input questions
  - evidence requirements
  - framework-specific cautions
  - escalation or tuning order
- This repository now has a more natural growth path: `llm-serving-opt-skill/skills/<name>/references/` can be added later without changing the top-level layout again.

## cuda-optimized-skill Findings

- The article's most important contribution is not a single CUDA trick but a closed-loop workflow:
  - preflight
  - correctness validation
  - benchmark
  - targeted NCU
  - full NCU
  - optimization proposal
  - strategy memory
  - strict best-version selection
- The upstream repository contains directly reusable source code, not just prose:
  - `kernel-benchmark/scripts/benchmark.py`
  - `operator-optimize-loop/scripts/optimize_loop.py`
  - `operator-optimize-loop/strategy-memory/global_strategy_memory.json`
  - backend-specific reference documents for CUDA, CUTLASS, and Triton
- Since the repository is MIT licensed, it is appropriate to vendor the upstream source into this project as long as the license is preserved.
- The right integration shape is to keep the upstream toolkit intact under one nested skill root rather than scattering its scripts into unrelated serving skills.
- Local path rewriting is mandatory after vendoring because the upstream code and docs hard-code `skills/optimized-skill/...` paths.

## Three-Module Findings

- The current repository structure is clearer when split into three sibling modules:
  - `e2e-inference-opt-skill` for non-LLM and small-model end-to-end optimization
  - `llm-serving-opt-skill` for service-level LLM optimization
  - `cuda-kernel-opt-skill` for operator and kernel-level optimization
- Not all sub-skills that mention serving belong inside the LLM-serving root; some of them are better treated as CUDA/operator skills once the bottleneck has already been narrowed to kernel behavior.
- `auto-profiling` works best as the common entrypoint only if it exposes scenario-specific aim templates instead of a single generic contract file.
- The minimal useful scenario split is:
  - `aim.e2e.md`
  - `aim.llm-serving.md`
  - `aim.cuda-kernel.md`
