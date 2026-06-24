# 2026-06-15 User-Perspective Review: Reasoning Task Optimization

## Review Goal

This review looks at `inference-agent` from the perspective of a user who wants help optimizing inference workloads that contain reasoning-heavy behavior, such as multi-step answers, tool-using agents, long-context analysis, code reasoning, retrieval-augmented synthesis, or structured decision workflows.

The key product question is not "can the system run a benchmark?" but:

> Can a user describe a reasoning task, receive a safe optimization plan, and trust that speed or cost gains did not degrade reasoning quality?

## Current Strengths for Users

1. **Single contract is easy to explain.**
   - Users only need to edit one `aim.md` contract instead of learning several internal modules.
   - This is a strong starting point for onboarding non-infra users.

2. **Exactness-first behavior is the right default.**
   - Reasoning optimizations can silently change outputs, tool choices, citation quality, or step ordering.
   - The existing exactness-first evaluator gives the product a conservative safety posture.

3. **Scenario routing already separates concerns.**
   - `e2e-inference`, `llm-serving`, and `cuda-kernel` are useful lanes for infrastructure-level optimization.
   - Reasoning tasks can reuse these lanes when the user's bottleneck is serving latency, KV cache behavior, batching, kernels, or end-to-end orchestration.

## User Gaps for Reasoning-Heavy Tasks

### P0 / Blocks User Trust

1. **The aim contract does not ask for a reasoning-quality contract.**
   - Current fields cover exactness artifacts and metric direction, but reasoning tasks often need semantic checks: answer correctness, instruction following, citation fidelity, tool-call validity, refusal correctness, or rubric score.
   - Without these fields, users may over-optimize latency while missing degraded answer quality.

2. **The user has no guided way to choose a reasoning evaluation mode.**
   - Exact parity is too strict for many generative reasoning outputs.
   - Bounded numeric tolerance is insufficient for natural-language or agentic outputs.
   - Users need explicit modes such as `semantic-rubric`, `golden-trace`, `tool-call-parity`, and `judge-assisted`.

3. **Reasoning workload intent is not captured.**
   - The contract does not currently distinguish between chat reasoning, code reasoning, RAG synthesis, tool-using agents, multi-turn planning, or batch offline evaluation.
   - Those differences change safe mutation lanes and what quality checks are required.

### P1 / Makes Onboarding Hard

1. **Users must translate product goals into infra metrics themselves.**
   - A user may say "make this agent answer faster without losing correctness," but the contract asks for `target_metric_name`, profile commands, exactness paths, and mutation surfaces.
   - The system should provide a guided intake that maps user intent to metrics and checks.

2. **No reasoning-specific examples.**
   - Existing examples are infrastructure-oriented.
   - Users need copyable examples for long-context QA, RAG, coding assistant, and tool-agent workflows.

3. **Safe optimization lanes are not explained in user language.**
   - Users understand "do not reduce answer quality" better than "blocked precision transitions."
   - Docs should translate low-level restrictions into user-visible risks: missing facts, invalid tool calls, lower citation coverage, shorter but wrong answers, or broken multi-turn memory.

## Recommended Product Additions

### 1. Add a Reasoning Task Intake Section to `aim.md`

Add optional fields that are ignored by existing runtime paths until implemented, but immediately improve user guidance:

```yaml
reasoning_task_type: chat_reasoning | code_reasoning | rag_synthesis | tool_agent | long_context | batch_eval
reasoning_quality_metric: exact_match | semantic_rubric | judge_score | tool_trace_parity | citation_fidelity
reasoning_quality_threshold:
golden_dataset_path:
golden_trace_path:
judge_command:
required_output_properties:
  - preserves final answer correctness
  - preserves required citations
  - preserves valid tool-call arguments
  - preserves refusal / safety policy behavior
```

### 2. Introduce Reasoning Evaluation Modes

| Mode | User need | Promotion rule |
| --- | --- | --- |
| `exact-parity` | Deterministic structured outputs | Candidate output must match baseline exactly. |
| `golden-trace` | Tool agents and planners | Required tool sequence, arguments, and final answer properties must pass. |
| `semantic-rubric` | Natural-language reasoning | Candidate must meet rubric threshold and cannot regress critical criteria. |
| `citation-fidelity` | RAG and grounded QA | Candidate must preserve answer support, citation coverage, and source validity. |
| `judge-assisted` | Open-ended tasks | A declared judge command scores baseline and candidate with fixed seeds and report artifacts. |

### 3. Provide User-Facing Presets

Add bootstrap presets that create a near-complete aim for common user requests:

```bash
python3 auto-profiling/bootstrap_aim.py \
  --mode reasoning \
  --profile rag \
  --project-name docs-qa-agent \
  --target-repo-path /path/to/agent \
  --output auto-profiling/aim.md
```

Recommended profiles:

- `rag`: retrieval + synthesis latency, citation fidelity, context-window budget.
- `tool-agent`: tool-call latency, trace validity, retry count, final answer correctness.
- `code-reasoning`: pass-rate, edit correctness, test latency, token/cost budget.
- `long-context`: prefill latency, memory footprint, answer completeness.
- `chat-reasoning`: p95 latency, rubric score, refusal/instruction-following behavior.

### 4. Translate Optimization Lanes Into User Language

For reasoning tasks, docs and candidate plans should state the user-visible promise before the infrastructure mutation:

- "Keep the same answer quality while reducing repeated context processing" -> prefix/KV cache optimization.
- "Keep tool calls valid while reducing orchestration latency" -> scheduler/concurrency tuning.
- "Keep citations grounded while shrinking prompt payload" -> retrieval and context packing changes gated by citation checks.
- "Keep test pass rate unchanged while reducing coding-agent turnaround" -> batching, cache, and harness improvements.

### 5. Make Candidate Reports Explain Quality Risk

Each `next_candidate.md` should include:

- user goal being optimized;
- expected user-visible improvement;
- reasoning-quality checks that must pass;
- what output behavior may change;
- exact rollback condition;
- artifacts a user can inspect without reading code.

## Suggested Implementation Order

1. **Documentation-only onboarding patch.**
   - Add reasoning-task review and link it from READMEs.
   - Add a copyable reasoning-oriented aim example.

2. **Schema-compatible aim fields.**
   - Add optional reasoning fields to `aim_schema.json`.
   - Keep required fields unchanged for backward compatibility.

3. **Bootstrap preset.**
   - Add `--mode reasoning` and profiles for `rag`, `tool-agent`, `code-reasoning`, `long-context`, and `chat-reasoning`.

4. **Evaluator extensions.**
   - Add semantic/rubric/judge artifacts as explicit inputs.
   - Persist baseline and candidate quality reports side-by-side.

5. **Candidate-plan UX.**
   - Teach candidate generation to write user-goal, quality-risk, and rollback sections.

## Acceptance Criteria

A reasoning-task optimization should not be considered user-ready until:

1. The user can fill an aim from product-language prompts rather than infra-only fields.
2. Baseline artifacts include both performance and reasoning-quality evidence.
3. Every candidate declares the quality gate it must pass before evaluation starts.
4. Promotion requires no regression in the declared quality contract.
5. Handoff explains the result in user terms: what got faster or cheaper, what quality checks passed, and what remains risky.
