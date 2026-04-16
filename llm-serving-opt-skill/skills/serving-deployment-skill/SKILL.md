---
name: "serving-deployment-skill"
description: "Use when launching, packaging, benchmarking, or rolling out LLM serving with SGLang, TensorRT-LLM, Triton, vLLM, or PyTorch services."
---

# Serving Deployment Skill

## Use When

- starting a new serving stack
- hardening a benchmark or rollout recipe
- turning a good local run into a reproducible service

## Deployment Mindset

- preserve exact launch commands
- preserve benchmark commands
- validate locally before scaling out
- keep a backend fallback plan

## Framework Notes

- **SGLang**: great for explicit launch and benchmark recipes
- **TensorRT-LLM**: separate build, serve, and benchmark clearly
- **Triton Inference Server**: always record model config and instance-group assumptions
- **vLLM**: record serve flags, cache flags, and concurrency assumptions

## Rollout Checklist

- golden prompts pass
- smoke benchmark passes
- memory headroom exists
- profiler hook is available
- rollback path is documented

## Deliverables

- launch recipe
- smoke validation recipe
- benchmark recipe
- rollout and rollback notes

## Deployment Sequence

1. local smoke validation
2. single-node benchmark
3. profile hook verification
4. service rollout with rollback path
5. scale-out only after single-node behavior is understood
