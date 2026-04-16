---
name: "remote-gpu-validation-skill"
description: "Use when validating LLM serving changes on remote GPU machines, containers, or shared clusters before trusting benchmark results."
---

# Remote GPU Validation Skill

## Use When

- the real benchmark target is a remote GPU host
- environment drift may invalidate local results
- container, driver, or cache setup differs from local development

## First Checks

1. hostname and user
2. repo path and git state
3. container status if applicable
4. `nvidia-smi`
5. model cache and token availability
6. framework version and launch command

## Safety Rules

- never benchmark before confirming the exact target environment
- do not trust stale containers blindly
- keep remote launch, benchmark, and cleanup commands explicit
- prefer smoke validation before long benchmarks

## Deliverables

- environment fingerprint
- validation status
- exact benchmark target
- risks or mismatches to fix first

## Minimum Remote Checklist

- driver and CUDA runtime
- visible GPU inventory
- container image or repo revision
- model cache path
- token and secret availability
- benchmark destination path for logs and traces
