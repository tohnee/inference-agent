# Task Plan

## Goal

Deep-review `auto-profiling` for runtime robustness and make environment detection degrade safely:

- prefer `zsh`, then fall back to `bash`, then system shell
- prefer `uv`, then fall back to `pip`
- surface detected environment in runtime artifacts and status output
- document the fallback behavior so operators know what the runtime will do

## Phases

| Phase | Status | Notes |
| --- | --- | --- |
| inspect runtime and docs | complete | reviewed `runner.py`, tests, README, and aim templates for environment assumptions |
| add RED tests | complete | added fallback-focused tests for shell detection, package-manager detection, and auto install selection |
| implement fallbacks | complete | added shell/package-manager detection, dynamic install command selection, and environment reporting |
| update docs | complete | documented `install_command: auto` and shell/package-manager fallback behavior in README and aim templates |
| verify and polish | complete | tests, compile checks, live environment probe, and final cleanup passed |

## Constraints

- exact-parity mode remains the default red line
- bounded tolerance remains opt-in and contract-driven
- runtime stays stdlib-first and avoids hard dependency on `uv`
- environment fallback must be deterministic and observable in logs/artifacts
- documentation must match actual runtime behavior
