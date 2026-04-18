# 09 Model Family Coverage Playbook

This reference is a rapid routing matrix for broad E2E inference optimization coverage.

## Family-to-Checklist

- Small models (classification/ranking):
  - prioritize preprocessing cost and CPU/GPU transfer overhead
  - enforce p95 and p99 under realistic concurrency
- Diffusion:
  - isolate UNet, VAE, text encoder costs
  - tune scheduler steps and CFG path
- Generic DL pipelines:
  - split preprocess / forward / postprocess explicitly
  - optimize queueing and stage overlap before kernel deep dives
- Transformer / ViT / SAM:
  - profile attention + MLP + embedding paths first
  - track memory format conversion and cache behavior
- Tree / tabular models:
  - prioritize feature-engineering hot path and memory locality
  - use thread pinning and vectorized feature transforms

## Fast-start sequence

1. define exactness contract
2. run baseline by family-specific command
3. collect one profile pass with stage decomposition
4. route to one bottleneck lane only
5. verify metric + exactness before promoting
