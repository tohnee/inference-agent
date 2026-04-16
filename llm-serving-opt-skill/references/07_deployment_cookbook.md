# 07 Deployment Cookbook

## Goal

Turn serving optimization into a reproducible deployment workflow instead of a pile of shell history.

## Recommended Deployment Mindset

Borrow from the SGLang cookbook style:

- keep launch commands explicit
- keep benchmark commands explicit
- separate local validation from service deployment
- prefer one reproducible launch path per environment

## Local Single-GPU Validation

Use a minimal server launch before scaling out:

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 30000
```

Then validate with a benchmark or a simple generate request.

## Docker Service Pattern

A common container launch pattern is:

```bash
docker run --gpus all \
  -p 30000:30000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HF_TOKEN=<secret>" \
  --ipc=host \
  lmsysorg/sglang:latest \
  python3 -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --host 0.0.0.0 --port 30000
```

## Service Rollout Checklist

Before calling a deployment ready, verify:

- exactness on a golden prompt set
- smoke benchmark on a running server
- memory headroom at target concurrency
- launch command saved in docs or infra config
- profiler path available for emergency triage
- fallback backend strategy documented

## Backend Fallback Example

If FlashInfer is expected but the target GPU or environment breaks that path, keep a fallback plan ready such as switching attention or sampling backend rather than blocking the whole rollout.

## Scale-Out Questions

Ask these before going multi-node or service-grade:

- is single-node behavior already understood?
- are TTFT and TPOT already bounded on one node?
- is the bottleneck model compute, scheduler overhead, or network transport?
- does the deployment need PD disaggregation or a simpler server first?

## Common Mistakes

- jumping to k8s before single-node metrics are stable
- not preserving the exact launch flags used for good runs
- treating deployment as separate from benchmark and profiling
- rolling out a faster backend without a parity gate
