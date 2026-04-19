# inference-agent 能力深度升级方案（E2E / LLM Serving / Operator）

日期：2026-04-17

## 目标

本方案把仓库能力从“知识库 + 最小运行时”升级到“可落地的推理优化工程系统”，重点覆盖：

1. 端到端模型推理优化：小模型、Diffusion、DL、Transformer、SAM、ViT、树模型等。
2. 大模型推理服务优化：SGLang / vLLM / TensorRT-LLM 等 backend，支持 profiling、分布式调度与 AI Infra 优化。
3. 算子深度优化：在 CUDA 基础上补齐 Triton，并支持“输入算子逻辑 -> 自动选择 Triton/CUDA -> CPU baseline 校验”的工程链路。

## 开源实践吸收（关键原则）

吸收典型开源推理优化工程的共性实践（包括腾讯 HPC-Ops 等）：

- 生产级目标：围绕真实在线吞吐/延迟，不做脱离场景的空洞优化。
- 算子工程化：Attention/GEMM/MoE 等热点算子做专门优化路径。
- 基线与正确性先行：先有可信 baseline，再做性能迭代。
- 与 serving 框架可集成：能接入 vLLM/SGLang 等运行时。
- profile 驱动优化：所有“保留变更”必须可被 benchmark + profiler 证据支持。

## 本次实现（落地项）

### A. 新增 Operator 后端合成能力

新增脚本：

- `cuda-kernel-opt-skill/skills/cuda-optimized-skill/operator-optimize-loop/scripts/operator_backend_synth.py`

能力：

- 输入算子逻辑描述（`--logic`）与算子类型（`matmul/elementwise_add/layernorm`）
- 自动后端选择（`--backend=auto`）或手动指定（`cuda/triton`）
- 自动生成 CPU 参考实现 `cpu_reference.py` 作为精度 baseline
- 自动生成 CUDA 或 Triton 的实现脚手架
- 自动生成 benchmark + exactness harness，输出 `metric.json` 与 `exactness.json`
- 输出 `manifest.json` 记录 backend 决策与文件清单

### B. 新增 operator-backend-synthesis-skill

新增 skill：

- `cuda-kernel-opt-skill/skills/operator-backend-synthesis-skill/SKILL.md`

能力：

- 把“算子逻辑 -> CPU 基线 -> backend scaffold -> 优化循环”串成标准流程
- 明确与 `optimize_loop.py` 的衔接方式
- 固化 correctness-first 与 benchmark/profile-first 约束

### C. 运行时路由增强

`auto-profiling/runner.py` 新增场景：

- `operator-kernel`

让运行时可以直接路由到：

- `cuda-kernel-opt-skill`
- `operator-backend-synthesis-skill`
- `cuda-optimized-skill`

### D. 目录/测试同步更新

- `cuda-kernel-opt-skill/SKILL.md` 新增子技能入口。
- `cuda-kernel-opt-skill/skills/README.md` 新增子技能清单。
- `tests/test_skill_catalog.py` 新增子技能存在性校验。
- `tests/test_operator_backend_synth.py` 新增合成脚本测试：
  - elementwise_add 自动选 Triton
  - large matmul 自动选 CUDA

### E. 新增 E2E/LLM 场景化 aim 快速生成

新增脚本：

- `auto-profiling/bootstrap_aim.py`

能力：

- `--mode=e2e` 覆盖 `small-model/diffusion/dl/transformer/sam/vit/tree`
- `--mode=llm-serving` 覆盖 `sglang/vllm/trtllm`
- 按 profile 预填 metric、baseline run/profile 命令、bottleneck 提示、safe lane 建议
- 直接输出可执行的 `aim.md`，降低落地门槛

### F. 新增 autopilot 一键自动迭代

增强 `auto-profiling/runner.py`：

- 新增 `autopilot` 子命令，支持 baseline 缺失时自动补齐
- 支持连续多轮 candidate 自动执行（`--iterations`）
- 新增 `command_retry_count`（在 aim 中可配置）提升命令执行稳定性
- 自动写出 `skill_route_plan.md`，让三大方向 skills 可在 Codex 中独立复用

## 如何使用（快速示例）

### 1) 根据算子逻辑自动生成实现骨架

```bash
python cuda-kernel-opt-skill/skills/cuda-optimized-skill/operator-optimize-loop/scripts/operator_backend_synth.py \
  --name=demo_gemm \
  --logic="batched matmul for inference" \
  --op-type=matmul \
  --backend=auto \
  --m=1024 --n=1024 --k=1024 \
  --output-dir=/tmp/op-workspace
```

### 2) 在生成目录中补全 kernel 实现

- CUDA：补全 `kernel_cuda.cu`
- Triton：补全 `kernel_triton.py`

### 3) 运行 harness 做 CPU 基线精度校验

```bash
python /tmp/op-workspace/demo_gemm/benchmark_harness.py --backend=cuda
```

### 4) 进入迭代优化循环

```bash
python cuda-kernel-opt-skill/skills/cuda-optimized-skill/operator-optimize-loop/scripts/optimize_loop.py \
  <kernel_file> --backend=<cuda|triton> --max-iterations=<N> --ref=<cpu_reference.py>
```

## 对三大目标的覆盖说明

### 1) 端到端推理优化覆盖

- 保留并强化 `e2e-inference-opt-skill` 的场景化路由（多模型类型覆盖）。
- 用 `auto-profiling` 的 baseline/candidate/evaluator 闭环控制实验质量。

### 2) 大模型 serving 优化覆盖

- `llm-serving-opt-skill` 维持 SGLang/vLLM/TensorRT-LLM 路由与 benchmark/profile/deployment 分层。
- 与算子层形成“服务瓶颈 -> kernel 下钻 -> 回灌 serving”的双向闭环。

### 3) 算子深度优化覆盖

- 从“仅 CUDA 优化”升级到“CUDA + Triton 可选生成”。
- CPU baseline 作为精度锚点，避免只追求速度。
- 支持从逻辑描述快速起步，降低算子优化门槛。

## 下一步建议（继续增强）

1. 增加 `conv2d`、`attention`、`softmax`、`rmsnorm` 等更多算子模板。
2. 增加自动生成 PyTorch extension 的 CUDA 构建脚本（减少手工编译成本）。
3. 把 harness 的输出 schema 与 `auto-profiling` aim 合约深度对齐（零改动接入）。
4. 引入多次采样与统计稳健性判定（p50/p95 + 最小收益阈值）。
5. 增加分布式 serving profile 入口（prefill/decode 分离 + interconnect 指标）。
