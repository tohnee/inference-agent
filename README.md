# Inference Agent

`inference-agent` 是一个面向**推理优化工程**的完整仓库：

- 不是只给“建议”，而是给“可执行流程”
- 不是只看吞吐/延迟，而是坚持 **exactness-first（结果一致优先）**
- 不是一次性调参，而是支持可恢复、可审计、可交接的持续优化

---

## 目录

1. [项目定位与目标](#项目定位与目标)
2. [核心设计原则](#核心设计原则)
3. [仓库总体架构](#仓库总体架构)
4. [四大模块详解](#四大模块详解)
5. [安装与环境准备](#安装与环境准备)
6. [快速开始（10 分钟）](#快速开始10-分钟)
7. [auto-profiling 详细用法](#auto-profiling-详细用法)
8. [三类优化场景最佳实践](#三类优化场景最佳实践)
9. [在 Codex/智能体中独立使用 skills](#在-codex智能体中独立使用-skills)
10. [常见问题与排障](#常见问题与排障)
11. [扩展建议](#扩展建议)

---

## 项目定位与目标

这个仓库解决的是三类核心问题：

1. **端到端推理链路优化（E2E）**
   - 小模型、Diffusion、通用 DL、Transformer、SAM、ViT、树模型等
2. **大模型在线推理服务优化（LLM Serving）**
   - SGLang / vLLM / TensorRT-LLM / Triton / PyTorch serving
3. **算子与 Kernel 深度优化（CUDA / CUTLASS / Triton）**
   - NCU 分析、崩溃排障、backend 选择、算子迭代闭环

仓库最终目标：

- 用户只需要在 `auto-profiling` 中编辑需求（`aim`）
- 系统可自动拉起 baseline、验证正确性、迭代候选实验
- 并把过程工件完整记录，便于下次继续和多人协作

---

## 核心设计原则

- **正确性优先**：性能提升不能以结果回归为代价
- **baseline 先行**：没有可信 baseline，不做优化结论
- **单变量实验**：每轮只做有边界的一个变化
- **证据驱动决策**：保留/回滚必须有 metric + exactness 证据
- **可恢复可交接**：中断后可继续、换 session 可续跑

---

## 仓库总体架构

可以把仓库理解为“3 个知识模块 + 1 个执行模块”：

- `e2e-inference-opt-skill/`：E2E 优化知识系统
- `llm-serving-opt-skill/`：LLM serving 优化知识系统
- `cuda-kernel-opt-skill/`：算子/kernel 优化知识系统
- `auto-profiling/`：执行优化闭环的运行时

一句话：

- `*-opt-skill` 决定“该优化什么、先走哪条 lane”
- `auto-profiling` 决定“如何稳定地把优化流程跑起来”

---

## 四大模块详解

## 1) `e2e-inference-opt-skill`

适用：

- 非 LLM 或小模型端到端链路
- 多阶段 pipeline：preprocess -> infer -> postprocess -> store/serve
- 覆盖模型家族：small-model / diffusion / DL / transformer / SAM / ViT / tree

入口：

- [e2e-inference-opt-skill/SKILL.md](e2e-inference-opt-skill/SKILL.md)
- [e2e-inference-opt-skill/references/09_model_family_coverage.md](e2e-inference-opt-skill/references/09_model_family_coverage.md)

---

## 2) `llm-serving-opt-skill`

适用：

- 在线 serving 场景（TTFT / TPOT / ITL / tokens/s / req/s）
- SGLang / vLLM / TensorRT-LLM 等 backend 对比与调优
- KV cache、scheduler、deployment、distributed AI-infra 深度优化

入口：

- [llm-serving-opt-skill/SKILL.md](llm-serving-opt-skill/SKILL.md)
- [llm-serving-opt-skill/references/09_distributed_ai_infra.md](llm-serving-opt-skill/references/09_distributed_ai_infra.md)

---

## 3) `cuda-kernel-opt-skill`

适用：

- 瓶颈已下钻到 operator/kernel
- CUDA crash triage、NCU 定位、backend 选择
- CUDA / CUTLASS / Triton 深度优化

新增能力（可直接用）：

- “算子逻辑 -> CPU baseline -> CUDA/Triton scaffold” 自动生成
- 脚本：`operator_backend_synth.py`

入口：

- [cuda-kernel-opt-skill/SKILL.md](cuda-kernel-opt-skill/SKILL.md)
- [operator-backend-synthesis-skill](cuda-kernel-opt-skill/skills/operator-backend-synthesis-skill/SKILL.md)

---

## 4) `auto-profiling`

这是项目的执行中枢。它负责：

- 初始化状态
- 环境采集
- baseline 记录
- candidate 评估与决策
- autopilot 自动迭代
- handoff 文档输出

核心文件：

- [auto-profiling/runner.py](auto-profiling/runner.py)
- [auto-profiling/scorer.py](auto-profiling/scorer.py)
- [auto-profiling/bootstrap_aim.py](auto-profiling/bootstrap_aim.py)
- [auto-profiling/README.md](auto-profiling/README.md)

---

## 安装与环境准备

建议：

- Python 3.10+
- Git
- 可选：`uv`（推荐）
- 可选（GPU/KERNEL 场景）：`nvidia-smi`、`nvcc`、`ncu`

进入运行时目录：

```bash
cd auto-profiling
```

建议使用 `uv`：

```bash
uv run runner.py status --aim aim.md
```

没有 `uv` 时：

```bash
python3 runner.py status --aim aim.md
```

---

## 快速开始（10 分钟）

下面以 LLM serving 为例。

### Step 1：生成或选择 `aim`

你有两种方式：

1) 手工编辑模板：
- `aim.e2e.md`
- `aim.llm-serving.md`
- `aim.cuda-kernel.md`

2) 用生成器快速初始化（推荐）：

```bash
python3 auto-profiling/bootstrap_aim.py \
  --mode llm-serving \
  --profile vllm \
  --project-name demo-vllm \
  --target-repo-path /path/to/target-repo \
  --output auto-profiling/aim.generated.md
```

### Step 2：初始化

```bash
uv run runner.py init --aim aim.generated.md
```

### Step 3：收集环境

```bash
uv run runner.py collect-env --aim aim.generated.md
```

### Step 4：记录 baseline

```bash
uv run runner.py baseline --aim aim.generated.md
```

### Step 5：自动迭代（one-click）

```bash
uv run runner.py autopilot --aim aim.generated.md --iterations 3 --label-prefix auto
```

---

## auto-profiling 详细用法

## 1) 命令概览

- `init`：初始化工作区与工件
- `collect-env`：采集环境指纹
- `baseline`：建立可信基线
- `candidate`：执行一轮候选实验并可晋升最优
- `evaluate`：执行评估但不晋升
- `loop`：单步循环（无 baseline 时先建 baseline）
- `autopilot`：连续多轮自动迭代
- `status`：查看当前状态
- `handoff`：输出下一轮接力说明

## 2) 关键参数

`aim` 至少应包含：

- `target_repo_path`
- `baseline_run_command`
- `baseline_profile_command`
- `metric_output_path`
- `exactness_output_path`
- `exactness_check_command`
- `target_metric_name`
- `target_metric_direction`

可选增强：

- `command_retry_count`：命令重试次数（提升稳定性）
- `allowed_mutations` / `blocked_by_default`
- `known_bottlenecks` / `suspected_safe_lanes`

## 3) 运行时工件（在目标仓库 `.auto-profiling/`）

- `current_contract.md`
- `evaluator_report.md`
- `next_handoff.md`
- `skill_route_plan.md`
- `session_state.json`
- `baseline_snapshot.json`
- `best_result.json`
- `experiment_log.md`
- `experiment_log.jsonl`
- `progress.md` / `worklog.md` / `findings.md`

## 4) autopilot 的行为

`autopilot` 会：

1. 若缺少 baseline，自动补建
2. 连续运行 `N` 轮 candidate（`--iterations`）
3. 每轮按 exactness + metric 决策 keep/reject
4. 自动刷新 `next_handoff.md` 与 `skill_route_plan.md`

这使得“只编辑需求 + 一键迭代”成为可行工作流。

---

## 三类优化场景最佳实践

## A. E2E 推理优化

推荐路径：

1. 用 `bootstrap_aim.py --mode e2e --profile <family>` 生成初始 aim
2. 跑 baseline 与 profile
3. 先修系统级瓶颈（IO/copy/schedule）
4. 必要时再下钻 kernel

常见 profile：

- `small-model`
- `diffusion`
- `dl`
- `transformer`
- `sam`
- `vit`
- `tree`

---

## B. LLM Serving 优化

推荐路径：

1. 用 `bootstrap_aim.py --mode llm-serving --profile <backend>`
2. 明确核心指标（TTFT/TPOT/ITL/throughput）
3. 先做服务层调度/KV 策略优化
4. 再针对瓶颈下钻到 operator/kernel

支持 backend：

- `sglang`
- `vllm`
- `trtllm`

---

## C. Operator / Kernel 优化

先生成算子脚手架：

```bash
python cuda-kernel-opt-skill/skills/cuda-optimized-skill/operator-optimize-loop/scripts/operator_backend_synth.py \
  --name demo_gemm \
  --logic "batched matmul for inference" \
  --op-type matmul \
  --backend auto \
  --m 1024 --n 1024 --k 1024 \
  --output-dir /tmp/op-workspace
```

输出内容：

- `cpu_reference.py`
- `kernel_cuda.cu` 或 `kernel_triton.py`
- `benchmark_harness.py`
- `manifest.json`

然后再进入 `optimize_loop.py` 做有边界的多轮优化。

---

## 在 Codex/智能体中独立使用 skills

即使不跑 `autopilot`，三条方向的 skills 也可以独立使用：

- E2E：`e2e-inference-opt-skill/SKILL.md`
- LLM：`llm-serving-opt-skill/SKILL.md`
- Kernel：`cuda-kernel-opt-skill/SKILL.md`

同时，`auto-profiling` 会产出 `skill_route_plan.md`，帮助新 session 快速对齐当前场景和推荐技能链路。

---

## 常见问题与排障

### Q1：为什么 baseline 必须先通过 exactness？

因为 baseline 是后续全部比较的参考点。如果 baseline 本身不可信，任何“提升”都没有意义。

### Q2：autopilot 会不会盲目保留劣化结果？

不会。每轮都经过 scorer 的 exactness + metric 判定，未通过 exactness 或指标未提升都会 reject。

### Q3：如何提升运行稳定性？

在 `aim` 中设置：

- `command_retry_count: 2`（或更高）

并尽量保证 `baseline_run_command` 和 `exactness_check_command` 可重复执行。

### Q4：如何在团队中交接？

优先查看目标仓库 `.auto-profiling/`：

- `next_handoff.md`
- `evaluator_report.md`
- `experiment_log.md`
- `skill_route_plan.md`

---

## 扩展建议

如果你要继续增强这个仓库，建议优先做：

1. 更丰富的算子模板（conv/attention/rmsnorm/softmax）
2. 分布式 serving 的更细粒度调度与资源隔离策略
3. 统计稳健判定（多次采样 + 置信区间 + 最小收益阈值）
4. 与 CI/CD 的自动回归集成

---

## 相关文档

- [README.en.md](README.en.md)
- [auto-profiling/README.md](auto-profiling/README.md)
- [e2e-inference-opt-skill/SKILL.md](e2e-inference-opt-skill/SKILL.md)
- [llm-serving-opt-skill/SKILL.md](llm-serving-opt-skill/SKILL.md)
- [cuda-kernel-opt-skill/SKILL.md](cuda-kernel-opt-skill/SKILL.md)

