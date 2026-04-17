# Inference Agent

`inference-agent` 是一个面向推理优化工程的系统化仓库。

它把“推理优化知识库”和“可运行的优化 harness”放在同一个工程里，目标不是只写一堆经验文档，而是让你可以围绕真实 baseline、真实环境、真实 exactness 契约去持续做性能优化。

这个仓库适合以下任务：

- 小模型或非 LLM 推理链路的端到端优化
- 大模型在线推理服务优化
- CUDA / CUTLASS / Triton kernel 级问题下钻
- 构建 exactness-first、可恢复、可审计的推理优化流程

## 核心原则

这个工程默认遵循以下原则：

- 结果一致优先，性能提升第二
- 先建立可信 baseline，再做优化
- 一次只做一个有边界的实验
- 所有保留决策都需要 exactness 和 metric 证据
- 优化要能恢复、能交接、能重复执行

如果你的目标只是“随便调快一点”，这个仓库并不是为那种工作方式设计的。

## 仓库结构

仓库由 4 个核心部分组成：

- `e2e-inference-opt-skill/`
  - 面向小模型、非 LLM、多阶段推理链路的端到端优化知识库
- `llm-serving-opt-skill/`
  - 面向 SGLang、vLLM、TensorRT-LLM、Triton、PyTorch serving 的服务层优化知识库
- `cuda-kernel-opt-skill/`
  - 面向 CUDA / CUTLASS / Triton kernel、算子、NCU 分析、crash triage 的深度优化知识库
- `auto-profiling/`
  - 面向真实项目执行 baseline、候选实验、评估、handoff 的最小运行时

一句话理解：

- 三个 `*-opt-skill` 模块负责告诉你“该优化什么、先走哪条 lane”
- `auto-profiling` 负责告诉你“如何把优化过程真正跑起来并留下工件”

## 三种使用场景

### 1. E2E 推理链路优化

适用情况：

- 业务模型不是典型 LLM serving
- 问题横跨 preprocess、forward、postprocess、IO、缓存、部署
- 你需要做系统级而不是单算子级优化

从这里开始：

- [e2e-inference-opt-skill](e2e-inference-opt-skill/SKILL.md)
- [aim.e2e.md](auto-profiling/aim.e2e.md)

### 2. LLM Serving 优化

适用情况：

- 你关心 TTFT、TPOT、ITL、tokens/s、req/s
- 你在做 SGLang、vLLM、TensorRT-LLM、Triton 或自定义 PyTorch 服务
- 你需要分析 scheduler、KV cache、prefix cache、deployment、benchmark

从这里开始：

- [llm-serving-opt-skill](llm-serving-opt-skill/SKILL.md)
- [aim.llm-serving.md](auto-profiling/aim.llm-serving.md)

### 3. CUDA / Kernel 级优化

适用情况：

- 你已经通过 profile 把瓶颈定位到算子或 kernel
- 你需要做 NCU、kernel benchmark、custom kernel workflow、CUDA crash triage
- 你需要从 serving 层继续下钻到 operator / kernel 层

从这里开始：

- [cuda-kernel-opt-skill](cuda-kernel-opt-skill/SKILL.md)
- [aim.cuda-kernel.md](auto-profiling/aim.cuda-kernel.md)

## 推荐上手路径

如果你是第一次使用这个仓库，推荐按下面的顺序：

1. 确定当前问题属于 `e2e-inference`、`llm-serving` 还是 `cuda-kernel`
2. 阅读对应的顶层 `SKILL.md`
3. 进入 `auto-profiling/`，选择对应的 `aim.*.md`
4. 在 `aim` 中填写目标仓库、baseline 命令、exactness 检查命令和 metric 输出路径
5. 运行 `init`、`collect-env`、`baseline`
6. 观察 `.auto-profiling/` 下的 contract、evaluator report 和 handoff
7. 再做单轮 bounded candidate experiment

## 5 分钟快速开始

以下示例以 `llm-serving` 场景为例。

先进入运行时目录：

```bash
cd auto-profiling
```

优先使用 `uv` 运行；如果环境里没有 `uv`，可以直接用 `python3`。

### 第一步：选择一个场景模板

编辑：

- `aim.e2e.md`
- `aim.llm-serving.md`
- `aim.cuda-kernel.md`

你至少需要填这些字段：

- `target_repo_path`
- `baseline_run_command`
- `baseline_profile_command`
- `metric_output_path`
- `exactness_output_path`
- `exactness_check_command`
- `target_metric_name`
- `target_metric_direction`

### 第二步：初始化工作区

```bash
uv run runner.py init --aim aim.llm-serving.md
```

如果没有 `uv`：

```bash
python3 runner.py init --aim aim.llm-serving.md
```

### 第三步：先摸清环境

```bash
uv run runner.py collect-env --aim aim.llm-serving.md
```

这一步会收集：

- shell 与包管理器回退结果
- Python / 平台 / 机器信息
- 常见工具路径
- 常见依赖包版本
- 可用时附加 `vllm collect-env` 摘要

### 第四步：记录可信 baseline

```bash
uv run runner.py baseline --aim aim.llm-serving.md
```

### 第五步：查看当前状态

```bash
uv run runner.py status --aim aim.llm-serving.md
```

### 第六步：做一轮 bounded candidate

```bash
uv run runner.py candidate --aim aim.llm-serving.md --label exp-001
```

## `auto-profiling` 是怎么工作的

`auto-profiling/` 是这个仓库里最接近“可执行系统”的部分。

关键文件：

- [runner.py](auto-profiling/runner.py)
- [scorer.py](auto-profiling/scorer.py)
- [README.md](auto-profiling/README.md)
- [aim.md](auto-profiling/aim.md)

运行时会在目标仓库写入 `.auto-profiling/` 工件，用于持续记录优化过程：

- `current_contract.md`
- `evaluator_report.md`
- `next_handoff.md`
- `session_state.json`
- `experiment_log.md`
- `experiment_log.jsonl`

这意味着：

- 你可以中断后继续
- 你可以在新 session 恢复
- 你可以审计每一轮实验为什么被保留或拒绝

## 三大模块怎么协作

### `e2e-inference-opt-skill`

负责：

- baseline
- profiling
- roofline
- memory / IO
- parallelism
- pipeline overlap
- deployment

适合“系统级端到端链路”问题。

### `llm-serving-opt-skill`

负责：

- serving baseline
- benchmark workflow
- profile analysis
- KV cache / prefix cache / scheduler
- deployment cookbook

适合“服务层 LLM 推理系统”问题。

### `cuda-kernel-opt-skill`

负责：

- CUDA crash debug
- profile triage
- backend selection
- custom kernel workflow
- vendored `cuda-optimized-skill`

适合“算子或 kernel 已经成为主瓶颈”的问题。

## 推荐工作流

一个常见的完整流程是：

1. 用 `llm-serving-opt-skill` 或 `e2e-inference-opt-skill` 先完成问题定性
2. 在 `auto-profiling` 中写好 `aim`
3. 跑 `collect-env`
4. 跑 `baseline`
5. 从 profile / benchmark / correctness 证据中选择一条最安全的优化 lane
6. 只做一个变更
7. 重新跑 exactness 和 metric
8. 看 evaluator report 决定 keep 或 revert
9. 必要时再升级到 `cuda-kernel-opt-skill`

## 包管理与环境策略

当前运行时的默认策略是：

- 项目安装优先 `uv`
- 没有 `uv` 时回退到 `pip`
- 如果你在 `conda` 环境中运行，也可以通过当前环境的 `python3` 直接执行
- 如果需要显式激活环境，可以在 `aim` 中设置 `python_env_command`

也就是说，最稳妥的方式通常是：

- 先进入你自己的目标 Python 环境
- 再运行 `runner.py`

## 如何验证仓库可用

### 技能目录测试

在仓库根目录运行：

```bash
python3 -m unittest tests/test_skill_catalog.py -v
```

### `auto-profiling` 运行时测试

在 `auto-profiling/` 目录运行：

```bash
cd auto-profiling
python3 -m unittest tests/test_runtime.py -v
```

## 适合谁使用

这个仓库适合：

- 做推理优化的工程师
- 做 LLM serving 系统优化的工程师
- 需要把优化流程沉淀成可复用知识和可执行 runtime 的团队
- 希望把 exactness-first 原则真正落到日常优化中的使用者

## 常见误区

- 直接开始调参，不先建立 baseline
- 把精度漂移当成性能优化的默认代价
- 一轮实验改太多变量
- 不输出结构化 metric / exactness JSON
- 不区分 E2E、serving、kernel 三种问题层次

## 进一步阅读

- [auto-profiling/README.md](auto-profiling/README.md)
- [e2e-inference-opt-skill/SKILL.md](e2e-inference-opt-skill/SKILL.md)
- [llm-serving-opt-skill/SKILL.md](llm-serving-opt-skill/SKILL.md)
- [cuda-kernel-opt-skill/SKILL.md](cuda-kernel-opt-skill/SKILL.md)

如果你只想记住一句话：

> 先选对场景，再写 `aim`，再让 `auto-profiling` 带着 exactness 契约去跑优化闭环。
