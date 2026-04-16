# Auto-Profiling

Auto-Profiling 是构建在本地 skills 之上的一个长运行推理优化 harness。

它面向这样的使用方式：

- 人类提供环境、baseline 代码和验证命令
- 人类只编辑 `aim.md`
- runtime 在目标仓库下初始化 `.auto-profiling/` 工作区
- agent 以“单轮单变量实验”的方式持续执行、评估、保留或回滚，并把下一轮 handoff 写到磁盘

默认哲学是：

- 结果一致优先
- 性能改进其次
- 每轮只做一个有边界的实验
- 评估必须独立、怀疑，而不是自我表扬

## 项目包含什么

### Runtime

- `runner.py`：执行 `aim.md` 契约
- `scorer.py`：统一比较 reference 与 candidate
- `pyproject.toml`：支持 `uv`
- `log_schema.json`：结构化日志 schema

runtime 现在还会做环境探测：

- 优先使用 `zsh`
- 如果没有 `zsh`，自动回退到 `bash`
- 如果两者都没有，则回退到系统默认 shell
- 项目安装时优先用 `uv`
- 如果没有 `uv`，自动回退到 `pip`
- 采集更丰富的 runtime 元数据，包括 Python 版本、平台、机器信息、工具路径、包版本，以及可选的 `vllm collect-env` 证据

### 人类控制面

- `aim.md`：唯一必填的人类运行契约
- `aim.zh-CN.md`：中文版模板
- `aim.e2e.md`：面向小模型或非 LLM 端到端链路优化的模板
- `aim.llm-serving.md`：面向大模型推理服务优化的模板
- `aim.cuda-kernel.md`：面向 CUDA/CUTLASS/Triton kernel 优化的模板

### Harness 状态目录

runtime 会在目标仓库中创建并维护 `.auto-profiling/`：

- `experiment_log.md`
- `experiment_log.jsonl`
- `baseline_snapshot.json`
- `best_result.json`
- `session_state.json`
- `current_contract.md`
- `evaluator_report.md`
- `next_handoff.md`
- `task_plan.md`
- `findings.md`
- `progress.md`
- `worklog.md`

### 参考文档

- `references/01_operating_model.md`
- `references/02_experiment_loop.md`
- `references/03_mutation_lanes.md`
- `references/04_exactness_gate.md`
- `references/05_artifacts_and_scoring.md`
- `references/06_harness_patterns.md`

## 架构总览

Auto-Profiling 采用 planner / generator / evaluator 风格的 harness 思想，但 runtime 保持尽量轻量。

### Planner 角色

Planner 角色通过以下文件体现：

- `aim.md`
- `.auto-profiling/current_contract.md`

这里负责写清楚当前目标、exactness mode、选定的 mutation lane，以及本轮实验边界。

现在 runtime 还会把 lane 作为显式工件写出来：

- 当前 `scenario`
- 目标优化模块
- 推荐 skill 路由
- 推荐 aim 模板

### Generator 角色

Generator 角色就是实际的有边界优化尝试：

- 跑 baseline
- 应用一处 candidate change
- 先重跑 exactness
- 再重跑性能测量

runtime 本身不把“变异引擎”写死。它负责提供循环、状态和工件，让 Claude 能在受控边界里改代码。

## 三种场景入口

Auto-Profiling 现在作为统一入口，支持三类场景：

- `e2e-inference`：使用 [e2e-inference-opt-skill](file:///Users/tc/Downloads/推理优化skills/e2e-inference-opt-skill/SKILL.md) 做小模型和非 LLM 端到端链路优化
- `llm-serving`：使用 [llm-serving-opt-skill](file:///Users/tc/Downloads/推理优化skills/llm-serving-opt-skill/SKILL.md) 做 SGLang、vLLM、TensorRT-LLM、Triton 或自定义 PyTorch 服务优化
- `cuda-kernel`：当瓶颈已经下钻到算子或 kernel 时，使用 [cuda-kernel-opt-skill](file:///Users/tc/Downloads/推理优化skills/cuda-kernel-opt-skill/SKILL.md)

### Evaluator 角色

Evaluator 角色通过以下部分与 generator 分离：

- `scorer.py`
- `.auto-profiling/evaluator_report.md`

Evaluator 是否保留当前候选，依据的是：

- exactness
- metric 是否提升
- 是否越界修改
- 稳定性
- 可复现性

这样可以避免系统一边做改动，一边对自己过度乐观。

### Handoff 角色

长时间运行时，依靠这些文件保持连续性：

- `.auto-profiling/session_state.json`
- `.auto-profiling/next_handoff.md`

这样即使你换一个新 session，也可以从磁盘恢复，而不依赖超长上下文。

## Exactness 模型

### 1. Exact-Parity 模式

这是默认模式。

适用条件：

- 逻辑不变
- 算法不变
- 设备类别基本一致
- 数值契约不变

规则：

- 任何输出不一致都判定失败

### 2. Bounded-Tolerance 模式

只有在以下条件都满足时才允许使用：

- 逻辑等价
- 算法等价
- 差异来自已声明的跨设备执行或 precision 切换
- `aim.md` 中明确声明了 `abs_tolerance` 和 `rel_tolerance`

典型场景：

- CPU reference vs GPU candidate
- FP32 reference vs BF16 candidate

规则：

- 任何一个误差阈值超出都失败
- 如果逻辑或算法本身不等价，容差模式不能“洗白”这个实验

## Runtime 命令

优先在本目录下用 `uv` 运行。

如果环境里没有 `uv`，可以直接用 Python 入口：

```bash
python3 runner.py status --aim aim.md
```

对于目标项目依赖安装，runtime 会自动选择：

- 如果有 `uv` 且目标项目有 `pyproject.toml`，就用 `uv sync`
- 如果目标项目有 `requirements.txt`，就用 `python -m pip install -r requirements.txt`
- 如果没有 `uv`，但目标项目可编辑安装，就用 `python -m pip install -e .`

### 初始化

```bash
uv run runner.py init --aim aim.md
```

创建或刷新 runtime workspace 与 handoff 骨架。

### 记录 Baseline

```bash
uv run runner.py baseline --aim aim.md
```

运行可信路径，记录 exactness 与性能，并保存 baseline snapshot。

### 评估一个 Candidate

```bash
uv run runner.py candidate --aim aim.md --label exp-001
```

执行一轮有边界 candidate 实验；如果它在 exactness 契约下优于当前 reference，就会被提升为新的 best。

### 只评估不晋升

```bash
uv run runner.py evaluate --aim aim.md --label eval-001
```

执行同样的评估逻辑，但不会把 candidate 晋升为 best-known result。

### Long-Running Loop 单步执行

```bash
uv run runner.py loop --aim aim.md --label loop-001
```

执行一个可恢复的 harness step：

- 如果还没有 baseline，就先创建 baseline
- 否则基于当前 best reference 跑一轮 candidate

### 查看状态

```bash
uv run runner.py status --aim aim.md
```

显示：

- baseline 是否存在
- best result 是否存在
- 当前 session state
- 当前场景、目标模块和推荐 skill 路由
- 当前检测到的 shell 和 package manager
- workspace 路径

### 强化环境检测

```bash
uv run runner.py collect-env --aim aim.llm-serving.md
```

在真正做优化前先采集更强的环境指纹，包括：

- shell 和 package manager 的回退结果
- Python、平台和机器元数据
- `git`、`nvidia-smi`、`nvcc`、`uv`、`pip`、`vllm` 等工具路径
- 轻量命令诊断结果
- 常见 serving / kernel 栈的已安装包版本
- 当环境里有 `vllm` 时，附加 `vllm collect-env` 摘要

### 生成 Handoff

```bash
uv run runner.py handoff --aim aim.md
```

写出并打印下一轮 resume 所需的 handoff 摘要。

## 如何填写 `aim.md`

最少需要这些字段：

- `target_repo_path`
- `baseline_run_command`
- `baseline_profile_command`
- `metric_output_path`
- `exactness_output_path`
- `exactness_check_command`
- `target_metric_name`
- `target_metric_direction`

强烈建议填写：

- `allowed_mutations`
- `blocked_by_default`
- `known_bottlenecks`
- `suspected_safe_lanes`
- `max_iterations_per_session`
- `max_runtime_per_experiment`

环境相关建议：

- `install_command` 留空时，runtime 会自动探测并选择 `uv` 或 `pip`
- 如果你想显式表达“自动选择”，也可以写 `install_command: auto`
- 只有在必须手动激活环境时，才填写 `python_env_command`

## 你的命令应输出什么

你的 baseline / candidate 命令需要输出机器可读 JSON。

### Metric payload

示例：

```json
{
  "metrics": {
    "p95_ms": 8.4,
    "throughput": 120.0
  }
}
```

### Exactness payload

Exact-parity 示例：

```json
{
  "passed": true,
  "mismatch_count": 0
}
```

Bounded-tolerance 示例：

```json
{
  "passed": false,
  "logic_equivalent": true,
  "algorithm_equivalent": true,
  "mismatch_count": 3,
  "max_abs_error": 0.000004,
  "max_rel_error": 0.000003
}
```

最终 exactness 是否通过，由 runtime 根据 `aim.md` 中声明的模式统一判断。

## 它和三大模块是什么关系

`auto-profiling` 是执行入口，三大技能模块提供优化知识和路由。

### `e2e-inference-opt-skill`

路径：

- [e2e-inference-opt-skill](file:///Users/tc/Downloads/推理优化skills/e2e-inference-opt-skill)

职责：

- 小模型和非 LLM 端到端链路的优化知识库
- preprocess / forward / postprocess / cache / overlap / deployment 的路由手册

它回答的是：

- 该分析什么
- 该走哪条优化路线

### `llm-serving-opt-skill`

路径：

- [llm-serving-opt-skill](file:///Users/tc/Downloads/推理优化skills/llm-serving-opt-skill)

职责：

- 大模型推理服务的优化知识库
- TTFT / TPOT / ITL / serving benchmark / 服务部署与服务层瓶颈的路由手册

### `cuda-kernel-opt-skill`

路径：

- [cuda-kernel-opt-skill](file:///Users/tc/Downloads/推理优化skills/cuda-kernel-opt-skill)

职责：

- CUDA / CUTLASS / Triton kernel 优化知识库
- correctness / benchmark / Nsight Compute / custom kernel workflow / strategy-memory 闭环的路由手册

### `auto-profiling`

路径：

- [auto-profiling](file:///Users/tc/Downloads/推理优化skills/auto-profiling)

职责：

- 编排 harness
- runtime 循环
- 工件管理
- keep-or-revert 执行引擎

它回答的是：

- 如何安全地、持续地、可恢复地跑优化实验

### 实际协作方式

一起使用时建议这样：

1. 先确定场景：`e2e-inference`、`llm-serving` 或 `cuda-kernel`
2. 用对应模块判断瓶颈和合理的优化 lane
3. 在对应的 aim 模板中写清楚项目契约
4. 用 `auto-profiling` 执行有边界实验并保存状态
5. 下一轮 mutation lane 选择时，再回到对应模块的 reference 文档

一句话总结：

- `e2e-inference-opt-skill` / `llm-serving-opt-skill` / `cuda-kernel-opt-skill` = 优化大脑
- `auto-profiling` = 优化 harness

## 推荐使用流程

1. 准备一个可运行的 baseline 仓库
2. 填写 `aim.e2e.md`、`aim.llm-serving.md` 或 `aim.cuda-kernel.md`
3. 执行 `uv run runner.py init --aim aim.llm-serving.md`
4. 执行 `uv run runner.py baseline --aim aim.llm-serving.md`
5. 阅读 `.auto-profiling/current_contract.md`
6. 执行 `uv run runner.py collect-env --aim aim.llm-serving.md`
7. 执行一轮 bounded candidate
8. 阅读 `.auto-profiling/evaluator_report.md`
9. 用 `.auto-profiling/next_handoff.md` 进入下一轮 session

## 常见错误

- 把 tolerance mode 当作逻辑改动的豁免
- 一轮实验改太多变量
- 没有输出 JSON metric / exactness 文件
- 没跑 baseline 就开始优化
- 忘记 `auto-profiling` 仍然需要对应场景模块提供优化策略脑图

## 建议的下一步增强

如果你继续演进这个项目，最自然的后续方向是：

- 自动多轮 loop
- 根据 profiler 证据自动生成 contract
- 更细粒度 evaluator rubric
- 直接把三大模块的 lane 知识接入 runtime 帮助器
