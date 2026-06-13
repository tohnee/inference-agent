# Inference Agent

`inference-agent` 现在收敛为一个更简单的产品形态：**用户提交一个 `aim.md`，`auto-profiling` 作为优化驱动引擎，按 profile 证据路由到三大底层 skills 库，并用 exactness-first 规则决定保留或回滚。**

## 核心入口

普通用户只需要关心一个文件：

- `auto-profiling/aim.md`：唯一推荐的用户目标合约。

`aim.md` 描述目标仓库、baseline/profile/exactness 命令、指标方向、允许改动范围和禁止改动范围。运行时会校验它是否完整，并把执行状态写入目标项目的 `.auto-profiling/` 目录。

## 架构

```text
inference-agent/
├── auto-profiling/              # aim -> baseline -> profile -> candidate plan -> evaluate -> handoff
├── e2e-inference-opt-skill/     # 端到端推理优化知识库
├── llm-serving-opt-skill/       # LLM serving 优化知识库
├── cuda-kernel-opt-skill/       # CUDA / operator / kernel 优化知识库
├── docs/                        # 架构说明、历史计划与归档材料
└── tests/                       # runtime、schema、目录结构和工具测试
```

## auto-profiling 的职责

`auto-profiling` 是优化驱动引擎，而不是单纯的 benchmark wrapper。它负责：

1. 读取并校验 `aim.md`。
2. 建立可信 baseline。
3. 执行 profile / exactness / metric 命令。
4. 根据 `scenario` 和 profile artifact 选择优化 lane。
5. 写出 `.auto-profiling/next_candidate.md` 和 `.auto-profiling/next_candidate.json`。
6. 只允许一个 bounded candidate 实验进入评估。
7. 用 exactness-first evaluator 决定 keep 或 revert。
8. 写出 evaluator report、experiment log 和 handoff 文档。

## 三大底层 skills 库

| 场景 | `aim.md` 中的 `scenario` | 底层 skill 库 |
| --- | --- | --- |
| 端到端推理链路 | `e2e-inference` | `e2e-inference-opt-skill/` |
| LLM 在线服务 | `llm-serving` | `llm-serving-opt-skill/` |
| CUDA / operator / kernel | `cuda-kernel` 或 `operator-kernel` | `cuda-kernel-opt-skill/` |

路由配置位于 `auto-profiling/skill_routes.json`，新增场景时优先改配置，而不是改 runtime 主逻辑。

## 快速开始

### 1. 生成或编辑唯一 aim

推荐直接生成到 `auto-profiling/aim.md`：

```bash
python3 auto-profiling/bootstrap_aim.py \
  --mode llm-serving \
  --profile vllm \
  --project-name demo-vllm \
  --target-repo-path /path/to/target-repo \
  --output auto-profiling/aim.md
```

CUDA/kernel 场景也走同一个入口：

```bash
python3 auto-profiling/bootstrap_aim.py \
  --mode cuda-kernel \
  --profile triton \
  --project-name demo-kernel \
  --target-repo-path /path/to/target-repo \
  --output auto-profiling/aim.md
```

历史/场景化模板已降级为 examples，位于 `auto-profiling/examples/`。

### 2. 运行优化闭环

```bash
cd auto-profiling
python3 runner.py init --aim aim.md
python3 runner.py doctor --aim aim.md
python3 runner.py collect-env --aim aim.md
python3 runner.py baseline --aim aim.md
python3 runner.py autopilot --aim aim.md --iterations 3
python3 runner.py status --aim aim.md
```

如果安装了 `uv`，可以把 `python3 runner.py` 替换为 `uv run runner.py`。

## 面向用户的预检

在真正启动 benchmark/profile 前，建议先运行：

```bash
cd auto-profiling
python3 runner.py doctor --aim aim.md
```

`doctor` 会生成 `<target_repo>/.auto-profiling/doctor_report.md` 和 `doctor_report.json`，集中检查 aim 完整性、目标仓库、git/revert 安全性、关键命令、指标/正确性合约、输出 artifact 路径和 mutation scope。它默认返回 0，便于在交互流程中阅读报告；如需在 CI 中遇到失败项直接失败，可追加 `--strict`。

## aim 是稳定用户 API

`auto-profiling/aim_schema.json` 定义了 `aim.md` 的必填字段，包括：

- `scenario`
- `target_repo_path`
- `baseline_run_command`
- `baseline_profile_command`
- `metric_output_path`
- `exactness_output_path`
- `exactness_check_command`
- `target_metric_name`
- `target_metric_direction`
- `allowed_mutations`
- `blocked_by_default`

如果字段缺失，runtime 会在执行前失败，并提示用户补齐 `aim.md`。

## 运行状态和模板边界

- `auto-profiling/templates/state/` 是仓库自带的状态模板。
- `<target_repo>/.auto-profiling/` 是用户目标项目运行时生成的状态目录。
- `docs/archive/` 保存历史计划、过程记录和探索材料，不是普通用户入口。

## 正确性原则

默认是 **exact-parity**：只要输出不一致，优化就失败。只有用户在 `aim.md` 中显式启用 bounded tolerance，并说明等价性和容忍范围，runtime 才允许有限数值误差。

## 生产化执行护栏

- `max_runtime_per_experiment` 会作为 install / warmup / baseline / exactness / profile 命令的默认超时，避免外部 benchmark 或 profiler 无限挂起。
- 如需给命令执行设置更严格的上限，可以在 `aim.md` 中添加 `command_timeout_seconds`；该字段优先于 `max_runtime_per_experiment`。
- 超时命令会以 `exit_code: 124`、`timed_out: true` 写入命令结果，并在 required command 路径抛出明确的 timeout 错误。
- 最新生产化 review 和后续硬化清单见 `docs/plans/2026-06-03-production-readiness-review.md`。
