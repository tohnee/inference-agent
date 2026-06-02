# Auto-Profiling

Auto-Profiling 是本仓库的优化驱动引擎。

普通用户流程只有一条主线：

1. 用户编写或生成一个 `aim.md`。
2. runtime 校验 aim 合约是否完整。
3. runtime 建立 baseline/profile/exactness 证据。
4. runtime 根据 `scenario` 路由到合适的底层 skill lane。
5. runtime 写出 bounded `next_candidate` 计划。
6. evaluator 按 exactness-first 规则决定 keep 或 reject。

## 用户需要知道的文件

- `aim.md`：唯一推荐的用户编辑入口。
- `bootstrap_aim.py`：为 E2E、LLM serving、CUDA/kernel 生成完整 aim。
- `runner.py`：执行优化闭环。
- `skill_routes.json`：把 `scenario` 映射到三大底层 skills 库。
- `aim_schema.json`：定义 aim 必填字段。
- `templates/state/`：仓库内置状态模板，会复制到目标项目的运行目录。
- `examples/`：历史场景化 aim 示例，不再是主入口。

## 支持场景

| `scenario` | 底层 skill 库 |
| --- | --- |
| `e2e-inference` | `../e2e-inference-opt-skill/` |
| `llm-serving` | `../llm-serving-opt-skill/` |
| `cuda-kernel` | `../cuda-kernel-opt-skill/` |
| `operator-kernel` | `../cuda-kernel-opt-skill/` 加 operator synthesis skills |

## 生成 aim

```bash
python3 bootstrap_aim.py \
  --mode e2e \
  --profile diffusion \
  --project-name demo-diffusion \
  --target-repo-path /path/to/target-repo \
  --output aim.md
```

```bash
python3 bootstrap_aim.py \
  --mode llm-serving \
  --profile vllm \
  --project-name demo-vllm \
  --target-repo-path /path/to/target-repo \
  --output aim.md
```

```bash
python3 bootstrap_aim.py \
  --mode cuda-kernel \
  --profile triton \
  --project-name demo-kernel \
  --target-repo-path /path/to/target-repo \
  --output aim.md
```

CUDA/kernel profiles 包括 `cuda`、`triton`、`cutlass`、`operator`。

## 运行闭环

```bash
python3 runner.py init --aim aim.md
python3 runner.py collect-env --aim aim.md
python3 runner.py baseline --aim aim.md
python3 runner.py autopilot --aim aim.md --iterations 3
python3 runner.py status --aim aim.md
```

`autopilot` 会在 candidate evaluation 前，把 `next_candidate.md` 和 `next_candidate.json` 写入目标项目的 `.auto-profiling/` 状态目录。

## 正确性策略

默认模式是 `exact-parity`。如果输出正确性、cache 语义、请求隔离或声明的算法等价性失败，即使性能提升也会被拒绝。

只有在 aim 显式声明逻辑等价和数值容忍范围时，才应使用 `bounded-tolerance`。
