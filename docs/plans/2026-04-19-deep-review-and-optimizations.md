# 深度代码审查与优化结果（E2E / CUDA-Triton / LLM Serving）

日期：2026-04-19

## 审查范围

- `auto-profiling/runner.py` 执行稳定性与自治能力
- `auto-profiling/bootstrap_aim.py` 场景覆盖能力
- `cuda-kernel-opt-skill/.../operator_backend_synth.py` 算子工程可用性
- 三个主技能模块（E2E / LLM Serving / CUDA-Kernel）路由与覆盖完整性

## 与主流工程实践的对齐目标

对齐以下实践共识：

- PyTorch / serving 工程：先 baseline + correctness，再做性能决策
- CUDA / Triton 工程：同一输入下进行 CPU vs GPU exactness 校验
- vLLM / SGLang / TRT-LLM 工程：优先 TTFT/TPOT/ITL 与尾延迟稳定性
- 生产优化流程：bounded experiment + 可恢复工件 + 清晰 handoff

## 关键问题与修复

### 1) Autopilot baseline 可信性缺口（已修复）

问题：

- `autopilot` 在 baseline 缺失时会自动建立 baseline，但之前未像 `baseline` 命令那样显式验证 exactness gate。

风险：

- 可能导致“不可信 baseline”进入后续比较链路。

修复：

- 在 autopilot baseline 分支中加入 `compare_runs(record, record)` 的 exactness 判定。
- baseline exactness 不通过时立即中止自动迭代。

### 2) Autopilot 迭代预算控制不充分（已修复）

问题：

- 命令行 `--iterations` 可能超过 aim 的会话预算。

修复：

- 加入 `max_iterations_per_session` 上限约束，最终执行轮数取两者最小值。

### 3) 命令链路稳定性（已增强）

问题：

- 运行命令在瞬时失败（短暂环境抖动）时缺少重试。

修复：

- 引入 `run_required_with_retry`。
- 支持从 aim 读取 `command_retry_count` 统一控制重试。

### 4) 算子合成 harness 的正确性问题（已修复）

问题：

- 之前 CPU 输入和 GPU 输入分别随机生成，可能不是同一组样本。

风险：

- exactness 对比不严格，可能出现伪差异或伪通过。

修复：

- 改为先生成 CPU 输入，再把同一输入搬到 CUDA 侧执行后端算子。
- 增加 `seed` 控制，提升可复现性。

### 5) CUDA scaffold 的可执行性不足（已修复）

问题：

- 之前仅生成 `kernel_cuda.cu`，但 harness 默认导入 `kernel_cuda` Python 模块，运行时不可直接使用。

修复：

- 生成 `kernel_cuda.py` loader（基于 `torch.utils.cpp_extension.load`）自动编译并导出 `op_cuda`。

### 6) 模板代码风格与依赖显式性（已修复）

问题：

- Triton 模板中存在 import try/except，风格与工程规范不一致。

修复：

- 改为显式 import，依赖缺失时由运行环境直接报错，失败更早更清晰。

## 覆盖能力结论

### E2E 层

- 覆盖 small-model/diffusion/dl/transformer/sam/vit/tree 场景启动模板。
- 已具备从业务链路到 bottleneck lane 的入口与流程约束。

### LLM Serving 层

- 覆盖 sglang/vllm/trtllm 预设模板。
- 指标体系与 profile 路由聚焦 TTFT/TPOT/ITL/throughput。

### CUDA/Triton 层

- 提供 operator synthesis 快速起步路径。
- 新版 harness 可进行同输入 CPU baseline exactness 校验，满足 kernel 迭代前置条件。

## 下一步深度优化建议

1. `autopilot` 增加“失败熔断 + 自动回退策略”选项（基于 consecutive failure）。
2. `autopilot` 支持 lane-specific candidate command（e2e/serving/kernel 差异化执行）。
3. operator synthesis 增加 `attention/softmax/rmsnorm/conv2d` 模板。
4. 引入统计稳健比较（多次采样 + 中位数 + 最小收益阈值）。
5. 增加分布式 serving 资源拓扑采集（NVLink/PCIe/RDMA）并纳入决策报告。
