# dpbench：DP 均衡 benchmark

## 项目目标

衡量并优化 MoE 推理中 DP attention + EP 部署（DEP）下，各 DP rank 之间负载不均衡造成的同步等待损失。
主分是“回收率” R：策略在轮询基线和 mirror 均衡参考之间补上了多大比例的吞吐差距。

    R = (X_policy - X_RR) / (X_mirror - X_RR)

详细设计见 `docs/` 里的《DP 均衡 Benchmark 设计文档 v0.1》和 S3 工程调研报告。
调研报告是对着 vLLM main 分支写的；在 v0.32.0 源码上核对后的结论见 `adapters/vllm/v0.32.0/README.md`，两者不一致时以后者为准。

**当前状态和下一步做什么写在 `docs/下一步.md`，每次改动仓库后都要更新它。**
开发用的 Windows 机器上不装环境、不跑测试；没跑过的代码要在那份文档里写明。

## 已定的决策（不要擅自改）

- 框架：规范框架无关，第一个适配 vLLM，版本钉 **v0.32.0**（需要 Python 3.11 以上）。第二个是 SGLang。
- 模型：第一模型 `Qwen/Qwen3.6-35B-A3B`。大模型候选 Qwen3.8-Flash-Next（FP8），尚未确认，在 Ada 卡上风险高。
- 硬件：4 × L40S 先做最小回路和模拟器校准；8 × L20 做 DP 8 和正式结果。单卡 48 GB，只有 PCIe，没有 NVLink。
- DP 规模：卡全用，rank 内 attention 做 TP，EP 规模恒定。最小回路只用 DP4 = TP1。
- 通信后端只能用默认的 `allgather_reducescatter`（DeepEP 需要 Hopper 和 NVLink）。
- mirror 的 N 份副本：长度相同、内容不同。
- request 表含 `prefill_start_ts`。
- SLO：TPOT 用固定值，TTFT 为固定项加每千 token 的增量；具体数等 S3 实测后填。

## 进度

| 步骤 | 状态 |
|---|---|
| S0 规范（schema、manifest） | 已交付；大模型选型未定 |
| S1 指标库 | 已交付，测试对着手算结果 |
| S2 负载生成与回放器 | 已交付，只对假服务测过 |
| S3 最小回路（4 × L40S） | **进行中**：step 埋点补丁和源码核对已做，未上 GPU |
| S4 模拟器 v0 | 未开始 |
| S5 扩到 8 × L20、DP 8 | 未开始 |
| S6 可靠性验收 V1–V8 | 未开始 |
| S7 基线全量，v1.0 | 未开始 |
| S8 SGLang 适配 | 未开始 |
| 阶段 3 策略优化（内环搜索、真机验证、外环自改进） | 只讨论过方向，未写进设计文档 |

## 下一步任务（S3）

已做（没跑过 pytest，也没上过 GPU）：

- step 埋点补丁 `adapters/vllm/v0.32.0/step-trace.patch`，记录器 `src/dpbench/vllm_step_trace.py`，
  合并命令 `dpbench merge-steps`（带对齐检查和工作量守恒检查）。
- 在 v0.32.0 源码上核对了均衡打分公式、async scheduling 参数名、issue #37856，结论见适配目录的 README。

还要做：

1. 定两件事（见适配目录 README 的“待定”）：`vllm_default` 策略删掉还是改成 v0.32.0 的新公式；
   CUDA Graph 下 DP 补齐对“同步等待损失”定义的影响。
2. 补 AgentX 多轮会话的加载器（场景 P3）。
3. 真机上先发一次请求确认回放器的请求体和请求头，再跑 D1、D2、P2 和 lag 运行，用 RR 和 JSQ 两个策略。
4. layer 埋点（L2）。

S3 的过关条件：工作量守恒检查通过；噪声地板（V1）和零点（V2）通过；lag 运行测得出吞吐下降。

## 还没验证的

- 回放器没连过真的 vLLM。请求体里的 `min_tokens`、`ignore_eos`、token id 形式的 `prompt`，
  以及请求头 `X-data-parallel-rank`，都要用一次真实请求确认。
- 场景的默认长度和到达率是起步值，要在真机上校准。
- step_id 对齐假设：关闭 async scheduling 后，各 rank 每个 wave 的前向次数是否严格相等。
  读 v0.32.0 的代码是成立的；真机上由 `dpbench merge-steps` 的对齐检查确认。
- step 埋点补丁本身，以及工作量守恒里“每个请求 输入 + 输出 − 1 个前向 token”的算法。
- DP 1 与 DP 4 对同一批请求的输出是否逐 token 一致（排除 dummy 前向对 Gated DeltaNet 状态的影响）。
- DP 4、PCIe、混合注意力结构三者叠加后，可回收空间是否超过噪声地板的 3 倍。

## 约束

- 评测时关闭前缀缓存、MTP 推测解码和 async scheduling（`--no-async-scheduling`）；加 `--language-model-only`。
  `--prefill-schedule-interval` 保持默认值 1。
- layer 表（L2）只能在单独的 `--enforce-eager` 运行里采：CUDA Graph 回放时 Python 钩子不执行。不要和计分运行混用。
- 时间戳一律用本机单调时钟，单位纳秒（`time.monotonic_ns()`）。
- 策略只能读在线信息，不能读输出长度。`RequestView` 故意没有这个字段，不要加。
- 模拟器只用于策略搜索的内环，报告分数一律来自真机。
- 指标只从三张表算，不读任何框架内部状态。
- 第 7、8 节里的阈值（1.1 倍 TPOT、1% 变异系数、5% 上界偏差等）都是暂定值，等 V1 实测后校准。

## 常用命令

```bash
pip install -e ".[dev]"
pytest                                   # 不需要 GPU
dpbench mock-server --port 8000 --ranks 4   # 假服务，联调用
dpbench gen | replay | merge-steps | metrics | recovery | validate
```

完整的联调步骤和真机步骤见 `README.md`。

## 代码结构

- `src/dpbench/schema.py`、`schemas/manifest.schema.json`：三张表与 manifest 的规范和校验。
- `src/dpbench/metrics.py`：全部指标。
- `src/dpbench/workload.py`：场景、trace 加载、副本、mirror / lag / 打散调度。
- `src/dpbench/policies.py`：dispatch 策略接口和基线。
- `src/dpbench/replay.py`：开环回放器兼外部路由。
- `src/dpbench/mock_server.py`：假服务，不是 S4 的模拟器。
- `src/dpbench/vllm_step_trace.py`：vLLM 引擎进程里的 step 记录器（只用标准库，不要在模块顶层引入 pandas 或 vLLM），以及合并、切窗、对齐检查。
- `adapters/vllm/v0.32.0/`：挂钩补丁和源码核对记录。vLLM 源码克隆在 `third_party/vllm`（不入库）；改了挂钩后用 `git -C third_party/vllm diff` 重新导出补丁。
- `tests/conftest.py`：一份可以手算的 trace，改指标时先看它。

## 与设计文档的三处差别

- 有空闲间隙时，SOL 和 W 只压缩 step 内的时间，间隙原样保留；没有间隙时与文档公式相同。
- L2 采样时，W 先在被采样的 step 上算，再按全部 step 的时长放大。
- 策略看到的 rank 状态由回放器记账，KV token 数按每个流式分片一个 token 估算。
