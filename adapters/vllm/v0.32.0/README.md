# vLLM v0.32.0 适配

基线：tag `v0.32.0`，commit `5df63a7203cdb9328d824a27b682e77c7b9928be`。

这里有两样东西：step 埋点补丁（L1），以及在 v0.32.0 源码上核对调研结论的记录。
调研报告是对着 main 分支写的，下面凡是与它不同的地方都以这份记录为准。

**状态：补丁只做过静态检查（能干净地打在 v0.32.0 上、语法通过）。配套的 pytest 还没有跑过，
也没有在 GPU 上跑过。**

## step 埋点补丁

`step-trace.patch` 只加挂钩，41 行，改两个文件：

| 文件 | 挂钩 |
| --- | --- |
| `vllm/v1/engine/core.py` | `EngineCore.step` 里取 t_start、t_end 和本步工作量；`DPEngineCoreProc.run_busy_loop` 里提交一行（真实前向或 dummy 前向），wave 结束时把缓冲交给写线程 |
| `vllm/v1/worker/gpu_worker.py` | `execute_dummy_batch` 之后等 GPU 做完，否则 dummy 步的 t_end 只是提交时刻 |

记录逻辑在 dpbench 里（`src/dpbench/vllm_step_trace.py`，只用标准库），所以 vLLM 的环境里要装 dpbench。

```bash
git clone --depth 1 --branch v0.32.0 https://github.com/vllm-project/vllm.git
cd vllm
git apply /path/to/dpbench/adapters/vllm/v0.32.0/step-trace.patch
pip install -e .                      # vLLM
pip install -e /path/to/dpbench       # 同一个环境
```

启动时设 `DPBENCH_STEP_TRACE_DIR`，每次启动用一个新目录：

```bash
DPBENCH_STEP_TRACE_DIR=traces/$(date +%s) VLLM_USE_DEEP_GEMM=0 \
vllm serve Qwen/Qwen3.6-35B-A3B \
  --data-parallel-size 4 --tensor-parallel-size 1 --enable-expert-parallel \
  --all2all-backend allgather_reducescatter \
  --language-model-only --no-enable-prefix-caching --no-async-scheduling \
  --max-model-len 32768 --max-num-seqs 256 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 --api-server-count 1
```

不设这个环境变量时，补丁不做任何事。开着 async scheduling 时 tracer 会拒绝启用并打一条警告。

每个 rank 写一个 `step.rank<k>.csv`。一次回放结束后，把这次运行的 step 切出来并入运行目录：

```bash
dpbench merge-steps runs/rr-0 --trace-dir traces/1731700000
```

这条命令按 request 表的首次 dispatch 和最后 finish 切时间窗，写 `step.parquet`，把 manifest 的
trace 级别改成 L1，并做两项检查，任何一项不过就以非零状态退出：

- **对齐检查**（S3 待验证项“step_id 对齐假设”）：每个 step_id 每个 rank 恰好一行；按各 rank
  自己的前向累计序号排序后 step_id 一致；同一步各 rank 的时间区间必须相交。第三条只看时间戳，
  编号自洽但对错了也能查出来。
- **工作量守恒**：每个 rank 的 step 表 token 总数等于 request 表推出的 `输入 − 缓存 + 输出 − 1`。
  “减 1”来自读代码（最后一个输出 token 不再送回模型），要用第一次真机运行确认。

### 列的取法

| 列 | 取法 |
| --- | --- |
| `step_id` | `wave × 2³² + wave 内前向序号`，序号从 1 起，dummy 前向也计数 |
| `t_start` | 真实步：`scheduler.schedule()` 之前。dummy 步：`execute_dummy_batch()` 之前 |
| `t_end` | 真实步：采样结果回到主机之后，`update_from_output` 之前。dummy 步：GPU 同步之后 |
| `prefill_tokens` / `decode_tokens` | 与 vLLM 自己的 `compute_iteration_details` 同一口径：还没产出过输出 token 的请求算 prefill |
| `num_reqs` | 本步被调度的请求数 |
| `kv_tokens` | 本步被调度请求的 `num_computed_tokens` 之和，是本步结束时的上下文长度 |
| `wave`、`dummy` | 引擎的 `current_wave`；本 rank 是否跑的 dummy 前向 |
| `step_in_wave`、`fwd_seq`、`t_exec_start` | 附加列：wave 内序号、进程生命期内的前向累计序号、`schedule()` 返回的时刻 |

## 在 v0.32.0 上核对的结论

### 与调研报告不同的

1. **默认内部均衡的打分公式不是 `waiting × 4 + running`。** v0.32.0 是
   （`vllm/v1/engine/core_client.py`，`DPLBAsyncMPClient.get_core_engine_for_request`）：

   ```
   score = max(client_count × inflight, waiting + running)
   if waiting: score += waiting × 6 × max(0, kv_cache_usage − 0.5)
   ```

   `inflight` 是本 API server 自己记的该引擎未完成请求数，是精确值；`waiting`、`running`、
   `kv_cache_usage` 来自协调器快照。取最小值，平分时取从轮转起点扫描到的第一个，起点每次请求加 1。
   单 API server、KV 占用不到一半时，它就是带轮转平分的 JSQ。
   `policies.py` 里的 `vllm_default` 仍是旧公式，见文末“待定”。

2. **step 计数器 vLLM 自己就有。** `DPEngineCoreProc.step_counter` 每次循环加 1、wave 结束清零，
   源码注释写明它“在各 DP rank 间相同”，并且已经随 `SchedulerStats` 发给协调器。补丁直接用它，
   没有另起计数器。

3. **默认走 Model Runner V2**（`vllm/v1/worker/gpu/model_runner.py`），不是 `gpu_model_runner.py`。
   每次前向开头的 DP 同步在 `vllm/v1/worker/gpu/dp_utils.py`，是一次 CPU 上的 all-reduce。

4. **开着 CUDA Graph 时，所有 rank 的 token 数会被补齐到各 rank 的最大值。**
   `sync_cudagraph_and_dp_padding` 取 `num_tokens_across_dp.max()` 再按 CUDA Graph 的档位取整，
   每个 rank 都按这个形状跑。也就是说在计分配置下，负载轻的 rank 不是“算完了在等”，
   而是“陪着算了一批补齐的 token”。这直接影响同步等待损失的解释和 L2 的空等占比，
   也会让各 rank 的 step 时长天然接近。只有某个 rank 要求 eager 时才不补齐。

5. **空闲检查的间隔是 16 步，不是 32**（`--dp-sync-interval`，默认 16；每个 wave 的第 1 步也检查）。
   所以每个 wave 末尾最多会有 15 个全员 dummy 的步，`merge-steps` 会报这个占比。

6. **v0.32.0 多了一个内置的 prefill 节流**：`--prefill-schedule-interval N`，各 rank 每 N 步才接纳
   新的 prefill，按 `step_counter` 对齐。默认 1，即关闭。它本身就是一个 admission 基线，
   计分运行要保持默认值并记进 manifest。

### 确认无误的

- **关闭 async scheduling：`--no-async-scheduling`。** 不传时默认解析为开启，此时并发批数为 2，
  `EngineCore` 走 `step_with_batch_queue`，step 边界与 GPU 执行错开。
- **每次循环恰好一次前向。** 关闭 async scheduling 后，`run_busy_loop` 的一次循环要么跑一次真实前向，
  要么跑一次 dummy 前向（本 rank 没有请求，或有请求但本步一个 token 也没调度上），要么全局空闲直接跳过。
  step_id 对齐假设在代码上成立，还要靠 `merge-steps` 的对齐检查在真机上确认。
- **请求侧参数**：`/v1/completions` 的请求体有 `min_tokens`、`ignore_eos`，`prompt` 接受 token id 列表；
  请求头 `X-data-parallel-rank` 在 `vllm/entrypoints/generate/base/serving.py` 读取。只核对了字段存在，没有发过请求。
- **`allgather_reducescatter` 是默认的 all2all 后端。**

### issue #37856

按代码看已经修了，没有在真机上验证。v0.32.0 的 `Qwen2MoeMLP`（Qwen3.5/3.6 的共享专家用的就是它）
里有 `disable_tp = disable_tp or is_sequence_parallel`：sequence-parallel MoE 生效时共享专家不再按 TP 切分，
每个 TP rank 对自己那一段 token 算完整的共享专家输出，不会再出现输出缩成 1/TP 的情况。
仓库里没有针对 Qwen 这条路径的回归测试，所以 TP 大于 1 的配置在用之前仍要先和 TP 1 逐 token 比对。

## 待定

- `vllm_default` 策略：按 v0.32.0 的公式，回放器一侧能模拟的部分与 `jsq` 完全相同（KV 占用率回放器看不到）。
  要么删掉这个模拟、只用 `server` 策略测真实的内部均衡，要么保留并改成新公式。还没改。
- 第 4 条（DP 补齐）对指标定义的影响需要在设计文档里定一个说法。
