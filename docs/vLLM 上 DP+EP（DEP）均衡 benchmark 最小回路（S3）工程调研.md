# vLLM 上 DP+EP（DEP）均衡 benchmark 最小回路（S3）工程调研

**结论：S3 可以在 vLLM 上直接搭，不需要动 vLLM 的内核代码。** 把请求钉到指定 rank 有现成的 `X-data-parallel-rank` 请求头，各 rank 的步级同步已经由 `DPEngineCoreProc.run_busy_loop` 加 dummy batch 保证。真正要补的只有两类：一是 step/layer 埋点，二是可插拔的 dispatch/admission 钩子。硬件上的硬约束是：L40S/L20（Ada，SM89，纯 PCIe）只能用默认的 `allgather_reducescatter` all2all 后端；Qwen3.8-Flash-Next 在 Ada 上属于高风险候选，没有任何官方验证。

## TL;DR

- **把请求钉到 rank：** 内部 LB 模式下发 `X-data-parallel-rank: k` 请求头即可。前端默认负载均衡的依据是协调器约每 100 ms 发布一次的各 rank waiting/running 计数，打分 `waiting*4+running`。自定义 dispatch 的最小改动点是 `vllm/v1/engine/core_client.py` 里的 `DPLBAsyncMPClient.get_core_engine_for_request`。step 埋点挂在 `vllm/v1/engine/core.py` 的 `DPEngineCoreProc.run_busy_loop` / `EngineCore.step`；layer 埋点挂在 `FusedMoEPrepareAndFinalize.prepare/finalize`（或 AG-RS 路径的 dispatch/combine）。
- **模型与硬件：** Qwen3.6-35B-A3B（256 专家，8 路由 + 1 共享，官方建议 vllm≥0.19.0）可以用 `--enable-expert-parallel`。在 4×L40S 上应选 DP4/EP4、TP=1、`allgather_reducescatter` 后端、FP8 MoE 走 Triton。DeepEP 需要 Hopper 和 NVLink，不可用。TP>1 与 DP+EP 组合有已知的 Qwen3.5 共享专家错误（#37856），默认应避开 TP2×DP2。Qwen3.8-Flash-Next FP8 在 DEP 下必须开 N-gram 表 CPU offload，主机内存每 worker 至少 51 GB，而且官方 recipe 没有 Ada 验证。
- **数据与模拟器：** Azure 2024、BurstGPT、AgentX、Mooncake 四份 trace 的字段已经确认，其中只有 AgentX（Apache-2.0）和 BurstGPT v2.0 的 BurstGPT_3 带会话。现有模拟器里只有 BalanceRoute 论文的 `a·x+b` 屏障模型和 Frontier 显式处理了"最慢 rank 决定步长"，Vidur 不支持 MoE。建议自写 step 级模拟器，按 rank 分别拟合"prefill token + decode batch + 全注意力层 KV token"的分段线性模型，再取各 rank 最大值。

## 版本基线与证据等级说明

- **版本基线：**
  - 最新稳定版是 vLLM v0.33.0（2026-11-02 发布，vllm.ai 首页与 deps.dev 一致）；更早的版本是 v0.32.0（10-18）和 v0.31.0（10-05）。\[1\]\[2\]
  - 本次引用的 main 源码锚点是 commit `913fccd11847f5b4d4769e3a69c8925f955bc29e`（来自 issue #64801，2026-10-30）。recipes\[3\] 站点显示的是 `vllm-project/vllm @ ade4fee`。
  - API 文档页（docs.vllm.ai/en/latest 与 /stable）的源码片段对应的就是这两个时间点附近的代码。
- **证据标记：** 下文用【已确认】表示官方文档、源码页或 issue/PR 原文有记载；【推测】表示根据代码结构推断或凭经验给出，需要真机验证。
- **未合并 PR：** 下文提到的未合并 PR 只说明"存在提案"，不代表行为已改变。

---

## 一、vLLM 的 DP / DP+EP 实现与可埋点位置

### 1. 内部 DP 负载均衡

- **【已确认】LB 位置和依据：** 官方 Data Parallel Deployment 文档原文是 "the internal DP load balancing is done within the API server process(es) and is based on the running and waiting queues in each of the engines"，并说明将来"could be made more sophisticated … by incorporating KV cache aware logic"。\[4\]
- **【已确认】类结构：** 都在 `vllm/v1/engine/core_client.py`。
  - `DPAsyncMPClient(AsyncMPClient)` 是多引擎异步客户端，"Assumes external load-balancing by default"。\[5\]
  - `DPLBAsyncMPClient` 是带内部 LB 的子类。
  - main 中 `self.lb_engines` 的注释是 "List of [waiting, running, kv_cache_usage] per engine"，较老的 v0.10 只有 [waiting, running]。
  - 后台 stats 任务从协调器收消息并解码：`counts, wave, running = msgspec.msgpack.decode(buf)`。\[5\]
  - `add_request_async` 里调用 `chosen_engine = self.get_core_engine_for_request(request)`。
- **【已确认】打分公式：** `score = waiting * 4 + running`，选最小值；平分时取扫描到的第一个（从 `eng_start_index` 开始轮转扫描）。vllm-ascend issue #20340 指出，这个平分规则会在混合长度负载下造成"initial avalanche"。\[6\]
- **【已确认、未合并】相关提案：**
  - PR #43279：DP>4 时改用 power-of-two-choices。\[7\]
  - PR #62900：单 API server 时按 `max(client_count*inflight, waiting+running)` 打分，理由是协调器快照"published at step boundaries, which can be seconds apart during long prefills"。\[8\]
  - PR #61386：协调器广播 5 字段快照，新增 `mean_queue_time` 和 `preempted_total`。\[9\]
  - 合入前，基线行为应以你所钉版本的源码为准。
- **【已确认】协调器：** `vllm/v1/engine/coordinator.py`，类 `DPCoordinator` / `DPCoordinatorProc`。
  - 文档字符串原文："Collects stats from each DP engine (currently just waiting and running …)"。\[10\]
  - 构造参数 `min_stats_update_interval_ms: int = 100`，代码注释是 "Send at stats_update_interval_ms interval if the stats have changed"。客户端注释写 "from the coordinator (which happen every 100ms)"。\[10\]\[11\]
  - 协调器还负责 request wave：任一引擎收到新请求时把全体切到 running 状态，并处理 stale wave。\[10\]
  - 引擎侧在 `run_busy_loop` 中每次 step 前后都调用 `_maybe_publish_request_counts()`，代码注释为 "Publish request counts before and after GPU step to ensure freshness"。\[12\]
- **对 benchmark 的含义：** 默认 LB 看到的是 ≤100 ms 粒度、按请求个数计的快照。它看不到 prefill token 量、KV token 量和输出长度，所以这本身就是一个很好的"弱基线"。
- **【推测】自定义 dispatch 的最小改动：**
  - 子类化 `DPLBAsyncMPClient`，重写 `get_core_engine_for_request(request)`：从 `request.prompt_token_ids` 取输入长度，再结合自维护的各 rank 状态返回 `self.core_engines[k]`。
  - 各 rank 的 KV token 数、待算 prefill token 数，协调器默认不发布，有两种取法：
    - (a) 前端自行记账：dispatch 时累加 input_tokens，收到输出时累加生成 token，结束时扣除。单 API server 下这是精确的，PR #62900 用的就是同一思路。
    - (b) 扩展 `SchedulerStats` 和协调器消息，参照 #61386 的做法。
  - S3 推荐 (a)，并配合 `--api-server-count 1`。
  - 比改源码更省事的替代方案：完全不改 vLLM，在外部 router 里实现策略，再用 `X-data-parallel-rank` 钉 rank（见第 2 点）。研究阶段首选这一方案。

### 2. 把请求钉到指定 DP rank

- **【已确认】机制：** OpenAI 兼容接口读取 HTTP 头 `X-data-parallel-rank`，作为 `data_parallel_rank` 传给引擎。
  - 越界时返回 400，例如 "data_parallel_rank 1 is out of range [0, 1)"（METR/hawk PR #2603 记录）。\[13\]
  - 校验逻辑在 `vllm/v1/engine/input_processor.py`（#L404-L405 @913fccd），拒绝 ≥ `data_parallel_size_local` 的值。\[3\]
  - 选引擎逻辑在 `core_client.py`（#L754-L756、#L1853 @913fccd）。\[3\]
- **【已确认】已知坑：** Responses API 曾经丢掉这个头，PR #62450 修复后才与 Chat Completions 对齐。回放时请用\[14\] `/v1/completions` 或 `/v1/chat/completions`。
- **三种模式下的用法：**
  - **内部 LB（单端口，`--data-parallel-size 4`）：** 头里填 0..DP-1，即全局 rank。单机时全局与本地编号一致，这是 S3 推荐模式。
  - **hybrid LB（`--data-parallel-hybrid-lb`）：** 每节点一个 API server，只调度本节点引擎。Python\[4\] 前端把头解释为本地下标 `[0, data_parallel_size_local)`，不是全局 rank；Rust 前端（`VLLM_USE_RUST_FRONTEND=1`）解释为全局 rank（issue #64801，修复 PR #64877 仍是 open）。单机下这两种解释没有区别。\[3\]
  - **external LB（`--data-parallel-external-lb`，或每个 rank 单独 `--data-parallel-rank k --port 800k`）：** 每个 rank 一个独立 HTTP 端口，钉 rank 等于"发到对应端口"，不需要请求头。官方文档要求共址的多个 rank 用 `CUDA_VISIBLE_DEVICES` 或 `--device-ids` 分开设备；协调器与 rank 0 共址。\[4\]
- **限制：**
  - 钉 rank 后会完全绕过内部 LB。
  - `--max-num-queued-reqs` / `--max-num-queued-tokens` 是整个服务的总限额，而 `--max-num-seqs` 是每个 rank 各自的限额（官方文档原文）。mirror\[4\] 回放时要把 max-num-seqs 设得足够大，以免 rank 内排队把不均衡效应掩盖掉。

### 3. step 循环、跨 rank 同步与 admission

- **【已确认】循环结构：** `vllm/v1/engine/core.py` → `DPEngineCoreProc.run_busy_loop()`：\[15\]
  1. `_process_input_queue()`；
  2. `_maybe_publish_request_counts()`；
  3. `executed = self._process_engine_step()`，内部调用 `EngineCore.step()`，即 scheduler.schedule → model_executor.execute_model → update_from_output；
  4. 若本 rank 本步没有真实前向但全局仍处于 running 状态，则执行 `self.execute_dummy_batch()`；
  5. `self.engines_running = self._has_global_unfinished_reqs(local_unfinished_reqs)`；
  6. 全局空闲时打印 "Wave %d finished"，`current_wave += 1`，step 计数器清零。
- **【已确认】all-reduce 频率：** v0.13 的 `_has_global_unfinished_reqs` 只在 `step_counter % 32 == 0` 时做 all-reduce（`ParallelConfig.has_unfinished_dp`）。latest 的注释是 "Sync step 1 too: an idle pause needs one dummy batch, not a full interval"。官方文档说法是\[16\]\[17\] "a collective operation performed every N steps to determine when all ranks become idle"。\[4\]
- **【已确认】步间对齐靠什么：** 每次前向里的 MoE 层都要做 EP 集合通信，因此 "Forward passes must be aligned"；没有请求的 rank 必须跑 dummy forward（官方文档原文）。\[4\]
- **【推测】step_id：** vLLM 没有对外的全局 step 编号。由于每步都有 EP 集合通信，同一 wave 内各 rank 的第 k 次前向（真实或 dummy）一一对应。因此 `(current_wave, 本 wave 内前向序号)` 可以作为全局对齐的 step_id。
  - 实现：在 `_process_engine_step` / `execute_dummy_batch` 两条路径上都把一个 rank 本地计数器加 1，计数器在 wave 切换时与 `current_wave` 一起记录。
  - 验证方法：先在真机上用 `torch.distributed` 的 barrier 时间戳或 NCCL 日志确认，在 async scheduling 和 MTP 都关闭时，计数严格对齐。
- **【推测】step 表各字段来源：** 都在 `EngineCore.step` 内取。
  - `SchedulerOutput.num_scheduled_tokens`（每请求本步 token 数）；
  - `scheduled_new_reqs` / `scheduled_cached_reqs`；
  - `total_num_scheduled_tokens`。
  - prefill token 等于 `num_computed_tokens < num_prompt_tokens` 的请求的本步 token 之和，decode token 等于其余请求的 token 之和。
  - `num_reqs = len(num_scheduled_tokens)`。
  - `kv_tokens = Σ(request.num_computed_tokens)`，从 `scheduler.running` 遍历得到。
  - t_start/t_end 用 `time.perf_counter_ns()` 包住 schedule 到 execute_model 返回。在同步调度下 execute_model 返回时 GPU 已完成。
  - **强烈建议 S3 关闭 async scheduling**（`--no-async-scheduling`，参数名需按所钉版本确认），否则 step 边界与 GPU 执行会错开一拍。PR #64457 也说明 async scheduling 会让某些 rank 出现"不跑前向"的步。\[18\]
  - dummy 步的 step 表行写 prefill=decode=0，并标记 dummy。
- **【推测，参数名为长期稳定的官方参数】admission 现状：**
  - 调度器在 `vllm/v1/core/sched/scheduler.py` 的 `Scheduler.schedule()`。
  - 先给 running 请求分配 token，再从 waiting 队列接纳。
  - 每步 token 预算是 `--max-num-batched-tokens`。
  - chunked prefill 按剩余预算切块。
  - `--long-prefill-token-threshold` 限制单个长 prompt 每步占用的预算。
  - `--max-num-partial-prefills` / `--max-long-partial-prefills` 控制并发的部分 prefill。
- **admission 插入点：** 子类化 `Scheduler`，在 waiting 循环里加一个 `should_admit(req, rank_state, global_state)` 判断，再通过 `--scheduler-cls`（SchedulerConfig 字段）加载，不用改上游文件。
  - 这里有一个难点：admission 若要用"其它 rank 的负载"，必须跨 rank 通信。最小方案是让 rank 在 `_maybe_publish_request_counts` 中多发 prefill backlog，由前端统一决定"何时把请求放出"。也就是把 admission 实现为前端 gating（请求先留在前端，满足条件才 dispatch），引擎侧不改。

### 4. MoE dispatch/combine 入口与层级埋点

- **【已确认】模块化内核：**
  - 设计文档 "Fused MoE Modular Kernel" 规定 `FusedMoEPrepareAndFinalize(Modular)::prepare()` 负责 "Quantization and All2All Dispatch"，`finalize()` 负责 "TopK Weight Application and Reduction and All2All Combine"。\[19\]
  - 代码在 `vllm/model_executor/layers/fused_moe/modular_kernel.py`：`FusedMoEModularKernel`，main 中为 `FusedMoEKernel` 与 `FusedMoEKernelModularImpl`。
  - 后端句柄来自 `get_ep_group().device_communicator.all2all_manager`，见 `vllm/model_executor/layers/fused_moe/layer.py`。\[20\]
  - 层入口是 `FusedMoE`（同一 `layer.py`）。
- **【推测】allgather_reducescatter 路径：** 默认的 `allgather_reducescatter` / `naive` 后端走 `FusedMoE` 前向里对 EP group 的 dispatch/combine 调用，实现位于 `vllm/distributed/device_communicators/all2all.py`（AG-RS / Naive All2AllManager）。在 Ada 上 S3 实际走的就是这条路径，开工前要在所钉 commit 上 grep `dispatch(`/`combine(` 确认挂点。
- **三个时间戳怎么挂：**
  - `t_ready`：进入 dispatch（prepare 或 all2all dispatch）之前；
  - `t_dispatch_done`：dispatch 返回、专家 GEMM 开始之前；
  - `t_combine_done`：combine（finalize）返回之后。
  - 跨 rank 的"等待"主要体现在 dispatch 内部：最快的 rank 会在 allgather 上等最慢的 rank。因此 `t_dispatch_done − t_ready` 的跨 rank 差异就是同步损失的层级度量。
- **【推测】计时方法与 CUDA Graph：**
  - 在 eager 模式下，用 `torch.cuda.Event(enable_timing=True).record()` 在当前流上打点，步结束后统一 `elapsed_time`。开销是微秒级，而且不需要 `synchronize`。若要跨 rank 对齐绝对时刻，每步在一个已知同步点（例如 step 开始）记一个参考 event，同时记 `perf_counter` 作为锚点。
  - piecewise cudagraph 默认只在注意力处切分，MoE 层位于被捕获的片段内，Python 侧 hook 在 replay 时不会执行。full cudagraph 更是如此。
  - 因此 S3 的做法是：layer 表采样跑用 `--enforce-eager`（或把 MoE 层设为 splitting op）；step 表和端到端跑开 cudagraph。两组都跑，用于量化插桩对步长的影响。
  - 另一个选择是 Nsight Systems 加 NVTX：`--profiler` / `VLLM_TORCH_PROFILER_DIR`（官方有 "Profiling vLLM" 文档页）。它能看到\[21\] cudagraph 内的 kernel，但不适合长时间采集。
- **【已确认】已有做法：**
  - EPLB 的 `log_balancedness`（EPLBConfig 字段）记录专家负载均衡度，可作为参考。\[22\]
  - BalanceRoute 论文附录 D.5 有 "Engine-side instrumentation for evaluation"（基于 vllm-ascend）。\[23\]
  - 我没有找到 vLLM 上游"每层 dispatch/combine 时间戳"的现成 PR。

### 5. 现有 metrics/tracing 能直接提供的 request 表字段

- **【已确认】可以直接拿到的：**
  - Prometheus：`vllm:num_requests_running/waiting`、`vllm:kv_cache_usage_perc`、`vllm:iteration_tokens_total`、`vllm:request_queue_time_seconds`、`vllm:request_prefill_time_seconds`、`vllm:request_decode_time_seconds`、`vllm:time_to_first_token_seconds`、`vllm:prompt_tokens_cached_total`。指标带 `engine` 标签（`vllm/v1/engine/async_llm.py` #L209），可以按 rank 拆分。\[3\]
  - 引擎核心记录 per-request 事件 QUEUED / SCHEDULED / PREEMPTED（Metrics 设计文档）；前端 `RequestStateStats` 有 `queued_ts`、`scheduled_ts`（首次调度，忽略抢占）等字段（`vllm/v1/metrics/stats.py`）。\[24\]\[25\]
  - 新的 Per-Request Metrics 特性会在响应体中返回 `queue_time_ms` 和 `time_to_first_token_ms`。注意文档对后者的定义是"from when the request was scheduled"，不是从到达开始。n>1 时这些字段为 null。\[26\]
- **字段对应：**
  - `arrival_ts`：客户端记录发送时刻最可靠；服务端 `arrival_time` 可作对照。
  - `dispatch_ts`：由前端或 router 记录。
  - `prefill_start_ts`：取 `scheduled_ts`，即 SCHEDULED 事件。
  - `rank`：钉 rank 时已知；未钉时可从 `engine` 标签或日志反查。
  - `input_tokens`：`len(prompt_token_ids)` 或 usage.prompt_tokens。
  - `cached_tokens`：RequestOutput 的 `num_cached_tokens`，或 usage 中的 prompt_tokens_details；关闭前缀缓存后应恒为 0，可用来校验。
  - `output_tokens`：usage.completion_tokens。
  - `first_token_ts`：客户端收到第一个非空流式 chunk 的时刻。
  - `finish_ts`：客户端收到最后一个 chunk 的时刻。
- **建议：** 所有时间戳以客户端为准，服务端事件作为辅证。服务端 per-request 时间戳在前端进程，与引擎进程的 step 时间同属一台机器的单调时钟，可以对齐。

---

## 二、模型与硬件兼容性

### 6. EP 支持与已知问题

- **【已确认】Qwen3.6-35B-A3B：**
  - 架构 `qwen3_5_moe`（`Qwen3_5MoeForConditionalGeneration`，多模态），256 专家，"8 Routed + 1 Shared"，HF 模型卡建议 "vllm>=0.19.0"。recipes\[27\]\[28\] 页标注 "vLLM 0.17.0+"。\[29\]
  - 官方 Qwen3.5/3.6 Usage Guide 说："use `--language-model-only` to skip loading the vision encoder and free up memory for KV cache as well as enabling Expert Parallelism"。\[30\]
  - 第三方 AWQ 卡指出 TP=8 时必须加 `--enable-expert-parallel`，否则专家无法均匀切分。由此可知\[31\] MoE 层走 FusedMoE，EP 自动生效。
- **【已确认】Qwen3-Next：** `vllm/model_executor/models/qwen3_next.py` 中 `Qwen3NextForCausalLM` 实现了 `QwenNextMixtureOfExperts` 和 `set_eplb_state`，读取 `eplb_config.num_redundant_experts`，因此支持 EPLB。\[32\]\[33\]
- **【推测】qwen3_5_moe 的 EPLB：** 它复用 `Qwen3NextSparseMoeBlock`，很可能同样支持 EPLB，但我没找到明确的一手说明。
- **【已确认】已知 issue：**
  - **#37856：** EP + TP>1 + DP>1 会触发 `use_sequence_parallel_moe`，`Qwen3NextSparseMoeBlock` 的共享专家输出被缩小为 1/TP，产生 "completely garbled output"（2026-03，H100）。我未能确认该问题在\[34\] v0.30–v0.33 是否已修复。
  - **#44185：** `Qwen3.5-35B-A3B --enable-expert-parallel --data-parallel-size 2` 加 MTP 时，长度接近 max_model_len 会卡死，原因是 DP 组间控制流分叉。修复 PR #44954 是否合入未确认。由于\[35\] benchmark 本来就关闭 MTP，影响有限。
- **【推测】GDN 与 dummy batch：** 没有找到 vLLM 侧的相关 issue。首次跑通后应检查 dummy 步是否影响 GDN 状态（对比 DP1 与 DP4 在相同请求上的输出 token 是否一致）。
- **【已确认】Qwen3.8-Flash-Next：** GDN + Qwen Sparse Attention，再加 N-gram/PLE。recipe 写明 "Stable ≥0.30.0 runs the model"，但只在 nightly 上调优验证过。recipe\[36\] 列出的硬件是 H200/H100/GB200/GB300/MI355X/RTX Pro 6000，**没有 Ada**；NVIDIA NVFP4 卡也只写 Hopper/Blackwell。QSA\[37\]\[38\] 的 indexer 内核能否在 SM89 上运行未确认，这是最大风险。

### 7. all2all 后端在 L40S/L20（SM89、仅 PCIe）上的可用性

- **【已确认】可用：** `allgather_reducescatter` 是默认后端，官方说明为 "Standard all2all using allgather/reducescatter primitives … works with any EP+DP configuration"。它基于\[39\] NCCL，不依赖 NVLink。**这是 Ada/PCIe 上唯一现实的选择。**
- **【已确认】naive：** "simple broadcast-based"，官方定位是 "Debugging, not recommended for production"。\[40\]
- **【已确认】不可用：**
  - DeepEP（`deepep_high_throughput`、`deepep_low_latency`、`deepep_v2`）：DeepEP 主 README 要求 "NVIDIA Hopper or newer GPUs" 和 "NVLink for intranode communication"；旧版 README 把 "PCIe kernel support for non-NVLink environments" 列为路线图项。DeepEP\[41\] issue #481 就是用户在 4×A6000 PCIe 上尝试失败。\[41\]\[42\]\[43\]
  - `flashinfer_nvlink_*` 面向 MNNVL，不适用。\[39\]\[44\]
  - pplx 在新 CLI 列表中仍然存在，但同样依赖\[45\] NVSHMEM，Ada/PCIe 下我没有找到可用证据【推测：不可用】。
- **【已确认/推测】DeepGEMM：** 不是硬依赖，只在 DeepEP 和高性能 FP8 路径中用到。PR #60171 在 SM89 上跳过了 "DeepGEMM tests below SM90"，因此\[46\] DeepGEMM 不支持 Ada。启动时建议 `VLLM_USE_DEEP_GEMM=0`。
- **【已确认】TP×DP×EP 组合：** 官方 EP 文档写明 "EP_SIZE = TP_SIZE × DP_SIZE"，所以 TP2×DP2 得到 EP4，这是设计上支持的。但正好会触发\[39\] #37856。**建议：** S3 默认用 DP4/TP1/EP4；做 DP2/TP2 对照时必须先跑正确性校验（与 TP1 输出逐 token 比对）。

### 8. 显存估算

- **【推测，基于公开配置估算，需真机验证】Qwen3.6-35B-A3B：**
  - HF 上 Qwen/Qwen3.6-35B-A3B 的 config.json 给出 `"hidden_size": 2048`、`"moe_intermediate_size": 512`、`"num_experts": 256`、`"num_experts_per_tok": 8`、`"num_hidden_layers": 40`、`"full_attention_interval": 4`、`"vocab_size": 248320`，视觉编码器为 `"depth": 27, "hidden_size": 1152`。据此路由专家约 3×2048×512×256×40 ≈ 32.2B，非专家部分（嵌入 248320 词表、GDN/注意力、共享专家、视觉编码器）约 3B。
  - BF16、DP4+EP4：每卡专家约 8.05B×2 B ≈ 16 GB，加非专家约 6 GB，权重共约 22 GB/卡。
  - FP8：专家约 8 GB，加非专家（部分保持 BF16）约 4–6 GB，权重约 12–14 GB/卡。
  - 在 48 GB 卡、`--gpu-memory-utilization 0.9` 下，KV/GDN 状态和激活还有约 20 GB（BF16）或约 28 GB（FP8）。`--language-model-only` 还能省掉视觉编码器。
  - 验证方法：看启动日志里的 "Model loading took X GiB" 和 KV cache 块数。
- **【已确认】Qwen3.8-Flash-Next FP8：**
  - checkpoint 172.78 GiB，其中 51B 的 N-gram 表为 per-tensor FP8，另有 4B MTP。\[36\]\[37\]
  - recipe 原文："PLE CPU offload is required for DEP … vLLM enables it by default; set it explicitly with `--engram-config '{\"cpu_offload\":true}'`"，以及 "DEP without it is not supported"。\[36\]
  - 表放在主机内存，按需异步预取行。\[47\]
  - Dynamo recipe 写 "Host memory: ≥ 51 GB per worker for N-gram embedding offload (VLLM_PLE_CPU_OFFLOAD=1)"，暗示每个 DP 进程各自持有一份【推测】。DEP8\[48\] 就可能需要约 8×51 ≈ 408 GB 以上主机内存，务必实测 RSS 或共享情况。
- **【推测】Flash-Next 能否放进 8×48 GB：** 去掉 N-gram 表后约 129B 参数（FP8 约 130 GB），按 EP8 切专家，再加上复制的非专家权重，每卡大约 20–30 GB，理论上放得下。但 recipe 在 8×H100 上都依赖 offload，而且 "plain TP8 is incompatible with … 128-wide quantization blocks"。\[36\]
- **【已确认】Ada 的 FP8：** SM89 支持 FP8 W8A8。PR\[49\] #60171 确认 Triton block-FP8 路径能在 SM89 上运行。#58241\[46\] 说明 CUTLASS block-FP8 线性层把 SM89 排除在外（"89 → False, 90 → True"，原因是 TMA 只在 Hopper 及以上），于是会回落到 Marlin W8A16。MoE\[49\] 层由 `fused_moe/oracle/fp8.py` 的 `select_fp8_moe_backend` 选择，在 SM89 上应落到 TRITON【推测，需看启动日志 "Using TRITON Fp8 MoE backend"】。\[50\]\[51\]
- **【已确认】FP8 实例：** vLLM 论坛帖（2026-07）有用户在 2×L40S PCIe 上跑 Qwen3.6-35B-A3B-FP8，用了 TP2，也尝试过 DP2+EP，没有报错。这只是用户报告，不是官方验证。\[52\]

### 9. 混合结构的 decode 步长特性

- **【已确认】** vLLM 官方推文："only a quarter of the layers hold a growing KV cache"，即 GDN 层的状态是常数大小。BalanceRoute\[53\] 论文 §2.1 把每 rank 每步耗时建模为 `a·x + b`，原文是 "x is the total KV-cache footprint of the worker's active batch, a is the bandwidth, and b is a load-independent overhead"，并指出 x 项相对 b 越大，DP 均衡越重要。
- **【推测】** 对 Qwen3.6，x 只来自 10 个全注意力层：HF 模型卡写明 "Hidden Layout: 10 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))"，Gated Attention 为 "16 for Q and 2 for KV"、head dim 256。GDN 层的代价与 batch 大小线性相关、与上下文无关。因此 decode 步长对上下文的斜率大约只有同规模全注意力模型的 1/4，不均衡损失会更多由 batch 大小差异和 prefill 混入决定，而不是上下文长度差异。
- **公开数据：** 我没有找到针对 Qwen3.5/3.6 GDN 的公开分步实测曲线。
- **需要做的标定实验：** 固定 B∈{1,8,32,64,128}、ctx∈{1k,4k,16k,64k}，单 rank（DP1/EP1，以及 DEP4 下其余 rank 跑 dummy），各测 200 步 decode 步长，拟合系数。

---

## 三、回放所需的服务端参数

### 10. 固定输出长度、token id 输入、TTFT、关缓存、关 MTP、thinking

以下参数都是 vLLM SamplingParams 和 OpenAI 兼容服务器长期提供的官方参数，本轮没有逐页重抓文档，请在所钉版本上用一次请求验证。

- **【推测】固定生成 N 个 token：** 在请求体里同时设 `max_tokens=N`、`min_tokens=N`、`ignore_eos=true`（后两者是 vLLM 扩展参数，直接放进 JSON body）。不要设 `stop` / `stop_token_ids`。用响应的 `usage.completion_tokens == N` 校验。
- **【推测】token id 输入：** `/v1/completions` 的 `prompt` 可以直接传 token id 列表（`[int]`），这是回放时绕开 chat template、精确控制 input_tokens 的首选方式。官方文档还有 "Tokens In <> Tokens Out API" 页可作参考。\[21\]
- **TTFT：** 用 `stream=true`，加 `stream_options={"include_usage": true}`，客户端记录第一个带非空 text/token 的 chunk 的到达时间。注意首个 chunk 可能是空的 role chunk（chat 接口），要跳过。服务端可另外开启 per-request metrics 作为对照。
- **前缀缓存：** 用 `--no-enable-prefix-caching` 关闭。\[54\]
- **MTP / 推测解码：** 不传 `--speculative-config` 就是关闭状态。HF 模型卡和 recipe 中的 MTP 示例都是显式加 `--speculative-config '{"method":"qwen3_next_mtp",...}'` 才开启的。注意\[28\]\[55\] Qwen3.8-Flash-Next 的 recipe 默认命令带 MTP，复制时要删掉。
- **【推测】thinking 模式：** 用 `/v1/completions` 加 token id 输入时，chat template 不参与，thinking 对长度没有影响，因为 ignore_eos 下生成就是 N 个 token，内容不重要。若用 chat 接口，可在请求中加 `chat_template_kwargs={"enable_thinking": false}`，同时不要启用 `--reasoning-parser`，避免把 token 拆成 reasoning_content 后统计口径不一致。

---

## 四、负载 trace 数据集格式

### 11. Azure LLM inference trace 2024

- **【已确认】** 来源是 `Azure/AzurePublicDataset` 的 `AzureLLMInferenceDataset2024.md`，采集于 2024-05-10 至 05-19，覆盖两类服务（Code、Conversation），对应 HPCA 2025 的 DynamoLLM 论文。\[56\]
- **【已确认】字段：** `TIMESTAMP`（Invocation time）、`ContextTokens`、`GeneratedTokens`。没有会话\[56\] ID，也没有前缀信息。下载链接在 `AzureLLMInferenceDataset2024Links.txt`。\[57\]
- **【推测，需下载核对】** 文件名形如 `AzureLLMInferenceTrace_code_1week.csv` / `AzureLLMInferenceTrace_conv_1week.csv`；TIMESTAMP 是带微秒和时区的日期时间字符串。
- **记录数：** 二手来源 Majumder, Yu & Xie（arXiv 2604.10769，2026-04）称其为 "44 million Azure LLM code and conversation requests [Stojkovic et al. (2025)]"；这不是一手数字，下载后仍应 `wc -l` 核对。
- **许可证：** 仓库对 LLM trace 使用 CC-BY 署名许可（第三方仓库转述 2023 版为 "CC-BY Attribution License"），以仓库 LICENSE 为准。\[58\]

### 12. BurstGPT（HPMLL/BurstGPT）

- **【已确认】文件：** `BurstGPT_1.csv`（前 2 个月，约 1429.7k 行）、`BurstGPT_2.csv`（后 2 个月，约 3858.4k 行）、`BurstGPT_3.csv`（另外 110 天），以及对应的 `BurstGPT_without_fails_*.csv`。\[59\]\[60\]
- **【已确认】列：** `Timestamp`（"seconds from 0:00:00 on the first day"，已校准到当地时区）、`Model`（ChatGPT/GPT-4）、`Request tokens`、`Response tokens`、`Total tokens`、`Log Type`（Conversation log / API log）。\[59\]\[60\]
- **【已确认】会话：** v2.0（2026-01-15 发布）在 BurstGPT_3 中新增 `Session ID` 和 `Elapsed time` 列，所以只有 BurstGPT_3 可以区分会话。\[59\]
- **【已确认】规模与许可：** 失败请求的 Response tokens 为 0。许可证 CC-BY-4.0。论文版本给出 10.31M 条 trace、213 天。\[60\]\[61\]

### 13. SemiAnalysis InferenceX AgentX v1.0 与 Mooncake Tool&Agent

- **【已确认】AgentX v1.0：**
  - 获取方式：HF `semianalysisai/cc-traces-weka-062126`（393 个会话，约 167k 请求），另有 `_256k` 变体（去掉了超过 256K 的请求）。许可证 apache-2.0。\[62\]\[63\]\[64\]
  - 格式：WEKA trace（JSON，按会话组织），由 AIPerf 直接消费：`agentx-harness==1.0.7`，或 `aiperf --public-dataset semianalysis_cc_traces_weka_062126`。\[64\]\[65\]
  - 字段：会话 ID；每个请求的时间戳（AgentX 发布版的子 agent 请求 `t` 是 root 绝对时间）；输入/输出 token 数；每 64-token 块一个 `hash_ids`（会话内链式哈希，可还原前缀复用）；子 agent 组（SPAWN/JOIN）；采集时的 TTFT/ITL。\[63\]\[66\]\[67\]
  - 不含 prompt 原文。\[67\]
- **【已确认】Mooncake FAST'25：**
  - 文件 `FAST25-release/traces/toolagent_trace.jsonl`：23,608 条请求，平均输入 8,596，平均输出 182，cache ratio 59%，按时间戳到达。\[68\]\[69\]
  - 每行 `{timestamp, input_length, output_length, hash_ids}`，timestamp 是相对 ms（0–3,600,000）；hash_ids 每块 512 token（NVIDIA Dynamo 文档："Public Mooncake/toolagent traces use 512"）。\[70\]\[71\]
  - 没有会话或轮次字段，只能通过 hash_ids 前缀推断会话关系。
  - 【推测】许可证随 Mooncake 仓库（Apache-2.0），需核对。

---

## 五、模拟器

### 14. 现有模拟器与推荐的耗时模型

- **【已确认】Vidur：**
  - 方法：先做算子级 profiling，再按算子类训练随机森林回归（特征包括 prompt 长度、KV 量、batch、分片数）。\[72\]
  - 论文报告的误差：静态 trace 中位误差 < 2.5%、P95 ≤ 3.33%；端到端 < 9%。\[72\]
  - 不足：没有 MoE 算子。Frontier 论文指出 Vidur "has no GroupedGEMM class"，在 MoE 上误差可达 45.5%，其 token-count predictor 在 FlashAttention 上平均误差 32.6%。\[73\]
  - 它支持 DP 副本，但副本之间独立，不建模步级屏障。
- **【已确认】Frontier（arXiv 2605.21312）：**
  - 显式建模 EP dispatch："Run pre-dispatch MoE work, wait for EP dispatch collective, then execute expert sub-batches"。\[74\]
  - MoE 算子用随机森林，特征包括 token-to-expert 计数的方差与最大值、专家选择比例等。\[73\]
  - 与 vLLM 对比误差在 9.37% 以内。\[73\]
  - 开源状态本轮未确认。
- **【已确认】BalanceRoute（arXiv 2605.06113，当前为 v3，2026-09-27，CC BY 4.0；作者 Bu、Lyu、Chen、Song、Liang、Gurung、Fan、Ye、Zhou，通讯作者为华为 Theory Lab 的 Yuwei Fan 与 HKUST 的 Yinyu Ye、Zijie Zhou）：**
  - 模型：步长由最慢 worker 决定，每 worker 步长约为 `a·x + b`（x 为总 KV 足迹）。\[23\]
  - 实现形态：部署为 PD 分离下的 decode 路由代理（附录 D：`DecodeInstanceState[g]`、`PromptPool`、`PrefillerBudget`、payload-mutation 协议；D.6 是 BR-0 的 pool-bypass 部署路径），运行在 vllm-ascend 上。\[23\]
  - 结果：相对最强基线 G=4 提升 +13.4%，G=16 提升 +34.5%。论文 §1 原文："Chen et al. (2026) reports that more than 40% of accelerator time is lost to barrier idle in production LLM deployments"。
  - **未能确认：** 匿名仓库 BR-Family-on-vllm-ascend 是否有正式公开地址，以及 BR-0 的实现文件名。本轮检索没有找到公开仓库。建议直接联系作者（HKUST / 华为 Theory Lab），并按论文 Algorithm 2 自行实现。
- **BF-IO 与其他模拟器：** BF-IO 对应 arXiv 2601.17855 "A Universal Load Balancing Principle and Its Application to Large Language Model Serving"，它提出 "Balance Future with Integer Optimization principle (BF-IO)"，面向 "sticky assignment systems with barrier synchronization"，与 BalanceRoute 所引 Chen et al. (2026) 吻合（仅见检索摘要，未获取全文，其附带模拟器的耗时模型未确认）。LLMServingSim 和 NanoCP 是否建模了 DP 屏障，本轮预算内没有拿到一手资料，需要后续阅读源码。
- **推荐自写 step 级模拟器（以下为方法建议）：**
  - 每 rank r 的步长：`T_r = c0 + c1·D_r + c2·K_r^{attn} + c3·P_r + c4·P_r·Kp_r`。其中 D_r 为 decode 请求数，K_r^{attn} 为全注意力层 KV token 总数（只算 1/4 的层），P_r 为本步 prefill token，Kp_r 为这些 prefill 的已有上下文。
  - 全局步长 `T = max_r T_r + c_comm(Σ_r tokens_r)`，通信项与总 token 数相关，因为 AG-RS 要搬运所有 rank 的 token。
  - 同步损失：`Σ_r (T − T_r)`。
  - 函数形式先用分段线性或非负最小二乘；只有残差呈现明显非线性（例如小 batch 的 cudagraph 档位跳变）时才改用随机森林。
  - 校准：DP4/EP4 下用 mirror 回放，构造各 rank 负载可控的网格（每格 ≥200 步）。取 step 表中同一 step_id 的各 rank 数据，"无等待"部分用 layer 表的 `t_ready` 差值近似。
  - 精度目标：步长 MAPE < 5%，端到端吞吐误差 < 10%（参照 Vidur/Frontier 的报告量级）。
  - cudagraph 档位（batch padding 到捕获尺寸）要作为离散特征处理。

---

## S3 最小回路具体改动清单

**环境：**
- vLLM v0.33.0 或钉一个 main commit（记录 SHA）；CUDA 12.9 已被 v0.33 移除，使用匹配的 CUDA 13 / PyTorch 2.14 轮子。\[1\]
- NCCL 环境变量 `NCCL_P2P_LEVEL` 视 PCIe 拓扑而定，先用 `nvidia-smi topo -m` 检查是否支持 P2P。
- 不安装 DeepEP/DeepGEMM，设 `VLLM_USE_DEEP_GEMM=0`。
- 模型：`Qwen/Qwen3.6-35B-A3B-FP8` 或 BF16。

**启动命令示例（4×L40S，DP4/EP4）：**
```bash
VLLM_USE_DEEP_GEMM=0 vllm serve Qwen/Qwen3.6-35B-A3B-FP8 \
  --data-parallel-size 4 --tensor-parallel-size 1 --enable-expert-parallel \
  --all2all-backend allgather_reducescatter \
  --language-model-only --no-enable-prefix-caching \
  --max-model-len 32768 --max-num-seqs 256 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 --api-server-count 1
# 层级采样跑追加 --enforce-eager；关闭 async scheduling（按所钉版本的参数名）
```

**改动和挂钩点：**
1. **外部 router 与回放器（不改 vLLM）：** 读 trace，按策略算出 rank 后加 `X-data-parallel-rank` 请求头。请求体为 `/v1/completions` + token id prompt + `max_tokens=min_tokens=N, ignore_eos=true, stream=true`。客户端记录 arrival/dispatch/first_token/finish 时间戳，写 request 表。
2. **`vllm/v1/engine/core.py`：**
   - 在 `DPEngineCoreProc.run_busy_loop` 中维护 `(current_wave, fwd_idx)`；`_process_engine_step` 和 `execute_dummy_batch` 两条路径都把 fwd_idx 加 1。
   - 在 `EngineCore.step` 中从 `SchedulerOutput` 和 `scheduler.running` 汇总 prefill/decode token、num_reqs、kv_tokens 以及 perf_counter 起止时间，异步写入每个 rank 一个 step 表文件（用 Parquet/CSV 缓冲队列）。
3. **`vllm/model_executor/layers/fused_moe/`（`modular_kernel.py` 的 prepare/finalize，或 `layer.py` 中 AG-RS 的 dispatch/combine 调用点）：** 采样（例如每 N 步一次）记录三个 CUDA event，在 step 结束时统一解析成 layer 表。step_id 通过 forward context 或全局变量传入。
4. **（可选）`vllm/v1/engine/core_client.py`：** 子类化 `DPLBAsyncMPClient.get_core_engine_for_request`，实现服务端内置 dispatch 策略，与外部 router 结果交叉验证。
5. **（可选）`vllm/v1/core/sched/scheduler.py`：** 子类化 `Scheduler`，通过 `--scheduler-cls` 注入 admission；跨 rank 的 admission 优先放在外部 router 中做前端 gating。
6. **校验清单：**
   - `usage.completion_tokens == N`；
   - `cached_tokens == 0`；
   - 各 rank 每个 wave 的 fwd_idx 计数一致；
   - DP1 与 DP4 同请求输出逐 token 一致（检查 GDN 和 dummy 步的影响）；
   - eager 与 cudagraph 两组步长差异 < 5%。

## Caveats

- 2026 年下半年 vLLM 的 DP-LB 正在活跃修改（#43279、#62900、#61386、#64877），务必钉住 commit，并在报告里注明基线 LB 的确切打分公式。
- TP>1 的 DEP 正确性（#37856）、Qwen3.8-Flash-Next 在 Ada 上的可运行性（QSA 内核、每 worker 51 GB 主机内存）、以及 step_id 对齐假设，都只是推测或尚未确认，必须先做真机实验。
- 本调研的 web 检索预算已用完。BalanceRoute 仓库地址、LLMServingSim/NanoCP 的建模细节、Azure 2024 的一手记录数（目前只有二手的约 4400 万条）和确切文件名仍待补充。

## Sources

1. [vLLM — Fast, Memory-Efficient LLM Inference & Serving](https://vllm.ai/)
2. [Versions](https://deps.dev/pypi/vllm/0.2.6/versions)
3. [\[Feature\]: Accept the global DP rank in X-data-parallel-rank under hybrid/external LB · Issue #64801 · vllm-project/vllm](https://github.com/vllm-project/vllm/issues/64801)
4. [Data Parallel Deployment - vLLM](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/)
5. [vllm.v1.engine.core\_client](https://docs.vllm.ai/en/latest/api/vllm/v1/engine/core_client/)
6. [\[Bug\]: DP load balancer initial avalanche selects the first tied DP and skews traffic under mixed-length workloads · Issue #20340 · vllm-project/vllm-ascend · GitHub](https://github.com/vllm-project/vllm-ascend/issues/20340)
7. [\[Perf\]\[Core\] Use power-of-two-choices for DP engine selection at large DP by abinggo · Pull Request #43279 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/pull/43279)
8. [\[Core\] Route by exact in-flight count when a single client owns the DP engines by jbelofamd · Pull Request #62900 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/pull/62900)
9. [\[Core\]\[DP\] Publish DP LB result-metric fields in engine snapshots by Jolin1993 · Pull Request #61386 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/pull/61386)
10. [vllm.v1.engine.coordinator - vLLM](https://docs.vllm.ai/en/v0.10.0/api/vllm/v1/engine/coordinator.html)
11. [vllm.v1.engine.core\_client - vLLM](https://docs.vllm.ai/en/v0.10.0/api/vllm/v1/engine/core_client.html)
12. [vllm.v1.engine.core](https://docs.vllm.ai/en/stable/api/vllm/v1/engine/core/)
13. [fix(infra): run vLLM with engine.dataParallelSize and route to the same ranks by VivaLaPanda · Pull Request #2603 · METR/hawk · GitHub](https://github.com/METR/hawk/pull/2603)
14. [\[Bugfix\]\[Frontend\] Preserve DP rank in Responses requests by jianzs · Pull Request #62450 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/pull/62450)
15. [vllm.v1.engine.core](https://docs.vllm.ai/en/v0.17.0/api/vllm/v1/engine/core/)
16. [vllm.v1.engine.core](https://docs.vllm.ai/en/latest/api/vllm/v1/engine/core/)
17. [vllm.v1.engine.core](https://docs.vllm.ai/en/v0.13.0/api/vllm/v1/engine/core/)
18. [\[Bugfix\]\[Core\] Keep offline DP ranks' collectives aligned with async scheduling by Juntian777 · Pull Request #64457 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/pull/64457)
19. [Fused MoE Modular Kernel - vLLM](https://docs.vllm.ai/en/stable/design/fused_moe_modular_kernel/)
20. [vllm.model\_executor.layers.fused\_moe.layer - vLLM](https://docs.vllm.ai/en/v0.10.1/api/vllm/model_executor/layers/fused_moe/layer.html)
21. [fp8 - vLLM](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/fused_moe/oracle/fp8/)
22. [Engine Arguments - vLLM](https://docs.vllm.ai/en/stable/configuration/engine_args/)
23. [Tackling the Data-Parallel Load Balancing Bottleneck in LLM Serving: Practical Online Routing at Scale](https://arxiv.org/html/2605.06113)
24. [Metrics - vLLM](https://docs.vllm.ai/en/stable/design/metrics/)
25. [vllm.v1.metrics.stats - vLLM](https://docs.vllm.ai/en/v0.9.2/api/vllm/v1/metrics/stats.html)
26. [Per-Request Metrics - vLLM](https://docs.vllm.ai/en/stable/features/per_request_metrics/)
27. [AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4 · Hugging Face](https://huggingface.co/AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4)
28. [Qwen/Qwen3.6-35B-A3B · Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
29. [Qwen3.6-35B-A3B on vLLM — Serve command for DGX Spark (GB10), H100, H200, MI300X, MI325X, MI355X, RTX Pro 6000, DGX Station (GB300), RTX 5090, Arc Pro B60, Arc Pro B70](https://recipes.vllm.ai/Qwen/Qwen3.6-35B-A3B)
30. [Qwen3.5 & Qwen3.6 Usage Guide - vLLM Recipes](https://docs.vllm.ai/projects/recipes/en/stable/Qwen/Qwen3.5.html)
31. [QuantTrio/Qwen3.6-35B-A3B-AWQ · Hugging Face](https://huggingface.co/QuantTrio/Qwen3.6-35B-A3B-AWQ)
32. [qwen3\_next - vLLM](https://docs.vllm.ai/en/v0.11.0/api/vllm/model_executor/models/qwen3_next.html)
33. [qwen3\_next - vLLM](https://docs.vllm.ai/en/v0.12.0/api/vllm/model_executor/models/qwen3_next/)
34. [\[Bug\]: Shared Expert output is incorrect under Sequence Parallel MoE (EP + TP \> 1 + DP \> 1) for Qwen3.5 MoE models](https://github.com/vllm-project/vllm/issues/37856)
35. [\[Bug\]: vLLM hangs during speculative decoding with MoE draft model near max\_model\_len under Data Parallelism (DP) · Issue #44185 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/issues/44185)
36. [Qwen3.8-Flash-Next on vLLM — Serve command for H200, H100, GB200 NVL4, GB300 NVL4, MI355X, RTX Pro 6000 4x, DGX Station (GB300)](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next)
37. [nvidia/Qwen3.8-Flash-Next-NVFP4 · Hugging Face](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)
38. [nvidia/Qwen3.6-35B-A3B-NVFP4 · Hugging Face](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4)
39. [vllm/docs/serving/expert\_parallel\_deployment.md at main · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/main/docs/serving/expert_parallel_deployment.md)
40. [Expert Parallel Deployment - vLLM](https://docs.vllm.ai/en/v0.18.0/serving/expert_parallel_deployment/)
41. [GitHub - deepseek-ai/DeepEP: DeepEP: an efficient expert-parallel communication library · GitHub](https://github.com/deepseek-ai/DeepEP)
42. [Does DeepEP support sm86?(4A6000 without NVL) · Issue #481 · deepseek-ai/DeepEP · GitHub](https://github.com/deepseek-ai/DeepEP/issues/481)
43. [GitHub - QingCheng-AI/DeepEP: DeepEP: an efficient expert-parallel communication library · GitHub](https://github.com/QingCheng-AI/DeepEP)
44. [vLLM Expert Parallel Deployment for DeepSeek and Other MoE Models](https://flatkey.ai/blog/vllm-expert-parallel-deployment)
45. [vllm serve - vLLM](https://docs.vllm.ai/en/v0.19.1/cli/serve/)
46. [\[CI/Build\] Enable block FP8 quantization tests on NVIDIA SM89 (Ada) by amankarki151 · Pull Request #60171 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/pull/60171)
47. [GitHub - QwenLM/Qwen3.8-Flash-Next: Qwen3.8-Flash-Next is the foundation model developed by Qwen Team, Alibaba Group. · GitHub](https://github.com/QwenLM/Qwen3.8-Flash-Next)
48. [dynamo/recipes/qwen3.8-flash-next/README.md at main · ai-dynamo/dynamo · GitHub](https://github.com/ai-dynamo/dynamo/blob/main/recipes/qwen3.8-flash-next/README.md)
49. [\[Bug\]: cutlass\_scaled\_mm\_supports\_block\_fp8 excludes SM89 (Ada Lovelace) despite native FP8 tensor core support](https://github.com/vllm-project/vllm/issues/58241)
50. [fp8 - vLLM](https://docs.vllm.ai/en/v0.18.0/api/vllm/model_executor/layers/fused_moe/oracle/fp8/)
51. [\[Performance\]: Qwen3.6-35B-A3B-FP8 on RTX PRO 6000 Blackwell has very low decode throughput; missing FP8 MoE config for E=256,N=256 · Issue #44688 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/issues/44688)
52. [Performance Tuning for L40s with Qwen36-35B](https://discuss.vllm.ai/t/performance-tuning-for-l40s-with-qwen36-35b/2848)
53. [vLLM on X: "Qwen3.8-Flash-Next from @Alibaba\_Qwen has day-0 support in vLLM, verified on NVIDIA and AMD GPUs. 🎉 Ultra-sparse multimodal MoE: 125B params, 6B active, 262K native, 1M via YaRN. On top of those sits a separate 51B N-gram table you can offload. Most of it will look familiar. The Gate… / X](https://x.com/vllm_project/status/2092600887873286157)
54. [\[Bug\]: Qwen3-Next FP8 error combining --tensor-parallel-size and --pipeline-parallel-size using MTP · Issue #27404 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/issues/27404)
55. [Qwen/Qwen3.6-35B-A3B-FP8 · Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)
56. [AzurePublicDataset/AzureLLMInferenceDataset2024.md at master · Azure/AzurePublicDataset · GitHub](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md)
57. [GitHub - winter131/cloudComputingPaper: ini proyek pengerjaan data azure cloud untuk paper cloud computing · GitHub](https://github.com/winter131/cloudComputingPaper)
58. [feat: run the CPU memory comparison on real inference traces by RiceXi · Pull Request #3 · xixii421/Professional-Internship · GitHub](https://github.com/xixii421/Professional-Internship/pull/3)
59. [Releases · HPMLL/BurstGPT · GitHub](https://github.com/HPMLL/BurstGPT/releases)
60. [GitHub - HPMLL/BurstGPT: A ChatGPT(GPT-3.5) & GPT-4 Workload Trace to Optimize LLM Serving Systems · GitHub](https://github.com/HPMLL/BurstGPT)
61. [BurstGPT: Ein Echtwelt-Arbeitslast-Datensatz zur Optimierung von LLM-Bereitstellungssystemen](https://www.alphaxiv.org/de/abs/2401.17644)
62. [GitHub - Semianalysis/agentx-conversation-explorer: AgentX Conversation Explorer — explore the shape of real agentic-inference traces: tokens, turns, timing (InferenceX AgentX datasets) · GitHub](https://github.com/Semianalysis/agentx-conversation-explorer)
63. [Agentic Benchmark Datasets - InferenceX - SemiAnalysis](https://inferencex.semianalysis.com/datasets)
64. [About](https://inferencex.semianalysis.com/about)
65. [InferenceX AgentX MVP Benchmark](https://docs.nvidia.com/aiperf/tutorials/datasets-inputs/inference-x-agent-x-mvp-benchmark)
66. [aiperf/docs/tutorials/weka-trace.md at main · ai-dynamo/aiperf · GitHub](https://github.com/ai-dynamo/aiperf/blob/main/docs/tutorials/weka-trace.md)
67. [AgentX Methodology](https://inferencex.semianalysis.com/agentx/methodology)
68. [Mooncake/FAST25-release/README.md at main · kvcache-ai/Mooncake · GitHub](https://github.com/kvcache-ai/Mooncake/blob/main/FAST25-release/README.md)
69. [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://dl.acm.org/doi/pdf/10.1145/3773772)
70. [Mocker Trace Replay](https://docs.nvidia.com/dynamo/v1.1.1/user-guides/mocker/mocker-offline-trace-replay)
71. [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://dl.acm.org/doi/10.1145/3773772)
72. [Vidur Framework: LLM Inference Simulation](https://www.emergentmind.com/topics/vidur-framework)
73. [Frontier: Towards Comprehensive and Accurate LLM Inference Simulation](https://arxiv.org/html/2605.21312)
74. [Frontier: Towards Comprehensive and Accurate LLM Inference Simulation](https://arxiv.org/pdf/2605.21312)
