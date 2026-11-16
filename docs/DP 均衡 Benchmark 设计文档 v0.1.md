# DP 均衡 Benchmark 设计文档 v0.1

Nov 11, 2026 · @Erkang Fei

## 1. 概要与范围

这个 benchmark 回答一个问题：给定模型、硬件和负载，一个 DP 均衡策略收回了多少因 rank 间同步等待而损失的吞吐，同时有没有违反延迟 SLO。主分是“回收率”：策略在轮询基线和完美均衡参考之间补上了多大比例的差距，取值 0 到 1，跨框架和硬件可比。

**目标**

- 用时间口径量化 DP attention + EP 部署中 rank 间不均衡造成的损失。
- 对 dispatch（请求进哪个 rank）和 admission（何时接纳 prefill）两类策略给出可复现的评分。
- 指标只依赖统一 trace，SGLang、vLLM、TensorRT-LLM 通过适配层接入。
- 覆盖 decode、prefill、混部三种阶段。

**非目标**

- 专家侧均衡（EPLB）不是被测对象，只作为固定条件记录。
- 不评多副本集群路由、扩缩容和模型输出质量。
- 模拟器不替代真机评分。它在最小回路跑通后用真机 trace 校准，只作为策略搜索的快速内环。

**已确定的决策**

| 事项 | 决定 | 对设计的影响 |
| --- | --- | --- |
| 硬件 | 4 × L40S 先做最小回路和模拟器校准，8 × L20 做 DP 8 和正式结果；单卡都是 48 GB | DP 规模在 8 卡机上覆盖 2、4、8，在 4 卡机上覆盖 2、4；没有跨节点时钟对齐问题；两种卡都只有 PCIe、没有 NVLink，同步阶段耗时占比会偏高 |
| 框架 | 规范框架无关，第一个适配 vLLM | 先定 trace 规范和指标；适配层顺序为 vLLM、SGLang |
| 阶段 | decode、prefill、混部全覆盖 | 三组场景，各有自己的 SLO 约束 |
| DP 规模的配法 | 8 卡全用，rank 内的 attention 做 TP | DP8 = TP1，DP4 = TP2，DP2 = TP4；EP 规模恒为 8，只有 DP 规模在变；4 卡机上 DP4 = TP1、DP2 = TP2，EP 恒为 4 |
| mirror 副本 | 长度相同、内容不同 | mirror 的专家侧负载与被测运行接近，回收率不被高估 |
| 模型 | 第一模型 Qwen3.6-35B-A3B；大模型待定 | 两台机器都放得下；混合注意力结构，decode 场景的可回收空间可能偏小 |
| vLLM 版本 | 钉 v0.32.0（2026-10-18 发布） | 埋点位置、默认均衡的打分公式和启动参数都以这个 tag 的源码为准；要求 Python 3.11 以上；manifest 记录版本和 commit |
| 第一步产出 | 本设计文档 | 代码在文档评审之后开始 |

## 2. 问题模型与术语

不均衡的代价来自一条规则：每个 MoE 层的 all-to-all 必须等所有 rank 到齐，所以每一步的耗时由最慢的 rank 决定。先到的 rank 只能空等，这段空等就是 benchmark 要量的损失。

```latex
T_{step} = \sum_{l}\left(\max_{r} a_{r,l} + c_{l}\right)
```

其中 a 是 rank r 在第 l 层的本地耗时（主要是 attention），c 是该层同步阶段（dispatch、专家计算、combine）的耗时。

| 术语 | 含义 |
| --- | --- |
| rank | 一个 DP 副本，持有自己的请求和 KV cache |
| DEP-N | attention 用 DP、MoE 用 EP 的部署，共 N 个 rank |
| step | 所有 rank 共同完成的一次前向 |
| 层调用 | 一个 step 内某个 MoE 层的一次执行 |
| 同步点 | dispatch 和 combine 两次 all-to-all，全体到齐才能完成 |
| 就绪时刻 | 某 rank 在某层算完本地部分、准备进入 dispatch 的时刻 |
| 空等 | 一个 rank 就绪后，等最后一个 rank 就绪的时间 |
| dispatch 策略 | 决定新请求进哪个 rank，分配后不可迁移 |
| admission 策略 | 决定某个 rank 在某一步接纳多少 prefill |

三种阶段的不均衡来源不同，策略能动的手段和要守的延迟约束也不同。

| 阶段 | 不均衡从哪来 | 策略能动什么 | 延迟约束 |
| --- | --- | --- | --- |
| decode（PD 分离） | 各 rank 的 KV 总量和 batch 大小不同；请求越生成越长，何时结束事先不知道 | dispatch | TPOT |
| prefill（PD 分离） | 各 rank 本步 token 数不同；前缀缓存命中不同；到达时刻错位 | dispatch、admission | TTFT |
| 混部（IFB） | 一个 rank 在做 prefill、其余在 decode，这一步全体被拖到 prefill 的耗时 | admission 为主，dispatch 为辅 | TTFT 和 TPOT |

哪种“负载”定义有效取决于模型。[vLLM 在 DeepSeek-V4.1-Flash 上](https://vllm.ai/blog/2026-10-23-deepseek-v41-flash-pd-wide-ep)发现 decode 单步耗时几乎不随上下文长度变化，[NanoCP 在 DeepSeek-V3 上](https://arxiv.org/pdf/2605.21100)则看到 KV 量和 batch 大小都起作用。所以本 benchmark 以时间口径为准，负载口径只做诊断。

## 3. 指标体系

主分是回收率，约束是延迟 SLO，其余指标用于诊断和校验。绝对吞吐随硬件、模型和框架变化，不同来源的结果无法直接比较；回收率把策略的贡献归一到轮询基线和均衡参考之间，可以跨环境比较。

```latex
R = \frac{X_{\pi} - X_{RR}}{X_{mirror} - X_{RR}}
```

X 是饱和吞吐，下标依次为被测策略、轮询基线、mirror 参考。R = 0 表示与轮询持平，R = 1 表示达到均衡参考。mirror 是均衡参考而不是严格上界，R 略大于 1 或小于 0 都照实报告。

| 层 | 指标 | 定义 | 角色 |
| --- | --- | --- | --- |
| 负载侧 | 均衡比 BR | 每步各 rank 负载的平均值 ÷ 最大值；负载分别取本步 token 数、KV token 总量、请求数 | 诊断 |
| 时间侧 | 注意力均衡 ALB | 每次层调用各 rank 本地耗时的平均值 ÷ 最大值 | 诊断 |
| 时间侧 | 空等占比 W | 所有 rank 在所有同步点前的空等时间之和 ÷（N × 总时长） | 主诊断 |
| 上界 | SOL 吞吐 | 假设每步都完美均衡时的理论吞吐，由 BR 推算 | 校验 |
| 上界 | mirror 吞吐 | 同一请求集合在 mirror 回放下的实测吞吐 | 回收率的分母 |
| 端到端 | 饱和吞吐 X | 固定请求集合的总 token 数 ÷ 完成全部请求的总时长 | 主分输入 |
| 端到端 | 可承载请求率 λ\* | SLO 达标率不低于 99% 时的最大到达率 | 完整档主分 |
| 端到端 | P99 TTFT、P99 TPOT | 按请求统计 | 约束 |

均衡比和空等占比的精确定义如下，L 是 rank r 在 step s 的负载，t 是 rank r 在层调用 i 的就绪时刻。两者分别沿用 [TensorRT-LLM 的 balance ratio](https://nvidia.github.io/TensorRT-LLM/latest/blogs/tech_blog/blog10_ADP_Balance_Strategy.html) 和 [CascadeEP 的 long-tail ratio](https://arxiv.org/html/2609.33252)。

```latex
BR_s = \frac{\frac{1}{N}\sum_{r} L_{r,s}}{\max_{r} L_{r,s}}
\qquad
W = \frac{\sum_{i}\sum_{r}\left(\max_{r'} t_{r',i} - t_{r,i}\right)}{N \cdot T_{e2e}}
```

SOL 吞吐沿用 TensorRT-LLM 的算法：把每步实测耗时 τ 乘上该步的均衡比，得到理想总时长。它假设单步耗时与最大负载成正比，只是近似，所以只用于和 mirror 实测互相校验。

```latex
T_{sol} = \sum_{s} \tau_s \cdot BR_s
\qquad
X_{sol} = X \cdot \frac{T_{e2e}}{T_{sol}}
```

评测分两档：

- **快速档**：回放一个固定请求集合，测饱和吞吐 X，计算回收率 R。一个配置跑一次即得一个分数。
- **完整档**：开环到达，扫描到达率求 λ\*，报告被测策略与轮询基线的 λ\* 之比。每个配置要跑 6 到 8 个到达率。

## 4. 统一 trace 与埋点规范

所有指标只从三张表算出来，框架只要能产出这三张表就能接入。计算指标的脚本不读任何框架内部状态。

| 记录 | 一行代表 | 字段 | 级别 |
| --- | --- | --- | --- |
| request | 一个请求 | req\_id、session\_id、arrival\_ts、dispatch\_ts、prefill\_start\_ts、rank、input\_tokens、cached\_tokens、output\_tokens、first\_token\_ts、finish\_ts | L0，必需 |
| step | 一个 rank 的一个 step | step\_id、rank、t\_start、t\_end、prefill\_tokens、decode\_tokens、num\_reqs、kv\_tokens | L1，必需 |
| layer | 一个 rank 的一次层调用 | step\_id、layer\_id、rank、t\_ready、t\_dispatch\_done、t\_combine\_done | L2，采样 |

L0 算端到端指标，L1 算均衡比和 SOL，L2 算 ALB 和空等占比。只有 L0 和 L1 也能出回收率，但没有时间侧诊断。

**采集规则**

- step\_id 全局对齐：同一步各 rank 用同一个编号。
- 时间戳用本机单调时钟，单位纳秒。
- GPU 异步执行，时间戳必须取在算子真正完成之后。L2 只对每 k 步采样一次，避免额外的同步调用改变被测行为。
- prefill\_tokens 只计实际计算的 token，前缀缓存命中的部分记在 cached\_tokens。
- kv\_tokens 是该 rank 本步所有活跃请求的上下文长度之和。
- 文件格式为 Parquet 或 JSONL。每次运行附一个 manifest，记录硬件、模型与精度、框架版本与 commit、DP/EP/TP 配置、EPLB 配置、batch 上限、埋点级别和负载种子。

**各框架适配层**

挂载位置是根据文档的预期，要读代码确认。

| 框架 | 预期挂载位置 | 已有的可复用部分 |
| --- | --- | --- |
| [SGLang](https://sgl-project.github.io/_sources/advanced_features/dp_dpa_smg_guide.md) | DP 控制器的请求分发处；调度器的 batch 循环；MoE dispatch 入口 | 原生 DP 自带 round\_robin、total\_requests、total\_tokens 三种分发 |
| [vLLM](https://docs.vllm.ai/en/v0.10.1/api/vllm/v1/engine/coordinator.html) | 前端选引擎处；引擎核心的 step 循环；MoE dispatch 入口 | DP coordinator 已汇总各引擎的 waiting、running 队列长度 |
| [TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/latest/blogs/tech_blog/blog10_ADP_Balance_Strategy.html) | 执行器的调度循环 | 官方博客已报告 balance ratio 和 SOL TPS |

**vLLM 适配要点**

下面是对 vLLM main 分支和官方文档的调研结论，要在 v0.32.0 上逐项确认后才算数。

- **钉 rank：** 请求头 `X-data-parallel-rank` 可以把请求发到指定 rank，mirror 和 lag 回放不用改 vLLM。
- **dispatch 策略：** 先实现在外部回放器里，由回放器自行记账各 rank 的请求数和 KV token 数，再用请求头钉 rank。服务端只开一个 API server。
- **step 埋点：** 挂在 `vllm/v1/engine/core.py` 的 `DPEngineCoreProc.run_busy_loop` 和 `EngineCore.step`。step\_id 取“wave 编号加 wave 内前向序号”，空 rank 的 dummy 前向也计数。
- **layer 埋点：** 挂在 MoE 层的 dispatch 和 combine 前后。CUDA Graph 回放时 Python 钩子不执行，所以 L2 要在单独的 `--enforce-eager` 运行里采，不和计分运行混在一起。
- **admission 策略：** 跨 rank 的等待先做成回放器里的前端门控，满足条件才把请求放出；引擎内的做法是子类化 `Scheduler`。
- **通信后端：** 只能用默认的 `allgather_reducescatter`。DeepEP 要求 Hopper 以上的卡和 NVLink，L40S 和 L20 用不了。
- **运行开关：** 关闭 async scheduling、前缀缓存和 MTP；加 `--language-model-only` 跳过视觉编码器。

## 5. 负载与场景设计

负载设计守一条原则：不同策略面对的总工作量必须完全相同，否则吞吐差异无法归因到均衡。为此有三条硬规则。

1. **输出长度固定。** 每个请求强制生成到 trace 记录的长度，忽略 EOS。
2. **快速档用固定请求集合。** 吞吐按“完成全部请求的总时长”算，不按固定时长算。[AgentX 的方法说明](https://inferencex.semianalysis.com/agentx)提到，闭环回放下更快的配置会遇到略有不同的负载组合。
3. **请求内容用真实语料按长度裁剪。** 随机 token 会改变专家路由的分布。每次运行同时记录 EP 侧均衡度，作为协变量。

**负载来源**

| 来源 | 提供什么 | 用途 |
| --- | --- | --- |
| [Azure LLM inference trace 2024](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md) | 时间戳、输入 token 数、输出 token 数 | 真实到达模式和长度分布；BalanceRoute 用过 |
| [BurstGPT](https://github.com/HPMLL/BurstGPT) | 时间戳、请求与响应 token 数 | 突发到达 |
| [AgentX v1.0](https://inferencex.semianalysis.com/agentx) | 393 个编码 agent 会话，保留请求长度、前缀复用和时序；输入中位数 14.2 万 token | 多轮、高前缀复用场景；CascadeEP 用过 |
| 合成负载 | 长度分布、长短混合比例、突发程度均可调 | 受控实验和剂量反应校验 |

**mirror 对照的构造**

mirror 参考和被测运行用同一个请求集合，只有到达时刻和分配方式不同。这个做法扩展自 [vLLM 的 mirror/lag 回放](https://vllm.ai/blog/2026-10-23-deepseek-v41-flash-pd-wide-ep)。

- **基础流 w**：从 trace 抽出的一段请求序列。请求集合是 w 的 N 份副本：各副本的到达时刻、输入长度、输出长度逐请求相同，文本取自语料的不同位置。
- **mirror 运行**：每个 rank 回放一份副本，同时到达，固定绑定 rank。各 rank 每一步的 token 数完全相同，文本不同，所以专家侧负载与被测运行接近。
- **被测运行**：同样 N 份副本，到达时刻按固定种子打散，由被测策略分配。
- **lag 运行**：rank r 的那份延迟 r × Δ 到达，固定绑定。只用于校验指标（见第 8 节）。

这组对照要求关闭前缀缓存，因为同一会话的后续请求能否命中缓存取决于它被分到哪个 rank，工作量会随策略变化。开启前缀缓存的场景不算回收率，只报告 λ\* 之比和 SOL。

**场景矩阵**

| 编号 | 阶段 | 负载 | 前缀缓存 | 考察点 |
| --- | --- | --- | --- | --- |
| D1 | decode | 等长输入、等长输出 | 关 | 零点：应几乎没有不均衡，用来测噪声地板 |
| D2 | decode | 输出长度长尾分布 | 关 | 请求结束时刻不可知带来的漂移 |
| D3 | decode | 短请求混入 1% 到 5% 长上下文 | 关 | KV 量与 batch 大小两个维度的冲突 |
| D4 | decode | D2 加突发到达 | 关 | 非平稳流量 |
| P1 | prefill | 等长输入 | 关 | 零点 |
| P2 | prefill | 变长输入 | 关 | 每步 token 数不均 |
| P3 | prefill | AgentX 多轮会话 | 开 | 缓存亲和与均衡的冲突 |
| M1 | 混部 | 短输入、长输出 | 关 | prefill 打断 decode；ADP Balance 的目标场景 |
| M2 | 混部 | 长输入、短输出 | 关 | prefill 占主导时的对齐 |
| M3 | 混部 | Azure trace 原始到达 | 关 | 真实流量下的综合表现 |

每个场景在 DP 规模 2、4、8 下各跑一遍，观察损失是否随规模增长。

## 6. 被测策略接口与基线

策略只能读在线信息，并在限定时间内给出决定。接口把策略能看到的字段写死，不同策略的分数才可比，也堵住了偷看输出长度这类捷径。

**两类接口**

| 项 | dispatch | admission |
| --- | --- | --- |
| 调用时机 | 每个新请求到达时一次 | 每个 rank 每步开始前一次 |
| 可读的请求信息 | input\_tokens、session\_id、到达时刻；开启前缀缓存时加上各 rank 的命中 token 数 | 本 rank 等待队列里每个请求的 input\_tokens、cached\_tokens、已等待时长 |
| 可读的 rank 状态 | 每个 rank 的 running 和 waiting 请求数、kv\_tokens、待算 prefill token 数、最近若干步的 step 耗时 | 同左；其他 rank 的状态允许滞后一步 |
| 输出 | 一个 rank 编号 | 本步接纳的请求列表，可以为空 |
| 不可读 | output\_tokens、未来的到达 | 同左 |
| 硬约束 | 分配后不可迁移 | 不得丢弃或拒绝请求；同一请求被连续跳过的步数有上限 |

- 允许使用预测器（例如输出长度预测），但训练数据不得包含测试负载。
- 有状态的策略，状态只能由上表的可读字段累积而来。
- 决策耗时计入端到端时间。dispatch 决策的 P99 暂定不超过 3 ms：[BalanceRoute](https://arxiv.org/pdf/2605.06113) 在 8 个 worker 的 DeepSeek-V3 上实测每轮 P99 约 2.8 ms，引擎单步约 60 ms。

**必测基线**

评 dispatch 策略时 admission 固定为“立即接纳”；评 admission 策略时 dispatch 固定为 RR 和 JSQ 各跑一遍。这样两类策略的贡献不会混在一起。

| 策略 | 类型 | 规则 | 现成实现 |
| --- | --- | --- | --- |
| RR 轮询 | dispatch | 按到达顺序循环分配；回收率的零点 | SGLang `round_robin` |
| P2C | dispatch | 随机抽两个 rank，选请求数较少的 | 需实现 |
| JSQ 最少请求数 | dispatch | 选 running + waiting 请求数最少的 rank | vLLM 内部均衡（调研显示它按 waiting × 4 + running 打分，单列为一个基线）；SGLang `total_requests` |
| 最少 token | dispatch | 选在算 token 数最少的 rank | SGLang `total_tokens` |
| 最少 KV | dispatch | 选 kv\_tokens 最小的 rank | 需实现 |
| 排序轮询 | dispatch | 一批请求按 token 数排序后轮询分配 | [TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/latest/blogs/tech_blog/blog10_ADP_Balance_Strategy.html) 的基线 |
| BR-0 | dispatch | 先填满各 rank 距最重 rank 的安全余量，不得不溢出时选代价最小的 rank；不需要输出长度预测 | [BalanceRoute](https://arxiv.org/pdf/2605.06113) 开源版，基于 vllm-ascend |
| 立即接纳 | admission | 有空位就接纳，不等待 | 各框架默认行为 |
| ADP Balance | admission | 某 rank 有 prefill 而其他 rank 在 decode 时，最多等 `timeout_iters` 步；对齐后可再等 `batching_wait_iters` 步攒批 | [TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/latest/blogs/tech_blog/blog10_ADP_Balance_Strategy.html) |

“现成实现”一列按文档和参数名对应，具体选择逻辑要读代码确认。

**参考策略（不参与排名）**

离线 oracle 预先知道全部请求的到达时刻和输出长度，按总工作量从大到小依次放到当前最轻的 rank。它把剩余差距拆成两段：在线策略到 oracle 的差距是预测还能补的部分，oracle 到 mirror 的差距是 dispatch 本身补不了、要靠 admission 的部分。

## 7. 评分与报告规则

每个场景报一个带置信区间的回收率，延迟超标的结果不计分。这些规则保证分数只能靠更好的均衡得到，不能靠改变工作量或牺牲延迟得到。

**主分**

- 快速档：每个“场景 × DP 规模”报 R 及其 95% 置信区间。
- 完整档：报被测策略与 RR 的 λ\* 之比。
- admission 策略必须跑完整档才能宣称改进。它用 TTFT 换吞吐，快速档看不到这笔代价。
- 按阶段汇总：decode、prefill、混部各取场景 R 的算术平均，并列出最差场景。不合成跨阶段总分。

**延迟约束**

| 档 | 约束 | 违反时 |
| --- | --- | --- |
| 快速档 | P99 TPOT 不超过 RR 运行的 1.1 倍（暂定） | 结果标“违约”，不计分 |
| 完整档 | SLO 达标率不低于 99%，SLO 数值写在 manifest 里 | 该到达率不算可承载 |

快速档不约束 TTFT：饱和回放下 TTFT 主要是排队时间，不反映策略好坏。纯 prefill 场景没有 TPOT，快速档不设延迟约束，延迟代价由完整档负责。

**重复与置信区间**

- 每个配置用同一负载种子重复 5 次，量系统噪声。
- 主结论在 3 个负载种子上复核，量负载抽样带来的波动。
- R 的置信区间用 bootstrap：对被测策略、RR、mirror 三组吞吐样本各自重采样，重算 R，取 2.5% 和 97.5% 分位。
- 两个策略的置信区间重叠时，报告为“无显著差异”。
- 计时前先跑一个单独的预热集合，预热请求不计入 X。
- 可回收空间太小的场景不报 R，只写“无可回收损失”。判据是 mirror 与 RR 的吞吐差占 mirror 吞吐的比例小于噪声地板的 3 倍，此时 R 的分母接近零。

**防作弊规则**

1. **工作量守恒。** 完成请求数等于集合大小，每个请求的输出长度等于 trace 值，总计算 token 数与 RR 运行相等。不满足的运行作废。
2. **配置冻结。** manifest 中除策略字段外，与基线运行逐项相同。
3. **参数不逐场景调。** 一套策略参数跑完同一阶段的全部场景；逐场景调参的结果单独标注。
4. **留出负载。** 开发用公开种子，报告分数用另一组留出种子。
5. **信息边界。** 策略只通过第 6 节的接口取数，接口不暴露 output\_tokens。
6. **决策耗时。** 计入端到端时间，超过预算的运行作废。

**随结果提交的内容**

manifest、三张表的原始 trace、R 与置信区间、空等占比 W、三种负载口径的 BR、P99 TTFT 和 P99 TPOT、EP 侧均衡度、是否违约。

## 8. benchmark 自身的可靠性验证

benchmark 在用来给策略打分之前，先要通过下面 8 个验收实验。它们检查三件事：噪声够小、指标确实在量不均衡、结果能复现。通过标准都是暂定值，V1 实测后再校准。

| 编号 | 检查什么 | 做法 | 通过标准 |
| --- | --- | --- | --- |
| V1 | 噪声地板 | D1 场景、RR 策略、同一种子重复 10 次 | 吞吐 X 的变异系数不超过 1%；这个值就是第 7 节用的噪声地板 |
| V2 | 零点 | D1、P1 上分别跑 mirror 和 RR；再跑一次副本内容相同的 mirror 作对照 | mirror 与 RR 的吞吐差小于噪声地板的 2 倍；mirror 的 W 记为 W 的地板；记录两种 mirror 的吞吐和 EP 侧均衡度之差 |
| V3 | 剂量反应 | lag 运行的 Δ 从 0 起取 5 档；合成负载的长请求比例从 0 调到 5% | W 随剂量单调上升，X 单调下降，秩相关不低于 0.9 |
| V4 | 三种上界互相印证 | 对同一次运行算三个均衡吞吐估计：mirror 实测、SOL、由 W 推算 | 两两相差不超过 5%；超出时按下表定位 |
| V5 | 埋点开销 | 同一配置分别开关 L1、L2 | L1 造成的吞吐下降不超过 0.5%，L2 不超过 1% |
| V6 | 工作量守恒 | 比较各策略运行的总计算 token 数和 EP 侧均衡度 | token 数完全相等；EP 侧均衡度的差异小于它自身的重复波动 |
| V7 | 已知结论复现 | M1 上对比 ADP Balance 与排序轮询；M3 上对比 BR-0 与 JSQ | 方向与公开结果一致（见表下） |
| V8 | 可复现 | 重启后隔天重跑同一 manifest；在 4 × L40S 上以 DP 规模 2、4 再跑一遍 | R 落在原置信区间内；策略排名的 Kendall τ 不低于 0.8 |

V7 对照的公开结果：[TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/latest/blogs/tech_blog/blog10_ADP_Balance_Strategy.html) 报告 ADP Balance 把平均 balance 从 54.11% 提到 87.70%，吞吐为基线的 1.33 倍；[BalanceRoute](https://arxiv.org/pdf/2605.06113) 报告 BR-0 在 Azure-2024 上吞吐 854 tok/s，JSQ 为 785 tok/s。只要求方向一致，不要求幅度相同，因为硬件和模型不同。

**V4 的第三个估计**

由 W 推算的均衡吞吐基于一个守恒假设：负载在 rank 间挪匀后，每层的本地耗时从各 rank 的最大值降到平均值，本地总耗时不变。省下的时间正好是全部空等时间除以 N，即 W 乘总时长。

```latex
X_{W} = \frac{X}{1 - W}
```

三个估计的假设不同，所以分歧本身能定位问题。

| 现象 | 可能原因 |
| --- | --- |
| SOL 偏离另外两个 | 所选负载口径与耗时不成正比；换 token、KV、请求数中的另一种重算 |
| W 推算值偏离 mirror | 本地耗时对负载不是线性的，例如有固定开销或 batch 效应 |
| mirror 偏离另外两个 | mirror 运行改变了专家侧负载（见 V2 的对照），或各 rank 没有真正对齐 |

## 9. 实施计划

计划分两段：先在 4 × L40S 上跑通最小回路并校准模拟器（S0 到 S4），再到 8 × L20 上做 DP 8、完整验收和正式结果（S5 到 S8）。每步有一个可检查的过关条件。S1 和 S2 不需要 GPU，可以并行，现在就能开始。

| 步骤 | 机器 | 做什么 | 产出 | 过关条件 |
| --- | --- | --- | --- | --- |
| S0 规范冻结 | 无 | 关闭第 10 节里需要拍板的问题 | trace 和 manifest 的 schema 文件 | 拍板项全部勾掉 |
| S1 指标库 | 无 | 写纯 Python 包，从三张表算出第 3 节的全部指标 | 指标库和单元测试 | 在手工构造的合成 trace 上，每个指标与手算结果一致 |
| S2 负载生成与回放 | 无 | 先做合成负载和 Azure trace；实现 mirror、lag、打散三种回放；强制输出长度 | 负载生成器、回放客户端 | 同一种子两次生成的请求集合完全相同 |
| S3 最小回路 | 4 × L40S | vLLM 加 Qwen3.6-35B-A3B，DP 4；接 L0、L1 埋点；dispatch 接 RR 和 JSQ；跑 D1、D2、P2 和 lag 运行 | 适配层；第一组回收率和均衡比 | 工作量守恒检查通过；V1、V2 通过；lag 运行测得出吞吐下降 |
| S4 模拟器 v0 | 4 × L40S 采数，模拟器只用 CPU | 加 L2 采样；用 trace 拟合单步耗时模型；写 step 级模拟器，复用同一套策略接口和指标库 | 模拟器、校准报告 | 在留出的真机运行上，预测吞吐误差不超过 5%；至少 3 个策略的排名与真机一致 |
| S5 扩到 8 卡 | 8 × L20 | 同一套代码跑 DP 8、4、2；加 admission 接口；重新校准模拟器 | DP 8 上的第一组结果 | D1 跑通；模拟器在 DP 8 上重新通过 S4 的过关条件 |
| S6 可靠性验收 | 8 × L20，V8 加用 4 × L40S | 跑 V1 到 V8 | 验收报告；校准后的阈值 | 8 项全部通过，未通过项有书面结论并回改设计 |
| S7 基线全量 | 8 × L20 | 必测基线 × 场景矩阵 × DP 规模，先快速档再完整档；加大模型 | 基线结果表，v1.0 发布 | 每个结果都带第 7 节要求的提交内容 |
| S8 第二个适配层 | 8 × L20 | 在 SGLang 上重复 S5 到 S7 的子集 | 跨框架对照 | 同一策略在两个框架上的 R 排名一致，差异有解释 |

先在 4 卡上做有三条限制。

- DP 最大只到 4。不均衡损失随 DP 规模增长，4 卡上的可回收空间更小；如果低于噪声地板的 3 倍，最小回路就靠 lag 运行人为制造不均衡来验证指标。
- 4 卡上校准的模拟器不能外推到 DP 8。EP 规模、同步耗时和卡的算力都变了，S5 必须重新校准。
- 模拟器只用于策略搜索的快速内环，报告分数一律来自真机。

## 10. 未决问题

原有 6 件拍板项已定，新增 1 件模型选择待定；另有 10 件要等实测数据。每项下面是结论或我的建议。

**需要你拍板**

- [x] GPU 型号和单卡显存是什么？

  已定：4 × L40S 先做最小回路和模拟器校准，8 × L20 做 DP 8 和正式结果；单卡都是 48 GB，只有 PCIe。模型选择见下面新增的一项。
- [x] 第一个适配哪个框架？

  已定：vLLM。SGLang 放在 S8。
- [x] mirror 的 N 份副本要不要改成“长度相同、内容不同”？

  已定：改。第 5 节的 mirror 构造已更新，V2 增加了与内容相同 mirror 的对照。
- [x] DP 规模 2 和 4 怎么配？

  已定：8 卡全用，rank 内的 attention 做 TP。vLLM 文档确认支持：TP=2、DP=4 时 EP 组大小为 8。但 Qwen3.5 架构在 TP 大于 1 加 DP 加 EP 时有输出错乱的已知问题（vLLM issue #37856），所以最小回路先只用 DP4 = TP1，用 DP2 = TP2 之前要先与 TP1 的输出逐 token 比对。
- [x] 完整档的 SLO 数值定多少？

  已定：TPOT 用固定值，TTFT 为固定项加每千 token 的增量。具体数等 S3 拿到单请求基准延迟后再填。
- [x] request 表要不要加 prefill\_start\_ts？

  已定：加。第 4 节的 request 表已更新。

* [ ] 用哪个 MoE 模型？

  第一模型已定：[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)。35B 总参数、3B 激活，BF16 约 70 GB，两台机器都放得下。[vLLM 配方](https://recipes.vllm.ai/Qwen/Qwen3.6-35B-A3B)要求 0.17.0 以上。

  大模型待你确认：建议用 [Qwen3.8-Flash-Next](https://www.datacamp.com/blog/qwen3-8-flash-next) 的 FP8 版，不等 Qwen4。它已经是 Qwen4 的架构，125B 总参数、6B 激活；Qwen4 目前只点名了 27B 这一档开放权重，规格和日期都没公布。按参数量粗算，FP8 版在 8 × L20 上 DP8 每卡约占 27 GB，只能在 8 卡机上跑。调研后风险上调：官方配方没有在 Ada 卡上验证过，稀疏注意力的内核能否在 L20 上运行未知，每个 DP 进程还可能各占约 51 GB 主机内存放查找表。

  两个模型都是混合结构：四分之三的层是 Gated DeltaNet，只有四分之一是注意力层。decode 本地耗时对上下文长度不敏感，decode 场景的可回收空间可能偏小，prefill 和混部场景不受影响。用之前要确认 vLLM 的 EP 支持这两个架构，评测时关闭 MTP 推测解码。

**需要实测才能定**

- [ ] 噪声地板的实际值（V1），以及由它推出的各项阈值。
- [ ] 基础流 w 取多长，一次快速档运行才既稳定又不太久。
- [ ] L2 采样间隔 k 取多少，埋点开销才低于 1%（V5）。
- [ ] lag 运行的 Δ 取哪 5 档。[vLLM 的做法](https://vllm.ai/blog/2026-10-23-deepseek-v41-flash-pd-wide-ep)是每个 rank 递增 5 秒。
- [ ] dispatch 决策 3 ms、快速档 TPOT 1.1 倍这两个暂定预算是否合适。

* [ ] PCIe 互联下，同步阶段占单步耗时的比例。占比过高时空等损失会被稀释，可回收空间可能低于噪声地板的 3 倍，那时要调场景参数来放大本地耗时。

- [ ] Qwen3.6-35B-A3B 的 decode 单步耗时随上下文长度和 batch 大小怎么变。它决定 decode 场景以哪种负载口径为主，以及 D3 是否还有意义。

* [ ] 在 v0.32.0 上确认 step 对齐：关闭 async scheduling 后，各 rank 每个 wave 的前向次数是否严格相等。
* [ ] DP 1 与 DP 4 对同一批请求的输出是否逐 token 一致。用来排除 dummy 前向对 Gated DeltaNet 状态的影响。
* [ ] 开 CUDA Graph 与 `--enforce-eager` 两种运行的单步耗时差多少。它决定 L2 数据能不能代表计分运行。

## 参考资料

**指标与方法**

- [TensorRT-LLM：ADP Balance Strategy](https://nvidia.github.io/TensorRT-LLM/latest/blogs/tech_blog/blog10_ADP_Balance_Strategy.html)：balance ratio、SOL TPS、ADP Balance 策略
- [vLLM：Serving DeepSeek-V4.1-Flash at Scale](https://vllm.ai/blog/2026-10-23-deepseek-v41-flash-pd-wide-ep)：mirror/lag 回放
- [CascadeEP（arXiv 2609.33252）](https://arxiv.org/html/2609.33252)：ALB、long-tail ratio
- [BalanceRoute（arXiv 2605.06113）](https://arxiv.org/pdf/2605.06113)：BR-0、基线策略、决策耗时实测
- [NanoCP（arXiv 2605.21100）](https://arxiv.org/pdf/2605.21100)：KV 量与 batch 大小两个负载维度、SLO 约束下的最大请求率
- [BF-IO（arXiv 2601.17855）](https://arxiv.org/abs/2601.17855v1)：同步屏障下的空等损失

**框架文档**

- [SGLang：DP, DPA and SGLang DP Router](https://sgl-project.github.io/_sources/advanced_features/dp_dpa_smg_guide.md)
- [vLLM：DP Coordinator](https://docs.vllm.ai/en/v0.10.1/api/vllm/v1/engine/coordinator.html)

* [vLLM：Expert Parallel Deployment](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment.html)：TP、DP、EP 的组合方式和 all2all 后端

**负载来源**

- [Azure LLM inference trace 2024](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md)
- [BurstGPT](https://github.com/HPMLL/BurstGPT)
- [AgentX v1.0](https://inferencex.semianalysis.com/agentx)
