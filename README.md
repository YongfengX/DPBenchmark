# dpbench

DP 均衡 benchmark：trace 与 manifest 的规范、指标库、负载生成与回放器（S0–S2），
以及 vLLM v0.32.0 的 step 埋点（S3 的第一部分，还没上过 GPU）。
对应设计文档《DP 均衡 Benchmark 设计文档 v0.1》的第 3–7 节。

layer 埋点（L2）和模拟器（S4）还没有。

## 安装

需要 Python 3.11 以上。

```bash
pip install -e ".[dev]"
pytest            # 不需要 GPU
```

## 不用 GPU 先跑通整条链路

`mock-server` 是一个假的 completions 服务：它认 rank 请求头、强制输出长度、按流式协议返回，
耗时是玩具模型。它只用来验证工具链，跑出来的回收率没有意义。

```bash
dpbench mock-server --port 8000 --ranks 4 &

# 同一个基础流的两种调度：mirror 参考和被测运行
dpbench gen --scenario D2 --n 50 --rate 100 --seed 1 --copies 4 --mode mirror --out wl/mirror.jsonl
dpbench gen --scenario D2 --n 50 --rate 100 --seed 1 --copies 4 --mode test   --out wl/test.jsonl

M=examples/manifest.4xL40S.vllm-0.32.json
dpbench replay --workload wl/mirror.jsonl --manifest $M --out runs/mirror-0
dpbench replay --workload wl/test.jsonl   --manifest $M --policy rr  --out runs/rr-0
dpbench replay --workload wl/test.jsonl   --manifest $M --policy jsq --out runs/jsq-0

dpbench validate runs/jsq-0
dpbench metrics  runs/jsq-0
dpbench recovery --policy runs/jsq-0 --rr runs/rr-0 --mirror runs/mirror-0
```

每个配置重复多次后，把多个目录一起传给 `recovery`，它会给出回收率的 95% 置信区间。

## 在 4 × L40S 上跑真机（S3 的起点）

1. 按 `examples/manifest.4xL40S.vllm-0.32.json` 里的 `launch_command` 启动 vLLM v0.32.0，
   把实际运行的 commit 填进 manifest。
2. 准备 token 语料：把一份真实文本用模型的 tokenizer 编码成一维 `int64` 数组，存成 `.npy`，
   生成负载时加 `--corpus corpus.npy`。不加语料时 prompt 是随机 token，会改变专家路由，只能用于联调。
3. 先发一个请求确认三件事：`usage.completion_tokens` 等于要求的输出长度；
   `X-data-parallel-rank` 确实把请求送到了指定 rank；`cached_tokens` 为 0。
4. 之后的命令与上面相同，把 `--base-url` 指向 vLLM。

要 step 表（L1）时，先按 [adapters/vllm/v0.32.0/README.md](adapters/vllm/v0.32.0/README.md) 给 vLLM 打补丁，
带着 `DPBENCH_STEP_TRACE_DIR` 启动，每次回放后跑 `dpbench merge-steps <运行目录> --trace-dir <目录>`。
layer 表（L2）还没有；`prefill_start_ts` 和 `cached_tokens` 两列也还要由服务端补。

## 目录

| 文件 | 内容 |
| --- | --- |
| `src/dpbench/schema.py` | 三张表的列定义与校验，manifest 校验，配置冻结检查 |
| `src/dpbench/schemas/manifest.schema.json` | manifest 的 JSON Schema |
| `src/dpbench/io.py` | 运行目录的读写（Parquet 或 JSONL） |
| `src/dpbench/metrics.py` | 均衡比、ALB、空等占比、SOL、吞吐、回收率与置信区间、延迟、SLO、λ*、工作量守恒 |
| `src/dpbench/workload.py` | 场景 D1–D4、P1–P2、M1–M2；Azure 与 BurstGPT 加载；N 份副本；mirror / lag / 打散三种调度 |
| `src/dpbench/policies.py` | dispatch 策略接口和基线：rr、p2c、jsq、least_tokens、least_kv、vllm_default、server |
| `src/dpbench/replay.py` | 开环回放器兼外部路由，写 request 表 |
| `src/dpbench/mock_server.py` | 联调用的假服务 |
| `src/dpbench/vllm_step_trace.py` | vLLM 引擎进程里的 step 记录器；per-rank 文件的合并、切窗和对齐检查 |
| `adapters/vllm/v0.32.0/` | vLLM 的挂钩补丁、用法，以及在 v0.32.0 源码上核对调研结论的记录 |
| `tests/conftest.py` | 一份可以手算的两 rank、两步、两层 trace，指标测试都对着它的手算结果 |

## 与设计文档的三处差别

- **有空闲间隙时的 SOL 和 W。** 文档的公式假设 step 首尾相接。代码把 step 之外的空闲时间原样保留，
  只压缩 step 内的时间；没有间隙时与文档公式相同。
- **L2 采样时的 W。** 先在被采样的 step 上算空等占比，再按全部 step 的时长放大到整次运行。
- **策略看到的 rank 状态由回放器记账。** 请求数是准确的；KV token 数按每个流式分片一个 token 估算。
  适配层写好后可以换成服务端的状态。

## 还没验证的

- 回放器只对假服务测过，没有连过真的 vLLM。请求体里的 `min_tokens`、`ignore_eos` 和 token id 形式的
  `prompt` 在 v0.32.0 的源码里确认存在，还要用一次真实请求确认行为。
- step 埋点补丁只做过静态检查，配套测试没跑过，也没有在 GPU 上跑过。下一步见 [docs/下一步.md](docs/下一步.md)。
- 场景的默认长度和到达率是起步值，要在真机上校准。
- AgentX 多轮会话（P3）的加载器还没写。
