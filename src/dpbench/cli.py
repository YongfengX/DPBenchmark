"""Command line: dpbench gen | replay | merge-steps | metrics | recovery | validate | mock-server."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

from . import io, metrics, policies, schema, workload


def _dump(obj) -> str:
    """JSON with NaN written as null, so the output is valid JSON."""

    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (float, np.floating)):
            return None if np.isnan(x) else float(x)
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.bool_):
            return bool(x)
        return x

    return json.dumps(clean(obj), indent=2, allow_nan=False)


def _meta_path(path: str) -> Path:
    return Path(path).with_suffix(".meta.json")


# --------------------------------------------------------------------------- gen


def cmd_gen(a: argparse.Namespace) -> int:
    if a.azure:
        base, source = workload.load_azure(a.azure, a.start, a.n, a.max_input, a.max_output), f"azure:{a.azure}"
    elif a.burstgpt:
        base, source = workload.load_burstgpt(a.burstgpt, a.start, a.n, a.max_input, a.max_output), f"burstgpt:{a.burstgpt}"
    else:
        if a.n is None or a.rate is None:
            print("gen: --n and --rate are required for a synthetic scenario", file=sys.stderr)
            return 2
        base, source = workload.scenario(a.scenario, a.n, a.rate, a.seed), "synthetic"
    corpus_len = int(np.load(a.corpus, mmap_mode="r").shape[0]) if a.corpus else None
    copies = workload.replicate(base, a.copies, corpus_len, a.seed)
    reqs = workload.schedule(copies, a.mode, a.lag_s, a.jitter_s, a.seed, a.time_scale)
    if a.at_rate:
        reqs = workload.at_rate(reqs, a.at_rate)
    workload.save(reqs, a.out)
    meta = {
        "scenario": a.scenario, "source": source, "seed": a.seed, "n_requests": len(reqs), "n_copies": a.copies,
        "corpus": a.corpus, "fingerprint": workload.fingerprint(reqs),
        "replay": {"mode": a.mode, "lag_s": a.lag_s, "jitter_s": a.jitter_s, "time_scale": a.time_scale,
                   "tier": "full" if a.at_rate else "fast", "arrival_rate": a.at_rate},
    }
    _meta_path(a.out).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    span = max(r.arrival_s for r in reqs)
    print(f"{len(reqs)} requests over {span:.1f} s -> {a.out}\nfingerprint {meta['fingerprint']}")
    return 0


# --------------------------------------------------------------------------- replay


def cmd_replay(a: argparse.Namespace) -> int:
    from .replay import replay  # aiohttp is only needed here

    reqs = workload.load(a.workload)
    meta = json.loads(_meta_path(a.workload).read_text(encoding="utf-8"))
    manifest = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    n_ranks = int(manifest["parallel"]["dp_size"])
    bound = {r.bound_rank for r in reqs}
    if None in bound and len(bound) > 1:
        print("replay: workload mixes bound and unbound requests", file=sys.stderr)
        return 2
    policy_name = a.policy if None in bound else "pinned"
    policy = policies.make_policy(a.policy) if None in bound else None
    corpus = np.load(meta["corpus"], mmap_mode="r") if meta.get("corpus") else None

    result = replay(reqs, base_url=a.base_url, model=manifest["model"]["name"], n_ranks=n_ranks, policy=policy,
                    corpus=corpus, rank_header=a.rank_header or None, max_inflight=a.max_inflight)

    manifest.update({
        "schema_version": schema.SCHEMA_VERSION,
        "run_id": a.run_id or Path(a.out).name,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "workload": {k: meta[k] for k in ("scenario", "source", "seed", "fingerprint", "n_requests", "n_copies", "corpus")},
        "replay": meta["replay"],
        "policy": {"dispatch": policy_name, "admission": "immediate"},
    })
    manifest.setdefault("trace", {"level": "L0", "layer_sample_every": None, "clock": "monotonic_ns"})
    io.save_run(io.Run(manifest, result.request), a.out)

    problems = list(result.errors)
    problems += metrics.check_work(result.request, workload.expected_frame(reqs))
    if policy_name == policies.SERVER:
        print("note: rank is -1 until the server-side trace is merged in")
    print(f"{len(result.request)}/{len(reqs)} requests finished -> {a.out}")
    print(f"dispatch decision P99: {result.decision_p99_ms:.3f} ms (budget {a.decision_budget_ms} ms)")
    if result.decision_p99_ms > a.decision_budget_ms:
        problems.append(f"dispatch decision P99 {result.decision_p99_ms:.3f} ms exceeds the budget")
    for p in problems[:20]:
        print(f"VOID: {p}", file=sys.stderr)
    return 1 if problems else 0


# --------------------------------------------------------------------------- metrics / validate / recovery


def cmd_metrics(a: argparse.Namespace) -> int:
    run = io.load_run(a.run_dir)
    out = metrics.summarize(run.request, run.step, run.layer, run.n_ranks, a.tokens)
    print(_dump(out))
    return 0


def cmd_validate(a: argparse.Namespace) -> int:
    problems = io.validate_run(io.load_run(a.run_dir))
    for p in problems:
        print(p)
    print("ok" if not problems else f"{len(problems)} problems")
    return 1 if problems else 0


def cmd_recovery(a: argparse.Namespace) -> int:
    groups = {name: [io.load_run(d) for d in dirs] for name, dirs in (("policy", a.policy), ("rr", a.rr), ("mirror", a.mirror))}
    frozen = []
    ref = groups["rr"][0].manifest
    for name, runs in groups.items():
        for run in runs:
            frozen += [f"{name}/{run.manifest.get('run_id')}: {d}" for d in schema.manifest_diff(ref, run.manifest)]
    xs = {name: [metrics.throughput(r.request, a.tokens) for r in runs] for name, runs in groups.items()}
    floor = a.floor if a.floor is not None else metrics.noise_floor(xs["rr"])
    rec = metrics.recovery(xs["policy"], xs["rr"], xs["mirror"], floor)
    guard = metrics.tpot_guard(groups["policy"][0].request, groups["rr"][0].request, a.tpot_factor)
    out = {
        "x_policy": xs["policy"], "x_rr": xs["rr"], "x_mirror": xs["mirror"], "noise_floor": floor,
        "recoverable": rec.recoverable, "reportable": rec.reportable,
        "r": rec.r if rec.reportable else None, "ci95": [rec.ci_low, rec.ci_high] if rec.reportable else None,
        "tpot_guard": guard, "violated": not guard["ok"], "config_freeze_violations": frozen,
    }
    print(_dump(out))
    if not rec.reportable:
        print("no recoverable loss in this scenario: R is not reported", file=sys.stderr)
    return 1 if frozen else 0


def cmd_merge_steps(a: argparse.Namespace) -> int:
    from . import vllm_step_trace as vst  # the only command that reads an engine-side trace

    run_dir = Path(a.run_dir)
    run = io.load_run(run_dir)
    trace = vst.load_trace_dir(a.trace_dir)
    step = vst.window(trace, int(run.request["dispatch_ts"].min()), int(run.request["finish_ts"].max()))
    problems = vst.check_alignment(step, run.n_ranks)
    if len(step):
        run.step = step
        trace_meta = run.manifest.setdefault("trace", {"clock": "monotonic_ns"})
        if trace_meta.get("level", "L0") == "L0":
            trace_meta["level"] = "L1"
        io.save_run(run, run_dir, fmt=io._find(run_dir, "request").suffix.lstrip("."))
        problems += metrics.check_step_work(run.request, step)
    print(f"{step['step_id'].nunique() if len(step) else 0} steps x {run.n_ranks} ranks -> {run_dir}")
    print(f"steps that are dummy on every rank: {vst.all_dummy_share(step):.1%}")
    for p in problems[:20]:
        print(f"VOID: {p}", file=sys.stderr)
    return 1 if problems else 0


def cmd_mock(a: argparse.Namespace) -> int:
    from .mock_server import serve

    print(f"mock completions server on http://{a.host}:{a.port} with {a.ranks} ranks")
    serve(a.host, a.port, n_ranks=a.ranks, prefill_ms_per_1k=a.prefill_ms_per_1k, tpot_ms=a.tpot_ms,
          per_active_ms=a.per_active_ms)
    return 0


# --------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dpbench", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="generate a scheduled workload")
    g.add_argument("--scenario", default="D2", help="D1-D4, P1-P2, M1-M2; a label only when a trace file is given")
    g.add_argument("--azure", help="Azure LLM inference trace 2024 CSV")
    g.add_argument("--burstgpt", help="BurstGPT CSV")
    g.add_argument("--n", type=int, help="requests in the base stream")
    g.add_argument("--rate", type=float, help="base-stream arrival rate, requests / s (synthetic scenarios)")
    g.add_argument("--start", type=int, default=0, help="first trace row to use")
    g.add_argument("--max-input", type=int)
    g.add_argument("--max-output", type=int)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--copies", type=int, required=True, help="number of copies = DP size")
    g.add_argument("--mode", choices=workload.MODES, required=True)
    g.add_argument("--lag-s", type=float, default=0.0)
    g.add_argument("--jitter-s", type=float)
    g.add_argument("--time-scale", type=float, default=1.0)
    g.add_argument("--at-rate", type=float, help="full tier: rescale the whole set to this arrival rate")
    g.add_argument("--corpus", help=".npy file of token ids to cut prompts from")
    g.add_argument("--out", required=True)
    g.set_defaults(fn=cmd_gen)

    r = sub.add_parser("replay", help="replay a workload against a server and write a run directory")
    r.add_argument("--workload", required=True)
    r.add_argument("--manifest", required=True, help="manifest template: hardware, model, framework, parallel, engine")
    r.add_argument("--base-url", default="http://127.0.0.1:8000")
    r.add_argument("--policy", default="rr", help=f"{sorted(policies.POLICIES)} or '{policies.SERVER}'")
    r.add_argument("--rank-header", default="X-data-parallel-rank")
    r.add_argument("--max-inflight", type=int)
    r.add_argument("--decision-budget-ms", type=float, default=3.0)
    r.add_argument("--run-id")
    r.add_argument("--out", required=True)
    r.set_defaults(fn=cmd_replay)

    m = sub.add_parser("metrics", help="metrics of one run")
    m.add_argument("run_dir")
    m.add_argument("--tokens", choices=("computed", "output", "input"), default="computed")
    m.set_defaults(fn=cmd_metrics)

    v = sub.add_parser("validate", help="check a run directory against the schemas")
    v.add_argument("run_dir")
    v.set_defaults(fn=cmd_validate)

    c = sub.add_parser("recovery", help="recovery R of a policy from repeats of three configurations")
    c.add_argument("--policy", nargs="+", required=True, help="run directories of the policy under test")
    c.add_argument("--rr", nargs="+", required=True, help="run directories of the round-robin baseline")
    c.add_argument("--mirror", nargs="+", required=True, help="run directories of the mirror reference")
    c.add_argument("--floor", type=float, help="noise floor (V1); default: CV of the round-robin repeats")
    c.add_argument("--tpot-factor", type=float, default=1.1)
    c.add_argument("--tokens", choices=("computed", "output", "input"), default="computed")
    c.set_defaults(fn=cmd_recovery)

    ms = sub.add_parser("merge-steps", help="cut a run's steps out of a vLLM step trace and add them as the step table")
    ms.add_argument("run_dir")
    ms.add_argument("--trace-dir", required=True, help="the server's DPBENCH_STEP_TRACE_DIR")
    ms.set_defaults(fn=cmd_merge_steps)

    s = sub.add_parser("mock-server", help="toy completions server for dry runs")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--ranks", type=int, default=4)
    s.add_argument("--prefill-ms-per-1k", type=float, default=2.0)
    s.add_argument("--tpot-ms", type=float, default=1.0)
    s.add_argument("--per-active-ms", type=float, default=0.0)
    s.set_defaults(fn=cmd_mock)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
