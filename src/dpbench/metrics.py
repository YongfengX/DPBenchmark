"""Metrics computed from the three trace tables only (design doc section 3).

Nothing here reads framework state. Every function takes DataFrames in the
canonical schema and returns plain numbers or DataFrames.

Notation (design doc):
    X        saturated throughput, tokens / s
    BR_s     balance ratio of step s           = mean_r L / max_r L
    W        idle-wait share                   = sum of waits / (N * T_e2e)
    ALB      attention load balance            = mean_r a / max_r a per layer invocation
    X_sol    throughput if every step were perfectly balanced, derived from BR
    X_W      throughput if no rank ever waited, derived from W
    R        recovery                          = (X_pi - X_RR) / (X_mirror - X_RR)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats as _stats

NS_PER_S = 1e9
NS_PER_MS = 1e6

LOAD_KINDS = ("tokens", "kv", "reqs")


def _n_ranks(df: pd.DataFrame, n_ranks: int | None) -> int:
    return int(n_ranks) if n_ranks else int(df["rank"].max()) + 1


def _load(step: pd.DataFrame, kind: str) -> pd.Series:
    if kind == "tokens":
        return step["prefill_tokens"] + step["decode_tokens"]
    if kind == "kv":
        return step["kv_tokens"]
    if kind == "reqs":
        return step["num_reqs"]
    raise ValueError(f"unknown load kind {kind!r}; expected one of {LOAD_KINDS}")


# --------------------------------------------------------------------------- step level (L1)


def step_durations(step: pd.DataFrame) -> pd.Series:
    """Wall time of each global step: latest rank end minus earliest rank start, in ns."""
    g = step.groupby("step_id", sort=True)
    return (g["t_end"].max() - g["t_start"].min()).astype("float64").rename("tau_ns")


def balance_ratio(step: pd.DataFrame, load: str = "tokens", n_ranks: int | None = None) -> pd.DataFrame:
    """Per-step balance ratio. A rank with no row in a step counts as zero load.

    Returns one row per step_id with tau_ns, mean_load, max_load and br.
    br is NaN for steps where every rank has zero load.
    """
    n = _n_ranks(step, n_ranks)
    tmp = pd.DataFrame({"step_id": step["step_id"], "load": _load(step, load).astype("float64")})
    g = tmp.groupby("step_id", sort=True)["load"]
    out = pd.DataFrame({"tau_ns": step_durations(step), "mean_load": g.sum() / n, "max_load": g.max()})
    out["br"] = np.where(out["max_load"] > 0, out["mean_load"] / out["max_load"].where(out["max_load"] > 0, 1.0), np.nan)
    return out


def balance_summary(br: pd.DataFrame) -> dict[str, float]:
    loaded = br.dropna(subset=["br"])
    if loaded.empty:
        return {"br_mean": float("nan"), "br_time_weighted": float("nan"), "n_steps": len(br), "n_loaded_steps": 0}
    return {
        "br_mean": float(loaded["br"].mean()),
        "br_time_weighted": float(np.average(loaded["br"], weights=loaded["tau_ns"])),
        "n_steps": int(len(br)),
        "n_loaded_steps": int(len(loaded)),
    }


def sol_throughput(step: pd.DataFrame, x: float, t_e2e_ns: float, load: str = "tokens",
                   n_ranks: int | None = None) -> float:
    """Speed-of-light throughput (TensorRT-LLM): each step shrunk by its balance ratio.

    Time outside steps (idle gaps) is kept as is, so with no gaps this equals
    X * T_e2e / sum(tau_s * BR_s).
    """
    br = balance_ratio(step, load, n_ranks)
    busy = float(br["tau_ns"].sum())
    ideal_busy = float((br["tau_ns"] * br["br"].fillna(1.0)).sum())
    total = max(float(t_e2e_ns), busy)
    return x * total / (total - busy + ideal_busy)


# --------------------------------------------------------------------------- layer level (L2)


def layer_waits(layer: pd.DataFrame) -> pd.DataFrame:
    """Per layer invocation: how long ranks waited for the last rank to become ready.

    wait_sum_ns = sum_r (max_r' t_ready - t_ready_r)
    """
    g = layer.groupby(["step_id", "layer_id"], sort=True)["t_ready"]
    last = g.transform("max")
    waits = (last - layer["t_ready"]).astype("float64")
    out = waits.groupby([layer["step_id"], layer["layer_id"]], sort=True).agg(wait_sum_ns="sum", n_ranks="count")
    return out


def wait_fraction(layer: pd.DataFrame, step: pd.DataFrame, t_e2e_ns: float | None = None,
                  n_ranks: int | None = None) -> dict[str, float]:
    """Idle-wait share W.

    w_step  share of rank-time lost to waiting within the sampled steps
            = sum of waits / (N * duration of the sampled steps)
    w       the same scaled to the whole run = w_step * busy time / T_e2e.
            With every step sampled and no idle gaps this is the design doc's W.
    """
    n = _n_ranks(step, n_ranks)
    waits = layer_waits(layer)
    tau = step_durations(step)
    sampled = waits.index.get_level_values("step_id").unique()
    missing = sampled.difference(tau.index)
    if len(missing):
        raise ValueError(f"layer table references {len(missing)} step_ids that are not in the step table")
    tau_sampled = float(tau.loc[sampled].sum())
    w_step = float(waits["wait_sum_ns"].sum()) / (n * tau_sampled) if tau_sampled > 0 else float("nan")
    busy = float(tau.sum())
    total = max(float(t_e2e_ns), busy) if t_e2e_ns else busy
    wait_time = w_step * busy
    return {
        "w": wait_time / total if total > 0 else float("nan"),
        "w_step": w_step,
        "wait_time_ns": wait_time,
        "sampled_steps": int(len(sampled)),
        "total_steps": int(len(tau)),
    }


def wait_throughput(x: float, w: float) -> float:
    """Balanced-throughput estimate from W: X / (1 - W) (design doc section 8, V4)."""
    return x / (1.0 - w)


def attention_load_balance(layer: pd.DataFrame, step: pd.DataFrame) -> pd.DataFrame:
    """ALB per layer invocation: mean over ranks of the local duration divided by the max.

    The local duration of a rank in a layer is t_ready minus the time the rank
    left the previous layer (its t_combine_done), or the step start for the first layer.
    """
    l = layer.sort_values(["step_id", "rank", "t_ready"], kind="stable").copy()
    prev_end = l.groupby(["step_id", "rank"], sort=False)["t_combine_done"].shift()
    starts = l[["step_id", "rank"]].merge(step[["step_id", "rank", "t_start"]], on=["step_id", "rank"], how="left")
    prev = prev_end.astype("float64").to_numpy()
    start = np.where(np.isnan(prev), starts["t_start"].astype("float64").to_numpy(), prev)
    l["local_ns"] = np.clip(l["t_ready"].astype("float64").to_numpy() - start, 0.0, None)
    g = l.groupby(["step_id", "layer_id"], sort=True)["local_ns"]
    out = pd.DataFrame({"mean_local_ns": g.mean(), "max_local_ns": g.max()})
    out["alb"] = np.where(out["max_local_ns"] > 0, out["mean_local_ns"] / out["max_local_ns"].where(out["max_local_ns"] > 0, 1.0), np.nan)
    return out


# --------------------------------------------------------------------------- request level (L0)


def run_duration_ns(request: pd.DataFrame) -> float:
    """T_e2e: first intended arrival to last finish."""
    return float(request["finish_ts"].max() - request["arrival_ts"].min())


def total_tokens(request: pd.DataFrame, tokens: str = "computed") -> int:
    """Token count used for throughput.

    computed  generated tokens plus prompt tokens that were actually computed (cache hits excluded)
    output    generated tokens only
    input     computed prompt tokens only
    """
    cached = request["cached_tokens"].fillna(0) if "cached_tokens" in request.columns else 0
    computed_in = int((request["input_tokens"] - cached).sum())
    out = int(request["output_tokens"].sum())
    if tokens == "computed":
        return computed_in + out
    if tokens == "output":
        return out
    if tokens == "input":
        return computed_in
    raise ValueError(f"unknown token kind {tokens!r}")


def throughput(request: pd.DataFrame, tokens: str = "computed") -> float:
    """Saturated throughput X in tokens / s: all tokens of the fixed request set over the time to finish them all."""
    return total_tokens(request, tokens) / (run_duration_ns(request) / NS_PER_S)


def latency_table(request: pd.DataFrame) -> pd.DataFrame:
    """Per-request latencies in ms. tpot_ms is NaN for requests with fewer than two output tokens."""
    arrival = request["arrival_ts"].astype("float64")
    first = request["first_token_ts"].astype("float64")
    finish = request["finish_ts"].astype("float64")
    n_out = request["output_tokens"].astype("float64")
    out = pd.DataFrame({"req_id": request["req_id"].to_numpy()})
    out["ttft_ms"] = ((first - arrival) / NS_PER_MS).to_numpy()
    out["tpot_ms"] = np.where(n_out > 1, (finish - first) / NS_PER_MS / (n_out - 1).where(n_out > 1, 1.0), np.nan)
    out["e2e_ms"] = ((finish - arrival) / NS_PER_MS).to_numpy()
    if "prefill_start_ts" in request.columns and request["prefill_start_ts"].notna().any():
        out["queue_ms"] = ((request["prefill_start_ts"].astype("float64") - arrival) / NS_PER_MS).to_numpy()
    return out


def latency_summary(request: pd.DataFrame, quantiles: Sequence[float] = (0.5, 0.9, 0.99)) -> dict[str, float]:
    lat = latency_table(request)
    out: dict[str, float] = {}
    for col in ("ttft_ms", "tpot_ms", "e2e_ms", "queue_ms"):
        if col not in lat.columns:
            continue
        values = lat[col].dropna()
        for q in quantiles:
            out[f"{col[:-3]}_p{round(q * 100)}_ms"] = float(values.quantile(q)) if len(values) else float("nan")
    return out


@dataclass(frozen=True)
class SLO:
    """Latency SLO. TTFT may grow with prompt length: ttft_base_ms + ttft_per_1k_input_ms * input_tokens / 1000."""

    tpot_ms: float | None = None
    ttft_base_ms: float | None = None
    ttft_per_1k_input_ms: float = 0.0


def slo_attainment(request: pd.DataFrame, slo: SLO) -> float:
    """Share of requests that meet every specified constraint."""
    lat = latency_table(request)
    ok = np.ones(len(lat), dtype=bool)
    if slo.tpot_ms is not None:
        tpot = lat["tpot_ms"].to_numpy()
        ok &= np.isnan(tpot) | (tpot <= slo.tpot_ms)
    if slo.ttft_base_ms is not None:
        limit = slo.ttft_base_ms + slo.ttft_per_1k_input_ms * request["input_tokens"].astype("float64").to_numpy() / 1000.0
        ok &= lat["ttft_ms"].to_numpy() <= limit
    return float(ok.mean())


def max_sustainable_rate(rates: Iterable[float], attainments: Iterable[float], target: float = 0.99) -> float | None:
    """lambda*: the highest tested arrival rate up to which every tested rate meets the target.

    Returns None when even the lowest tested rate misses the target.
    """
    pairs = sorted(zip(rates, attainments))
    best: float | None = None
    for rate, att in pairs:
        if att < target:
            break
        best = float(rate)
    return best


def tpot_guard(request_policy: pd.DataFrame, request_rr: pd.DataFrame, factor: float = 1.1) -> dict[str, float | bool]:
    """Fast-tier latency constraint: P99 TPOT must not exceed factor x the round-robin run's."""
    p = latency_summary(request_policy, (0.99,)).get("tpot_p99_ms", float("nan"))
    b = latency_summary(request_rr, (0.99,)).get("tpot_p99_ms", float("nan"))
    ratio = p / b if b and not np.isnan(b) else float("nan")
    return {"tpot_p99_ms": p, "rr_tpot_p99_ms": b, "ratio": ratio, "ok": bool(np.isnan(ratio) or ratio <= factor)}


# --------------------------------------------------------------------------- scoring


def noise_floor(xs: Sequence[float]) -> float:
    """Coefficient of variation of throughput across repeats of one configuration (V1)."""
    a = np.asarray(xs, dtype="float64")
    if len(a) < 2:
        return float("nan")
    return float(a.std(ddof=1) / a.mean())


@dataclass(frozen=True)
class Recovery:
    r: float                 # point estimate from the means of the three groups
    ci_low: float            # bootstrap 2.5 %
    ci_high: float           # bootstrap 97.5 %
    recoverable: float       # (X_mirror - X_RR) / X_mirror
    reportable: bool         # False when the recoverable share is under 3 x the noise floor


def recovery(x_policy: Sequence[float], x_rr: Sequence[float], x_mirror: Sequence[float],
             floor: float | None = None, n_boot: int = 10_000, seed: int = 0, level: float = 0.95) -> Recovery:
    """Recovery R with a bootstrap confidence interval (design doc section 7).

    Each argument holds the throughputs of the repeats of one configuration.
    The scenario has no recoverable loss, and R is not reportable, when the mirror
    reference is not faster than round robin, or when `floor` (the noise floor, V1)
    is given and the recoverable share is below 3 x floor.
    """
    p, b, m = (np.asarray(v, dtype="float64") for v in (x_policy, x_rr, x_mirror))
    if min(len(p), len(b), len(m)) == 0:
        raise ValueError("each group needs at least one throughput sample")
    gap = m.mean() - b.mean()
    r = (p.mean() - b.mean()) / gap if gap != 0 else float("nan")
    recoverable = gap / m.mean()
    lo = hi = float("nan")
    if min(len(p), len(b), len(m)) >= 2:
        rng = np.random.default_rng(seed)
        pb = rng.choice(p, (n_boot, len(p))).mean(axis=1)
        bb = rng.choice(b, (n_boot, len(b))).mean(axis=1)
        mb = rng.choice(m, (n_boot, len(m))).mean(axis=1)
        denom = mb - bb
        with np.errstate(divide="ignore", invalid="ignore"):
            rb = np.where(denom > 0, (pb - bb) / denom, np.nan)
        rb = rb[~np.isnan(rb)]
        if len(rb):
            alpha = (1.0 - level) / 2.0
            lo, hi = (float(q) for q in np.quantile(rb, [alpha, 1.0 - alpha]))
    # No recoverable loss: the mirror reference is not faster than round robin, or the gap is inside the noise.
    reportable = recoverable > 0 and not (floor is not None and not np.isnan(floor) and recoverable < 3.0 * floor)
    return Recovery(float(r), lo, hi, float(recoverable), bool(reportable))


def rank_agreement(a: Sequence[float], b: Sequence[float]) -> dict[str, float]:
    """Spearman and Kendall rank correlation, used by V3 (dose response) and V8 (reproducibility)."""
    return {
        "spearman": float(_stats.spearmanr(a, b).statistic),
        "kendall": float(_stats.kendalltau(a, b).statistic),
    }


# --------------------------------------------------------------------------- work conservation


def check_work(request: pd.DataFrame, expected: pd.DataFrame) -> list[str]:
    """Anti-cheating rule 1: every request finished, with exactly the lengths the workload asked for.

    `expected` has req_id, input_tokens, output_tokens.
    """
    problems: list[str] = []
    got = request.set_index("req_id")
    want = expected.set_index("req_id")
    missing = want.index.difference(got.index)
    extra = got.index.difference(want.index)
    if len(missing):
        problems.append(f"{len(missing)} requests of the workload did not finish")
    if len(extra):
        problems.append(f"{len(extra)} finished requests are not in the workload")
    common = want.index.intersection(got.index)
    for col in ("input_tokens", "output_tokens"):
        bad = int((got.loc[common, col].astype("int64") != want.loc[common, col].astype("int64")).sum())
        if bad:
            problems.append(f"{bad} requests have {col} different from the workload")
    return problems


def computed_tokens(step: pd.DataFrame) -> int:
    """Total tokens the engines computed; must be equal across runs of the same request set."""
    return int((step["prefill_tokens"] + step["decode_tokens"]).sum())


def check_step_work(request: pd.DataFrame, step: pd.DataFrame) -> list[str]:
    """Work conservation between the request table and the step table, per rank.

    A request with I computed prompt tokens and O output tokens costs the engine
    I + O - 1 forward tokens: the last prompt position yields the first output
    token, and the last output token is never fed back in.
    """
    cached = request["cached_tokens"].fillna(0) if "cached_tokens" in request.columns else 0
    want = (request["input_tokens"] - cached + request["output_tokens"] - 1).astype("int64")
    got = (step["prefill_tokens"] + step["decode_tokens"]).astype("int64")
    if (request["rank"] < 0).any():  # rank unknown until the server-side trace is merged: compare totals only
        w, g = int(want.sum()), int(got.sum())
        return [] if w == g else [f"step table has {g} computed tokens, request table implies {w}"]
    want_by_rank = want.groupby(request["rank"].astype("int64")).sum()
    got_by_rank = got.groupby(step["rank"].astype("int64")).sum()
    problems = []
    for rank in sorted(set(want_by_rank.index) | set(got_by_rank.index)):
        w, g = int(want_by_rank.get(rank, 0)), int(got_by_rank.get(rank, 0))
        if w != g:
            problems.append(f"rank {rank}: step table has {g} computed tokens, request table implies {w}")
    return problems


# --------------------------------------------------------------------------- one-call summary


def summarize(request: pd.DataFrame, step: pd.DataFrame | None = None, layer: pd.DataFrame | None = None,
              n_ranks: int | None = None, tokens: str = "computed") -> dict[str, float]:
    """Everything one run can report on its own (R needs three runs; see `recovery`)."""
    t_e2e = run_duration_ns(request)
    x = throughput(request, tokens)
    out: dict[str, float] = {"x_tok_s": x, "t_e2e_s": t_e2e / NS_PER_S, "n_requests": int(len(request))}
    out.update(latency_summary(request))
    if step is not None:
        out["computed_tokens"] = computed_tokens(step)
        for kind in LOAD_KINDS:
            s = balance_summary(balance_ratio(step, kind, n_ranks))
            out[f"br_{kind}_mean"] = s["br_mean"]
            out[f"br_{kind}_time_weighted"] = s["br_time_weighted"]
            out[f"x_sol_{kind}_tok_s"] = sol_throughput(step, x, t_e2e, kind, n_ranks)
        if layer is not None:
            wf = wait_fraction(layer, step, t_e2e, n_ranks)
            out["w"] = wf["w"]
            out["w_step"] = wf["w_step"]
            out["x_w_tok_s"] = wait_throughput(x, wf["w"])
            out["alb_mean"] = float(attention_load_balance(layer, step)["alb"].mean())
            out["l2_sampled_steps"] = wf["sampled_steps"]
    return out
