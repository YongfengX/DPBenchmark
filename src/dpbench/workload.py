"""Workload generation (design doc section 5).

Pipeline:

    base stream w          a list of (arrival, input_tokens, output_tokens)
      -> replicate()       N copies: same arrivals and lengths, different text
      -> schedule()        mirror | lag | test   -> the request set that is replayed

Three hard rules of the design are enforced here:
  * output length is part of the workload (the replayer forces it),
  * a run replays a fixed request set (throughput = tokens / time to finish all),
  * the copies differ in content only, never in length.

Everything is deterministic in the seed; `fingerprint()` proves it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

MODES = ("mirror", "lag", "test")


@dataclass(frozen=True)
class BaseReq:
    idx: int
    arrival_s: float
    input_tokens: int
    output_tokens: int
    session_id: str | None = None


@dataclass(frozen=True)
class Req:
    req_id: str
    session_id: str
    copy: int                 # which of the N copies of the base stream
    idx: int                  # position in the base stream
    arrival_s: float          # seconds from the start of the replay
    input_tokens: int
    output_tokens: int
    corpus_offset: int        # where in the token corpus the prompt text starts
    bound_rank: int | None    # fixed rank for mirror / lag runs, None when a policy decides


# --------------------------------------------------------------------------- arrivals


def poisson_arrivals(n: int, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Arrival times of a Poisson process with `rate` requests / s."""
    return np.cumsum(rng.exponential(1.0 / rate, n))


def gamma_arrivals(n: int, rate: float, cv: float, rng: np.random.Generator) -> np.ndarray:
    """Bursty arrivals: gamma inter-arrival times with the given coefficient of variation (cv > 1 is bursty)."""
    shape = 1.0 / (cv * cv)
    return np.cumsum(rng.gamma(shape, 1.0 / (rate * shape), n))


def uniform_arrivals(n: int, rate: float) -> np.ndarray:
    return (np.arange(n) + 1) / rate


# --------------------------------------------------------------------------- lengths


def lognormal_lengths(n: int, median: float, sigma: float, lo: int, hi: int, rng: np.random.Generator) -> np.ndarray:
    """Long-tailed lengths: lognormal with the given median, clipped to [lo, hi]."""
    x = rng.lognormal(np.log(median), sigma, n)
    return np.clip(np.rint(x), lo, hi).astype(np.int64)


def _build(arrivals: np.ndarray, inputs: np.ndarray, outputs: np.ndarray) -> list[BaseReq]:
    return [BaseReq(i, float(a), int(x), int(y)) for i, (a, x, y) in enumerate(zip(arrivals, inputs, outputs))]


# --------------------------------------------------------------------------- scenarios

# Default shapes of the scenario matrix. The numbers are starting points to be
# calibrated on the real machine (design doc section 10); override with **params.
SCENARIO_DEFAULTS: dict[str, dict] = {
    "D1": dict(input=1024, output=256),
    "D2": dict(input=1024, output_median=256, output_sigma=1.0, output_lo=16, output_hi=4096),
    "D3": dict(short_input=512, long_input=32768, long_frac=0.03, output=256),
    "D4": dict(input=1024, output_median=256, output_sigma=1.0, output_lo=16, output_hi=4096, burst_cv=4.0),
    "P1": dict(input=4096, output=1),
    "P2": dict(input_median=2048, input_sigma=1.0, input_lo=64, input_hi=32768, output=1),
    "M1": dict(input=256, output_median=1024, output_sigma=0.8, output_lo=64, output_hi=8192),
    "M2": dict(input_median=8192, input_sigma=0.8, input_lo=512, input_hi=32768, output=64),
}


def scenario(name: str, n: int, rate: float, seed: int, **params) -> list[BaseReq]:
    """Base stream for one scenario of the matrix (D1-D4, P1-P2, M1-M2).

    P3 (AgentX sessions) and M3 (Azure arrivals) come from trace loaders instead.
    """
    if name not in SCENARIO_DEFAULTS:
        raise ValueError(f"unknown scenario {name!r}; expected one of {sorted(SCENARIO_DEFAULTS)}")
    p = {**SCENARIO_DEFAULTS[name], **params}
    rng = np.random.default_rng([seed, sum(name.encode())])
    arrivals = gamma_arrivals(n, rate, p["burst_cv"], rng) if "burst_cv" in p else poisson_arrivals(n, rate, rng)

    if "input" in p:
        inputs = np.full(n, p["input"], dtype=np.int64)
    elif "input_median" in p:
        inputs = lognormal_lengths(n, p["input_median"], p["input_sigma"], p["input_lo"], p["input_hi"], rng)
    else:  # D3: short requests with a small share of long-context ones
        is_long = rng.random(n) < p["long_frac"]
        inputs = np.where(is_long, p["long_input"], p["short_input"]).astype(np.int64)

    if "output" in p:
        outputs = np.full(n, p["output"], dtype=np.int64)
    else:
        outputs = lognormal_lengths(n, p["output_median"], p["output_sigma"], p["output_lo"], p["output_hi"], rng)
    return _build(arrivals, inputs, outputs)


# --------------------------------------------------------------------------- trace loaders


def _from_frame(ts_s: pd.Series, inputs: pd.Series, outputs: pd.Series, sessions: pd.Series | None,
                start: int, n: int | None, max_input: int | None, max_output: int | None) -> list[BaseReq]:
    df = pd.DataFrame({"t": ts_s.astype("float64"), "i": inputs, "o": outputs})
    if sessions is not None:
        df["s"] = sessions.astype("string")
    df = df[(df["i"] > 0) & (df["o"] > 0)].sort_values("t", kind="stable")
    df = df.iloc[start: start + n if n is not None else None]
    if df.empty:
        raise ValueError("no usable rows in the selected trace window")
    t0 = df["t"].iloc[0]
    i = df["i"].clip(upper=max_input).astype("int64") if max_input else df["i"].astype("int64")
    o = df["o"].clip(upper=max_output).astype("int64") if max_output else df["o"].astype("int64")
    sess = df["s"].tolist() if "s" in df.columns else [None] * len(df)
    return [BaseReq(k, float(t - t0), int(x), int(y), None if s is None or pd.isna(s) else str(s))
            for k, (t, x, y, s) in enumerate(zip(df["t"], i, o, sess))]


def load_azure(path: str | Path, start: int = 0, n: int | None = None, max_input: int | None = None,
               max_output: int | None = None, cols: tuple[str, str, str] = ("TIMESTAMP", "ContextTokens", "GeneratedTokens")
               ) -> list[BaseReq]:
    """Azure LLM inference trace 2024. Columns: TIMESTAMP, ContextTokens, GeneratedTokens (no session id)."""
    df = pd.read_csv(path)
    ts = pd.to_datetime(df[cols[0]], utc=True, format="mixed")
    ts_s = (ts - ts.min()).dt.total_seconds()
    return _from_frame(ts_s, df[cols[1]], df[cols[2]], None, start, n, max_input, max_output)


def load_burstgpt(path: str | Path, start: int = 0, n: int | None = None, max_input: int | None = None,
                  max_output: int | None = None) -> list[BaseReq]:
    """BurstGPT. Columns: Timestamp (seconds), Request tokens, Response tokens; BurstGPT_3 also has Session ID.

    Failed requests (Response tokens == 0) are dropped.
    """
    df = pd.read_csv(path)
    sessions = df["Session ID"] if "Session ID" in df.columns else None
    return _from_frame(df["Timestamp"], df["Request tokens"], df["Response tokens"], sessions, start, n,
                       max_input, max_output)


# --------------------------------------------------------------------------- copies and schedules


def replicate(base: Sequence[BaseReq], n_copies: int, corpus_len: int | None, seed: int) -> list[Req]:
    """N copies of the base stream: identical arrivals and lengths, text taken from different corpus positions.

    Copies with different content keep the expert-side load of a mirror run close to
    that of the run under test. With corpus_len=None the offset is only a content seed
    (synthetic token ids; fine for dry runs, not for scored runs).
    """
    out: list[Req] = []
    space = corpus_len if corpus_len is not None else 2**31 - 1
    for c in range(n_copies):
        rng = np.random.default_rng([seed, 7919, c])
        for b in base:
            hi = space - b.input_tokens
            if hi < 0:
                raise ValueError(f"corpus has {space} tokens, request {b.idx} needs {b.input_tokens}")
            offset = int(rng.integers(0, hi + 1))
            session = f"c{c}-{b.session_id}" if b.session_id else f"c{c}-r{b.idx}"
            out.append(Req(f"c{c}-r{b.idx}", session, c, b.idx, b.arrival_s, b.input_tokens, b.output_tokens,
                           offset, None))
    return out


def schedule(reqs: Sequence[Req], mode: str, lag_s: float = 0.0, jitter_s: float | None = None, seed: int = 0,
             time_scale: float = 1.0) -> list[Req]:
    """Turn the copies into the request set of one run.

    mirror  copy c is bound to rank c; all copies arrive together. Every rank sees the same load at every step.
    lag     copy c is bound to rank c and arrives c * lag_s later. Same work, ranks out of step.
    test    no binding; arrivals are spread by a seeded jitter of up to jitter_s
            (default: the mean inter-arrival time of the base stream); a policy picks the rank.

    time_scale multiplies every arrival time; values below 1 compress the schedule to push the run into saturation.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    out: list[Req] = []
    if mode == "test":
        base_times = sorted({r.arrival_s for r in reqs if r.copy == 0})
        if jitter_s is None:
            jitter_s = base_times[-1] / max(len(base_times), 1)
        rng = np.random.default_rng([seed, 104729])
        jitter = rng.uniform(0.0, jitter_s, len(reqs)) if jitter_s > 0 else np.zeros(len(reqs))
        for r, j in zip(reqs, jitter):
            out.append(replace(r, arrival_s=(r.arrival_s + float(j)) * time_scale, bound_rank=None))
    else:
        for r in reqs:
            shift = r.copy * lag_s if mode == "lag" else 0.0
            out.append(replace(r, arrival_s=r.arrival_s * time_scale + shift, bound_rank=r.copy))
    out.sort(key=lambda r: (r.arrival_s, r.copy, r.idx))
    return out


def at_rate(reqs: Sequence[Req], rate: float) -> list[Req]:
    """Rescale arrival times so the whole set arrives at `rate` requests / s (full tier, open loop)."""
    span = max(r.arrival_s for r in reqs)
    if span <= 0:
        return list(reqs)
    scale = (len(reqs) / rate) / span
    return [replace(r, arrival_s=r.arrival_s * scale) for r in reqs]


# --------------------------------------------------------------------------- persistence


def fingerprint(reqs: Iterable[Req]) -> str:
    """sha256 of the request set. Two generations with the same seed must give the same value."""
    h = hashlib.sha256()
    for r in reqs:
        h.update(json.dumps([r.req_id, r.session_id, r.copy, r.idx, round(r.arrival_s, 9), r.input_tokens,
                             r.output_tokens, r.corpus_offset, r.bound_rank], separators=(",", ":")).encode())
    return h.hexdigest()


def save(reqs: Sequence[Req], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in reqs:
            f.write(json.dumps(asdict(r), separators=(",", ":")) + "\n")


def load(path: str | Path) -> list[Req]:
    with open(path, encoding="utf-8") as f:
        return [Req(**json.loads(line)) for line in f if line.strip()]


def expected_frame(reqs: Sequence[Req]) -> pd.DataFrame:
    """The lengths the workload asks for, in the shape `metrics.check_work` expects."""
    return pd.DataFrame({"req_id": [r.req_id for r in reqs],
                         "input_tokens": [r.input_tokens for r in reqs],
                         "output_tokens": [r.output_tokens for r in reqs]})
