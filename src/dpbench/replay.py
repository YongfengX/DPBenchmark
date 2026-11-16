"""Open-loop replayer and external router (design doc sections 5 and 6).

The replayer sends each request of a scheduled workload at its arrival time to an
OpenAI-compatible `/v1/completions` endpoint, forces the output length, and
records the client-side columns of the request table.

It is also the router: for requests that are not bound to a rank it asks a
dispatch policy, then pins the request with a header (vLLM: `X-data-parallel-rank`).
This keeps the framework unmodified.

Request body (vLLM names; verify on the pinned version with one request):
    prompt       list of token ids, so input_tokens is exact and no chat template runs
    max_tokens = min_tokens = N, ignore_eos = true      -> exactly N output tokens
    stream = true, stream_options.include_usage = true   -> first-token time and token counts
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import aiohttp
import numpy as np
import pandas as pd

from .policies import DispatchPolicy, RankState, RequestView
from .workload import Req

VLLM_RANK_HEADER = "X-data-parallel-rank"


@dataclass
class ReplayResult:
    request: pd.DataFrame             # request table (client-side columns filled)
    errors: list[str] = field(default_factory=list)
    decision_ns: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def decision_p99_ms(self) -> float:
        return float(np.quantile(self.decision_ns, 0.99) / 1e6) if len(self.decision_ns) else 0.0


class Replayer:
    def __init__(self, base_url: str, model: str, n_ranks: int, policy: DispatchPolicy | None,
                 corpus: np.ndarray | None = None, vocab_range: tuple[int, int] = (1000, 30000),
                 rank_header: str | None = VLLM_RANK_HEADER, endpoint: str = "/v1/completions",
                 max_inflight: int | None = None, read_timeout_s: float = 3600.0,
                 extra_body: dict[str, Any] | None = None, lead_s: float = 0.2) -> None:
        """
        policy        dispatch policy for unbound requests; None = send no rank header and let the server decide
        corpus        1-D array of token ids; prompts are slices of it. None = synthetic ids (dry runs only:
                      random tokens change expert routing, so scored runs need a real corpus)
        max_inflight  optional cap on concurrent requests (closed-loop limit); None = pure open loop
        """
        self.url = base_url.rstrip("/") + endpoint
        self.model = model
        self.n_ranks = n_ranks
        self.policy = policy
        self.corpus = corpus
        self.vocab_range = vocab_range
        self.rank_header = rank_header
        self.max_inflight = max_inflight
        self.read_timeout_s = read_timeout_s
        self.extra_body = extra_body or {}
        self.lead_s = lead_s
        self.states = [RankState(r) for r in range(n_ranks)]

    # ------------------------------------------------------------------ pieces

    def _prompt(self, req: Req) -> list[int]:
        if self.corpus is not None:
            ids = self.corpus[req.corpus_offset: req.corpus_offset + req.input_tokens]
            if len(ids) != req.input_tokens:
                raise ValueError(f"{req.req_id}: corpus slice is {len(ids)} tokens, need {req.input_tokens}")
            return [int(t) for t in ids]
        rng = np.random.default_rng(req.corpus_offset)
        return rng.integers(self.vocab_range[0], self.vocab_range[1], req.input_tokens).tolist()

    def _body(self, req: Req) -> dict[str, Any]:
        body = {
            "model": self.model,
            "prompt": self._prompt(req),
            "max_tokens": req.output_tokens,
            "min_tokens": req.output_tokens,
            "ignore_eos": True,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        body.update(self.extra_body)
        return body

    def _pick_rank(self, req: Req) -> tuple[int | None, int]:
        """Rank for the request and the time the decision took (ns). None = let the server decide."""
        if req.bound_rank is not None:
            if not 0 <= req.bound_rank < self.n_ranks:
                raise ValueError(f"{req.req_id}: bound to rank {req.bound_rank}, but there are {self.n_ranks} ranks")
            return req.bound_rank, 0
        if self.policy is None:
            return None, 0
        t = time.perf_counter_ns()
        rank = self.policy.choose(RequestView(req.input_tokens, req.session_id, req.arrival_s), self.states)
        dt = time.perf_counter_ns() - t
        if not 0 <= rank < self.n_ranks:
            raise ValueError(f"policy {self.policy.name} returned rank {rank} outside [0, {self.n_ranks})")
        return rank, dt

    # ------------------------------------------------------------------ one request

    async def _one(self, session: aiohttp.ClientSession, req: Req, t0: int, sem: asyncio.Semaphore | None,
                   rows: list[dict[str, Any]], errors: list[str], decisions: list[int]) -> None:
        arrival_ts = t0 + int(round(req.arrival_s * 1e9))
        delay = (arrival_ts - time.monotonic_ns()) / 1e9
        if delay > 0:
            await asyncio.sleep(delay)
        if sem is not None:
            await sem.acquire()
        state: RankState | None = None
        got_first = False
        n_chunks = 0
        try:
            rank, dt = self._pick_rank(req)
            decisions.append(dt)
            headers = {}
            if rank is not None:
                state = self.states[rank]
                state.waiting += 1
                state.pending_prefill_tokens += req.input_tokens
                if self.rank_header:
                    headers[self.rank_header] = str(rank)
            body = self._body(req)
            dispatch_ts = time.monotonic_ns()
            first_ts = last_ts = None
            usage: dict[str, Any] = {}
            async with session.post(self.url, json=body, headers=headers) as resp:
                if resp.status != 200:
                    text = (await resp.text())[:300]
                    raise RuntimeError(f"HTTP {resp.status}: {text}")
                async for raw in resp.content:
                    line = raw.strip()
                    if not line.startswith(b"data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == b"[DONE]":
                        break
                    chunk = json.loads(payload)
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    choices = chunk.get("choices") or []
                    if not choices or not (choices[0].get("text") or choices[0].get("token_ids")):
                        continue
                    now = time.monotonic_ns()
                    n_chunks += 1
                    last_ts = now
                    if not got_first:
                        got_first = True
                        first_ts = now
                        if state is not None:
                            state.waiting -= 1
                            state.running += 1
                            state.pending_prefill_tokens -= req.input_tokens
                            state.kv_tokens += req.input_tokens
                    if state is not None:
                        state.kv_tokens += 1
            if first_ts is None:
                raise RuntimeError("stream ended without any output token")
            details = usage.get("prompt_tokens_details") or {}
            rows.append({
                "req_id": req.req_id,
                "session_id": req.session_id,
                "arrival_ts": arrival_ts,
                "dispatch_ts": dispatch_ts,
                "prefill_start_ts": None,
                "rank": rank if rank is not None else -1,
                "input_tokens": int(usage.get("prompt_tokens", req.input_tokens)),
                "cached_tokens": details.get("cached_tokens"),
                "output_tokens": int(usage.get("completion_tokens", n_chunks)),
                "first_token_ts": first_ts,
                "finish_ts": last_ts,
            })
        except Exception as exc:  # noqa: BLE001 - every failure is recorded; the run is void if any occur
            errors.append(f"{req.req_id}: {type(exc).__name__}: {exc}")
        finally:
            if state is not None:
                if got_first:
                    state.running -= 1
                    state.kv_tokens -= req.input_tokens + n_chunks
                else:
                    state.waiting -= 1
                    state.pending_prefill_tokens -= req.input_tokens
            if sem is not None:
                sem.release()

    # ------------------------------------------------------------------ whole run

    async def run(self, reqs: Sequence[Req]) -> ReplayResult:
        rows: list[dict[str, Any]] = []
        errors: list[str] = []
        decisions: list[int] = []
        sem = asyncio.Semaphore(self.max_inflight) if self.max_inflight else None
        timeout = aiohttp.ClientTimeout(total=None, sock_read=self.read_timeout_s)
        connector = aiohttp.TCPConnector(limit=0)
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            t0 = time.monotonic_ns() + int(self.lead_s * 1e9)
            await asyncio.gather(*(self._one(session, r, t0, sem, rows, errors, decisions) for r in reqs))
        cols = ["req_id", "session_id", "arrival_ts", "dispatch_ts", "prefill_start_ts", "rank", "input_tokens",
                "cached_tokens", "output_tokens", "first_token_ts", "finish_ts"]
        table = pd.DataFrame(rows, columns=cols).sort_values("arrival_ts", kind="stable").reset_index(drop=True)
        return ReplayResult(table, errors, np.asarray(decisions, dtype=np.int64))


def replay(reqs: Sequence[Req], **kwargs) -> ReplayResult:
    """Synchronous wrapper: `replay(reqs, base_url=..., model=..., n_ranks=..., policy=...)`."""
    return asyncio.run(Replayer(**kwargs).run(reqs))
