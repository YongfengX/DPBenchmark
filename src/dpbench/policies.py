"""Dispatch policies (design doc section 6).

A dispatch policy picks the DP rank of each new request. It sees only online
information: the request's prompt length, session and arrival time, and the
state of each rank. It never sees the output length; `RequestView` has no such field.

Rank state here is kept by the replayer from what a client can observe
(dispatches, first tokens, streamed tokens, completions). With one client and
one API server this bookkeeping is exact for request counts and close for KV
tokens. Server-side state can replace it once the framework adapter exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import numpy as np


@dataclass
class RankState:
    rank: int
    waiting: int = 0                  # dispatched, first token not seen yet
    running: int = 0                  # decoding
    kv_tokens: int = 0                # context tokens of the running requests
    pending_prefill_tokens: int = 0   # prompt tokens of the waiting requests

    @property
    def num_reqs(self) -> int:
        return self.waiting + self.running


@dataclass(frozen=True)
class RequestView:
    """What a policy may read about a request. There is deliberately no output length."""

    input_tokens: int
    session_id: str
    arrival_s: float


class DispatchPolicy(Protocol):
    name: str

    def choose(self, req: RequestView, ranks: Sequence[RankState]) -> int: ...


def _argmin(scores: Sequence[float], start: int = 0) -> int:
    """Index of the smallest score; ties go to the first one found scanning from `start`."""
    n = len(scores)
    best = start % n
    for k in range(1, n):
        i = (start + k) % n
        if scores[i] < scores[best]:
            best = i
    return best


class RoundRobin:
    """Cycle through the ranks in arrival order. The zero point of the recovery score."""

    name = "rr"

    def __init__(self) -> None:
        self._next = 0

    def choose(self, req: RequestView, ranks: Sequence[RankState]) -> int:
        r = self._next % len(ranks)
        self._next += 1
        return r


class PowerOfTwo:
    """Sample two ranks, take the one with fewer requests."""

    name = "p2c"

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def choose(self, req: RequestView, ranks: Sequence[RankState]) -> int:
        if len(ranks) == 1:
            return 0
        a, b = (int(i) for i in self._rng.choice(len(ranks), 2, replace=False))
        return a if ranks[a].num_reqs <= ranks[b].num_reqs else b


class JoinShortestQueue:
    """Fewest running + waiting requests."""

    name = "jsq"

    def __init__(self) -> None:
        self._start = 0

    def choose(self, req: RequestView, ranks: Sequence[RankState]) -> int:
        r = _argmin([s.num_reqs for s in ranks], self._start)
        self._start += 1
        return r


class LeastTokens:
    """Fewest tokens in flight: context of running requests plus prompts still to prefill."""

    name = "least_tokens"

    def choose(self, req: RequestView, ranks: Sequence[RankState]) -> int:
        return _argmin([s.kv_tokens + s.pending_prefill_tokens for s in ranks])


class LeastKV:
    """Smallest KV footprint."""

    name = "least_kv"

    def choose(self, req: RequestView, ranks: Sequence[RankState]) -> int:
        return _argmin([s.kv_tokens for s in ranks])


class VllmDefault:
    """Client-side emulation of vLLM's internal DP load balancer: score = waiting * 4 + running,
    ties to the first engine found scanning from a rotating start index.

    The real balancer works from coordinator snapshots that can be up to ~100 ms old and counts
    "waiting" inside the engine; this emulation uses the replayer's own bookkeeping. To measure the
    real thing, use the `server` policy, which sends no rank header.
    """

    name = "vllm_default"

    def __init__(self) -> None:
        self._start = 0

    def choose(self, req: RequestView, ranks: Sequence[RankState]) -> int:
        r = _argmin([s.waiting * 4 + s.running for s in ranks], self._start)
        self._start += 1
        return r


SERVER = "server"  # no rank header: the framework's own balancer decides; rank is filled in from server traces

POLICIES: dict[str, Callable[..., DispatchPolicy]] = {
    "rr": RoundRobin,
    "p2c": PowerOfTwo,
    "jsq": JoinShortestQueue,
    "least_tokens": LeastTokens,
    "least_kv": LeastKV,
    "vllm_default": VllmDefault,
}


def make_policy(name: str, **params) -> DispatchPolicy | None:
    if name == SERVER:
        return None
    if name not in POLICIES:
        raise ValueError(f"unknown dispatch policy {name!r}; expected one of {sorted(POLICIES) + [SERVER]}")
    return POLICIES[name](**params)
