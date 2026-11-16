"""A stand-in for an OpenAI-compatible completions server, for dry runs without a GPU.

It honours the rank header, the forced output length and the streaming protocol,
and nothing else: timing is a toy model (prefill time proportional to prompt
length, a fixed time per output token plus a small cost per request active on the
same rank). It is not the simulator of step S4.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from dataclasses import dataclass, field

from aiohttp import web

from .replay import VLLM_RANK_HEADER


@dataclass
class MockState:
    seen: Counter = field(default_factory=Counter)      # requests per rank
    active: Counter = field(default_factory=Counter)    # requests currently decoding per rank
    next: int = 0                                       # round-robin cursor for requests without a rank header


STATE: web.AppKey[MockState] = web.AppKey("state", MockState)


def make_app(n_ranks: int = 4, prefill_ms_per_1k: float = 2.0, tpot_ms: float = 1.0,
             per_active_ms: float = 0.0, rank_header: str = VLLM_RANK_HEADER) -> web.Application:
    app = web.Application(client_max_size=64 * 1024 * 1024)
    state = app[STATE] = MockState()

    async def completions(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        header = request.headers.get(rank_header)
        if header is None:
            rank = state.next % n_ranks
            state.next += 1
        else:
            rank = int(header)
            if not 0 <= rank < n_ranks:
                return web.json_response({"error": f"data_parallel_rank {rank} is out of range [0, {n_ranks})"},
                                         status=400)
        prompt = body["prompt"]
        n_in = len(prompt) if isinstance(prompt, list) else len(str(prompt).split())
        n_out = int(body.get("max_tokens", 16))
        state.seen[rank] += 1

        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await asyncio.sleep(n_in / 1000.0 * prefill_ms_per_1k / 1000.0)
        state.active[rank] += 1
        try:
            for _ in range(n_out):
                await asyncio.sleep((tpot_ms + per_active_ms * state.active[rank]) / 1000.0)
                chunk = {"choices": [{"index": 0, "text": "x"}]}
                await resp.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
        finally:
            state.active[rank] -= 1
        usage = {"choices": [], "usage": {"prompt_tokens": n_in, "completion_tokens": n_out,
                                          "prompt_tokens_details": {"cached_tokens": 0}}}
        await resp.write(b"data: " + json.dumps(usage).encode() + b"\n\n")
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp

    app.router.add_post("/v1/completions", completions)
    return app


def serve(host: str = "127.0.0.1", port: int = 8000, **kwargs) -> None:
    web.run_app(make_app(**kwargs), host=host, port=port, print=None)
