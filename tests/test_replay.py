import asyncio
import json
import socket
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from aiohttp import web

from dpbench import io, metrics, policies, schema, workload as w
from dpbench.cli import main as cli
from dpbench.mock_server import STATE, make_app
from dpbench.replay import Replayer

ROOT = Path(__file__).parent.parent


def small(mode, copies=4, n=12, **kw):
    base = w.scenario("D1", n, rate=200.0, seed=1, input=64, output=8)
    return w.schedule(w.replicate(base, copies, None, seed=1), mode, seed=1, **kw)


def run_against_mock(reqs, policy, n_ranks=4, **replayer_kw):
    async def go():
        app = make_app(n_ranks=n_ranks, prefill_ms_per_1k=1.0, tpot_ms=1.0)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            rep = Replayer(f"http://127.0.0.1:{port}", "mock", n_ranks, policy, lead_s=0.05, **replayer_kw)
            return await rep.run(reqs), app[STATE].seen, rep
        finally:
            await runner.cleanup()

    return asyncio.run(go())


def test_mirror_pins_each_copy_to_its_rank():
    reqs = small("mirror")
    result, seen, rep = run_against_mock(reqs, None)
    assert result.errors == []
    assert dict(seen) == {0: 12, 1: 12, 2: 12, 3: 12}
    table = result.request.set_index("req_id")
    assert all(table.loc[r.req_id, "rank"] == r.bound_rank for r in reqs)
    assert schema.validate_table(schema.coerce(result.request, "request"), "request") == []
    assert metrics.check_work(result.request, w.expected_frame(reqs)) == []


def test_output_length_is_forced_and_timestamps_are_ordered():
    result, _, _ = run_against_mock(small("test"), policies.RoundRobin())
    t = result.request
    assert (t["output_tokens"] == 8).all() and (t["input_tokens"] == 64).all()
    assert (t["cached_tokens"] == 0).all()
    assert (t["arrival_ts"] <= t["dispatch_ts"]).all()
    assert (t["dispatch_ts"] < t["first_token_ts"]).all() and (t["first_token_ts"] < t["finish_ts"]).all()
    # 8 tokens at >= 1 ms each: decoding takes at least 7 ms after the first token
    assert ((t["finish_ts"] - t["first_token_ts"]) >= 7_000_000).all()


def test_arrivals_follow_the_schedule():
    reqs = small("test", n=20)
    result, _, _ = run_against_mock(reqs, policies.RoundRobin())
    t = result.request.set_index("req_id")
    t0 = t["arrival_ts"].min() - round(min(r.arrival_s for r in reqs) * 1e9)
    for r in reqs:
        assert t.loc[r.req_id, "arrival_ts"] - t0 == pytest.approx(r.arrival_s * 1e9, abs=2)
    late = (t["dispatch_ts"] - t["arrival_ts"]).max() / 1e6
    assert late < 50, f"requests left up to {late:.1f} ms after their arrival time"


def test_policies_route_and_bookkeeping_returns_to_zero():
    for name in sorted(policies.POLICIES):
        result, seen, rep = run_against_mock(small("test"), policies.make_policy(name))
        assert result.errors == [], name
        assert sum(seen.values()) == 48
        assert Counter(result.request["rank"].tolist()) == seen
        assert all(s.num_reqs == 0 and s.kv_tokens == 0 and s.pending_prefill_tokens == 0 for s in rep.states), name
    result, seen, _ = run_against_mock(small("test"), policies.RoundRobin())
    assert dict(seen) == {0: 12, 1: 12, 2: 12, 3: 12}


def test_server_policy_sends_no_header():
    result, seen, _ = run_against_mock(small("test"), policies.make_policy("server"))
    assert result.errors == [] and sum(seen.values()) == 48
    assert (result.request["rank"] == -1).all()


def test_policy_never_sees_output_length():
    assert not hasattr(policies.RequestView(1, "s", 0.0), "output_tokens")

    class Peek:
        name = "peek"

        def choose(self, req, ranks):
            assert set(vars(req)) == {"input_tokens", "session_id", "arrival_s"}
            return 0

    result, seen, _ = run_against_mock(small("test"), Peek())
    assert result.errors == [] and dict(seen) == {0: 48}


def test_failures_void_the_run():
    # Requests bound to ranks 2 and 3 cannot run on a 2-rank server: they are recorded as errors, not dropped silently.
    reqs = small("mirror", copies=4)
    result, seen, _ = run_against_mock(reqs, None, n_ranks=2)
    assert len(result.errors) == 24 and len(result.request) == 24 and dict(seen) == {0: 12, 1: 12}
    assert any("did not finish" in p for p in metrics.check_work(result.request, w.expected_frame(reqs)))

    class Bad:
        name = "bad"

        def choose(self, req, ranks):
            return 99

    result, _, _ = run_against_mock(small("test"), Bad())
    assert len(result.errors) == 48 and len(result.request) == 0


def test_corpus_slices_become_prompts():
    corpus = np.arange(10_000, dtype=np.int64)
    base = w.scenario("D1", 3, 100.0, 0, input=16, output=2)
    reqs = w.schedule(w.replicate(base, 2, len(corpus), 0), "mirror")
    rep = Replayer("http://x", "m", 2, None, corpus=corpus)
    for r in reqs:
        assert rep._prompt(r) == list(range(r.corpus_offset, r.corpus_offset + 16))
    a, b = (r for r in reqs if r.idx == 0)
    assert rep._prompt(a) != rep._prompt(b)
    body = rep._body(a)
    assert body["max_tokens"] == body["min_tokens"] == 2 and body["ignore_eos"] is True and body["stream"] is True


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_cli_end_to_end(tmp_path, capsys):
    """gen -> replay (mirror, rr, jsq) against the mock server -> validate -> metrics -> recovery."""
    port = free_port()
    server = subprocess.Popen([sys.executable, "-m", "dpbench.cli", "mock-server", "--port", str(port), "--ranks", "4",
                               "--per-active-ms", "0.2"], env={"PYTHONPATH": str(ROOT / "src"), "PATH": ""})
    try:
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.1).close()
                break
            except OSError:
                time.sleep(0.05)
        template = tmp_path / "manifest.json"
        m = json.loads((ROOT / "examples" / "manifest.4xL40S.vllm-0.32.json").read_text())
        m["framework"] = {"name": "mock", "version": "0"}
        template.write_text(json.dumps(m))
        common = ["--scenario", "D1", "--n", "10", "--rate", "300", "--seed", "5", "--copies", "4"]
        assert cli(["gen", *common, "--mode", "mirror", "--out", str(tmp_path / "mirror.jsonl")]) == 0
        assert cli(["gen", *common, "--mode", "test", "--out", str(tmp_path / "test.jsonl")]) == 0
        runs = {}
        for name, wl, policy in (("mirror", "mirror", "rr"), ("rr", "test", "rr"), ("jsq", "test", "jsq")):
            for rep in range(2):
                out = tmp_path / f"{name}-{rep}"
                rc = cli(["replay", "--workload", str(tmp_path / f"{wl}.jsonl"), "--manifest", str(template),
                          "--base-url", f"http://127.0.0.1:{port}", "--policy", policy, "--out", str(out),
                          "--decision-budget-ms", "50"])
                assert rc == 0
                assert cli(["validate", str(out)]) == 0
                runs.setdefault(name, []).append(str(out))
        assert io.load_run(runs["mirror"][0]).manifest["policy"]["dispatch"] == "pinned"
        assert io.load_run(runs["jsq"][0]).manifest["replay"]["mode"] == "test"
        capsys.readouterr()
        assert cli(["metrics", runs["rr"][0]]) == 0
        summary = json.loads(capsys.readouterr().out)
        assert summary["n_requests"] == 40 and summary["x_tok_s"] > 0
        assert cli(["recovery", "--policy", *runs["jsq"], "--rr", *runs["rr"], "--mirror", *runs["mirror"]]) == 0
        rec = json.loads(capsys.readouterr().out)
        assert rec["config_freeze_violations"] == []
        assert len(rec["x_policy"]) == 2 and "tpot_guard" in rec
        assert cli(["recovery", "--policy", runs["jsq"][0], "--rr", runs["rr"][0], "--mirror", runs["mirror"][0]]) == 0
        single = json.loads(capsys.readouterr().out)      # one repeat each: NaN interval must still be valid JSON
        assert single["noise_floor"] is None
    finally:
        server.terminate()
        server.wait(timeout=5)
