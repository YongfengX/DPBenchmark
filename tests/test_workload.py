from collections import Counter

import numpy as np
import pandas as pd
import pytest

from dpbench import workload as w


def make(mode, seed=3, n=50, copies=4, **kw):
    base = w.scenario("D2", n, rate=10.0, seed=seed)
    return w.schedule(w.replicate(base, copies, corpus_len=1_000_000, seed=seed), mode, seed=seed, **kw)


def test_same_seed_same_set():
    assert w.fingerprint(make("test")) == w.fingerprint(make("test"))
    assert w.fingerprint(make("test")) != w.fingerprint(make("test", seed=4))
    assert w.fingerprint(make("test")) != w.fingerprint(make("mirror"))


def test_copies_share_lengths_and_differ_in_content():
    reqs = make("mirror")
    by_idx = {}
    for r in reqs:
        by_idx.setdefault(r.idx, []).append(r)
    for group in by_idx.values():
        assert len(group) == 4
        assert len({(r.arrival_s, r.input_tokens, r.output_tokens) for r in group}) == 1
        assert len({r.corpus_offset for r in group}) == 4
        assert all(r.corpus_offset + r.input_tokens <= 1_000_000 for r in group)


def test_mirror_binds_copy_to_rank():
    reqs = make("mirror")
    assert all(r.bound_rank == r.copy for r in reqs)
    assert Counter(r.bound_rank for r in reqs) == {0: 50, 1: 50, 2: 50, 3: 50}


def test_lag_shifts_each_rank():
    mirror = {r.req_id: r for r in make("mirror")}
    for r in make("lag", lag_s=5.0):
        assert r.bound_rank == r.copy
        assert r.arrival_s == pytest.approx(mirror[r.req_id].arrival_s + 5.0 * r.copy)


def test_all_modes_carry_the_same_work():
    def work(reqs):
        return sorted((r.req_id, r.input_tokens, r.output_tokens, r.corpus_offset) for r in reqs)

    assert work(make("mirror")) == work(make("lag", lag_s=5.0)) == work(make("test"))


def test_test_mode_is_unbound_and_jittered():
    reqs = make("test", jitter_s=0.5)
    mirror = {r.req_id: r for r in make("mirror")}
    assert all(r.bound_rank is None for r in reqs)
    shifts = [r.arrival_s - mirror[r.req_id].arrival_s for r in reqs]
    assert min(shifts) >= 0 and max(shifts) <= 0.5 and len(set(shifts)) > 100
    assert [r.arrival_s for r in reqs] == sorted(r.arrival_s for r in reqs)


def test_time_scale_and_rate():
    fast = make("mirror", time_scale=0.1)
    slow = make("mirror")
    assert max(r.arrival_s for r in fast) == pytest.approx(0.1 * max(r.arrival_s for r in slow))
    rated = w.at_rate(slow, rate=100.0)
    assert max(r.arrival_s for r in rated) == pytest.approx(len(slow) / 100.0)


def test_scenarios_have_the_intended_shape():
    d1 = w.scenario("D1", 100, 10.0, 0)
    assert {(r.input_tokens, r.output_tokens) for r in d1} == {(1024, 256)}
    d2 = [r.output_tokens for r in w.scenario("D2", 5000, 10.0, 0)]
    assert min(d2) >= 16 and max(d2) <= 4096 and np.mean(d2) > np.median(d2)  # long tail
    d3 = w.scenario("D3", 5000, 10.0, 0, long_frac=0.05)
    assert 0.03 < np.mean([r.input_tokens == 32768 for r in d3]) < 0.07
    assert {r.output_tokens for r in w.scenario("P2", 100, 10.0, 0)} == {1}
    gaps = lambda rs: np.diff([r.arrival_s for r in rs])
    cv = lambda g: g.std() / g.mean()
    assert cv(gaps(w.scenario("D4", 5000, 10.0, 0))) > 2 * cv(gaps(w.scenario("D2", 5000, 10.0, 0)))  # bursty
    with pytest.raises(ValueError):
        w.scenario("Z9", 10, 1.0, 0)


def test_corpus_must_be_long_enough():
    base = w.scenario("D1", 5, 10.0, 0)
    with pytest.raises(ValueError):
        w.replicate(base, 2, corpus_len=100, seed=0)


def test_save_load_roundtrip(tmp_path):
    reqs = make("lag", lag_s=2.0)
    w.save(reqs, tmp_path / "wl.jsonl")
    assert w.load(tmp_path / "wl.jsonl") == reqs
    exp = w.expected_frame(reqs)
    assert len(exp) == 200 and exp["output_tokens"].sum() == sum(r.output_tokens for r in reqs)


def test_load_azure(tmp_path):
    p = tmp_path / "azure.csv"
    pd.DataFrame({
        "TIMESTAMP": ["2024-05-10 00:00:01.500000+00:00", "2024-05-10 00:00:00.000000+00:00",
                      "2024-05-10 00:00:03.000000+00:00", "2024-05-10 00:00:04.000000+00:00"],
        "ContextTokens": [300, 100, 50000, 10],
        "GeneratedTokens": [30, 10, 500, 0],
    }).to_csv(p, index=False)
    base = w.load_azure(p, max_input=32768)
    assert [(b.arrival_s, b.input_tokens, b.output_tokens) for b in base] == [(0.0, 100, 10), (1.5, 300, 30), (3.0, 32768, 500)]


def test_load_burstgpt(tmp_path):
    p = tmp_path / "burst.csv"
    pd.DataFrame({
        "Timestamp": [5, 45, 60], "Model": ["ChatGPT"] * 3, "Request tokens": [472, 1087, 100],
        "Response tokens": [18, 0, 7], "Total tokens": [490, 1087, 107], "Log Type": ["Conversation log"] * 3,
        "Session ID": ["s1", "s1", "s2"],
    }).to_csv(p, index=False)
    base = w.load_burstgpt(p)
    assert [(b.arrival_s, b.input_tokens, b.output_tokens, b.session_id) for b in base] == [(0.0, 472, 18, "s1"), (55.0, 100, 7, "s2")]
    copies = w.replicate(base, 2, None, 0)
    assert {r.session_id for r in copies} == {"c0-s1", "c0-s2", "c1-s1", "c1-s2"}
