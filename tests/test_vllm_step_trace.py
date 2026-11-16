"""The vLLM step tracer, driven the way the hook patch drives it.

Two ranks replay one small wave by hand:

    forward   rank 0                          rank 1
    1         prefill 100 of request a        dummy
    2         decode a (1 tok)                prefill 40 of request b
    3         decode a (1 tok)                decode b (1 tok)
    4         dummy                           dummy          (wave-closing step)

Request a: 100 prompt + 3 output tokens = 102 forward tokens on rank 0.
Request b:  40 prompt + 2 output tokens =  41 forward tokens on rank 1.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from dpbench import cli, io, metrics, schema
from dpbench import vllm_step_trace as vst

EXAMPLE = Path(__file__).parent.parent / "examples" / "manifest.4xL40S.vllm-0.32.json"


class Cached:
    """Stands in for vLLM's CachedRequestData: knows which running requests have no output yet."""

    def __init__(self, in_prefill=()):
        self._in_prefill = set(in_prefill)

    def is_context_phase(self, req_id):
        return req_id in self._in_prefill


def sched_out(tokens, new=(), in_prefill=()):
    return SimpleNamespace(
        num_scheduled_tokens=dict(tokens),
        scheduled_new_reqs=[SimpleNamespace(req_id=r) for r in new],
        scheduled_cached_reqs=Cached(in_prefill),
    )


def real(tracer, wave, k, tokens, computed, **kw):
    tracer.begin()
    tracer.mark_exec()
    tracer.end(sched_out(tokens, **kw), {r: SimpleNamespace(num_computed_tokens=n) for r, n in computed.items()})
    tracer.commit(wave, k)


def dummy(tracer, wave, k):
    tracer.begin()
    tracer.commit_dummy(wave, k)


@pytest.fixture
def trace_dir(tmp_path):
    t0, t1 = vst.StepTracer(tmp_path, 0), vst.StepTracer(tmp_path, 1)
    # forward 1
    real(t0, 0, 1, {"a": 100}, {"a": 100}, new=["a"])
    dummy(t1, 0, 1)
    # forward 2
    real(t0, 0, 2, {"a": 1}, {"a": 101})
    real(t1, 0, 2, {"b": 40}, {"b": 40}, new=["b"])
    # forward 3
    real(t0, 0, 3, {"a": 1}, {"a": 102})
    real(t1, 0, 3, {"b": 1}, {"b": 41})
    # forward 4
    dummy(t0, 0, 4)
    dummy(t1, 0, 4)
    t0.close()
    t1.close()
    return tmp_path


def test_rows_match_the_hand_trace(trace_dir):
    step = vst.load_trace_dir(trace_dir)
    assert schema.validate_table(step, "step") == []
    assert len(step) == 8
    r0 = step[step["rank"] == 0].reset_index(drop=True)
    r1 = step[step["rank"] == 1].reset_index(drop=True)
    assert r0["prefill_tokens"].tolist() == [100, 0, 0, 0]
    assert r0["decode_tokens"].tolist() == [0, 1, 1, 0]
    assert r0["kv_tokens"].tolist() == [100, 101, 102, 0]
    assert r0["dummy"].tolist() == [False, False, False, True]
    assert r1["prefill_tokens"].tolist() == [0, 40, 0, 0]
    assert r1["num_reqs"].tolist() == [0, 1, 1, 0]
    assert r1["dummy"].tolist() == [True, False, False, True]
    assert r0["step_id"].tolist() == r1["step_id"].tolist() == [1, 2, 3, 4]
    assert (step["t_start"] <= step["t_exec_start"]).all() and (step["t_exec_start"] <= step["t_end"]).all()


def test_a_chunked_prefill_stays_prefill_until_the_first_output_token():
    out = sched_out({"a": 512, "b": 1}, in_prefill=["a"])
    tracer = SimpleNamespace(_pending=None)
    vst.StepTracer.end(tracer, out, {"a": SimpleNamespace(num_computed_tokens=1024), "b": SimpleNamespace(num_computed_tokens=77)})
    _, prefill, decode, num_reqs, kv = tracer._pending
    assert (prefill, decode, num_reqs, kv) == (512, 1, 2, 1101)


def test_step_id_carries_the_wave():
    assert vst.make_step_id(0, 7) == 7
    assert vst.make_step_id(3, 1) == 3 * 2**32 + 1
    assert vst.make_step_id(3, 1) > vst.make_step_id(2, 10**6)


def test_a_step_that_scheduled_nothing_is_recorded_as_dummy_only(tmp_path):
    t = vst.StepTracer(tmp_path, 0)
    t.begin()
    t.end(sched_out({}), {})  # scheduler had requests but could not run any
    t.begin()
    t.commit_dummy(0, 1)  # the loop then runs a dummy forward
    t.close()
    step = vst.load_trace_dir(tmp_path)
    assert len(step) == 1 and bool(step["dummy"].iloc[0])


def test_aligned_trace_passes(trace_dir):
    step = vst.load_trace_dir(trace_dir)
    assert vst.check_alignment(step, 2) == []
    assert vst.all_dummy_share(step) == pytest.approx(0.25)


def test_alignment_problems_are_reported(trace_dir):
    step = vst.load_trace_dir(trace_dir)
    assert any("ranks present" in p for p in vst.check_alignment(step, 4))

    lost = step.drop(step[(step["rank"] == 1) & (step["fwd_seq"] == 2)].index)
    assert any("one row per rank" in p for p in vst.check_alignment(lost, 2))

    # rank 1 believes it is one wave ahead from forward 3 on
    skew = step.copy()
    late = (skew["rank"] == 1) & (skew["fwd_seq"] >= 3)
    skew.loc[late, "step_id"] = skew.loc[late, "step_id"] + 2**32
    assert any("disagree on step_id" in p for p in vst.check_alignment(skew, 2))

    # consistent ids, but rank 1's forward 2 ends before rank 0's begins
    apart = step.copy()
    hit = (apart["rank"] == 1) & (apart["fwd_seq"] == 2)
    apart.loc[hit, ["t_start", "t_end"]] = [0, 1]
    assert any("do not overlap" in p for p in vst.check_alignment(apart, 2))


def test_window_keeps_whole_steps(trace_dir):
    step = vst.load_trace_dir(trace_dir)
    r0 = step[step["rank"] == 0].set_index("fwd_seq")
    cut = vst.window(step, int(r0.loc[2, "t_start"]), int(r0.loc[3, "t_end"]))
    assert set(cut["fwd_seq"]) >= {2, 3}
    assert (cut.groupby("step_id").size() == 2).all()
    assert vst.window(step, int(step["t_end"].max()) + 10**12, int(step["t_end"].max()) + 2 * 10**12).empty


def test_one_directory_holds_one_server_launch(trace_dir):
    again = vst.StepTracer(trace_dir, 0)
    dummy(again, 0, 1)
    again.close()
    with pytest.raises(ValueError, match="more than one server launch"):
        vst.load_trace_dir(trace_dir)


def requests_for(step):
    """Client-side rows for requests a and b, timed inside the traced steps."""
    lo, hi = int(step["t_start"].min()), int(step["t_end"].max())
    return pd.DataFrame({
        "req_id": ["a", "b"], "session_id": ["s", "s"], "arrival_ts": [lo, lo], "dispatch_ts": [lo, lo],
        "rank": [0, 1], "input_tokens": [100, 40], "cached_tokens": [0, 0], "output_tokens": [3, 2],
        "first_token_ts": [hi, hi], "finish_ts": [hi, hi],
    })


def test_step_work_matches_the_request_table(trace_dir):
    step = vst.load_trace_dir(trace_dir)
    req = requests_for(step)
    assert metrics.check_step_work(req, step) == []
    short = req.assign(output_tokens=[3, 5])
    assert metrics.check_step_work(short, step) == ["rank 1: step table has 41 computed tokens, request table implies 44"]
    unknown_rank = req.assign(rank=-1)
    assert metrics.check_step_work(unknown_rank, step) == []


def test_merge_steps_command(trace_dir, tmp_path_factory, capsys):
    step = vst.load_trace_dir(trace_dir)
    manifest = json.loads(EXAMPLE.read_text())
    manifest["parallel"].update(dp_size=2, ep_size=2)
    run_dir = tmp_path_factory.mktemp("run")
    io.save_run(io.Run(manifest, schema.coerce(requests_for(step), "request")), run_dir)

    assert cli.main(["merge-steps", str(run_dir), "--trace-dir", str(trace_dir)]) == 0
    run = io.load_run(run_dir)
    assert run.manifest["trace"]["level"] == "L1"
    assert len(run.step) == 8
    assert io.validate_run(run) == []
    assert "4 steps x 2 ranks" in capsys.readouterr().out
