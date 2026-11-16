import numpy as np
import pandas as pd
import pytest

from dpbench import metrics as m

MS = 1_000_000


def test_throughput_and_duration(request_df):
    assert m.run_duration_ns(request_df) == 160 * MS
    assert m.total_tokens(request_df, "computed") == 332
    assert m.total_tokens(request_df, "output") == 32
    assert m.throughput(request_df) == pytest.approx(332 / 0.160)


def test_cached_tokens_are_not_counted(request_df):
    request_df.loc[0, "cached_tokens"] = 40
    assert m.total_tokens(request_df, "computed") == 292


def test_balance_ratio_per_load_kind(step_df):
    tokens = m.balance_ratio(step_df, "tokens")
    assert tokens["br"].tolist() == pytest.approx([20 / 30, 1.0])
    assert tokens["tau_ns"].tolist() == [100 * MS, 60 * MS]
    assert m.balance_ratio(step_df, "kv")["br"].tolist() == pytest.approx([1.0, 4000 / 6000])
    assert m.balance_ratio(step_df, "reqs")["br"].tolist() == pytest.approx([20 / 30, 1.0])
    s = m.balance_summary(tokens)
    assert s["br_mean"] == pytest.approx((20 / 30 + 1) / 2)
    assert s["br_time_weighted"] == pytest.approx((100 * 20 / 30 + 60) / 160)


def test_missing_rank_counts_as_zero_load(step_df):
    one_rank_idle = step_df[~((step_df.step_id == 1) & (step_df["rank"] == 1))]
    br = m.balance_ratio(one_rank_idle, "tokens", n_ranks=2)
    assert br.loc[1, "br"] == pytest.approx(10 / 20)


def test_all_idle_step_has_no_ratio(step_df):
    step_df.loc[step_df.step_id == 1, ["decode_tokens", "num_reqs", "kv_tokens"]] = 0
    br = m.balance_ratio(step_df, "tokens")
    assert np.isnan(br.loc[1, "br"])
    assert m.balance_summary(br)["n_loaded_steps"] == 1


def test_sol_throughput(step_df):
    # ideal busy time = 100 * 2/3 + 60 = 126.67 ms out of 160 ms
    assert m.sol_throughput(step_df, 1000.0, 160 * MS) == pytest.approx(1000 * 160 / (100 * 20 / 30 + 60))
    # 40 ms of idle gaps outside the steps are kept
    assert m.sol_throughput(step_df, 1000.0, 200 * MS) == pytest.approx(1000 * 200 / (40 + 100 * 20 / 30 + 60))


def test_wait_fraction_full_sampling(layer_df, step_df):
    waits = m.layer_waits(layer_df)
    assert (waits["wait_sum_ns"] / MS).tolist() == [20, 30, 0, 10]
    wf = m.wait_fraction(layer_df, step_df, 160 * MS)
    assert wf["w_step"] == pytest.approx(60 / (2 * 160))
    assert wf["w"] == pytest.approx(0.1875)
    assert m.wait_throughput(1000.0, wf["w"]) == pytest.approx(1000 / 0.8125)


def test_wait_fraction_sampled_steps_scale_up(layer_df, step_df):
    only_step0 = layer_df[layer_df.step_id == 0]
    wf = m.wait_fraction(only_step0, step_df, 160 * MS)
    assert wf["sampled_steps"] == 1 and wf["total_steps"] == 2
    assert wf["w_step"] == pytest.approx(50 / (2 * 100))
    assert wf["w"] == pytest.approx(0.25)


def test_w_matches_time_saved_by_balancing(layer_df, step_df):
    # Balanced, each layer costs the mean instead of the max of the local durations:
    # saved = sum(max - mean) = 10 + 15 + 0 + 5 = 30 ms = W * T.
    w = m.wait_fraction(layer_df, step_df, 160 * MS)["w"]
    assert w * 160 == pytest.approx(30)


def test_attention_load_balance(layer_df, step_df):
    alb = m.attention_load_balance(layer_df, step_df)
    assert alb["alb"].tolist() == pytest.approx([0.75, 0.625, 1.0, 0.75])
    assert alb["alb"].mean() == pytest.approx(0.78125)


def test_layer_table_must_reference_known_steps(layer_df, step_df):
    with pytest.raises(ValueError):
        m.wait_fraction(layer_df, step_df[step_df.step_id == 0])


def test_latency(request_df):
    lat = m.latency_table(request_df)
    assert lat["ttft_ms"].tolist() == pytest.approx([50, 50])
    assert lat["tpot_ms"].tolist() == pytest.approx([10, 5])
    assert lat["queue_ms"].tolist() == pytest.approx([5, 10])
    assert m.latency_summary(request_df)["tpot_p50_ms"] == pytest.approx(7.5)


def test_single_token_output_has_no_tpot(request_df):
    request_df.loc[0, "output_tokens"] = 1
    assert np.isnan(m.latency_table(request_df)["tpot_ms"].iloc[0])
    assert m.slo_attainment(request_df, m.SLO(tpot_ms=1.0)) == 0.5  # the one-token request cannot violate TPOT


def test_slo_attainment(request_df):
    assert m.slo_attainment(request_df, m.SLO(tpot_ms=8)) == 0.5
    # TTFT limit = 40 + 60 * input / 1000 -> 46 ms (fail) and 52 ms (pass)
    assert m.slo_attainment(request_df, m.SLO(ttft_base_ms=40, ttft_per_1k_input_ms=60)) == 0.5
    assert m.slo_attainment(request_df, m.SLO(tpot_ms=20, ttft_base_ms=100)) == 1.0


def test_max_sustainable_rate():
    assert m.max_sustainable_rate([1, 2, 3, 4], [1.0, 0.995, 0.98, 0.999]) == 2
    assert m.max_sustainable_rate([1, 2], [0.5, 1.0]) is None
    assert m.max_sustainable_rate([2, 1], [0.99, 1.0]) == 2


def test_recovery_point_and_interval():
    rec = m.recovery([90, 92], [80, 80], [100, 100])
    assert rec.r == pytest.approx(0.55)
    assert rec.recoverable == pytest.approx(0.2)
    assert 0.5 <= rec.ci_low <= rec.r <= rec.ci_high <= 0.6
    assert rec.reportable


def test_recovery_endpoints():
    assert m.recovery([80], [80], [100]).r == 0.0
    assert m.recovery([100], [80], [100]).r == 1.0
    assert np.isnan(m.recovery([100], [80], [100]).ci_low)  # one repeat: no interval


def test_recovery_not_reportable_when_gap_is_inside_noise():
    assert not m.recovery([90, 92], [80, 80], [100, 100], floor=0.1).reportable   # 0.2 < 3 * 0.1
    assert m.recovery([90, 92], [80, 80], [100, 100], floor=0.05).reportable      # 0.2 >= 3 * 0.05


def test_recovery_not_reportable_when_mirror_is_not_faster():
    rec = m.recovery([85], [90], [88])
    assert rec.recoverable < 0 and not rec.reportable


def test_recovery_is_deterministic():
    a = m.recovery([90, 95, 93], [80, 82, 81], [100, 99, 101], seed=7)
    b = m.recovery([90, 95, 93], [80, 82, 81], [100, 99, 101], seed=7)
    assert a == b


def test_noise_floor():
    assert m.noise_floor([100, 100, 100]) == 0
    assert m.noise_floor([99, 101]) == pytest.approx(np.std([99, 101], ddof=1) / 100)
    assert np.isnan(m.noise_floor([100]))


def test_tpot_guard(request_df):
    slow = request_df.copy()
    slow["finish_ts"] = slow["first_token_ts"] + (slow["finish_ts"] - slow["first_token_ts"]) * 2
    assert m.tpot_guard(request_df, request_df)["ok"]
    assert not m.tpot_guard(slow, request_df)["ok"]


def test_check_work(request_df):
    expected = pd.DataFrame({"req_id": ["a", "b"], "input_tokens": [100, 200], "output_tokens": [11, 21]})
    assert m.check_work(request_df, expected) == []
    short = expected.assign(output_tokens=[11, 22])
    assert any("output_tokens" in p for p in m.check_work(request_df, short))
    more = pd.concat([expected, pd.DataFrame({"req_id": ["c"], "input_tokens": [1], "output_tokens": [1]})])
    assert any("did not finish" in p for p in m.check_work(request_df, more))


def test_summarize(request_df, step_df, layer_df):
    s = m.summarize(request_df, step_df, layer_df)
    x = 332 / 0.160
    assert s["x_tok_s"] == pytest.approx(x)
    assert s["computed_tokens"] == 80
    assert s["w"] == pytest.approx(0.1875)
    assert s["x_w_tok_s"] == pytest.approx(x / 0.8125)
    assert s["x_sol_tokens_tok_s"] == pytest.approx(x * 160 / (100 * 20 / 30 + 60))
    assert s["alb_mean"] == pytest.approx(0.78125)
    assert "w" not in m.summarize(request_df, step_df)
