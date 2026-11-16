"""A hand-built trace whose metrics can be computed on paper.

Two ranks, two steps, two MoE layers per step. Times are in milliseconds here and
converted to nanoseconds, so latency numbers read naturally.

    step 0  0..100 ms   tokens  rank0 10, rank1 30      kv 1000 / 1000    reqs 10 / 30
    step 1  100..160 ms tokens  rank0 20, rank1 20      kv 2000 / 6000    reqs 20 / 20

    ready times (ms)        rank0  rank1   wait   local durations   ALB
    step 0 layer 0            20     40     20       20, 40         0.75
    step 0 layer 1            60     90     30       10, 40         0.625
    step 1 layer 0           110    110      0       10, 10         1.0
    step 1 layer 1           140    130     10       20, 10         0.75
"""

import pandas as pd
import pytest

MS = 1_000_000


@pytest.fixture
def request_df():
    return pd.DataFrame({
        "req_id": ["a", "b"],
        "session_id": ["s", "s"],
        "arrival_ts": [0 * MS, 10 * MS],
        "dispatch_ts": [0 * MS, 10 * MS],
        "prefill_start_ts": [5 * MS, 20 * MS],
        "rank": [0, 1],
        "input_tokens": [100, 200],
        "cached_tokens": [0, 0],
        "output_tokens": [11, 21],
        "first_token_ts": [50 * MS, 60 * MS],
        "finish_ts": [150 * MS, 160 * MS],
    })


@pytest.fixture
def step_df():
    return pd.DataFrame({
        "step_id": [0, 0, 1, 1],
        "rank": [0, 1, 0, 1],
        "t_start": [0 * MS, 0 * MS, 100 * MS, 100 * MS],
        "t_end": [100 * MS, 100 * MS, 160 * MS, 160 * MS],
        "prefill_tokens": [0, 0, 0, 0],
        "decode_tokens": [10, 30, 20, 20],
        "num_reqs": [10, 30, 20, 20],
        "kv_tokens": [1000, 1000, 2000, 6000],
    })


@pytest.fixture
def layer_df():
    rows = [
        # step, layer, rank, ready, dispatch_done, combine_done   (ms)
        (0, 0, 0, 20, 45, 50), (0, 0, 1, 40, 45, 50),
        (0, 1, 0, 60, 95, 100), (0, 1, 1, 90, 95, 100),
        (1, 0, 0, 110, 115, 120), (1, 0, 1, 110, 115, 120),
        (1, 1, 0, 140, 150, 160), (1, 1, 1, 130, 150, 160),
    ]
    df = pd.DataFrame(rows, columns=["step_id", "layer_id", "rank", "t_ready", "t_dispatch_done", "t_combine_done"])
    for c in ("t_ready", "t_dispatch_done", "t_combine_done"):
        df[c] = df[c] * MS
    return df
