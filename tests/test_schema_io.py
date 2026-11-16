import json
from pathlib import Path

import pandas as pd

from dpbench import io, schema

EXAMPLE = Path(__file__).parent.parent / "examples" / "manifest.4xL40S.vllm-0.32.json"


def manifest():
    return json.loads(EXAMPLE.read_text())


def test_example_manifest_is_valid():
    assert schema.validate_manifest(manifest()) == []


def test_manifest_rejects_missing_and_unknown_fields():
    m = manifest()
    del m["parallel"]["ep_size"]
    m["hardware"]["interconnect"] = "wifi"
    problems = schema.validate_manifest(m)
    assert any("ep_size" in p for p in problems)
    assert any("interconnect" in p for p in problems)


def test_valid_tables(request_df, step_df, layer_df):
    assert schema.validate_table(request_df, "request") == []
    assert schema.validate_table(step_df, "step") == []
    assert schema.validate_table(layer_df, "layer") == []


def test_table_problems_are_reported(request_df, step_df):
    assert any("missing" in p for p in schema.validate_table(step_df.drop(columns="kv_tokens"), "step"))
    backwards = step_df.assign(t_end=step_df["t_start"] - 1)
    assert any("earlier" in p for p in schema.validate_table(backwards, "step"))
    doubled = pd.concat([step_df, step_df.iloc[:1]])
    assert any("repeat" in p for p in schema.validate_table(doubled, "step"))
    unmerged = request_df.assign(rank=-1)
    assert any("negative" in p for p in schema.validate_table(unmerged, "request"))


def test_server_side_columns_may_be_empty(request_df):
    client_only = request_df.drop(columns=["prefill_start_ts", "cached_tokens"])
    assert schema.validate_table(schema.coerce(client_only, "request"), "request") == []


def test_run_roundtrip(tmp_path, request_df, step_df, layer_df):
    m = manifest()
    m["trace"]["level"] = "L2"
    for fmt in ("parquet", "jsonl"):
        d = tmp_path / fmt
        io.save_run(io.Run(m, request_df, step_df, layer_df), d, fmt)
        run = io.load_run(d)
        assert io.validate_run(run) == []
        assert run.n_ranks == 4
        assert run.request["finish_ts"].tolist() == request_df["finish_ts"].tolist()
        assert run.layer["t_ready"].tolist() == layer_df["t_ready"].tolist()


def test_trace_level_must_match_tables(tmp_path, request_df):
    m = manifest()
    m["trace"]["level"] = "L1"
    io.save_run(io.Run(m, request_df), tmp_path)
    assert any("no step table" in p for p in io.validate_run(io.load_run(tmp_path)))


def test_config_freeze_diff():
    a, b = manifest(), manifest()
    b["run_id"] = "other"
    b["policy"]["dispatch"] = "jsq"
    b["replay"]["mode"] = "mirror"
    b["workload"]["fingerprint"] = "different schedule of the same set"
    assert schema.manifest_diff(a, b) == []
    b["engine"]["max_num_seqs"] = 64
    b["framework"]["version"] = "0.33.0"
    diff = schema.manifest_diff(a, b)
    assert len(diff) == 2 and any("max_num_seqs" in d for d in diff) and any("version" in d for d in diff)
