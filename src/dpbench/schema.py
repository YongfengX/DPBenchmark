"""Trace-table and manifest schemas (design doc section 4).

Three tables carry every metric. All timestamps are integers in nanoseconds
read from the machine's monotonic clock (``time.monotonic_ns()``), which is
shared by all processes on one host.

    request  one row per request                      level L0
    step     one row per (rank, step)                 level L1
    layer    one row per (rank, MoE layer invocation) level L2, sampled
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Any

import jsonschema
import pandas as pd

SCHEMA_VERSION = "0.1"


@dataclass(frozen=True)
class Column:
    name: str
    kind: str  # "int" | "str" | "bool"
    required: bool
    doc: str


REQUEST = (
    Column("req_id", "str", True, "unique request id"),
    Column("session_id", "str", True, "conversation / agent session the request belongs to"),
    Column("arrival_ts", "int", True, "intended arrival time of the request (open-loop schedule)"),
    Column("dispatch_ts", "int", True, "time the request was sent to its rank"),
    Column("prefill_start_ts", "int", False, "time the engine first scheduled the request (server side)"),
    Column("rank", "int", True, "DP rank that served the request"),
    Column("input_tokens", "int", True, "prompt length in tokens"),
    Column("cached_tokens", "int", False, "prompt tokens served from the prefix cache (server side)"),
    Column("output_tokens", "int", True, "tokens actually generated"),
    Column("first_token_ts", "int", True, "time the first output token reached the client"),
    Column("finish_ts", "int", True, "time the last output token reached the client"),
)

STEP = (
    Column("step_id", "int", True, "globally aligned step number; the same step has the same id on every rank"),
    Column("rank", "int", True, "DP rank"),
    Column("t_start", "int", True, "step start on this rank"),
    Column("t_end", "int", True, "step end on this rank, taken after the GPU work really finished"),
    Column("prefill_tokens", "int", True, "prompt tokens computed in this step (cache hits excluded)"),
    Column("decode_tokens", "int", True, "tokens decoded in this step"),
    Column("num_reqs", "int", True, "active requests on this rank in this step"),
    Column("kv_tokens", "int", True, "sum of context lengths of the active requests"),
    Column("wave", "int", False, "engine request-wave number, if the framework has one"),
    Column("dummy", "bool", False, "true when the rank ran a dummy forward to stay aligned"),
)

LAYER = (
    Column("step_id", "int", True, "step the layer invocation belongs to"),
    Column("layer_id", "int", True, "MoE layer index within the forward pass"),
    Column("rank", "int", True, "DP rank"),
    Column("t_ready", "int", True, "local work done, rank ready to enter dispatch"),
    Column("t_dispatch_done", "int", True, "dispatch all-to-all finished"),
    Column("t_combine_done", "int", True, "combine all-to-all finished"),
)

TABLES: dict[str, tuple[Column, ...]] = {"request": REQUEST, "step": STEP, "layer": LAYER}
KEYS: dict[str, list[str]] = {
    "request": ["req_id"],
    "step": ["step_id", "rank"],
    "layer": ["step_id", "layer_id", "rank"],
}
# Timestamp columns that must be non-decreasing in this order within a row.
ORDERED: dict[str, list[str]] = {
    "request": ["arrival_ts", "dispatch_ts", "first_token_ts", "finish_ts"],
    "step": ["t_start", "t_end"],
    "layer": ["t_ready", "t_dispatch_done", "t_combine_done"],
}


def columns(kind: str, required_only: bool = False) -> list[str]:
    return [c.name for c in TABLES[kind] if c.required or not required_only]


def coerce(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Return a copy with canonical dtypes. Optional columns that are missing are added as nulls."""
    out = df.copy()
    for col in TABLES[kind]:
        if col.name not in out.columns:
            if col.required:
                continue
            out[col.name] = pd.NA
        if col.kind == "int":
            out[col.name] = pd.to_numeric(out[col.name], errors="coerce").astype("Int64")
        elif col.kind == "bool":
            out[col.name] = out[col.name].astype("boolean")
        else:
            out[col.name] = out[col.name].astype("string")
    return out[[c.name for c in TABLES[kind]] + [c for c in out.columns if c not in columns(kind)]]


def validate_table(df: pd.DataFrame, kind: str) -> list[str]:
    """Return a list of problems; an empty list means the table is valid."""
    problems: list[str] = []
    spec = TABLES[kind]
    missing = [c.name for c in spec if c.required and c.name not in df.columns]
    if missing:
        return [f"{kind}: missing required columns {missing}"]
    if len(df) == 0:
        return [f"{kind}: table is empty"]
    for col in spec:
        if col.name not in df.columns:
            continue
        s = df[col.name]
        if col.required and s.isna().any():
            problems.append(f"{kind}.{col.name}: {int(s.isna().sum())} null values in a required column")
        if col.kind == "int" and not pd.api.types.is_integer_dtype(s.dropna().infer_objects()):
            problems.append(f"{kind}.{col.name}: expected integers, got {s.dtype}")
    dup = int(df.duplicated(KEYS[kind]).sum())
    if dup:
        problems.append(f"{kind}: {dup} rows repeat the key {KEYS[kind]}")
    order = ORDERED[kind]
    for a, b in zip(order, order[1:]):
        bad = int((df[b] < df[a]).sum())
        if bad:
            problems.append(f"{kind}: {bad} rows have {b} earlier than {a}")
    for name in ("input_tokens", "output_tokens", "prefill_tokens", "decode_tokens", "num_reqs", "kv_tokens", "rank"):
        if name in df.columns and (df[name].dropna() < 0).any():
            problems.append(f"{kind}.{name}: negative values")
    return problems


def manifest_schema() -> dict[str, Any]:
    text = resources.files("dpbench").joinpath("schemas/manifest.schema.json").read_text(encoding="utf-8")
    return json.loads(text)


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    validator = jsonschema.Draft202012Validator(manifest_schema())
    return [
        f"manifest.{'.'.join(str(p) for p in err.absolute_path) or '<root>'}: {err.message}"
        for err in sorted(validator.iter_errors(manifest), key=lambda e: list(e.absolute_path))
    ]


# Fields that may differ between two runs that are being compared (design doc 7, rule 2).
_FREE_FIELDS = {("run_id",), ("created_utc",), ("notes",), ("policy",), ("replay", "mode"), ("replay", "lag_s"),
                ("replay", "jitter_s"), ("framework", "launch_command")}


def manifest_diff(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """Fields that differ between two manifests and are not allowed to (config-freeze rule)."""
    diffs: list[str] = []

    def walk(x: Any, y: Any, path: tuple[str, ...]) -> None:
        if path in _FREE_FIELDS:
            return
        if isinstance(x, dict) and isinstance(y, dict):
            for key in sorted(set(x) | set(y)):
                walk(x.get(key), y.get(key), path + (key,))
        elif x != y:
            diffs.append(f"{'.'.join(path)}: {x!r} != {y!r}")

    walk(a, b, ())
    # The workload fingerprint legitimately differs between mirror/lag/test schedules of the same base set.
    return [d for d in diffs if not d.startswith("workload.fingerprint")]
