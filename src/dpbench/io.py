"""Run directories: one manifest plus up to three trace tables.

    run_dir/
        manifest.json
        request.parquet      (or .jsonl)
        step.parquet         optional, level L1
        layer.parquet        optional, level L2
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from . import schema


@dataclass
class Run:
    manifest: dict[str, Any]
    request: pd.DataFrame
    step: pd.DataFrame | None = None
    layer: pd.DataFrame | None = None

    @property
    def n_ranks(self) -> int:
        return int(self.manifest["parallel"]["dp_size"])


def read_table(path: Path, kind: str) -> pd.DataFrame:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".jsonl":
        df = pd.read_json(path, lines=True, dtype=False)
    else:
        raise ValueError(f"unsupported table format: {path}")
    return schema.coerce(df, kind)


def write_table(df: pd.DataFrame, path: Path, kind: str) -> None:
    df = schema.coerce(df, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix == ".jsonl":
        df.to_json(path, orient="records", lines=True)
    else:
        raise ValueError(f"unsupported table format: {path}")


def _find(run_dir: Path, kind: str) -> Path | None:
    for suffix in (".parquet", ".jsonl"):
        p = run_dir / f"{kind}{suffix}"
        if p.exists():
            return p
    return None


def load_run(run_dir: str | Path) -> Run:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    req_path = _find(run_dir, "request")
    if req_path is None:
        raise FileNotFoundError(f"{run_dir}: no request table")
    step_path, layer_path = _find(run_dir, "step"), _find(run_dir, "layer")
    return Run(
        manifest=manifest,
        request=read_table(req_path, "request"),
        step=read_table(step_path, "step") if step_path else None,
        layer=read_table(layer_path, "layer") if layer_path else None,
    )


def save_run(run: Run, run_dir: str | Path, fmt: str = "parquet") -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps(run.manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_table(run.request, run_dir / f"request.{fmt}", "request")
    if run.step is not None:
        write_table(run.step, run_dir / f"step.{fmt}", "step")
    if run.layer is not None:
        write_table(run.layer, run_dir / f"layer.{fmt}", "layer")


def validate_run(run: Run) -> list[str]:
    problems = schema.validate_manifest(run.manifest)
    problems += schema.validate_table(run.request, "request")
    if run.step is not None:
        problems += schema.validate_table(run.step, "step")
    if run.layer is not None:
        problems += schema.validate_table(run.layer, "layer")
        if run.step is not None:
            orphan = set(run.layer["step_id"].unique()) - set(run.step["step_id"].unique())
            if orphan:
                problems.append(f"layer: {len(orphan)} step_ids are not in the step table")
    level = run.manifest.get("trace", {}).get("level")
    if level in ("L1", "L2") and run.step is None:
        problems.append(f"manifest says trace level {level} but there is no step table")
    if level == "L2" and run.layer is None:
        problems.append("manifest says trace level L2 but there is no layer table")
    return problems
