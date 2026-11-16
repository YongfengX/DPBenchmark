"""Step-table (L1) tracing for vLLM, engine side and merge side.

Engine side
    ``StepTracer`` runs inside every vLLM DP engine-core process. The hook
    patch in ``adapters/vllm/v0.32.0/step-trace.patch`` calls it from
    ``EngineCore.step`` and ``DPEngineCoreProc.run_busy_loop``. It is switched
    on by the environment variable ``DPBENCH_STEP_TRACE_DIR`` and writes one
    CSV per rank: ``<dir>/step.rank<k>.csv``.

    This half uses the standard library only and never imports vLLM, torch or
    pandas, so importing it costs the engine nothing.

Merge side
    ``load_trace_dir`` reads the per-rank files, ``window`` cuts out the steps
    of one replay run, and ``check_alignment`` tests the assumption the whole
    step table rests on: that forward pass k of a wave is the same step on
    every rank.

step_id
    ``wave * 2**32 + step_in_wave``, where ``step_in_wave`` counts forward
    passes inside the wave from 1 and dummy forwards count too. In v0.32.0
    ``DPEngineCoreProc.step_counter`` is exactly this counter minus one.
"""

from __future__ import annotations

import atexit
import os
import queue
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pandas is only needed on the merge side
    import pandas as pd

ENV_DIR = "DPBENCH_STEP_TRACE_DIR"
ENV_FLUSH_EVERY = "DPBENCH_STEP_TRACE_FLUSH_EVERY"
WAVE_SHIFT = 32

# Schema columns first, in schema order; the rest are adapter extras.
COLUMNS = (
    "step_id", "rank", "t_start", "t_end", "prefill_tokens", "decode_tokens", "num_reqs", "kv_tokens",
    "wave", "dummy", "step_in_wave", "fwd_seq", "t_exec_start",
)


def make_step_id(wave: int, step_in_wave: int) -> int:
    return (wave << WAVE_SHIFT) | step_in_wave


# --------------------------------------------------------------------------- engine side


class StepTracer:
    """Collects one row per forward pass of one DP rank.

    Call order per busy-loop iteration:

        begin()                     before scheduler.schedule()
        mark_exec()                 after schedule(), before execute_model()
        end(scheduler_output, ...)  after the model output is back on the host
        commit(wave, step_in_wave)  if the step ran a real forward

    or, when the rank had nothing to run:

        begin()
        commit_dummy(wave, step_in_wave)   after execute_dummy_batch()

    Rows are handed to a writer thread in blocks, at the end of each wave and
    whenever ``flush_every`` rows have piled up, so the engine loop never
    touches the file system.
    """

    def __init__(self, trace_dir: str | os.PathLike[str], rank: int, flush_every: int = 2048):
        self.rank = int(rank)
        self.path = Path(trace_dir) / f"step.rank{self.rank}.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_every = max(1, int(flush_every))
        self.fwd_seq = 0
        self._rows: list[tuple] = []
        self._t_start = 0
        self._t_exec = 0
        self._pending: tuple[int, int, int, int, int] | None = None
        self._queue: queue.SimpleQueue[list[tuple] | None] = queue.SimpleQueue()
        self._writer = threading.Thread(target=self._write_loop, name=f"dpbench-step-trace-{rank}", daemon=True)
        self._writer.start()
        atexit.register(self.close)

    @classmethod
    def from_env(cls, rank: int) -> StepTracer | None:
        trace_dir = os.environ.get(ENV_DIR)
        if not trace_dir:
            return None
        return cls(trace_dir, rank, int(os.environ.get(ENV_FLUSH_EVERY, "2048")))

    # -- hooks, in call order

    def begin(self) -> None:
        self._t_start = self._t_exec = time.monotonic_ns()
        self._pending = None

    def mark_exec(self) -> None:
        self._t_exec = time.monotonic_ns()

    def end(self, scheduler_output: Any, requests: dict[str, Any]) -> None:
        """Take t_end and count the step's work.

        Must run after the sampled tokens are back on the host (that is the
        proof the GPU work finished) and before ``update_from_output``, while
        every scheduled request is still in ``requests``.

        The prefill / decode split follows vLLM's own ``compute_iteration_details``:
        a request is in prefill while it has produced no output token.
        ``kv_tokens`` is the context length at the end of the step, because
        ``schedule()`` has already advanced ``num_computed_tokens``.
        """
        t_end = time.monotonic_ns()
        num_scheduled = scheduler_output.num_scheduled_tokens
        new_ids = {r.req_id for r in scheduler_output.scheduled_new_reqs}
        cached = scheduler_output.scheduled_cached_reqs
        prefill = decode = kv = 0
        for req_id, n in num_scheduled.items():
            if req_id in new_ids or cached.is_context_phase(req_id):
                prefill += n
            else:
                decode += n
            req = requests.get(req_id)
            if req is not None:
                kv += req.num_computed_tokens
        self._pending = (t_end, prefill, decode, len(num_scheduled), kv)

    def commit(self, wave: int, step_in_wave: int) -> None:
        if self._pending is None:
            return
        t_end, prefill, decode, num_reqs, kv = self._pending
        self._pending = None
        self._append(wave, step_in_wave, t_end, prefill, decode, num_reqs, kv, 0)

    def commit_dummy(self, wave: int, step_in_wave: int) -> None:
        self._pending = None
        self._append(wave, step_in_wave, time.monotonic_ns(), 0, 0, 0, 0, 1)

    def flush(self) -> None:
        if self._rows:
            self._queue.put(self._rows)
            self._rows = []

    def close(self) -> None:
        if not self._writer.is_alive():
            return
        self.flush()
        self._queue.put(None)
        self._writer.join(timeout=10)

    # -- internals

    def _append(self, wave: int, step_in_wave: int, t_end: int, prefill: int, decode: int, num_reqs: int, kv: int,
                dummy: int) -> None:
        self.fwd_seq += 1
        self._rows.append((
            make_step_id(wave, step_in_wave), self.rank, self._t_start, t_end, prefill, decode, num_reqs, kv,
            wave, dummy, step_in_wave, self.fwd_seq, self._t_exec,
        ))
        if len(self._rows) >= self.flush_every:
            self.flush()

    def _write_loop(self) -> None:
        new_file = not self.path.exists() or self.path.stat().st_size == 0
        with open(self.path, "a", encoding="ascii", newline="\n") as f:
            if new_file:
                f.write(",".join(COLUMNS) + "\n")
            while True:
                rows = self._queue.get()
                if rows is None:
                    break
                f.write("".join(",".join(map(str, row)) + "\n" for row in rows))
                f.flush()


# --------------------------------------------------------------------------- merge side


def load_trace_dir(trace_dir: str | os.PathLike[str]) -> pd.DataFrame:
    """All ranks' step rows from one trace directory, in schema dtypes."""
    import pandas as pd

    from . import schema

    files = sorted(Path(trace_dir).glob("step.rank*.csv"))
    if not files:
        raise FileNotFoundError(f"{trace_dir}: no step.rank*.csv files")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if df.duplicated(["rank", "fwd_seq"]).any():
        raise ValueError(f"{trace_dir}: holds the trace of more than one server launch; use a fresh directory per launch")
    df["dummy"] = df["dummy"].astype(bool)
    return schema.coerce(df, "step").sort_values(["fwd_seq", "rank"], ignore_index=True)


def window(step: pd.DataFrame, t_first: int, t_last: int) -> pd.DataFrame:
    """Steps of one run: every step that overlaps [t_first, t_last] on any rank.

    A step is kept or dropped as a whole, so each kept step_id still has a row
    for every rank. The server keeps tracing across replay runs; the request
    table's first dispatch and last finish give the cut.
    """
    hit = step.loc[(step["t_end"] >= t_first) & (step["t_start"] <= t_last), "fwd_seq"]
    if hit.empty:
        return step.iloc[0:0].copy()
    keep = step["fwd_seq"].between(hit.min(), hit.max())
    return step.loc[keep].reset_index(drop=True)


def check_alignment(step: pd.DataFrame, n_ranks: int, max_report: int = 5) -> list[str]:
    """Problems with cross-rank step alignment; an empty list means aligned.

    Three independent checks:

    1. every step_id has exactly one row from each rank;
    2. ordering by the per-rank lifetime counter ``fwd_seq`` gives the same
       step_id on every rank, so the wave-based id and the plain forward count
       agree;
    3. the ranks' time intervals for one step overlap. Every forward pass has
       EP collectives, so the same forward on two ranks cannot be disjoint in
       time. This one uses timestamps only and would catch an id that is
       consistent but wrong.
    """
    problems: list[str] = []
    if step.empty:
        return ["step: no rows"]
    ranks = sorted(int(r) for r in step["rank"].unique())
    if ranks != list(range(n_ranks)):
        problems.append(f"step: ranks present {ranks}, expected 0..{n_ranks - 1}")

    per_step = step.groupby("step_id")["rank"].agg(["size", "nunique"])
    bad = per_step[(per_step["size"] != n_ranks) | (per_step["nunique"] != n_ranks)]
    if len(bad):
        problems.append(f"step: {len(bad)} of {len(per_step)} step_ids do not have one row per rank, "
                        f"e.g. {[int(i) for i in bad.index[:max_report]]}")

    ids = step.groupby("fwd_seq")["step_id"].nunique()
    split = ids[ids > 1]
    if len(split):
        problems.append(f"step: at {len(split)} forward counts the ranks disagree on step_id, "
                        f"first at fwd_seq {int(split.index[0])}")

    span = step.groupby("step_id").agg(latest_start=("t_start", "max"), earliest_end=("t_end", "min"))
    disjoint = span[span["latest_start"] > span["earliest_end"]]
    if len(disjoint):
        problems.append(f"step: in {len(disjoint)} of {len(span)} steps the ranks' time intervals do not overlap, "
                        f"e.g. {[int(i) for i in disjoint.index[:max_report]]}")
    return problems


def all_dummy_share(step: pd.DataFrame) -> float:
    """Share of steps where every rank ran a dummy forward.

    Not an error: after the last request of a wave finishes, the ranks keep
    stepping until the next idle check, which v0.32.0 runs every
    ``--dp-sync-interval`` (default 16) steps.
    """
    return float(step.groupby("step_id")["dummy"].all().mean()) if len(step) else 0.0
