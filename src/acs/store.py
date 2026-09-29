"""Append-only `runs.jsonl` access: resume bookkeeping and the ReAct pool index that
Reflexion's trial-1 reuse depends on (§2.4)."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from . import config


class MissingPoolRun(RuntimeError):
    """Raised when a Reflexion run cannot find the ReAct pool trajectory it must reuse."""


@dataclass
class PoolRecord:
    """The part of a ReAct pool run that a Reflexion run needs for its trial 1."""

    run_id: str
    final_answer: str
    halted: bool
    n_steps: int
    n_llm_calls: int
    n_tool_calls: dict[str, int]
    tokens_in: int
    tokens_out: int
    usd_nocache: float
    usd_empirical: float
    transcript: str
    trace: list[dict] = field(default_factory=list)

    @classmethod
    def from_row(cls, row: dict, keep_trace: bool = True) -> "PoolRecord":
        return cls(
            run_id=row["run_id"],
            final_answer=row.get("final_answer") or "",
            halted=bool(row.get("halted")),
            n_steps=int(row.get("n_steps") or 0),
            n_llm_calls=int(row.get("n_llm_calls") or 0),
            n_tool_calls=dict(row.get("n_tool_calls") or {}),
            tokens_in=int(row.get("tokens_in") or 0),
            tokens_out=int(row.get("tokens_out") or 0),
            usd_nocache=float(row.get("usd_nocache") or 0.0),
            usd_empirical=float(row.get("usd_empirical") or 0.0),
            transcript=(row.get("extra") or {}).get("transcript", ""),
            trace=list(row.get("trace") or []) if keep_trace else [],
        )


class RunStore:
    def __init__(self, path: Path | None, keep_trace_below_replicate: int = config.R) -> None:
        self.path = Path(path) if path is not None else None
        self.keep_trace_below_replicate = keep_trace_below_replicate
        self.status: dict[str, str] = {}
        self.pool: dict[str, PoolRecord] = {}
        self._lock = asyncio.Lock()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._load()

    # ------------------------------------------------------------------ loading

    def _index(self, row: dict) -> None:
        run_id = row.get("run_id")
        if not run_id:
            return
        self.status[run_id] = row.get("status", "ok")
        if row.get("scaffold") == "react" and row.get("status") == "ok":
            keep = int(row.get("replicate", 0)) < self.keep_trace_below_replicate
            self.pool[run_id] = PoolRecord.from_row(row, keep_trace=keep)

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        for row in iter_jsonl(self.path):
            self._index(row)

    # ------------------------------------------------------------------ writing

    async def append(self, row: dict) -> None:
        async with self._lock:
            self._index(row)
            if self.path is None:
                return
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())

    # ------------------------------------------------------------------ queries

    def is_done(self, run_id: str) -> bool:
        """Resume rule (§T11): skip `ok` runs, re-run `error` ones."""
        return self.status.get(run_id) == "ok"

    def pool_record(self, benchmark: str, task_id: str, replicate: int) -> PoolRecord:
        run_id = f"{benchmark}:{task_id}:react:{replicate}"
        record = self.pool.get(run_id)
        if record is None:
            raise MissingPoolRun(
                f"pool run {run_id} is missing — stage `pool` must complete for "
                f"replicates r < R before stage `others` runs Reflexion"
            )
        return record


def iter_jsonl(path: Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def read_jsonl(path: Path) -> list[dict]:
    return list(iter_jsonl(path))
