"""T14.1 — read a run tag's JSONL logs into DataFrames.

`calls` and `runs` are the only inputs any number in the write-up depends on.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .. import config
from ..store import iter_jsonl

INVALID_ACTION_RE = re.compile(r"Invalid action\.")
NOT_FOUND_RE = re.compile(r"Could not find")

REFLEXION_SCAFFOLDS = ("reflexion_gt", "reflexion_self")


@dataclass
class Dataset:
    calls: pd.DataFrame
    runs: pd.DataFrame
    ownership: pd.DataFrame  # (run_id, call_id): which calls a run is billed for
    prompt_tokens: dict[str, int]
    dropped: pd.DataFrame  # runs with status != "ok"
    run_tag: str

    @property
    def benchmarks(self) -> list[str]:
        return sorted(self.runs["benchmark"].unique())

    def scaffold_runs(self, benchmark: str, scaffold: str) -> pd.DataFrame:
        return self.runs[
            (self.runs["benchmark"] == benchmark) & (self.runs["scaffold"] == scaffold)
        ]

    def drop_report(self) -> pd.DataFrame:
        """Per scaffold: how many runs were dropped. Must stay under 2% (§T14.1)."""
        keys = ["benchmark", "scaffold"]
        everything = pd.concat([self.runs, self.dropped], ignore_index=True)
        if not len(everything):
            return pd.DataFrame(columns=[*keys, "n_total", "n_dropped", "frac_dropped"])
        total = everything.groupby(keys, observed=True).size().rename("n_total")
        dropped = (
            self.dropped.groupby(keys, observed=True).size().rename("n_dropped")
            if len(self.dropped)
            else None
        )
        out = total.to_frame()
        out["n_dropped"] = (
            dropped.reindex(out.index).fillna(0).astype(int) if dropped is not None else 0
        )
        out["frac_dropped"] = out["n_dropped"] / out["n_total"]
        return out.reset_index().rename(columns={"level_0": "benchmark", "level_1": "scaffold"})


# --------------------------------------------------------------------------- derived fields


def _trace_features(trace: list[dict] | None, transcript: str) -> dict:
    trace = trace or []
    observations = [e.get("observation", "") for e in trace if e.get("kind") == "tool"]
    llm_texts = [e.get("text", "") or "" for e in trace if e.get("kind") == "llm"]
    blob = transcript or "\n".join(f"{t}" for t in llm_texts)
    return {
        "n_invalid_actions": len(INVALID_ACTION_RE.findall(blob)),
        "n_could_not_find": sum(bool(NOT_FOUND_RE.search(o)) for o in observations),
    }


def _explode_run(row: dict) -> dict:
    extra = row.get("extra") or {}
    tools = row.get("n_tool_calls") or {}
    transcript = extra.get("transcript", "")
    out = {
        "run_id": row["run_id"],
        "benchmark": row["benchmark"],
        "task_id": row["task_id"],
        "scaffold": row["scaffold"],
        "replicate": row["replicate"],
        "status": row.get("status", "ok"),
        "error": row.get("error"),
        "final_answer": row.get("final_answer") or "",
        "gold": row.get("gold") or "",
        "em": row.get("em", 0),
        "f1": row.get("f1", 0.0),
        "correct": row.get("correct", 0),
        "n_llm_calls": row.get("n_llm_calls", 0),
        "n_steps": row.get("n_steps", 0),
        "halted": bool(row.get("halted", False)),
        "tokens_in": row.get("tokens_in", 0),
        "tokens_out": row.get("tokens_out", 0),
        "usd_nocache_run": row.get("usd_nocache", 0.0),
        "usd_empirical_run": row.get("usd_empirical", 0.0),
        "wall_s": row.get("wall_s", 0.0),
        "n_search": tools.get("Search", 0),
        "n_lookup": tools.get("Lookup", 0),
        "n_calc": tools.get("Calculate", 0),
        "n_llm_tool": tools.get("LLM", 0),
        "answer_len_chars": len(row.get("final_answer") or ""),
        # scaffold-specific `extra` fields, flattened for analysis
        "plan_parse_ok": extra.get("plan_parse_ok"),
        "n_evidence": extra.get("n_evidence"),
        "n_plan_steps": len(extra["plan_steps"]) if "plan_steps" in extra else None,
        "n_trials": extra.get("n_trials"),
        "judge_parse_fail": extra.get("judge_parse_fail"),
        "judge_verdicts": extra.get("judge_verdicts"),
        "trial_correct": extra.get("trial_correct"),
        "trial_answers": extra.get("trial_answers"),
        "trial1_source_run_id": extra.get("trial1_source_run_id"),
    }
    out["n_tool_calls_total"] = sum(tools.values())
    out.update(_trace_features(row.get("trace"), transcript))
    return out


def build_ownership(runs_rows: list[dict]) -> pd.DataFrame:
    """Which run is billed for which call.

    A Reflexion run is billed for its own calls **and** for the trial-1 pool run's calls
    (§2.4); those calls are also billed to the pool run itself, which is correct — each arm
    is priced as if it had been run alone.
    """
    own: dict[str, str] = {r["run_id"]: r["run_id"] for r in runs_rows}
    pairs: list[tuple[str, str]] = []
    for r in runs_rows:
        source = (r.get("extra") or {}).get("trial1_source_run_id")
        if source and source in own:
            pairs.append((r["run_id"], source))
    return pd.DataFrame(pairs, columns=["run_id", "source_run_id"])


def load(run_tag: str, run_dir: Path | None = None, keep_traces: bool = False) -> Dataset:
    d = Path(run_dir) if run_dir is not None else config.run_dir(run_tag)
    runs_path, calls_path = d / "runs.jsonl", d / "calls.jsonl"
    if not runs_path.exists():
        raise FileNotFoundError(f"{runs_path} not found")

    # Later rows win, so a retried run replaces its earlier error row.
    latest: dict[str, dict] = {}
    for row in iter_jsonl(runs_path):
        latest[row["run_id"]] = row
    rows = list(latest.values())

    ownership_pairs = build_ownership(rows)
    exploded = [_explode_run(r) for r in rows]
    runs = pd.DataFrame(exploded)
    if keep_traces:
        runs["trace"] = [r.get("trace") for r in rows]

    dropped = runs[runs["status"] != "ok"].copy()
    runs = runs[runs["status"] == "ok"].copy()

    calls = pd.DataFrame(iter_jsonl(calls_path)) if calls_path.exists() else pd.DataFrame()
    if len(calls):
        for f in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens",
                  "output_tokens"):
            calls[f] = calls[f].fillna(0).astype("int64")
        calls["I"] = (
            calls["input_tokens"]
            + calls["cache_creation_input_tokens"]
            + calls["cache_read_input_tokens"]
        )
        calls["O"] = calls["output_tokens"]

    prompt_tokens_path = d / "prompt_tokens.json"
    prompt_tokens = (
        json.loads(prompt_tokens_path.read_text(encoding="utf-8"))
        if prompt_tokens_path.exists()
        else {}
    )

    # Ownership: each run owns its own calls, plus any Reflexion trial-1 source's calls.
    ownership = _ownership_table(calls, runs, ownership_pairs)

    return Dataset(
        calls=calls,
        runs=runs,
        ownership=ownership,
        prompt_tokens=prompt_tokens,
        dropped=dropped,
        run_tag=run_tag,
    )


def _ownership_table(
    calls: pd.DataFrame, runs: pd.DataFrame, pairs: pd.DataFrame
) -> pd.DataFrame:
    if not len(calls):
        return pd.DataFrame(columns=["run_id", "call_id"])
    direct = calls[["run_id", "call_id"]].copy()
    direct = direct[direct["run_id"].isin(set(runs["run_id"]))]
    if not len(pairs):
        return direct
    inherited = pairs.merge(
        calls[["run_id", "call_id"]].rename(columns={"run_id": "source_run_id"}),
        on="source_run_id",
        how="inner",
    )[["run_id", "call_id"]]
    return pd.concat([direct, inherited], ignore_index=True).drop_duplicates()
