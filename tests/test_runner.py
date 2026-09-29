"""T11 acceptance: a FakeLLM end-to-end run over all scaffolds, plus resume."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from acs import config
from acs.data import Task
from acs.llm import CallLog, FakeLLM
from acs.runner import build_work, execute_run, gold_for, run_stage, shuffle_work
from acs.store import RunStore, read_jsonl
from tests.conftest import FakeWiki

TASKS = [
    Task(task_id=f"t{i}", question=f"Question {i}?", answer="Arthur's Magazine",
         benchmark="hotpotqa", type="comparison")
    for i in range(3)
]

PAGES = {
    "Arthur's Magazine": (
        "Arthur's Magazine (1844-1846) was an American literary periodical.",
        "Arthur's Magazine (1844-1846) was an American literary periodical.",
    )
}


def scripted(role: str, messages: list[dict]) -> str:
    """One deterministic reply per role, good enough to drive every scaffold."""
    if role == "actor" or role == "executor":
        text = messages[-1]["content"]
        text = text if isinstance(text, str) else "".join(b["text"] for b in text)
        if "Observation" in text:
            return "Thought 2: done.\nAction 2: Finish[Arthur's Magazine]"
        return "Thought 1: search.\nAction 1: Search[Arthur's Magazine]"
    if role == "planner":
        return (
            "Plan: Search for the magazine.\n#E1 = Search[Arthur's Magazine]\n"
            "Plan: Decide.\n#E2 = LLM[Which was first? Given context: #E1]\n"
            "1. Find when Arthur's Magazine started.\n"
        )
    if role in ("solver", "worker_llm"):
        return "Arthur's Magazine"
    if role == "reflector":
        return "I should have searched both pages."
    if role == "judge":
        return "Matches the evidence.\nVERDICT: CORRECT"
    raise AssertionError(f"unexpected role {role}")


def make_llm(tmp_path: Path) -> FakeLLM:
    return FakeLLM(scripted, call_log=CallLog(tmp_path / "calls.jsonl", max_usd=1e9))


async def run_all(tmp_path: Path, k: int = 3, r: int = 2) -> tuple[RunStore, FakeLLM]:
    llm = make_llm(tmp_path)
    store = RunStore(tmp_path / "runs.jsonl", keep_trace_below_replicate=r)
    wiki = FakeWiki(PAGES)
    for stage in ("pool", "others"):
        work = shuffle_work(build_work(stage, ["hotpotqa"], {"hotpotqa": TASKS}, k, r))
        await run_stage(stage, work, llm, store, wiki, max_in_flight=4, quiet=True)
    return store, llm


async def test_end_to_end_run_is_internally_consistent(tmp_path):
    store, llm = await run_all(tmp_path)

    runs = read_jsonl(tmp_path / "runs.jsonl")
    calls = read_jsonl(tmp_path / "calls.jsonl")
    assert runs and calls

    # Every run succeeded and every scaffold is present with the right replicate count.
    assert all(r["status"] == "ok" for r in runs), [
        r["error"] for r in runs if r["status"] != "ok"
    ]
    by_scaffold: dict[str, int] = {}
    for r in runs:
        by_scaffold[r["scaffold"]] = by_scaffold.get(r["scaffold"], 0) + 1
    assert by_scaffold == {
        "react": 3 * 3, "rewoo": 3 * 2, "plan_exec": 3 * 2,
        "reflexion_gt": 3 * 2, "reflexion_self": 3 * 2,
    }

    # Every call belongs to a run that exists.
    run_ids = {r["run_id"] for r in runs}
    assert {c["run_id"] for c in calls} <= run_ids

    # Per-run token sums match that run's own calls, plus trial 1 for Reflexion runs.
    own_in: dict[str, int] = {}
    own_out: dict[str, int] = {}
    for c in calls:
        own_in[c["run_id"]] = own_in.get(c["run_id"], 0) + c["input_tokens"]
        own_out[c["run_id"]] = own_out.get(c["run_id"], 0) + c["output_tokens"]
    for r in runs:
        extra = r["extra"]
        expected_in = own_in.get(r["run_id"], 0) + extra.get("trial1_tokens_in", 0)
        expected_out = own_out.get(r["run_id"], 0) + extra.get("trial1_tokens_out", 0)
        assert r["tokens_in"] == expected_in, r["run_id"]
        assert r["tokens_out"] == expected_out, r["run_id"]

    # Reflexion runs reference pool runs that exist.
    for r in runs:
        if r["scaffold"].startswith("reflexion"):
            assert r["extra"]["trial1_source_run_id"] in run_ids

    # Conversations are numbered from 0 with no gaps.
    by_conv: dict[str, list[int]] = {}
    for c in calls:
        by_conv.setdefault(c["conversation_id"], []).append(c["turn_index"])
    for conv, turns in by_conv.items():
        assert sorted(turns) == list(range(len(turns))), conv


async def test_resume_only_completes_the_missing_runs(tmp_path):
    llm = make_llm(tmp_path)
    store = RunStore(tmp_path / "runs.jsonl", keep_trace_below_replicate=2)
    wiki = FakeWiki(PAGES)

    work = shuffle_work(build_work("pool", ["hotpotqa"], {"hotpotqa": TASKS}, 3, 2))
    # Simulate a crash after the first four pool runs.
    for item in work[:4]:
        await store.append(await execute_run(item, llm, store, wiki=wiki))
    assert len(read_jsonl(tmp_path / "runs.jsonl")) == 4

    store2 = RunStore(tmp_path / "runs.jsonl", keep_trace_below_replicate=2)
    stats = await run_stage("pool", work, llm, store2, wiki, 4, quiet=True)
    assert stats["skipped"] == 4
    assert stats["done"] == len(work) - 4
    rows = read_jsonl(tmp_path / "runs.jsonl")
    assert len(rows) == len(work)
    assert len({r["run_id"] for r in rows}) == len(work)


async def test_error_runs_are_recorded_and_retried(tmp_path):
    llm = make_llm(tmp_path)
    store = RunStore(tmp_path / "runs.jsonl", keep_trace_below_replicate=2)
    # A Reflexion run with no pool run behind it must fail cleanly, not crash the process.
    work = build_work("others", ["hotpotqa"], {"hotpotqa": TASKS[:1]}, 3, 1)
    stats = await run_stage("others", work, llm, store, FakeWiki(PAGES), 4, quiet=True)
    assert stats["errors"] == 2  # reflexion_gt and reflexion_self
    errors = [r for r in read_jsonl(tmp_path / "runs.jsonl") if r["status"] == "error"]
    assert all("MissingPoolRun" in r["error"] for r in errors)
    assert not any(store.is_done(r["run_id"]) for r in errors)


async def test_spend_guard_stops_launching_new_runs(tmp_path):
    llm = FakeLLM(scripted, call_log=CallLog(tmp_path / "calls.jsonl", max_usd=1e-9))
    store = RunStore(tmp_path / "runs.jsonl")
    work = build_work("pool", ["hotpotqa"], {"hotpotqa": TASKS}, 3, 1)
    stats = await run_stage("pool", work, llm, store, FakeWiki(PAGES), 1, quiet=True)
    assert stats["stopped"] > 0
    assert stats["done"] < len(work)


def test_work_lists_and_shuffle_are_deterministic():
    tasks = {"hotpotqa": TASKS, "gsm8k": []}
    pool = build_work("pool", ["hotpotqa"], tasks, k=10, r=3)
    assert len(pool) == 30 and {w.scaffold for w in pool} == {"react"}
    others = build_work("others", ["hotpotqa"], tasks, k=10, r=3)
    assert len(others) == 3 * 3 * 4
    assert {w.scaffold for w in others} == set(config.OTHER_SCAFFOLDS)
    assert [w.run_id for w in shuffle_work(pool)] == [w.run_id for w in shuffle_work(pool)]
    assert [w.run_id for w in shuffle_work(pool)] != [w.run_id for w in pool]


def test_gold_for_extracts_the_gsm8k_number():
    t = Task(task_id="g", question="q", answer="steps\n#### 1,234", benchmark="gsm8k")
    assert gold_for(t) == "1234"
    h = Task(task_id="h", question="q", answer="yes", benchmark="hotpotqa")
    assert gold_for(h) == "yes"
