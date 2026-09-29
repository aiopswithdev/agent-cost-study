"""Synthetic `runs.jsonl` / `calls.jsonl` for exercising the analysis chain offline."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np

from acs import config
from acs.pricing import usd_nocache
from acs.scaffolds.base import make_run_id

ANSWERS = {
    "hotpotqa": ["Arthur's Magazine", "First for Women", "The Saimaa Gesture", ""],
    "gsm8k": ["72", "24", "10", ""],
}
GOLD = {"hotpotqa": "Arthur's Magazine", "gsm8k": "72"}

#: Per scaffold: (P(correct answer), LLM calls per run, input tokens per call).
PROFILE = {
    "react": (0.45, 4, 1400),
    "rewoo": (0.40, 3, 1800),
    "plan_exec": (0.50, 7, 1600),
    "reflexion_gt": (0.70, 9, 1500),
    "reflexion_self": (0.55, 8, 1500),
}

SYSTEM_PROMPT_IDS = {
    "actor": "react_{b}_v1",
    "planner": "rewoo_planner_{b}_v1",
    "executor": "pe_executor_{b}_v1",
    "solver": "solver_{b}_v1",
    "worker_llm": "worker_llm_v1",
    "reflector": "reflector_{b}_v1",
    "judge": "judge_{b}_v1",
}
ROLE_BY_SCAFFOLD = {
    "react": "actor",
    "rewoo": "planner",
    "plan_exec": "executor",
    "reflexion_gt": "actor",
    "reflexion_self": "actor",
}


def _call_rows(run_id: str, row: dict, role: str, n_calls: int, base_in: int, clock: list[int]):
    benchmark = row["benchmark"]
    prompt_id = SYSTEM_PROMPT_IDS[role].format(b=benchmark)
    conversation = f"{run_id}:c0"
    calls = []
    for turn in range(n_calls):
        clock[0] += 1
        calls.append(
            {
                "call_id": str(uuid.uuid4()),
                "run_id": run_id,
                "benchmark": benchmark,
                "task_id": row["task_id"],
                "scaffold": row["scaffold"],
                "replicate": row["replicate"],
                "role": role,
                "conversation_id": conversation,
                "turn_index": turn,
                "system_prompt_id": prompt_id,
                "model": config.MODEL,
                "temperature": 0.7,
                "max_tokens": 512,
                "input_tokens": base_in + 600 * turn,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 90,
                "stop_reason": "stop_sequence",
                "latency_s": 1.0,
                "ts": f"2026-03-01T00:{clock[0] // 60:02d}:{clock[0] % 60:02d}",
                "attempts": 1,
                "error": None,
            }
        )
    return calls


def generate(
    out_dir: Path,
    n_tasks: int = 12,
    k: int = config.POOL_K,
    r: int = config.R,
    benchmarks: tuple[str, ...] = config.BENCHMARKS,
    seed: int = 7,
    n_errors: int = 0,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    runs: list[dict] = []
    calls: list[dict] = []
    clock = [0]

    for benchmark in benchmarks:
        gold = GOLD[benchmark]
        for t in range(n_tasks):
            task_id = f"{benchmark}_task{t}"
            for scaffold, (p_correct, n_calls, base_in) in PROFILE.items():
                reps = k if scaffold == "react" else r
                for rep in range(reps):
                    run_id = make_run_id(benchmark, task_id, scaffold, rep)
                    correct = bool(rng.random() < p_correct)
                    answer = gold if correct else str(
                        rng.choice([a for a in ANSWERS[benchmark] if a != gold])
                    )
                    row = {
                        "run_id": run_id,
                        "benchmark": benchmark,
                        "task_id": task_id,
                        "scaffold": scaffold,
                        "replicate": rep,
                        "status": "ok",
                        "error": None,
                        "final_answer": answer,
                        "gold": gold,
                        "em": int(answer == gold),
                        "f1": float(answer == gold),
                        "correct": int(answer == gold),
                        "n_llm_calls": n_calls,
                        "n_tool_calls": {
                            "Search": 2 if benchmark == "hotpotqa" else 0,
                            "Lookup": 1 if benchmark == "hotpotqa" else 0,
                            "Calculate": 0 if benchmark == "hotpotqa" else 2,
                            "LLM": 1 if scaffold == "rewoo" else 0,
                        },
                        "n_steps": n_calls,
                        "halted": bool(answer == ""),
                        "wall_s": 6.0 + float(rng.random()),
                        "extra": {},
                        "trace": [
                            {"kind": "llm", "role": ROLE_BY_SCAFFOLD[scaffold],
                             "call_id": "x", "text": "Thought 1: go."},
                            {"kind": "tool", "tool": "Search", "input": "x",
                             "observation": "Could not find x. Similar: []."},
                        ],
                    }
                    if scaffold == "react":
                        row["extra"]["transcript"] = (
                            "Thought 1: go.\nAction 1: Search[x]\nObservation 1: y"
                            + ("\nObservation 2: Invalid action." if rng.random() < 0.1 else "")
                        )
                    if scaffold == "rewoo":
                        row["extra"].update(
                            plan_parse_ok=bool(rng.random() < 0.9), n_evidence=3,
                            plan_text="Plan: x\n#E1 = Search[x]",
                        )
                    if scaffold == "plan_exec":
                        row["extra"].update(
                            plan_steps=["a", "b"], step_results=["r1", "r2"],
                            plan_parse_ok=True,
                        )
                    run_calls = _call_rows(
                        run_id, row, ROLE_BY_SCAFFOLD[scaffold], n_calls, base_in, clock
                    )
                    if scaffold.startswith("reflexion"):
                        source = make_run_id(benchmark, task_id, "react", rep)
                        source_row = next(x for x in runs if x["run_id"] == source)
                        n_trials = int(rng.integers(1, config.REFLEXION_MAX_TRIALS + 1))
                        row["extra"].update(
                            n_trials=n_trials,
                            trial_answers=[answer] * n_trials,
                            trial_correct=[int(answer == gold)] * n_trials,
                            judge_verdicts=(
                                [str(rng.choice(["CORRECT", "INCORRECT"])) for _ in range(n_trials)]
                                if scaffold == "reflexion_self"
                                else [None] * n_trials
                            ),
                            judge_parse_fail=0,
                            reflections=["try harder"] * max(n_trials - 1, 0),
                            trial1_source_run_id=source,
                            trial1_tokens_in=source_row["tokens_in"],
                            trial1_tokens_out=source_row["tokens_out"],
                            trial1_usd_nocache=source_row["usd_nocache"],
                            trial1_usd_empirical=source_row["usd_empirical"],
                            trial1_n_llm_calls=source_row["n_llm_calls"],
                        )
                    row["tokens_in"] = sum(c["input_tokens"] for c in run_calls)
                    row["tokens_out"] = sum(c["output_tokens"] for c in run_calls)
                    row["usd_nocache"] = sum(usd_nocache(c) for c in run_calls)
                    row["usd_empirical"] = row["usd_nocache"]
                    if scaffold.startswith("reflexion"):
                        row["tokens_in"] += row["extra"]["trial1_tokens_in"]
                        row["tokens_out"] += row["extra"]["trial1_tokens_out"]
                        row["usd_nocache"] += row["extra"]["trial1_usd_nocache"]
                        row["usd_empirical"] += row["extra"]["trial1_usd_empirical"]
                    runs.append(row)
                    calls.extend(run_calls)

    for i in range(n_errors):
        broken = dict(runs[i])
        broken["run_id"] = broken["run_id"] + ":broken"
        broken.update(status="error", error="RuntimeError: synthetic", correct=0, em=0, f1=0.0)
        runs.append(broken)

    _write(out_dir / "runs.jsonl", runs)
    _write(out_dir / "calls.jsonl", calls)
    (out_dir / "prompt_tokens.json").write_text(
        json.dumps(
            {
                pid.format(b=b): 1500 + 100 * i
                for i, pid in enumerate(SYSTEM_PROMPT_IDS.values())
                for b in config.BENCHMARKS
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_dir


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
