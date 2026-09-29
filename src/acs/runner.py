"""T11 — the experiment runner.

    python -m acs.runner --run-tag pilot --benchmarks hotpotqa gsm8k \
           --stages pool others --pilot --k 3 --r 1

Stages run in the order given. `others` needs the pool runs with `r < R` to already exist,
because Reflexion's trial 1 *is* a ReAct pool trajectory (§2.4).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from . import config
from .data import Task, load_tasks
from .grading import gsm8k_gold, grade
from .llm import BaseLLM, CallLog, FakeLLM, SpendLimitExceeded, make_llm
from .prompts import library as lib
from .scaffolds.base import RunContext, RunResult, make_run_id
from .scaffolds.plan_exec import run_plan_exec
from .scaffolds.react import run_react
from .scaffolds.reflexion import run_reflexion_gt, run_reflexion_self
from .scaffolds.rewoo import run_rewoo
from .store import MissingPoolRun, RunStore
from .tools.toolbox import ToolBox
from .tools.wiki import WikiClient, close_default_client, default_client

SCAFFOLD_FNS = {
    "react": run_react,
    "rewoo": run_rewoo,
    "plan_exec": run_plan_exec,
    "reflexion_gt": run_reflexion_gt,
    "reflexion_self": run_reflexion_self,
}
REFLEXION_SCAFFOLDS = ("reflexion_gt", "reflexion_self")


@dataclass(frozen=True)
class WorkItem:
    benchmark: str
    task: Task
    scaffold: str
    replicate: int

    @property
    def run_id(self) -> str:
        return make_run_id(self.benchmark, self.task.task_id, self.scaffold, self.replicate)


# --------------------------------------------------------------------------- one run


def gold_for(task: Task) -> str:
    """What goes in `runs.jsonl`'s `gold` field; both forms are accepted by `grade`."""
    if task.benchmark == "gsm8k":
        g = gsm8k_gold(task.answer)
        return str(int(g)) if abs(g - round(g)) < 1e-9 else str(g)
    return task.answer


async def execute_run(
    item: WorkItem,
    llm: BaseLLM,
    store: RunStore,
    wiki: WikiClient | None = None,
) -> dict:
    """Build a fresh ToolBox and RunContext, run the scaffold, grade, return the row."""
    started = time.perf_counter()
    gold = gold_for(item.task)
    task = Task(
        task_id=item.task.task_id,
        question=item.task.question,
        answer=gold,
        benchmark=item.benchmark,
        type=item.task.type,
    )
    trace: list[dict] = []
    toolbox = ToolBox(
        item.benchmark,
        wiki=wiki,
        trace=trace,
        with_llm_worker=(item.scaffold == "rewoo"),
    )
    ctx = RunContext(
        run_id=item.run_id,
        benchmark=item.benchmark,
        task=task,
        scaffold=item.scaffold,
        replicate=item.replicate,
        llm=llm,
        toolbox=toolbox,
        trace=trace,
        attempt_id=uuid.uuid4().hex,
    )

    try:
        if item.scaffold in REFLEXION_SCAFFOLDS:
            trial1 = store.pool_record(item.benchmark, item.task.task_id, item.replicate)
            result = await SCAFFOLD_FNS[item.scaffold](ctx, trial1)
        else:
            result = await SCAFFOLD_FNS[item.scaffold](ctx)
        grades = grade(item.benchmark, result.final_answer, gold)
        result.em, result.f1, result.correct = grades["em"], grades["f1"], grades["correct"]
    except Exception as exc:
        c = ctx.counters
        result = RunResult(
            run_id=item.run_id,
            benchmark=item.benchmark,
            task_id=item.task.task_id,
            scaffold=item.scaffold,
            replicate=item.replicate,
            status="error",
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            gold=gold,
            n_llm_calls=c["n_llm_calls"],
            n_tool_calls=dict(toolbox.n_tool_calls),
            tokens_in=c["tokens_in"],
            tokens_out=c["tokens_out"],
            usd_nocache=c["usd_nocache"],
            usd_empirical=c["usd_empirical"],
            trace=list(trace),
        )
    result.wall_s = time.perf_counter() - started
    row = result.to_dict()
    row["attempt_id"] = ctx.attempt_id
    return row


# --------------------------------------------------------------------------- work lists


def build_work(
    stage: str,
    benchmarks: list[str],
    tasks_by_benchmark: dict[str, list[Task]],
    k: int,
    r: int,
    scaffolds: list[str] | tuple[str, ...] = config.OTHER_SCAFFOLDS,
) -> list[WorkItem]:
    """`scaffolds` restricts the `others` stage (e.g. R=2 for ReWOO/Plan-Exec only, §6.3)."""
    work: list[WorkItem] = []
    for benchmark in benchmarks:
        for task in tasks_by_benchmark[benchmark]:
            if stage == "pool":
                for rep in range(k):
                    work.append(WorkItem(benchmark, task, "react", rep))
            else:
                for rep in range(r):
                    for scaffold in config.OTHER_SCAFFOLDS:
                        if scaffold in scaffolds:
                            work.append(WorkItem(benchmark, task, scaffold, rep))
    return work


def shuffle_work(work: list[WorkItem], seed: int = config.SEED) -> list[WorkItem]:
    """Interleave scaffolds and tasks so no arm systematically gets a warmer cache."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(work))
    return [work[int(i)] for i in order]


# --------------------------------------------------------------------------- run-tag setup


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=config.ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def write_config_json(run_dir: Path, args: argparse.Namespace) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config.snapshot(),
        "git_commit": git_commit(),
        "cli_args": vars(args),
        "argv": sys.argv,
        "launched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (run_dir / "config.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )


async def write_prompt_tokens(run_dir: Path, llm: BaseLLM) -> dict[str, int]:
    """§4.3 — one `count_tokens` call per distinct system prompt; needed by T14.2."""
    path = run_dir / "prompt_tokens.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    for prompt_id, text in lib.all_system_prompts().items():
        counts[prompt_id] = await llm.count_tokens(text, [{"role": "user", "content": "x"}])
    path.write_text(json.dumps(counts, indent=2), encoding="utf-8")
    return counts


# --------------------------------------------------------------------------- offline self-test


def _smoke_responder(role: str, messages: list[dict]) -> str:
    """A deterministic stand-in used only by `--fake-llm`, to exercise the whole pipeline
    (runner -> JSONL -> analysis) without spending anything. Never used in a real run."""
    last = messages[-1]["content"]
    last = last if isinstance(last, str) else "".join(b.get("text", "") for b in last)
    is_math = "math word problem" in last or "Calculate[" in last or "$" in last

    if role in ("actor", "executor"):
        if "Observation" in last:
            return "Thought 2: that is enough.\nAction 2: Finish[42]" if is_math else (
                "Thought 2: that is enough.\nAction 2: Finish[The Saimaa Gesture]"
            )
        return (
            "Thought 1: compute it.\nAction 1: Calculate[6 * 7]"
            if is_math
            else "Thought 1: look it up.\nAction 1: Search[The Saimaa Gesture]"
        )
    if role == "planner":
        return (
            "Plan: Compute the value.\n#E1 = Calculate[6 * 7]\n1. Compute the value.\n"
            if is_math
            else "Plan: Look up the film.\n#E1 = Search[The Saimaa Gesture]\n"
                 "1. Look up the film.\n"
        )
    if role in ("solver", "worker_llm"):
        return "42" if is_math else "The Saimaa Gesture"
    if role == "reflector":
        return "I should have searched both entities before answering."
    if role == "judge":
        return "The trajectory supports it.\nVERDICT: CORRECT"
    raise AssertionError(f"unexpected role {role!r}")


# --------------------------------------------------------------------------- the loop


async def run_stage(
    stage: str,
    work: list[WorkItem],
    llm: BaseLLM,
    store: RunStore,
    wiki: WikiClient | None,
    max_in_flight: int,
    quiet: bool = False,
) -> dict[str, int]:
    pending = [w for w in work if not store.is_done(w.run_id)]
    stats = {"done": 0, "errors": 0, "skipped": len(work) - len(pending), "stopped": 0}
    if not pending:
        if not quiet:
            print(f"[{stage}] nothing to do ({stats['skipped']} runs already ok)")
        return stats

    sem = asyncio.Semaphore(max_in_flight)
    started = time.perf_counter()
    bar = tqdm(total=len(pending), desc=stage, disable=quiet)

    async def one(item: WorkItem) -> None:
        async with sem:
            if llm.stopped:
                stats["stopped"] += 1
                bar.update(1)
                return
            row = await execute_run(item, llm, store, wiki=wiki)
            await store.append(row)
            stats["done"] += 1
            stats["errors"] += int(row["status"] != "ok")
            bar.update(1)
            n = stats["done"]
            if n % 50 == 0 and not quiet:
                rate = n / max(time.perf_counter() - started, 1e-9) * 60
                tqdm.write(
                    f"[{stage}] {n}/{len(pending)} runs | "
                    f"billed ${llm.call_log.total_usd_billed:.3f} of "
                    f"${llm.call_log.max_usd:.2f} | "
                    f"{rate:.1f} runs/min | errors={stats['errors']}"
                    + (f" | wiki throttled={wiki.throttled}" if wiki is not None
                       and hasattr(wiki, "throttled") else "")
                )

    try:
        await asyncio.gather(*(one(w) for w in pending))
    finally:
        bar.close()

    if stats["stopped"] and not quiet:
        print(f"[{stage}] SPEND GUARD tripped — {stats['stopped']} runs were not launched")
    return stats


async def async_main(args: argparse.Namespace) -> int:
    run_dir = config.run_dir(args.run_tag)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_config_json(run_dir, args)

    call_log = CallLog(run_dir / "calls.jsonl", max_usd=args.max_usd)
    llm: BaseLLM = (
        FakeLLM(_smoke_responder, call_log=call_log)
        if args.fake_llm
        else make_llm(call_log=call_log, concurrency=args.llm_concurrency)
    )
    if args.fake_llm:
        print("!! --fake-llm: no API calls are made; this output is for pipeline self-test only")
    else:
        print(f"provider={config.PROVIDER} model={llm.model} | spend guard ${args.max_usd:.2f} billed")
    store = RunStore(run_dir / "runs.jsonl", keep_trace_below_replicate=args.r)
    wiki = default_client() if "hotpotqa" in args.benchmarks else None

    if call_log.n_calls:
        print(
            f"resuming: {call_log.n_calls} calls already logged "
            f"(${call_log.total_usd_billed:.3f} billed), "
            f"{sum(1 for s in store.status.values() if s == 'ok')} runs ok"
        )

    await write_prompt_tokens(run_dir, llm)

    tasks_by_benchmark = {
        b: load_tasks(b, pilot=args.pilot, n=args.n) for b in args.benchmarks
    }
    if args.limit is not None:
        # The frozen sample is stored in shuffled order, so a prefix is a random subset
        # (§6.3 "keep the first 200 IDs of the frozen sample").
        tasks_by_benchmark = {b: ts[: args.limit] for b, ts in tasks_by_benchmark.items()}
    for b, ts in tasks_by_benchmark.items():
        print(f"{b}: {len(ts)} tasks")

    total_errors = 0
    try:
        for stage in args.stages:
            work = shuffle_work(
                build_work(stage, args.benchmarks, tasks_by_benchmark, args.k, args.r,
                           scaffolds=args.scaffolds),
                seed=config.SEED,
            )
            stats = await run_stage(
                stage, work, llm, store, wiki, args.max_runs_in_flight
            )
            total_errors += stats["errors"]
            print(f"[{stage}] {stats}")
            if llm.stopped:
                print("spend guard is set — not starting further stages")
                break
    finally:
        if hasattr(llm, "aclose"):
            await llm.aclose()
        await close_default_client()

    print(
        f"total: ${call_log.total_usd_billed:.3f} billed "
        f"(list price: ${call_log.total_usd_empirical:.3f} with cache, "
        f"${call_log.total_usd_nocache:.3f} without) over {call_log.n_calls} calls; "
        f"errors={total_errors}"
    )
    return 1 if total_errors else 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m acs.runner")
    ap.add_argument("--run-tag", required=True, help="e.g. pilot or main")
    ap.add_argument("--benchmarks", nargs="+", default=list(config.BENCHMARKS),
                    choices=list(config.BENCHMARKS))
    ap.add_argument("--stages", nargs="+", default=["pool", "others"],
                    choices=["pool", "others"])
    ap.add_argument("--pilot", action=argparse.BooleanOptionalAction, default=False,
                    help="use only the first PILOT_N tasks of each frozen sample")
    ap.add_argument("--k", type=int, default=config.POOL_K, help="ReAct pool replicates")
    ap.add_argument("--r", type=int, default=config.R, help="replicates for other scaffolds")
    ap.add_argument("--n", type=int, default=config.N, help="frozen sample size to load")
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first LIMIT tasks of the frozen sample (scope-down)")
    ap.add_argument("--scaffolds", nargs="+", default=list(config.OTHER_SCAFFOLDS),
                    choices=list(config.OTHER_SCAFFOLDS),
                    help="restrict the `others` stage to these scaffolds")
    ap.add_argument("--max-usd", type=float, default=config.MAX_USD,
                    help="spend guard on billed dollars; no new runs start past it")
    ap.add_argument("--max-runs-in-flight", type=int, default=config.MAX_RUNS_IN_FLIGHT)
    ap.add_argument("--llm-concurrency", type=int, default=config.LLM_CONCURRENCY)
    ap.add_argument("--fake-llm", action="store_true",
                    help="offline self-test: scripted replies, no API calls, no spend")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
