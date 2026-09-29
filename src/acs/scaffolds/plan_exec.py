"""T9 — Plan-and-Execute: one plan, a ReAct executor per step, then the shared solver.
No replanning (Zhou et al. 2026's "no replanning" variant)."""

from __future__ import annotations

import re
import time

from .. import config
from ..prompts import library as lib
from .base import RunContext, RunResult, extract_finish, finalize
from .react import react_loop

_PLAN_LINE_RE = re.compile(r"^\s*(\d+)[.)]\s+(.+)$")

FALLBACK_STEP = "Answer the question."


def parse_plan(text: str, max_steps: int = config.PE_MAX_PLAN_STEPS) -> tuple[list[str], bool]:
    """Numbered lines, in order, capped at `max_steps`. Returns `(steps, parse_ok)`."""
    steps: list[str] = []
    for line in (text or "").split("\n"):
        m = _PLAN_LINE_RE.match(line)
        if m:
            steps.append(m.group(2).strip())
            if len(steps) >= max_steps:
                break
    if not steps:
        return [FALLBACK_STEP], False
    return steps, True


async def run_plan_exec(ctx: RunContext) -> RunResult:
    started = time.perf_counter()
    benchmark = ctx.benchmark

    # 1. Planner -----------------------------------------------------------------
    plan_res = await ctx.call(
        role="planner",
        conversation_id=f"{ctx.run_id}:plan",
        turn_index=0,
        system_prompt_id=lib.system_prompt_id("pe_planner", benchmark),
        system=lib.pe_planner_system(benchmark),
        messages=[{"role": "user", "content": lib.pe_planner_user(ctx.question)}],
    )
    plan_steps, plan_parse_ok = parse_plan(plan_res.text)

    # 2. Executor (one ReAct conversation per step; the ToolBox is shared) --------
    executor_system = lib.pe_executor_system(benchmark)
    executor_id = lib.system_prompt_id("pe_executor", benchmark)
    step_results: list[str] = []
    total_steps = 0
    any_halt = False

    for j, _step in enumerate(plan_steps, start=1):
        outcome = await react_loop(
            ctx,
            system=executor_system,
            system_prompt_id=executor_id,
            first_user_msg=lib.pe_executor_user(ctx.question, plan_steps, step_results, j),
            conversation_label=f"step{j}",
            max_steps=config.PE_MAX_EXEC_STEPS,
            role="executor",
        )
        total_steps += outcome.n_steps
        any_halt = any_halt or outcome.halted
        step_results.append(outcome.answer if outcome.answer else "No result.")

    # 3. Solver (identical to ReWOO's) -------------------------------------------
    pairs = list(zip(plan_steps, step_results))
    solver_res = await ctx.call(
        role="solver",
        conversation_id=f"{ctx.run_id}:solve",
        turn_index=0,
        system_prompt_id=lib.system_prompt_id("solver", benchmark),
        system=lib.solver_system(benchmark),
        messages=[{"role": "user", "content": lib.solver_user(ctx.question, pairs)}],
    )
    final_answer = solver_res.text.strip()
    finished = extract_finish(final_answer)
    if finished is not None:
        final_answer = finished.strip()

    return finalize(
        ctx,
        final_answer,
        halted=any_halt,
        n_steps=total_steps,
        extra={
            "plan_steps": plan_steps,
            "step_results": step_results,
            "plan_parse_ok": plan_parse_ok,
        },
        started=started,
    )
