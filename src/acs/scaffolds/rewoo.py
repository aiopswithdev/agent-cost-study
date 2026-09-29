"""T8 — ReWOO: plan everything up front, execute the workers, then solve once."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from .. import config
from ..prompts import library as lib
from .base import RunContext, RunResult, extract_finish, finalize

_PLAN_RE = re.compile(r"^\s*Plan\s*:\s*(.*)$")
_EVIDENCE_RE = re.compile(r"^\s*#E(\d+)\s*=\s*(\w+)\[(.*)\]\s*$")
_EVIDENCE_REF_RE = re.compile(r"#E(\d+)")


@dataclass
class PlanStep:
    index: int
    plan: str
    tool: str
    tool_input: str


def parse_plan(text: str, max_evidence: int = config.REWOO_MAX_EVIDENCE) -> list[PlanStep]:
    """`Plan: …` lines set the current plan text; `#Ek = Tool[input]` lines define evidence."""
    steps: list[PlanStep] = []
    current_plan = ""
    for line in (text or "").split("\n"):
        pm = _PLAN_RE.match(line)
        if pm:
            current_plan = pm.group(1).strip()
            continue
        em = _EVIDENCE_RE.match(line)
        if em:
            steps.append(
                PlanStep(
                    index=int(em.group(1)),
                    plan=current_plan,
                    tool=em.group(2).strip(),
                    tool_input=em.group(3),
                )
            )
            if len(steps) >= max_evidence:
                break
    return steps


def substitute_evidence(text: str, evidence: dict[int, str]) -> str:
    """Raw `#Ej` substitution, longest index first so `#E10` is not clobbered by `#E1`."""
    out = text
    for j in sorted(evidence, key=lambda k: (-len(str(k)), -k)):
        out = out.replace(f"#E{j}", evidence[j])
    return out


async def run_rewoo(ctx: RunContext) -> RunResult:
    started = time.perf_counter()
    benchmark = ctx.benchmark

    # 1. Planner -----------------------------------------------------------------
    planner_id = lib.system_prompt_id("rewoo_planner", benchmark)
    plan_res = await ctx.call(
        role="planner",
        conversation_id=f"{ctx.run_id}:plan",
        turn_index=0,
        system_prompt_id=planner_id,
        system=lib.rewoo_planner_system(benchmark),
        messages=[{"role": "user", "content": lib.rewoo_planner_user(ctx.question)}],
    )
    plan_text = plan_res.text

    # 2. Parse -------------------------------------------------------------------
    steps = parse_plan(plan_text)
    plan_parse_ok = len(steps) >= 1

    # 3. Workers (sequential, in order) ------------------------------------------
    evidence: dict[int, str] = {}
    pairs: list[tuple[str, str]] = []
    for k, step in enumerate(steps, start=1):
        tool_input = substitute_evidence(step.tool_input, evidence)
        if step.tool == "LLM":
            worker_res = await ctx.call(
                role="worker_llm",
                conversation_id=f"{ctx.run_id}:w{k}",
                turn_index=0,
                system_prompt_id=lib.system_prompt_id("worker_llm", benchmark),
                system=lib.WORKER_LLM_SYSTEM,
                messages=[{"role": "user", "content": tool_input or "(no input)"}],
            )
            value = worker_res.text.strip()
            ctx.toolbox.n_tool_calls["LLM"] += 1
            ctx.toolbox.trace.append(
                {"kind": "tool", "tool": "LLM", "input": tool_input, "observation": value}
            )
        else:
            value = await ctx.toolbox.run(step.tool, tool_input)
        evidence[step.index] = value
        pairs.append((step.plan, value))

    # 4. Solver ------------------------------------------------------------------
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
        halted=False,
        n_steps=len(steps),
        extra={
            "plan_parse_ok": plan_parse_ok,
            "n_evidence": len(steps),
            "plan_text": plan_text,
        },
        started=started,
    )
