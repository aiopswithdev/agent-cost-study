"""T10 — Reflexion in two modes that differ **only** in the source of the stopping bit:

* `gt`   — stop when exact-match against the gold answer is true (the original paper's
           HotpotQA setup; the stopping oracle whose premium RQ2 measures).
* `self` — stop when an LLM judge with no gold access says CORRECT.

Everything else — the actor prompt, the reflector prompt, the memory, the budget — is
identical, so the accuracy gap is attributable to the oracle and nothing else.
"""

from __future__ import annotations

import time

from .. import config
from ..grading import grade
from ..prompts import library as lib
from ..store import PoolRecord
from .base import RunContext, RunResult, finalize
from .react import react_loop


def _seed_from_trial1(ctx: RunContext, trial1: PoolRecord) -> None:
    """Trial 1 is a ReAct pool trajectory; its cost is counted in full in this run (§2.4)."""
    ctx.counters["n_llm_calls"] += trial1.n_llm_calls
    ctx.counters["tokens_in"] += trial1.tokens_in
    ctx.counters["tokens_out"] += trial1.tokens_out
    ctx.counters["usd_nocache"] += trial1.usd_nocache
    ctx.counters["usd_empirical"] += trial1.usd_empirical
    for tool, n in trial1.n_tool_calls.items():
        ctx.toolbox.n_tool_calls[tool] = ctx.toolbox.n_tool_calls.get(tool, 0) + n
    ctx.trace.extend(trial1.trace)


async def _judge(ctx: RunContext, trial: int, answer: str, transcript: str) -> tuple[str, bool]:
    res = await ctx.call(
        role="judge",
        conversation_id=f"{ctx.run_id}:judge{trial}",
        turn_index=0,
        system_prompt_id=lib.system_prompt_id("judge", ctx.benchmark),
        system=lib.JUDGE_SYSTEM,
        messages=[
            {"role": "user", "content": lib.judge_user(ctx.question, transcript, answer)}
        ],
    )
    return lib.parse_verdict(res.text)


async def _reflect(ctx: RunContext, trial: int, transcript: str) -> str:
    """The reflector sees only the question and its own failed trajectory — never the gold
    answer and never the judge's justification, so the two modes stay comparable."""
    res = await ctx.call(
        role="reflector",
        conversation_id=f"{ctx.run_id}:refl{trial}",
        turn_index=0,
        system_prompt_id=lib.system_prompt_id("reflector", ctx.benchmark),
        system=lib.reflector_system(ctx.benchmark),
        messages=[
            {"role": "user", "content": lib.reflector_user(ctx.question, transcript)}
        ],
    )
    return res.text.strip()


async def run_reflexion(ctx: RunContext, mode: str, trial1: PoolRecord) -> RunResult:
    if mode not in ("gt", "self"):
        raise ValueError(f"unknown Reflexion mode {mode!r}")
    started = time.perf_counter()

    _seed_from_trial1(ctx, trial1)

    answer: str | None = trial1.final_answer or None
    transcript = trial1.transcript
    halted = trial1.halted
    n_steps = trial1.n_steps

    reflections: list[str] = []
    trial_answers: list[str] = []
    trial_correct: list[int] = []
    judge_verdicts: list[str | None] = []
    judge_parse_fail = 0
    n_trials = 0

    for t in range(1, config.REFLEXION_MAX_TRIALS + 1):
        n_trials = t
        if t > 1:
            header = lib.format_reflections(reflections[-config.REFLEXION_MEMORY:])
            outcome = await react_loop(
                ctx,
                system=lib.react_system(ctx.benchmark),
                system_prompt_id=lib.system_prompt_id("react", ctx.benchmark),
                first_user_msg=f"{header}\n\nQuestion: {ctx.question}",
                conversation_label=f"trial{t}",
                max_steps=config.REACT_MAX_STEPS,
                role="actor",
            )
            answer, transcript, halted = outcome.answer, outcome.transcript, outcome.halted
            n_steps += outcome.n_steps

        # --- the stopping signal: the ONLY difference between the two modes ------
        verdict: str | None = None
        if not answer:
            success = False  # no judge call for an empty answer
        elif mode == "gt":
            success = grade(ctx.benchmark, answer, ctx.gold)["correct"] == 1
        else:
            verdict, failed = await _judge(ctx, t, answer, transcript)
            judge_parse_fail += int(failed)
            success = verdict == "CORRECT"

        trial_answers.append(answer or "")
        trial_correct.append(grade(ctx.benchmark, answer or "", ctx.gold)["correct"])
        judge_verdicts.append(verdict)

        if success or t == config.REFLEXION_MAX_TRIALS:
            break

        reflections.append(await _reflect(ctx, t, transcript))

    return finalize(
        ctx,
        answer or "",
        halted=halted,
        n_steps=n_steps,
        extra={
            "n_trials": n_trials,
            "trial_answers": trial_answers,
            "trial_correct": trial_correct,
            "judge_verdicts": judge_verdicts,
            "judge_parse_fail": judge_parse_fail,
            "reflections": reflections,
            "trial1_source_run_id": trial1.run_id,
            "trial1_tokens_in": trial1.tokens_in,
            "trial1_tokens_out": trial1.tokens_out,
            "trial1_usd_nocache": trial1.usd_nocache,
            "trial1_usd_empirical": trial1.usd_empirical,
            "trial1_n_llm_calls": trial1.n_llm_calls,
        },
        started=started,
    )


async def run_reflexion_gt(ctx: RunContext, trial1: PoolRecord) -> RunResult:
    return await run_reflexion(ctx, "gt", trial1)


async def run_reflexion_self(ctx: RunContext, trial1: PoolRecord) -> RunResult:
    return await run_reflexion(ctx, "self", trial1)
