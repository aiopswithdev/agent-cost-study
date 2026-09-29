"""T7 — the ReAct loop. Also the executor used by Plan-and-Execute and by Reflexion trials."""

from __future__ import annotations

import time
from dataclasses import dataclass

from .. import config
from ..prompts import library as lib
from .base import RunContext, RunResult, finalize, parse_action, truncate_after_first_action


@dataclass
class LoopOutcome:
    answer: str | None
    transcript: str
    halted: bool
    n_steps: int


async def react_loop(
    ctx: RunContext,
    *,
    system: str,
    system_prompt_id: str,
    first_user_msg: str,
    conversation_label: str,
    max_steps: int = config.REACT_MAX_STEPS,
    role: str = "actor",
    finish_is_final: bool = True,
) -> LoopOutcome:
    """Thought / Action / Observation until `Finish[...]` or `max_steps` is exhausted.

    The conversation is append-only: call *t*'s message list is call *t−1*'s list plus the
    assistant reply we stored plus one new user message (§4.1), which is what makes the
    ideal-cache counterfactual of T14.2 valid.
    """
    conversation_id = f"{ctx.run_id}:{conversation_label}"
    messages: list[dict] = [{"role": "user", "content": first_user_msg}]
    transcript_parts: list[str] = []
    n_steps = 0

    for i in range(1, max_steps + 1):
        n_steps = i
        res = await ctx.call(
            role=role,
            conversation_id=conversation_id,
            turn_index=i - 1,
            system_prompt_id=system_prompt_id,
            system=system,
            messages=messages,
            max_tokens=config.MAX_TOKENS[role],
            stop=["\nObservation", "\nQuestion:"],
        )
        text = truncate_after_first_action(res.text).strip() or "(no output)"
        messages.append({"role": "assistant", "content": text})
        transcript_parts.append(text)

        act = parse_action(text)
        if act is None:
            obs = "Invalid action. Valid actions are " + ctx.toolbox.valid_actions_str()
        elif act[0] == "Finish" and finish_is_final:
            return LoopOutcome(
                answer=act[1].strip(),
                transcript="\n".join(transcript_parts),
                halted=False,
                n_steps=n_steps,
            )
        else:
            obs = await ctx.toolbox.run(act[0], act[1])

        observation_line = f"Observation {i}: {obs}"
        transcript_parts.append(observation_line)
        messages.append({"role": "user", "content": observation_line})

    return LoopOutcome(
        answer=None,
        transcript="\n".join(transcript_parts),
        halted=True,
        n_steps=n_steps,
    )


async def run_react(ctx: RunContext) -> RunResult:
    """The `react` scaffold (and one pool sample)."""
    started = time.perf_counter()
    outcome = await react_loop(
        ctx,
        system=lib.react_system(ctx.benchmark),
        system_prompt_id=lib.system_prompt_id("react", ctx.benchmark),
        first_user_msg=f"Question: {ctx.question}",
        conversation_label="react",
        max_steps=config.REACT_MAX_STEPS,
        role="actor",
    )
    return finalize(
        ctx,
        outcome.answer or "",
        halted=outcome.halted,
        n_steps=outcome.n_steps,
        extra={"transcript": outcome.transcript},
        started=started,
    )
