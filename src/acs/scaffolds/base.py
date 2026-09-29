"""T7 — the run context, the run record, and the action parser shared by all scaffolds."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .. import config
from ..data import Task
from ..llm import BaseLLM, LLMResult
from ..pricing import usd_empirical, usd_nocache

if TYPE_CHECKING:  # pragma: no cover
    from ..tools.toolbox import ToolBox

#: Canonical spellings, so a lower/upper-cased tool name still dispatches.
_CANONICAL = {"search": "Search", "lookup": "Lookup", "calculate": "Calculate",
              "finish": "Finish", "llm": "LLM"}

_ACTION_START = re.compile(r"Action\s*\d*\s*:\s*(\w+)\[")
_ACTION_LINE = re.compile(r"Action\s*\d*\s*:\s*\w+\[.*\]", re.M)
_FINISH_START = re.compile(r"Finish\[")


def make_run_id(benchmark: str, task_id: str, scaffold: str, replicate: int) -> str:
    return f"{benchmark}:{task_id}:{scaffold}:{replicate}"


def parse_action(text: str) -> tuple[str, str] | None:
    """First `Action N: Tool[arg]` in `text`, or None.

    The argument runs from the first `[` after the tool name to the **last** `]` on that
    same line, so answers containing brackets survive (`Finish[Arthur's Magazine [1844]]`).
    """
    m = _ACTION_START.search(text or "")
    if m is None:
        return None
    tool = m.group(1).strip().title()
    tool = _CANONICAL.get(tool.lower(), tool)
    line = (text[m.end():].split("\n", 1)[0])
    close = line.rfind("]")
    if close == -1:
        return None
    return tool, line[:close]


def extract_finish(text: str) -> str | None:
    """`Finish[answer]` anywhere in `text` (no `Action:` prefix needed).

    The solver is asked for a bare answer but sometimes wraps it in the actor's syntax.
    """
    m = _FINISH_START.search(text or "")
    if m is None:
        return None
    line = text[m.end():].split("\n", 1)[0]
    close = line.rfind("]")
    if close == -1:
        return None
    return line[:close]


def truncate_after_first_action(text: str) -> str:
    """Keep only up to and including the first complete `Action …]` line.

    The model sometimes hallucinates several steps at once; what we store is exactly what
    we append to the conversation, which is what keeps conversations append-only (§4.1).
    """
    m = _ACTION_LINE.search(text or "")
    if m is None:
        return text or ""
    return text[: m.end()]


@dataclass
class RunContext:
    """One scaffold run on one task for one replicate."""

    run_id: str
    benchmark: str
    task: Task
    scaffold: str
    replicate: int
    llm: BaseLLM
    toolbox: "ToolBox"
    trace: list[dict] = field(default_factory=list)
    counters: dict[str, Any] = field(
        default_factory=lambda: {
            "n_llm_calls": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "usd_nocache": 0.0,
            "usd_empirical": 0.0,
            "n_steps": 0,
        }
    )

    @property
    def task_id(self) -> str:
        return self.task.task_id

    @property
    def question(self) -> str:
        return self.task.question

    @property
    def gold(self) -> str:
        return self.task.answer

    async def call(
        self,
        *,
        role: str,
        conversation_id: str,
        turn_index: int,
        system_prompt_id: str,
        system: str,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: list[str] | None = None,
    ) -> LLMResult:
        """The one path to the LLM: applies the frozen per-role settings and books the cost."""
        if max_tokens is None:
            max_tokens = config.MAX_TOKENS[role]
        if temperature is None:
            temperature = config.TEMPERATURE_JUDGE if role == "judge" else config.TEMPERATURE

        res = await self.llm.call(
            ctx=self,
            role=role,
            conversation_id=conversation_id,
            turn_index=turn_index,
            system_prompt_id=system_prompt_id,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )

        self.counters["n_llm_calls"] += 1
        self.counters["tokens_in"] += res.tokens_in
        self.counters["tokens_out"] += res.tokens_out
        self.counters["usd_nocache"] += usd_nocache(res.usage)
        self.counters["usd_empirical"] += usd_empirical(res.usage)
        self.trace.append(
            {"kind": "llm", "role": role, "call_id": res.call_id, "text": res.text}
        )
        return res


@dataclass
class RunResult:
    """One line of `runs.jsonl` (§4.2)."""

    run_id: str
    benchmark: str
    task_id: str
    scaffold: str
    replicate: int
    status: str = "ok"
    error: str | None = None
    final_answer: str = ""
    gold: str = ""
    em: int = 0
    f1: float = 0.0
    correct: int = 0
    n_llm_calls: int = 0
    n_tool_calls: dict[str, int] = field(
        default_factory=lambda: {"Search": 0, "Lookup": 0, "Calculate": 0, "LLM": 0}
    )
    n_steps: int = 0
    halted: bool = False
    tokens_in: int = 0
    tokens_out: int = 0
    usd_nocache: float = 0.0
    usd_empirical: float = 0.0
    wall_s: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)
    trace: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "benchmark": self.benchmark,
            "task_id": self.task_id,
            "scaffold": self.scaffold,
            "replicate": self.replicate,
            "status": self.status,
            "error": self.error,
            "final_answer": self.final_answer,
            "gold": self.gold,
            "em": self.em,
            "f1": self.f1,
            "correct": self.correct,
            "n_llm_calls": self.n_llm_calls,
            "n_tool_calls": self.n_tool_calls,
            "n_steps": self.n_steps,
            "halted": self.halted,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "usd_nocache": self.usd_nocache,
            "usd_empirical": self.usd_empirical,
            "wall_s": self.wall_s,
            "extra": self.extra,
            "trace": self.trace,
        }


def finalize(
    ctx: RunContext,
    final_answer: str,
    *,
    halted: bool,
    n_steps: int,
    extra: dict | None = None,
    started: float | None = None,
) -> RunResult:
    """Build the `RunResult` from the context's counters; grading is added by the runner."""
    c = ctx.counters
    return RunResult(
        run_id=ctx.run_id,
        benchmark=ctx.benchmark,
        task_id=ctx.task_id,
        scaffold=ctx.scaffold,
        replicate=ctx.replicate,
        final_answer=final_answer,
        gold=ctx.gold,
        n_llm_calls=c["n_llm_calls"],
        n_tool_calls=dict(ctx.toolbox.n_tool_calls),
        n_steps=n_steps,
        halted=halted,
        tokens_in=c["tokens_in"],
        tokens_out=c["tokens_out"],
        usd_nocache=c["usd_nocache"],
        usd_empirical=c["usd_empirical"],
        wall_s=(time.perf_counter() - started) if started is not None else 0.0,
        extra=extra or {},
        trace=list(ctx.trace),
    )
