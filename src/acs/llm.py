"""T3 — the single gateway for every LLM call in the study.

Every scaffold calls `BaseLLM.call`; nothing else talks to a model API. That is what
makes the token/dollar accounting complete and identical across scaffolds. The provider
(DeepSeek by default, Anthropic optionally) is picked by `make_llm` from `config.PROVIDER`.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

from . import config
from .pricing import usd_billed, usd_empirical, usd_nocache

if TYPE_CHECKING:  # pragma: no cover
    from .scaffolds.base import RunContext

USAGE_FIELDS = (
    "input_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "output_tokens",
)

EMPTY_TEXT_PLACEHOLDER = "(no output)"


class SpendLimitExceeded(RuntimeError):
    """Raised when the billed spend passes `MAX_USD`, or the provider reports that the
    account is out of credit or the key is invalid (nothing further can succeed)."""


@dataclass
class LLMResult:
    text: str
    stop_reason: str
    usage: dict[str, int]
    call_id: str
    latency_s: float = 0.0
    attempts: int = 1

    @property
    def tokens_in(self) -> int:
        return sum(self.usage.get(f, 0) or 0 for f in USAGE_FIELDS[:3])

    @property
    def tokens_out(self) -> int:
        return self.usage.get("output_tokens", 0) or 0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _zero_usage() -> dict[str, int]:
    return dict.fromkeys(USAGE_FIELDS, 0)


# --------------------------------------------------------------------------- request building


def build_request(system: str, messages: Sequence[dict]) -> tuple[list[dict], list[dict]]:
    """Return `(system_blocks, messages)` with cache breakpoints placed per §T3.2.

    Exactly two ephemeral breakpoints: the system block, and the last block of the last
    message. Earlier messages keep plain-string content so breakpoints never accumulate
    (the API allows at most four).
    """
    system_blocks = [
        {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
    ]
    msgs = copy.deepcopy(list(messages))
    if not msgs:
        raise ValueError("messages must be non-empty and must start and end with a user turn")

    for i, m in enumerate(msgs):
        content = m.get("content")
        if isinstance(content, list):
            for block in content:
                block.pop("cache_control", None)
        else:
            m["content"] = "" if content is None else str(content)
        if i == len(msgs) - 1:
            text = m["content"] if isinstance(m["content"], str) else None
            if text is not None:
                m["content"] = [{"type": "text", "text": text}]
            m["content"][-1]["cache_control"] = {"type": "ephemeral"}
    return system_blocks, msgs


def validate_messages(messages: Sequence[dict]) -> None:
    """Alternating user/assistant, starts with user, ends with user, no empty content."""
    if not messages:
        raise ValueError("empty message list")
    if messages[0]["role"] != "user":
        raise ValueError("conversation must start with a user message")
    if messages[-1]["role"] != "user":
        raise ValueError("conversation must end with a user message (no assistant prefill)")
    for i, m in enumerate(messages):
        expected = "user" if i % 2 == 0 else "assistant"
        if m["role"] != expected:
            raise ValueError(f"messages must alternate user/assistant; index {i} is {m['role']}")
        content = m["content"]
        text = content if isinstance(content, str) else "".join(
            b.get("text", "") for b in content
        )
        if not text.strip():
            raise ValueError(
                f"message {i} is empty; callers must substitute {EMPTY_TEXT_PLACEHOLDER!r}"
            )


# --------------------------------------------------------------------------- call log


class CallLog:
    """Append-only `calls.jsonl` writer plus the running spend guard (§T3.6–7)."""

    def __init__(self, path: Path | None, max_usd: float = config.MAX_USD) -> None:
        self.path = Path(path) if path is not None else None
        self.max_usd = max_usd
        self.total_usd_nocache = 0.0
        self.total_usd_empirical = 0.0
        self.total_usd_billed = 0.0
        self.n_calls = 0
        self.stop = False
        self._lock = asyncio.Lock()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._resume()

    def _resume(self) -> None:
        """Re-initialise the running spend from an existing log, so resumes stay bounded."""
        if self.path is None or not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._add(row)
        self._check_guard()

    def _add(self, row: dict) -> None:
        self.total_usd_nocache += usd_nocache(row)
        self.total_usd_empirical += usd_empirical(row)
        self.total_usd_billed += usd_billed(row)
        self.n_calls += 1

    def _check_guard(self) -> None:
        # The guard watches what the account is actually charged (time-of-day rate).
        if self.total_usd_billed >= self.max_usd:
            self.stop = True

    async def write(self, row: dict) -> None:
        async with self._lock:
            self._add(row)
            self._check_guard()
            if self.path is None:
                return
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())


# --------------------------------------------------------------------------- the wrapper


class BaseLLM:
    """Shared logging, validation and spend-guard behaviour."""

    def __init__(self, call_log: CallLog | None = None, model: str = config.MODEL) -> None:
        self.call_log = call_log if call_log is not None else CallLog(None)
        self.model = model

    @property
    def stopped(self) -> bool:
        return self.call_log.stop

    async def call(
        self,
        *,
        ctx: "RunContext | None" = None,
        role: str,
        conversation_id: str,
        turn_index: int,
        system_prompt_id: str,
        system: str,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        stop: list[str] | None = None,
    ) -> LLMResult:
        validate_messages(messages)
        if self.call_log.stop:
            raise SpendLimitExceeded(
                f"billed spend reached MAX_USD={self.call_log.max_usd}, "
                "or the provider rejected the key / reported no credit"
            )

        call_id = str(uuid.uuid4())
        system_blocks, api_messages = build_request(system, messages)
        started = time.perf_counter()
        error: str | None = None
        attempts = 1
        try:
            text, stop_reason, usage, attempts = await self._invoke(
                system_blocks=system_blocks,
                messages=api_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                role=role,
            )
        except Exception as exc:  # logged, then re-raised (§T3.6)
            error = f"{type(exc).__name__}: {exc}"
            text, stop_reason, usage = "", "error", _zero_usage()
            raise
        finally:
            latency = time.perf_counter() - started
            await self.call_log.write(
                self._row(
                    call_id=call_id,
                    ctx=ctx,
                    role=role,
                    conversation_id=conversation_id,
                    turn_index=turn_index,
                    system_prompt_id=system_prompt_id,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    usage=usage,
                    stop_reason=stop_reason,
                    latency_s=latency,
                    attempts=attempts,
                    error=error,
                )
            )

        return LLMResult(
            text=text,
            stop_reason=stop_reason,
            usage=usage,
            call_id=call_id,
            latency_s=latency,
            attempts=attempts,
        )

    def _row(
        self,
        *,
        call_id: str,
        ctx: "RunContext | None",
        role: str,
        conversation_id: str,
        turn_index: int,
        system_prompt_id: str,
        max_tokens: int,
        temperature: float,
        usage: dict[str, int],
        stop_reason: str,
        latency_s: float,
        attempts: int,
        error: str | None,
    ) -> dict:
        return {
            "call_id": call_id,
            "run_id": getattr(ctx, "run_id", None),
            "benchmark": getattr(ctx, "benchmark", None),
            "task_id": getattr(ctx, "task_id", None),
            "scaffold": getattr(ctx, "scaffold", None),
            "replicate": getattr(ctx, "replicate", None),
            "attempt_id": getattr(ctx, "attempt_id", None),
            "role": role,
            "conversation_id": conversation_id,
            "turn_index": turn_index,
            "system_prompt_id": system_prompt_id,
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **{k: int(usage.get(k, 0) or 0) for k in USAGE_FIELDS},
            "stop_reason": stop_reason,
            "latency_s": round(latency_s, 4),
            "ts": _now_iso(),
            "attempts": attempts,
            "error": error,
        }

    async def _invoke(self, **kwargs) -> tuple[str, str, dict[str, int], int]:
        raise NotImplementedError

    async def count_tokens(self, system: str, messages: list[dict]) -> int:
        raise NotImplementedError


class LLM(BaseLLM):
    """The Anthropic client (`ACS_PROVIDER=anthropic`)."""

    def __init__(
        self,
        call_log: CallLog | None = None,
        model: str = config.MODEL,
        concurrency: int = config.LLM_CONCURRENCY,
        api_key: str | None = None,
    ) -> None:
        super().__init__(call_log=call_log, model=model)
        import anthropic

        self._client = anthropic.AsyncAnthropic(
            api_key=api_key or config.api_key("anthropic"),
            max_retries=config.LLM_MAX_RETRIES,
            timeout=config.LLM_TIMEOUT_S,
        )
        self._sem = asyncio.Semaphore(concurrency)

    async def _invoke(
        self,
        *,
        system_blocks: list[dict],
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        stop: list[str] | None,
        role: str,
    ) -> tuple[str, str, dict[str, int], int]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_blocks,
            "messages": messages,
        }
        if stop:
            kwargs["stop_sequences"] = list(stop)

        async with self._sem:
            resp = await self._client.messages.create(**kwargs)

        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        usage = {
            "input_tokens": resp.usage.input_tokens or 0,
            "cache_creation_input_tokens": getattr(
                resp.usage, "cache_creation_input_tokens", 0
            )
            or 0,
            "cache_read_input_tokens": getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
            "output_tokens": resp.usage.output_tokens or 0,
        }
        return text, resp.stop_reason or "end_turn", usage, 1

    async def count_tokens(self, system: str, messages: list[dict]) -> int:
        """Used once per distinct system prompt for `prompt_tokens.json` (§4.3)."""
        async with self._sem:
            resp = await self._client.messages.count_tokens(
                model=self.model,
                system=system,
                messages=messages or [{"role": "user", "content": "x"}],
            )
        return int(resp.input_tokens)

    async def aclose(self) -> None:
        await self._client.close()


AnthropicLLM = LLM


class ThinkingModeError(RuntimeError):
    """DeepSeek returned reasoning output although thinking was disabled on the request.

    Hidden reasoning tokens would be billed as output and change the Thought/Action format,
    which would invalidate the cost comparison, so the call fails loudly instead."""


def to_openai_messages(system_blocks: Sequence[dict], messages: Sequence[dict]) -> list[dict]:
    """Anthropic-shaped request -> OpenAI chat messages. `cache_control` markers are dropped:
    DeepSeek caches every identical prefix automatically, so none are needed."""
    out = [{"role": "system", "content": "".join(b["text"] for b in system_blocks)}]
    for m in messages:
        content = m["content"]
        text = content if isinstance(content, str) else "".join(
            b.get("text", "") for b in content
        )
        out.append({"role": m["role"], "content": text})
    return out


def usage_from_openai(usage: Any) -> dict[str, int]:
    """Map DeepSeek usage onto the four fields of `calls.jsonl` (§4.1).

    cache miss -> `input_tokens`, cache hit -> `cache_read_input_tokens`. DeepSeek has no
    cache-write charge, so `cache_creation_input_tokens` is always 0.
    """
    if usage is None:
        return _zero_usage()
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    hit = getattr(usage, "prompt_cache_hit_tokens", None)
    if hit is None:
        details = getattr(usage, "prompt_tokens_details", None)
        hit = getattr(details, "cached_tokens", 0) if details is not None else 0
    hit = int(hit or 0)
    miss = getattr(usage, "prompt_cache_miss_tokens", None)
    miss = int(miss) if miss is not None else prompt - hit
    if prompt and hit + miss != prompt:  # never let the fields disagree with the total
        miss = prompt - hit
    return {
        "input_tokens": max(miss, 0),
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": hit,
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
    }


def reasoning_tokens(usage: Any) -> int:
    details = getattr(usage, "completion_tokens_details", None) if usage is not None else None
    return int(getattr(details, "reasoning_tokens", 0) or 0) if details is not None else 0


_FINISH_TO_STOP_REASON = {"stop": "end_turn", "length": "max_tokens"}

#: Sent on every request. DeepSeek's thinking mode is ON by default.
DEEPSEEK_EXTRA_BODY = {"thinking": {"type": "disabled"}}


class DeepSeekLLM(BaseLLM):
    """The DeepSeek client (`ACS_PROVIDER=deepseek`, the default). OpenAI-compatible API,
    thinking mode disabled on every request."""

    def __init__(
        self,
        call_log: CallLog | None = None,
        model: str | None = None,
        concurrency: int = config.LLM_CONCURRENCY,
        api_key: str | None = None,
    ) -> None:
        model = model or (
            config.MODEL if config.PROVIDER == "deepseek" else config.DEFAULT_MODEL["deepseek"]
        )
        super().__init__(call_log=call_log, model=model)
        import openai

        self._openai = openai
        self._client = openai.AsyncOpenAI(
            api_key=api_key or config.api_key("deepseek"),
            base_url=config.DEEPSEEK_BASE_URL,
            max_retries=config.LLM_MAX_RETRIES,
            timeout=config.LLM_TIMEOUT_S,
        )
        self._sem = asyncio.Semaphore(concurrency)
        self._template_overhead: int | None = None

    async def _create(self, **kwargs: Any) -> Any:
        async with self._sem:
            try:
                return await self._client.chat.completions.create(
                    model=self.model, extra_body=DEEPSEEK_EXTRA_BODY, **kwargs
                )
            except self._openai.APIStatusError as exc:
                if exc.status_code in (401, 402):
                    # Bad key or no credit: every further call would fail the same way.
                    self.call_log.stop = True
                    what = "invalid API key" if exc.status_code == 401 else "insufficient balance"
                    raise SpendLimitExceeded(
                        f"DeepSeek HTTP {exc.status_code} ({what}): {exc}"
                    ) from exc
                raise

    async def _invoke(
        self,
        *,
        system_blocks: list[dict],
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        stop: list[str] | None,
        role: str,
    ) -> tuple[str, str, dict[str, int], int]:
        kwargs: dict[str, Any] = {
            "messages": to_openai_messages(system_blocks, messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if stop:
            kwargs["stop"] = list(stop)
        resp = await self._create(**kwargs)

        choice = resp.choices[0]
        if getattr(choice.message, "reasoning_content", None) or reasoning_tokens(resp.usage):
            raise ThinkingModeError(
                "DeepSeek returned reasoning output with thinking disabled; refusing to log "
                "a call whose cost includes hidden reasoning tokens"
            )
        text = choice.message.content or ""
        stop_reason = _FINISH_TO_STOP_REASON.get(choice.finish_reason, choice.finish_reason)
        return text, stop_reason or "end_turn", usage_from_openai(resp.usage), 1

    async def _prompt_tokens(self, system: str, user: str) -> int:
        resp = await self._create(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=1,
            temperature=0.0,
        )
        return int(resp.usage.prompt_tokens)

    async def count_tokens(self, system: str, messages: list[dict]) -> int:
        """System-prompt tokens for `prompt_tokens.json` (§4.3).

        DeepSeek has no token-counting endpoint, so this makes one 1-token call per system
        prompt and subtracts the chat-template overhead measured once with a 1-character
        system prompt. These few calls cost well under a cent in total and are not logged
        to `calls.jsonl`, which holds study calls only.
        """
        user = "x"
        if self._template_overhead is None:
            self._template_overhead = await self._prompt_tokens(".", user) - 1
        return max(0, await self._prompt_tokens(system, user) - self._template_overhead)

    async def aclose(self) -> None:
        await self._client.close()


def make_llm(
    call_log: CallLog | None = None,
    concurrency: int = config.LLM_CONCURRENCY,
    provider: str | None = None,
) -> BaseLLM:
    """The real client for `config.PROVIDER` (or `provider`)."""
    provider = provider or config.PROVIDER
    if provider == "deepseek":
        return DeepSeekLLM(call_log=call_log, concurrency=concurrency)
    if provider == "anthropic":
        return LLM(call_log=call_log, concurrency=concurrency,
                   model=config.MODEL if config.PROVIDER == "anthropic"
                   else config.DEFAULT_MODEL["anthropic"])
    raise ValueError(f"unknown provider {provider!r}; expected one of {list(config.API_KEY_ENV)}")


# --------------------------------------------------------------------------- test double


class FakeLLM(BaseLLM):
    """Scripted stand-in with the identical interface. Every scaffold test uses it."""

    def __init__(
        self,
        responses: Sequence[str] | Callable[[str, list[dict]], str] | None = None,
        call_log: CallLog | None = None,
        model: str = config.MODEL,
        stop_reason: str = "end_turn",
    ) -> None:
        super().__init__(call_log=call_log, model=model)
        self._script = list(responses) if isinstance(responses, (list, tuple)) else responses
        self._i = 0
        self._stop_reason = stop_reason
        self.requests: list[dict] = []

    def _next_text(self, role: str, messages: list[dict]) -> str:
        if self._script is None:
            return "Thought 1: done.\nAction 1: Finish[unknown]"
        if callable(self._script):
            return self._script(role, messages)
        if self._i >= len(self._script):
            raise AssertionError(
                f"FakeLLM script exhausted after {self._i} responses (role={role})"
            )
        text = self._script[self._i]
        self._i += 1
        return text

    async def _invoke(
        self,
        *,
        system_blocks: list[dict],
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        stop: list[str] | None,
        role: str,
    ) -> tuple[str, str, dict[str, int], int]:
        self.requests.append(
            {
                "system": system_blocks,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": stop,
                "role": role,
            }
        )
        text = self._next_text(role, messages)
        if stop:
            for s in stop:
                idx = text.find(s)
                if idx != -1:
                    text = text[:idx]
        prompt_chars = sum(len(b["text"]) for b in system_blocks)
        for m in messages:
            content = m["content"]
            prompt_chars += (
                len(content)
                if isinstance(content, str)
                else sum(len(b.get("text", "")) for b in content)
            )
        usage = {
            "input_tokens": prompt_chars // 4,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "output_tokens": len(text) // 4,
        }
        return text, self._stop_reason, usage, 1

    async def count_tokens(self, system: str, messages: list[dict]) -> int:
        return len(system) // 4
