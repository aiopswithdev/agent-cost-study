"""T2 — dollar cost of a logged LLM call under the three pricing regimes (§2.6).

A *call* here is any mapping with the four usage fields of `calls.jsonl` (§4.1);
that covers dicts, pandas Series and ``LLMResult.usage``.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from . import config

_M = 1_000_000.0


def _u(call: Mapping[str, Any], field: str) -> int:
    value = call.get(field, 0)
    if value is None:
        return 0
    return int(value)


def total_input(call: Mapping[str, Any]) -> int:
    """``I`` = input + cache-creation + cache-read tokens (§2.6)."""
    return (
        _u(call, "input_tokens")
        + _u(call, "cache_creation_input_tokens")
        + _u(call, "cache_read_input_tokens")
    )


def total_output(call: Mapping[str, Any]) -> int:
    """``O`` = output tokens."""
    return _u(call, "output_tokens")


def usd_nocache(call: Mapping[str, Any]) -> float:
    """Regime 1: every input token billed at full price."""
    return (
        total_input(call) * config.PRICE_INPUT_PER_MTOK
        + total_output(call) * config.PRICE_OUTPUT_PER_MTOK
    ) / _M


def usd_empirical(call: Mapping[str, Any]) -> float:
    """Regime 2: what the API actually charged, given our cache breakpoints."""
    return (
        _u(call, "input_tokens") * config.PRICE_INPUT_PER_MTOK
        + _u(call, "cache_creation_input_tokens") * config.PRICE_CACHE_WRITE_PER_MTOK
        + _u(call, "cache_read_input_tokens") * config.PRICE_CACHE_READ_PER_MTOK
        + total_output(call) * config.PRICE_OUTPUT_PER_MTOK
    ) / _M


def usd_from_split(
    *, uncached: int, write: int, read: int, output: int
) -> float:
    """Price an already-decomposed call. Shared by `usd_empirical` and `usd_ideal`."""
    return (
        uncached * config.PRICE_INPUT_PER_MTOK
        + write * config.PRICE_CACHE_WRITE_PER_MTOK
        + read * config.PRICE_CACHE_READ_PER_MTOK
        + output * config.PRICE_OUTPUT_PER_MTOK
    ) / _M


def usd_ideal(
    call_rows: Sequence[Mapping[str, Any]] | Iterable[Mapping[str, Any]],
    L: int,
    prompt_tokens: Mapping[str, int] | None = None,
) -> list[float]:
    """Regime 3: schedule-independent perfect-prefix-caching counterfactual (T14.2).

    Implemented in `acs.analysis.costs`; re-exported here so `pricing` stays the one
    place that knows about money. Returns one USD figure per input row, in input order.
    """
    from .analysis.costs import usd_ideal as _impl

    return _impl(call_rows, L, prompt_tokens=prompt_tokens)
