"""T2 — dollar cost of a logged LLM call under the pricing regimes of §2.6.

A *call* here is any mapping with the four usage fields of `calls.jsonl` (§4.1);
that covers dicts, pandas Series and ``LLMResult.usage``. Every call is priced by its own
``model`` field (falling back to the active ``config.MODEL``), so a log is always priced
with the price sheet of the model that produced it.

All regimes use the model's *list* prices. `usd_billed` is the only figure that applies
DeepSeek's time-of-day discount; it is what the account is actually charged and is what
the spend guard watches. Analysis ratios use list prices, which the discount cannot change
because it halves every price line alike.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from . import config

_M = 1_000_000.0


def _u(call: Mapping[str, Any], field: str) -> int:
    value = call.get(field, 0)
    if value is None or value != value:  # None or NaN
        return 0
    return int(value)


def prices_for(call: Mapping[str, Any] | None = None) -> dict[str, float]:
    """The price sheet of the model that made `call` (or of the active model)."""
    model = call.get("model") if call is not None else None
    if not isinstance(model, str) or not model:  # absent, None or NaN
        model = config.MODEL
    if model not in config.MODEL_PROFILES:
        raise KeyError(f"no price sheet for model {model!r}; add it to MODEL_PROFILES")
    return config.MODEL_PROFILES[model]


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
    p = prices_for(call)
    return (total_input(call) * p["input"] + total_output(call) * p["output"]) / _M


def usd_empirical(call: Mapping[str, Any]) -> float:
    """Regime 2: what the API charged for the cache hits it reported, at list price."""
    p = prices_for(call)
    return (
        _u(call, "input_tokens") * p["input"]
        + _u(call, "cache_creation_input_tokens") * p["cache_write"]
        + _u(call, "cache_read_input_tokens") * p["cache_read"]
        + total_output(call) * p["output"]
    ) / _M


def usd_from_split(
    *, uncached: int, write: int, read: int, output: int, model: str | None = None
) -> float:
    """Price an already-decomposed call. Shared by `usd_empirical` and `usd_ideal`."""
    p = prices_for({"model": model})
    return (
        uncached * p["input"]
        + write * p["cache_write"]
        + read * p["cache_read"]
        + output * p["output"]
    ) / _M


# --------------------------------------------------------------------------- the actual bill


def is_deepseek_peak(ts: datetime) -> bool:
    """DeepSeek peak = Mon-Fri inside `DEEPSEEK_PEAK_HOURS_UTC`; everything else off-peak."""
    ts = ts.astimezone(timezone.utc)
    if ts.weekday() >= 5:
        return False
    return any(lo <= ts.hour < hi for lo, hi in config.DEEPSEEK_PEAK_HOURS_UTC)


def billing_multiplier(call: Mapping[str, Any]) -> float:
    """Fraction of list price actually charged for `call` (1.0 unless DeepSeek off-peak)."""
    if prices_for(call)["provider"] != "deepseek":
        return 1.0
    ts = call.get("ts")
    if not ts:
        return 1.0  # unknown time: assume peak, which can only over-estimate
    when = datetime.fromisoformat(str(ts))
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return 1.0 if is_deepseek_peak(when) else config.DEEPSEEK_OFFPEAK_MULTIPLIER


def usd_billed(call: Mapping[str, Any]) -> float:
    """What the account is charged: `usd_empirical` at the call's time-of-day rate."""
    return usd_empirical(call) * billing_multiplier(call)


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
