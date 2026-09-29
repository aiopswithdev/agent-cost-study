from datetime import datetime, timezone

import pytest

from acs import config
from acs.pricing import (
    billing_multiplier,
    is_deepseek_peak,
    prices_for,
    total_input,
    total_output,
    usd_billed,
    usd_empirical,
    usd_nocache,
)

HAIKU = "claude-haiku-4-5"
DEEPSEEK = "deepseek-flash"


def test_usd_nocache_accept_haiku():
    call = {"model": HAIKU, "input_tokens": 10_000, "output_tokens": 1_000}
    assert usd_nocache(call) == 0.015


def test_usd_empirical_accept_haiku():
    call = {"model": HAIKU, "input_tokens": 2_000, "cache_read_input_tokens": 8_000,
            "output_tokens": 1_000}
    assert abs(usd_empirical(call) - 0.0078) < 1e-12


def test_usd_deepseek_list_prices():
    # 2,000 miss + 8,000 hit input, 1,000 output at deepseek-flash list (peak) prices.
    call = {"model": DEEPSEEK, "input_tokens": 2_000, "cache_read_input_tokens": 8_000,
            "output_tokens": 1_000}
    assert abs(usd_nocache(call) - (10_000 * 0.30 + 1_000 * 1.20) / 1e6) < 1e-12
    assert abs(usd_empirical(call) - (2_000 * 0.30 + 8_000 * 0.006 + 1_000 * 1.20) / 1e6) < 1e-12


def test_total_input_sums_the_three_input_fields():
    call = {
        "input_tokens": 1,
        "cache_creation_input_tokens": 2,
        "cache_read_input_tokens": 4,
        "output_tokens": 8,
    }
    assert total_input(call) == 7
    assert total_output(call) == 8


@pytest.mark.parametrize("model", [HAIKU, DEEPSEEK])
def test_nocache_is_never_cheaper_than_empirical(model):
    call = {"model": model, "input_tokens": 100, "cache_creation_input_tokens": 500,
            "cache_read_input_tokens": 9_000, "output_tokens": 50}
    assert usd_nocache(call) > usd_empirical(call)


def test_price_sheets_are_the_documented_ones():
    h, d = config.MODEL_PROFILES[HAIKU], config.MODEL_PROFILES[DEEPSEEK]
    assert (h["input"], h["output"], h["cache_write"], h["cache_read"]) == (1.00, 5.00, 1.25, 0.10)
    assert config.CACHE_MIN_TOKENS[HAIKU] == 4096
    assert (d["input"], d["output"], d["cache_write"], d["cache_read"]) == (0.30, 1.20, 0.30, 0.006)


def test_rows_without_a_model_use_the_active_model():
    assert prices_for({}) is config.MODEL_PROFILES[config.MODEL]
    assert prices_for({"model": float("nan")}) is config.MODEL_PROFILES[config.MODEL]


def test_unknown_model_is_an_error():
    with pytest.raises(KeyError):
        usd_nocache({"model": "no-such-model", "input_tokens": 1})


@pytest.mark.parametrize(
    "when, peak",
    [
        ("2026-09-29T02:30:00+00:00", True),   # Tue 02:30 UTC
        ("2026-09-29T07:00:00+00:00", True),   # Tue 07:00 UTC
        ("2026-09-29T04:00:00+00:00", False),  # Tue 04:00 UTC: first off-peak hour
        ("2026-09-29T10:00:00+00:00", False),  # Tue 10:00 UTC
        ("2026-09-29T15:00:00+00:00", False),  # Tue afternoon UTC
        ("2026-10-03T02:30:00+00:00", False),  # Saturday
        ("2026-09-29T08:00:00+05:30", True),   # 02:30 UTC expressed in IST
    ],
)
def test_deepseek_peak_hours(when, peak):
    assert is_deepseek_peak(datetime.fromisoformat(when)) is peak


def test_billed_applies_the_offpeak_discount_to_deepseek_only():
    base = {"input_tokens": 10_000, "output_tokens": 1_000}
    ds_peak = {**base, "model": DEEPSEEK, "ts": "2026-09-29T02:30:00+00:00"}
    ds_off = {**base, "model": DEEPSEEK, "ts": "2026-09-29T15:00:00+00:00"}
    haiku_off = {**base, "model": HAIKU, "ts": "2026-09-29T15:00:00+00:00"}
    assert usd_billed(ds_peak) == usd_empirical(ds_peak)
    assert abs(usd_billed(ds_off) - 0.5 * usd_empirical(ds_off)) < 1e-15
    assert billing_multiplier(haiku_off) == 1.0
    # An unknown time is assumed to be peak, which can only over-estimate the bill.
    assert billing_multiplier({**base, "model": DEEPSEEK}) == 1.0
    assert datetime(2026, 9, 29, tzinfo=timezone.utc).weekday() == 1  # sanity: Tuesday
