from acs import config
import pandas as pd
import pytest

from acs.analysis.costs import add_ideal_regime, ideal_col, run_costs
from acs.pricing import usd_ideal


def calls(rows: list[dict]) -> pd.DataFrame:
    base = dict(
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        output_tokens=0,
        system_prompt_id="p1",
        conversation_id="c1",
    )
    out = []
    for i, r in enumerate(rows):
        row = {**base, **r}
        row.setdefault("call_id", f"call{i}")
        row.setdefault("ts", f"2026-01-01T00:00:{i:02d}")
        row.setdefault("turn_index", i)
        row.setdefault("input_tokens", row.pop("I_tokens", 0))
        out.append(row)
    return pd.DataFrame(out)


def test_three_turn_conversation_reads_the_previous_turns_prefix():
    df = calls([{"input_tokens": 5000}, {"input_tokens": 5600}, {"input_tokens": 6200}])
    out = add_ideal_regime(df, 4096, {"p1": 100})
    assert out["ideal_4096_read"].tolist() == [0, 5000, 5600]
    assert out["ideal_4096_write"].tolist() == [5000, 600, 600]
    assert out["ideal_4096_uncached"].tolist() == [0, 0, 0]


def test_no_caching_when_every_prompt_is_below_L():
    df = calls([{"input_tokens": 5000}, {"input_tokens": 5600}, {"input_tokens": 6200}])
    out = add_ideal_regime(df, 8000, {"p1": 100})
    assert out["ideal_8000_read"].tolist() == [0, 0, 0]
    assert out["ideal_8000_write"].tolist() == [0, 0, 0]
    assert out["ideal_8000_uncached"].tolist() == [5000, 5600, 6200]


def test_system_prompt_breakpoint_is_shared_across_tasks():
    df = calls(
        [
            {"input_tokens": 5200, "conversation_id": "task1", "turn_index": 0},
            {"input_tokens": 5300, "conversation_id": "task2", "turn_index": 0},
        ]
    )
    out = add_ideal_regime(df, 4096, {"p1": 5000})
    # The first call for this system prompt pays the write; the second reads it.
    assert out["ideal_4096_read"].tolist() == [0, 5000]
    assert out["ideal_4096_write"].tolist() == [5200, 300]


def test_system_prompt_shorter_than_L_never_caches_across_tasks():
    df = calls(
        [
            {"input_tokens": 5200, "conversation_id": "t1", "turn_index": 0},
            {"input_tokens": 5300, "conversation_id": "t2", "turn_index": 0},
        ]
    )
    out = add_ideal_regime(df, 4096, {"p1": 1000})
    assert out["ideal_4096_read"].tolist() == [0, 0]


def test_L_zero_caches_everything_after_the_first_call():
    df = calls([{"input_tokens": 100}, {"input_tokens": 200}, {"input_tokens": 300}])
    out = add_ideal_regime(df, 0, {"p1": 50})
    assert out["ideal_0_read"].tolist() == [0, 100, 200]
    assert out["ideal_0_uncached"].tolist() == [0, 0, 0]


def test_ideal_price_matches_the_regime_formula():
    df = calls([{"input_tokens": 5000, "output_tokens": 100}])
    out = add_ideal_regime(df, 4096, {"p1": 100})
    p = config.MODEL_PROFILES[config.MODEL]
    expected = (0 * p["input"] + 5000 * p["cache_write"] + 0 * p["cache_read"]
                + 100 * p["output"]) / 1e6
    assert abs(out[ideal_col(4096)].iloc[0] - expected) < 1e-15


def test_ideal_prices_each_row_with_its_own_model():
    """A log is priced with the sheet of the model that produced it, whatever is active."""
    df = calls([{"input_tokens": 5000, "output_tokens": 100}])
    df["model"] = "claude-haiku-4-5"
    out = add_ideal_regime(df, 4096, {"p1": 100})
    expected = (5000 * 1.25 + 100 * 5.0) / 1e6
    assert abs(out[ideal_col(4096)].iloc[0] - expected) < 1e-15


def test_pricing_module_reexports_usd_ideal():
    rows = calls([{"input_tokens": 5000}, {"input_tokens": 5600}]).to_dict("records")
    values = usd_ideal(rows, 4096, {"p1": 100})
    assert len(values) == 2 and all(v > 0 for v in values)


def test_run_costs_bills_reflexion_for_its_trial1_calls():
    df = calls(
        [
            {"input_tokens": 1000, "output_tokens": 10, "call_id": "a", "conversation_id": "r0"},
            {"input_tokens": 2000, "output_tokens": 20, "call_id": "b", "conversation_id": "rx"},
        ]
    )
    ownership = pd.DataFrame(
        [("pool0", "a"), ("refl0", "a"), ("refl0", "b")], columns=["run_id", "call_id"]
    )
    out = run_costs(df, ownership, {"p1": 100}).set_index("run_id")
    assert out.loc["pool0", "tokens_in_calls"] == 1000
    assert out.loc["refl0", "tokens_in_calls"] == 3000
    assert out.loc["refl0", "n_calls_billed"] == 2
    assert out.loc["refl0", "usd_nocache"] > out.loc["pool0", "usd_nocache"]
