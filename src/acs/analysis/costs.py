"""T14.1–T14.2 — the three pricing regimes, and the schedule-independent ideal-cache model.

`usd_empirical` is what the API charged on the day we ran; it depends on the run schedule
(TTL, concurrency). `usd_ideal[L]` is the counterfactual "perfect prefix caching with
minimum cacheable length L" in a warm steady state, which is comparable across scaffolds
regardless of when each run happened to execute.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .. import config
from ..pricing import total_input, total_output, usd_empirical, usd_from_split, usd_nocache

REGIMES = ("usd_nocache", "usd_empirical", "usd_ideal_0", "usd_ideal_1024", "usd_ideal_4096")


def ideal_col(L: int) -> str:
    return f"usd_ideal_{L}"


# --------------------------------------------------------------------------- T14.2


def _ideal_split(
    I: int, O: int, S: int, prev_I_ge_L: list[int], seen_before: bool, L: int
) -> tuple[int, int, int]:
    """Returns `(uncached, write, read)` for one call. See T14.2 for the algorithm."""
    read = 0
    if prev_I_ge_L:
        # The entry written at the previous call's last block covers that call's whole prefix.
        read = max(prev_I_ge_L)
    elif S >= L and seen_before:
        # Otherwise the shared system-prompt breakpoint still hits, across tasks.
        read = S
    read = min(read, I)
    write = (I - read) if I >= L else 0
    uncached = I - read - write
    return uncached, write, read


def usd_ideal(
    call_rows: Sequence[Mapping[str, Any]] | Iterable[Mapping[str, Any]],
    L: int,
    prompt_tokens: Mapping[str, int] | None = None,
) -> list[float]:
    """Per-call ideal-cache USD, in the order the rows were given."""
    rows = list(call_rows)
    if not rows:
        return []
    df = add_ideal_regime(pd.DataFrame(rows), L, prompt_tokens or {})
    return df[ideal_col(L)].tolist()


def add_ideal_regime(
    calls: pd.DataFrame, L: int, prompt_tokens: Mapping[str, int]
) -> pd.DataFrame:
    """Add `usd_ideal_{L}` (and its token split) to a copy of `calls`."""
    df = calls.copy()
    if not len(df):
        df[ideal_col(L)] = []
        return df

    if "I" not in df:
        df["I"] = df.apply(lambda r: total_input(r), axis=1)
    if "O" not in df:
        df["O"] = df.apply(lambda r: total_output(r), axis=1)

    df["_S"] = df["system_prompt_id"].map(lambda p: int(prompt_tokens.get(p, 0))).astype("int64")

    # "Seen before" = an earlier call with the same system_prompt_id in `ts` order;
    # the very first call for each system prompt pays the write.
    order = df.sort_values(["ts", "call_id"], kind="stable").index
    rank_in_prompt = (
        df.loc[order].groupby("system_prompt_id", observed=True).cumcount()
    )
    df["_seen_before"] = rank_in_prompt.reindex(df.index) > 0

    # The largest earlier `I` within the same conversation that is itself >= L.
    df["_prev_max_I"] = 0
    if "conversation_id" in df:
        conv_order = df.sort_values(["conversation_id", "turn_index"], kind="stable").index
        sub = df.loc[conv_order, ["conversation_id", "I"]].copy()
        eligible = sub["I"].where(sub["I"] >= L, np.nan)
        prev_max = (
            eligible.groupby(sub["conversation_id"], observed=True)
            .apply(lambda s: s.shift(1).cummax())
            .reset_index(level=0, drop=True)
        )
        df["_prev_max_I"] = prev_max.reindex(df.index).fillna(0).astype("int64")

    read = np.where(
        df["_prev_max_I"] > 0,
        df["_prev_max_I"],
        np.where((df["_S"] >= L) & df["_seen_before"], df["_S"], 0),
    )
    read = np.minimum(read, df["I"].to_numpy())
    write = np.where(df["I"].to_numpy() >= L, df["I"].to_numpy() - read, 0)
    uncached = df["I"].to_numpy() - read - write

    df[f"ideal_{L}_read"] = read
    df[f"ideal_{L}_write"] = write
    df[f"ideal_{L}_uncached"] = uncached
    df[ideal_col(L)] = (
        uncached * config.PRICE_INPUT_PER_MTOK
        + write * config.PRICE_CACHE_WRITE_PER_MTOK
        + read * config.PRICE_CACHE_READ_PER_MTOK
        + df["O"].to_numpy() * config.PRICE_OUTPUT_PER_MTOK
    ) / 1e6
    return df.drop(columns=["_S", "_seen_before", "_prev_max_I"])


# --------------------------------------------------------------------------- per-call regimes


def add_all_regimes(
    calls: pd.DataFrame,
    prompt_tokens: Mapping[str, int],
    Ls: Sequence[int] = config.IDEAL_CACHE_L,
) -> pd.DataFrame:
    """Every regime of §2.6 as a column on a copy of `calls`."""
    df = calls.copy()
    if not len(df):
        for c in REGIMES:
            df[c] = []
        return df
    df["usd_nocache"] = df.apply(lambda r: usd_nocache(r), axis=1)
    df["usd_empirical"] = df.apply(lambda r: usd_empirical(r), axis=1)
    for L in Ls:
        df = add_ideal_regime(df, L, prompt_tokens)
    return df


# --------------------------------------------------------------------------- per-run rollup


def run_costs(
    calls: pd.DataFrame,
    ownership: pd.DataFrame,
    prompt_tokens: Mapping[str, int],
    Ls: Sequence[int] = config.IDEAL_CACHE_L,
) -> pd.DataFrame:
    """Per-run tokens and dollars in every regime, following the ownership table.

    Reflexion runs inherit their trial-1 pool call rows, so their cost includes trial 1 in
    full (§2.4) even though those rows carry the pool run's `run_id`.
    """
    priced = add_all_regimes(calls, prompt_tokens, Ls)
    cols = ["I", "O", "usd_nocache", "usd_empirical"] + [ideal_col(L) for L in Ls]
    joined = ownership.merge(
        priced[["call_id", *cols]], on="call_id", how="left", validate="many_to_one"
    )
    agg = joined.groupby("run_id", observed=True)[cols].sum()
    agg = agg.rename(columns={"I": "tokens_in_calls", "O": "tokens_out_calls"})
    agg["tokens_total"] = agg["tokens_in_calls"] + agg["tokens_out_calls"]
    agg["n_calls_billed"] = joined.groupby("run_id", observed=True).size()
    return agg.reset_index()


def attach_costs(runs: pd.DataFrame, costs: pd.DataFrame) -> pd.DataFrame:
    """Left-join `run_costs` onto the runs table, filling runs that made no calls."""
    out = runs.merge(costs, on="run_id", how="left")
    numeric = [c for c in costs.columns if c != "run_id"]
    out[numeric] = out[numeric].fillna(0.0)
    return out


# --------------------------------------------------------------------------- headline metric


def cost_of_pass(mean_cost: float, accuracy: float) -> float:
    """Erol et al. (2025): expected cost per *successful* task."""
    if accuracy <= 0:
        return float("inf")
    return mean_cost / accuracy


def empirical_cache_read_share(calls: pd.DataFrame) -> float:
    """Σ cache_read / Σ I — expected ≈ 0 on Haiku 4.5 (4,096-token threshold)."""
    if not len(calls):
        return 0.0
    total = calls["I"].sum() if "I" in calls else calls.apply(total_input, axis=1).sum()
    if total == 0:
        return 0.0
    return float(calls["cache_read_input_tokens"].sum() / total)
