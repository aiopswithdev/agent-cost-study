"""T14.3 — `react_vote@k`, the budget-matched control.

The ReAct pool gives, post hoc and at no extra API cost, a whole accuracy-vs-budget curve
for the *simplest* scaffold. Every other scaffold is then compared not against ReAct at
k = 1 but against the point on this curve that spends the same tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .. import config
from ..grading import gsm8k_pred, grade, normalize_answer
from .costs import REGIMES

TOKEN_COL = "tokens_total"


def vote_key(benchmark: str, answer: str) -> str | None:
    """The string votes are counted over. `None` means "abstain" (empty / unparsable)."""
    if answer is None:
        return None
    if benchmark == "gsm8k":
        value = gsm8k_pred(answer)
        return None if value is None else repr(value)
    key = normalize_answer(answer)
    return key or None


def majority_vote(benchmark: str, answers: Sequence[str]) -> str:
    """Majority over normalised answers, ignoring empty ones.

    Ties go to the tied answer from the earliest-drawn sample. Returns the **original**
    answer string of the winning group's first member, so grading is unchanged.
    """
    counts: dict[str, int] = {}
    first_index: dict[str, int] = {}
    first_answer: dict[str, str] = {}
    for i, a in enumerate(answers):
        key = vote_key(benchmark, a)
        if key is None:
            continue
        counts[key] = counts.get(key, 0) + 1
        if key not in first_index:
            first_index[key] = i
            first_answer[key] = a
    if not counts:
        return ""
    best = max(counts.values())
    winner = min(
        (k for k, c in counts.items() if c == best), key=lambda k: first_index[k]
    )
    return first_answer[winner]


# --------------------------------------------------------------------------- the curve


@dataclass
class VoteCurve:
    benchmark: str
    per_task: pd.DataFrame  # task_id x k -> acc, tokens, usd_*
    aggregate: pd.DataFrame  # k -> mean acc, mean tokens, mean usd_*
    cost_cols: tuple[str, ...]

    def at(self, k: int) -> pd.Series:
        return self.aggregate.loc[k]


def compute_vote_curve(
    pool: pd.DataFrame,
    benchmark: str,
    ks: Sequence[int] | None = None,
    draws: int = config.VOTE_DRAWS,
    seed: int = config.SEED,
    cost_cols: Sequence[str] = REGIMES,
) -> VoteCurve:
    """`pool` = the ReAct pool runs for one benchmark, with costs already attached."""
    pool = pool[pool["benchmark"] == benchmark]
    cost_cols = tuple(c for c in cost_cols if c in pool.columns)
    value_cols = (TOKEN_COL,) + cost_cols
    max_k = int(pool.groupby("task_id", observed=True).size().max())
    ks = list(ks) if ks is not None else list(range(1, min(config.POOL_K, max_k) + 1))

    rng = np.random.default_rng(seed)
    records: list[dict] = []

    for task_id, group in pool.groupby("task_id", observed=True, sort=True):
        group = group.sort_values("replicate")
        answers = group["final_answer"].fillna("").tolist()
        gold = group["gold"].iloc[0]
        values = {c: group[c].to_numpy(dtype=float) for c in value_cols}
        n = len(answers)

        for k in ks:
            kk = min(k, n)
            acc_sum = 0.0
            cost_sums = {c: 0.0 for c in value_cols}
            n_draws = draws if kk < n else 1  # drawing all n samples is deterministic
            for _ in range(n_draws):
                idx = rng.choice(n, size=kk, replace=False) if kk < n else np.arange(n)
                voted = majority_vote(benchmark, [answers[i] for i in idx])
                acc_sum += grade(benchmark, voted, gold)["correct"]
                for c in value_cols:
                    cost_sums[c] += float(values[c][idx].sum())
            rec = {"task_id": task_id, "k": k, "acc": acc_sum / n_draws, "n_samples": n}
            rec.update({c: cost_sums[c] / n_draws for c in value_cols})
            records.append(rec)

    per_task = pd.DataFrame(records)
    aggregate = (
        per_task.groupby("k", observed=True)[["acc", *value_cols]].mean().sort_index()
    )
    return VoteCurve(
        benchmark=benchmark, per_task=per_task, aggregate=aggregate, cost_cols=cost_cols
    )


# --------------------------------------------------------------------------- budget matching


@dataclass
class MatchPoint:
    k_star: float
    accuracy: float
    cost: float
    beyond_pool: bool  # True when the scaffold outspends vote@k_max (k* > k_max)
    cost_col: str


def interpolate_k(curve: VoteCurve, target_cost: float, cost_col: str = TOKEN_COL) -> MatchPoint:
    """The (fractional) k whose mean cost equals `target_cost`, by linear interpolation.

    If the scaffold outspends the whole pool, `k*` is reported as the maximum k and
    flagged: vote@k_max is then a *conservative lower bound* on the matched control.
    """
    agg = curve.aggregate
    ks = agg.index.to_numpy(dtype=float)
    costs = agg[cost_col].to_numpy(dtype=float)
    accs = agg["acc"].to_numpy(dtype=float)

    if target_cost <= costs[0]:
        return MatchPoint(float(ks[0]), float(accs[0]), float(costs[0]), False, cost_col)
    if target_cost >= costs[-1]:
        return MatchPoint(float(ks[-1]), float(accs[-1]), float(costs[-1]), True, cost_col)

    k_star = float(np.interp(target_cost, costs, ks))
    acc = float(np.interp(target_cost, costs, accs))
    return MatchPoint(k_star, acc, float(target_cost), False, cost_col)


def per_task_at_k(curve: VoteCurve, k_star: float) -> pd.Series:
    """Per-task accuracy at a fractional k, interpolated between the bracketing integers.

    Needed so the paired bootstrap can resample the control on the same task set.
    """
    lo, hi = int(np.floor(k_star)), int(np.ceil(k_star))
    pivot = curve.per_task.pivot(index="task_id", columns="k", values="acc")
    lo = max(lo, int(pivot.columns.min()))
    hi = min(hi, int(pivot.columns.max()))
    if lo == hi:
        return pivot[lo]
    w = k_star - lo
    return pivot[lo] * (1 - w) + pivot[hi] * w
