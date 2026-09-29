"""T14.6 — RQ4: can cheap first-attempt features predict when Reflexion-Self pays off?

Target: the first ReAct attempt was wrong **and** Reflexion-Self (replicate 0) got it
right. Features come only from the first attempt, so a router could compute them before
deciding to spend anything extra.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .. import config
from .costs import REGIMES
from .vote import TOKEN_COL, vote_key

BASE_FEATURES = (
    "n_steps",
    "n_search",
    "n_lookup",
    "n_calc",
    "n_invalid_actions",
    "halted",
    "n_could_not_find",
    "answer_len_chars",
    "trial1_tokens",
)
AGREEMENT_FEATURE = "distinct_answers_in_first3"
BENCHMARK_FEATURES = tuple(f"benchmark_{b}" for b in config.BENCHMARKS)
TAUS = (0.2, 0.3, 0.5)


# --------------------------------------------------------------------------- features


def build_features(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per task: first-attempt features, the target, and the routing costs."""
    pool0 = runs[(runs["scaffold"] == "react") & (runs["replicate"] == 0)].set_index(
        ["benchmark", "task_id"]
    )
    first3 = runs[(runs["scaffold"] == "react") & (runs["replicate"] < 3)]
    refl = runs[
        (runs["scaffold"] == "reflexion_self") & (runs["replicate"] == 0)
    ].set_index(["benchmark", "task_id"])

    agreement = (
        first3.groupby(["benchmark", "task_id"], observed=True)
        .apply(
            lambda g: len(
                {
                    k
                    for k in (vote_key(g.name[0], a) for a in g["final_answer"].fillna(""))
                    if k is not None
                }
            )
            or 1,
            include_groups=False,
        )
        .rename(AGREEMENT_FEATURE)
    )
    extra_pool_cost = (
        first3[first3["replicate"].between(1, 2)]
        .groupby(["benchmark", "task_id"], observed=True)[
            [TOKEN_COL, *[c for c in REGIMES if c in first3.columns]]
        ]
        .sum()
        .add_prefix("agreement_cost_")
    )

    shared = pool0.index.intersection(refl.index)
    rows = pd.DataFrame(index=shared)
    rows["n_steps"] = pool0.loc[shared, "n_steps"]
    rows["n_search"] = pool0.loc[shared, "n_search"]
    rows["n_lookup"] = pool0.loc[shared, "n_lookup"]
    rows["n_calc"] = pool0.loc[shared, "n_calc"]
    rows["n_invalid_actions"] = pool0.loc[shared, "n_invalid_actions"]
    rows["halted"] = pool0.loc[shared, "halted"].astype(float)
    rows["n_could_not_find"] = pool0.loc[shared, "n_could_not_find"]
    rows["answer_len_chars"] = pool0.loc[shared, "answer_len_chars"]
    rows["trial1_tokens"] = pool0.loc[shared, TOKEN_COL]
    rows[AGREEMENT_FEATURE] = agreement.reindex(shared).fillna(1)

    for b in config.BENCHMARKS:
        rows[f"benchmark_{b}"] = (
            rows.index.get_level_values("benchmark") == b
        ).astype(float)

    rows["react_correct"] = pool0.loc[shared, "correct"]
    rows["reflexion_correct"] = refl.loc[shared, "correct"]
    rows["y"] = (
        (rows["react_correct"] == 0) & (rows["reflexion_correct"] == 1)
    ).astype(int)

    for regime in (TOKEN_COL, *[c for c in REGIMES if c in runs.columns]):
        rows[f"react_cost_{regime}"] = pool0.loc[shared, regime]
        rows[f"reflexion_cost_{regime}"] = refl.loc[shared, regime]
        trial1 = refl.loc[shared, "trial1_source_run_id"]
        rows[f"reflexion_extra_{regime}"] = (
            rows[f"reflexion_cost_{regime}"] - rows[f"react_cost_{regime}"]
        ).clip(lower=0)
        agreement_col = f"agreement_cost_{regime}"
        rows[f"agreement_cost_{regime}"] = (
            extra_pool_cost[agreement_col].reindex(shared).fillna(0.0)
            if agreement_col in extra_pool_cost
            else 0.0
        )
        _ = trial1  # kept for provenance; the cost itself is already inside the run row

    return rows.reset_index()


# --------------------------------------------------------------------------- model


def make_model() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000)),
        ]
    )


@dataclass
class CVResult:
    name: str
    auc: float
    auc_ci_lo: float
    auc_ci_hi: float
    y_true: np.ndarray
    y_score: np.ndarray
    coefficients: dict[str, float] = field(default_factory=dict)

    def as_row(self) -> dict:
        return {
            "model": self.name,
            "cv_auc": self.auc,
            "auc_ci_lo": self.auc_ci_lo,
            "auc_ci_hi": self.auc_ci_hi,
            "n": len(self.y_true),
            "positives": int(self.y_true.sum()),
        }


def _bootstrap_auc(
    y: np.ndarray, scores: np.ndarray, B: int = 1000, seed: int = config.SEED
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    out = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        out.append(roc_auc_score(y[idx], scores[idx]))
    if not out:
        return float("nan"), float("nan")
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def cross_validated(
    features: pd.DataFrame,
    columns: Sequence[str],
    name: str,
    n_splits: int = 5,
    seed: int = config.SEED,
) -> CVResult:
    """5-fold stratified CV; the returned scores are strictly out-of-fold."""
    X = features[list(columns)].to_numpy(dtype=float)
    y = features["y"].to_numpy(dtype=int)
    scores = np.zeros(len(y), dtype=float)

    if len(np.unique(y)) < 2 or y.sum() < n_splits:
        return CVResult(name, float("nan"), float("nan"), float("nan"), y, scores)

    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train, test in folds.split(X, y):
        model = make_model()
        model.fit(X[train], y[train])
        scores[test] = model.predict_proba(X[test])[:, 1]

    auc = float(roc_auc_score(y, scores))
    lo, hi = _bootstrap_auc(y, scores, seed=seed)

    full = make_model().fit(X, y)
    coefficients = dict(zip(columns, full.named_steps["clf"].coef_[0].tolist()))
    return CVResult(name, auc, lo, hi, y, scores, coefficients)


def fit_all(features: pd.DataFrame, seed: int = config.SEED) -> dict[str, CVResult]:
    """H4's three models: the baseline, the cheap features, and those plus agreement."""
    benchmark_only = [c for c in BENCHMARK_FEATURES if c in features]
    base = [c for c in BASE_FEATURES if c in features] + benchmark_only
    with_agreement = base + [AGREEMENT_FEATURE]
    return {
        "benchmark_only": cross_validated(features, benchmark_only, "benchmark_only", seed=seed),
        "first_attempt": cross_validated(features, base, "first_attempt", seed=seed),
        "first_attempt_plus_agreement": cross_validated(
            features, with_agreement, "first_attempt_plus_agreement", seed=seed
        ),
    }


def h4_verdict(results: dict[str, CVResult]) -> dict:
    baseline = results["benchmark_only"].auc
    best_name = max(
        ("first_attempt", "first_attempt_plus_agreement"),
        key=lambda n: results[n].auc if np.isfinite(results[n].auc) else -1,
    )
    best = results[best_name]
    return {
        "baseline_auc": baseline,
        "best_model": best_name,
        "best_auc": best.auc,
        "best_auc_ci_lo": best.auc_ci_lo,
        "best_auc_ci_hi": best.auc_ci_hi,
        "h4_supported": bool(np.isfinite(best.auc) and np.isfinite(baseline)
                             and best.auc_ci_lo > baseline),
    }


# --------------------------------------------------------------------------- routing policy


def routing_policies(
    features: pd.DataFrame,
    result: CVResult,
    cost_col: str = TOKEN_COL,
    taus: Sequence[float] = TAUS,
    include_agreement_cost: bool = False,
) -> pd.DataFrame:
    """Evaluate "route to Reflexion-Self when p > τ" on the held-out scores.

    When the agreement feature is used, the two extra pool samples it needs are a real
    cost and are charged to every task, routed or not (§T14.6).
    """
    scores = result.y_score
    react_correct = features["react_correct"].to_numpy(dtype=float)
    reflexion_correct = features["reflexion_correct"].to_numpy(dtype=float)
    react_cost = features[f"react_cost_{cost_col}"].to_numpy(dtype=float)
    extra_cost = features[f"reflexion_extra_{cost_col}"].to_numpy(dtype=float)
    agreement_cost = (
        features[f"agreement_cost_{cost_col}"].to_numpy(dtype=float)
        if include_agreement_cost
        else np.zeros(len(features))
    )

    rows = [
        {
            "policy": "always_react",
            "tau": float("nan"),
            "accuracy": float(react_correct.mean()),
            "cost": float(react_cost.mean()),
            "routed_frac": 0.0,
        },
        {
            "policy": "always_reflexion",
            "tau": float("nan"),
            "accuracy": float(reflexion_correct.mean()),
            "cost": float((react_cost + extra_cost).mean()),
            "routed_frac": 1.0,
        },
    ]
    for tau in taus:
        route = scores > tau
        accuracy = np.where(route, reflexion_correct, react_correct).mean()
        cost = (react_cost + np.where(route, extra_cost, 0.0) + agreement_cost).mean()
        rows.append(
            {
                "policy": f"route_p>{tau}",
                "tau": float(tau),
                "accuracy": float(accuracy),
                "cost": float(cost),
                "routed_frac": float(route.mean()),
            }
        )
    out = pd.DataFrame(rows)
    out["cost_col"] = cost_col
    out["includes_agreement_cost"] = include_agreement_cost
    return out


def roc_points(result: CVResult) -> pd.DataFrame:
    if len(np.unique(result.y_true)) < 2:
        return pd.DataFrame(columns=["fpr", "tpr"])
    fpr, tpr, _ = roc_curve(result.y_true, result.y_score)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr})
