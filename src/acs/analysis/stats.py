"""T14.4 — cluster bootstrap over tasks, exact McNemar, and Holm correction.

The unit of analysis is the **task** (§2.8): a scaffold's per-task score is the mean over
its replicates, and every interval resamples tasks, not runs. Paired comparisons use the
same resampled task set for both arms, which is what makes the differences interpretable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy import stats

from .. import config


@dataclass
class BootstrapResult:
    point: float
    ci_lo: float
    ci_hi: float
    p_value: float | None
    samples: np.ndarray

    def as_dict(self, prefix: str = "") -> dict:
        return {
            f"{prefix}point": self.point,
            f"{prefix}ci_lo": self.ci_lo,
            f"{prefix}ci_hi": self.ci_hi,
            f"{prefix}p": self.p_value,
        }

    def __str__(self) -> str:
        return f"{self.point:.4f} [{self.ci_lo:.4f}, {self.ci_hi:.4f}]"


def _percentile_ci(samples: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def two_sided_p(samples: np.ndarray) -> float:
    """`p = 2 · min(P(Δ* ≤ 0), P(Δ* ≥ 0))`, clipped to [0, 1]."""
    n = len(samples)
    if n == 0:
        return float("nan")
    p_le = float(np.mean(samples <= 0))
    p_ge = float(np.mean(samples >= 0))
    return float(min(1.0, 2 * min(p_le, p_ge)))


def cluster_bootstrap(
    task_ids: Sequence[str],
    stat_fn: Callable[[np.ndarray], float],
    B: int = config.BOOT_B,
    seed: int = config.SEED,
    alpha: float = 0.05,
    with_p: bool = False,
) -> BootstrapResult:
    """Resample tasks with replacement; `stat_fn` receives the resampled index array."""
    tasks = np.asarray(task_ids)
    n = len(tasks)
    if n == 0:
        return BootstrapResult(float("nan"), float("nan"), float("nan"), None, np.array([]))

    point = float(stat_fn(np.arange(n)))
    rng = np.random.default_rng(seed)
    samples = np.empty(B, dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        samples[b] = stat_fn(idx)

    finite = samples[np.isfinite(samples)]
    lo, hi = _percentile_ci(finite, alpha) if len(finite) else (float("nan"), float("nan"))
    return BootstrapResult(
        point=point,
        ci_lo=lo,
        ci_hi=hi,
        p_value=two_sided_p(finite) if with_p else None,
        samples=samples,
    )


def bootstrap_mean(
    values: Sequence[float], B: int = config.BOOT_B, seed: int = config.SEED
) -> BootstrapResult:
    arr = np.asarray(values, dtype=float)
    return cluster_bootstrap(
        np.arange(len(arr)).astype(str), lambda idx: float(np.mean(arr[idx])), B=B, seed=seed
    )


def bootstrap_paired_diff(
    a: Sequence[float],
    b: Sequence[float],
    B: int = config.BOOT_B,
    seed: int = config.SEED,
) -> BootstrapResult:
    """`mean(a) − mean(b)` with both arms resampled on the same task set."""
    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"paired arrays must align: {a_arr.shape} vs {b_arr.shape}")
    return cluster_bootstrap(
        np.arange(len(a_arr)).astype(str),
        lambda idx: float(np.mean(a_arr[idx]) - np.mean(b_arr[idx])),
        B=B,
        seed=seed,
        with_p=True,
    )


def bootstrap_paired_ratio(
    a: Sequence[float],
    b: Sequence[float],
    B: int = config.BOOT_B,
    seed: int = config.SEED,
) -> BootstrapResult:
    """`mean(a) / mean(b)` on the same resampled task set (used for the H3 cost ratios)."""
    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)

    def stat(idx: np.ndarray) -> float:
        denom = float(np.mean(b_arr[idx]))
        return float(np.mean(a_arr[idx]) / denom) if denom else float("nan")

    return cluster_bootstrap(
        np.arange(len(a_arr)).astype(str), stat, B=B, seed=seed
    )


def bootstrap_cost_of_pass(
    cost: Sequence[float],
    correct: Sequence[float],
    B: int = config.BOOT_B,
    seed: int = config.SEED,
) -> BootstrapResult:
    """Cost-of-pass = mean cost / accuracy, resampled as one statistic."""
    cost_arr = np.asarray(cost, dtype=float)
    acc_arr = np.asarray(correct, dtype=float)

    def stat(idx: np.ndarray) -> float:
        acc = float(np.mean(acc_arr[idx]))
        return float(np.mean(cost_arr[idx]) / acc) if acc > 0 else float("inf")

    return cluster_bootstrap(np.arange(len(cost_arr)).astype(str), stat, B=B, seed=seed)


# --------------------------------------------------------------------------- McNemar


@dataclass
class McNemarResult:
    b: int
    c: int
    p_value: float

    def as_dict(self) -> dict:
        return {"mcnemar_b": self.b, "mcnemar_c": self.c, "mcnemar_p": self.p_value}


def mcnemar_exact(b: int, c: int) -> McNemarResult:
    """Exact (binomial) McNemar test on the discordant pairs."""
    n = b + c
    if n == 0:
        return McNemarResult(b, c, 1.0)
    p = float(stats.binomtest(b, n, 0.5, alternative="two-sided").pvalue)
    return McNemarResult(b, c, p)


def mcnemar_from_pairs(a: Sequence[int], b: Sequence[int]) -> McNemarResult:
    """`a` and `b` are paired 0/1 outcomes (replicate 0, per §2.8)."""
    a_arr, b_arr = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    if a_arr.shape != b_arr.shape:
        raise ValueError("paired outcome arrays must align")
    only_a = int(np.sum((a_arr == 1) & (b_arr == 0)))
    only_b = int(np.sum((a_arr == 0) & (b_arr == 1)))
    return mcnemar_exact(only_a, only_b)


# --------------------------------------------------------------------------- Holm


def holm(pvals: Sequence[float]) -> list[float]:
    """Holm–Bonferroni step-down adjusted p-values, in the input order."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(n, dtype=float)
    running = 0.0
    for rank, i in enumerate(order):
        value = (n - rank) * p[i]
        running = max(running, value)
        adjusted[i] = min(1.0, running)
    return adjusted.tolist()
