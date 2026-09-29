"""T14.5 — per-task aggregation, the metrics table, and the pre-registered comparisons.

The unit of analysis is the task; everything below first collapses replicates to one row
per (benchmark, scaffold, task) and only then aggregates across tasks (§2.8).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from .. import config
from .costs import REGIMES, attach_costs, cost_of_pass, ideal_col, run_costs
from .load import Dataset
from .stats import (
    BootstrapResult,
    bootstrap_cost_of_pass,
    bootstrap_mean,
    bootstrap_paired_diff,
    bootstrap_paired_ratio,
    holm,
    mcnemar_from_pairs,
)
from .vote import TOKEN_COL, MatchPoint, VoteCurve, compute_vote_curve, interpolate_k, per_task_at_k

BASELINE = "react"
VOTE_PREFIX = "react_vote@"

MEAN_COLS = (
    "correct",
    "em",
    "f1",
    "n_llm_calls",
    "n_tool_calls_total",
    "n_steps",
    "halted",
    "wall_s",
    "tokens_in_calls",
    "tokens_out_calls",
    TOKEN_COL,
    *REGIMES,
)


# --------------------------------------------------------------------------- preparation


def costed_runs(ds: Dataset) -> pd.DataFrame:
    """Runs with every pricing regime attached, ready for per-task aggregation."""
    costs = run_costs(ds.calls, ds.ownership, ds.prompt_tokens)
    return attach_costs(ds.runs, costs)


def per_task(runs: pd.DataFrame, r: int = config.R) -> pd.DataFrame:
    """One row per (benchmark, scaffold, task): the mean over that scaffold's replicates.

    The ReAct **baseline** uses pool samples `r = 0..R−1` (§2.4); the remaining pool
    samples exist only to build the vote@k curve.
    """
    df = runs.copy()
    df = df[~((df["scaffold"] == BASELINE) & (df["replicate"] >= r))]
    cols = [c for c in MEAN_COLS if c in df.columns]
    grouped = (
        df.groupby(["benchmark", "scaffold", "task_id"], observed=True)[cols]
        .mean()
        .reset_index()
    )
    grouped["n_replicates"] = (
        df.groupby(["benchmark", "scaffold", "task_id"], observed=True).size().to_numpy()
    )
    return grouped


def aligned(
    table: pd.DataFrame, benchmark: str, scaffold_a: str, scaffold_b: str, column: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """The two arms' per-task values on the tasks both arms completed, in one task order."""
    a = table[(table["benchmark"] == benchmark) & (table["scaffold"] == scaffold_a)]
    b = table[(table["benchmark"] == benchmark) & (table["scaffold"] == scaffold_b)]
    merged = a.merge(b, on="task_id", suffixes=("_a", "_b")).sort_values("task_id")
    return (
        merged[f"{column}_a"].to_numpy(dtype=float),
        merged[f"{column}_b"].to_numpy(dtype=float),
        merged["task_id"].tolist(),
    )


# --------------------------------------------------------------------------- vote@k control


def vote_curves(ds_runs: pd.DataFrame, **kw) -> dict[str, VoteCurve]:
    pool = ds_runs[ds_runs["scaffold"] == BASELINE]
    return {
        b: compute_vote_curve(pool, b, **kw)
        for b in sorted(pool["benchmark"].unique())
    }


@dataclass
class MatchedControl:
    scaffold: str
    benchmark: str
    point: MatchPoint
    per_task_acc: pd.Series

    @property
    def label(self) -> str:
        return f"{VOTE_PREFIX}k*({self.scaffold})"


def matched_control(
    curve: VoteCurve,
    table: pd.DataFrame,
    benchmark: str,
    scaffold: str,
    cost_col: str = TOKEN_COL,
) -> MatchedControl:
    """The vote@k point that spends what `scaffold` spends (H1's budget match)."""
    arm = table[(table["benchmark"] == benchmark) & (table["scaffold"] == scaffold)]
    target = float(arm[cost_col].mean())
    point = interpolate_k(curve, target, cost_col=cost_col)
    return MatchedControl(
        scaffold=scaffold,
        benchmark=benchmark,
        point=point,
        per_task_acc=per_task_at_k(curve, point.k_star),
    )


# --------------------------------------------------------------------------- Table 1


def metrics_table(
    table: pd.DataFrame,
    curves: dict[str, VoteCurve],
    controls: dict[tuple[str, str], MatchedControl] | None = None,
    B: int = config.BOOT_B,
    vote_ks: Sequence[int] = (1, 3, 5, 10),
) -> pd.DataFrame:
    """T14.5 — one row per benchmark × arm, with bootstrap CIs on the headline numbers."""
    rows: list[dict] = []

    for benchmark in sorted(table["benchmark"].unique()):
        for scaffold in config.SCAFFOLDS:
            arm = table[
                (table["benchmark"] == benchmark) & (table["scaffold"] == scaffold)
            ].sort_values("task_id")
            if not len(arm):
                continue
            rows.append(_arm_row(benchmark, scaffold, arm, B))

        curve = curves.get(benchmark)
        if curve is not None:
            for k in vote_ks:
                if k not in curve.aggregate.index:
                    continue
                rows.append(_vote_row(benchmark, f"{VOTE_PREFIX}{k}", curve, k, B))
            for (bench, scaffold), control in (controls or {}).items():
                if bench != benchmark:
                    continue
                rows.append(_control_row(benchmark, control, curve, B))

    return pd.DataFrame(rows)


def _arm_row(benchmark: str, scaffold: str, arm: pd.DataFrame, B: int) -> dict:
    acc = bootstrap_mean(arm["correct"], B=B)
    row = {
        "benchmark": benchmark,
        "arm": scaffold,
        "n_tasks": len(arm),
        "acc": acc.point,
        "acc_ci_lo": acc.ci_lo,
        "acc_ci_hi": acc.ci_hi,
        "em": float(arm["em"].mean()),
        "f1": float(arm["f1"].mean()),
        "tokens_in": float(arm["tokens_in_calls"].mean()),
        "tokens_out": float(arm["tokens_out_calls"].mean()),
        "tokens_total": float(arm[TOKEN_COL].mean()),
        "n_llm_calls": float(arm["n_llm_calls"].mean()),
        "n_tool_calls": float(arm["n_tool_calls_total"].mean()),
        "n_steps": float(arm["n_steps"].mean()),
        "halt_rate": float(arm["halted"].mean()),
        "wall_s": float(arm["wall_s"].mean()),
    }
    for regime in REGIMES:
        if regime not in arm:
            continue
        row[regime] = float(arm[regime].mean())
        cop = bootstrap_cost_of_pass(arm[regime], arm["correct"], B=B)
        row[f"cop_{regime}"] = cop.point
        row[f"cop_{regime}_ci_lo"] = cop.ci_lo
        row[f"cop_{regime}_ci_hi"] = cop.ci_hi
    return row


def _vote_row(benchmark: str, label: str, curve: VoteCurve, k: int, B: int) -> dict:
    at_k = curve.per_task[curve.per_task["k"] == k].sort_values("task_id")
    acc = bootstrap_mean(at_k["acc"], B=B)
    row = {
        "benchmark": benchmark,
        "arm": label,
        "n_tasks": len(at_k),
        "acc": acc.point,
        "acc_ci_lo": acc.ci_lo,
        "acc_ci_hi": acc.ci_hi,
        "tokens_total": float(at_k[TOKEN_COL].mean()),
        "n_llm_calls": float("nan"),
        "vote_k": float(k),
    }
    for regime in curve.cost_cols:
        row[regime] = float(at_k[regime].mean())
        cop = bootstrap_cost_of_pass(at_k[regime], at_k["acc"], B=B)
        row[f"cop_{regime}"] = cop.point
        row[f"cop_{regime}_ci_lo"] = cop.ci_lo
        row[f"cop_{regime}_ci_hi"] = cop.ci_hi
    return row


def _control_row(benchmark: str, control: MatchedControl, curve: VoteCurve, B: int) -> dict:
    acc = bootstrap_mean(control.per_task_acc.to_numpy(), B=B)
    return {
        "benchmark": benchmark,
        "arm": control.label,
        "n_tasks": len(control.per_task_acc),
        "acc": acc.point,
        "acc_ci_lo": acc.ci_lo,
        "acc_ci_hi": acc.ci_hi,
        "tokens_total": control.point.cost,
        "vote_k": control.point.k_star,
        "k_star_beyond_pool": control.point.beyond_pool,
    }


# --------------------------------------------------------------------------- Table 2


@dataclass
class Comparison:
    benchmark: str
    family: str
    arm_a: str
    arm_b: str
    delta: BootstrapResult
    mcnemar_p: float
    holm_p: float = float("nan")

    def as_row(self) -> dict:
        return {
            "benchmark": self.benchmark,
            "family": self.family,
            "arm_a": self.arm_a,
            "arm_b": self.arm_b,
            "delta": self.delta.point,
            "ci_lo": self.delta.ci_lo,
            "ci_hi": self.delta.ci_hi,
            "bootstrap_p": self.delta.p_value,
            "holm_p": self.holm_p,
            "mcnemar_p": self.mcnemar_p,
        }


def _replicate0_pairs(
    runs: pd.DataFrame, benchmark: str, scaffold_a: str, scaffold_b: str
) -> tuple[np.ndarray, np.ndarray]:
    """Binary outcomes at replicate 0, for the secondary McNemar test (§2.8)."""
    sel = runs[(runs["benchmark"] == benchmark) & (runs["replicate"] == 0)]
    a = sel[sel["scaffold"] == scaffold_a][["task_id", "correct"]]
    b = sel[sel["scaffold"] == scaffold_b][["task_id", "correct"]]
    merged = a.merge(b, on="task_id", suffixes=("_a", "_b"))
    return merged["correct_a"].to_numpy(int), merged["correct_b"].to_numpy(int)


def preregistered_comparisons(
    table: pd.DataFrame,
    runs: pd.DataFrame,
    controls: dict[tuple[str, str], MatchedControl],
    B: int = config.BOOT_B,
) -> pd.DataFrame:
    """The nine pre-registered comparisons per benchmark, Holm-corrected within benchmark."""
    out: list[Comparison] = []

    for benchmark in sorted(table["benchmark"].unique()):
        family: list[Comparison] = []

        for scaffold in config.OTHER_SCAFFOLDS:
            a, b, _ = aligned(table, benchmark, scaffold, BASELINE, "correct")
            if not len(a):
                continue
            pairs_a, pairs_b = _replicate0_pairs(runs, benchmark, scaffold, BASELINE)
            family.append(
                Comparison(
                    benchmark=benchmark,
                    family="X_vs_react",
                    arm_a=scaffold,
                    arm_b=BASELINE,
                    delta=bootstrap_paired_diff(a, b, B=B),
                    mcnemar_p=mcnemar_from_pairs(pairs_a, pairs_b).p_value,
                )
            )

        for scaffold in config.OTHER_SCAFFOLDS:
            control = controls.get((benchmark, scaffold))
            if control is None:
                continue
            arm = table[
                (table["benchmark"] == benchmark) & (table["scaffold"] == scaffold)
            ].set_index("task_id")["correct"]
            shared = arm.index.intersection(control.per_task_acc.index)
            if not len(shared):
                continue
            family.append(
                Comparison(
                    benchmark=benchmark,
                    family="X_vs_matched_vote",
                    arm_a=scaffold,
                    arm_b=control.label,
                    delta=bootstrap_paired_diff(
                        arm.loc[shared].to_numpy(),
                        control.per_task_acc.loc[shared].to_numpy(),
                        B=B,
                    ),
                    mcnemar_p=float("nan"),  # vote@k is an expectation, not a binary draw
                )
            )

        a, b, _ = aligned(table, benchmark, "reflexion_gt", "reflexion_self", "correct")
        if len(a):
            pairs_a, pairs_b = _replicate0_pairs(
                runs, benchmark, "reflexion_gt", "reflexion_self"
            )
            family.append(
                Comparison(
                    benchmark=benchmark,
                    family="oracle_premium",
                    arm_a="reflexion_gt",
                    arm_b="reflexion_self",
                    delta=bootstrap_paired_diff(a, b, B=B),
                    mcnemar_p=mcnemar_from_pairs(pairs_a, pairs_b).p_value,
                )
            )

        adjusted = holm([c.delta.p_value or 1.0 for c in family])
        for comparison, p in zip(family, adjusted):
            comparison.holm_p = p
        out.extend(family)

    return pd.DataFrame([c.as_row() for c in out])


# --------------------------------------------------------------------------- H1


def h1_table(
    table: pd.DataFrame,
    controls: dict[tuple[str, str], MatchedControl],
    B: int = config.BOOT_B,
) -> pd.DataFrame:
    """H1: does budget matching absorb at least half of each scaffold's naive gain?"""
    rows = []
    for (benchmark, scaffold), control in sorted(controls.items()):
        a, b, _ = aligned(table, benchmark, scaffold, BASELINE, "correct")
        if not len(a):
            continue
        delta_def = bootstrap_paired_diff(a, b, B=B)

        arm = table[
            (table["benchmark"] == benchmark) & (table["scaffold"] == scaffold)
        ].set_index("task_id")["correct"]
        shared = arm.index.intersection(control.per_task_acc.index)
        delta_match = bootstrap_paired_diff(
            arm.loc[shared].to_numpy(), control.per_task_acc.loc[shared].to_numpy(), B=B
        )

        supported = (
            bool(delta_match.point <= 0.5 * delta_def.point)
            if delta_def.point > 0
            else None  # H1 only predicts anything for scaffolds that beat ReAct
        )
        rows.append(
            {
                "benchmark": benchmark,
                "scaffold": scaffold,
                "k_star": control.point.k_star,
                "k_star_beyond_pool": control.point.beyond_pool,
                "matched_tokens": control.point.cost,
                "delta_default": delta_def.point,
                "delta_default_ci_lo": delta_def.ci_lo,
                "delta_default_ci_hi": delta_def.ci_hi,
                "delta_matched": delta_match.point,
                "delta_matched_ci_lo": delta_match.ci_lo,
                "delta_matched_ci_hi": delta_match.ci_hi,
                "h1_supported": supported,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- Table 3 (H3)

H3_COST_COLS = (TOKEN_COL, "usd_nocache", ideal_col(0), ideal_col(1024), ideal_col(4096),
                "usd_empirical")


def h3_cost_ratios(table: pd.DataFrame, B: int = config.BOOT_B) -> pd.DataFrame:
    """Each scaffold's cost relative to ReAct, under every cost measure."""
    rows = []
    for benchmark in sorted(table["benchmark"].unique()):
        for scaffold in config.SCAFFOLDS:
            if scaffold == BASELINE:
                continue
            for column in H3_COST_COLS:
                if column not in table.columns:
                    continue
                a, b, _ = aligned(table, benchmark, scaffold, BASELINE, column)
                if not len(a):
                    continue
                ratio = bootstrap_paired_ratio(a, b, B=B)
                rows.append(
                    {
                        "benchmark": benchmark,
                        "scaffold": scaffold,
                        "measure": column,
                        "ratio": ratio.point,
                        "ci_lo": ratio.ci_lo,
                        "ci_hi": ratio.ci_hi,
                    }
                )
    return pd.DataFrame(rows)


def h3_verdict(ratios: pd.DataFrame, scaffold: str = "rewoo") -> pd.DataFrame:
    """H3: is `cost(X)/cost(react)` closer to 1 under `usd_ideal[1024]` than under tokens?"""
    rows = []
    for benchmark in sorted(ratios["benchmark"].unique()):
        sel = ratios[(ratios["benchmark"] == benchmark) & (ratios["scaffold"] == scaffold)]
        by_measure = sel.set_index("measure")["ratio"]
        if TOKEN_COL not in by_measure or ideal_col(1024) not in by_measure:
            continue
        tokens = abs(by_measure[TOKEN_COL] - 1.0)
        ideal_1024 = abs(by_measure[ideal_col(1024)] - 1.0)
        rows.append(
            {
                "benchmark": benchmark,
                "scaffold": scaffold,
                "ratio_tokens": by_measure[TOKEN_COL],
                "ratio_ideal_1024": by_measure[ideal_col(1024)],
                "ratio_ideal_4096": by_measure.get(ideal_col(4096), float("nan")),
                "ratio_empirical": by_measure.get("usd_empirical", float("nan")),
                "h3_supported": bool(ideal_1024 < tokens),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- Table 4


def judge_confusion(runs: pd.DataFrame) -> pd.DataFrame:
    """RQ2's explanation: the self-judge's verdicts against gold, over all judged trials."""
    rows = []
    sel = runs[runs["scaffold"] == "reflexion_self"]
    for benchmark in sorted(sel["benchmark"].unique()):
        tp = fp = tn = fn = 0
        for _, run in sel[sel["benchmark"] == benchmark].iterrows():
            verdicts = run.get("judge_verdicts") or []
            correct = run.get("trial_correct") or []
            for verdict, is_correct in zip(verdicts, correct):
                if verdict is None:
                    continue
                said_correct = verdict == "CORRECT"
                if said_correct and is_correct:
                    tp += 1
                elif said_correct and not is_correct:
                    fp += 1
                elif not said_correct and is_correct:
                    fn += 1
                else:
                    tn += 1
        judged = tp + fp + tn + fn
        rows.append(
            {
                "benchmark": benchmark,
                "n_judged_trials": judged,
                "true_accept": tp,
                "false_accept": fp,
                "true_reject": tn,
                "false_reject": fn,
                "false_accept_rate": fp / (fp + tn) if (fp + tn) else float("nan"),
                "false_reject_rate": fn / (fn + tp) if (fn + tp) else float("nan"),
                "judge_accuracy": (tp + tn) / judged if judged else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def failure_modes(runs: pd.DataFrame) -> pd.DataFrame:
    """Table 4 — parse failures, halts, invalid actions, judge parse failures, trials."""
    rows = []
    for benchmark in sorted(runs["benchmark"].unique()):
        for scaffold in config.SCAFFOLDS:
            sel = runs[(runs["benchmark"] == benchmark) & (runs["scaffold"] == scaffold)]
            if not len(sel):
                continue
            rows.append(
                {
                    "benchmark": benchmark,
                    "scaffold": scaffold,
                    "n_runs": len(sel),
                    "halt_rate": float(sel["halted"].mean()),
                    "invalid_action_rate": float((sel["n_invalid_actions"] > 0).mean()),
                    "mean_invalid_actions": float(sel["n_invalid_actions"].mean()),
                    "could_not_find_rate": float((sel["n_could_not_find"] > 0).mean()),
                    "plan_parse_ok_rate": (
                        float(sel["plan_parse_ok"].astype(float).mean())
                        if sel["plan_parse_ok"].notna().any()
                        else float("nan")
                    ),
                    "mean_trials": (
                        float(sel["n_trials"].mean())
                        if sel["n_trials"].notna().any()
                        else float("nan")
                    ),
                    "judge_parse_fail_rate": (
                        float((sel["judge_parse_fail"].fillna(0) > 0).mean())
                        if sel["judge_parse_fail"].notna().any()
                        else float("nan")
                    ),
                    "empty_answer_rate": float((sel["answer_len_chars"] == 0).mean()),
                }
            )
    return pd.DataFrame(rows)
