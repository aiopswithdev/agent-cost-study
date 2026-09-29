"""T14.7 — every figure and table, from the JSONL logs alone.

    python -m acs.analysis.report --run-tag main

Writes `results/{run_tag}/analysis/`: CSVs for each table, PNG+PDF for each figure, and
`SUMMARY.md` tying them together.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import config
from . import metrics as M
from . import rq4 as R4
from .costs import REGIMES, empirical_cache_read_share, ideal_col
from .load import Dataset, load
from .vote import TOKEN_COL, VoteCurve

PRIMARY_COST = "usd_nocache"
SCAFFOLD_COLORS = {
    "react": "#4C72B0",
    "rewoo": "#DD8452",
    "plan_exec": "#55A868",
    "reflexion_gt": "#C44E52",
    "reflexion_self": "#8172B3",
}
VOTE_COLOR = "#333333"


# --------------------------------------------------------------------------- driver


class Report:
    def __init__(
        self,
        ds: Dataset,
        out_dir: Path,
        B: int = config.BOOT_B,
        vote_draws: int = config.VOTE_DRAWS,
    ) -> None:
        self.ds = ds
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.B = B

        self.runs = M.costed_runs(ds)
        self.table = M.per_task(self.runs)
        self.curves = M.vote_curves(self.runs, draws=vote_draws)
        self.controls = {
            (b, s): M.matched_control(self.curves[b], self.table, b, s)
            for b in self.curves
            for s in config.OTHER_SCAFFOLDS
            if len(self.table[(self.table["benchmark"] == b) & (self.table["scaffold"] == s)])
        }

    # ------------------------------------------------------------------ tables

    def build_tables(self) -> dict[str, pd.DataFrame]:
        tables = {
            "table1_metrics": M.metrics_table(
                self.table, self.curves, self.controls, B=self.B
            ),
            "table2_comparisons": M.preregistered_comparisons(
                self.table, self.runs, self.controls, B=self.B
            ),
            "table3_cost_ratios": M.h3_cost_ratios(self.table, B=self.B),
            "table4_failure_modes": M.failure_modes(self.runs),
            "h1_budget_matching": M.h1_table(self.table, self.controls, B=self.B),
            "h3_verdict": M.h3_verdict(M.h3_cost_ratios(self.table, B=self.B)),
            "judge_confusion": M.judge_confusion(self.runs),
            "dropped_runs": self.ds.drop_report(),
        }
        for benchmark, curve in self.curves.items():
            tables[f"vote_curve_{benchmark}"] = curve.aggregate.reset_index()
        return tables

    def build_rq4(self) -> tuple[dict, dict]:
        features = R4.build_features(self.runs)
        if not len(features) or features["y"].sum() == 0:
            return {"rq4_features": features}, {}
        results = R4.fit_all(features)
        policies = pd.concat(
            [
                R4.routing_policies(features, results["first_attempt"], TOKEN_COL),
                R4.routing_policies(
                    features,
                    results["first_attempt_plus_agreement"],
                    TOKEN_COL,
                    include_agreement_cost=True,
                ),
                R4.routing_policies(features, results["first_attempt"], PRIMARY_COST),
            ],
            ignore_index=True,
        )
        tables = {
            "rq4_features": features,
            "rq4_auc": pd.DataFrame([r.as_row() for r in results.values()]),
            "rq4_coefficients": pd.DataFrame(
                [
                    {"model": name, "feature": f, "coefficient": c}
                    for name, res in results.items()
                    for f, c in res.coefficients.items()
                ]
            ),
            "rq4_routing_policies": policies,
        }
        return tables, results

    # ------------------------------------------------------------------ figures

    def figure1_pareto(self) -> None:
        benchmarks = sorted(self.curves)
        fig, axes = plt.subplots(
            1, max(len(benchmarks), 1), figsize=(7 * max(len(benchmarks), 1), 5.5),
            squeeze=False,
        )
        for ax, benchmark in zip(axes[0], benchmarks):
            curve = self.curves[benchmark]
            agg = curve.aggregate
            ax.plot(
                agg[PRIMARY_COST], agg["acc"], "-o", color=VOTE_COLOR, ms=4, lw=1.5,
                label="react_vote@k (k=1..%d)" % int(agg.index.max()), zorder=3,
            )
            self._vote_band(ax, curve)

            for scaffold in config.SCAFFOLDS:
                arm = self.table[
                    (self.table["benchmark"] == benchmark)
                    & (self.table["scaffold"] == scaffold)
                ]
                if not len(arm):
                    continue
                x, y = arm[PRIMARY_COST].mean(), arm["correct"].mean()
                xerr, yerr = self._errorbars(arm)
                ax.errorbar(
                    x, y, xerr=xerr, yerr=yerr, fmt="o", ms=9, capsize=3,
                    color=SCAFFOLD_COLORS.get(scaffold, "#666666"), label=scaffold, zorder=4,
                )

            pareto = self._pareto_frontier(benchmark)
            if len(pareto) > 1:
                ax.plot(pareto[:, 0], pareto[:, 1], "--", color="#999999", lw=1,
                        label="Pareto frontier", zorder=2)

            ax.set_xscale("log")
            ax.set_xlabel(f"mean {PRIMARY_COST} per task (USD, log scale)")
            ax.set_ylabel("accuracy (EM)")
            ax.set_title(benchmark)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, loc="lower right")
        fig.suptitle("Fig 1 — accuracy vs cost; is any scaffold above the vote@k curve?")
        self._save(fig, "fig1_pareto")

    def _vote_band(self, ax, curve: VoteCurve) -> None:
        from .stats import bootstrap_mean

        lo, hi = [], []
        for k in curve.aggregate.index:
            at_k = curve.per_task[curve.per_task["k"] == k]
            res = bootstrap_mean(at_k["acc"], B=min(self.B, 500))
            lo.append(res.ci_lo)
            hi.append(res.ci_hi)
        ax.fill_between(curve.aggregate[PRIMARY_COST], lo, hi, color=VOTE_COLOR, alpha=0.12,
                        zorder=1)

    def _errorbars(self, arm: pd.DataFrame):
        from .stats import bootstrap_mean

        acc = bootstrap_mean(arm["correct"], B=min(self.B, 500))
        cost = bootstrap_mean(arm[PRIMARY_COST], B=min(self.B, 500))
        return (
            [[cost.point - cost.ci_lo], [cost.ci_hi - cost.point]],
            [[acc.point - acc.ci_lo], [acc.ci_hi - acc.point]],
        )

    def _pareto_frontier(self, benchmark: str) -> np.ndarray:
        points = []
        curve = self.curves[benchmark]
        points += list(zip(curve.aggregate[PRIMARY_COST], curve.aggregate["acc"]))
        for scaffold in config.SCAFFOLDS:
            arm = self.table[
                (self.table["benchmark"] == benchmark) & (self.table["scaffold"] == scaffold)
            ]
            if len(arm):
                points.append((arm[PRIMARY_COST].mean(), arm["correct"].mean()))
        points.sort()
        frontier, best = [], -np.inf
        for x, y in points:
            if y > best:
                frontier.append((x, y))
                best = y
        return np.array(frontier)

    def figure2_reflexion(self) -> None:
        benchmarks = sorted(self.table["benchmark"].unique())
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))

        acc_ax, cost_ax, trials_ax, conf_ax = axes.ravel()
        width = 0.35
        x = np.arange(len(benchmarks))
        for i, scaffold in enumerate(("reflexion_gt", "reflexion_self")):
            accs = [
                self.table[
                    (self.table["benchmark"] == b) & (self.table["scaffold"] == scaffold)
                ]["correct"].mean()
                for b in benchmarks
            ]
            costs = [
                self.table[
                    (self.table["benchmark"] == b) & (self.table["scaffold"] == scaffold)
                ][PRIMARY_COST].mean()
                for b in benchmarks
            ]
            acc_ax.bar(x + i * width, accs, width, label=scaffold,
                       color=SCAFFOLD_COLORS[scaffold])
            cost_ax.bar(x + i * width, costs, width, label=scaffold,
                        color=SCAFFOLD_COLORS[scaffold])
        for ax, title, ylabel in (
            (acc_ax, "accuracy", "EM"),
            (cost_ax, f"cost ({PRIMARY_COST})", "USD/task"),
        ):
            ax.set_xticks(x + width / 2)
            ax.set_xticklabels(benchmarks)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3, axis="y")

        for scaffold in ("reflexion_gt", "reflexion_self"):
            trials = self.runs[self.runs["scaffold"] == scaffold]["n_trials"].dropna()
            if len(trials):
                trials_ax.hist(
                    trials, bins=np.arange(0.5, config.REFLEXION_MAX_TRIALS + 1.5),
                    alpha=0.6, label=scaffold, color=SCAFFOLD_COLORS[scaffold],
                )
        trials_ax.set_title("distribution of trials")
        trials_ax.set_xlabel("n_trials")
        trials_ax.legend(fontsize=8)
        trials_ax.grid(alpha=0.3, axis="y")

        confusion = M.judge_confusion(self.runs)
        if len(confusion):
            labels = ["false accept", "false reject"]
            for i, benchmark in enumerate(confusion["benchmark"]):
                row = confusion.iloc[i]
                conf_ax.bar(
                    np.arange(2) + i * 0.35,
                    [row["false_accept_rate"], row["false_reject_rate"]],
                    0.35, label=benchmark,
                )
            conf_ax.set_xticks(np.arange(2) + 0.175)
            conf_ax.set_xticklabels(labels)
            conf_ax.set_ylabel("rate")
        conf_ax.set_title("self-judge errors vs gold")
        conf_ax.legend(fontsize=8)
        conf_ax.grid(alpha=0.3, axis="y")

        fig.suptitle("Fig 2 — RQ2: what does the ground-truth stopping oracle buy?")
        fig.tight_layout()
        self._save(fig, "fig2_reflexion")

    def figure3_cost_measures(self, ratios: pd.DataFrame) -> None:
        if not len(ratios):
            return
        benchmarks = sorted(ratios["benchmark"].unique())
        fig, axes = plt.subplots(
            1, len(benchmarks), figsize=(7 * len(benchmarks), 5), squeeze=False
        )
        measures = [m for m in M.H3_COST_COLS if m in set(ratios["measure"])]
        for ax, benchmark in zip(axes[0], benchmarks):
            scaffolds = [s for s in config.SCAFFOLDS if s != M.BASELINE]
            width = 0.8 / len(measures)
            x = np.arange(len(scaffolds))
            for j, measure in enumerate(measures):
                values, lo, hi = [], [], []
                for scaffold in scaffolds:
                    sel = ratios[
                        (ratios["benchmark"] == benchmark)
                        & (ratios["scaffold"] == scaffold)
                        & (ratios["measure"] == measure)
                    ]
                    values.append(sel["ratio"].iloc[0] if len(sel) else np.nan)
                    lo.append(values[-1] - sel["ci_lo"].iloc[0] if len(sel) else 0)
                    hi.append(sel["ci_hi"].iloc[0] - values[-1] if len(sel) else 0)
                ax.bar(x + j * width, values, width, yerr=[lo, hi], capsize=2, label=measure)
            ax.axhline(1.0, color="black", lw=1, ls="--")
            ax.set_xticks(x + 0.4 - width / 2)
            ax.set_xticklabels(scaffolds, rotation=20, ha="right")
            ax.set_ylabel("cost relative to ReAct")
            ax.set_title(benchmark)
            ax.grid(alpha=0.3, axis="y")
            ax.legend(fontsize=7)
        fig.suptitle("Fig 3 — H3: do cost rankings depend on the cost measure?")
        fig.tight_layout()
        self._save(fig, "fig3_cost_measures")

    def figure4_budget_curve(self) -> None:
        """Cumulative accuracy against a token budget, cheapest task first."""
        benchmarks = sorted(self.table["benchmark"].unique())
        fig, axes = plt.subplots(
            1, len(benchmarks), figsize=(7 * len(benchmarks), 5), squeeze=False
        )
        for ax, benchmark in zip(axes[0], benchmarks):
            for scaffold in config.SCAFFOLDS:
                arm = self.table[
                    (self.table["benchmark"] == benchmark)
                    & (self.table["scaffold"] == scaffold)
                ].sort_values(TOKEN_COL)
                if not len(arm):
                    continue
                ax.plot(
                    np.cumsum(arm[TOKEN_COL]),
                    np.cumsum(arm["correct"]),
                    label=scaffold,
                    color=SCAFFOLD_COLORS.get(scaffold, "#666666"),
                )
            ax.set_xlabel("cumulative tokens (cheapest task first)")
            ax.set_ylabel("cumulative tasks solved")
            ax.set_title(benchmark)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle("Fig 4 — cumulative accuracy against a token budget")
        fig.tight_layout()
        self._save(fig, "fig4_budget_curve")

    def figure5_rq4(self, results: dict, policies: pd.DataFrame) -> None:
        if not results:
            return
        fig, (roc_ax, pareto_ax) = plt.subplots(1, 2, figsize=(13, 5.5))
        for name, result in results.items():
            points = R4.roc_points(result)
            if len(points):
                roc_ax.plot(points["fpr"], points["tpr"],
                            label=f"{name} (AUC={result.auc:.3f})")
        roc_ax.plot([0, 1], [0, 1], "--", color="#999999", lw=1)
        roc_ax.set_xlabel("false positive rate")
        roc_ax.set_ylabel("true positive rate")
        roc_ax.set_title("RQ4 — predicting when Reflexion-Self fixes a failure")
        roc_ax.legend(fontsize=8)
        roc_ax.grid(alpha=0.3)

        sel = policies[
            (policies["cost_col"] == TOKEN_COL) & (~policies["includes_agreement_cost"])
        ]
        for _, row in sel.iterrows():
            pareto_ax.scatter(row["cost"], row["accuracy"], s=60)
            pareto_ax.annotate(
                row["policy"], (row["cost"], row["accuracy"]),
                textcoords="offset points", xytext=(6, 4), fontsize=8,
            )
        pareto_ax.set_xlabel("mean tokens per task")
        pareto_ax.set_ylabel("accuracy")
        pareto_ax.set_title("routing policies on the cost/accuracy plane")
        pareto_ax.grid(alpha=0.3)
        fig.suptitle("Fig 5 — RQ4")
        fig.tight_layout()
        self._save(fig, "fig5_rq4")

    def _save(self, fig, name: str) -> None:
        fig.savefig(self.out / f"{name}.png", dpi=160, bbox_inches="tight")
        fig.savefig(self.out / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------ summary

    def write_summary(self, tables: dict[str, pd.DataFrame], h4: dict) -> Path:
        lines = [
            f"# Analysis summary — run tag `{self.ds.run_tag}`",
            "",
            f"- runs analysed: **{len(self.runs)}** "
            f"(dropped for `status != ok`: {len(self.ds.dropped)})",
            f"- LLM calls: **{len(self.ds.calls)}**",
            f"- empirical cache-read share (Σcache_read / ΣI): "
            f"**{empirical_cache_read_share(self.ds.calls):.4f}**",
            "",
            "## Dropped runs per scaffold (must stay < 2%)",
            "",
            tables["dropped_runs"].to_markdown(index=False, floatfmt=".4f"),
            "",
            "## Table 1 — metrics (Fig 1)",
            "",
            _md(tables["table1_metrics"], [
                "benchmark", "arm", "n_tasks", "acc", "acc_ci_lo", "acc_ci_hi", "f1",
                "tokens_total", "n_llm_calls", "usd_nocache", "usd_empirical",
                ideal_col(1024), ideal_col(4096), "cop_usd_nocache",
            ]),
            "",
            "## Table 2 — the nine pre-registered comparisons (Holm-corrected per benchmark)",
            "",
            _md(tables["table2_comparisons"]),
            "",
            "## H1 — budget matching (RQ1)",
            "",
            _md(tables["h1_budget_matching"]),
            "",
            "## H2 — the oracle premium (RQ2)",
            "",
            _md(tables["judge_confusion"]),
            "",
            "## Table 3 / H3 — cost ratios by cost measure (RQ3)",
            "",
            _md(tables["h3_verdict"]),
            "",
            "## Table 4 — failure modes",
            "",
            _md(tables["table4_failure_modes"]),
            "",
            "## H4 — predictability (RQ4)",
            "",
            "```json",
            json.dumps(h4, indent=2, default=str),
            "```",
            "",
            "Figures: `fig1_pareto`, `fig2_reflexion`, `fig3_cost_measures`, "
            "`fig4_budget_curve`, `fig5_rq4` (PNG + PDF in this directory).",
            "",
        ]
        path = self.out / "SUMMARY.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    # ------------------------------------------------------------------ run it all

    def run(self) -> Path:
        tables = self.build_tables()
        rq4_tables, rq4_results = self.build_rq4()
        tables.update(rq4_tables)

        for name, frame in tables.items():
            frame.to_csv(self.out / f"{name}.csv", index=False)

        self.figure1_pareto()
        self.figure2_reflexion()
        self.figure3_cost_measures(tables["table3_cost_ratios"])
        self.figure4_budget_curve()
        if rq4_results:
            self.figure5_rq4(rq4_results, tables["rq4_routing_policies"])

        h4 = R4.h4_verdict(rq4_results) if rq4_results else {"h4_supported": None}
        return self.write_summary(tables, h4)


def _md(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    if not len(frame):
        return "_(no rows)_"
    if columns:
        columns = [c for c in columns if c in frame.columns]
        frame = frame[columns]
    return frame.to_markdown(index=False, floatfmt=".4f")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m acs.analysis.report")
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--out", default=None, help="default: results/{run_tag}/analysis")
    ap.add_argument("--boot-b", type=int, default=config.BOOT_B)
    ap.add_argument("--vote-draws", type=int, default=config.VOTE_DRAWS)
    args = ap.parse_args(argv)

    ds = load(args.run_tag)
    out = Path(args.out) if args.out else config.run_dir(args.run_tag) / "analysis"
    path = Report(ds, out, B=args.boot_b, vote_draws=args.vote_draws).run()
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
