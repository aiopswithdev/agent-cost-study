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

# Categorical palette: slots 1, 2, 3, 7 of the validated reference palette. These four pass
# every check (lightness, chroma, colour-blind separation and normal-vision separation)
# even when *all* pairs are compared, as in a scatter. Reflexion-GT and Reflexion-Self are
# the same mechanism with a different stop signal, so they share one hue and are told
# apart by fill: GT solid, Self hollow / hatched. A colour always means the same scaffold
# in every figure; things that are not scaffolds are drawn in neutral greys.
SCAFFOLD_COLORS = {
    "react": "#2a78d6",
    "rewoo": "#eb6834",
    "plan_exec": "#1baf7a",
    "reflexion_gt": "#4a3aa7",
    "reflexion_self": "#4a3aa7",
}
SCAFFOLD_LABELS = {
    "react": "ReAct",
    "rewoo": "ReWOO",
    "plan_exec": "Plan-and-Execute",
    "reflexion_gt": "Reflexion-GT (answer key)",
    "reflexion_self": "Reflexion-Self (own judge)",
}
HOLLOW = {"reflexion_self"}
#: Direct-label nudges for Fig 1 where the default position collides with another mark.
FIG1_LABEL_OFFSETS = {
    ("gsm8k", "react"): (-10, -14),
    ("hotpotqa", "plan_exec"): (-10, 10),
}
#: The same, for the smaller paper panels (in points).
FIG1_PAPER_OFFSETS = {
    ("gsm8k", "react"): (-6, -8),
    ("gsm8k", "reflexion_gt"): (-6, 3),
    ("gsm8k", "reflexion_self"): (5, -9),
    ("hotpotqa", "plan_exec"): (-6, 6),
    ("hotpotqa", "reflexion_self"): (6, 4),
}
INK = "#2b2b2b"  # primary text / the ReAct budget curve
MUTED = "#6b6b6b"  # secondary marks that are not scaffolds
LIGHT = "#b5b5b0"
GRID = "#e6e6e3"
VOTE_COLOR = INK
SURFACE = "white"


def _style(ax, lw: float = 0.8) -> None:
    """Recessive, solid hairline grid and light spines."""
    ax.grid(True, color=GRID, lw=lw, ls="-", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(LIGHT)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def _marker_kw(scaffold: str, ms: float = 9, mew: float = 2) -> dict:
    color = SCAFFOLD_COLORS.get(scaffold, MUTED)
    return dict(
        color=color, marker="o", ms=ms, mec=color, mew=mew,
        mfc=SURFACE if scaffold in HOLLOW else color,
    )


def _bar_kw(scaffold: str, lw: float = 1.5) -> dict:
    color = SCAFFOLD_COLORS.get(scaffold, MUTED)
    if scaffold in HOLLOW:
        return dict(facecolor=SURFACE, edgecolor=color, hatch="////", linewidth=lw)
    return dict(color=color)


def _ratio_axis(ax) -> None:
    """Log-scale 'relative to ReAct' axis with plain-number ticks (0.25, 0.5, 1, 2, …)."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    ax.set_yscale("log")
    ticks = [0.125, 0.25, 0.5, 1, 2, 4, 8]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))


def _end_label_offset(value: float) -> tuple[int, int]:
    """Lift an end label off the dashed 'same cost as ReAct' line when it sits on it."""
    return (9, 9) if abs(np.log(value)) < 0.08 else (9, 0)


def _bottom_legend(fig, ax, ncol: int = 4, fontsize: float = 8.5) -> None:
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False,
               fontsize=fontsize, bbox_to_anchor=(0.5, -0.02))


def _line_kw(scaffold: str, lw: float = 2) -> dict:
    return dict(
        color=SCAFFOLD_COLORS.get(scaffold, MUTED), lw=lw,
        ls="--" if scaffold in HOLLOW else "-", solid_capstyle="round",
    )


# --------------------------------------------------------------------------- paper mode
# `--paper` redraws every figure at its printed size in the ACL template, so no figure is
# scaled by LaTeX and every font prints at its nominal size. Without the flag the report
# figures are unchanged.

#: ACL two-column page: \textwidth = 16 cm, \columnwidth = (16 - 0.6) / 2 = 7.7 cm.
COLUMN_IN = 7.7 / 2.54
TEXT_IN = 16 / 2.54
#: Per figure: (rows, cols, (width, height) in inches). Column-width figures stack their
#: benchmark panels vertically; full-width figures keep them side by side.
PAPER_LAYOUT = {
    "fig1_pareto": (1, 2, (TEXT_IN, 2.6)),
    "fig2_reflexion": (1, 4, (TEXT_IN, 2.1)),
    "fig3_cost_measures": (2, 1, (COLUMN_IN, 4.6)),
    "fig4_budget_curve": (2, 1, (COLUMN_IN, 4.2)),
    "fig5_rq4": (1, 2, (TEXT_IN, 2.5)),
    "fig6_rq1_gains": (2, 1, (COLUMN_IN, 3.6)),
    "fig7_rq3_threshold_sweep": (2, 1, (COLUMN_IN, 4.6)),
}
#: Times-metric TrueType serif to match the body text, embedded as TrueType (no Type 3
#: fonts; OpenType/CFF fonts such as Nimbus Roman embed in a form poppler flags).
PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
#: Explicit sizes in the drawing code are report sizes; paper mode scales them by these.
PAPER_SCALE = {"font": 0.8, "marker": 0.55, "line": 0.6}
PAPER_BENCHMARK = {"hotpotqa": "HotpotQA", "gsm8k": "GSM8K"}


# --------------------------------------------------------------------------- driver


class Report:
    def __init__(
        self,
        ds: Dataset,
        out_dir: Path,
        B: int = config.BOOT_B,
        vote_draws: int = config.VOTE_DRAWS,
        paper: bool = False,
    ) -> None:
        self.ds = ds
        self.paper = paper
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

    # ------------------------------------------------------------------ figure helpers
    # Report sizes are written in the drawing code; in paper mode they are scaled down.

    def _fs(self, size: float) -> float:
        return size * PAPER_SCALE["font"] if self.paper else size

    def _ms(self, size: float) -> float:
        return size * PAPER_SCALE["marker"] if self.paper else size

    def _lw(self, width: float) -> float:
        return width * PAPER_SCALE["line"] if self.paper else width

    def _label(self, scaffold: str) -> str:
        """Paper figures use the short name; the caption explains GT vs Self."""
        label = SCAFFOLD_LABELS[scaffold]
        return label.split(" (")[0] if self.paper else label

    def _mk(self, scaffold: str, ms: float = 9) -> dict:
        return _marker_kw(scaffold, ms=self._ms(ms), mew=self._lw(2))

    def _bar(self, scaffold: str) -> dict:
        return _bar_kw(scaffold, lw=self._lw(1.5))

    def _style(self, ax) -> None:
        _style(ax, lw=self._lw(0.8))

    def _halo(self) -> dict:
        """White outline behind direct labels so they stay legible over lines (paper only)."""
        if not self.paper:
            return {}
        from matplotlib import patheffects

        return {"path_effects": [patheffects.withStroke(linewidth=2, foreground=SURFACE)]}

    def _title(self, ax, report: str, paper: str | None = None) -> None:
        ax.set_title(paper if self.paper and paper is not None else report, color=INK)

    def _suptitle(self, fig, text: str) -> None:
        """Report figures carry their own title; paper figures get it from the caption."""
        if not self.paper:
            fig.suptitle(text, color=INK)

    def _subplots(self, name: str, nrows: int, ncols: int, figsize: tuple, **kw):
        """Axes as a flat array, in the paper layout when in paper mode."""
        if self.paper:
            nrows, ncols, figsize = PAPER_LAYOUT[name]
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, **kw)
        return fig, axes.ravel()

    # ------------------------------------------------------------------ figures

    def figure1_pareto(self) -> None:
        """RQ1 — every scaffold against ReAct's budget curve (accuracy vs cost)."""
        benchmarks = sorted(self.curves)
        n = max(len(benchmarks), 1)
        fig, axes = self._subplots("fig1_pareto", 1, n, (7 * n, 5.5))
        for i, (ax, benchmark) in enumerate(zip(axes, benchmarks)):
            curve = self.curves[benchmark]
            agg = curve.aggregate
            k_max = int(agg.index.max())
            ax.plot(
                agg[PRIMARY_COST], agg["acc"], "-o", color=VOTE_COLOR, ms=self._ms(4),
                lw=self._lw(2),
                label=(f"ReAct vote@k (k = 1..{k_max})" if self.paper
                       else f"ReAct, majority of k attempts (k = 1..{k_max})"),
                zorder=3,
            )
            self._vote_band(ax, curve)
            ax.annotate(f"k={k_max}", (agg[PRIMARY_COST].iloc[-1],
                        agg["acc"].iloc[-1]), textcoords="offset points", xytext=(4, 6),
                        fontsize=self._fs(8), color=MUTED)

            pareto = self._pareto_frontier(benchmark)
            if len(pareto) > 1:
                ax.plot(pareto[:, 0], pareto[:, 1], ":", color=MUTED, lw=self._lw(1.2),
                        label="Pareto frontier", zorder=2)

            for scaffold in config.SCAFFOLDS:
                arm = self.table[
                    (self.table["benchmark"] == benchmark)
                    & (self.table["scaffold"] == scaffold)
                ]
                if not len(arm):
                    continue
                x, y = arm[PRIMARY_COST].mean(), arm["correct"].mean()
                xerr, yerr = self._errorbars(arm)
                kw = self._mk(scaffold)
                ax.errorbar(x, y, xerr=xerr, yerr=yerr, ls="none", capsize=self._ms(3),
                            elinewidth=self._lw(1.5), label=self._label(scaffold), zorder=4,
                            **kw)
                if self.paper:
                    offset = FIG1_PAPER_OFFSETS.get((benchmark, scaffold), (5, -7))
                else:
                    offset = FIG1_LABEL_OFFSETS.get((benchmark, scaffold), (8, -12))
                ax.annotate(SCAFFOLD_LABELS[scaffold].split(" (")[0], (x, y),
                            textcoords="offset points", xytext=offset, fontsize=self._fs(8),
                            ha="right" if offset[0] < 0 else "left", color=INK, zorder=5,
                            **self._halo())

            ax.set_xscale("log")
            if self.paper:
                # Dollars per 1,000 questions on 1-2-5 ticks: short labels that never collide.
                from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

                ax.xaxis.set_major_locator(FixedLocator(
                    [m * 10.0 ** e for e in range(-5, 0) for m in (1, 2, 5)]))
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 1000:g}"))
                ax.xaxis.set_minor_formatter(NullFormatter())
                ax.set_xlabel("USD per 1,000 questions (no cache, log scale)")
                ax.set_ylabel("accuracy" if i == 0 else "")
            else:
                ax.set_xlabel("mean cost per question, USD, no-cache list price (log scale)")
                ax.set_ylabel("accuracy (exact match)")
            self._title(ax, benchmark, PAPER_BENCHMARK.get(benchmark, benchmark))
            self._style(ax)
            if not self.paper:
                ax.legend(fontsize=7.5, loc="lower right", frameon=False)
        if self.paper:
            # Scaffolds are labelled directly; the legend only explains the reference marks.
            handles, labels = axes[0].get_legend_handles_labels()
            keep = [j for j, lab in enumerate(labels) if lab.startswith(("ReAct vote", "Pareto"))]
            axes[0].legend([handles[j] for j in keep], [labels[j] for j in keep],
                           loc="lower right", frameon=False)
        self._suptitle(fig, "Fig 1 — RQ1: accuracy vs cost. A scaffold beats budget-matched "
                       "ReAct only if it sits above the black curve")
        fig.tight_layout()
        self._save(fig, "fig1_pareto")

    def _vote_band(self, ax, curve: VoteCurve) -> None:
        from .stats import bootstrap_mean

        lo, hi = [], []
        for k in curve.aggregate.index:
            at_k = curve.per_task[curve.per_task["k"] == k]
            res = bootstrap_mean(at_k["acc"], B=min(self.B, 500))
            lo.append(res.ci_lo)
            hi.append(res.ci_hi)
        ax.fill_between(curve.aggregate[PRIMARY_COST], lo, hi, color=VOTE_COLOR, alpha=0.10,
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
        """RQ2 — what the answer-key oracle buys, and why the self-judge falls short."""
        from matplotlib.ticker import PercentFormatter

        benchmarks = sorted(self.table["benchmark"].unique())
        fig, axes = self._subplots("fig2_reflexion", 2, 2, (12, 9))
        acc_ax, cost_ax, trials_ax, conf_ax = axes
        width = 0.36
        x = np.arange(len(benchmarks))
        pair = ("reflexion_gt", "reflexion_self")
        # Paper: accuracy as plain percentages and cost per 1,000 questions, so the bar
        # labels stay short enough for a quarter-width panel.
        if self.paper:
            acc_fmt, cost_fmt, cost_scale = "{:.1f}", "{:.1f}", 1000
        else:
            acc_fmt, cost_fmt, cost_scale = "{:.1%}", "${:.4f}", 1
        for i, scaffold in enumerate(pair):
            sel = [self.table[(self.table["benchmark"] == b)
                              & (self.table["scaffold"] == scaffold)] for b in benchmarks]
            accs = [s_["correct"].mean() for s_ in sel]
            costs = [s_[PRIMARY_COST].mean() * cost_scale for s_ in sel]
            for ax, vals, fmt, shown in ((acc_ax, accs, acc_fmt, 100 if self.paper else 1),
                                         (cost_ax, costs, cost_fmt, 1)):
                bars = ax.bar(x + i * width, vals, width * 0.94, label=SCAFFOLD_LABELS[scaffold],
                              **self._bar(scaffold))
                for bar, v in zip(bars, vals):
                    ax.annotate(fmt.format(v * shown), (bar.get_x() + bar.get_width() / 2, v),
                                textcoords="offset points", xytext=(0, 3), ha="center",
                                fontsize=self._fs(8), color=INK)
        panels = (
            (acc_ax, "accuracy", "exact match", "accuracy", "accuracy (%)"),
            (cost_ax, "cost per question (no-cache list price)", "USD",
             "cost (no cache)", "USD per 1,000 questions"),
        )
        for ax, title, ylabel, paper_title, paper_ylabel in panels:
            ax.set_xticks(x + width / 2)
            ax.set_xticklabels([PAPER_BENCHMARK.get(b, b) for b in benchmarks]
                               if self.paper else benchmarks)
            self._title(ax, title, paper_title)
            ax.set_ylabel(paper_ylabel if self.paper else ylabel)
            if not self.paper:
                ax.legend(fontsize=8, frameon=False)
            self._style(ax)
        if self.paper:
            acc_ax.yaxis.set_major_formatter(PercentFormatter(1.0, symbol=""))
            acc_ax.set_ylim(0, 1.12)
            cost_ax.set_ylim(0, cost_ax.get_ylim()[1] * 1.1)

        bins = np.arange(1, config.REFLEXION_MAX_TRIALS + 1)
        for i, scaffold in enumerate(pair):
            trials = self.runs[self.runs["scaffold"] == scaffold]["n_trials"].dropna()
            counts = [(trials == b).sum() for b in bins]
            trials_ax.bar(bins + (i - 0.5) * width, counts, width * 0.94,
                          label=SCAFFOLD_LABELS[scaffold], **self._bar(scaffold))
        trials_ax.set_xticks(bins)
        self._title(trials_ax, "number of trials used per run (both benchmarks)",
                    "trials used per run")
        trials_ax.set_xlabel("trials")
        trials_ax.set_ylabel("runs")
        if not self.paper:
            trials_ax.legend(fontsize=8, frameon=False)
        self._style(trials_ax)

        confusion = M.judge_confusion(self.runs)
        if len(confusion):
            if self.paper:
                labels = ["false\naccept", "false\nreject"]
            else:
                labels = ["accepts a WRONG answer\n(false accept)",
                          "rejects a RIGHT answer\n(false reject)"]
            greys = (MUTED, LIGHT)
            for i, benchmark in enumerate(confusion["benchmark"]):
                row = confusion.iloc[i]
                vals = [row["false_accept_rate"], row["false_reject_rate"]]
                bars = conf_ax.bar(np.arange(2) + i * width, vals, width * 0.94,
                                   color=greys[i % 2],
                                   label=PAPER_BENCHMARK.get(benchmark, benchmark)
                                   if self.paper else benchmark)
                for bar, v in zip(bars, vals):
                    conf_ax.annotate(f"{v * 100:.1f}" if self.paper else f"{v:.1%}",
                                     (bar.get_x() + bar.get_width() / 2, v),
                                     textcoords="offset points", xytext=(0, 3),
                                     ha="center", fontsize=self._fs(8), color=INK)
            conf_ax.set_xticks(np.arange(2) + width / 2)
            conf_ax.set_xticklabels(labels, fontsize=self._fs(8))
            if self.paper:
                conf_ax.yaxis.set_major_formatter(PercentFormatter(1.0, symbol=""))
                conf_ax.set_ylabel("rate (%)")
                conf_ax.set_ylim(0, 1.12)
            else:
                conf_ax.set_ylabel("rate")
                conf_ax.set_ylim(0, 1.08)
        self._title(conf_ax, "Reflexion-Self's judge, checked against the true answers",
                    "Self judge vs. true answers")
        if self.paper:
            # Compact, in the empty space above the (near-zero) false-reject bars.
            conf_ax.legend(fontsize=self._fs(8), frameon=False, loc="center right",
                           bbox_to_anchor=(1.06, 0.55), handlelength=0.9, handletextpad=0.4)
        else:
            conf_ax.legend(fontsize=self._fs(8), frameon=False)
        self._style(conf_ax)

        if self.paper:
            handles, labels = acc_ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
                       bbox_to_anchor=(0.5, 1.0))
            fig.tight_layout(rect=(0, 0, 1, 0.9))
        else:
            fig.suptitle("Fig 2 — RQ2: what does the answer-key (oracle) stopping rule buy?",
                         color=INK)
            fig.tight_layout()
        self._save(fig, "fig2_reflexion")

    # Pre-registered cost measures, in the order a reader should scan them.
    MEASURE_LABELS = {
        TOKEN_COL: "tokens",
        "usd_nocache": "$ no cache",
        ideal_col(4096): "$ cache ≥4096",
        ideal_col(1024): "$ cache ≥1024",
        ideal_col(0): "$ cache, any length",
        "usd_empirical": "$ DeepSeek bill",
    }

    def figure3_cost_measures(self, ratios: pd.DataFrame) -> None:
        """RQ3 — each scaffold's cost relative to ReAct, under every pre-registered measure."""
        if not len(ratios):
            return
        benchmarks = sorted(ratios["benchmark"].unique())
        fig, axes = self._subplots("fig3_cost_measures", 1, len(benchmarks),
                                   (7.5 * len(benchmarks), 5.2),
                                   **({"sharex": True} if self.paper else {}))
        measures = [m for m in self.MEASURE_LABELS if m in set(ratios["measure"])]
        x = np.arange(len(measures))
        scaffolds = [s_ for s_ in config.SCAFFOLDS if s_ != M.BASELINE]
        for i, (ax, benchmark) in enumerate(zip(axes, benchmarks)):
            for j, scaffold in enumerate(scaffolds):
                sel = ratios[(ratios["benchmark"] == benchmark)
                             & (ratios["scaffold"] == scaffold)].set_index("measure")
                vals = np.array([sel.loc[m, "ratio"] for m in measures])
                lo = vals - np.array([sel.loc[m, "ci_lo"] for m in measures])
                hi = np.array([sel.loc[m, "ci_hi"] for m in measures]) - vals
                offset = (j - 1.5) * 0.06
                kw = self._mk(scaffold, ms=7)
                ax.errorbar(x + offset, vals, yerr=[lo, hi], capsize=self._ms(2),
                            elinewidth=self._lw(1.2), label=self._label(scaffold), zorder=3,
                            ls="--" if scaffold in HOLLOW else "-", lw=self._lw(1.5), **kw)
                ax.annotate(SCAFFOLD_LABELS[scaffold].split(" (")[0],
                            (x[-1] + offset, vals[-1]), textcoords="offset points",
                            xytext=_end_label_offset(vals[-1]), va="center",
                            fontsize=self._fs(8), color=INK)
            ax.axhline(1.0, color=INK, lw=self._lw(1), ls="--", zorder=2)
            ax.annotate("same cost as ReAct", (x[0], 1.0), textcoords="offset points",
                        xytext=(0, -11 if not self.paper else -8), fontsize=self._fs(8),
                        color=MUTED)
            _ratio_axis(ax)
            ax.set_xticks(x)
            ax.set_xticklabels([self.MEASURE_LABELS[m] for m in measures],
                               rotation=30 if self.paper else 20, ha="right",
                               fontsize=self._fs(8.5))
            if self.paper:
                # Room on the right for the end labels.
                ax.set_xlim(-0.4, len(measures) + 1.1)
                ax.set_ylabel("cost relative to ReAct")
            else:
                ax.set_xlim(-0.4, len(measures) - 0.2)
                ax.set_ylabel("cost relative to ReAct (log scale)")
            self._title(ax, benchmark, PAPER_BENCHMARK.get(benchmark, benchmark))
            self._style(ax)
        if self.paper:
            fig.tight_layout()
        else:
            _bottom_legend(fig, axes[0])
            fig.suptitle("Fig 3 — RQ3: does a scaffold's cost relative to ReAct depend on how "
                         "cost is measured?", color=INK)
            fig.tight_layout(rect=(0, 0.06, 1, 1))
        self._save(fig, "fig3_cost_measures")

    def figure4_budget_curve(self) -> None:
        """Cumulative accuracy against a token budget, cheapest task first."""
        benchmarks = sorted(self.table["benchmark"].unique())
        fig, axes = self._subplots("fig4_budget_curve", 1, len(benchmarks),
                                   (7 * len(benchmarks), 5))
        for i, (ax, benchmark) in enumerate(zip(axes, benchmarks)):
            for scaffold in config.SCAFFOLDS:
                arm = self.table[
                    (self.table["benchmark"] == benchmark)
                    & (self.table["scaffold"] == scaffold)
                ].sort_values(TOKEN_COL)
                if not len(arm):
                    continue
                xs, ys = np.cumsum(arm[TOKEN_COL]), np.cumsum(arm["correct"])
                ax.plot(xs, ys, label=self._label(scaffold),
                        **_line_kw(scaffold, lw=self._lw(2)))
            if self.paper:
                ax.set_xlabel("cumulative tokens (cheapest question first)"
                              if i == len(benchmarks) - 1 else "")
                ax.set_ylabel("questions solved")
            else:
                ax.set_xlabel("cumulative tokens (cheapest question first)")
                ax.set_ylabel("cumulative questions solved")
            if self.paper:
                from matplotlib.ticker import FuncFormatter

                ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e6:g}M"))
            self._title(ax, benchmark, PAPER_BENCHMARK.get(benchmark, benchmark))
            self._style(ax)
            if not self.paper or i == 0:
                ax.legend(fontsize=self._fs(8), frameon=False)
        self._suptitle(fig, "Fig 4 — questions solved as a token budget is spent "
                       "(secondary view)")
        fig.tight_layout()
        self._save(fig, "fig4_budget_curve")

    def figure5_rq4(self, results: dict, policies: pd.DataFrame) -> None:
        if not results:
            return
        fig, (roc_ax, pareto_ax) = self._subplots("fig5_rq4", 1, 2, (13, 5.5))
        lw = self._lw(2)
        styles = {
            "benchmark_only": dict(color=LIGHT, ls=":", lw=lw),
            "first_attempt": dict(color=MUTED, ls="--", lw=lw),
            "first_attempt_plus_agreement": dict(color=INK, ls="-", lw=lw),
        }
        names = {
            "benchmark_only": "benchmark only (baseline)",
            "first_attempt": "first-attempt features",
            "first_attempt_plus_agreement": "first-attempt + agreement of 3 attempts",
        }
        if self.paper:
            names["first_attempt_plus_agreement"] = "+ agreement of 3 attempts"
        for name, result in results.items():
            points = R4.roc_points(result)
            if len(points):
                roc_ax.plot(points["fpr"], points["tpr"],
                            label=f"{names.get(name, name)} (AUC = {result.auc:.2f})",
                            **styles.get(name, dict(color=MUTED)))
        roc_ax.plot([0, 1], [0, 1], color=GRID, lw=self._lw(1))
        roc_ax.set_xlabel("false positive rate")
        roc_ax.set_ylabel("true positive rate")
        positives = int(results[next(iter(results))].y_true.sum())
        n = len(results[next(iter(results))].y_true)
        self._title(roc_ax,
                    "predicting when Reflexion-Self fixes a failed first attempt\n"
                    f"(only {positives} positive questions of {n})",
                    f"ROC: Reflexion-Self fixes a failed first attempt ({positives} of {n})")
        roc_ax.legend(fontsize=self._fs(8), frameon=False, loc="lower right",
                      **({"bbox_to_anchor": (1.0, 0.02)} if self.paper else {}))
        self._style(roc_ax)

        sel = policies[
            (policies["cost_col"] == TOKEN_COL) & (~policies["includes_agreement_cost"])
        ]
        readable = {"always_react": "always ReAct", "always_reflexion": "always Reflexion-Self"}
        grouped = sel.groupby(["cost", "accuracy"], sort=False)["policy"].apply(
            lambda ps: " = ".join(readable.get(p, p.replace("route_p>", "route if p > "))
                                  for p in ps))
        for (cost, acc), label in grouped.items():
            pareto_ax.plot(cost, acc, "o", ms=self._ms(9), color=INK, mfc=INK, mec=SURFACE,
                           mew=self._lw(2))
            if not self.paper:
                offset = (8, 4)
            else:
                offset = (5, -9) if label.startswith("always ReAct") else (5, 3)
            pareto_ax.annotate(label, (cost, acc), textcoords="offset points",
                               xytext=offset, fontsize=self._fs(8), color=INK)
        pareto_ax.set_xlabel("mean tokens per question")
        pareto_ax.set_ylabel("accuracy (both benchmarks pooled)")
        self._title(pareto_ax, "routing policies: send a question to Reflexion-Self "
                    "only if predicted to help", "routing policies (held-out folds)")
        pareto_ax.margins(x=0.25, y=0.2)
        if self.paper:
            pareto_ax.margins(x=0.35, y=0.2)
        self._style(pareto_ax)
        self._suptitle(fig, "Fig 5 — RQ4: can cheap first-attempt features predict the payoff?")
        fig.tight_layout()
        self._save(fig, "fig5_rq4")

    def figure6_rq1_gains(self, h1: pd.DataFrame) -> None:
        """RQ1 — each scaffold's gain over ReAct, before and after budget matching."""
        if not len(h1):
            return
        benchmarks = sorted(h1["benchmark"].unique())
        scaffolds = [s_ for s_ in config.SCAFFOLDS if s_ != M.BASELINE]
        fig, axes = self._subplots("fig6_rq1_gains", 1, len(benchmarks),
                                   (7 * len(benchmarks), 4.6), sharey=True)
        ms, mew = self._ms(10), self._lw(2)
        for i, (ax, benchmark) in enumerate(zip(axes, benchmarks)):
            for row_i, scaffold in enumerate(reversed(scaffolds)):
                r = h1[(h1["benchmark"] == benchmark) & (h1["scaffold"] == scaffold)]
                if not len(r):
                    continue
                r = r.iloc[0]
                color = SCAFFOLD_COLORS[scaffold]
                d0, d1 = 100 * r["delta_default"], 100 * r["delta_matched"]
                ax.plot([d0, d1], [row_i, row_i], color=color, lw=self._lw(2), zorder=2)
                ax.errorbar(d1, row_i, xerr=[[d1 - 100 * r["delta_matched_ci_lo"]],
                                             [100 * r["delta_matched_ci_hi"] - d1]],
                            fmt="o", ms=ms, color=color, mfc=color, mec=SURFACE, mew=mew,
                            capsize=self._ms(3), elinewidth=self._lw(1.5), zorder=4)
                ax.plot(d0, row_i, "o", ms=ms, color=color, mfc=SURFACE, mec=color, mew=mew,
                        zorder=3)
                if self.paper:
                    # Right of the naive dot and the CI, centred on the row.
                    right = max(d0, d1, 100 * r["delta_matched_ci_hi"])
                    ax.annotate(f"{d0:+.1f} → {d1:+.1f}", (right, row_i),
                                textcoords="offset points", xytext=(5, 0), va="center",
                                fontsize=self._fs(8), color=INK, **self._halo())
                else:
                    ax.annotate(f"{d0:+.1f} → {d1:+.1f}", (max(d0, d1), row_i),
                                textcoords="offset points", xytext=(10, 6),
                                fontsize=self._fs(8), color=INK)
            ax.axvline(0, color=INK, lw=self._lw(1), zorder=1)
            ax.set_yticks(range(len(scaffolds)))
            ax.set_yticklabels([self._label(s_) for s_ in reversed(scaffolds)])
            if self.paper:
                ax.set_xlabel("gain over ReAct (pp)" if i == len(benchmarks) - 1 else "")
                sub = h1[h1["benchmark"] == benchmark]
                lo = 100 * min(sub["delta_matched_ci_lo"].min(), sub["delta_default"].min())
                hi = 100 * max(sub["delta_matched_ci_hi"].max(), sub["delta_default"].max(), 0)
                span = hi - min(lo, 0)
                ax.set_xlim(min(lo, 0) - 0.08 * span, hi + 0.55 * span)
                ax.margins(y=0.15)
            else:
                ax.set_xlabel("accuracy gain over ReAct (percentage points)")
                ax.margins(x=0.18)
            self._title(ax, benchmark, PAPER_BENCHMARK.get(benchmark, benchmark))
            self._style(ax)
        lm, lmew = self._ms(9), self._lw(2)
        if self.paper:
            naive, matched = "vs one ReAct attempt (naive)", "vs ReAct at the same budget (95% CI)"
        else:
            naive, matched = ("vs ONE ReAct attempt (naive)",
                              "vs ReAct given the SAME budget (95% CI)")
        handles = [
            plt.Line2D([], [], ls="none", marker="o", ms=lm, mfc=SURFACE, mec=MUTED, mew=lmew,
                       label=naive),
            plt.Line2D([], [], ls="none", marker="o", ms=lm, mfc=MUTED, mec=SURFACE, mew=lmew,
                       label=matched),
        ]
        if self.paper:
            fig.legend(handles=handles, loc="lower center", ncol=1, frameon=False,
                       bbox_to_anchor=(0.5, 0.0))
            fig.tight_layout(rect=(0, 0.09, 1, 1))
        else:
            fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9,
                       bbox_to_anchor=(0.5, -0.04))
            fig.suptitle("Fig 6 — RQ1: how much of each scaffold's gain survives budget "
                         "matching?", color=INK)
            fig.tight_layout(rect=(0, 0.05, 1, 1))
        self._save(fig, "fig6_rq1_gains")

    # Exploratory (not pre-registered) — the cache-threshold sweep and dollar matching.
    SWEEP_L = (0, 256, 512, 768, 1024, 1536, 2048, 4096)

    def threshold_sweep(self) -> pd.DataFrame:
        from .costs import run_costs

        rc = run_costs(self.ds.calls, self.ds.ownership, self.ds.prompt_tokens, Ls=self.SWEEP_L)
        runs = self.ds.runs[["run_id", "benchmark", "scaffold", "replicate"]].merge(
            rc, on="run_id")
        runs = runs[~((runs["scaffold"] == M.BASELINE) & (runs["replicate"] >= config.R))]
        cols = ["tokens_total", "usd_empirical"] + [ideal_col(L) for L in self.SWEEP_L]
        # Every arm has the same number of runs per question, so the mean over runs equals
        # the mean of per-question means used everywhere else.
        means = runs.groupby(["benchmark", "scaffold"])[cols].mean()
        rows = []
        for benchmark in sorted(runs["benchmark"].unique()):
            base = means.loc[(benchmark, M.BASELINE)]
            for scaffold in config.OTHER_SCAFFOLDS:
                if (benchmark, scaffold) not in means.index:
                    continue
                ratio = means.loc[(benchmark, scaffold)] / base
                row = {"benchmark": benchmark, "scaffold": scaffold,
                       "tokens": ratio["tokens_total"]}
                row.update({f"L={L}": ratio[ideal_col(L)] for L in self.SWEEP_L})
                row["DeepSeek bill"] = ratio["usd_empirical"]
                rows.append(row)
        return pd.DataFrame(rows)

    def dollar_matched(self, table1: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for benchmark, curve in self.curves.items():
            v = curve.aggregate.reset_index()
            for scaffold in ("plan_exec", "reflexion_gt", "reflexion_self"):
                r = table1[(table1["benchmark"] == benchmark) & (table1["arm"] == scaffold)]
                if not len(r):
                    continue
                r = r.iloc[0]
                for col in (TOKEN_COL, "usd_empirical"):
                    xs, ks, acc = v[col].to_numpy(), v["k"].to_numpy(), v["acc"].to_numpy()
                    a = float(np.interp(r[col], xs, acc))
                    rows.append({"benchmark": benchmark, "scaffold": scaffold,
                                 "matched_on": col, "k_star": float(np.interp(r[col], xs, ks)),
                                 "beyond_pool": bool(r[col] > xs[-1]), "acc_X": r["acc"],
                                 "acc_vote": a, "delta": r["acc"] - a})
        return pd.DataFrame(rows)

    def figure7_threshold_sweep(self, sweep: pd.DataFrame) -> None:
        """RQ3, exploratory — relative cost as the provider's cache minimum L varies."""
        if not len(sweep):
            return
        benchmarks = sorted(sweep["benchmark"].unique())
        cols = ["tokens"] + [f"L={L}" for L in self.SWEEP_L] + ["DeepSeek bill"]
        xlabels = ["tokens"] + [f"{L}" for L in self.SWEEP_L] + ["DeepSeek\nbill"]
        # Visual gap separates the two reference columns from the sweep itself.
        x = np.array([0] + [1.6 + i for i in range(len(self.SWEEP_L))]
                     + [2.2 + len(self.SWEEP_L)])
        fig, axes = self._subplots("fig7_rq3_threshold_sweep", 1, len(benchmarks),
                                   (7.5 * len(benchmarks), 5.2),
                                   **({"sharex": True} if self.paper else {}))
        for i, (ax, benchmark) in enumerate(zip(axes, benchmarks)):
            sub = sweep[sweep["benchmark"] == benchmark].set_index("scaffold")
            for scaffold in config.OTHER_SCAFFOLDS:
                if scaffold not in sub.index:
                    continue
                vals = sub.loc[scaffold, cols].to_numpy(dtype=float)
                kw = self._mk(scaffold, ms=6)
                mid = slice(1, 1 + len(self.SWEEP_L))
                ax.plot(x[mid], vals[mid], ls="--" if scaffold in HOLLOW else "-",
                        lw=self._lw(2), label=self._label(scaffold), **kw)
                ax.plot(x[[0]], vals[[0]], ls="none", **kw)
                ax.plot(x[[-1]], vals[[-1]], ls="none", **kw)
                ax.annotate(SCAFFOLD_LABELS[scaffold].split(" (")[0], (x[-1], vals[-1]),
                            textcoords="offset points", xytext=_end_label_offset(vals[-1]),
                            va="center", fontsize=self._fs(8), color=INK)
            ax.axhline(1.0, color=INK, lw=self._lw(1), ls="--")
            ax.annotate("same cost as ReAct", (x[1], 1.0), textcoords="offset points",
                        xytext=(0, -11 if not self.paper else -8), fontsize=self._fs(8),
                        color=MUTED)
            ax.axvspan(x[1] - 0.4, x[1 + len(self.SWEEP_L) - 1] + 0.4, color=GRID, alpha=0.35,
                       zorder=0)
            _ratio_axis(ax)
            ax.set_xticks(x)
            ax.set_xticklabels(xlabels, fontsize=self._fs(8),
                               rotation=45 if self.paper else 0)
            if self.paper:
                ax.set_xlim(-0.6, x[-1] + 3.2)
                ax.set_xlabel("shaded: ideal caching, minimum prefix L (tokens)"
                              if i == len(benchmarks) - 1 else "")
                ax.set_ylabel("cost relative to ReAct")
            else:
                ax.set_xlim(-0.6, x[-1] + 1.6)
                ax.set_xlabel("shaded: perfect caching with minimum cacheable prefix L (tokens)")
                ax.set_ylabel("cost relative to ReAct (log scale)")
            self._title(ax, benchmark, PAPER_BENCHMARK.get(benchmark, benchmark))
            self._style(ax)
        if self.paper:
            fig.tight_layout()
        else:
            _bottom_legend(fig, axes[0])
            fig.suptitle("Fig 7 — RQ3 (exploratory): a scaffold's cost rank depends on the "
                         "provider's caching rules", color=INK)
            fig.tight_layout(rect=(0, 0.06, 1, 1))
        self._save(fig, "fig7_rq3_threshold_sweep")

    def _save(self, fig, name: str) -> None:
        pad = {"pad_inches": 0.02} if self.paper else {}
        fig.savefig(self.out / f"{name}.png", dpi=300 if self.paper else 160,
                    bbox_inches="tight", **pad)
        fig.savefig(self.out / f"{name}.pdf", bbox_inches="tight", **pad)
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
            "`fig4_budget_curve`, `fig5_rq4`, `fig6_rq1_gains`, and the exploratory "
            "`fig7_rq3_threshold_sweep` (PNG + PDF in this directory).",
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

        sweep = self.threshold_sweep()
        sweep.to_csv(self.out / "exploratory_h3_threshold_sweep.csv", index=False)
        self.dollar_matched(tables["table1_metrics"]).to_csv(
            self.out / "exploratory_dollar_matched.csv", index=False)

        self.draw_figures(tables, rq4_results, sweep)

        h4 = R4.h4_verdict(rq4_results) if rq4_results else {"h4_supported": None}
        return self.write_summary(tables, h4)

    def run_figures(self) -> Path:
        """Figures only (used by `--paper`); the tables are computed but not written."""
        tables = self.build_tables()
        rq4_tables, rq4_results = self.build_rq4()
        tables.update(rq4_tables)
        self.draw_figures(tables, rq4_results, self.threshold_sweep())
        return self.out

    def draw_figures(self, tables: dict[str, pd.DataFrame], rq4_results: dict,
                     sweep: pd.DataFrame) -> None:
        with plt.rc_context(PAPER_RC if self.paper else {}):
            self.figure1_pareto()
            self.figure2_reflexion()
            self.figure3_cost_measures(tables["table3_cost_ratios"])
            self.figure4_budget_curve()
            if rq4_results:
                self.figure5_rq4(rq4_results, tables["rq4_routing_policies"])
            self.figure6_rq1_gains(tables["h1_budget_matching"])
            self.figure7_threshold_sweep(sweep)


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
    ap.add_argument("--paper", action="store_true",
                    help="figures only, at ACL print size (default out: analysis/paper)")
    args = ap.parse_args(argv)

    ds = load(args.run_tag)
    out = config.run_dir(args.run_tag) / "analysis"
    if args.paper:
        out = out / "paper"
    out = Path(args.out) if args.out else out
    report = Report(ds, out, B=args.boot_b, vote_draws=args.vote_draws, paper=args.paper)
    path = report.run_figures() if args.paper else report.run()
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
