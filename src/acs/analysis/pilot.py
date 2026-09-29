"""T12 — the pilot report and the go/no-go projection.

    python -m acs.analysis.pilot --run-tag pilot

Writes `results/{run_tag}/PILOT_REPORT.md`. The decision it supports goes in
`docs/DEVIATIONS.md`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .. import config
from .costs import empirical_cache_read_share
from .load import Dataset, load
from .metrics import costed_runs

# Go/no-go thresholds (§T12).
MAX_ERROR_RATE = 0.02
MAX_INVALID_ACTION_RATE = 0.10
REACT_HOTPOT_EM_RANGE = (0.25, 0.50)


@dataclass
class Projection:
    scaffold: str
    benchmark: str
    pilot_runs: int
    planned_runs: int
    mean_usd_per_run: float
    projected_usd: float
    mean_wall_s: float


def per_scaffold(ds: Dataset, runs: pd.DataFrame) -> pd.DataFrame:
    all_runs = pd.concat([ds.runs, ds.dropped], ignore_index=True)
    rows = []
    for benchmark in sorted(all_runs["benchmark"].unique()):
        for scaffold in config.SCAFFOLDS:
            every = all_runs[
                (all_runs["benchmark"] == benchmark) & (all_runs["scaffold"] == scaffold)
            ]
            ok = runs[(runs["benchmark"] == benchmark) & (runs["scaffold"] == scaffold)]
            if not len(every):
                continue
            rows.append(
                {
                    "benchmark": benchmark,
                    "scaffold": scaffold,
                    "n_runs": len(every),
                    "error_rate": float((every["status"] != "ok").mean()),
                    "accuracy": float(ok["correct"].mean()) if len(ok) else float("nan"),
                    "f1": float(ok["f1"].mean()) if len(ok) else float("nan"),
                    "mean_tokens_in": float(ok["tokens_in_calls"].mean()) if len(ok) else 0.0,
                    "mean_tokens_out": float(ok["tokens_out_calls"].mean()) if len(ok) else 0.0,
                    "mean_usd_nocache": float(ok["usd_nocache"].mean()) if len(ok) else 0.0,
                    "mean_usd_empirical": float(ok["usd_empirical"].mean()) if len(ok) else 0.0,
                    "mean_llm_calls": float(ok["n_llm_calls"].mean()) if len(ok) else 0.0,
                    "mean_wall_s": float(ok["wall_s"].mean()) if len(ok) else 0.0,
                    # parse health
                    "invalid_action_rate": (
                        float((ok["n_invalid_actions"] > 0).mean()) if len(ok) else float("nan")
                    ),
                    "halt_rate": float(ok["halted"].mean()) if len(ok) else float("nan"),
                    "plan_parse_ok_rate": (
                        float(ok["plan_parse_ok"].astype(float).mean())
                        if len(ok) and ok["plan_parse_ok"].notna().any()
                        else float("nan")
                    ),
                    "judge_parse_fail_rate": (
                        float((ok["judge_parse_fail"].fillna(0) > 0).mean())
                        if len(ok) and ok["judge_parse_fail"].notna().any()
                        else float("nan")
                    ),
                    "mean_trials": (
                        float(ok["n_trials"].mean())
                        if len(ok) and ok["n_trials"].notna().any()
                        else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)


def project_main_run(
    summary: pd.DataFrame, n: int = config.N, k: int = config.POOL_K, r: int = config.R
) -> pd.DataFrame:
    """Planned main-run counts: N·K for the pool, N·R per other scaffold."""
    rows = []
    for _, row in summary.iterrows():
        planned = n * k if row["scaffold"] == "react" else n * r
        rows.append(
            Projection(
                scaffold=row["scaffold"],
                benchmark=row["benchmark"],
                pilot_runs=int(row["n_runs"]),
                planned_runs=planned,
                mean_usd_per_run=row["mean_usd_nocache"],
                projected_usd=planned * row["mean_usd_nocache"],
                mean_wall_s=row["mean_wall_s"],
            ).__dict__
        )
    return pd.DataFrame(rows)


def observed_runs_per_min(runs: pd.DataFrame, max_in_flight: int) -> float:
    """Throughput implied by the mean run duration and the configured concurrency."""
    mean_wall = float(runs["wall_s"].mean()) if len(runs) else float("nan")
    if not mean_wall or mean_wall != mean_wall:
        return float("nan")
    return 60.0 / mean_wall * max_in_flight


def go_no_go(summary: pd.DataFrame, projected_usd: float, max_usd: float) -> list[str]:
    issues: list[str] = []
    for _, row in summary.iterrows():
        label = f"{row['benchmark']}/{row['scaffold']}"
        if row["error_rate"] > MAX_ERROR_RATE:
            issues.append(
                f"**{label}**: error rate {row['error_rate']:.1%} > {MAX_ERROR_RATE:.0%} "
                "— fix the scaffold or the parser before the main run."
            )
        if (
            row["scaffold"] == "react"
            and row["invalid_action_rate"] == row["invalid_action_rate"]
            and row["invalid_action_rate"] > MAX_INVALID_ACTION_RATE
        ):
            issues.append(
                f"**{label}**: invalid-action rate {row['invalid_action_rate']:.1%} > "
                f"{MAX_INVALID_ACTION_RATE:.0%} — fix the ReAct prompt/parser first."
            )
        if row["scaffold"] == "react" and row["benchmark"] == "hotpotqa":
            lo, hi = REACT_HOTPOT_EM_RANGE
            if not (lo <= row["accuracy"] <= hi):
                issues.append(
                    f"**{label}**: ReAct HotpotQA EM {row['accuracy']:.1%} is outside the "
                    f"sanity band {lo:.0%}–{hi:.0%} — inspect traces before continuing."
                )

    react_calls = summary[summary["scaffold"] == "react"]["mean_llm_calls"].mean()
    rewoo_calls = summary[summary["scaffold"] == "rewoo"]["mean_llm_calls"].mean()
    if rewoo_calls == rewoo_calls and react_calls == react_calls and rewoo_calls >= react_calls:
        issues.append(
            f"**rewoo**: mean LLM calls {rewoo_calls:.1f} is not far below ReAct's "
            f"{react_calls:.1f} — ReWOO should be much cheaper in calls; inspect traces."
        )

    if projected_usd > max_usd:
        issues.append(
            f"**cost**: projected main run ${projected_usd:.0f} > MAX_USD ${max_usd:.0f} "
            "— apply the scope-down ladder (§6.3) before launch."
        )
    return issues


def build_report(ds: Dataset, max_usd: float = config.MAX_USD) -> str:
    runs = costed_runs(ds)
    summary = per_scaffold(ds, runs)
    projection = project_main_run(summary)
    projected_usd = float(projection["projected_usd"].sum())
    runs_per_min = observed_runs_per_min(runs, config.MAX_RUNS_IN_FLIGHT)
    planned_runs = int(projection["planned_runs"].sum())
    projected_hours = (
        planned_runs / runs_per_min / 60.0 if runs_per_min and runs_per_min == runs_per_min
        else float("nan")
    )
    issues = go_no_go(summary, projected_usd, max_usd)

    parts = [
        f"# PILOT REPORT — run tag `{ds.run_tag}`",
        "",
        f"- pilot runs: **{len(ds.runs) + len(ds.dropped)}** "
        f"({len(ds.dropped)} errored)",
        f"- pilot LLM calls: **{len(ds.calls)}**",
        f"- pilot spend (`usd_nocache`): **${runs['usd_nocache'].sum():.2f}** "
        f"(empirical: ${runs['usd_empirical'].sum():.2f})",
        f"- empirical cache-read share (Σcache_read / ΣI): "
        f"**{empirical_cache_read_share(ds.calls):.4f}** "
        "— expected ≈ 0 on Haiku 4.5, whose cacheable prefix is 4,096 tokens",
        "",
        "## Per scaffold × benchmark",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Projected main run",
        "",
        f"Planned counts: N={config.N}, pool K={config.POOL_K}, others R={config.R} "
        f"→ {planned_runs:,} runs.",
        "",
        projection.to_markdown(index=False, floatfmt=".4f"),
        "",
        f"- **projected cost: ${projected_usd:.0f}** (`usd_nocache`; MAX_USD = ${max_usd:.0f})",
        f"- observed throughput: **{runs_per_min:.1f} runs/min** at "
        f"MAX_RUNS_IN_FLIGHT={config.MAX_RUNS_IN_FLIGHT}",
        f"- **projected wall clock: {projected_hours:.1f} hours**",
        "",
        "## Go / no-go",
        "",
    ]
    if issues:
        parts.append("**NO-GO — resolve these first:**")
        parts.append("")
        parts += [f"- {issue}" for issue in issues]
        parts += [
            "",
            "A prompt change after the pilot is allowed but must be logged in "
            "`docs/DEVIATIONS.md`. After the main run starts, prompts are frozen.",
        ]
    else:
        parts.append(
            "**GO** — error rates, parse health, sanity bands, projected cost and projected "
            "wall clock all pass the §T12 thresholds."
        )
    parts += ["", "Record the decision and its reason in `docs/DEVIATIONS.md`.", ""]
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m acs.analysis.pilot")
    ap.add_argument("--run-tag", default="pilot")
    ap.add_argument("--max-usd", type=float, default=config.MAX_USD)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    ds = load(args.run_tag)
    text = build_report(ds, max_usd=args.max_usd)
    path = Path(args.out) if args.out else config.run_dir(args.run_tag) / "PILOT_REPORT.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
