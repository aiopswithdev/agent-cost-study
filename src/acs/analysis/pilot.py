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
from ..pricing import usd_billed, usd_empirical, usd_nocache
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


def new_spend_per_run(ds: Dataset) -> pd.DataFrame:
    """Mean *new* spend per run by (benchmark, scaffold), at list price with caching.

    Uses only the calls logged under each run's own `run_id`. A Reflexion run's trial 1 is
    a pool run that is already paid for, so it is correctly not counted twice here (the
    analysis still bills it to Reflexion in full, §2.4 — that is a comparison, not a bill).
    """
    all_runs = pd.concat([ds.runs, ds.dropped], ignore_index=True)
    n_runs = all_runs.groupby(["benchmark", "scaffold"]).size().rename("n_runs")
    if not len(ds.calls):
        out = n_runs.to_frame()
        out["usd_list"] = 0.0
        out["usd_billed"] = 0.0
        return out.reset_index()
    calls = ds.calls.copy()
    calls["usd_list"] = calls.apply(lambda r: usd_empirical(r), axis=1)
    calls["usd_billed"] = calls.apply(lambda r: usd_billed(r), axis=1)
    spent = calls.groupby(["benchmark", "scaffold"])[["usd_list", "usd_billed"]].sum()
    out = n_runs.to_frame().join(spent, how="left").fillna(0.0)
    out["usd_list"] = out["usd_list"] / out["n_runs"]
    out["usd_billed"] = out["usd_billed"] / out["n_runs"]
    return out.reset_index()


def project_main_run(
    per_run: pd.DataFrame, summary: pd.DataFrame,
    n: int = config.N, k: int = config.POOL_K, r: int = config.R,
) -> pd.DataFrame:
    """Planned main-run counts: N·K for the pool, N·R per other scaffold."""
    wall = summary.set_index(["benchmark", "scaffold"])["mean_wall_s"]
    rows = []
    for _, row in per_run.iterrows():
        planned = n * k if row["scaffold"] == "react" else n * r
        rows.append(
            Projection(
                scaffold=row["scaffold"],
                benchmark=row["benchmark"],
                pilot_runs=int(row["n_runs"]),
                planned_runs=planned,
                mean_usd_per_run=row["usd_list"],
                projected_usd=planned * row["usd_list"],
                mean_wall_s=float(wall.get((row["benchmark"], row["scaffold"]), float("nan"))),
            ).__dict__
        )
    return pd.DataFrame(rows)


_R_ALL3 = {s: 3 for s in config.OTHER_SCAFFOLDS}
_R_PE2 = {**_R_ALL3, "rewoo": 2, "plan_exec": 2}
#: The §6.3 scope-down ladder, then two further steps for a very small budget.
#: (label, N, pool K per benchmark, R per other scaffold)
LADDER = [
    ("as pre-registered", 300, {"hotpotqa": 10, "gsm8k": 10}, _R_ALL3),
    ("§6.3 step 1: GSM8K K=6", 300, {"hotpotqa": 10, "gsm8k": 6}, _R_ALL3),
    ("§6.3 step 2: N=200", 200, {"hotpotqa": 10, "gsm8k": 6}, _R_ALL3),
    ("§6.3 step 3: R=2 ReWOO/Plan-Exec", 200, {"hotpotqa": 10, "gsm8k": 6}, _R_PE2),
    ("§6.3 step 4: HotpotQA K=6", 200, {"hotpotqa": 6, "gsm8k": 6}, _R_PE2),
    ("extra: N=150", 150, {"hotpotqa": 6, "gsm8k": 6}, _R_PE2),
    ("extra: N=100", 100, {"hotpotqa": 6, "gsm8k": 6}, _R_PE2),
]


def ladder_projection(per_run: pd.DataFrame, budget_usd: float) -> pd.DataFrame:
    """Projected bill of every ladder configuration, at list (peak) and off-peak rates."""
    u = per_run.set_index(["benchmark", "scaffold"])["usd_list"]
    benchmarks = sorted(per_run["benchmark"].unique())
    rows = []
    for label, n, ks, rs in LADDER:
        runs, usd = 0, 0.0
        for b in benchmarks:
            runs += n * ks[b]
            usd += n * ks[b] * float(u.get((b, "react"), 0.0))
            for s_, r_ in rs.items():
                runs += n * r_
                usd += n * r_ * float(u.get((b, s_), 0.0))
        offpeak = usd * config.DEEPSEEK_OFFPEAK_MULTIPLIER
        rows.append({
            "configuration": label, "N": n,
            "K (hotpot/gsm8k)": f"{ks['hotpotqa']}/{ks['gsm8k']}",
            "R (rewoo,pe / reflexion)": f"{rs['rewoo']} / {rs['reflexion_gt']}",
            "runs": runs, "usd_peak": usd, "usd_offpeak": offpeak,
            "fits_offpeak": offpeak <= budget_usd, "fits_peak": usd <= budget_usd,
        })
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
            f"**cost**: the pre-registered main run projects to ${projected_usd:.2f} off-peak, "
            f"more than the remaining budget ${max_usd:.2f} — launch the largest configuration "
            "marked as fitting in the scope-down table below, and log it in DEVIATIONS.md."
        )
    return issues


def build_report(ds: Dataset, max_usd: float = config.MAX_USD) -> str:
    runs = costed_runs(ds)
    summary = per_scaffold(ds, runs)
    per_run = new_spend_per_run(ds)
    projection = project_main_run(per_run, summary)
    projected_list = float(projection["projected_usd"].sum())
    def _spent(fn) -> float:
        return float(ds.calls.apply(lambda r: fn(r), axis=1).sum()) if len(ds.calls) else 0.0

    pilot_billed = _spent(usd_billed)
    pilot_list, pilot_nocache = _spent(usd_empirical), _spent(usd_nocache)
    remaining = max(max_usd - pilot_billed, 0.0)
    ladder = ladder_projection(per_run, remaining)
    fitting = ladder[ladder["fits_offpeak"]]
    projected_usd = projected_list * config.DEEPSEEK_OFFPEAK_MULTIPLIER
    runs_per_min = observed_runs_per_min(runs, config.MAX_RUNS_IN_FLIGHT)
    planned_runs = int(projection["planned_runs"].sum())
    projected_hours = (
        planned_runs / runs_per_min / 60.0 if runs_per_min and runs_per_min == runs_per_min
        else float("nan")
    )
    issues = go_no_go(summary, projected_usd, remaining)

    parts = [
        f"# PILOT REPORT — run tag `{ds.run_tag}`",
        "",
        f"- pilot runs: **{len(ds.runs) + len(ds.dropped)}** "
        f"({len(ds.dropped)} errored)",
        f"- pilot LLM calls: **{len(ds.calls)}**",
        f"- pilot bill: **${pilot_billed:.3f}** actually charged "
        f"(list price: ${pilot_list:.3f} with cache, ${pilot_nocache:.3f} without)",
        f"- budget: MAX_USD ${max_usd:.2f} − pilot ${pilot_billed:.3f} = "
        f"**${remaining:.2f} remaining** for the main run",
        f"- empirical cache-read share (Σcache_read / ΣI): "
        f"**{empirical_cache_read_share(ds.calls):.4f}** "
        "(share of input tokens the provider served from its prefix cache)",
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
        f"- **projected bill, as pre-registered: ${projected_list:.2f} at peak rates, "
        f"${projected_usd:.2f} if run entirely off-peak** "
        "(new spend per run at list price with caching; Reflexion trial 1 not double-counted)",
        "",
        "## Scope-down ladder vs. the remaining budget",
        "",
        ladder.to_markdown(index=False, floatfmt=".2f"),
        "",
        (
            f"**Recommended:** `{fitting.iloc[0]['configuration']}` — the largest configuration "
            f"that fits ${remaining:.2f} off-peak (${fitting.iloc[0]['usd_offpeak']:.2f}). "
            "DeepSeek peak = 01:00–04:00 and 06:00–10:00 UTC, Mon–Fri; launch outside it."
            if len(fitting) else
            f"**No configuration fits ${remaining:.2f}** — top up the account or reduce N further."
        ),
        "",
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
