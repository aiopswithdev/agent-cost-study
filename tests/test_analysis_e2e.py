"""T14 acceptance: every table and figure is reproducible from the JSONL files alone."""

from __future__ import annotations

import pandas as pd
import pytest

from acs import config
from acs.analysis import metrics as M
from acs.analysis import report as REPORT
from acs.analysis.load import load
from acs.analysis.pilot import build_report
from tests import synth


@pytest.fixture(scope="module")
def synth_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("synthrun")
    synth.generate(d, n_tasks=12, n_errors=2)
    return d


@pytest.fixture(scope="module")
def dataset(synth_dir):
    return load("synth", run_dir=synth_dir)


def test_load_splits_ok_and_errored_runs(dataset):
    assert len(dataset.dropped) == 2
    assert (dataset.runs["status"] == "ok").all()
    report = dataset.drop_report()
    assert set(report.columns) >= {"benchmark", "scaffold", "n_dropped", "frac_dropped"}


def test_ownership_bills_reflexion_for_trial1(dataset):
    refl = dataset.runs[dataset.runs["scaffold"] == "reflexion_gt"].iloc[0]
    owned = dataset.ownership[dataset.ownership["run_id"] == refl["run_id"]]
    own_calls = dataset.calls[dataset.calls["run_id"] == refl["run_id"]]
    assert len(owned) > len(own_calls)


def test_costed_runs_match_the_logged_run_totals(dataset):
    runs = M.costed_runs(dataset)
    assert (runs["tokens_in_calls"] == runs["tokens_in"]).all()
    assert (runs["tokens_out_calls"] == runs["tokens_out"]).all()
    assert (runs["usd_nocache"] - runs["usd_nocache_run"]).abs().max() < 1e-12
    # Caching can only ever reduce the bill.
    for L in config.IDEAL_CACHE_L:
        assert (runs[f"usd_ideal_{L}"] <= runs["usd_nocache"] + 1e-12).all()


def test_per_task_uses_only_the_first_R_react_replicates(dataset):
    runs = M.costed_runs(dataset)
    table = M.per_task(runs)
    react = table[table["scaffold"] == "react"]
    assert (react["n_replicates"] == config.R).all()
    others = table[table["scaffold"] == "rewoo"]
    assert (others["n_replicates"] == config.R).all()


def test_vote_curve_and_matched_controls(dataset):
    runs = M.costed_runs(dataset)
    table = M.per_task(runs)
    curves = M.vote_curves(runs, draws=20)
    for benchmark, curve in curves.items():
        assert list(curve.aggregate.index) == list(range(1, config.POOL_K + 1))
        tokens = curve.aggregate["tokens_total"].to_numpy()
        assert (tokens[1:] > tokens[:-1]).all()

        control = M.matched_control(curve, table, benchmark, "plan_exec")
        assert 1.0 <= control.point.k_star <= config.POOL_K
        assert len(control.per_task_acc) == 12


def test_full_report_runs_and_writes_every_artifact(dataset, tmp_path):
    out = tmp_path / "analysis"
    summary = REPORT.Report(dataset, out, B=120, vote_draws=20).run()
    assert summary.exists() and "Analysis summary" in summary.read_text()

    for name in (
        "table1_metrics", "table2_comparisons", "table3_cost_ratios",
        "table4_failure_modes", "h1_budget_matching", "h3_verdict",
        "judge_confusion", "dropped_runs", "rq4_auc", "rq4_routing_policies",
    ):
        assert (out / f"{name}.csv").exists(), name
    for fig in ("fig1_pareto", "fig2_reflexion", "fig3_cost_measures",
                "fig4_budget_curve", "fig5_rq4"):
        assert (out / f"{fig}.png").exists() and (out / f"{fig}.pdf").exists(), fig

    comparisons = pd.read_csv(out / "table2_comparisons.csv")
    # Nine pre-registered comparisons per benchmark (§2.8).
    assert comparisons.groupby("benchmark").size().tolist() == [9, 9]
    assert (comparisons["holm_p"] >= comparisons["bootstrap_p"] - 1e-12).all()
    assert set(comparisons["family"]) == {
        "X_vs_react", "X_vs_matched_vote", "oracle_premium"
    }

    metrics = pd.read_csv(out / "table1_metrics.csv")
    assert set(config.SCAFFOLDS) <= set(metrics["arm"])
    assert any(a.startswith("react_vote@") for a in metrics["arm"])
    assert (metrics["acc_ci_lo"] <= metrics["acc"] + 1e-9).all()
    assert (metrics["acc"] <= metrics["acc_ci_hi"] + 1e-9).all()


def test_paper_figures_are_print_size(dataset, tmp_path):
    """`--paper` writes only figures, each no wider than its ACL slot and close to it."""
    import re

    out = tmp_path / "paper"
    REPORT.Report(dataset, out, B=80, vote_draws=20, paper=True).run_figures()
    assert not list(out.glob("*.csv")) and not (out / "SUMMARY.md").exists()
    for name, (_, _, (width_in, _)) in REPORT.PAPER_LAYOUT.items():
        pdf = (out / f"{name}.pdf").read_bytes()
        box = re.search(rb"/MediaBox \[\s*0 0 ([\d.]+) ([\d.]+)\s*\]", pdf)
        width = float(box.group(1)) / 72
        # The tight bounding box trims whitespace; it must not grow past the slot.
        assert 0.85 * width_in <= width <= width_in + 0.05, (name, width)


def test_report_is_deterministic(dataset, tmp_path):
    a = REPORT.Report(dataset, tmp_path / "a", B=100, vote_draws=20).run().read_text()
    b = REPORT.Report(dataset, tmp_path / "b", B=100, vote_draws=20).run().read_text()
    assert a == b


def test_report_cli_from_jsonl_only(synth_dir, tmp_path, monkeypatch):
    """Acceptance: `python -m acs.analysis.report --run-tag X` needs only the JSONL files."""
    import shutil

    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    target = tmp_path / "results" / "unittest_synth"
    target.mkdir(parents=True)
    for name in ("runs.jsonl", "calls.jsonl", "prompt_tokens.json"):
        shutil.copy(synth_dir / name, target / name)

    assert REPORT.main(["--run-tag", "unittest_synth", "--boot-b", "80", "--vote-draws", "20"]) == 0
    assert (target / "analysis" / "SUMMARY.md").exists()
    assert (target / "analysis" / "fig1_pareto.png").exists()


def test_pilot_report_renders(dataset):
    text = build_report(dataset, max_usd=250.0)
    assert "# PILOT REPORT" in text
    assert "Projected main run" in text
    assert "Go / no-go" in text
    assert "projected bill" in text
    assert "Scope-down ladder" in text
