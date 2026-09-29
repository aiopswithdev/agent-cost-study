import numpy as np
import pandas as pd
import pytest

from acs.analysis.vote import (
    compute_vote_curve,
    interpolate_k,
    majority_vote,
    per_task_at_k,
    vote_key,
)


def test_vote_key_normalises_and_abstains():
    assert vote_key("hotpotqa", "The Saimaa Gesture") == vote_key("hotpotqa", "saimaa gesture")
    assert vote_key("hotpotqa", "") is None
    assert vote_key("gsm8k", "the answer is 72") == vote_key("gsm8k", "72")
    assert vote_key("gsm8k", "no digits") is None


def test_majority_vote_picks_the_plurality():
    assert majority_vote("hotpotqa", ["Ajax", "Bravo", "Ajax"]) == "Ajax"
    assert majority_vote("gsm8k", ["72", "71", "seventy-two is 72"]) == "72"


def test_empty_answers_are_ignored_and_all_empty_returns_empty():
    assert majority_vote("hotpotqa", ["", "Bravo", ""]) == "Bravo"
    assert majority_vote("hotpotqa", ["", "", ""]) == ""


def test_tie_breaks_to_the_earliest_drawn_sample():
    assert majority_vote("hotpotqa", ["Bravo", "Ajax"]) == "Bravo"
    assert majority_vote("hotpotqa", ["Ajax", "Bravo"]) == "Ajax"
    assert majority_vote("hotpotqa", ["Cee", "Ajax", "Bravo", "Ajax", "Cee"]) == "Cee"


def test_a_lone_article_normalises_away_and_counts_as_an_abstention():
    """Official HotpotQA normalisation strips articles, so "a" carries no vote."""
    assert vote_key("hotpotqa", "a") is None
    assert majority_vote("hotpotqa", ["a", "Ajax"]) == "Ajax"


def test_winning_group_keeps_its_original_answer_string():
    """Grading uses the original string, not the normalised vote key."""
    assert majority_vote("hotpotqa", ["The Saimaa Gesture", "saimaa gesture"]) == (
        "The Saimaa Gesture"
    )


def pool_frame(answers_by_task: dict[str, list[str]], gold: str = "Ajax") -> pd.DataFrame:
    rows = []
    for task_id, answers in answers_by_task.items():
        for r, a in enumerate(answers):
            rows.append(
                {
                    "benchmark": "hotpotqa",
                    "task_id": task_id,
                    "replicate": r,
                    "final_answer": a,
                    "gold": gold,
                    "tokens_total": 1000 * (r + 1),
                    "usd_nocache": 0.001 * (r + 1),
                }
            )
    return pd.DataFrame(rows)


def test_curve_shapes_and_monotone_cost():
    pool = pool_frame({"t1": ["Ajax", "Ajax", "Bravo", "Bravo", "Ajax"], "t2": ["Bravo", "Bravo", "Bravo", "Ajax", "Bravo"]})
    curve = compute_vote_curve(pool, "hotpotqa", draws=50, cost_cols=("usd_nocache",))
    assert list(curve.aggregate.index) == [1, 2, 3, 4, 5]
    tokens = curve.aggregate["tokens_total"].to_numpy()
    assert np.all(np.diff(tokens) > 0)
    # vote@1 accuracy equals the plain per-sample accuracy (3/5 and 1/5, averaged).
    assert abs(curve.aggregate.loc[1, "acc"] - 0.4) < 0.06
    # Voting helps the task where the correct answer is the plurality.
    assert curve.per_task.query("task_id == 't1' and k == 5")["acc"].iloc[0] == 1.0
    assert curve.per_task.query("task_id == 't2' and k == 5")["acc"].iloc[0] == 0.0


def test_curve_is_reproducible():
    pool = pool_frame({"t1": ["Ajax", "Bravo", "Ajax", "Bravo", "Ajax"]})
    a = compute_vote_curve(pool, "hotpotqa", draws=20, cost_cols=("usd_nocache",))
    b = compute_vote_curve(pool, "hotpotqa", draws=20, cost_cols=("usd_nocache",))
    pd.testing.assert_frame_equal(a.aggregate, b.aggregate)


def test_interpolate_k_matches_a_budget():
    pool = pool_frame({"t1": ["Ajax", "Ajax", "Bravo", "Bravo", "Ajax"], "t2": ["Ajax", "Bravo", "Bravo", "Ajax", "Ajax"]})
    curve = compute_vote_curve(pool, "hotpotqa", draws=50, cost_cols=("usd_nocache",))
    mid = float(curve.aggregate["tokens_total"].iloc[1])
    point = interpolate_k(curve, mid)
    assert abs(point.k_star - 2.0) < 1e-6 and point.beyond_pool is False

    beyond = interpolate_k(curve, 1e9)
    assert beyond.beyond_pool is True and beyond.k_star == 5.0

    below = interpolate_k(curve, 0.0)
    assert below.k_star == 1.0 and below.beyond_pool is False


def test_per_task_at_fractional_k_interpolates():
    pool = pool_frame({"t1": ["Ajax", "Ajax", "Bravo", "Bravo", "Ajax"], "t2": ["Ajax", "Bravo", "Bravo", "Ajax", "Ajax"]})
    curve = compute_vote_curve(pool, "hotpotqa", draws=50, cost_cols=("usd_nocache",))
    lo = per_task_at_k(curve, 2.0)
    hi = per_task_at_k(curve, 3.0)
    mid = per_task_at_k(curve, 2.5)
    assert np.allclose(mid.to_numpy(), (lo.to_numpy() + hi.to_numpy()) / 2)
