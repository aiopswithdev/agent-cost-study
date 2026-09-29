import numpy as np
import pytest

from acs.analysis.stats import (
    bootstrap_cost_of_pass,
    bootstrap_mean,
    bootstrap_paired_diff,
    bootstrap_paired_ratio,
    cluster_bootstrap,
    holm,
    mcnemar_exact,
    mcnemar_from_pairs,
    two_sided_p,
)


def test_bootstrap_mean_covers_the_truth():
    rng = np.random.default_rng(0)
    values = rng.normal(0.5, 0.1, size=400)
    res = bootstrap_mean(values, B=500)
    assert abs(res.point - values.mean()) < 1e-12
    assert res.ci_lo < values.mean() < res.ci_hi
    assert res.ci_hi - res.ci_lo < 0.05


def test_paired_diff_is_significant_for_a_real_effect():
    n = 300
    a = np.array([1.0] * 200 + [0.0] * 100)
    b = np.array([1.0] * 120 + [0.0] * 180)
    res = bootstrap_paired_diff(a, b, B=500)
    assert res.point == pytest.approx(a.mean() - b.mean())
    assert res.ci_lo > 0 and res.p_value < 0.05


def test_paired_diff_is_not_significant_for_no_effect():
    x = np.array([1.0] * 150 + [0.0] * 150)
    res = bootstrap_paired_diff(x, x.copy(), B=500)
    assert res.point == 0.0 and res.p_value == 1.0


def test_pairing_uses_the_same_resampled_tasks():
    """A constant per-task difference must give a zero-width interval."""
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, size=200)
    b = a - 0.25
    res = bootstrap_paired_diff(a, b, B=300)
    assert res.point == pytest.approx(0.25)
    assert res.ci_hi - res.ci_lo < 1e-9


def test_paired_arrays_must_align():
    with pytest.raises(ValueError):
        bootstrap_paired_diff([1, 2, 3], [1, 2])


def test_paired_ratio():
    a = np.full(100, 2.0)
    b = np.full(100, 4.0)
    res = bootstrap_paired_ratio(a, b, B=200)
    assert res.point == pytest.approx(0.5)


def test_cost_of_pass_divides_by_accuracy():
    cost = np.full(100, 0.01)
    correct = np.array([1.0] * 50 + [0.0] * 50)
    res = bootstrap_cost_of_pass(cost, correct, B=200)
    assert res.point == pytest.approx(0.02)
    assert bootstrap_cost_of_pass(cost, np.zeros(100), B=50).point == float("inf")


def test_two_sided_p_bounds():
    assert two_sided_p(np.array([1.0, 2.0, 3.0])) == 0.0
    assert two_sided_p(np.array([-1.0, 1.0])) == 1.0
    assert 0.0 <= two_sided_p(np.array([-1.0, 1.0, 1.0, 1.0])) <= 1.0


def test_cluster_bootstrap_is_reproducible():
    values = np.arange(50, dtype=float)
    fn = lambda idx: float(values[idx].mean())
    a = cluster_bootstrap([str(i) for i in range(50)], fn, B=100)
    b = cluster_bootstrap([str(i) for i in range(50)], fn, B=100)
    assert np.array_equal(a.samples, b.samples)


def test_mcnemar_exact_matches_the_binomial():
    assert mcnemar_exact(0, 0).p_value == 1.0
    assert mcnemar_exact(10, 10).p_value == 1.0
    assert mcnemar_exact(20, 2).p_value < 0.001
    assert mcnemar_exact(5, 5).p_value == pytest.approx(1.0)


def test_mcnemar_from_paired_outcomes():
    a = [1, 1, 1, 0, 0]
    b = [0, 0, 1, 0, 1]
    res = mcnemar_from_pairs(a, b)
    assert (res.b, res.c) == (2, 1)


def test_holm_step_down():
    assert holm([]) == []
    assert holm([0.01]) == [0.01]
    # Classic worked example.
    adj = holm([0.01, 0.02, 0.03])
    assert adj == pytest.approx([0.03, 0.04, 0.04])
    # Monotone non-decreasing in the sorted order, and never above 1.
    adj = holm([0.5, 0.4, 0.001, 0.9])
    assert all(0 <= v <= 1 for v in adj)
    assert adj[2] < adj[1] <= adj[0] <= adj[3]


def test_holm_is_conservative_relative_to_raw_p():
    raw = [0.001, 0.02, 0.03, 0.04, 0.5]
    assert all(a >= r for a, r in zip(holm(raw), raw))
