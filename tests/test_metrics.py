import numpy as np
import pytest
from scipy.stats import norm

from credit_sim.copula import gaussian_conditional_pd
from credit_sim.metrics import (
    es_contributions,
    expected_shortfall,
    value_at_risk,
    vasicek_expected_shortfall,
    vasicek_quantile,
)
from credit_sim.portfolio import Portfolio


def _book(pds: list[float], lgd: float = 0.4, ead: float = 1_000.0) -> Portfolio:
    n = len(pds)
    return Portfolio(
        sector=np.full(n, "Corporate"),
        rating=np.full(n, "BB"),
        pd=np.array(pds, dtype=float),
        lgd=np.full(n, lgd),
        ead=np.full(n, ead),
        sector_names=("Corporate",),
    )


def test_expected_shortfall_is_at_least_var() -> None:
    losses = np.array([1.0, 2.0, 3.0, 10.0, 4.0, 5.0, 8.0, 6.0, 7.0, 9.0])
    for alpha in (0.5, 0.8, 0.9):
        var = value_at_risk(losses, alpha)
        es = expected_shortfall(losses, alpha)
        assert es >= var
        tail = losses[losses >= var]
        assert es == np.mean(tail)


def test_es_contributions_sum_to_expected_shortfall() -> None:
    rng = np.random.default_rng(0)
    sector_losses = rng.random((4_000, 4)) * np.array([3.0, 1.0, 2.0, 0.5])
    contributions = es_contributions(sector_losses, 0.99)
    assert np.all(contributions >= 0.0)
    assert contributions.sum() == expected_shortfall(sector_losses.sum(axis=1), 0.99)


def test_bad_systematic_factor_raises_conditional_pd() -> None:
    pds = np.array([0.02, 0.05, 0.10])
    bad = gaussian_conditional_pd(pds, np.array([-2.0]), rho=0.2)
    good = gaussian_conditional_pd(pds, np.array([2.0]), rho=0.2)
    assert np.all(bad > pds)
    assert np.all(good < pds)


def test_vasicek_known_value_and_monotonicity() -> None:
    portfolio = _book([0.01])
    numer = norm.ppf(0.01) + np.sqrt(0.2) * norm.ppf(0.99)
    expected = 1_000.0 * 0.4 * norm.cdf(numer / np.sqrt(0.8))
    assert vasicek_quantile(portfolio, 0.2, 0.99) == pytest.approx(expected)
    assert vasicek_quantile(portfolio, 0.2, 0.99) == pytest.approx(30.100315774198478)
    homogeneous = _book([0.02] * 20, lgd=0.45, ead=100.0)
    quantiles = [vasicek_quantile(homogeneous, rho, 0.999) for rho in (0.05, 0.12, 0.2, 0.35, 0.5)]
    pairs = zip(quantiles, quantiles[1:], strict=False)
    assert all(later > earlier for earlier, later in pairs)
    assert vasicek_expected_shortfall(homogeneous, 0.2, 0.99) > vasicek_quantile(homogeneous, 0.2, 0.99)
