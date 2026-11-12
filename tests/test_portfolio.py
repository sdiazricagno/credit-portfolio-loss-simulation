import numpy as np
import pytest

from credit_sim.portfolio import (
    RATING_PD,
    analytic_expected_loss,
    generate_portfolio,
    scale_pds,
    shock_sector_pds,
)


def test_portfolio_is_reproducible_and_calibrated() -> None:
    first = generate_portfolio(600, seed=7)
    second = generate_portfolio(600, seed=7)
    other = generate_portfolio(600, seed=8)
    assert np.array_equal(first.sector, second.sector)
    assert np.array_equal(first.rating, second.rating)
    assert np.array_equal(first.pd, second.pd)
    assert np.array_equal(first.lgd, second.lgd)
    assert np.array_equal(first.ead, second.ead)
    assert not np.array_equal(first.ead, other.ead)
    assert first.n_loans == 600
    assert set(first.sector).issubset(set(first.sector_names))
    assert set(first.rating).issubset(RATING_PD)
    assert np.all((first.lgd > 0.0) & (first.lgd < 1.0))
    assert np.all(first.ead > 0.0)
    expected_pd = np.array([RATING_PD[rating] for rating in first.rating])
    assert np.array_equal(first.pd, expected_pd)
    manual = float(np.sum(first.pd * first.lgd * first.ead))
    assert analytic_expected_loss(first) == manual


def test_pd_shock_is_local_and_does_not_mutate() -> None:
    portfolio = generate_portfolio(500, seed=3)
    original = portfolio.pd.copy()
    scaled = scale_pds(portfolio, multiplier=10.0, cap=0.4)
    shocked = shock_sector_pds(portfolio, "CRE", multiplier=2.0, cap=0.95)
    assert np.array_equal(portfolio.pd, original)
    assert scaled.pd.max() == pytest.approx(0.4)
    assert np.all(scaled.pd <= 0.4 + 1e-12)
    assert np.all(scaled.pd + 1e-15 >= original)
    cre = portfolio.sector == "CRE"
    assert np.any(cre)
    assert np.allclose(shocked.pd[cre], np.minimum(original[cre] * 2.0, 0.95))
    assert np.array_equal(shocked.pd[~cre], original[~cre])
    assert np.array_equal(shocked.ead, portfolio.ead)
    assert np.array_equal(shocked.lgd, portfolio.lgd)
