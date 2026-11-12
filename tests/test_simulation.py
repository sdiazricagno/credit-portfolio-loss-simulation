import numpy as np

from credit_sim.copula import simulate_gaussian_rho_grid, simulate_losses
from credit_sim.metrics import expected_shortfall, value_at_risk, vasicek_quantile
from credit_sim.portfolio import Portfolio, analytic_expected_loss, generate_portfolio


def _homogeneous(n: int = 1_200, pd: float = 0.02) -> Portfolio:
    return Portfolio(
        sector=np.full(n, "Corporate"),
        rating=np.full(n, "B"),
        pd=np.full(n, pd),
        lgd=np.full(n, 0.45),
        ead=np.full(n, 100.0),
        sector_names=("Corporate",),
    )


def test_simulation_is_reproducible() -> None:
    portfolio = generate_portfolio(300, seed=4)
    first = simulate_losses(portfolio, n_sims=1_500, rho=0.15, seed=4)
    second = simulate_losses(portfolio, n_sims=1_500, rho=0.15, seed=4)
    assert np.array_equal(first.losses, second.losses)


def test_simulated_el_matches_analytic() -> None:
    portfolio = generate_portfolio(700, seed=9)
    analytic = analytic_expected_loss(portfolio)
    gaussian = simulate_losses(portfolio, n_sims=12_000, rho=0.15, seed=9, copula="gaussian")
    student = simulate_losses(portfolio, n_sims=12_000, rho=0.15, seed=9, copula="student_t", df=5.0)
    np.testing.assert_allclose(gaussian.losses.mean(), analytic, rtol=0.05)
    np.testing.assert_allclose(student.losses.mean(), analytic, rtol=0.08)
    for alpha in (0.95, 0.99):
        assert expected_shortfall(gaussian.losses, alpha) >= value_at_risk(gaussian.losses, alpha)
        assert expected_shortfall(student.losses, alpha) >= value_at_risk(student.losses, alpha)


def test_var_increases_with_asset_correlation() -> None:
    portfolio = generate_portfolio(800, seed=11)
    rhos = (0.05, 0.15, 0.30, 0.45)
    grid = simulate_gaussian_rho_grid(portfolio, rhos, n_sims=6_000, seed=11)
    vars_99 = [value_at_risk(grid[rho], 0.99) for rho in rhos]
    pairs = zip(vars_99, vars_99[1:], strict=False)
    assert all(later > earlier for earlier, later in pairs)


def test_student_t_tail_heavier_than_gaussian_at_999() -> None:
    portfolio = _homogeneous(n=1_500, pd=0.02)
    gaussian = simulate_losses(portfolio, n_sims=20_000, rho=0.20, seed=5, copula="gaussian")
    student = simulate_losses(portfolio, n_sims=20_000, rho=0.20, seed=5, copula="student_t", df=5.0)
    assert value_at_risk(student.losses, 0.999) > value_at_risk(gaussian.losses, 0.999)


def test_factor_only_quantile_tracks_vasicek_limit() -> None:
    portfolio = generate_portfolio(400, seed=6)
    factor_only = simulate_losses(
        portfolio,
        n_sims=20_000,
        rho=0.15,
        seed=6,
        copula="gaussian",
        idiosyncratic=False,
    )
    closed_form = vasicek_quantile(portfolio, 0.15, 0.99)
    np.testing.assert_allclose(value_at_risk(factor_only.losses, 0.99), closed_form, rtol=0.03)
