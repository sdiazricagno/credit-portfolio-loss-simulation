"""Expected loss, VaR, expected shortfall, and the Vasicek large-portfolio limit."""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

from credit_sim.copula import gaussian_conditional_pd
from credit_sim.portfolio import Portfolio


def _require_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    return alpha


def value_at_risk(losses: np.ndarray, alpha: float) -> float:
    alpha = _require_alpha(alpha)
    sample = np.asarray(losses, dtype=float)
    if sample.size == 0:
        raise ValueError("losses must be non-empty")
    return float(np.quantile(sample, alpha, method="linear"))


def expected_shortfall(losses: np.ndarray, alpha: float) -> float:
    sample = np.asarray(losses, dtype=float)
    var = value_at_risk(sample, alpha)
    tail = sample[sample >= var]
    if tail.size == 0:
        return var
    return float(tail.mean())


def es_contributions(sector_losses: np.ndarray, alpha: float) -> np.ndarray:
    """Average sector loss on scenarios at or above portfolio VaR. Columns sum to ES."""
    losses = np.asarray(sector_losses, dtype=float)
    if losses.ndim != 2 or losses.shape[0] == 0:
        raise ValueError("sector_losses must be a non-empty matrix of scenarios by sectors")
    total = losses.sum(axis=1)
    tail = total >= value_at_risk(total, alpha)
    if not np.any(tail):
        tail = total == total.max()
    return losses[tail].mean(axis=0)


def loss_given_factor(portfolio: Portfolio, rho: float, systematic: float) -> float:
    conditional = gaussian_conditional_pd(portfolio.pd, np.array([systematic]), rho)
    return float(np.sum(conditional * portfolio.ead * portfolio.lgd))


def vasicek_quantile(portfolio: Portfolio, rho: float, alpha: float) -> float:
    """Large-portfolio loss quantile for a single-factor Gaussian copula."""
    alpha = _require_alpha(alpha)
    stressed_factor = -norm.ppf(alpha)
    return loss_given_factor(portfolio, rho, float(stressed_factor))


def vasicek_expected_shortfall(portfolio: Portfolio, rho: float, alpha: float) -> float:
    """Expected shortfall of the infinitely granular single-factor loss."""
    alpha = _require_alpha(alpha)

    def quantile(level: float) -> float:
        factor = float(norm.ppf(np.clip(1.0 - level, 1e-12, 1.0 - 1e-12)))
        return loss_given_factor(portfolio, rho, factor)

    integral, _ = quad(quantile, alpha, 1.0, epsabs=1e-3, limit=400)
    return float(integral / (1.0 - alpha))
