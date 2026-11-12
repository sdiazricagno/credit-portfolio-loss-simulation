"""One-factor Gaussian and Student-t credit copulas."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, t

from credit_sim.portfolio import Portfolio


@dataclass(frozen=True)
class SimulationResult:
    losses: np.ndarray
    sector_losses: np.ndarray | None
    sector_names: tuple[str, ...]
    copula: str
    rho: float
    df: float | None
    n_sims: int
    seed: int


def _require_rho(rho: float) -> float:
    rho = float(rho)
    if not 0.0 < rho < 1.0:
        raise ValueError("rho must lie in (0, 1)")
    return rho


def _clip_pds(pds: np.ndarray) -> np.ndarray:
    return np.clip(pds, 1e-8, 1.0 - 1e-8)


def gaussian_conditional_pd(pds: np.ndarray, systematic: np.ndarray, rho: float) -> np.ndarray:
    """P(default | systematic). A higher systematic draw is a healthier economy."""
    rho = _require_rho(rho)
    factor = np.asarray(systematic, dtype=float)
    numer = norm.ppf(_clip_pds(pds))[None, :] - np.sqrt(rho) * factor[:, None]
    return norm.cdf(numer / np.sqrt(1.0 - rho))


def student_t_conditional_pd(
    pds: np.ndarray,
    systematic: np.ndarray,
    mixing: np.ndarray,
    rho: float,
    df: float,
) -> np.ndarray:
    """Conditional PD for a one-factor t copula with a shared chi-square draw."""
    rho = _require_rho(rho)
    if df <= 2:
        raise ValueError("df must be greater than 2")
    factor = np.asarray(systematic, dtype=float)
    shock = np.sqrt(np.asarray(mixing, dtype=float) / df)
    numer = t.ppf(_clip_pds(pds), df)[None, :] * shock[:, None] - np.sqrt(rho) * factor[:, None]
    return norm.cdf(numer / np.sqrt(1.0 - rho))


def _aggregate(
    weights: np.ndarray,
    severity: np.ndarray,
    masks: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray | None]:
    total = weights @ severity
    if not masks:
        return total, None
    by_sector = np.column_stack([weights[:, mask] @ severity[mask] for mask in masks])
    return total, by_sector


def simulate_losses(
    portfolio: Portfolio,
    n_sims: int,
    rho: float,
    seed: int,
    copula: str = "gaussian",
    df: float = 5.0,
    idiosyncratic: bool = True,
    by_sector: bool = False,
    chunk_size: int = 4_000,
) -> SimulationResult:
    rho = _require_rho(rho)
    if n_sims < 1:
        raise ValueError("n_sims must be positive")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if copula not in {"gaussian", "student_t"}:
        raise ValueError("copula must be 'gaussian' or 'student_t'")
    if copula == "student_t" and df <= 2:
        raise ValueError("df must be greater than 2")

    rng = np.random.default_rng(seed)
    severity = portfolio.ead * portfolio.lgd
    pds = _clip_pds(portfolio.pd)
    threshold = norm.ppf(pds) if copula == "gaussian" else t.ppf(pds, df)
    masks = [portfolio.sector == name for name in portfolio.sector_names] if by_sector else []
    losses = np.empty(n_sims, dtype=float)
    sector_losses = np.empty((n_sims, len(portfolio.sector_names)), dtype=float) if by_sector else None
    sqrt_rho = np.sqrt(rho)
    sqrt_idio = np.sqrt(1.0 - rho)

    offset = 0
    while offset < n_sims:
        size = min(chunk_size, n_sims - offset)
        systematic = rng.standard_normal(size)
        if idiosyncratic:
            eps = rng.standard_normal((size, portfolio.n_loans))
            latent = sqrt_rho * systematic[:, None] + sqrt_idio * eps
            if copula == "student_t":
                latent *= np.sqrt(df / rng.chisquare(df, size=size))[:, None]
            weights = (latent < threshold).astype(np.float64)
        elif copula == "gaussian":
            weights = gaussian_conditional_pd(pds, systematic, rho)
        else:
            mixing = rng.chisquare(df, size=size)
            weights = student_t_conditional_pd(pds, systematic, mixing, rho, df)
        chunk_losses, chunk_sectors = _aggregate(weights, severity, masks)
        losses[offset : offset + size] = chunk_losses
        if sector_losses is not None and chunk_sectors is not None:
            sector_losses[offset : offset + size] = chunk_sectors
        offset += size

    return SimulationResult(
        losses=losses,
        sector_losses=sector_losses,
        sector_names=portfolio.sector_names,
        copula=copula,
        rho=rho,
        df=float(df) if copula == "student_t" else None,
        n_sims=n_sims,
        seed=seed,
    )


def simulate_gaussian_rho_grid(
    portfolio: Portfolio,
    rhos: tuple[float, ...] | list[float],
    n_sims: int,
    seed: int,
    chunk_size: int = 4_000,
) -> dict[float, np.ndarray]:
    """Gaussian losses on a rho grid, reusing the same factor and idiosyncratic draws."""
    if n_sims < 1:
        raise ValueError("n_sims must be positive")
    if not rhos:
        raise ValueError("rhos must be non-empty")
    checked = tuple(_require_rho(rho) for rho in rhos)
    rng = np.random.default_rng(seed)
    severity = portfolio.ead * portfolio.lgd
    threshold = norm.ppf(_clip_pds(portfolio.pd))
    out = {rho: np.empty(n_sims, dtype=float) for rho in checked}
    offset = 0
    while offset < n_sims:
        size = min(chunk_size, n_sims - offset)
        systematic = rng.standard_normal(size)
        idiosyncratic = rng.standard_normal((size, portfolio.n_loans))
        stop = offset + size
        for rho in checked:
            latent = np.sqrt(rho) * systematic[:, None] + np.sqrt(1.0 - rho) * idiosyncratic
            out[rho][offset:stop] = (latent < threshold).astype(np.float64) @ severity
        offset = stop
    return out
