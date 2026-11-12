"""Synthetic loan book: sectors, rating PDs, beta LGDs, lognormal exposures."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

RATING_PD: dict[str, float] = {
    "AAA": 0.0002,
    "AA": 0.0005,
    "A": 0.0008,
    "BBB": 0.0025,
    "BB": 0.0120,
    "B": 0.0450,
    "CCC": 0.1500,
}


@dataclass(frozen=True)
class SectorSpec:
    name: str
    weight: float
    mean_lgd: float
    lgd_concentration: float
    median_ead: float
    ead_sigma: float
    rating_weights: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if not 0.0 < self.mean_lgd < 1.0:
            raise ValueError("mean_lgd must lie in (0, 1)")
        if self.lgd_concentration <= 0 or self.median_ead <= 0 or self.ead_sigma <= 0 or self.weight <= 0:
            raise ValueError("sector weight, LGD concentration, and exposure parameters must be positive")
        for rating, weight in self.rating_weights:
            if rating not in RATING_PD:
                raise ValueError(f"unknown rating: {rating}")
            if weight <= 0:
                raise ValueError("rating weights must be positive")


SECTOR_SPECS: tuple[SectorSpec, ...] = (
    SectorSpec(
        name="Mortgage",
        weight=0.30,
        mean_lgd=0.22,
        lgd_concentration=40.0,
        median_ead=160_000.0,
        ead_sigma=0.35,
        rating_weights=(("A", 0.15), ("BBB", 0.40), ("BB", 0.30), ("B", 0.12), ("CCC", 0.03)),
    ),
    SectorSpec(
        name="Consumer",
        weight=0.28,
        mean_lgd=0.55,
        lgd_concentration=28.0,
        median_ead=12_000.0,
        ead_sigma=0.55,
        rating_weights=(("BBB", 0.10), ("BB", 0.30), ("B", 0.40), ("CCC", 0.20)),
    ),
    SectorSpec(
        name="SME",
        weight=0.22,
        mean_lgd=0.42,
        lgd_concentration=36.0,
        median_ead=80_000.0,
        ead_sigma=0.60,
        rating_weights=(("BBB", 0.15), ("BB", 0.35), ("B", 0.35), ("CCC", 0.15)),
    ),
    SectorSpec(
        name="Corporate",
        weight=0.12,
        mean_lgd=0.38,
        lgd_concentration=40.0,
        median_ead=450_000.0,
        ead_sigma=0.75,
        rating_weights=(
            ("AA", 0.05),
            ("A", 0.15),
            ("BBB", 0.35),
            ("BB", 0.25),
            ("B", 0.15),
            ("CCC", 0.05),
        ),
    ),
    SectorSpec(
        name="CRE",
        weight=0.08,
        mean_lgd=0.32,
        lgd_concentration=36.0,
        median_ead=400_000.0,
        ead_sigma=0.70,
        rating_weights=(("A", 0.05), ("BBB", 0.25), ("BB", 0.35), ("B", 0.25), ("CCC", 0.10)),
    ),
)


@dataclass(frozen=True)
class Portfolio:
    sector: np.ndarray
    rating: np.ndarray
    pd: np.ndarray
    lgd: np.ndarray
    ead: np.ndarray
    sector_names: tuple[str, ...]

    def __post_init__(self) -> None:
        n = len(self.pd)
        if n == 0:
            raise ValueError("portfolio must contain at least one loan")
        if not self.sector_names:
            raise ValueError("sector_names must be non-empty")
        for column in (self.sector, self.rating, self.lgd, self.ead):
            if len(column) != n:
                raise ValueError("portfolio columns must share one length")

    @property
    def n_loans(self) -> int:
        return int(self.pd.shape[0])

    @property
    def total_ead(self) -> float:
        return float(self.ead.sum())


def analytic_expected_loss(portfolio: Portfolio) -> float:
    return float(np.sum(portfolio.pd * portfolio.lgd * portfolio.ead))


def effective_loan_count(portfolio: Portfolio) -> float:
    weights = portfolio.ead / portfolio.ead.sum()
    return float(1.0 / np.sum(weights * weights))


def generate_portfolio(n_loans: int = 2_500, seed: int = 42) -> Portfolio:
    if n_loans < 1:
        raise ValueError("n_loans must be positive")
    rng = np.random.default_rng(seed)
    weights = np.array([spec.weight for spec in SECTOR_SPECS], dtype=float)
    weights /= weights.sum()
    counts = rng.multinomial(n_loans, weights)

    sectors: list[np.ndarray] = []
    ratings: list[np.ndarray] = []
    pds: list[np.ndarray] = []
    lgds: list[np.ndarray] = []
    eads: list[np.ndarray] = []
    for spec, count in zip(SECTOR_SPECS, counts, strict=True):
        count = int(count)
        if count == 0:
            continue
        names = [rating for rating, _ in spec.rating_weights]
        probs = np.array([weight for _, weight in spec.rating_weights], dtype=float)
        probs /= probs.sum()
        pd_lookup = np.array([RATING_PD[rating] for rating in names], dtype=float)
        drawn = rng.choice(len(names), size=count, p=probs)
        sectors.append(np.full(count, spec.name, dtype="U16"))
        ratings.append(np.array([names[i] for i in drawn], dtype="U3"))
        pds.append(pd_lookup[drawn])
        alpha = spec.mean_lgd * spec.lgd_concentration
        beta = (1.0 - spec.mean_lgd) * spec.lgd_concentration
        lgds.append(rng.beta(alpha, beta, size=count))
        eads.append(rng.lognormal(np.log(spec.median_ead), spec.ead_sigma, size=count))

    return Portfolio(
        sector=np.concatenate(sectors),
        rating=np.concatenate(ratings),
        pd=np.concatenate(pds),
        lgd=np.concatenate(lgds),
        ead=np.concatenate(eads),
        sector_names=tuple(spec.name for spec in SECTOR_SPECS),
    )


def portfolio_summary(portfolio: Portfolio) -> pd.DataFrame:
    total_ead = portfolio.total_ead
    loan_el = portfolio.pd * portfolio.lgd * portfolio.ead
    total_el = float(loan_el.sum())
    rows: list[dict[str, float | int | str]] = []
    for name in portfolio.sector_names:
        mask = portfolio.sector == name
        ead = float(portfolio.ead[mask].sum()) if np.any(mask) else 0.0
        el = float(loan_el[mask].sum()) if np.any(mask) else 0.0
        if ead > 0.0:
            weighted_pd = float(np.average(portfolio.pd[mask], weights=portfolio.ead[mask]))
            weighted_lgd = float(np.average(portfolio.lgd[mask], weights=portfolio.ead[mask]))
        else:
            weighted_pd = 0.0
            weighted_lgd = 0.0
        rows.append(
            {
                "Sector": name,
                "Loans": int(np.sum(mask)),
                "EAD": ead,
                "EAD share": ead / total_ead if total_ead else 0.0,
                "PD": weighted_pd,
                "LGD": weighted_lgd,
                "EL": el,
                "EL rate": el / ead if ead else 0.0,
                "EL share": el / total_el if total_el else 0.0,
            }
        )
    return pd.DataFrame(rows)


def scale_pds(portfolio: Portfolio, multiplier: float, cap: float = 0.95) -> Portfolio:
    if multiplier <= 0:
        raise ValueError("multiplier must be positive")
    if not 0.0 < cap < 1.0:
        raise ValueError("cap must lie in (0, 1)")
    pds = np.minimum(portfolio.pd * multiplier, cap)
    return replace(portfolio, pd=pds)


def shock_sector_pds(portfolio: Portfolio, sector: str, multiplier: float, cap: float = 0.95) -> Portfolio:
    if sector not in portfolio.sector_names:
        raise ValueError(f"unknown sector: {sector}")
    if multiplier <= 0:
        raise ValueError("multiplier must be positive")
    if not 0.0 < cap < 1.0:
        raise ValueError("cap must lie in (0, 1)")
    pds = portfolio.pd.copy()
    mask = portfolio.sector == sector
    pds[mask] = np.minimum(pds[mask] * multiplier, cap)
    return replace(portfolio, pd=pds)
