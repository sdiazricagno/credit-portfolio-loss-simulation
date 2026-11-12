"""Charts for the published simulation run."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from credit_sim.metrics import expected_shortfall, value_at_risk

NAVY = "#1F4E79"
RUST = "#A8431F"
GOLD = "#C47B17"
GREEN = "#1F7A4D"
SLATE = "#5D6D7E"
PALE = "#8FA4B8"

_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#1C2833",
    "axes.labelcolor": "#1C2833",
    "xtick.color": "#1C2833",
    "ytick.color": "#1C2833",
    "text.color": "#1C2833",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#E5E8EB",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titlepad": 12,
    "legend.frameon": False,
    "figure.dpi": 120,
    "savefig.dpi": 150,
}


def _save(fig: Figure, path: Path | None) -> None:
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")


def _rates(losses: np.ndarray, total_ead: float) -> np.ndarray:
    return 100.0 * np.asarray(losses, dtype=float) / total_ead


def plot_loss_distribution(
    losses: np.ndarray,
    total_ead: float,
    rho: float,
    n_sims: int,
    path: Path | None = None,
) -> Figure:
    rates = _rates(losses, total_ead)
    el = float(rates.mean())
    var_99 = value_at_risk(rates, 0.99)
    es_99 = expected_shortfall(rates, 0.99)
    var_999 = value_at_risk(rates, 0.999)
    upper = var_999 * 1.25
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        ax.hist(
            rates,
            bins=55,
            range=(0.0, upper),
            color=NAVY,
            alpha=0.9,
            density=True,
            label="Simulated loss",
        )
        guides = (
            ("Expected loss", el, GREEN, "-"),
            ("VaR 99%", var_99, GOLD, "--"),
            ("ES 99%", es_99, RUST, "-"),
            ("VaR 99.9%", var_999, NAVY, "-."),
        )
        for label, level, color, style in guides:
            ax.axvline(level, color=color, linestyle=style, linewidth=1.6, label=f"{label}  {level:.2f}%")
        ax.set_xlim(0.0, upper)
        ax.set_xlabel("Portfolio loss, % of exposure")
        ax.set_ylabel("Density")
        ax.set_title(
            f"Portfolio loss distribution\nGaussian copula, asset correlation {rho:.2f}, {n_sims:,} scenarios"
        )
        ax.legend(loc="upper right")
        fig.tight_layout()
        _save(fig, path)
    return fig


def plot_tail_comparison(
    gaussian_losses: np.ndarray,
    student_losses: np.ndarray,
    total_ead: float,
    df: float,
    path: Path | None = None,
) -> Figure:
    def curve(losses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rates = np.sort(_rates(losses, total_ead))
        n = rates.size
        survival = (n - np.arange(n)) / n
        start = int(np.floor(0.90 * n))
        return rates[start:], survival[start:]

    g_x, g_y = curve(gaussian_losses)
    t_x, t_y = curve(student_losses)
    g_var = value_at_risk(_rates(gaussian_losses, total_ead), 0.999)
    t_var = value_at_risk(_rates(student_losses, total_ead), 0.999)
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        ax.semilogy(g_x, g_y, color=NAVY, linewidth=1.8, label="Gaussian copula")
        ax.semilogy(t_x, t_y, color=RUST, linewidth=1.8, label=f"Student-t copula, ν = {df:.0f}")
        ax.axvline(
            g_var,
            color=NAVY,
            linestyle="--",
            linewidth=1.1,
            label=f"Gaussian VaR 99.9%  {g_var:.2f}%",
        )
        ax.axvline(
            t_var,
            color=RUST,
            linestyle="--",
            linewidth=1.1,
            label=f"Student-t VaR 99.9%  {t_var:.2f}%",
        )
        ax.set_xlabel("Portfolio loss, % of exposure")
        ax.set_ylabel("Probability of a larger loss")
        ax.set_title(
            "Upper tail under two dependence assumptions\nFrom the 90th percentile of each simulated sample"
        )
        ax.legend(loc="upper right")
        fig.tight_layout()
        _save(fig, path)
    return fig


def plot_correlation_sensitivity(sensitivity: pd.DataFrame, path: Path | None = None) -> Figure:
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        x = sensitivity["Asset correlation"]
        series = (
            ("EL %", "Expected loss", GREEN, "-"),
            ("VaR 99%", "VaR 99%", GOLD, "--"),
            ("ES 99%", "ES 99%", RUST, "-"),
            ("VaR 99.9%", "VaR 99.9%", NAVY, "-."),
        )
        for column, label, color, style in series:
            ax.plot(
                x,
                sensitivity[column],
                color=color,
                linestyle=style,
                marker="o",
                linewidth=1.7,
                label=label,
            )
        ax.set_xlabel("Asset correlation")
        ax.set_ylabel("Portfolio loss, % of exposure")
        ax.set_title(
            "How much extra capital correlation demands\n"
            "Gaussian copula, common random numbers across the grid"
        )
        ax.legend(loc="upper left")
        fig.tight_layout()
        _save(fig, path)
    return fig


def plot_stress_comparison(stress: pd.DataFrame, path: Path | None = None) -> Figure:
    metrics = ["EL %", "VaR 99%", "ES 99%", "VaR 99.9%"]
    scenarios = list(stress["Scenario"])
    colors = (PALE, NAVY, RUST)
    x = np.arange(len(metrics))
    width = 0.24
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(8.6, 4.8))
        for i, scenario in enumerate(scenarios):
            values = [float(stress.loc[stress["Scenario"] == scenario, metric].iloc[0]) for metric in metrics]
            offset = (i - (len(scenarios) - 1) / 2) * width
            ax.bar(x + offset, values, width=width, color=colors[i % len(colors)], label=scenario)
        ax.set_xticks(x, metrics)
        ax.set_ylabel("Portfolio loss, % of exposure")
        ax.set_title(
            "Baseline against a PD uplift and a CRE sector shock\n"
            "Gaussian copula, same seed and asset correlation"
        )
        ax.legend(loc="upper left")
        fig.tight_layout()
        _save(fig, path)
    return fig


def plot_sector_contributions(contributions: pd.DataFrame, path: Path | None = None) -> Figure:
    frame = contributions.sort_values("ES share", ascending=True)
    y = np.arange(len(frame))
    height = 0.36
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        ax.barh(
            y - height / 2,
            frame["EAD share"] * 100,
            height=height,
            color=PALE,
            label="Share of exposure",
        )
        ax.barh(y + height / 2, frame["ES share"] * 100, height=height, color=NAVY, label="Share of ES 99%")
        ax.set_yticks(y, frame["Sector"])
        ax.set_xlabel("Share of portfolio, %")
        ax.set_title(
            "Where tail loss sits, versus where the exposure sits\nGaussian copula, expected shortfall at 99%"
        )
        ax.legend(loc="lower right")
        fig.tight_layout()
        _save(fig, path)
    return fig
