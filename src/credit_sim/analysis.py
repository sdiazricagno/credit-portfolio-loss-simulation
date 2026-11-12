"""Run the published capital study and write the charts."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from credit_sim.config import (
    ALPHA_LABEL,
    ALPHAS,
    CHUNK_SIZE,
    N_LOANS,
    N_SIMS,
    N_SIMS_RHO,
    PD_CAP,
    PD_STRESS_MULTIPLIER,
    RHO,
    RHO_GRID,
    SEED,
    SHOCK_SECTOR,
    SHOCK_SECTOR_MULTIPLIER,
    T_DF,
)
from credit_sim.copula import simulate_gaussian_rho_grid, simulate_losses
from credit_sim.metrics import (
    es_contributions,
    expected_shortfall,
    value_at_risk,
    vasicek_expected_shortfall,
    vasicek_quantile,
)
from credit_sim.plots import (
    plot_correlation_sensitivity,
    plot_loss_distribution,
    plot_sector_contributions,
    plot_stress_comparison,
    plot_tail_comparison,
)
from credit_sim.portfolio import (
    analytic_expected_loss,
    effective_loan_count,
    generate_portfolio,
    portfolio_summary,
    scale_pds,
    shock_sector_pds,
)


@dataclass
class RunResult:
    summary: pd.DataFrame
    capital: pd.DataFrame
    sensitivity: pd.DataFrame
    stress: pd.DataFrame
    contributions: pd.DataFrame
    checks: dict[str, float | int | str]
    elapsed_seconds: float


def _pct(amount: float, total_ead: float) -> float:
    return 100.0 * amount / total_ead


def _loss_columns(losses: np.ndarray, total_ead: float) -> dict[str, float]:
    columns = {"EL %": _pct(float(np.mean(losses)), total_ead)}
    for alpha in ALPHAS:
        label = ALPHA_LABEL[alpha]
        columns[f"VaR {label}"] = _pct(value_at_risk(losses, alpha), total_ead)
        columns[f"ES {label}"] = _pct(expected_shortfall(losses, alpha), total_ead)
    return columns


def run_analysis(output_dir: Path | str = "docs") -> RunResult:
    started = time.perf_counter()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    portfolio = generate_portfolio(N_LOANS, SEED)
    total_ead = portfolio.total_ead
    analytic_el = analytic_expected_loss(portfolio)
    summary = portfolio_summary(portfolio)

    gaussian = simulate_losses(
        portfolio,
        n_sims=N_SIMS,
        rho=RHO,
        seed=SEED,
        copula="gaussian",
        by_sector=True,
        chunk_size=CHUNK_SIZE,
    )
    student = simulate_losses(
        portfolio,
        n_sims=N_SIMS,
        rho=RHO,
        seed=SEED,
        copula="student_t",
        df=T_DF,
        chunk_size=CHUNK_SIZE,
    )
    factor_only = simulate_losses(
        portfolio,
        n_sims=N_SIMS,
        rho=RHO,
        seed=SEED,
        copula="gaussian",
        idiosyncratic=False,
        chunk_size=CHUNK_SIZE,
    )

    capital_rows: list[dict[str, float | str]] = []
    limit_row: dict[str, float | str] = {
        "Model": "Vasicek large-portfolio limit",
        "EL %": _pct(analytic_el, total_ead),
    }
    for alpha in ALPHAS:
        label = ALPHA_LABEL[alpha]
        limit_row[f"VaR {label}"] = _pct(vasicek_quantile(portfolio, RHO, alpha), total_ead)
        limit_row[f"ES {label}"] = _pct(vasicek_expected_shortfall(portfolio, RHO, alpha), total_ead)
    capital_rows.append(limit_row)
    capital_rows.append({"Model": "Gaussian, factor only", **_loss_columns(factor_only.losses, total_ead)})
    capital_rows.append({"Model": "Gaussian copula", **_loss_columns(gaussian.losses, total_ead)})
    capital_rows.append(
        {"Model": f"Student-t copula (ν = {T_DF:.0f})", **_loss_columns(student.losses, total_ead)}
    )
    capital = pd.DataFrame(capital_rows)

    grid = simulate_gaussian_rho_grid(portfolio, RHO_GRID, N_SIMS_RHO, SEED, chunk_size=CHUNK_SIZE)
    sensitivity = pd.DataFrame(
        [{"Asset correlation": rho, **_loss_columns(losses, total_ead)} for rho, losses in grid.items()]
    )

    higher_pds = scale_pds(portfolio, PD_STRESS_MULTIPLIER, cap=PD_CAP)
    cre_shock = shock_sector_pds(higher_pds, SHOCK_SECTOR, SHOCK_SECTOR_MULTIPLIER, cap=PD_CAP)
    stressed_losses = {
        "Baseline": gaussian.losses,
        "PDs x 1.5": simulate_losses(
            higher_pds, n_sims=N_SIMS, rho=RHO, seed=SEED, copula="gaussian", chunk_size=CHUNK_SIZE
        ).losses,
        "PDs x 1.5, CRE x 2": simulate_losses(
            cre_shock, n_sims=N_SIMS, rho=RHO, seed=SEED, copula="gaussian", chunk_size=CHUNK_SIZE
        ).losses,
    }
    stress_books = {"Baseline": portfolio, "PDs x 1.5": higher_pds, "PDs x 1.5, CRE x 2": cre_shock}
    stress_rows: list[dict[str, float | str]] = []
    for name, losses in stressed_losses.items():
        book = stress_books[name]
        row: dict[str, float | str] = {
            "Scenario": name,
            "EL %": _pct(analytic_expected_loss(book), book.total_ead),
        }
        for alpha, key in ((0.99, "99%"), (0.999, "99.9%")):
            row[f"VaR {key}"] = _pct(value_at_risk(losses, alpha), book.total_ead)
            row[f"ES {key}"] = _pct(expected_shortfall(losses, alpha), book.total_ead)
        stress_rows.append(row)
    stress = pd.DataFrame(stress_rows)

    assert gaussian.sector_losses is not None
    contrib = es_contributions(gaussian.sector_losses, 0.99)
    loan_el = portfolio.pd * portfolio.lgd * portfolio.ead
    contribution_rows = []
    for i, name in enumerate(portfolio.sector_names):
        mask = portfolio.sector == name
        contribution_rows.append(
            {
                "Sector": name,
                "EAD share": float(portfolio.ead[mask].sum() / total_ead),
                "EL share": float(loan_el[mask].sum() / analytic_el),
                "ES share": float(contrib[i] / contrib.sum()),
            }
        )
    contributions = pd.DataFrame(contribution_rows).sort_values("ES share", ascending=False)

    plot_loss_distribution(gaussian.losses, total_ead, RHO, N_SIMS, output / "loss_distribution.png")
    plot_tail_comparison(gaussian.losses, student.losses, total_ead, T_DF, output / "gaussian_vs_t_tails.png")
    plot_correlation_sensitivity(sensitivity, output / "correlation_sensitivity.png")
    plot_stress_comparison(stress, output / "stress_comparison.png")
    plot_sector_contributions(contributions, output / "sector_contributions.png")
    plt.close("all")

    gaussian_el = float(np.mean(gaussian.losses))
    student_el = float(np.mean(student.losses))
    checks: dict[str, float | int | str] = {
        "n_loans": portfolio.n_loans,
        "n_sims": N_SIMS,
        "n_sims_rho": N_SIMS_RHO,
        "seed": SEED,
        "rho": RHO,
        "t_df": T_DF,
        "total_ead": total_ead,
        "effective_loans": effective_loan_count(portfolio),
        "largest_exposure_share": float(portfolio.ead.max() / total_ead),
        "analytic_el": analytic_el,
        "gaussian_el": gaussian_el,
        "student_el": student_el,
        "gaussian_el_rel_error": (gaussian_el - analytic_el) / analytic_el,
        "student_el_rel_error": (student_el - analytic_el) / analytic_el,
        "max_pd_baseline": float(portfolio.pd.max()),
        "max_pd_stressed": float(cre_shock.pd.max()),
    }
    for alpha in ALPHAS:
        label = ALPHA_LABEL[alpha].replace("%", "").replace(".", "")
        checks[f"vasicek_var_{label}"] = vasicek_quantile(portfolio, RHO, alpha)
        checks[f"factor_var_{label}"] = value_at_risk(factor_only.losses, alpha)
        checks[f"gaussian_var_{label}"] = value_at_risk(gaussian.losses, alpha)
        checks[f"gaussian_es_{label}"] = expected_shortfall(gaussian.losses, alpha)
        checks[f"student_var_{label}"] = value_at_risk(student.losses, alpha)
        checks[f"student_es_{label}"] = expected_shortfall(student.losses, alpha)
    elapsed = time.perf_counter() - started
    return RunResult(
        summary=summary,
        capital=capital,
        sensitivity=sensitivity,
        stress=stress,
        contributions=contributions,
        checks=checks,
        elapsed_seconds=elapsed,
    )


def _print_frame(title: str, frame: pd.DataFrame) -> None:
    print(f"\n{title}")
    print(frame.to_string(index=False, float_format=lambda value: f"{value:,.6f}"))


def print_report(result: RunResult) -> None:
    checks = result.checks
    total_ead = float(checks["total_ead"])
    print("Credit portfolio loss simulation")
    print(f"Loans: {checks['n_loans']:,}")
    print(f"Total EAD: {total_ead:,.2f}")
    print(f"Effective loans (1/HHI): {float(checks['effective_loans']):,.1f}")
    print(f"Largest exposure share: {float(checks['largest_exposure_share']):.2%}")
    print(f"Asset correlation: {checks['rho']}")
    print(f"Student-t degrees of freedom: {checks['t_df']}")
    print(f"Scenarios: {checks['n_sims']:,} (rho grid {checks['n_sims_rho']:,})")
    print(f"Seed: {checks['seed']}")
    print(f"Runtime seconds: {result.elapsed_seconds:.1f}")
    print(f"Max PD baseline / full stress: {checks['max_pd_baseline']:.4f} / {checks['max_pd_stressed']:.4f}")
    print(
        "EL relative error gaussian / student-t: "
        f"{float(checks['gaussian_el_rel_error']):+.4%} / {float(checks['student_el_rel_error']):+.4%}"
    )
    _print_frame("Portfolio", result.summary)
    _print_frame("Capital, percent of EAD", result.capital)
    _print_frame("Correlation sensitivity, percent of EAD", result.sensitivity)
    _print_frame("Stress, percent of EAD", result.stress)
    _print_frame("Sector contributions", result.contributions)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the credit portfolio loss simulation")
    parser.add_argument("--output", type=Path, default=Path("docs"))
    parser.add_argument("--results-json", type=Path, default=None)
    args = parser.parse_args(argv)
    result = run_analysis(args.output)
    print_report(result)
    if args.results_json is not None:
        payload = {
            "checks": result.checks,
            "elapsed_seconds": result.elapsed_seconds,
            "summary": json.loads(result.summary.to_json(orient="records", double_precision=10)),
            "capital": json.loads(result.capital.to_json(orient="records", double_precision=10)),
            "sensitivity": json.loads(result.sensitivity.to_json(orient="records", double_precision=10)),
            "stress": json.loads(result.stress.to_json(orient="records", double_precision=10)),
            "contributions": json.loads(result.contributions.to_json(orient="records", double_precision=10)),
        }
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        args.results_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
