"""Tier-2 within-city equity gradient regressions.

Population-weighted least squares of cell deprivation on SES covariates
(income/rent proxy, age structure, household composition — whatever
`ses_*` columns the city's ingest joined). Reported per regime with robust
(HC1) standard errors. Cross-sectional, descriptive gradients — no causal
claims.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def gradient_regression(surfaces: pd.DataFrame, outcome: str,
                        ses_cols: list[str]) -> pd.DataFrame:
    """WLS of ``outcome`` on standardised SES covariates, weights=population.

    Returns a tidy frame: term, coef, se, t, p, n. Covariates are z-scored so
    coefficients are comparable within a city.
    """
    import statsmodels.api as sm

    cols = [outcome, "population", *ses_cols]
    df = surfaces[cols].replace([np.inf, -np.inf], np.nan).dropna()
    df = df[df.population > 0]
    if len(df) < len(ses_cols) + 5:
        raise ValueError(
            f"Too few complete cells ({len(df)}) for regression on {ses_cols}"
        )
    X = df[ses_cols].apply(lambda c: (c - c.mean()) / c.std(ddof=0))
    X = sm.add_constant(X)
    model = sm.WLS(df[outcome], X, weights=df.population)
    fit = model.fit(cov_type="HC1")
    return pd.DataFrame({
        "term": fit.params.index,
        "coef": fit.params.to_numpy(),
        "se": fit.bse.to_numpy(),
        "t": fit.tvalues.to_numpy(),
        "p": fit.pvalues.to_numpy(),
        "n": len(df),
        "r2": fit.rsquared,
    })


def density_gradient(surfaces: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Deprivation-density slope: WLS of outcome on log10 population density
    (population per cell is density on a uniform 100 m grid)."""
    df = surfaces[[outcome, "population"]].dropna()
    df = df[df.population > 0]
    x = np.log10(df.population)
    return gradient_regression(
        df.assign(ses_log10_density=x), outcome, ["ses_log10_density"]
    )
