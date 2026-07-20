"""Equity stage: population-weighted indices per regime + Tier-2 gradient
regressions on SES covariates."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from depacc.equity.indices import concentration_index, weighted_gini, weighted_mean
from depacc.equity.regressions import density_gradient, gradient_regression
from depacc.ingest.pipeline import derived_dir


def run_equity(cfg: dict, city: str, root: Path) -> None:
    out = derived_dir(cfg, city, root)
    surfaces = pd.read_parquet(out / "surfaces.parquet")
    pop = surfaces["population"]
    ses_cols = sorted(c for c in surfaces.columns if c.startswith("ses_"))

    rows = []
    for regime in ("everyday", "emergency"):
        dep = surfaces[f"deprivation_{regime}"]
        row = {
            "regime": regime,
            "weighted_mean": weighted_mean(dep, pop),
            "gini": weighted_gini(dep, pop),
        }
        # Concentration index against the first income/rent-like SES column.
        ses_rank_col = next(
            (c for c in ses_cols if any(k in c for k in ("income", "rent", "filosofi", "imd"))),
            None,
        )
        if ses_rank_col:
            row["concentration_index"] = concentration_index(
                dep, surfaces[ses_rank_col], pop
            )
            row["concentration_ses_col"] = ses_rank_col
        rows.append(row)
    indices = pd.DataFrame(rows)
    indices.to_csv(out / "equity_indices.csv", index=False)
    print(indices.to_string(index=False))

    reg_frames = []
    for regime in ("everyday", "emergency"):
        outcome = f"deprivation_{regime}"
        try:
            d = density_gradient(surfaces, outcome).assign(regime=regime, model="density")
            reg_frames.append(d)
        except (ValueError, ImportError) as err:
            print(f"NOTE: density gradient skipped for {regime}: {err}")
        if ses_cols:
            try:
                g = gradient_regression(surfaces, outcome, ses_cols).assign(
                    regime=regime, model="ses")
                reg_frames.append(g)
            except (ValueError, ImportError) as err:
                print(f"NOTE: SES regression skipped for {regime}: {err}")
    if reg_frames:
        regs = pd.concat(reg_frames, ignore_index=True)
        regs.to_csv(out / "equity_regressions.csv", index=False)
        slopes = regs[regs.term != "const"]
        print(slopes[["regime", "model", "term", "coef", "p"]].to_string(index=False))
