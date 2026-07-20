"""Per-city feature vectors for cross-city analysis.

Assembled from cityplane.csv plus each city's equity outputs:

    mean_everyday, mean_emergency        population-weighted levels
    gini_everyday, gini_emergency        inequity
    gini_divergence, rank_corr           everyday-emergency divergence
    hh_pop_share                         compounding share
    slope_density_<regime>               deprivation-density gradient
    slope_ses_<regime>                   deprivation-income/rent gradient (Tier 2)
    log10_population                     the size axis (space-for-time)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
    "mean_everyday", "mean_emergency",
    "gini_everyday", "gini_emergency",
    "gini_divergence", "rank_corr", "hh_pop_share",
    "slope_density_everyday", "slope_density_emergency",
    "slope_ses_everyday", "slope_ses_emergency",
]


def build_city_vectors(cfg: dict, root: Path) -> pd.DataFrame:
    derived = root / cfg["output"]["root"]
    plane = pd.read_csv(derived / "cityplane.csv")
    rows = []
    for _, r in plane.iterrows():
        row = r.to_dict()
        row["log10_population"] = float(np.log10(max(r.population, 1.0)))
        regs_path = derived / r.city / "equity_regressions.csv"
        if regs_path.exists():
            regs = pd.read_csv(regs_path)
            slopes = regs[regs.term != "const"]
            for _, s in slopes.iterrows():
                if s.model == "density":
                    row[f"slope_density_{s.regime}"] = s.coef
                elif s.model == "ses" and ("income" in str(s.term) or "rent" in str(s.term)):
                    row[f"slope_ses_{s.regime}"] = s.coef
        rows.append(row)
    vectors = pd.DataFrame(rows)
    vectors.to_csv(derived / "cityvector.csv", index=False)
    return vectors
