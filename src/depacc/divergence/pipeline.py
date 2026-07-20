"""Divergence stage: bivariate typology per cell + city-plane row."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from depacc.divergence.cityplane import city_row, upsert_cityplane
from depacc.divergence.typology import bivariate_typology
from depacc.ingest.pipeline import derived_dir


def run_divergence(cfg: dict, city: str, root: Path) -> None:
    out = derived_dir(cfg, city, root)
    surfaces = pd.read_parquet(out / "surfaces.parquet")

    cells, summary = bivariate_typology(
        surfaces, quantile=float(cfg["typology"]["quantile"])
    )
    cells.to_parquet(out / "typology.parquet")
    summary_out = summary.reset_index()
    summary_out["unclassified_pop_share"] = summary.attrs["unclassified_pop_share"]
    summary_out.to_csv(out / "typology_summary.csv", index=False)
    print("typology population shares:")
    for cls, r in summary.iterrows():
        print(f"  {cls}: {r.population_share:6.1%}  ({int(r.n_cells)} cells)")

    row = city_row(surfaces, summary, cfg, city)
    # One-row per-city summary: the unit persisted to the depacc-results
    # branch so separate (batch) runs accumulate into one cross-city table.
    pd.DataFrame([row]).to_csv(out / "cityplane_row.csv", index=False)
    table = upsert_cityplane(row, root / cfg["output"]["root"] / "cityplane.csv")
    print(f"cityplane.csv: {len(table)} cities; {city}: "
          f"gini_ev={row['gini_everyday']:.3f} gini_em={row['gini_emergency']:.3f} "
          f"HH share={row['hh_pop_share']:.1%}")
