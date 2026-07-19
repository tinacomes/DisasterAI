"""Cross-city clustering and the size-gradient (space-for-time) read.

k-means and Ward hierarchical clustering on standardised city feature
vectors, plus OLS of the everyday-emergency divergence measures on
log10(FUA population). All inference is cross-sectional: the size gradient
is read space-for-time, never as observed temporal change (stated on every
output).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from depacc.cityvector.features import FEATURES

MIN_CITIES = 5  # below this, clustering is meaningless; only the table is written


def cluster_cities(vectors: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    from sklearn.cluster import AgglomerativeClustering, KMeans
    from sklearn.preprocessing import StandardScaler

    feats = [f for f in FEATURES if f in vectors.columns]
    x = vectors[feats].astype(float)
    keep = x.notna().all(axis=1)
    if keep.sum() < max(MIN_CITIES, n_clusters):
        print(f"NOTE: only {int(keep.sum())} complete city vectors; "
              f"clustering skipped (needs >= {max(MIN_CITIES, n_clusters)})")
        vectors["cluster_kmeans"] = pd.NA
        vectors["cluster_ward"] = pd.NA
        return vectors
    z = StandardScaler().fit_transform(x[keep])
    vectors.loc[keep, "cluster_kmeans"] = (
        KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(z)
    )
    vectors.loc[keep, "cluster_ward"] = (
        AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(z)
    )
    return vectors


def size_gradient(vectors: pd.DataFrame,
                  outcomes: tuple[str, ...] = ("gini_divergence", "rank_corr",
                                               "hh_pop_share")) -> pd.DataFrame:
    """OLS of each divergence outcome on log10 population (HC1 errors).
    Cross-sectional space-for-time trajectory read."""
    import statsmodels.api as sm

    rows = []
    for outcome in outcomes:
        df = vectors[["log10_population", outcome]].dropna()
        if len(df) < 4:
            continue
        X = sm.add_constant(df["log10_population"])
        fit = sm.OLS(df[outcome], X).fit(cov_type="HC1")
        rows.append({
            "outcome": outcome,
            "slope_per_log10_pop": fit.params["log10_population"],
            "se": fit.bse["log10_population"],
            "p": fit.pvalues["log10_population"],
            "n_cities": len(df),
            "r2": fit.rsquared,
            "inference": "cross-sectional space-for-time",
        })
    return pd.DataFrame(rows)


def run_cross_city(cfg: dict, root: Path, n_clusters: int = 3) -> None:
    from depacc.cityvector.features import build_city_vectors

    derived = root / cfg["output"]["root"]
    vectors = build_city_vectors(cfg, root)
    real = vectors[~vectors.synthetic.astype(bool)] if "synthetic" in vectors else vectors
    if len(real) < len(vectors):
        print(f"NOTE: {len(vectors) - len(real)} synthetic fixture(s) excluded "
              f"from cross-city statistics")
    real = cluster_cities(real.copy(), n_clusters=n_clusters)
    real.to_csv(derived / "cityvector_clustered.csv", index=False)

    grad = size_gradient(real)
    if not grad.empty:
        grad.to_csv(derived / "size_gradient.csv", index=False)
        print(grad.to_string(index=False))
    else:
        print("NOTE: too few real cities for the size-gradient regression")

    _plot_cross_city(real, derived)


def _plot_cross_city(vectors: pd.DataFrame, derived: Path) -> None:
    if len(vectors) < 2:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = derived / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    # Categorical cluster colors: fixed assignment order (skill-validated set).
    cluster_colors = ["#5ac8c8", "#be64ac", "#3b4994", "#e0a04e"]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    have_clusters = vectors["cluster_kmeans"].notna().any()
    for i, (_, r) in enumerate(vectors.iterrows()):
        color = (cluster_colors[int(r.cluster_kmeans) % len(cluster_colors)]
                 if have_clusters and pd.notna(r.cluster_kmeans) else "#3b4994")
        ax.scatter(r.log10_population, r.gini_divergence, c=color, s=40,
                   linewidths=0)
        ax.annotate(r["name"], (r.log10_population, r.gini_divergence),
                    textcoords="offset points", xytext=(6, 4), fontsize=8,
                    color="0.25")
    ax.axhline(0, color="0.75", linewidth=1, linestyle="--")
    ax.set_xlabel("log10 FUA population")
    ax.set_ylabel("Gini(emergency) − Gini(everyday)")
    ax.set_title("Everyday-emergency divergence along the city-size gradient\n"
                 "(cross-sectional space-for-time reading"
                 + ("; color = k-means cluster)" if have_clusters else ")"),
                 fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(figdir / "size_gradient.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"cross-city figure: {figdir / 'size_gradient.png'}")
