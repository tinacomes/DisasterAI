"""Viz stage: static matplotlib figures per city + cross-city plane.

Design conventions: sequential single-hue ramps for magnitude (Oranges =
everyday, Purples = emergency), the Stevens 2x2 bivariate scheme for the
compounding typology (labelled legend; population shares also written as CSV
for a non-color reading), recessive axes, no dual axes. Every figure carries
the cross-sectional (space-for-time) framing in its caption where relevant,
and synthetic fixtures are watermarked.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from depacc.ingest.pipeline import derived_dir

# Stevens bivariate 2x2 corner scheme (LL near-neutral by construction;
# classes carry a labelled legend + CSV table, CVD ΔE validated).
TYPOLOGY_COLORS = {"LL": "#e8e8e8", "HL": "#5ac8c8", "LH": "#be64ac", "HH": "#3b4994"}
TYPOLOGY_LABELS = {
    "LL": "low both",
    "HL": "high everyday only",
    "LH": "high emergency only",
    "HH": "compounding (high both)",
}
REGIME_CMAP = {"everyday": "Oranges", "emergency": "Purples"}


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.set_axisbelow(True)


def _watermark(fig, cfg):
    if cfg["city"].get("synthetic"):
        fig.text(0.5, 0.5, "SYNTHETIC FIXTURE", fontsize=28, color="0.75",
                 alpha=0.35, ha="center", va="center", rotation=25, zorder=0)


def run_viz(cfg: dict, city: str, root: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = derived_dir(cfg, city, root)
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)
    surfaces = pd.read_parquet(out / "surfaces.parquet")
    typology = pd.read_parquet(out / "typology.parquet")
    name = cfg["city"].get("name", city)

    # --- 1. deprivation choropleths (one sequential hue per regime) ---------
    for regime in ("everyday", "emergency"):
        fig, ax = plt.subplots(figsize=(7, 6.5))
        dep = surfaces[f"deprivation_{regime}"]
        vmax = np.nanquantile(dep, 0.98)
        sc = ax.scatter(surfaces.x, surfaces.y, c=dep, s=8, marker="s",
                        cmap=REGIME_CMAP[regime], vmin=0, vmax=vmax, linewidths=0)
        unreach = surfaces[surfaces[f"unreachable_{regime}"]]
        if len(unreach):
            ax.scatter(unreach.x, unreach.y, s=5, marker="x", c="#666666",
                       linewidths=0.5, label=f"unreachable ({len(unreach)} cells)")
            ax.legend(loc="lower right", frameon=False, fontsize=8)
        kind = surfaces[f"deprivation_kind_{regime}"].iloc[0]
        ax.set_title(f"{name}: {regime} potential deprivation ({kind})", fontsize=11)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        cb = fig.colorbar(sc, ax=ax, shrink=0.75)
        cb.set_label(f"deprivation ({'dimensionless' if kind == 'DLF' else 'monetary'})",
                     fontsize=9)
        cb.outline.set_visible(False)
        _watermark(fig, cfg)
        fig.tight_layout()
        fig.savefig(figdir / f"deprivation_{regime}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    # --- 2. bivariate compounding map ---------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for cls, color in TYPOLOGY_COLORS.items():
        sub = typology[typology.typology == cls]
        ax.scatter(sub.x, sub.y, c=color, s=8, marker="s", linewidths=0,
                   label=f"{cls} — {TYPOLOGY_LABELS[cls]}")
    ax.set_title(f"{name}: everyday x emergency co-location "
                 f"(pop-weighted median split)", fontsize=11)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8)
    _watermark(fig, cfg)
    fig.tight_layout()
    fig.savefig(figdir / "compounding_map.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # --- 3. cross-city everyday-vs-emergency plane --------------------------
    plane_path = root / cfg["output"]["root"] / "cityplane.csv"
    if plane_path.exists():
        plane = pd.read_csv(plane_path)
        fig, ax = plt.subplots(figsize=(6.5, 6))
        lim = max(plane.gini_everyday.max(), plane.gini_emergency.max()) * 1.15 + 1e-9
        ax.plot([0, lim], [0, lim], color="0.75", linewidth=1, linestyle="--",
                zorder=1)
        ax.scatter(plane.gini_everyday, plane.gini_emergency,
                   s=20 + 40 * np.log10(plane.population.clip(lower=1) + 1),
                   c="#3b4994", alpha=0.85, linewidths=0, zorder=2)
        for _, r in plane.iterrows():
            label = r["name"] + (" (synthetic)" if r.get("synthetic") else "")
            ax.annotate(label, (r.gini_everyday, r.gini_emergency),
                        textcoords="offset points", xytext=(6, 4), fontsize=8,
                        color="0.25")
        ax.set_xlabel("Gini of everyday deprivation")
        ax.set_ylabel("Gini of emergency deprivation")
        ax.set_title("Cities in the everyday-vs-emergency inequity plane\n"
                     "(cross-sectional; size ~ log population)", fontsize=10)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        _style(ax)
        fig.tight_layout()
        crossdir = root / cfg["output"]["root"] / "figures"
        crossdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(crossdir / "cityplane.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    print(f"figures written to {figdir} (+ cross-city cityplane)")
