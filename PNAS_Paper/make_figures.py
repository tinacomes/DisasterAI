#!/usr/bin/env python3
"""Regenerate the PNAS manuscript figures from the archived result JSONs.

All data are read from the canonical archives (results-mechfix/,
results-gap-sweep-mechfix/, results-robustness/); nothing is re-simulated
and no figure data are hand-edited. Run from the repository root:

    python3 PNAS_Paper/make_figures.py

Outputs go to PNAS_Paper/Figures/.
"""
import glob
import json
import os
import re

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "PNAS_Paper", "Figures")
os.makedirs(OUT, exist_ok=True)

MECHFIX = os.path.join(ROOT, "results-mechfix")

C_MAIN = "#d95f02"     # main model (network-bounded)
C_CTRL = "#1f78b4"     # control (global access)
LBL_MAIN = "main model (network-bounded)"
LBL_CTRL = "control (unrestricted access)"

plt.rcParams.update({
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
})


def load_sweep(config):
    path = os.path.join(MECHFIX, f"plots-config-{config}", "experiment_results.json")
    d = json.load(open(path))
    return d["alignment_sweep"], d["all_results"]


def series(all_results, name):
    """Steady-state mean and SE per alpha from the per-seed ss_runs."""
    means, ses = [], []
    for e in all_results:
        runs = np.asarray(e[f"{name}_ss_runs"], dtype=float)
        runs = runs[~np.isnan(runs)]
        means.append(runs.mean())
        ses.append(runs.std(ddof=1) / np.sqrt(len(runs)))
    return np.asarray(means), np.asarray(ses)


def panel_label(ax, letter):
    ax.text(-0.14, 1.06, letter, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top")


def sig_marks(ax, alphas, levels, y=None):
    """Mark alpha levels at which the paired main-minus-control contrast is
    Holm-significant (N=50 boundary table for alpha>=0.8, M4 grid below)."""
    if y is None:
        y = ax.get_ylim()[1]
    for a in levels:
        ax.annotate("*", (a, y), ha="center", va="bottom",
                    fontsize=10, color="0.25", annotation_clip=False)


# ----------------------------------------------------------------------
# Figure 2 - configuration comparison, per agent type
# ----------------------------------------------------------------------
def fig2():
    alphas, sw = load_sweep("switches")
    _, bl = load_sweep("baseline")

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    for cfg, res, color, lbl in [("main", sw, C_MAIN, LBL_MAIN),
                                 ("ctrl", bl, C_CTRL, LBL_CTRL)]:
        m, se = series(res, "seci_exploit")
        ax_a.errorbar(alphas, m, yerr=se, color=color, marker="o", ms=3,
                      lw=1.4, capsize=2, label=lbl)
        m, se = series(res, "seci_explor")
        ax_b.errorbar(alphas, m, yerr=se, color=color, marker="o", ms=3,
                      lw=1.4, capsize=2, label=lbl)
        for typ, ls in [("exploit", "--"), ("explor", "-")]:
            m, se = series(res, f"mae_{typ}")
            ax_c.errorbar(alphas, m, yerr=se, color=color, ls=ls, marker="o",
                          ms=2.5, lw=1.2, capsize=2,
                          label=f"{'confirmation' if typ=='exploit' else 'accuracy'}-seekers, "
                                f"{'main' if cfg=='main' else 'control'}")
        m, se = series(res, "unmet_needs")
        ax_d.errorbar(alphas, m, yerr=se, color=color, marker="o", ms=3,
                      lw=1.4, capsize=2, label=lbl)

    for ax in (ax_a, ax_b):
        ax.axhline(0, color="0.6", lw=0.7, ls=":")
        ax.set_ylabel("SECI (negative = echo chamber)")
        ax.set_ylim(-0.55, 0.12)
    ax_a.set_title("Confirmation-seeking communities")
    ax_b.set_title("Accuracy-seeking communities")
    ax_c.set_title("Belief error by agent type")
    ax_c.set_ylabel("Belief MAE (disaster cells)")
    ax_d.set_title("Unmet needs")
    ax_d.set_ylabel("Unmet high-need cells per tick")
    for ax in axes.flat:
        ax.set_xlabel(r"AI alignment $\alpha$")
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # Holm-significant paired contrasts (N=50 boundary at alpha>=0.8; M4 grid below)
    sig_marks(ax_a, alphas, [0.8, 0.9, 1.0], y=0.05)
    sig_marks(ax_b, alphas, [0.8, 0.9], y=0.05)
    sig_marks(ax_d, alphas, [0.8, 0.9, 1.0], y=9.7)

    ax_a.legend(loc="lower left", frameon=False)
    ax_c.legend(loc="center left", bbox_to_anchor=(0.0, 0.68), frameon=False,
                fontsize=6.5)
    for ax, letter in zip(axes.flat, "abcd"):
        panel_label(ax, letter)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_configuration.png"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig2_configuration.pdf"), bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 3 - the interior optimum and its robustness
# ----------------------------------------------------------------------
def norm01(x):
    x = np.asarray(x, dtype=float)
    return (x - x.min()) / (x.max() - x.min())


def composite(res, keys):
    """Range-normalised sum of |metric| means (within-sweep normalisation)."""
    total = None
    for k in keys:
        m, _ = series(res, k)
        n = norm01(np.abs(m))
        total = n if total is None else total + n
    return total


def gap_cell_optima():
    """alpha* per (g, d_mid) cell for the unmet-needs minimum and the
    population bubble composite, from results-gap-sweep-mechfix/."""
    cells = {}
    for d in sorted(glob.glob(os.path.join(ROOT, "results-gap-sweep-mechfix", "gap-cell-*"))):
        m = re.match(r".*gap-cell-g([\d.]+)-dm([\d.]+)-a([\d.]+)$", d)
        if not m:
            continue
        g, dm, a = float(m.group(1)), float(m.group(2)), float(m.group(3))
        jf = glob.glob(os.path.join(d, "*.json"))
        if not jf:
            continue
        r = json.load(open(jf[0]))["result"]
        cells.setdefault((g, dm), {})[a] = r
    optima = {}
    for key, per_alpha in cells.items():
        alphas = sorted(per_alpha)
        unmet = [np.nanmean(per_alpha[a]["unmet_needs_ss_runs"]) for a in alphas]
        seci_pop = np.abs([np.nanmean(per_alpha[a]["seci_pop_ss_runs"]) for a in alphas])
        aeci_pop = np.abs([np.nanmean(per_alpha[a]["aeci_ie_chan_pop_ss_runs"]) for a in alphas])
        bubble = norm01(seci_pop) + norm01(aeci_pop)
        optima[key] = (alphas[int(np.argmin(unmet))], alphas[int(np.argmin(bubble))])
    return optima


def fig3():
    alphas, sw = load_sweep("switches")
    _, bl = load_sweep("baseline")

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 3.0),
                                     gridspec_kw={"width_ratios": [1, 1.15]})

    pop_sw = composite(sw, ["seci_pop", "aeci_ie_chan_pop"])
    pop_bl = composite(bl, ["seci_pop", "aeci_ie_chan_pop"])
    # per-type channel composite from the archived summary table (combined
    # SECI + combined AECI-IE-chan, the "SECI + AECI-IE-chan" alpha* variant)
    st = np.genfromtxt(os.path.join(MECHFIX, "plots-config-switches",
                                    "summary_table.csv"),
                       delimiter=",", names=True)
    typ_sw = norm01(np.abs(st["seci"])) + norm01(np.abs(st["aeci_ie_chan"]))

    ax_a.plot(alphas, pop_sw, color=C_MAIN, marker="o", ms=3.5, lw=1.6,
              label="population composite, main")
    ax_a.plot(alphas, typ_sw, color=C_MAIN, marker="s", ms=3, lw=1.1, alpha=0.45,
              label="per-type composite, main")
    ax_a.plot(alphas, pop_bl, color=C_CTRL, marker="o", ms=3.5, lw=1.4,
              label="population composite, control")
    for curve, color, ls in [(pop_sw, C_MAIN, "-"), (pop_bl, C_CTRL, "-")]:
        astar = alphas[int(np.argmin(curve))]
        ax_a.axvline(astar, color=color, lw=0.9, ls=":", alpha=0.8)
    ax_a.set_xlabel(r"AI alignment $\alpha$")
    ax_a.set_ylabel("Bubble composite (range-normalised,\nwithin sweep; lower = fewer chambers)")
    ax_a.set_title("Interior optimum in both configurations")
    ax_a.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_a.legend(loc="upper center", frameon=False, fontsize=6.5)

    optima = gap_cell_optima()
    keys = sorted(optima)
    x = np.arange(len(keys))
    unmet_star = [optima[k][0] for k in keys]
    bubble_star = [optima[k][1] for k in keys]
    ax_b.axhspan(0.02, 0.98, color="0.94", zorder=0)
    ax_b.scatter(x - 0.12, unmet_star, marker="o", s=26, color="#33a02c",
                 label=r"$\alpha^{*}$ (unmet needs)", zorder=3)
    ax_b.scatter(x + 0.12, bubble_star, marker="D", s=22, color="#6a3d9a",
                 label=r"$\alpha^{*}$ (population bubble)", zorder=3)
    ax_b.set_ylim(-0.04, 1.04)
    ax_b.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([f"{g:g}/{dm:g}" for g, dm in keys], rotation=60,
                         fontsize=6.5)
    ax_b.set_xlabel(r"cognitive profile: gap $g$ / acceptance midpoint $d_{\mathrm{mid}}$")
    ax_b.set_ylabel(r"$\alpha^{*}$")
    ax_b.set_title("The optimum is interior in every cognitive profile")
    ax_b.axhline(0, color="0.5", lw=0.7)
    ax_b.axhline(1, color="0.5", lw=0.7)
    ax_b.legend(loc="lower right", frameon=False, fontsize=6.5)

    panel_label(ax_a, "a")
    panel_label(ax_b, "b")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_goldilocks.png"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig3_goldilocks.pdf"), bbox_inches="tight")
    plt.close(fig)
    return {"astar_pop_main": alphas[int(np.argmin(pop_sw))],
            "astar_pop_ctrl": alphas[int(np.argmin(pop_bl))],
            "astar_type_main": alphas[int(np.argmin(typ_sw))],
            "gap_unmet": sorted(set(unmet_star)),
            "gap_bubble": sorted(set(bubble_star))}


# ----------------------------------------------------------------------
# Figure 4 - spatial periphery
# ----------------------------------------------------------------------
def fig4():
    alphas, sw = load_sweep("switches")
    _, bl = load_sweep("baseline")
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for res, color, lbl in [(sw, C_MAIN, LBL_MAIN), (bl, C_CTRL, LBL_CTRL)]:
        m, se = series(res, "periph_sp_mae_gap")
        ax_a.errorbar(alphas, m, yerr=se, color=color, marker="o", ms=3,
                      lw=1.4, capsize=2, label=lbl)
        m, se = series(res, "periph_sp_aid_gap")
        ax_b.errorbar(alphas, m, yerr=se, color=color, marker="o", ms=3,
                      lw=1.4, capsize=2, label=lbl)
    ax_a.axhline(0, color="0.6", lw=0.7, ls=":")
    ax_b.axhline(0, color="0.6", lw=0.7, ls=":")
    ax_a.set_ylabel("Belief-error gap (far $-$ near quartile)")
    ax_a.set_title("Accuracy loss concentrates on the periphery")
    ax_b.set_ylabel("Aid-contribution gap (far $-$ near)")
    ax_b.set_title("The periphery withdraws from the response")
    for ax in (ax_a, ax_b):
        ax.set_xlabel(r"AI alignment $\alpha$")
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_a.legend(loc="upper left", frameon=False, fontsize=6.5)
    panel_label(ax_a, "a")
    panel_label(ax_b, "b")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_periphery.png"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig4_periphery.pdf"), bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# SI figures replotted from the archives
# ----------------------------------------------------------------------
def fig_s2():
    alphas, sw = load_sweep("switches")
    _, bl = load_sweep("baseline")
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for res, color, cfg in [(sw, C_MAIN, "main"), (bl, C_CTRL, "control")]:
        for typ, ls in [("exploit", "--"), ("explor", "-")]:
            m, se = series(res, f"l1pool_{typ}")
            ax_a.errorbar(alphas, m, yerr=se, color=color, ls=ls, marker="o",
                          ms=2.5, lw=1.2, capsize=2,
                          label=f"{'confirmation' if typ=='exploit' else 'accuracy'}-seekers, {cfg}")
            m, se = series(res, f"lockin_{typ}")
            ax_b.errorbar(alphas, m, yerr=se, color=color, ls=ls, marker="o",
                          ms=2.5, lw=1.2, capsize=2)
    ax_a.set_ylabel("L1+ beliefs per agent (pool size)")
    ax_a.set_title("Belief starvation: the informative pool collapses")
    ax_b.axhline(0, color="0.6", lw=0.7, ls=":")
    ax_b.set_ylabel("AECI-LockIn (negative = beliefs freeze)")
    ax_b.set_title("Individual lock-in of AI-heavy agents")
    for ax in (ax_a, ax_b):
        ax.set_xlabel(r"AI alignment $\alpha$")
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_a.legend(loc="center left", frameon=False, fontsize=6.5)
    panel_label(ax_a, "a")
    panel_label(ax_b, "b")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figS2_starvation.png"), bbox_inches="tight")
    plt.close(fig)


def fig_s3():
    alphas, sw = load_sweep("switches")
    _, bl = load_sweep("baseline")
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for res, color, cfg in [(sw, C_MAIN, "main"), (bl, C_CTRL, "control")]:
        for typ, ls in [("exploit", "--"), ("explor", "-")]:
            m, se = series(res, f"ai_query_ratio_{typ}")
            ax_a.errorbar(alphas, m, yerr=se, color=color, ls=ls, marker="o",
                          ms=2.5, lw=1.2, capsize=2,
                          label=f"{'confirmation' if typ=='exploit' else 'accuracy'}-seekers, {cfg}")
            m, se = series(res, f"trust_ai_{typ}")
            ax_b.errorbar(alphas, m, yerr=se, color=color, ls=ls, marker="o",
                          ms=2.5, lw=1.2, capsize=2)
    ax_a.set_ylabel("AI share of queries")
    ax_a.set_title("Capture: confirmation-seekers route queries to the AI")
    ax_b.set_ylabel("Trust in the AI source")
    ax_b.set_ylim(0.3, 1.0)
    ax_b.set_title("Trust never punishes the confirming AI")
    for ax in (ax_a, ax_b):
        ax.set_xlabel(r"AI alignment $\alpha$")
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_a.legend(loc="upper left", frameon=False, fontsize=6.5)
    panel_label(ax_a, "a")
    panel_label(ax_b, "b")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figS3_feedback.png"), bbox_inches="tight")
    plt.close(fig)


def fig_s7():
    """Robustness envelope (M5): three headline quantities across the four
    perturbation families."""
    sweeps = {
        "pop": ("Population size", ["100", "300", "500"]),
        "net": ("Network generator", ["smallworld"]),
        "ai": ("AI supply", ["1", "5", "10"]),
        "verif": ("Verification prob.", ["0.1", "0.3", "0.5"]),
    }
    rows = [("seci_exploit", "SECI confirmation-seekers"),
            ("mae_explor", "MAE accuracy-seekers"),
            ("unmet_needs", "Unmet needs")]
    fig, axes = plt.subplots(3, 4, figsize=(9.5, 6.5), sharex=True)
    cmap = plt.get_cmap("viridis")
    for j, (sweep, (title, levels)) in enumerate(sweeps.items()):
        for li, lev in enumerate(levels):
            dirs = sorted(glob.glob(os.path.join(
                ROOT, "results-robustness", f"robust-{sweep}-{lev}-*")))
            alphas, per_alpha = [], []
            for d in dirs:
                jf = glob.glob(os.path.join(d, "*.json"))
                if not jf:
                    continue
                dd = json.load(open(jf[0]))
                alphas.append(dd["alpha"])
                per_alpha.append(dd["result"])
            order = np.argsort(alphas)
            alphas = np.asarray(alphas)[order]
            per_alpha = [per_alpha[i] for i in order]
            color = cmap(0.15 + 0.7 * li / max(len(levels) - 1, 1))
            for i, (key, _) in enumerate(rows):
                m = [np.nanmean(r[f"{key}_ss_runs"]) for r in per_alpha]
                se = [np.nanstd(r[f"{key}_ss_runs"], ddof=1) /
                      np.sqrt(len(r[f"{key}_ss_runs"])) for r in per_alpha]
                axes[i, j].errorbar(alphas, m, yerr=se, color=color, marker="o",
                                    ms=3, lw=1.2, capsize=2, label=str(lev))
        axes[0, j].set_title(title)
        axes[-1, j].set_xlabel(r"$\alpha$")
        axes[0, j].legend(frameon=False, fontsize=6.5)
    for i, (_, lbl) in enumerate(rows):
        axes[i, 0].set_ylabel(lbl)
    for ax in axes[0]:
        ax.axhline(0, color="0.6", lw=0.7, ls=":")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figS7_robustness.png"), bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Copies of archived figures used verbatim in the SI
# ----------------------------------------------------------------------
def copy_archived():
    import shutil
    copies = {
        os.path.join(MECHFIX, "plots-config-switches", "echo_chamber_lifecycle.png"):
            "figS1_lifecycle.png",
        os.path.join(MECHFIX, "plots-config-switches", "periphery_gap_evolution.png"):
            "figS4_periphery_evolution.png",
        os.path.join(MECHFIX, "plots-config-switches", "population_evolution.png"):
            "figS5_population_evolution.png",
        os.path.join(ROOT, "results-docking", "dyadic-docking", "dyadic_results",
                     "dyadic_docking.png"):
            "figS6_dyadic_docking.png",
    }
    for src, dst in copies.items():
        shutil.copyfile(src, os.path.join(OUT, dst))


if __name__ == "__main__":
    fig2()
    info = fig3()
    fig4()
    fig_s2()
    fig_s3()
    fig_s7()
    copy_archived()
    print("Sanity checks (must match the archived alpha* tables):")
    print("  population composite alpha*: main =", info["astar_pop_main"],
          "(expect 0.6), control =", info["astar_pop_ctrl"], "(expect 0.9)")
    print("  per-type channel composite alpha*, main =", info["astar_type_main"])
    print("  gap-sweep alpha*(unmet) values:", info["gap_unmet"], "(expect within [0.6, 0.7])")
    print("  gap-sweep alpha*(pop bubble) values:", info["gap_bubble"], "(expect within [0.6, 0.8])")
    print("Figures written to", OUT)
