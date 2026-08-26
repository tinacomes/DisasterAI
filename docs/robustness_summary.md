# Robustness envelope (M5) — summary

One paragraph per sweep, each stating whether (i) the **structural
precondition** (network-bounded access preserves the confirmation-seekers'
community echo chamber that unrestricted access dissolves), (ii) the
**interior optimum** (α\* interior under the population composite and the
unmet-needs minimum), and (iii) the **starvation / capture mechanisms**
(explorer L1+ pool collapse and MAE growth with α; exploiter query-share
capture) hold under the perturbation.

All sweeps: revised model (`30f89e0` mechanisms:
`confirmation_reference=network`, `report_rounding=stochastic`,
`confirmation_target=individual`), main-model configuration (mobility=1,
spatial network, network query scope), N=20 seeded replications
(replicate *i* ← seed *i*), 200 ticks, α ∈ {0.0, 0.5, 0.7, 0.9, 1.0}.

> **Status: COMPLETE.** Run
> [32894002318](https://github.com/tinacomes/DisasterAI/actions/runs/32894002318)
> (2026-08-25, code `2836a18`), all 50 cells succeeded; per-cell JSONs and
> `robustness_tables.md` archived in `results-robustness/`. The M5 cells
> run the MAIN model only, so criterion (i) is verified here in its
> *persistence* half (the exploiter chamber does not dissolve under
> bounded access at any α); the *dissolution-under-global-access* half is
> the mechfix control (`results-mechfix/plots-config-baseline/`) and the
> N=50 boundary deltas (`results-boundary-n50/`).

**Overall verdict: PASS on all three criteria at every one of the ten
perturbation levels — with striking quantitative stability.** Across
population 100→500, the alternative generator, AI supply 1→10, and
verification probability 0.1→0.5: α\*(unmet) ∈ {0.5, 0.7} and
α\*(population bubble) ∈ {0.5, 0.7} everywhere (interior in every cell);
exploiter SECI −0.36…−0.43 at α=0 and still −0.20…−0.28 at α=1
(persistence); explorer SECI ≈0 → −0.31…−0.35; explorer MAE
≈0.55 → ≈1.75; explorer L1+ pool ≈110–124 → ≈27–30; exploiter AI query
share 0.53–0.56 → 0.68–0.71. The headline quantities of the canonical run
reproduce almost unchanged under every perturbation.

## Population size (100 / 300 / 500)

Community scaling: the spatial networks keep `n_communities_per_type=4`,
so communities grow from ~12 to ~62 members.

**(i) PASS** — SECI_exploit −0.41/−0.36/−0.38 (α=0) staying deep at α=1
(−0.28/−0.24/−0.20) for N=100/300/500. **(ii) PASS** — α\*(unmet)
0.7/0.5/0.5, α\*(pop-bubble) 0.7 at all three sizes. **(iii) PASS** —
explorer MAE 0.55→1.74–1.76, pool 110–115→27–28, exploiter share
0.53–0.56→0.68 at every size. No size dependence beyond seed noise; the
N=100 mainline is not a small-population artifact.

## Alternative within-community generator (`spatial_smallworld`)

Watts–Strogatz within-community wiring (ring lattice, k=4, rewire 0.1);
bridges and spatial embedding identical to `spatial_bridged`.

**(i) PASS** — SECI_exploit −0.41 → −0.24. **(ii) PASS** — α\*(unmet)
0.7, α\*(pop-bubble) 0.5. **(iii) PASS** — explorer MAE 0.53→1.78, pool
112→27, exploiter share 0.56→0.71. The findings do not depend on the
within-community wiring rule.

## AI information supply (`num_ai` 1 / 5 / 10)

**(i) PASS** — SECI_exploit −0.36…−0.43 (α=0) → −0.23…−0.26 (α=1) at all
three supply levels. **(ii) PASS** — α\*(unmet) 0.7/0.7/0.5,
α\*(pop-bubble) 0.7 throughout. **(iii) PASS** — explorer MAE
0.53–0.68→1.73–1.76 (the single-AI cell starts slightly worse at α=0:
0.68, thinner coverage), pool 113–124→28–30, exploiter share
0.53–0.54→0.69–0.70. A monopoly AI and a ten-provider ecosystem produce
the same dynamics: what matters is the alignment policy, not the number
of providers serving it.

## Verification probability (0.1 / 0.3 / 0.5)

**(i) PASS** — SECI_exploit −0.38…−0.41 → −0.24…−0.25. **(ii) PASS** —
α\*(unmet) 0.7 and α\*(pop-bubble) 0.7 at all three levels. **(iii)
PASS** — explorer MAE 0.54–0.55→1.73–1.75, pool 113–114→27–30, exploiter
share 0.53–0.55→0.69–0.70. Making external verification scarcer (0.1) or
more available (0.5) than the mainline 0.3 leaves every headline
quantity within seed noise — consistent with C12: the verification
channel's base-rate structure, not its throughput, is what limits
accuracy-seeking.
