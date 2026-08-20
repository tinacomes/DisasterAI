# Robustness envelope (M5) — summary

One paragraph per sweep, each stating whether (i) the **structural
precondition** (network-bounded access preserves the confirmation-seekers'
community echo chamber that unrestricted access dissolves), (ii) the
**interior optimum** (α\* interior under the M1 composite |SECI| + |AECI-IE|),
and (iii) the **starvation / capture mechanisms** (explorer L1+ pool
collapse and MAE growth with α; exploiter query-share capture and lock-in)
hold under the perturbation.

All sweeps: fixed model (`confirmation_target=individual`), main-model
configuration (mobility=1, spatial network, network query scope), N=20
seeded replications (replicate *i* ← seed *i*), 200 ticks,
α ∈ {0.0, 0.5, 0.7, 0.9, 1.0}. Dispatched via
`.github/workflows/run-robustness-sweeps.yml`; the per-condition tables are
produced by `tools/robustness_summary.py` (artifact `robustness-tables`).

> **Status: PENDING RUNS.** The paragraphs below are templates to be filled
> from the `robustness_tables.md` artifact of the dispatched run — do not
> cite until the run id is recorded here.

## Population size (100 / 300 / 500)

Community scaling: the spatial networks keep `n_communities_per_type=4`, so
communities grow from ~12 to ~62 members; the `components` control network
instead scales the community count (~N_type/25).

*TODO after run: verdict on (i)–(iii).*

## Alternative within-community generator (`spatial_smallworld`)

Watts–Strogatz within-community wiring (ring lattice, k=4, rewire 0.1)
behind `--network-type spatial_smallworld`; bridges and spatial embedding
identical to `spatial_bridged`. Compare against the `results-final`
main-model tables.

*TODO after run: verdict on (i)–(iii).*

## AI information supply (`num_ai` 1 / 5 / 10)

*TODO after run: verdict on (i)–(iii).*

## Verification probability (0.1 / 0.3 / 0.5)

Recovery ticks (`SECIbreak` transition scalars) are reported alongside in
the tables.

*TODO after run: verdict on (i)–(iii).*
