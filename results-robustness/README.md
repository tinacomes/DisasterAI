# M5 robustness envelope — PASS at every perturbation level

Four sweeps in the main-model configuration, revised model (`2836a18` =
`30f89e0` mechanisms), N=20 seed-paired replications, 200 ticks,
α ∈ {0.0, 0.5, 0.7, 0.9, 1.0}: population {100, 300, 500}; alternative
within-community generator (`spatial_smallworld`); AI supply {1, 5, 10};
verification probability {0.1, 0.3, 0.5}.

- **Run**: [32894002318](https://github.com/tinacomes/DisasterAI/actions/runs/32894002318),
  2026-08-25/26, workflow *Run Robustness Sweeps (M5)*. All 50 cells
  succeeded.
- Per-cell JSONs in `robust-{sweep}-{level}-{α}/`; late-run tables in
  `robustness-tables/robustness_tables.md`.

**Verdict: all three criteria hold at all ten levels** — interior optimum
(α\* ∈ {0.5, 0.7} for both the unmet-needs minimum and the population
bubble composite, every cell), chamber persistence under bounded access
(SECI_exploit −0.36…−0.43 at α=0, still −0.20…−0.28 at α=1), and
starvation/capture (explorer MAE ≈0.55→1.75, L1+ pool ≈110–124→27–30,
exploiter AI share ≈0.54→0.69) — with the canonical run's numbers
reproducing almost unchanged under every perturbation. Full per-sweep
paragraphs: `docs/robustness_summary.md`. Notable single observations:
a monopoly AI (num_ai=1) behaves like a ten-provider ecosystem, and
verification throughput (0.1–0.5) is immaterial — consistent with C12's
base-rate mechanism.
