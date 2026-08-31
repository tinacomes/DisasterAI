# E1 docking fit report

Targets (Glickman & Sharot 2025, computed from published data; see `experiments/docking_fit/targets.json` for provenance):

- kappa_ai (transmission from AI partner): **0.748** (se 0.117)
- kappa_human (transmission from human partner): **0.309** (se 0.494)

| quantity | target | default params | fitted params |
|---|---|---|---|
| kappa_ai | 0.748 | 0.129 | 0.158 |
| kappa_human | 0.309 | 0.233 | 0.229 |
| retention alpha=1 (> retention human) | ordering | 1.000 vs 0.767 | 1.000 vs 0.771 |
| monotonicity rho(final bias, alpha) | ~1 | 0.841 | 0.854 |
| loss | -- | 28.51 | 25.85 |

Fitted parameters (defaults in brackets):

- `d_exploit` = **2.37** (default 2.0)
- `delta_exploit` = **2.15** (default 3.5)
- `d_explor` = **3.28** (default 4.0)
- `delta_explor` = **1.28** (default 1.2)
- `initial_trust` = **0.31** (default 0.3)
- `initial_ai_trust` = **0.40** (default 0.25)
- `rounds` = **40** (default 40)

## Identifiability

Coarse-pass top 10 (loss-ranked); parameter ranges within this set indicate ridge directions -- the fit constrains combinations, not every coordinate:

|   d_exploit |   delta_exploit |   d_explor |   delta_explor |   initial_trust |   initial_ai_trust |   rounds |   loss |
|------------:|----------------:|-----------:|---------------:|----------------:|-------------------:|---------:|-------:|
|       1.179 |           3.804 |      4.915 |          2.085 |           0.537 |              0.318 |       50 | 25.362 |
|       2.375 |           2.154 |      3.28  |          1.278 |           0.307 |              0.403 |       40 | 25.387 |
|       2     |           3.5   |      4     |          1.2   |           0.3   |              0.25  |       40 | 27.955 |

Trust learning rates are not identified by the dyad (trust is held fixed there by design); the fitted trust-side quantities are the initial trust levels.

## Refined comparison (all refined points)

| point_id   |   d_exploit |   delta_exploit |   d_explor |   delta_explor |   initial_trust |   initial_ai_trust |   rounds |   kappa_ai |   kappa_hh |   retention_a1 |   retention_hh |   mono_rho |   loss |
|:-----------|------------:|----------------:|-----------:|---------------:|----------------:|-------------------:|---------:|-----------:|-----------:|---------------:|---------------:|-----------:|-------:|
| lhs000     |       2.375 |           2.154 |      3.28  |          1.278 |           0.307 |              0.403 |       40 |      0.158 |      0.229 |              1 |          0.771 |      0.854 | 25.852 |
| lhs001     |       1.179 |           3.804 |      4.915 |          2.085 |           0.537 |              0.318 |       50 |      0.158 |      0.329 |              1 |          0.671 |      0.841 | 25.918 |
| default    |       2     |           3.5   |      4     |          1.2   |           0.3   |              0.25  |       40 |      0.129 |      0.233 |              1 |          0.767 |      0.841 | 28.511 |
