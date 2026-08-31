# E1 docking fit report

Targets (Glickman & Sharot 2025, computed from published data; see `experiments/docking_fit/targets.json` for provenance):

- kappa_ai (transmission from AI partner): **0.748** (se 0.117)
- kappa_human (transmission from human partner): **0.309** (se 0.494)

| quantity | target | default params | fitted params |
|---|---|---|---|
| kappa_ai | 0.748 | 0.162 | 0.268 |
| kappa_human | 0.309 | 0.265 | 0.282 |
| retention alpha=1 (> retention human) | ordering | 1.000 vs 0.735 | 1.000 vs 0.718 |
| monotonicity rho(final bias, alpha) | ~1 | 0.935 | 0.947 |
| loss | -- | 25.16 | 16.86 |

Fitted parameters (defaults in brackets):

- `d_exploit` = **2.70** (default 2.0)
- `delta_exploit` = **4.20** (default 3.5)
- `d_explor` = **5.37** (default 4.0)
- `delta_explor` = **1.87** (default 1.2)
- `initial_trust` = **0.26** (default 0.3)
- `initial_ai_trust` = **0.48** (default 0.25)
- `rounds` = **60** (default 40)

## Per-type transmission

| quantity | G&S (population) | default | fitted |
|---|---|---|---|
| kappa_ai, exploitative | 0.748 | 0.046 | 0.047 |
| kappa_ai, exploratory | 0.748 | 0.277 | 0.489 |
| kappa_hh, exploitative | 0.309 | 0.274 | 0.283 |
| kappa_hh, exploratory | 0.309 | 0.255 | 0.280 |

## Interpretation (for the SI text)

1. **Human-human transmission is matched.** Both cognitive types transmit ~0.28 of the partner-self gap; the measured value is 0.309 (wide CI) -- inside it at default and fitted parameters alike.
2. **AI transmission is identified but ceilinged.** The single clearly identified direction is the initial AI trust level: every top set roughly doubles the default (0.44-0.59 vs 0.25). At the fitted point the accuracy-seeker reaches kappa_ai = 0.49, at the edge of the measured CI [0.52, 0.98]; the confirmation-seeker stays near zero (0.05) at every parameter set in the box -- its D/delta acceptance window rejects strongly disconfirming reports by construction (the same mechanism behind C12). The population mean therefore under-transmits AI influence relative to Glickman & Sharot (0.16 default, 0.27 fitted, vs 0.748 measured).
3. **The mismatch is conservative.** The model's humans adopt AI judgments more reluctantly than measured participants, so the population-scale harms are not driven by an over-credulous human model; if anything the model understates AI influence.
4. **All orderings reproduce**: the confirming AI (alpha=1) retains the implanted bias fully while the human partner erodes it, and final bias is monotone in alpha (Spearman ~0.95).

## Identifiability

Coarse-pass top 10 (loss-ranked); parameter ranges within this set indicate ridge directions -- the fit constrains combinations, not every coordinate:

|   d_exploit |   delta_exploit |   d_explor |   delta_explor |   initial_trust |   initial_ai_trust |   rounds |   loss |
|------------:|----------------:|-----------:|---------------:|----------------:|-------------------:|---------:|-------:|
|       1.676 |           3.911 |      4.821 |          1.926 |           0.592 |              0.592 |       50 | 16.657 |
|       2.474 |           3.533 |      5.306 |          1.567 |           0.523 |              0.583 |       40 | 16.81  |
|       2.16  |           4.453 |      5.104 |          1.814 |           0.54  |              0.535 |       50 | 17.779 |
|       2.699 |           4.203 |      5.371 |          1.871 |           0.263 |              0.48  |       60 | 17.79  |
|       1.944 |           3.951 |      4.291 |          1.727 |           0.505 |              0.589 |       40 | 17.978 |
|       2.838 |           2.777 |      4.595 |          1.933 |           0.304 |              0.438 |       60 | 18.551 |
|       2.78  |           2.1   |      4.998 |          1.77  |           0.597 |              0.555 |       30 | 18.685 |
|       1.874 |           2.692 |      4.679 |          1.65  |           0.391 |              0.47  |       50 | 18.783 |
|       2.745 |           2.525 |      4.357 |          1.362 |           0.195 |              0.489 |       50 | 18.945 |
|       2.215 |           4.082 |      3.701 |          0.9   |           0.578 |              0.58  |       40 | 19.114 |

Trust learning rates are not identified by the dyad (trust is held fixed there by design); the fitted trust-side quantities are the initial trust levels.

## Refined comparison (all refined points)

| point_id   |   d_exploit |   delta_exploit |   d_explor |   delta_explor |   initial_trust |   initial_ai_trust |   rounds |   kappa_ai |   kappa_hh |   retention_a1 |   retention_hh |   mono_rho |   loss |
|:-----------|------------:|----------------:|-----------:|---------------:|----------------:|-------------------:|---------:|-----------:|-----------:|---------------:|---------------:|-----------:|-------:|
| lhs021     |       2.699 |           4.203 |      5.371 |          1.871 |           0.263 |              0.48  |       60 |      0.268 |      0.282 |              1 |          0.718 |      0.947 | 16.859 |
| lhs019     |       2.474 |           3.533 |      5.306 |          1.567 |           0.523 |              0.583 |       40 |      0.265 |      0.377 |              1 |          0.623 |      0.947 | 17.08  |
| lhs090     |       1.676 |           3.911 |      4.821 |          1.926 |           0.592 |              0.592 |       50 |      0.265 |      0.373 |              1 |          0.627 |      0.947 | 17.107 |
| lhs071     |       2.16  |           4.453 |      5.104 |          1.814 |           0.54  |              0.535 |       50 |      0.26  |      0.381 |              1 |          0.619 |      0.947 | 17.407 |
| lhs031     |       2.838 |           2.777 |      4.595 |          1.933 |           0.304 |              0.438 |       60 |      0.258 |      0.289 |              1 |          0.711 |      0.947 | 17.565 |
| lhs058     |       2.78  |           2.1   |      4.998 |          1.77  |           0.597 |              0.555 |       30 |      0.255 |      0.38  |              1 |          0.62  |      0.947 | 17.764 |
| lhs068     |       1.944 |           3.951 |      4.291 |          1.727 |           0.505 |              0.589 |       40 |      0.253 |      0.362 |              1 |          0.638 |      0.947 | 17.904 |
| lhs044     |       1.874 |           2.692 |      4.679 |          1.65  |           0.391 |              0.47  |       50 |      0.251 |      0.298 |              1 |          0.702 |      0.947 | 18.074 |
| default    |       2     |           3.5   |      4     |          1.2   |           0.3   |              0.25  |       40 |      0.162 |      0.265 |              1 |          0.735 |      0.935 | 25.16  |
