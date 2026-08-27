# Echo-chamber lifecycle metrics (PROTOTYPE)

Computed from the cross-replication **mean** trajectories in the archived `experiment_results.json` files; thresholds follow `test_filter_bubbles.py` (formation SECI < -0.1, dissolution sustained > -0.05 for 5 samples). Chambers can be multi-episode (form, dissolve mid-run, re-form), so each row lists every episode; an episode ending at the run horizon with a `+` was still open when the run ended (right-censored), and `dissolved? = no` means the LAST episode never closed. **No CIs: per-seed trajectories are not archived** — publication statistics need the re-run with per-seed lifecycle columns.

## Chamber lifecycle — confirmation-seeking (exploitative) communities

| config | alpha | formation | peak SECI | episodes (start-end, + = censored) | final dissolution | dissolved? | standing at end? | persistence |
|---|---|---|---|---|---|---|---|---|
| main | 0.0 | 5 | -0.54 | 5-55; 100-200+ | 200 | no | yes | 0.68 |
| main | 0.1 | 5 | -0.51 | 5-50; 100-200+ | 200 | no | yes | 0.68 |
| main | 0.2 | 5 | -0.50 | 5-45; 100-200+ | 200 | no | yes | 0.68 |
| main | 0.3 | 5 | -0.50 | 5-40; 105-200+ | 200 | no | yes | 0.65 |
| main | 0.4 | 5 | -0.50 | 5-40; 105-200+ | 200 | no | yes | 0.65 |
| main | 0.5 | 5 | -0.47 | 5-40; 110-200+ | 200 | no | yes | 0.62 |
| main | 0.6 | 5 | -0.49 | 5-40; 115-200+ | 200 | no | yes | 0.60 |
| main | 0.7 | 5 | -0.47 | 5-50; 115-200+ | 200 | no | yes | 0.60 |
| main | 0.8 | 5 | -0.43 | 5-60; 120-200+ | 200 | no | yes | 0.60 |
| main | 0.9 | 5 | -0.36 | 5-75; 115-200+ | 200 | no | yes | 0.57 |
| main | 1.0 | 15 | -0.47 | 15-200+ | 200 | no | yes | 0.72 |
| control | 0.0 | 5 | -0.63 | 5-200+ | 200 | no | yes | 0.85 |
| control | 0.1 | 5 | -0.61 | 5-200+ | 200 | no | yes | 0.82 |
| control | 0.2 | 5 | -0.60 | 5-75; 110-200+ | 200 | no | yes | 0.75 |
| control | 0.3 | 5 | -0.60 | 5-70; 110-200+ | 200 | no | yes | 0.72 |
| control | 0.4 | 5 | -0.59 | 5-70; 115-200+ | 200 | no | yes | 0.70 |
| control | 0.5 | 5 | -0.59 | 5-65; 120-200+ | 200 | no | yes | 0.68 |
| control | 0.6 | 5 | -0.52 | 5-65; 120-200+ | 200 | no | yes | 0.68 |
| control | 0.7 | 5 | -0.41 | 5-75; 130-200+ | 200 | no | yes | 0.65 |
| control | 0.8 | 5 | -0.35 | 5-75; 135-200+ | 200 | no | yes | 0.57 |
| control | 0.9 | 5 | -0.34 | 5-80; 140-170 | 170 | yes | no | 0.47 |
| control | 1.0 | 10 | -0.38 | 10-95 | 95 | yes | no | 0.40 |

## Chamber lifecycle — accuracy-seeking (exploratory) communities

| config | alpha | formation | peak SECI | episodes (start-end, + = censored) | final dissolution | dissolved? | standing at end? | persistence |
|---|---|---|---|---|---|---|---|---|
| main | 0.0 | 20 | -0.36 | 20-100 | 100 | yes | no | 0.38 |
| main | 0.1 | 20 | -0.37 | 20-105 | 105 | yes | no | 0.38 |
| main | 0.2 | 20 | -0.41 | 20-105 | 105 | yes | no | 0.40 |
| main | 0.3 | 20 | -0.43 | 20-110 | 110 | yes | no | 0.40 |
| main | 0.4 | 20 | -0.48 | 20-120 | 120 | yes | no | 0.42 |
| main | 0.5 | 15 | -0.50 | 15-125 | 125 | yes | no | 0.47 |
| main | 0.6 | 20 | -0.52 | 20-200+ | 200 | no | no | 0.50 |
| main | 0.7 | 20 | -0.49 | 20-200+ | 200 | no | no | 0.55 |
| main | 0.8 | 25 | -0.45 | 25-200+ | 200 | no | yes | 0.88 |
| main | 0.9 | 45 | -0.37 | 45-200+ | 200 | no | yes | 0.78 |
| main | 1.0 | 60 | -0.35 | 60-200+ | 200 | no | yes | 0.70 |
| control | 0.0 | 20 | -0.40 | 20-100 | 100 | yes | no | 0.38 |
| control | 0.1 | 20 | -0.43 | 20-105 | 105 | yes | no | 0.38 |
| control | 0.2 | 20 | -0.47 | 20-105 | 105 | yes | no | 0.40 |
| control | 0.3 | 20 | -0.51 | 20-105 | 105 | yes | no | 0.40 |
| control | 0.4 | 20 | -0.55 | 20-110 | 110 | yes | no | 0.42 |
| control | 0.5 | 20 | -0.58 | 20-115 | 115 | yes | no | 0.45 |
| control | 0.6 | 20 | -0.61 | 20-120 | 120 | yes | no | 0.47 |
| control | 0.7 | 25 | -0.64 | 25-125 | 125 | yes | no | 0.50 |
| control | 0.8 | 35 | -0.62 | 35-200+ | 200 | no | no | 0.47 |
| control | 0.9 | 45 | -0.57 | 45-200+ | 200 | no | yes | 0.78 |
| control | 1.0 | 60 | -0.47 | 60-200+ | 200 | no | yes | 0.70 |

## Chamber lifecycle — population (societal) index

| config | alpha | formation | peak SECI | episodes (start-end, + = censored) | final dissolution | dissolved? | standing at end? | persistence |
|---|---|---|---|---|---|---|---|---|
| main | 0.0 | 5 | -0.27 | 5-200+ | 200 | no | yes | 0.80 |
| main | 0.1 | 5 | -0.26 | 5-200+ | 200 | no | yes | 0.72 |
| main | 0.2 | 5 | -0.26 | 5-200+ | 200 | no | yes | 0.75 |
| main | 0.3 | 5 | -0.28 | 5-200+ | 200 | no | yes | 0.70 |
| main | 0.4 | 5 | -0.28 | 5-80; 115-200+ | 200 | no | yes | 0.68 |
| main | 0.5 | 5 | -0.28 | 5-80; 115-200+ | 200 | no | yes | 0.68 |
| main | 0.6 | 10 | -0.27 | 10-80; 115-200+ | 200 | no | yes | 0.65 |
| main | 0.7 | 10 | -0.27 | 10-85; 115-200+ | 200 | no | yes | 0.72 |
| main | 0.8 | 15 | -0.28 | 15-200+ | 200 | no | yes | 0.70 |
| main | 0.9 | 30 | -0.28 | 30-200+ | 200 | no | yes | 0.70 |
| main | 1.0 | 45 | -0.39 | 45-200+ | 200 | no | yes | 0.78 |
| control | 0.0 | 10 | -0.33 | 10-200+ | 200 | no | yes | 0.78 |
| control | 0.1 | 10 | -0.34 | 10-200+ | 200 | no | yes | 0.75 |
| control | 0.2 | 10 | -0.36 | 10-200+ | 200 | no | yes | 0.78 |
| control | 0.3 | 10 | -0.37 | 10-200+ | 200 | no | yes | 0.75 |
| control | 0.4 | 10 | -0.38 | 10-90; 120-200+ | 200 | no | yes | 0.68 |
| control | 0.5 | 10 | -0.39 | 10-90; 120-200+ | 200 | no | yes | 0.72 |
| control | 0.6 | 10 | -0.39 | 10-90; 125-200+ | 200 | no | yes | 0.68 |
| control | 0.7 | 15 | -0.37 | 15-95; 130-200+ | 200 | no | yes | 0.60 |
| control | 0.8 | 25 | -0.34 | 25-100; 140-200+ | 200 | no | no | 0.47 |
| control | 0.9 | 30 | -0.32 | 30-100 | 100 | yes | no | 0.35 |
| control | 1.0 | 40 | -0.28 | 40-105 | 105 | yes | no | 0.33 |

## Capture onset (first sustained AI-majority tick) and final-window slopes (per 100 ticks; non-zero = not equilibrated at horizon)

| config | alpha | capture onset (exploit) | capture onset (explor) | MAE-gap slope | aid-gap slope | unmet slope |
|---|---|---|---|---|---|---|
| main | 0.0 | 57 | 0 | -0.02 | -0.24 | -0.45 |
| main | 0.1 | 57 | 0 | -0.02 | -0.05 | -2.09 |
| main | 0.2 | 54 | 0 | -0.08 | 0.37 | -0.42 |
| main | 0.3 | 51 | 0 | -0.06 | 0.77 | -0.22 |
| main | 0.4 | 48 | 0 | -0.10 | 2.46 | 0.27 |
| main | 0.5 | 45 | 0 | -0.13 | 2.14 | -0.36 |
| main | 0.6 | 35 | 0 | -0.04 | 2.76 | 0.17 |
| main | 0.7 | 13 | 0 | -0.06 | 3.48 | -0.07 |
| main | 0.8 | 3 | 0 | -0.07 | 2.31 | -0.67 |
| main | 0.9 | 0 | 0 | 0.04 | 1.22 | -1.90 |
| main | 1.0 | 1 | 0 | 0.12 | -1.85 | -4.57 |
| control | 0.0 | 74 | 0 | 0.01 | 0.18 | -1.06 |
| control | 0.1 | 65 | 0 | -0.01 | 0.36 | 0.05 |
| control | 0.2 | 60 | 0 | -0.01 | 0.52 | 0.52 |
| control | 0.3 | 6 | 0 | -0.07 | 1.14 | 1.07 |
| control | 0.4 | 47 | 0 | -0.02 | 1.21 | -0.02 |
| control | 0.5 | 33 | 0 | -0.03 | -0.88 | -0.04 |
| control | 0.6 | 21 | 0 | -0.04 | 0.65 | -0.43 |
| control | 0.7 | 16 | 0 | -0.04 | -0.35 | -0.25 |
| control | 0.8 | 17 | 0 | -0.02 | 0.03 | -2.52 |
| control | 0.9 | 8 | 0 | -0.01 | 0.67 | -7.23 |
| control | 1.0 | 1 | 0 | -0.02 | 0.27 | -17.20 |

## Mechanism cascade — half-crossing tick per mechanism (-- = no meaningful net transition at that alpha)

| config | alpha | AI capture (exploit query share) | Belief starvation (explor L1+ pool) | Explorer chamber (SECI) | Precision decline (explor) | Periphery aid gap (spatial) |
|---|---|---|---|---|---|---|
| main | 0.0 | -- | 65 | -- | 45 | 45 |
| main | 0.1 | -- | 60 | -- | 50 | 45 |
| main | 0.2 | -- | 60 | -- | 50 | 40 |
| main | 0.3 | -- | 60 | -- | 55 | 45 |
| main | 0.4 | -- | 65 | -- | 60 | 45 |
| main | 0.5 | -- | 65 | -- | 65 | 50 |
| main | 0.6 | -- | 70 | -- | 75 | 60 |
| main | 0.7 | 22 | 75 | -- | 85 | 70 |
| main | 0.8 | 5 | 85 | 25 | 105 | 90 |
| main | 0.9 | 5 | 100 | 45 | 115 | 100 |
| main | 1.0 | 4 | 80 | 70 | 120 | 110 |
| control | 0.0 | -- | 70 | -- | 55 | 135 |
| control | 0.1 | -- | 70 | -- | 55 | -- |
| control | 0.2 | -- | 70 | -- | 60 | -- |
| control | 0.3 | -- | 70 | -- | 60 | -- |
| control | 0.4 | -- | 75 | -- | 65 | -- |
| control | 0.5 | 35 | 75 | -- | 75 | 130 |
| control | 0.6 | 25 | 85 | -- | 85 | 105 |
| control | 0.7 | 22 | 90 | -- | 105 | -- |
| control | 0.8 | 19 | 100 | -- | 120 | -- |
| control | 0.9 | -- | 115 | -- | 130 | -- |
| control | 1.0 | 18 | 90 | 65 | 140 | -- |

