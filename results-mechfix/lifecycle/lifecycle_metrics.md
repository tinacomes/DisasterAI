# Echo-chamber lifecycle metrics (PROTOTYPE)

Computed from the cross-replication **mean** trajectories in the archived `experiment_results.json` files; thresholds follow `test_filter_bubbles.py` (formation SECI < -0.1, dissolution sustained > -0.05 for 5 samples). Chambers can be multi-episode (form, dissolve mid-run, re-form), so each row lists every episode; an episode ending at the run horizon with a `+` was still open when the run ended (right-censored), and `dissolved? = no` means the LAST episode never closed. **No CIs: per-seed trajectories are not archived** — publication statistics need the re-run with per-seed lifecycle columns.

## Chamber lifecycle — confirmation-seeking (exploitative) communities

| config | alpha | formation | peak SECI | episodes (start-end, + = censored) | final dissolution | dissolved? | standing at end? | persistence |
|---|---|---|---|---|---|---|---|---|
| main | 0.0 | 5 | -0.52 | 5-55; 100-200+ | 200 | no | yes | 0.68 |
| main | 0.1 | 5 | -0.51 | 5-50; 100-200+ | 200 | no | yes | 0.68 |
| main | 0.2 | 5 | -0.52 | 5-45; 100-200+ | 200 | no | yes | 0.68 |
| main | 0.3 | 5 | -0.49 | 5-40; 105-200+ | 200 | no | yes | 0.65 |
| main | 0.4 | 5 | -0.47 | 5-40; 110-200+ | 200 | no | yes | 0.62 |
| main | 0.5 | 5 | -0.50 | 5-40; 110-200+ | 200 | no | yes | 0.62 |
| main | 0.6 | 5 | -0.47 | 5-40; 115-200+ | 200 | no | yes | 0.60 |
| main | 0.7 | 5 | -0.51 | 5-50; 120-200+ | 200 | no | yes | 0.57 |
| main | 0.8 | 5 | -0.46 | 5-60; 120-200+ | 200 | no | yes | 0.57 |
| main | 0.9 | 5 | -0.32 | 5-75; 115-200+ | 200 | no | yes | 0.60 |
| main | 1.0 | 15 | -0.40 | 15-200+ | 200 | no | yes | 0.70 |
| control | 0.0 | 5 | -0.64 | 5-200+ | 200 | no | yes | 0.85 |
| control | 0.1 | 5 | -0.59 | 5-200+ | 200 | no | yes | 0.80 |
| control | 0.2 | 5 | -0.58 | 5-75; 110-200+ | 200 | no | yes | 0.75 |
| control | 0.3 | 5 | -0.59 | 5-70; 110-200+ | 200 | no | yes | 0.72 |
| control | 0.4 | 5 | -0.59 | 5-70; 115-200+ | 200 | no | yes | 0.70 |
| control | 0.5 | 5 | -0.55 | 5-65; 120-200+ | 200 | no | yes | 0.68 |
| control | 0.6 | 5 | -0.54 | 5-65; 120-200+ | 200 | no | yes | 0.68 |
| control | 0.7 | 5 | -0.41 | 5-70; 130-200+ | 200 | no | yes | 0.65 |
| control | 0.8 | 5 | -0.35 | 5-80; 135-200+ | 200 | no | yes | 0.55 |
| control | 0.9 | 5 | -0.34 | 5-85; 145-175 | 175 | yes | no | 0.45 |
| control | 1.0 | 10 | -0.38 | 10-95 | 95 | yes | no | 0.40 |

## Chamber lifecycle — accuracy-seeking (exploratory) communities

| config | alpha | formation | peak SECI | episodes (start-end, + = censored) | final dissolution | dissolved? | standing at end? | persistence |
|---|---|---|---|---|---|---|---|---|
| main | 0.0 | 20 | -0.35 | 20-100 | 100 | yes | no | 0.35 |
| main | 0.1 | 20 | -0.37 | 20-100 | 100 | yes | no | 0.38 |
| main | 0.2 | 20 | -0.41 | 20-105 | 105 | yes | no | 0.38 |
| main | 0.3 | 20 | -0.43 | 20-110 | 110 | yes | no | 0.40 |
| main | 0.4 | 20 | -0.48 | 20-120 | 120 | yes | no | 0.42 |
| main | 0.5 | 15 | -0.51 | 15-120 | 120 | yes | no | 0.47 |
| main | 0.6 | 20 | -0.51 | 20-200+ | 200 | no | no | 0.47 |
| main | 0.7 | 20 | -0.49 | 20-200+ | 200 | no | no | 0.55 |
| main | 0.8 | 25 | -0.46 | 25-200+ | 200 | no | yes | 0.88 |
| main | 0.9 | 45 | -0.37 | 45-200+ | 200 | no | yes | 0.78 |
| main | 1.0 | 60 | -0.35 | 60-200+ | 200 | no | yes | 0.70 |
| control | 0.0 | 20 | -0.40 | 20-100 | 100 | yes | no | 0.38 |
| control | 0.1 | 20 | -0.43 | 20-105 | 105 | yes | no | 0.40 |
| control | 0.2 | 20 | -0.47 | 20-105 | 105 | yes | no | 0.40 |
| control | 0.3 | 20 | -0.52 | 20-105 | 105 | yes | no | 0.40 |
| control | 0.4 | 20 | -0.56 | 20-115 | 115 | yes | no | 0.42 |
| control | 0.5 | 20 | -0.58 | 20-115 | 115 | yes | no | 0.45 |
| control | 0.6 | 20 | -0.61 | 20-120 | 120 | yes | no | 0.47 |
| control | 0.7 | 25 | -0.63 | 25-125 | 125 | yes | no | 0.50 |
| control | 0.8 | 35 | -0.62 | 35-200+ | 200 | no | no | 0.50 |
| control | 0.9 | 45 | -0.57 | 45-200+ | 200 | no | yes | 0.78 |
| control | 1.0 | 60 | -0.47 | 60-200+ | 200 | no | yes | 0.70 |

## Chamber lifecycle — population (societal) index

| config | alpha | formation | peak SECI | episodes (start-end, + = censored) | final dissolution | dissolved? | standing at end? | persistence |
|---|---|---|---|---|---|---|---|---|
| main | 0.0 | 5 | -0.26 | 5-200+ | 200 | no | yes | 0.78 |
| main | 0.1 | 5 | -0.26 | 5-200+ | 200 | no | yes | 0.75 |
| main | 0.2 | 5 | -0.26 | 5-200+ | 200 | no | yes | 0.72 |
| main | 0.3 | 5 | -0.28 | 5-200+ | 200 | no | yes | 0.72 |
| main | 0.4 | 5 | -0.28 | 5-80; 115-200+ | 200 | no | yes | 0.68 |
| main | 0.5 | 5 | -0.28 | 5-75; 115-200+ | 200 | no | yes | 0.70 |
| main | 0.6 | 10 | -0.26 | 10-80; 115-200+ | 200 | no | yes | 0.65 |
| main | 0.7 | 10 | -0.29 | 10-85; 120-200+ | 200 | no | yes | 0.68 |
| main | 0.8 | 15 | -0.29 | 15-200+ | 200 | no | yes | 0.70 |
| main | 0.9 | 30 | -0.26 | 30-200+ | 200 | no | yes | 0.75 |
| main | 1.0 | 45 | -0.34 | 45-200+ | 200 | no | yes | 0.78 |
| control | 0.0 | 10 | -0.33 | 10-200+ | 200 | no | yes | 0.82 |
| control | 0.1 | 10 | -0.34 | 10-200+ | 200 | no | yes | 0.80 |
| control | 0.2 | 10 | -0.36 | 10-200+ | 200 | no | yes | 0.78 |
| control | 0.3 | 10 | -0.37 | 10-200+ | 200 | no | yes | 0.72 |
| control | 0.4 | 10 | -0.38 | 10-90; 120-200+ | 200 | no | yes | 0.70 |
| control | 0.5 | 10 | -0.39 | 10-90; 120-200+ | 200 | no | yes | 0.72 |
| control | 0.6 | 10 | -0.39 | 10-90; 125-200+ | 200 | no | yes | 0.68 |
| control | 0.7 | 15 | -0.36 | 15-95; 130-200+ | 200 | no | yes | 0.65 |
| control | 0.8 | 25 | -0.33 | 25-100; 140-175 | 175 | yes | no | 0.47 |
| control | 0.9 | 30 | -0.33 | 30-100 | 100 | yes | no | 0.35 |
| control | 1.0 | 40 | -0.28 | 40-105 | 105 | yes | no | 0.30 |

## Capture onset (first sustained AI-majority tick) and final-window slopes (per 100 ticks; non-zero = not equilibrated at horizon)

| config | alpha | capture onset (exploit) | capture onset (explor) | MAE-gap slope | aid-gap slope | unmet slope |
|---|---|---|---|---|---|---|
| main | 0.0 | 57 | 0 | -0.01 | 0.67 | -1.42 |
| main | 0.1 | 57 | 0 | -0.07 | 0.57 | -0.63 |
| main | 0.2 | 54 | 0 | -0.05 | 0.34 | 0.11 |
| main | 0.3 | 51 | 0 | -0.06 | 1.20 | -0.11 |
| main | 0.4 | 48 | 0 | -0.05 | 0.89 | -0.27 |
| main | 0.5 | 46 | 0 | -0.06 | 3.01 | 0.24 |
| main | 0.6 | 35 | 0 | -0.10 | 3.57 | -0.14 |
| main | 0.7 | 13 | 0 | -0.00 | 2.52 | -0.45 |
| main | 0.8 | 3 | 0 | 0.02 | 1.05 | -0.74 |
| main | 0.9 | 0 | 0 | 0.06 | 0.95 | -1.77 |
| main | 1.0 | 1 | 0 | 0.15 | -1.97 | -4.75 |
| control | 0.0 | 74 | 0 | -0.03 | 0.35 | -2.22 |
| control | 0.1 | 64 | 0 | -0.01 | 0.93 | -0.51 |
| control | 0.2 | 59 | 0 | -0.02 | 0.85 | -0.66 |
| control | 0.3 | 6 | 0 | -0.07 | 0.71 | 0.61 |
| control | 0.4 | 47 | 0 | -0.03 | 0.64 | 0.10 |
| control | 0.5 | 33 | 0 | -0.00 | 0.37 | -0.06 |
| control | 0.6 | 21 | 0 | -0.06 | 0.41 | -0.33 |
| control | 0.7 | 16 | 0 | -0.03 | -0.24 | -0.89 |
| control | 0.8 | 17 | 0 | -0.06 | 1.31 | -2.61 |
| control | 0.9 | 8 | 0 | -0.01 | 0.75 | -5.89 |
| control | 1.0 | 1 | 0 | -0.01 | 0.17 | -16.52 |

## Mechanism cascade — half-crossing tick per mechanism (-- = no meaningful net transition at that alpha)

| config | alpha | AI capture (exploit query share) | Belief starvation (explor L1+ pool) | Explorer chamber (SECI) | Precision decline (explor) | Periphery aid gap (spatial) |
|---|---|---|---|---|---|---|
| main | 0.0 | -- | 60 | -- | 45 | 45 |
| main | 0.1 | -- | 60 | -- | 50 | 45 |
| main | 0.2 | -- | 60 | -- | 50 | 40 |
| main | 0.3 | -- | 60 | -- | 55 | 45 |
| main | 0.4 | -- | 65 | -- | 60 | 45 |
| main | 0.5 | -- | 65 | -- | 65 | 50 |
| main | 0.6 | -- | 70 | -- | 75 | 60 |
| main | 0.7 | 21 | 75 | -- | 85 | 70 |
| main | 0.8 | 5 | 85 | 25 | 105 | 90 |
| main | 0.9 | 6 | 105 | 45 | 115 | 100 |
| main | 1.0 | 5 | 80 | 70 | 120 | 110 |
| control | 0.0 | -- | 70 | -- | 55 | 105 |
| control | 0.1 | -- | 70 | -- | 55 | -- |
| control | 0.2 | -- | 70 | -- | 60 | -- |
| control | 0.3 | -- | 70 | -- | 60 | -- |
| control | 0.4 | -- | 75 | -- | 65 | 145 |
| control | 0.5 | 36 | 75 | -- | 75 | 110 |
| control | 0.6 | -- | 85 | -- | 85 | -- |
| control | 0.7 | 22 | 90 | -- | 105 | -- |
| control | 0.8 | 19 | 100 | -- | 120 | -- |
| control | 0.9 | 18 | 115 | -- | 130 | -- |
| control | 1.0 | 17 | 85 | 65 | 140 | -- |

