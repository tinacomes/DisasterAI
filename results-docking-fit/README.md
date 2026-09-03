# E1 docking fit — effect-size-targeted calibration of the dyad (fit executed; population sweep pending)

Fit of the model's free dyadic micro-parameters (acceptance windows
D/δ for both types, initial human and AI trust, interaction length) to
transmission coefficients computed from Glickman & Sharot's published
per-trial data (Nat. Hum. Behav. 9, 345–359, 2025; data repo
`affective-brain-lab/BiasedHumanAI`, commit `ed7b660`). Pipeline and
correspondence rules: `experiments/docking_fit/README.md`. Plan and
status: `docs/EMPIRICAL_FOUNDATIONS_PLAN.md` §2; review with the open
items: `docs/development/EMPIRICAL_REVIEW_2026-09-03.md` §2.

- **Run**: local (in-session), 2026-08-31, commit `a20599b` on
  `claude/disasterai-measurement-protocol-bnog4g`; no CI workflow for
  the fit itself. Settings: 96 Latin-hypercube points × 8 coarse seeds
  × 3 conditions; top-8 (+ default) refined at 20 seeds × 5 α levels.
  `fit_docking.py --report-only` regenerates `fit_report.md` and
  `docking_fit.png` byte-identically from the committed files
  (verified 2026-09-03).
- **Files**: `fitted_params.json` (fitted set `lhs021`, defaults,
  targets, search box, coarse top-10), `coarse_grid.csv` /
  `refined.csv` (raw per-run rows), `fit_report.md` (tables,
  interpretation, identifiability), `docking_fit.png`,
  `population_check_fitted.json` (reduced-grid structure check,
  α ∈ {0, 0.6, 1.0} × both configurations × N=5; all four checks pass).

## Verdict — fit executed; deliverable pending

Headline numbers (fit_report.md): κ_AI 0.16 (default) → 0.27 (fitted)
vs measured 0.75 [0.52, 0.98]; κ_human 0.26 → 0.28 vs measured 0.31
[−0.59, 1.34]; both orderings reproduce; loss 25.2 → 16.9. The one
identified direction is the initial AI trust (≈2× default). Read the
numbers with the review's caveats: the human–human target is
uninformative (its CI spans zero), the model under-transmits AI
influence for both types (below the measured CI; near zero for
confirmation-seekers by construction of the D/δ window), and the
fitted `rounds` (60) and `d_explor` (5.37) lie on or near the
S9-certified search-box boundary.

Still to do before the SI can carry the claim "micro-parameters
estimated from the published experiment; population results intact":
the canonical N=20 × 11-α × two-configuration sweep at the fitted
parameters (workflow `Run Docking-Fit Sweep (E1)`,
`.github/workflows/run-docking-fit-sweep.yml`; archive as
`results-docking-fit-sweep/`), and the fit rerun with the Exp2
accurate-AI error ratio as a second target (review A6).

## Caveats

- Effect-size-targeted, not parameter-matched: Glickman & Sharot's
  tasks share no units with the severity grid; only dimensionless
  transmission fractions and orderings are fitted.
- Trust learning rates are inert in the dyad (no relief loop) and are
  not fitted; `rounds` is dyad-only and is not carried into the
  population model.
