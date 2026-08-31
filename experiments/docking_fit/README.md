# E1 — Effect-size-targeted docking fit

Implements Tier-1 step E1 (`docs/EMPIRICAL_FOUNDATIONS_PLAN.md` §2):
fit the model's free dyadic micro-parameters to dimensionless effect
sizes computed from the published data of the docking target experiment
(Glickman & Sharot, *How human–AI feedback loops alter human
perceptual, emotional and social judgements*, Nat. Hum. Behav. 9,
345–359, 2025), then verify the population results are intact at the
fitted parameters. Upgrades SI §M6 from qualitative pattern
reproduction to indirect calibration.

## Pipeline

1. **`compute_targets.py`** → `targets.json` (committed).
   Reads a local clone of the authors' public data repo
   (https://github.com/affective-brain-lab/BiasedHumanAI — clone it
   yourself; raw data is not redistributed here) and computes, with
   the authors' own trial/block conventions, the transmission
   coefficients κ = induced bias / (partner − self gap) for the
   human×AI and human×human conditions (Exp1), plus the accurate-AI
   correction and biased-AI induction checks (Exp2). Bootstrap CIs
   over subjects. The clone's commit hash is recorded in the output.

   ```
   python3 experiments/docking_fit/compute_targets.py \
       --data-dir /path/to/biasedhumanai
   ```

2. **`fit_docking.py`** → `results-docking-fit/`.
   Latin-hypercube search over the dyadic free parameters
   (`d_exploit`, `delta_exploit`, `d_explor`, `delta_explor`,
   `initial_trust`, `initial_ai_trust`, `rounds`), reusing the M6
   harness (`experiments/dyadic_docking.py`, unchanged behaviour at
   defaults). Inverse-CI-weighted quadratic loss on κ_AI and κ_human
   with ordering/monotonicity penalties; coarse pass then top-set
   refinement on the full α grid. Trust *learning rates* are not
   fitted — the dyad holds trust fixed by design (no relief loop, no
   reward signal), so the trust-side free parameters are the initial
   levels. Outputs: `fitted_params.json`, `fit_report.md` (with the
   identifiability note), `docking_fit.png`, and the raw grids.

   ```
   python3 experiments/docking_fit/fit_docking.py
   ```

3. **`population_check.py`** → `results-docking-fit/population_check_*.json`.
   Reruns the population model at the fitted parameters on a reduced
   grid (default α ∈ {0, 0.6, 1.0} × both configurations × N=5) and
   checks the headline structure: starvation gradient, interior
   operational optimum, structural precondition. Qualitative gate
   only — the SI-grade comparison is the canonical N=20 sweep with the
   fitted overrides, run on CI.

   ```
   python3 experiments/docking_fit/population_check.py
   ```

## Correspondence and phrasing (binding for the paper)

- κ is measured identically in spirit on both sides: the fraction of
  the partner–self judgment gap the focal human has adopted by the end
  of the interaction. G&S's biased-AI partner and the model's truthful
  AI both hold a fixed judgment the human does not share; the
  transmission fraction is direction-agnostic.
- The dyad's interaction length (`rounds`) is a free parameter because
  experiment blocks and model rounds share no clock.
- Claim template: "dyadic micro-parameters estimated from the published
  human–AI interaction effect sizes; the population-scale results are
  unchanged" — never "the model is calibrated to disaster data".
- Expect ridges, not a point estimate: the fit constrains parameter
  *combinations*; `fit_report.md` reports the top-set spread as the
  identifiability statement.
