# Methods

Every modelling choice in the pipeline, with the literature source of every
parameter. Sections marked **TODO(cite)** must be completed from the named
papers before the corresponding computation is run — the code enforces this
(`depacc.config.require_params` refuses null placeholders and repeats the
citation in its error message). **Never** fill a parameter with an invented
or "reasonable-looking" value.

## 1. Design

Cross-sectional, multi-city study modelled on Musso et al. (PNAS, 2026,
"Large cities lose their growth advantage as countries urbanize"): one
harmonised city definition across countries (Eurostat-OECD Functional Urban
Areas, cross-checked against JRC GHS-UCDB urban centres), and *trajectories
read from the cross-sectional city-size gradient* (space-for-time
substitution). **There is deliberately no longitudinal/temporal component**;
all statements about "trajectories" are cross-sectional inferences and are
labelled as such in every output.

## 2. Potential deprivation

For each populated 100 m grid cell *i* and service type *s*, potential
deprivation is an estimated deprivation function evaluated at an *effective*
travel time. The deprivation function is the impedance function of a gravity
model run in the opposite direction: increasing and convex in travel time,
rather than a decreasing discount. Two regimes, computed as separate
surfaces per city:

### 2.1 Everyday regime (chosen, repeated, substitutable)

1. **2SFCA congestion factor.** Step 1 computes the supply-to-demand ratio
   per facility *j*: `R_j = S_j / Σ_i P_i K(t_ij)` with kernel *K*
   (gaussian, mode-specific bandwidth; config `catchment.kernel`). The
   decreasing kernel appears *only* inside this competition weighting.
   Step 2 converts crowding into a travel-time inflation
   `c_j = (R_ref / R_j)^γ` (reference: demand-weighted median ratio in the
   city; exponent γ and clip bounds in config; γ = 0 disables congestion).
2. **Soft-minimum reducer.** Effective deprivation time
   `t_eff(i) = -(1/κ) ln Σ_j exp(-κ · t_ij · c_j)`, a smooth minimum with
   the deliberate property that several similar-time options reduce the
   effective time (substitutability bonus); bounds
   `min - ln(n)/κ ≤ t_eff ≤ min` (unit-tested). κ in config
   (`softmin.kappa`), sensitivity-swept.
3. **Deprivation.** `D_ev(i) = g_DLF(t_eff(i))`.

### 2.2 Emergency regime (non-substitutable, time-critical)

`D_em(i) = g_DCF(min_j t_ij)` — nearest facility only; the convexity of the
DCF is where its shape matters most.

### 2.3 Baseline

For **both** regimes, the plain nearest-facility travel time is always
computed and reported as a comparison baseline.

### 2.4 Unreachable cells

Cells with no reachable facility of a service within `routing.max_time_min`
are flagged explicitly and handled by config policy — `cap_at_max_time`
(default: deprivation at the cutoff time) or `exclude` (NaN, dropped from
aggregates) — and their population share is always reported.

## 3. Deprivation functions (transferred, not estimated)

| Regime | Kind | Form | Parameters | Source |
|---|---|---|---|---|
| Everyday | DLF (dimensionless) | exponential `g(t) = scale·(e^{βt} − 1)` | β = **TODO(cite)**, scale = **TODO(cite)** | **TODO(cite):** Wang et al., NRS-based Deprivation Level Functions — record full reference, table/equation here when filled |
| Emergency | DCF (monetary) | Box-Cox `g(t) = scale·((t+shift)^λ − shift^λ)/λ` | λ, scale, shift = **TODO(cite)** | **TODO(cite):** Holguín-Veras et al.; Cantillo et al.; Macea et al.; Delgado-Lindeman et al. — econometric DCF estimates; record reference, currency-year, time unit here when filled |

Both forms satisfy g(0)=0, g′>0, g″≥0 (unit-tested). Alternative
specifications for sensitivity analysis live in
`config/deprivation.yaml → deprivation.alternatives`.

## 4. Divergence outputs (the central result)

1. **Cell-level co-location:** bivariate typology at population-weighted
   median thresholds (quantile configurable): LL/LH/HL/**HH** (compounding
   deprivation), population-weighted and mapped.
2. **City-level divergence:** each city as a point in the
   (Gini of everyday deprivation, Gini of emergency deprivation) plane;
   off-diagonal spread measured alongside population-weighted mean levels.
3. **Trajectory:** cities ordered along the FUA-population size gradient;
   test whether everyday and emergency deprivation/inequity co-evolve or
   diverge with size. Cross-sectional (space-for-time) inference only.

## 5. Travel times

Single routing engine (R5 via r5py, JDK 21) for all modes, so Tier-1 and
Tier-2 matrices are methodologically identical: walk + car for every city
(OSM); walk+transit added for Tier-2 cities with GTFS (departure window in
`routing.departure`). Origins: centroids of populated GHS-POP 100 m cells
within the FUA; destinations: OSM facilities per service
(`config/services.yaml`, including capacity sources; uniform-capacity
proxies are flagged). Optional robustness check against the Weiss et al.
motorised friction surface (1 km).

## 6. Equity statistics

Population-weighted mean deprivation; population-weighted Gini (covariance
form); concentration index against SES rank (Tier 2: income/rent proxies
from national 100–200 m grids); within-city gradient regressions
(deprivation on income/rent proxy, age structure, household composition).

## 7. Data quality

OSM facility completeness characterised **per country** by benchmarking OSM
hospital/pharmacy counts against national registries (or OSM intrinsic
quality metrics); completeness table produced by `quality/`; the Tier-1
sample can be restricted to cities above `quality.completeness_threshold`.
For Tier 2 we test whether adding transit changes city *rankings* and
clustering, not just levels.

## 8. Reproducibility

Config-driven (YAML per city + tier); cached downloads with SHA-256
provenance sidecars; no raw data committed; unit tests on the DLF/DCF
mapping, soft-min reducer, 2SFCA factor, unreachable handling and the
divergence typology; CI runs the tests on every push.
