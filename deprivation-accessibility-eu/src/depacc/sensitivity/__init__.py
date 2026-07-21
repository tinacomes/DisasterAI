"""Robustness harness (structured, NOT probabilistic).

Recomputes only STANDARDISED / rank-based targets (Ginis, typology shares,
city rankings, cluster membership, divergence_gap) for each parameter variant;
raw deprivation magnitudes are never tracked. Curvature and form-swap layers
are evaluated on the saved deprivation-free travel times (t_regime_*), so no
re-routing is needed; the accessibility layer is the comparison axis and is
reported when accessibility-variant runs are supplied.
"""

from depacc.sensitivity.harness import (  # noqa: F401
    city_stable_targets,
    expand_variants,
    flip_cells,
    run_sensitivity,
)
