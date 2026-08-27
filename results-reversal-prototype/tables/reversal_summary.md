# Alpha-reversal (hysteresis) experiment — summary

Per-seed post-switch recovery of the accuracy-seekers' chamber (standing = SECI episode containing the switch tick; dissolution thresholds as in test_filter_bubbles), plus endpoint (last-75-tick) means ± 95% CI. Compare each switch row against BOTH constant anchors on the same seeds: recovery to the truthful anchor = reversible; endpoint at the aligned anchor = hysteresis.

| label | n | standing_at_switch | dissolved_after | median_lag | end_SECI_explor | end_SECI_exploit | end_MAE_explor | end_L1pool_explor | end_prec_explor | end_sp_MAE_gap | end_sp_aid_gap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| constant α=0.0 | 6 | 2/6 | 1/2 | 10.0 | -0.006 ± 0.020 | -0.205 ± 0.075 | +0.450 ± 0.071 | +117.789 ± 13.116 | +0.998 ± 0.001 | +0.173 ± 0.111 | -2.982 ± 2.287 |
| α=1.0→0.0 @t=100 | 6 | 5/6 | 4/5 | 20.0 | -0.013 ± 0.029 | -0.178 ± 0.095 | +0.620 ± 0.032 | +92.935 ± 7.217 | +0.990 ± 0.004 | +0.205 ± 0.159 | -4.087 ± 3.076 |
| constant α=1.0 | 6 | 5/6 | 0/5 | -- | -0.296 ± 0.067 | -0.031 ± 0.064 | +1.658 ± 0.205 | +29.020 ± 7.178 | +0.858 ± 0.056 | +0.500 ± 0.195 | -10.100 ± 4.042 |
