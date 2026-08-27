# Alpha-reversal (hysteresis) experiment — summary

Per-seed post-switch recovery of the accuracy-seekers' chamber (standing = SECI episode containing the switch tick; dissolution thresholds as in test_filter_bubbles), plus endpoint (last-75-tick) means ± 95% CI. Compare each switch row against BOTH constant anchors on the same seeds: recovery to the truthful anchor = reversible; endpoint at the aligned anchor = hysteresis.

| label | n | standing_at_switch | dissolved_after | median_lag | end_SECI_explor | end_SECI_exploit | end_MAE_explor | end_L1pool_explor | end_prec_explor | end_sp_MAE_gap | end_sp_aid_gap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| constant α=0.0 | 20 | 5/20 | 2/5 | 7.5 | -0.010 ± 0.011 | -0.069 ± 0.039 | +0.412 ± 0.029 | +120.160 ± 6.307 | +0.999 ± 0.000 | +0.108 ± 0.169 | -1.570 ± 2.115 |
| α=0.0→1.0 @t=100 | 20 | 5/20 | 1/5 | 15.0 | -0.023 ± 0.014 | -0.017 ± 0.057 | +0.567 ± 0.087 | +107.433 ± 8.178 | +0.998 ± 0.002 | +0.169 ± 0.171 | -1.497 ± 2.495 |
| α=0.8→0.0 @t=100 | 20 | 20/20 | 20/20 | 30.0 | -0.010 ± 0.008 | -0.072 ± 0.034 | +0.465 ± 0.021 | +108.927 ± 5.975 | +0.998 ± 0.001 | +0.085 ± 0.163 | -1.515 ± 1.818 |
| constant α=0.8 | 20 | 20/20 | 13/20 | 30.0 | -0.080 ± 0.028 | +0.014 ± 0.031 | +0.893 ± 0.052 | +80.581 ± 4.596 | +0.995 ± 0.002 | +0.246 ± 0.115 | -3.010 ± 1.660 |
| α=1.0→0.0 @t=100 | 20 | 19/20 | 18/19 | 50.0 | -0.011 ± 0.009 | -0.087 ± 0.041 | +0.486 ± 0.033 | +102.039 ± 5.870 | +0.997 ± 0.001 | +0.118 ± 0.162 | -1.911 ± 2.072 |
| constant α=1.0 | 20 | 19/20 | 4/19 | 32.5 | -0.232 ± 0.062 | -0.010 ± 0.074 | +1.592 ± 0.103 | +29.793 ± 5.255 | +0.860 ± 0.043 | +0.466 ± 0.106 | -8.221 ± 1.894 |
