### Per seed (test fold; selection used the validation Cox loss only)

| Dataset | Seed | Val Cox loss | Train Cox loss (null) | Harrell C | Uno C | td-AUC 1y / 3y / 5y | IBS | Calibration E/O (horizon) | Hidden | k | Inputs used | Events / param | Wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GBSG2 | 0 | 3.9802 | 5.381 (5.532) | 0.675 | 0.642 | 0.722 / 0.754 / 0.713 | 0.114 | 0.966 (1825 d) | 0 | 5 | 5 | 38.2 | 13.1 |
| GBSG2 | **1** | 3.9802 | 5.381 (5.532) | 0.675 | 0.642 | 0.722 / 0.754 / 0.713 | 0.114 | 0.966 (1825 d) | 0 | 5 | 5 | 38.2 | 13.1 |
| GBSG2 | 2 | 3.9802 | 5.381 (5.532) | 0.675 | 0.642 | 0.722 / 0.754 / 0.713 | 0.114 | 0.966 (1825 d) | 0 | 5 | 5 | 38.2 | 12.0 |
| GBSG2 | 3 | 3.9802 | 5.381 (5.532) | 0.675 | 0.642 | 0.722 / 0.754 / 0.713 | 0.114 | 0.966 (1825 d) | 0 | 5 | 5 | 38.2 | 12.0 |
| GBSG2 | 4 | 3.9802 | 5.381 (5.532) | 0.675 | 0.642 | 0.722 / 0.754 / 0.713 | 0.114 | 0.966 (1825 d) | 0 | 5 | 5 | 38.2 | 12.1 |
| WHAS500 | **0** | 3.6001 | 4.766 (5.234) | 0.830 | 0.833 | 0.848 / 0.913 / 0.862 | 0.138 | 0.885 (1095 d) | 0 | 9 | 9 | 15.33 | 12.4 |
| WHAS500 | 1 | 3.6383 | 4.739 (5.234) | 0.826 | 0.830 | 0.837 / 0.903 / 0.862 | 0.137 | 0.885 (1095 d) | 0 | 8 | 8 | 17.25 | 12.5 |
| WHAS500 | 2 | 3.6262 | 4.764 (5.234) | 0.810 | 0.816 | 0.814 / 0.887 / 0.830 | 0.149 | 0.874 (1095 d) | 0 | 10 | 10 | 13.8 | 12.4 |
| WHAS500 | 3 | 3.6277 | 4.800 (5.234) | 0.823 | 0.825 | 0.831 / 0.901 / 0.830 | 0.145 | 0.872 (1095 d) | 0 | 12 | 12 | 11.5 | 12.8 |
| WHAS500 | 4 | 3.6391 | 4.800 (5.234) | 0.826 | 0.832 | 0.850 / 0.907 / 0.837 | 0.143 | 0.866 (1095 d) | 0 | 9 | 9 | 15.33 | 12.8 |

The selected seed is in bold. Val / train Cox loss: Efron partial likelihood per event on that fold; null = the constant model on the training fold.

### References on the same split (test fold)

| Dataset | Model | Params | Harrell C | Uno C | td-AUC 1y / 3y / 5y | IBS | Brier at horizon | Calibration E/O (horizon) | Max group gap | Test Cox loss |
|---|---|---|---|---|---|---|---|---|---|---|
| GBSG2 | PropNEAT-Cox (selected seed) | 5 | 0.675 | 0.642 | 0.722 / 0.754 / 0.713 | 0.114 | 0.240 | 0.966 (1825 d) | 0.211 | 4.196 |
| GBSG2 | Ridge Cox (alpha on validation) | 9 | 0.680 | 0.632 | 0.753 / 0.756 / 0.685 | 0.114 | 0.245 | 0.956 (1825 d) | 0.338 | 4.205 |
| GBSG2 | Cox PH, unpenalised | 9 | 0.680 | 0.632 | 0.753 / 0.756 / 0.685 | 0.114 | 0.245 | 0.956 (1825 d) | 0.338 | 4.205 |
| GBSG2 | Random survival forest | 500 trees | 0.700 | 0.669 | 0.720 / 0.751 / 0.752 | 0.105 | 0.222 | 0.920 (1825 d) | 0.302 | -- |
| WHAS500 | PropNEAT-Cox (selected seed) | 9 | 0.830 | 0.833 | 0.848 / 0.913 / 0.862 | 0.138 | 0.146 | 0.885 (1095 d) | 0.453 | 3.588 |
| WHAS500 | Ridge Cox (alpha on validation) | 14 | 0.819 | 0.823 | 0.829 / 0.900 / 0.853 | 0.149 | 0.163 | 0.908 (1095 d) | 0.324 | 3.715 |
| WHAS500 | Cox PH, unpenalised | 14 | 0.818 | 0.820 | 0.824 / 0.893 / 0.852 | 0.141 | 0.147 | 0.888 (1095 d) | 0.414 | 3.623 |
| WHAS500 | Random survival forest | 500 trees | 0.808 | 0.815 | 0.833 / 0.883 / 0.866 | 0.150 | 0.159 | 0.885 (1095 d) | 0.362 | -- |

### Explanation, the three levels and the verifier

| Dataset | Label | Leaves | Compositions | Identity nodes | Structural / compositional coverage | Faithful: max abs err | Closed-form err | Granularity | Understood: meaningful / partial / closed-form only | Recheck mismatch / needs review | Verifier |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GBSG2 | cox-gbsg2-s1 | 3 | 1 | 3 | 1.00 / 1.00 | $4.4\times10^{-16}$ | $3.5\times10^{-3}$ | -- | 3 / 0 / 0 | 0 / 0 | upheld (64 findings: 0 refuted, 0 weakened) |
| WHAS500 | cox-whas500-s0 | 4 | 1 | 4 | 1.00 / 1.00 | $4.4\times10^{-16}$ | $1.4\times10^{-3}$ | -- | 4 / 0 / 0 | 0 / 0 | weakened (39 findings: 0 refuted, 1 weakened) |

