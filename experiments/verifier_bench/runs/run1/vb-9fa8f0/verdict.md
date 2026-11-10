# Verdict vb-9fa8f0 (churn, genome 6a26c823)

**Overall: weakened.** Coverage (1.0/1.0), function preservation (7.2e-16), the closed forms (error 0.0077), all 158 claims and all 120 prose numbers reproduce, and each part behaves as described under probing and ablation. Three technical statements are imprecise. Nothing is refuted.

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| `operations.churn_logit.hypothesis` (ann_95) | L1 | 22 unconnected inputs: 4 numeric, area code 408 and 16 states | 4 numeric + area code 408 + **17** states = 22 (16 does not add up) | root entry nodes mapped to feature names |
| `operations.csc_hinge_7208.evidence[1]` (ann_92) | L1 | active_rows 304 | active on **306** rows (318 with 4+ calls minus 12 in West Virginia), as C127 says | `ablate --part ann_92 --to zero`; `subgroup` CSC>=4 by state |
| `operations.csc_usage_gate.hypothesis` (ann_91) | L2 | churn 0.88 low-day vs 0.24 high-day for 4+ calls, "which the network reproduces" | observed rates hold, but the model predicts 65% vs 88.5% observed (<160 min) and 41% vs 25% (>=180.5 min): the order is right, the levels are off by about 20 points | `subgroup` CSC>=4 by day minutes |

Upheld:
- **ann_95, the whole model:** the sum of the readouts and the intercept match. Linearising reproduces: all parts 0.837, the gate restored 0.899, the hinge restored 0.8505. The rare-state offsets (-2.2/-2.1/-1.4) and their observed log-odds (-1.22/-0.83/-0.47) reproduce.
- **ann_90, fixed-weight inputs:** exactly linear, with the stated weights. Evening minutes and charge are identical after scaling. AUC is 0.539 with the part off.
- **ann_91, calls with low usage:** daytime minutes stop lowering the chance at about 170 min (1 call). The kink moves right with calls from 3 onward. Evening charge changes the effect of calls (+2.49 at $12 vs flat at $22). The mean gate logit goes from 1.14 to 7.41 (0 to 7 calls).
- **ann_93, voice-mail discount:** exactly linear, -0.288 logit per z of the plan. 2.1% in Minnesota vs 1.5% in West Virginia.
- **ann_92, many service calls:** nothing up to 3 calls, then +0.61 logit per call. 16% without vs 26% with for the typical active customer. Off for all 12 West Virginia customers with 4+ calls. It does switch on at 9 calls in West Virginia, a case absent from the data. The "2.75 logits" is the whole-model change, of which this part gives 2.43.
- **ann_94, state units:** Tennessee +0.40, Ohio +0.22, Nevada -0.45, California -0.21. The AUC is unchanged with the part off.
- **The low-usage overprediction in `resisted`:** the effect is real (e.g. 11.0% predicted vs 3.1% observed, n=159). I did not identify the exact 93-row subgroup.

Flags (L4, data not contradicted):
- A voice-mail plan lowers the chance while more messages raise it, on two strongly linked inputs.
- West Virginia is exempted from the many-calls part, fitted to 12 customers.
- Nevada's chance is lowered, although Nevada customers leave more often (18.2% vs 14.1%).
- The state offsets are about twice the observed log-odds, on 73 to 80 customers per state.

Not checked: the SHAP and sensitivity rankings, for which there is no tool.
