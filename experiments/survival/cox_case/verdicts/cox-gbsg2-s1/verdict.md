# Verdict: cox-gbsg2-s1 (genome e4c62516-fd1f-4e10-a997-adc574bbcef3)

**Overall: upheld.** Every stated number, structural claim, hazard ratio, subgroup count, ablation effect and survival metric reproduces, and the four hypotheses survive adversarial probing; two domain flags are left for the reviewer.

Verifier: independent blind verifier (explanation-verifier skill). Started 2026-11-12T14:02:24+13:00, finished 2026-11-12T14:11:11+13:00 (9 minutes). Database read only. Findings: 62 upheld, 0 weakened, 0 refuted, 2 flags. The recheck carries no `target_hash` for this record target; sha256 of the record JSON is 71f43de7…213dfc (in verdict.json).

## Refuted and weakened findings

None.

## Parts

| Part | Grade | What was checked |
|---|---|---|
| cox_log_risk (ann_15, root) | upheld | exact sum with unit child weights and bias -0.747; 0 direct edges at the output; the four unconnected inputs change 0 of 438 predictions (node ablation and sweeps); Harrell 0.682, Uno 0.651, td-AUC 0.765/0.731/0.707, IBS 0.132, calibration 50% vs 49%, log-risk HR 5.82 recomputed independently with `explaneat.survival.metrics` on the model's log-risk; all three parts off gives AUC 0.50 and risk 56% to 48%; single-part AUC losses rank 0.6258 < 0.6675 < 0.6767 |
| tumour_biology (ann_12) | upheld | weights -0.3161/0.3208/0.3453; HR per unit 2.264 (III), 1.960 (II), 0.99838 per fmol/l (0.729 per SD) from the training SDs; spans 1.92/2.26/1.96; the three grade cells (-1.1995, -0.5265, -0.3825; 34/56/61%) and the progrec sweep (-0.4787 to -0.9644, 58% to 41%, linear); subgroups 102/285/51/0 with the stated risks, event rates and contributions; part and edge ablations (0.6675; 0.6742, 0.6387, 0.6751); concept sweep (progrec > 29.5, 0.905, R2 0.30; ordinal grade 0.36; progrec 0.70); additivity holds at other inputs (grade gaps 0.673 and 0.817 at pnodes 15, progrec 300, therapy yes) |
| nodal_burden (ann_13) | upheld | weight 0.2926, HR 1.0513 per node, 1.34 per SD; spans 2.12 and 1.92; sweep 1 to 29 (-0.6266 to 0.7748, 53% to 95%) and 1 to 51 with constant steps of 0.05005 per node; one vs two nodes -0.6266/-0.5765; subgroups 71 (71%, 73%, 0.520) and 234 (43%, 31%, -0.175); ablation 6%, max 49%, AUC 0.6258; concept pnodes 1.00 |
| hormone_therapy (ann_14) | upheld | weight -0.2062, HR 0.6534, 0.81 per SD, span 1.53; treated/untreated cells -0.9520/-0.5265 (42%/56%, contributions -0.265/+0.160); the gap -0.4255 is identical at grade III, pnodes 15, progrec 300; subgroups 165 (41%, 37%) and 273 (55%, 48%); ablation 6%, max 10%, AUC 0.6767; concept 1.00; called an association, not a treatment effect |

Record fields (coverage 1.0/1.0, hierarchy complete, preservation 4.4e-16, closed-form error 0.0035, 3 identity nodes, 0 splits, 0 hidden nodes, summary, resisted, hazard_ratio_summary, meaning, ph_caveat) all reproduce. The recheck reported 0 mismatch / 0 needs review; its 77 unchecked sentences were checked by hand with probe, subgroup, ablate, concept, linearise and a scratch recomputation.

## Flags (domain, for the reviewer; they do not change the verdict)

- **tumour_biology**: the progesterone-receptor effect is log-linear per fmol/l over the whole training range (0 to 1600 fmol/l) with no saturation: the model gives HR 0.20 at 1000 fmol/l against 0, and 0.52 at the 95th percentile (403 fmol/l). Clinical use usually reads receptor status at a positive/negative cut-off. The data do not contradict the model; the text states the linear form.
- **cox_log_risk**: age, oestrogen receptor, menopausal status and tumour size have zero effect (0 of 438 predictions change when fixed). Tumour size and age are standard prognostic factors in this cohort's usual analyses; the explanation states the omission as a model property and claims nothing clinical for it.

## Remarks

- The "meaningful" grade of tumour_biology rests on the concept tool's rule (score 0.905 from the AUC 0.953 of a sign-level threshold); R2 is 0.30 and the text says the concept carries the direction, not the value.
- The three leaves are the explainer's grouping of five independent additive edges (the model has no hidden node); the grouping is clinically coherent but any grouping would preserve the function.
- The proportional-hazards caveat is present in the record and in all four hypotheses.
