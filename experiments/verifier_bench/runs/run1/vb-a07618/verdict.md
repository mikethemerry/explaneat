# Verdict vb-a07618 (genome b19b9536, Heart Disease)

**Overall: upheld.** Every number, closed form, claim, count, threshold, interaction and ablation figure I checked reproduces on the model. Nothing is refuted or weakened.

Started 2026-11-10T23:14:33+13:00, finished 2026-11-10T23:25:08+13:00 (11 minutes).

## Refuted and weakened findings

None.

## Parts upheld

- **ann_35 The whole model**: coverage 1.0/1.0, single root, function preserved (3.3e-16). All 15 linear-core coefficients and the -0.954 intercept reproduce. Core SD 4.96, readouts SD 1.41. AUC 0.9437 falls to 0.9295 with every readout at its mean. SHAP ranking reproduces within sampling noise. Exercise angina and ST depression have no direct edge. The resting ECG has no effect.
- **ann_33 ST depression hinge**: switches on at 1.49 mm (stated about 1.5 mm), then rises linearly: 45.4% at 2.8 mm. It is active for 70 patients, 50 of whom had heart disease. With the part at its mean, AUC is 0.9400.
- **ann_31 No chest pain on exercise, by sex**: the output is 0 for all 82 patients with exercise angina. The contribution is -0.43 for the 96 men without it and -1.67 for the 64 women without it. Being a woman lowers the score by 4.89 without exercise angina and by 3.65 with it.
- **ann_32 Fixed defect with normal sugar**: with normal blood sugar, a fixed defect takes the typical patient from 19.7% to 1.0%. With high sugar it barely changes the risk (3.4% to 3.3%). The part acts on n=10 (5 positive). All ten per-patient logits match the evidence record exactly.
- **ann_30 Vessel count boost**: never clips (minimum 0.330). The effective ca weight is 2.599. A thallium result coded 4 gives 92.9%. With the part at its mean, AUC is 0.9408.
- **ann_34 Atypical angina offset**: contributes -0.044 for exactly the 42 patients with atypical angina. Switching it off changes risk by at most 0.0085, and AUC goes from 0.9437 to 0.9434.

## Flags (domain sense only, L4)

- High fasting blood sugar lowers predicted risk (19.7% to 3.4%) [C20].
- A fixed thallium defect lowers predicted risk below that of a normal scan (to 1.0%). This rests on 10 patients and is likely overfitted [C17, C41].
- Going from 0 to 2 vessels takes the typical patient from 20% to 99%. The direction is expected but the effect is very steep [C01].

The explainer already raised all three.
