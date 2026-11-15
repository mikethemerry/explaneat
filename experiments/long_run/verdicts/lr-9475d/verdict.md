# Verdict: lr-9475d (genome f7d31b31)

**Overall: weakened.** Every stated number, count and threshold reproduces, and the model contradicts no hypothesis. The weakenings are shape words and readings that the model only partly bears out. Findings: 0 refuted, 11 weakened, 24 upheld, 3 flags.

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| record.summary, sodium_U, root | L3 | risk lowest at 138 mmol/L (6%) | lowest at 138.5-139 (5.3%); the source sweep used 2 mmol/L steps | probe sodium 137-141 in 0.25 steps |
| sodium_U | L3 | the direct sodium edge adds a "gentle" downward tilt | -0.21 logit per mmol/L: +2.24 at 126 to -3.66 at 154, a 5.9 logit swing | probe --wide sodium sweep |
| record.summary | L3 | noise has a "small" fitted effect | small on average (AUC 0.904 to 0.901 with noise fixed), but at sodium 145, noise -2 to 2 moves risk 63% to 91%; single patients move by up to 31 points | counterfactual on training rows |
| root | L3 | age raises the risk "steadily" | for the typical (non-diabetic male) patient: 11.6% at 75, then a jump to 49% at 85 as the renal unit opens, then flat; for a woman a steady rise to 22.5% at 90 | age sweep at 2.5-year steps, with node 228 |
| root | L3 | BMI curve "dips in the middle" | rises from 8.3% (16) to 12.5% (20), then a cliff to 4.1% at 23 as the gate chain closes, then a slow climb to 10.4% at 40 | BMI sweep at 1-unit steps |
| renal_228 | L3 | reads as "low sodium, or being a non-diabetic man" | needed but not enough: it fires for 215 of the 457 rows with either open (96 of 234 gate-only, 71 of 160 low-sodium-only) | forward pass on 1500 rows |
| gate_and_renal | L3 | the renal unit fires "when either" the gate or the low-sodium chain is open | same as above: a threshold on their sum (bias -1.19) | forward pass |
| age_bmi, sex_diabetes_gate | L1 | AUC 0.904 to 0.895 when switched off | 0.8955 (rounds to 0.896); rounded twice | ablate |

**Upheld parts** (every number and direction reproduces):
- low_sodium (ann_24): an exact hinge at 134 mmol/L, 223 rows, 0 exceptions; exit values, the 119 changed rows and AUC 0.889 all reproduce.
- high_sodium (ann_25): unit 1173 is an exact hinge at 140 (635 rows); unit 338 is on for 796 rows; the noise swing and AUC 0.757 reproduce.
- age_bmi (ann_34): "older and leaner" holds in the probes at BMI 22 and 34 (weakened only for the rounding).
- sex_diabetes_gate (ann_35): open for 0 women or diabetic patients; every stated gate value reproduces (weakened only for the rounding).
- bp_hr (ann_37): opposite slopes hold for 99.6% and 100% of training rows; all probe values and the four subgroup cells reproduce. The slopes are step-like, mostly between 120 and 150 mmHg.
- direct (ann_38): the coefficients, creatinine scores, smoking and diabetes steps, and AUC 0.806 reproduce.
- Record: coverage 1.0, function preservation 7.2e-16, closed-form error 0.016, operation counts, the summary's subgroup numbers, the 28% vs 15% diabetes reversal, and the per-row "age, creatinine and smoking raise the risk" (no exceptions) and "mostly diabetes" (91.5% of rows) all reproduce.

**Leaf meaning:** low_sodium, high_sodium, age_bmi, sex_diabetes_gate and bp_hr are meaningful; renal_228 is partial; direct is closed-form-only.

**Flags (L4, domain only):**
- "Shock-index-like" describes only the low-pressure, fast-pulse arm. The high-pressure, slow-pulse arm is a different pattern.
- Diabetes lowers the predicted risk for older, lean men (28% to 15%).
- Unit 228 is called "renal", but creatinine enters it weakly (0.142 per SD) and never opens it alone. "Alarm unit" fits better.

Note: the record target's recheck.json has no `target_hash`, so `target_hash` is null in verdict.json. Nothing was written to the database.
