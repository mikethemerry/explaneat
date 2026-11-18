## Reading (hand-written, Claude, 2026-11-18; draft for Mike's review)

**In one paragraph.** Resting blood pressure reaches the output through three switches, and each patient's
curve is those three switches in some order. (1) *A low-pressure discount* (unit 2116, fitness_pathway):
below about 113–115 mm Hg the model lowers the risk, less and less as pressure rises, so **every**
patient's risk rises from 94 up to about 113 mm Hg. (2) *The ischaemia gate* (unit 855, ischaemia_gate):
above a patient-specific pressure the risk **rises steeply** (+0.13 log-odds per mm Hg). The gate opens
earlier with a reversible thallium defect, ST depression, high fasting sugar or exercise angina. It opens
later with an upsloping ST segment, and typical angina moves it 113 mm Hg higher, which for most patients puts it beyond the observed range. (3) *The
fitness readout* (unit 4037 via the fitness score 2445, fitness_pathway): above another patient-specific
pressure the risk **falls** (−0.07 log-odds per mm Hg). That pressure is lower for younger patients,
patients with a high peak heart rate and patients without exercise angina. The "strange" shapes are
which of (2) and (3) comes first and whether it falls inside 94–200 mm Hg.

**The families (the model, not the world):**

- **F1, 76 patients: rises, levels off, rises again.** The discount ends at about 113. Then the curve is flat
  until the gate opens (median 142 mm Hg, but anywhere from 121 to 177), and after that it climbs. For the
  typical member the risk is 5% at 94, 25% from 113 to 142 and 94% at 200 mm Hg. Older patients without a
  reversible defect are the clearest members (age > 65.5 and no reversible defect: 20 of 20). The family is
  the "default" shape, though, and the tree describes it only loosely (balanced accuracy 0.75).
- **F2, 66 patients: rises all the way.** The gate is open from low pressure (median 107 mm Hg, or open
  over the whole range), so every bend is a change of steepness, never a change of direction. Typical
  member: 56% at 94 rising to almost certain disease. Who: a reversible thallium defect without typical
  angina or an upsloping ST segment (47 of 53), or ST depression > 2 mm.
- **F3, 50 patients: rises, flat, falls, then rises again at the very top.** Here the fitness readout
  switches on before the gate (median 129 mm Hg). From there the risk falls until the gate opens at
  about 180 mm Hg (typical member: 10% at 129 falls to 0.9% at 180, then 4% at 200). Who: middle-aged
  (45–63) with little ST depression (< 0.8 mm). This captures 41 of 50, but the rule also takes in 36
  patients from other families.
- **F4, 36 patients: a short rise, then a long fall.** The fitness readout is on from about 100 mm Hg and
  the discount ends at about 105, so the risk falls from 105 to 189 mm Hg (typical member 13% to under
  0.1%). The gate opens only at 189 mm Hg, where the formula's constant plus the upsloping-ST term puts
  it for patients with no other gate inputs (159.4 + 29.3 = 188.7). Who: younger than 44.5 with no
  exercise angina (30 of 33).
- **F5, 14 patients: rises, flat, falls.** Typical angina keeps the gate shut in the observed range (it
  adds 113 mm Hg to the gate's threshold), so after the flat stretch only the fitness readout acts: the risk falls above about
  164 mm Hg. Who: typical angina, younger than 63.5, peak heart rate below 174.5 (10 of 11). The family is
  small.

**Where the shapes rest on little data.** Only 13 of the 242 training patients have a resting BP above
160 mm Hg and only 2 above 180. Only 41 are below 113 mm Hg. So:

- The first rise (94 to 113 mm Hg) is common to every patient but sits where 41 patients are. It agrees
  in direction with the crude data: 34% disease below 113 mm Hg against 42–48% from 113 to 160.
- The long falls of F3, F4 and F5, and the late rises at 180–200 mm Hg, lie mostly where there are almost
  no patients. In F4 only 1 member is near the 189 mm Hg bend (flagged thin); 30 of its 36 members sit on
  the falling stretch, but almost all of them between 105 and 145 mm Hg. F5's fall above 164 mm Hg holds
  2 of its 14 members.
- Within families the observed rates are too small to show or refute a bend. F4 has 2 of 19 with disease
  at ≤ 120 mm Hg and 0 of 17 above, which neither contradicts the falling curve nor supports it. F3's
  rates jump about (27%, 38%, 12%, 0%, 33% across the five bands). The model's mean predicted risk tracks
  the observed rates band by band in every family, but its curves hold the patient fixed and the band
  rates do not.
- The model's claim that higher blood pressure *lowers* the risk for younger, fitter patients (F3, F4, F5:
  100 patients, 41%) runs against clinical expectation. The v4 record already flags it as "worth
  checking". The dossier shows where that claim comes from (unit 4037 and its threshold formula) and that it
  is a statement about 130–200 mm Hg, where the data is thin.

**The panel's curve.** The line plot Mike was looking at (others at the training median) is rises → flat
→ (a 0.005 log-odds/mm Hg dip from 2445 at 131) → rises steeply after 148 mm Hg. There the gate (855 on at
148.3) and the fitness readout (4037 on at 148.6) switch almost together, and the gate's +0.128 beats the
readout's −0.071. That curve is for a patient with *no* chest-pain type, *no* resting-ECG category and
*no* ST slope (the median of each one-hot column is 0), so no real patient has this curve. The families are the shapes real patients have.

**Unresolved.** The trees are readable approximations (the multi-class tree is 63% accurate on training and
59% in cross-validation). Exact membership is set by the threshold formulas in the units table, and those
are linear in age, peak heart rate, ST depression and the flags. A two-variable picture (age against peak
heart rate, coloured by family) would probably describe F3/F4 better than any axis-aligned tree.
