# Binary one-step units: linear on the data (Experiment 2)

Branch `analysis/binary-step-linear`. Code: `explaneat/analysis/plain_language_linearity.py`.
Recomputation: `experiments/10seed/linearity_v2.py` → `experiments/10seed/linearity_v2/<label>.json`
and `summary.json`. The frozen v4 files are untouched.

## The observation

Many hidden units in the Experiment 2 networks read a single yes/no input: one level of a
one-hot category (a state, an occupation, a country) or a binary column such as international
plan. Their ReLU does switch on the data: it is off for one answer and on for the other. But the
input only ever takes those two values, so on the data the unit has exactly two outputs, one per
answer. Two points always lie on a straight line, so the unit is nothing more than an
adjustment to that input's fixed effect, plus a constant. The old classifier counted every
switching unit as non-linear, so the parts holding these units were labelled non-linear and their
inputs "mixed", although they bend nothing on the data.

## The rule

A hidden unit is **binary one-step** (`binary_step`) when all of these hold:

1. it is a ReLU that switches on the training rows (on for some rows, off for others);
2. its direct inputs are exactly one node, and that node is an input node (no hidden units, no
   second input);
3. that input's data column takes exactly two distinct values on the training rows;
4. the numerical check passes (below).

Such a unit counts as linear on the data, like a unit that is always on. An input whose paths go
only through linear, always-on or binary one-step units is `support_linear` ("a fixed rate across
the observed data"). A part whose hidden units are all identity, always on, binary one-step or dead
is `support_linear` too. I kept the existing class name, so every renderer and summary that already
handles "linear on the data" works unchanged. The reason is still visible: the claim's evidence
carries `binary_step_units: n` (variables, parts, model), and the layer's graph record lists each
unit under `linearity.binary_step` with its input, its two outputs and its check.

Not reclassified (they stay `switching`):

- units downstream of a binary one-step unit (e.g. adult-s0's `wife_chain`, 5009 → 1579);
- units with several inputs, even when all of them are levels of one one-hot category. These are
  also exact on the data by the same argument (one output per level), so they are listed
  separately below, for Mike to decide.

## The numerical check

For each candidate unit, the code fits its activation as `α + β·x` (x = the input column) by
least squares on every training row. The largest error must be below 1e-9, or the unit is not
reclassified. The check is recorded per unit (`max_affine_error`, `passed`). Over the 30 networks
no check failed, and the largest error was 3.8e-11.

The slope check behind each input's class changes for these columns. A finite-difference step at
the data point would read the unit's local slope (0 or its weight), not its effect, so for a
column that feeds a binary one-step unit the slope is now the change in the score between the
column's two values, divided by their difference: the per-level effect, on every training row.
The coefficient this check compares against is the linear-path coefficient, with the unit's
contribution included at its slope on the data (its output change over its pre-activation change
between the two values). The direct edge and the unit's jump add up.

The linearised model does not change, by construction. The linearisation replaces every switching
unit by its least-squares affine fit, and a binary one-step unit's fit is already exact. The code
checks this (`binary_step_linearisation_max_diff`: the largest change in the linearised score on
the training rows if these units were replaced as well). It is at most 3.1e-11 over the 30
networks, and the linearised train and test AUCs are identical before and after in every network.

Two small fixes for two-valued numbers (not one-hot categories): the per-unit text gives the change
between the two values, and the typical-case risk change runs between the two values instead of
the 10th and 90th percentiles. The percentiles can coincide for a yes/no column, which would give
no change.

## Reproduction first

With the new kind switched off (`linearity_claims(..., binary_step=False)`), the recomputation
reproduces the frozen v4 linearity claims in all 30 networks: the same variable classes, part
classes and unit kinds, and the same linear share (to 1e-5). The "before" numbers below are the v4
claims.

## Counts

E/S/M/N = exact linear / linear on the data (support) / mixed / non-linear.
"Linear share" = the model's share of score variation carried by the linear (fixed-rate) paths.

| study | units reclassified | variables E/S/M/N before → after | parts E/S/M/N before → after | linear share, mean (median) before → after |
|---|---|---|---|---|
| adult (10) | 32 | 70/7/58/4 → 70/15/50/4 | 16/13/0/65 → 16/31/0/47 | 71.3% (80.3%) → 72.8% (80.5%) |
| churn100 (10) | 27 | 75/5/37/31 → 75/8/36/29 | 14/5/0/65 → 14/13/0/57 | 39.4% (34.2%) → 43.2% (40.2%) |
| heart100 (10) | 11 | 41/1/61/20 → 41/2/61/19 | 9/1/0/50 → 9/6/0/45 | 64.4% (66.2%) → 64.4% (66.3%) |
| all 30 | 70 | 186/13/156/55 → 186/25/147/52 | 39/19/0/180 → 39/50/0/149 | |

31 parts (18 adult, 8 churn100, 5 heart100) go from non-linear to linear on the data. These are
the candidates to fold into the linear-factor analysis rather than annotate as non-linear parts.
13 variables change class: 12 become linear on the data, and heart100-s8's `thal` goes from
non-linear to mixed (its binary one-step unit is now a fixed-rate path next to a switching one).

The linear share is the variance of the fixed-rate part of the score over (its variance + the
variance of the rest). Moving a unit into the fixed-rate part usually raises it, but it can lower
it slightly when the moved term cancels part of the fixed-rate score (adult-s7, churn100-s3,
heart100-s3, heart100-s8). The largest gains come where yes/no gates carry a lot of the score:
churn100-s9 (23.0% → 42.7%, international plan), churn100-s7 (8.1% → 24.1%, international plan)
and adult-s8 (63.2% → 71.0%).

### Per network

| network | units reclassified | variables E/S/M/N before → after | parts E/S/M/N before → after | linear share before → after | parts now linear on the data |
|---|---|---|---|---|---|
| adult-s0 | 2 | 7/2/4/1 → 7/3/3/1 | 2/3/0/5 → 2/5/0/3 | 97.9% → 98.1% | workclass_missing_pair, exec_managerial_hinge |
| adult-s1 | 4 | 9/1/4/0 → 9/2/3/0 | 2/0/0/5 → 2/2/0/3 | 82.8% → 83.2% | household, origin |
| adult-s2 | 1 | 6/0/8/0 → 6/1/7/0 | 1/2/0/5 → 1/2/0/5 | 48.5% → 48.6% | – |
| adult-s3 | 6 | 6/1/6/1 → 6/3/4/1 | 1/2/0/9 → 1/8/0/3 | 10.4% → 13.4% | us_born_offset, scotland_gate, el_salvador_gate, federal_gov_gate, other_relative_gate, unmarried_gate |
| adult-s4 | 2 | 9/1/4/0 → 9/1/4/0 | 4/3/0/7 → 4/5/0/5 | 92.3% → 93.1% | clerical, occupation |
| adult-s5 | 2 | 6/1/6/1 → 6/1/6/1 | 1/1/0/7 → 1/1/0/7 | 77.9% → 77.9% | – |
| adult-s6 | 2 | 9/0/4/1 → 9/0/4/1 | 2/0/0/6 → 2/1/0/5 | 66.9% → 66.9% | china_gate |
| adult-s7 | 5 | 7/0/7/0 → 7/1/6/0 | 1/2/0/9 → 1/4/0/7 | 83.5% → 83.4% | other_service_unit, hungary_unit |
| adult-s8 | 4 | 5/0/8/0 → 5/1/7/0 | 1/0/0/6 → 1/1/0/5 | 63.2% → 71.0% | marital_switches |
| adult-s9 | 4 | 6/1/7/0 → 6/2/6/0 | 1/0/0/6 → 1/2/0/4 | 89.2% → 92.3% | never_married_chain, category_offset_units |
| churn100-s0 | 3 | 4/2/5/4 → 4/3/4/4 | 1/0/0/7 → 1/0/0/7 | 31.0% → 31.8% | – |
| churn100-s1 | 3 | 4/1/5/4 → 4/1/5/4 | 1/0/0/5 → 1/1/0/4 | 31.7% → 31.7% | single_state_hinges |
| churn100-s2 | 3 | 9/1/4/1 → 9/1/4/1 | 1/1/0/6 → 1/1/0/6 | 59.5% → 59.5% | – |
| churn100-s3 | 2 | 8/0/3/5 → 8/0/3/5 | 2/0/0/12 → 2/2/0/10 | 29.0% → 27.9% | state2_gate, state25_gate |
| churn100-s4 | 3 | 7/1/3/3 → 7/1/3/3 | 2/1/0/7 → 2/1/0/7 | 58.4% → 59.9% | – |
| churn100-s5 | 2 | 11/0/3/2 → 11/0/3/2 | 1/0/0/4 → 1/1/0/3 | 55.7% → 56.3% | state_offsets |
| churn100-s6 | 3 | 8/0/4/2 → 8/0/4/2 | 1/0/0/7 → 1/0/0/7 | 36.7% → 37.7% | – |
| churn100-s7 | 5 | 6/0/4/4 → 6/1/4/3 | 1/1/0/6 → 1/3/0/4 | 8.1% → 24.1% | intlPlanGate, stateGates |
| churn100-s8 | 0 | 11/0/4/2 → 11/0/4/2 | 2/2/0/5 → 2/2/0/5 | 60.7% → 60.7% | – |
| churn100-s9 | 3 | 7/0/2/4 → 7/1/2/3 | 2/0/0/6 → 2/2/0/4 | 23.0% → 42.7% | state_gates, intl_plan_gate |
| heart100-s0 | 1 | 5/0/6/1 → 5/0/6/1 | 1/0/0/5 → 1/0/0/5 | 78.6% → 78.6% | – |
| heart100-s1 | 0 | 5/1/4/1 → 5/1/4/1 | 1/1/0/2 → 1/1/0/2 | 86.7% → 86.7% | – |
| heart100-s2 | 1 | 3/0/5/3 → 3/0/5/3 | 1/0/0/6 → 1/0/0/6 | 34.1% → 34.1% | – |
| heart100-s3 | 1 | 4/0/8/1 → 4/0/8/1 | 1/0/0/7 → 1/1/0/6 | 88.3% → 87.7% | cp2_offset |
| heart100-s4 | 1 | 4/0/7/2 → 4/0/7/2 | 1/0/0/5 → 1/0/0/5 | 85.2% → 85.2% | – |
| heart100-s5 | 1 | 6/0/6/1 → 6/0/6/1 | 1/0/0/5 → 1/1/0/4 | 56.3% → 56.4% | normal_sugar_flag |
| heart100-s6 | 1 | 4/0/5/2 → 4/0/5/2 | 1/0/0/6 → 1/1/0/5 | 76.1% → 76.1% | thal4Unit |
| heart100-s7 | 3 | 3/0/6/4 → 3/1/5/4 | 0/0/0/7 → 0/1/0/6 | 55.8% → 56.4% | direct_terms |
| heart100-s8 | 1 | 3/0/6/4 → 3/0/7/3 | 1/0/0/4 → 1/1/0/3 | 29.7% → 29.5% | thal4_hinge |
| heart100-s9 | 1 | 4/0/8/1 → 4/0/8/1 | 1/0/0/3 → 1/0/0/3 | 53.0% → 53.0% | – |

## Example: adult-s0

Before (v4): workclass and relationship mixed, sex non-linear, 5 of 10 parts non-linear.
After: 2 units reclassified, and workclass becomes linear on the data. 3 of 10 parts stay
non-linear: `capital_gain_hinge` and `female_gate` (unit 369 reads occupation:9, sex and a
capital-gain unit) are non-linear on the data. `wife_chain` (5009 → 1579, relationship: wife) is
affine on the data but stays non-linear under this rule, because 1579 is downstream of a unit.
The linear share goes from 97.9% to 98.1%, and the linearised AUC is unchanged (0.9051 train).

**Executive or managerial (occupation:4), part `exec_managerial_hinge`.** The input is split into
two copies. One goes straight to the score (through the identity node): −0.52 per level. The
other feeds unit 2058 = ReLU(0.575·z − 0.439), which is 0 for everyone else and 1.088 for
executives/managers (4,853 of 39,073 training rows). It is read out with weight −0.270, so it
adds −0.29 for executives/managers. Together, being an executive or manager lowers the score
(log-odds of an income of $50,000 or less) by **0.82**, and this is the same on every one of the
39,073 training rows (the per-row change from 0 to 1 is −0.816312 on all of them). Before, the
classifier only counted the direct −0.52 and called the part non-linear. Now the part is linear
on the data (affine error 3.7e-13).

**Workclass missing (workclass:0), part `workclass_missing_pair`.** Unit 4158 = ReLU(1.056·z −
2.04) is on only for workclass "missing (?)" (output 2.26) and adds −0.83 to the score there.
With it, every workclass path is linear on the data, so workclass is linear on the data, and so
is the part.

## Also exact by the same argument, not reclassified (Mike to decide)

Units with two or more inputs, all levels of one one-hot category, exactly affine on the training
rows (max error < 1e-9). On the data at most one level is on per row, so the unit has one value
per level: the same argument as above, but with several inputs. There are 10 units in 8 networks (churn100-s1 3854_a and 3854_b are two split copies of one unit):

| network | unit | category | columns read | max affine error |
|---|---|---|---|---|
| adult-s1 | 4030 | occupation | occupation:2, occupation:3 | 1.5e-11 |
| adult-s5 | 2134 | native-country | native-country:21, native-country:37 | 2.3e-11 |
| adult-s5 | 2391 | native-country | native-country:23, native-country:27, native-country:30, native-country:34 | 4.3e-11 |
| adult-s8 | 2342 | native-country | native-country:27, native-country:3 | 3.4e-12 |
| churn100-s1 | 3854_a | state | state:32, state:44 | 2.0e-15 |
| churn100-s1 | 3854_b | state | state:32, state:44 | 2.0e-15 |
| churn100-s2 | 161 | state | state:32, state:4 | 2.5e-13 |
| churn100-s3 | 5143 | state | state:22, state:38 | 8.6e-14 |
| churn100-s4 | 1529 | state | state:41, state:44 | 6.5e-14 |
| churn100-s6 | 10379 | state | state:0, state:22, state:46 | 4.3e-13 |

Other switching units that are still exactly affine on the data in the columns that reach them
(mostly units downstream of a binary one-step or always-on unit fed by the same levels). They are
not reclassified and are listed for review only. There are 24; 7 of them depend on a single yes/no
column through a chain (adult-s0 1579, adult-s1 16792, adult-s5 5559, adult-s7 10878 and 771,
heart100-s0 3280 and 4331), and these account for the difference from the earlier scratch count
of 77 single-binary-input units, which went by the inputs reaching a unit rather than its direct
input. The full list is in each `linearity_v2/<label>.json` under
`other_exact_affine_not_reclassified`.

## Caveat

This holds only on the data's two values. The unit is a step, not a line: an input value the
data never shows (a level code between 0 and 1, or a combination where two levels of one category
are on at once) would land somewhere else. For one-hot and yes/no inputs those values cannot
occur, so on every possible case the unit is exactly an adjustment to the input's effect. The
rule makes no claim about anything else the unit could do. It uses the training rows only, and
nothing was selected on test.
