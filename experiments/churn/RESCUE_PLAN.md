# Study D: knockout-and-rescue with feature aliases

Written 2026-11-05, before any D computation. Requested by Mike: find two features that are
potential **aliases** (hypothesised to capture the same signal in different representations),
knock one out, and test whether transplanting the other rescues the lost function.

## Why

C2 found no feature that a recipient genuinely lacks: donor signals are on average about 90%
linearly reproducible from the recipient's own hidden units. So instead of looking for a gap,
D **creates one**. It removes a recipient's own explained feature, then asks whether an
explained alias from another seed can stand in for it. If the alias rescues the loss and a
sham or a wrong-family feature does not, explained features behave as interchangeable
modules.

## Alias hypotheses

These come from the churn100 explanation records (leaf names, hypotheses, entry features).
They were formed by Claude before any D computation. Each family is one mechanism, built
differently in different seeds.

| Family | Mechanism | Feature (seed, annotation) | Representation |
|---|---|---|---|
| A | service calls × low day usage | s5 `csc_threshold_low_day` | 3 hidden; day usage + calls |
| A | | s8 `csc_lowusage_hinge` | 7 hidden; day + eve usage + calls |
| B | heavy day + evening usage hinge | s4 `day_eve_minutes_hinge` | 2 hidden; day + eve minutes |
| B | | s6 `heavy_day_eve_usage` | 1 hidden; day minutes + eve charge (+ 2 state inputs) |
| C | heavy day usage, only without voicemail | s1 `dayUsageHinges` | 2 hidden; day usage gated by **voice mail plan** |
| C | | s0 `day_charge_no_voicemail` | 1 hidden; day charge gated by **number of vmail messages** |

Family C is the clearest case of "same signal, different representation": one network gates
on the plan flag, the other on the message count.

## Cases (recipient feature knocked out ← alias donor; N = wrong-family control donor)

| Case | Recipient (KO feature) | Alias donor | Wrong-family donor N |
|---|---|---|---|
| A1 | s8 `csc_lowusage_hinge` | s5 `csc_threshold_low_day` | s6 `heavy_day_eve_usage` |
| A2 | s5 `csc_threshold_low_day` | s8 `csc_lowusage_hinge` | s6 `heavy_day_eve_usage` |
| B1 | s4 `day_eve_minutes_hinge` | s6 `heavy_day_eve_usage` | s5 `csc_threshold_low_day` |
| B2 | s6 `heavy_day_eve_usage` | s4 `day_eve_minutes_hinge` | s5 `csc_threshold_low_day` |
| C1 | s1 `dayUsageHinges` | s0 `day_charge_no_voicemail` | s5 `csc_threshold_low_day` |
| C2 | s0 `day_charge_no_voicemail` | s1 `dayUsageHinges` | s5 `csc_threshold_low_day` |

## Train-only confirmation (computed and written to `rescue_confirm.csv` before any test evaluation)

For each case, three checks decide whether it is analysed:
1. **Alias similarity:** Pearson correlation, on train, between the recipient feature's logit
   contribution and the alias donor's logit contribution, each computed with its own genome on
   the same inputs. **Alias confirmed if r ≥ 0.7.**
2. **Different representation:** the two features differ in hidden-node count or in their
   entry base-feature set. This is reported; every listed pair differs by construction.
3. **Knockout cost:** the train AUC drop when the recipient feature is knocked out with no
   fine-tuning. **The case is analysed if the drop is ≥ 0.005.**

Cases that fail checks 1 or 3 are still run and reported, but are excluded from the primary
pooled outcome. No substitute pairs are chosen after seeing these numbers.

## Knockout

Disable every enabled connection of the recipient feature's subgraph: entry → feature,
internal connections, and the feature's readout edges to the output. All operations are on
genome connection keys, resolved through the explanation boundary (study C machinery).

## Conditions per case (masked fine-tune, E = 500; refit = study C's readout refit on train)

1. **base:** the original recipient
2. **KO:** knockout, no fine-tune
3. **KO + ft:** the "relearn on its own" control
4. **KO + alias + refit**, no fine-tune
5. **KO + alias + refit + ft**
6. **KO + sham(alias) + refit + ft:** same topology and entries as the alias, N(0,1)
   weights, seed = recipient seed
7. **KO + wrong-family N + refit + ft**

## Outcomes (test split, paired bootstrap 95% CIs, 10,000 resamples, seed 42)

- **Primary, per case:**
  - Δ(5 − 3), alias rescue against relearning;
  - Δ(5 − 6), alias against sham;
  - Δ(5 − 7), alias against wrong family.
- **Rescue fraction:** (AUC5 − AUC3) / (AUC1 − AUC3), reported when AUC1 − AUC3 > 0.
- **Pooled over confirmed cases:** median deltas and sign tests.
- **Secondary:**
  - Δ(4 − 2), how much the alias restores with only a readout refit;
  - the knockout cost on test, AUC1 − AUC2.
- **Hypothesis:** the alias rescues more than relearning, the sham, or the wrong-family
  feature. Any result is reported.
