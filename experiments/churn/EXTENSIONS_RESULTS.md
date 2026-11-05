# Churn extensions: results

The design and amendments are in `EXTENSIONS_PLAN.md`, all fixed before the real runs. The raw
data is in `validation.csv`, `repair.csv` and `transplant.csv`. Fine-tuning used masked backprop
(genome weights only). Results for E = 500 epochs are primary; E = 50 is secondary and is in the
CSVs. Test AUC CIs are paired bootstrap intervals (10,000 resamples, seed 42) on the 1,000-row
churn test split. Deltas are paired differences.

## A. Do the named features hold on unseen data?

**Yes, the nonlinear features do. The state offsets do not.**

- **Nonlinear features:** all **23 of 23** named nonlinear features with train restore gain
  ≥ 0.01 have a test restore gain whose 95% CI lies entirely above 0.
  - Median gain is +0.038 on train and +0.044 on test.
  - These are the service-calls × usage gates, the heavy-usage hinges and the
    voicemail-gated hinges.
- **Per seed:** the gap between each network and its own all-linear version is +0.058 to
  +0.096 test AUC, against +0.054 to +0.085 on train.
- **State offsets** flagged as rare-category switches: of the **46 large** ones (|effect| ≥ 1
  logit), only **18 (39%)** have the right sign on test, and 15 fall inside the test-set 95%
  interval. That is no better than chance. Caveat: each state has only 11–36 test rows.

## B. Explanation-guided repair (10 churn100 seeds)

| Condition | Median test AUC | Δ vs ft-only, median [min, max] | Seeds with Δ > 0 | Connections |
|---|---|---|---|---|
| base | 0.886 | — | — | 82 |
| ft-only (E = 500) | 0.883 | 0 | — | 82 |
| **prune all state edges + ft** | **0.895** | **+0.0085** [+0.0017, +0.0140] | **10/10** | **38** |
| prune flagged states + ft | 0.892 | +0.0073 [+0.0022, +0.0155] | 10/10 | 70 |
| prune all state edges, no ft | 0.894 | +0.0110 [−0.0021, +0.0177] | 9/10 | 38 |

- **Repair works:** removing the state structure that the explanations (and A) identified as
  overfit improves test AUC on **every** seed, while about halving the network.
  - 10 of 10 positive has a sign-test p ≈ 0.002.
  - Individual seeds' CIs mostly cross 0, because the shift is small.
- **The control fails:** fine-tuning without the edit slightly *lowers* test AUC.

## C. Feature transplant (recipients s9, s4; selection fixed on train before evaluation)

**Donors:**
- **Original donor:** the largest-gain leaf is s2's `csc_usage_gate` (service calls × low usage).
  The recipients already have a similar mechanism.
- **Complementary donors** (amendment 2, with amendment 3's one-hot collapse):
  - for s9: s6 `day_charge_no_voicemail`;
  - for s4: s7 `usagePathway`.

Δ is the test-AUC change; E = 500.

| Recipient | Donor | Condition | Δ vs ft-only [95% CI] | Δ vs matching sham [95% CI] |
|---|---|---|---|---|
| s9 | s2 gate | graft, no ft | −0.075 [−0.108, −0.042] | −0.043 [−0.076, −0.009] |
| s9 | s2 gate | graft + ft | +0.010 [+0.000, +0.020] | +0.005 [−0.007, +0.017] |
| s9 | s2 gate | **graft + refit + ft** | **+0.013 [+0.003, +0.024]** | **+0.012 [+0.003, +0.021]** |
| s4 | s2 gate | graft, no ft | −0.049 [−0.077, −0.019] | (sham collapses to 0.62) |
| s4 | s2 gate | graft + ft | +0.013 [+0.002, +0.025] | +0.006 [−0.005, +0.016] |
| s4 | s2 gate | **graft + refit + ft** | **+0.013 [+0.002, +0.025]** | **+0.010 [−0.001, +0.023]** |
| s9 | s6 compl. | graft + refit + ft | +0.012 [+0.002, +0.022] | +0.003 [−0.015, +0.022] |
| s4 | s7 compl. | graft + refit + ft | +0.023 [+0.010, +0.038] | +0.002 [−0.013, +0.016] |

**Reading:**
- **Grafting as-is hurts:** the donor's readout double-counts a mechanism the recipient
  already has.
- **Graft, refit the readout, then fine-tune helps:** with the explained service-call gate
  this improves both recipients by about +0.013 test AUC over fine-tuning alone.
  - It also beats a size-matched random-weight graft by +0.010 to +0.012 (CI excludes 0 for
    s9; borderline for s4).
  - So the *explained function* carries signal beyond the extra capacity.
- **The complementary donors' gains are matched by their shams:** there the gain is
  capacity, not the transplanted function.
- **Evidence is modest:** two recipients, and the prespecified graft + ft (no refit) does not
  clearly beat its sham.

## Not completed

**churn100-masked** (PropNEAT with the phantom-weight mask) crashed twice. Postgres.app refused
connections ("failed to verify trust authentication"). The training loop then silently set the
population fitness to −1000 and carried on (`DatabaseBackpropPopulation._evaluate_population`).
The runs were killed and marked failed (`RUN_INCIDENTS.md`). This is on hold until the
Postgres.app app permissions are fixed. No reported run was affected (checked).
