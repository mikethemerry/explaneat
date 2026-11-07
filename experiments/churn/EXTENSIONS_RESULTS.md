# Churn extensions: results

> **Thesis scope** (decision of 7 Nov 2026). **In Ch 7:**
> - A: explained features generalise;
> - B: explanation-guided repair of state edges;
> - C2, algorithmic arm only: cross-seed signal reproducibility;
> - D2: knockout-and-rescue with algorithmic aliases;
> - E: re-initialisation;
> - training details;
> - MONK-2 grafting (F2, light; `GRAFT_F_G_RESULTS.md`).
> **Future work, not in the thesis:**
> - C: transplant;
> - C2, judged arm;
> - D and D+: aliases chosen by Claude;
> - F1, F3, G, H.
> Results are kept as recorded.

Each study's design is in its plan file:
- A–C: `EXTENSIONS_PLAN.md`
- C2: `TRANSPLANT2_PLAN.md`
- D: `RESCUE_PLAN.md`
- D+, D2 and E: `RESCUE2_REINIT_PLAN.md`
- early-stopping reruns: `EARLY_STOPPING_PLAN.md`

Every plan and amendment was committed before the runs it covers. Each section names its CSV.

All fine-tuning used masked backprop (genome weights only). For A–C, results for E = 500 epochs are primary; E = 50 is secondary and is in the
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
| prune all state edges, no ft | 0.894 | +0.0110 [−0.0021, +0.0177] | 8/10 | 38 |

- **Repair works:** removing the state structure that the explanations (and A) identified as
  overfit improves test AUC on **every** seed, while about halving the network.
  - 10 of 10 positive has a sign-test p ≈ 0.002.
  - Individual seeds' CIs mostly cross 0, because the shift is small.
- **The control fails:** fine-tuning without the edit slightly *lowers* test AUC.

**B with early stopping** (`repair_es.csv`; the stopping rule is in the last section):

| Condition | Median test AUC | Δ vs ES ft-only: sign, median | Median best epoch |
|---|---|---|---|
| ft-only (ES) | 0.886 | — | 0 (9/10 stop at once) |
| prune all state edges, no ft | 0.894 | +10/−0, +0.0063 (p = 0.002) | — |
| prune all state edges + ft (ES) | 0.894 | +7/−3, +0.0037 (p = 0.34) | 511 |
| prune flagged states + ft (ES) | 0.892 | +8/−2, +0.0067 (p = 0.11) | 0 (6/10 at once) |

- **ES fine-tuning leaves the unedited network unchanged:** it stops at epoch 0, so ES
  ft-only is the base network.
- **Pruning alone beats it on 10/10 seeds.** Fine-tuning after pruning doesn't add reliably.
- **Part of the fixed-epoch result was the control getting worse:** with fixed E = 500,
  some of prune + ft's advantage came from fixed fine-tuning slightly lowering the unedited
  control. Under ES the edit is still positive on every seed, but the margin is smaller.

## C. Feature transplant (recipients s9, s4; selection fixed on train before evaluation)

> **Future work, not in the thesis** (scope decision, 7 Nov 2026). Results kept as recorded; not cited in Ch 7.

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

## C2. Transplanting features whose explained signal the recipient lacks

### Algorithmic arm (`transplant2_selection.csv`)

No donor qualified for any of the 10 recipients.
- Over all candidate pairs, the novelty median is 0.098 and the maximum is 0.355, against a
  threshold of 0.5.
- So every graftable, useful explained feature in any churn100 seed is, on average, about 90%
  linearly reproducible from any other seed's own inputs plus hidden units.

The study found nothing to transplant. That is itself a result: **cross-seed convergence**.
Ten independent runs found the same signals, built differently.

### Judged arm (`transplant2_judged.csv`)

> **Future work, not in the thesis** (scope decision, 7 Nov 2026). Results kept as recorded; not cited in Ch 7.

Claude picked donors from the explanation records alone.

| Recipient ← donor | Δ (graft + refit + ft) vs ft-only [95% CI] | Δ vs sham [95% CI] |
|---|---|---|
| s2 ← s6 `heavy_day_eve_usage` | +0.002 [−0.005, +0.009] | +0.002 [−0.004, +0.009] |
| s3 ← s6 `day_charge_no_voicemail` | +0.008 [−0.001, +0.017] | +0.007 [−0.001, +0.016] |
| s4 ← s5 `csc_threshold_low_day` | +0.004 [−0.014, +0.021] | −0.003 [−0.018, +0.010] |
| s5 ← s6 `day_charge_no_voicemail` | **+0.013 [+0.002, +0.026]** | **+0.016 [+0.003, +0.031]** |
| s8 ← s6 `heavy_day_eve_usage` | +0.003 [−0.001, +0.008] | +0.000 [−0.007, +0.007] |
| s9 ← s5 `csc_threshold_low_day` | −0.002 [−0.023, +0.018] | −0.007 [−0.015, +0.001] |

- **Pooled:** median +0.004 vs ft-only (5/6 positive, sign p = 0.22) and +0.001 vs sham (4/6,
  p = 0.69).
- **Without fine-tuning:** graft + refit alone gives median +0.005 vs ft-only (5/6 positive).
- **Reading:**
  - Choosing a mechanism the recipient's explanation lacks gives small, mostly positive gains,
    but only s5 is clearly above both controls.
  - This matches the algorithmic arm. The "missing" mechanisms are mostly present implicitly,
    so a graft adds little that the recipient couldn't already express.

## D. Knockout and rescue with feature aliases chosen by Claude (`rescue_confirm.csv`, `rescue.csv`)

> **Future work, not in the thesis** (scope decision, 7 Nov 2026). Results kept as recorded; not cited in Ch 7.

Six cases from three alias families. On train:
- families B (r = 0.96) and C (r = 0.99) confirmed as aliases;
- family A (r = 0.44) did not, so A1 and A2 are reported but excluded from the pooled results.

Test AUC, E = 500 masked fine-tuning:

| Case | base | KO | KO + ft | KO + alias + refit | KO + alias + refit + ft | KO + sham + refit + ft | KO + wrong + refit + ft |
|---|---|---|---|---|---|---|---|
| B1 | 0.883 | 0.818 | 0.868 | 0.878 | 0.880 | 0.880 | 0.871 |
| B2 | 0.896 | 0.858 | 0.874 | 0.897 | 0.891 | 0.866 | 0.879 |
| C1 | 0.904 | 0.897 | 0.908 | 0.903 | 0.909 | 0.909 | 0.910 |
| C2 | 0.895 | 0.886 | 0.882 | 0.895 | 0.895 | 0.891 | 0.882 |
| A1 (excluded) | 0.874 | 0.804 | 0.866 | 0.858 | 0.866 | 0.866 | 0.882 |
| A2 (excluded) | 0.881 | 0.855 | 0.878 | 0.865 | 0.894 | 0.896 | 0.895 |

- **The alias alone, with only a readout refit, restores almost all of the knocked-out
  function:** for B1, B2 and C2 it lands within 0.005 of base.
- **After fine-tuning, the controls mostly catch up:** sham and wrong-family grafts plus 500
  epochs reach similar AUCs. Only B2 clearly separates.

That raised the question D+ answers: is the alias better *before* fine-tuning?

## D+. No-fine-tune controls for D (`rescue_plus.csv`; added after D's results, labelled as such)

> **Future work, not in the thesis** (scope decision, 7 Nov 2026). Results kept as recorded; not cited in Ch 7.

| Case | KO + alias + refit | KO + sham + refit | KO + wrong + refit | Δ alias − sham [95% CI] | Δ alias − wrong [95% CI] |
|---|---|---|---|---|---|
| A1 | 0.858 | 0.804 | 0.803 | +0.054 [+0.035, +0.076] | +0.055 [+0.036, +0.076] |
| A2 | 0.865 | 0.856 | 0.856 | +0.009 [−0.002, +0.020] | +0.008 [−0.003, +0.020] |
| B1 | 0.878 | 0.837 | 0.820 | +0.041 [+0.019, +0.064] | +0.058 [+0.030, +0.088] |
| B2 | 0.897 | 0.862 | 0.857 | +0.035 [+0.017, +0.057] | +0.040 [+0.021, +0.063] |
| C1 | 0.903 | 0.895 | 0.896 | +0.008 [+0.001, +0.016] | +0.007 [−0.003, +0.018] |
| C2 | 0.895 | 0.891 | 0.886 | +0.004 [−0.004, +0.013] | +0.009 [−0.002, +0.022] |

With only a readout refit, the alias's explained weights beat both controls in all six cases:
- a same-topology random-weight graft, with the CI excluding 0 in 4/6;
- a wrong-family feature, with the CI excluding 0 in 3/6. **The function is in the weights. The
topology alone doesn't carry it.**

## D2. Knockout and rescue with algorithmic aliases (`rescue2_selection.csv`, `rescue2.csv`)

- **Cases:**
  - 36 ordered alias pairs (r ≥ 0.9, different representation, KO cost ≥ 0.005).
  - They come from 19 recipient features. 35 are runnable; one graft collides with existing
    node ids.
  - The selection was committed before any test evaluation.
- **Median test AUC:**

  | | Median test AUC |
  |---|---|
  | base | 0.895 |
  | KO | 0.848 |
  | KO + ft | 0.878 |
  | KO + alias + refit | 0.892 |
  | KO + alias + refit + ft | 0.890 |
  | KO + sham + refit + ft | 0.884 |
  | KO + wrong + refit + ft | 0.887 |
  | KO + sham + refit | 0.872 |
  | KO + wrong + refit | 0.857 |

Sign tests are over cases; the per-recipient-feature version uses the median over each feature's
aliases:

| Δ | Over 35 cases | Over 19 recipient features |
|---|---|---|
| **alias + refit − sham + refit** | **+0.020, 35/0, p < 1e-9** | +0.020, 19/0 |
| **alias + refit − wrong + refit** | **+0.030, 32/3, p < 1e-6** | +0.032, 18/1 |
| alias + refit + ft − KO + ft | +0.010, 30/5, p < 1e-5 | +0.010, 15/4, p = 0.02 |
| alias + refit + ft − sham + refit + ft | +0.001, 24/11, p = 0.04 | +0.001, 13/6, p = 0.17 |
| alias + refit + ft − wrong + refit + ft | −0.001, 17/18, p = 1.0 | −0.001, 8/11 |
| alias + refit − KO | +0.039, 35/0 | +0.040, 19/0 |

- **Rescue fraction:** (AUC_x − KO) / (base − KO) has median 0.94 with refit only and 0.95 with
  refit + fine-tune.
- **What this shows:**
  - D+ generalises. An algorithmically found alias, wired in with only a readout refit, recovers
    about 94% of the knocked-out function.
  - It beats a random-weight graft of the same topology in 35/35 cases.
  - It beats a wrong-family feature in 32/35.
- **After 500 epochs of fine-tuning the advantage mostly disappears:**
  - Any added capacity plus gradient descent recovers about as much.
  - The alias is still ahead of plain relearning (KO + ft) in 30/35 cases.

### D2 with early stopping (`rescue2_es.csv`; refits on the 3200 fit rows)

| Δ | Over cases | Over recipient features |
|---|---|---|
| alias + refit − sham + refit | +0.019, 35/0 | +0.020, 19/0 |
| alias + refit − wrong + refit | +0.030, 32/3 | +0.032, 18/1 |
| alias + refit + ft − KO + ft | +0.019, 33/2 | +0.018, 18/1 |
| alias + refit + ft − sham + refit + ft | +0.010, 31/4, p < 1e-5 | +0.012, 17/2, p < 0.001 |
| alias + refit + ft − wrong + refit + ft | +0.007, 27/8, p = 0.002 | +0.005, 13/6, p = 0.17 |

Under early stopping, the alias stays ahead of every control after fine-tuning. **This comparison
is not symmetric:**
- **The alias barely fine-tunes:** in 31/35 cases its best validation epoch is 0, so its
  fine-tuned AUC is its refit-only AUC (0.892).
- **The controls do train, and stop early:**
  - KO + ft: median best epoch 409, test 0.869.
  - sham: 221, test 0.877.
  - wrong family: 242, test 0.877.
- **The controls are weaker under ES than at fixed E = 500.** They train on 80% of the rows and
  stop on a noisy 800-row validation set.

So the ES result shows that the alias starts at a validation optimum and the controls don't reach
it under this stopping rule. It does not show that the controls can't reach it at all: at fixed
E = 500 they nearly did.

## E. Re-initialisation consistency: given a topology, how consistent are the weights?

### E2, network level (`reinit_network.csv`, `reinit_network_es.csv`)

Setup: 10 churn100 topologies × 10 N(0,1) re-inits of every weight and bias, then masked
training.

| | Fixed E = 500 | Fixed E = 2000 | Early stopping |
|---|---|---|---|
| Re-init test AUC, median | 0.867 | 0.873 | 0.862 |
| Δ vs evolved weights, median | −0.022 | −0.014 | −0.027 |
| Inits ≥ evolved test AUC | 10/100 | 21/100 | 5/100 |
| Δ train AUC vs evolved, median | −0.021 | −0.011 | — |
| Logit r vs evolved / pairwise | 0.875 / 0.852 | 0.879 / 0.859 | 0.881 / 0.859 |
| Matched-unit weight cosine vs evolved | −0.10 | −0.11 | −0.11 |
| Equivalent (r ≥ 0.99 and cosine ≥ 0.9) | 0/100 | 0/100 | 0/100 |
| Dead hidden units (evolved: 0.05) | 0.24 | 0.25 | 0.25 |

Under ES the re-inits stop at a median of 931 epochs (best epoch 881). ES fine-tuning of the
evolved weights stops at once (best epoch 0 for 9/10), so they are already at a validation
optimum.

**Answers to the plan's questions:**
1. **A topology does not determine the function.**
   - Logit r is about 0.88 on average.
   - Some topologies (s7, s9) are much less consistent under ES: test AUC medians 0.760 and
     0.790, against 0.900 and 0.875 evolved.
2. **Independent inits find functionally similar but different weights.**
   - The matched-unit cosine is about 0, both against the evolved weights and between inits.
   - None of the 300 trained inits is equivalent to the evolved weights.
   - In NEAT topologies, hidden units are almost all distinguishable by connectivity
     (largest symmetry group: 2), so this is not a matching artefact.
3. **Training from scratch rarely matches the evolved network:**
   - 5–21% of inits reach the evolved test AUC, depending on the training budget.
   - The gap is also present on **train** and shrinks with more epochs.
   - So it is mainly an optimisation gap, not a generalisation one.
   - Likely causes:
     - a quarter of units die under N(0,1) init (a weak correlation with the gap, Spearman
       ≈ −0.25, p ≈ 0.02);
     - sparse topologies have little redundancy;
     - the evolved weights were reached incrementally, each structure added to an
       already-working network and trained in that context;
     - the topology was selected together with its weights, as in lottery tickets.
   - These causes have not been tested separately. A fan-in-scaled init rerun would separate
     the init effect from the topology effect.

### E1, feature level (`reinit_feature.csv`, `reinit_feature_es.csv`)

Setup: 35 D2 cases × 10 N(0,1) re-inits of the grafted alias subgraph, then refit and fine-tune.

| | Fixed E = 500 | Early stopping |
|---|---|---|
| Δ test AUC vs alias-weights reference, median | −0.002 | −0.012 |
| Inits ≥ reference | 105/350 | 38/350 |
| Inits with Δ CI entirely below 0 | 58/350 | 185/350 |
| Contribution r vs donor feature / pairwise | 0.20 / 0.18 | 0.37 / 0.23 |
| Weight cosine vs donor | 0.07 | 0.07 |
| Equivalent | 2/350 | 2/350 |

- **Same in both regimes:**
  - re-initialised alias topologies don't recover the donor feature's function (r about
    0.2–0.4) or its weights (cosine about 0);
  - they don't agree with each other.
- **Different between regimes:**
  - At fixed E = 500 the rest of the network compensates and test AUC is about the same.
  - Under ES most re-inits stop early (89/350 at epoch 0) and fall short of the alias-weights
    reference.

**The structure permits the feature, but the explained feature is a specific weight setting
that random starts don't rediscover.**

## Training details that affect these results

### Phantom weights and the connectivity mask

- **Phantom weights:**
  - `NeuralNeat` stores each layer as a dense matrix. Entries with no connection gene start at
    0 (verified: all 1,728 phantom entries in a churn genome).
  - They still receive gradient: ∂L/∂W_ij = (∂L/∂z_j)·a_i is non-zero whenever the *source*
    is active.
  - So they trained during backprop, and `update_genome_weights` then discarded them.
- **The mask:** a fixed W ⊙ M mask (`mask_phantom=True`) keeps them at zero.
- **Settings by study:**
  - All extension fine-tuning (B–E) uses the mask.
  - The evolutionary runs used for the 10-seed experiment, churn100 and churn200 did not, and
    their reproduction runs still don't.
  - Since commit 929fd76, `BackpropPopulation` and `DatabaseBackpropPopulation` default to
    masked, and each experiment records `config_json.training.mask_phantom`.
- **Effect on evolution, tested by rerunning churn100 masked** (`runs_churn100mask.csv`):
  - median test AUC 0.884 vs 0.886 unmasked;
  - slightly sparser networks: median 38 vs 45.5 connected inputs, 17.5 vs 17 hidden units.
  - So the unmasked runs are not materially compromised. With 5 epochs per generation, the
    phantom updates mostly perturbed the discarded entries and the forward pass during those
    epochs.

### Early stopping (`EARLY_STOPPING_PLAN.md`, `val_split.json`)

- **Validation set:** a stratified 20% of train (800 rows, random_state 42), never test.
- **Stopping rule:** patience 50, min_delta 1e-5, maximum 5000 epochs, best weights restored.
- **What it covers:** B, D2 and E.
- **Caveat:** the validation rows were part of the evolved genomes' training data. ES
  fine-tuning of evolved or alias weights therefore tends to stop at once. Re-initialised and
  control networks train on fit rows only and stop on a noisy validation set.

### Incidents (`../10seed/RUN_INCIDENTS.md`)

The first two churn100-masked attempts crashed:
- **What happened:** Postgres.app refused connections ("failed to verify trust
  authentication"). `DatabaseBackpropPopulation._evaluate_population` then silently set
  population fitness to −1000 and carried on.
- **Cleanup:** the runs were renamed `churn100mask-crashed*` and marked failed.
- **The third attempt** completed under a watchdog.
- **Reported runs:** none is affected, checked by logs and by confirming no stored fitness of −1000.
- **Open:** the silent fallback should become a hard failure (pending decision).

## F and G: grafting onto unconverged or wrong recipients, and a graft protection phase

> **Future work, not in the thesis** (scope decision, 7 Nov 2026). Results kept as recorded; not cited in Ch 7.

Full results: `GRAFT_F_G_RESULTS.md`. Plan: `GRAFT_F_G_PLAN.md`.
- **F1 (churn, 60 recipients from generations 0–99, 120 graft cases):**
  - Graft gain rises with the signal's novelty to the recipient: graft + refit − base has
    Spearman ρ = +0.74 [0.62, 0.82], and the partial ρ given generation is +0.42.
  - Gain is ≈0 (within ±0.005) for nov < 0.1.
  - The graft-specific part, over a same-topology random graft after fine-tuning, is small:
    +0.003, 84/36.
- **F2 (MONK-2):** the correct seeds' low-count gate, grafted, refit and fine-tuned, fixes all 7
  wrong networks (test AUC ≥ 0.996, ES). Fine-tuning alone fixes one.
- **F3 (3 seeds, descriptive):** a grafted elite inserted at generation 10 reaches the original
  run's generation-50 fitness first on every seed.
- **G:** freezing the graft for the first 25 or 100 fine-tuning epochs changes test AUC by ≈0
  in every context and regime. The C2, D2 and F1 P = 0 runs reproduce the earlier CSVs exactly.

## H: capacity thresholds along lineages (lineage replay)

> **Future work, not in the thesis** (scope decision, 7 Nov 2026). Results kept as recorded; not cited in Ch 7.

Full results: `../lineage/RESULTS_H.md`. Plan: `../lineage/PLAN_H.md`.

**Parents:** reconstructed by replaying every run in memory. 50 experiments were verified
bit-identical against the DB, then backfilled into `parent1_id`/`parent2_id`.

- **Improvement is discrete:**
  - jump steps (top 5% of lineage Δ) carry 62–84% of each winner-lineage's train-AUC gain;
  - backprop-only (elite) steps carry 3–5%;
  - every non-elite lineage step is structural.
- **The jumps are the initial climb:** after generation 10, structural steps no longer
  predict jumps in churn100 or heart100. churn200's second climb does show it.
- **Leaf signals:**
  - About half emerge step-like.
  - They become decodable from small *partial* realisations (median 1 hidden node), long
    before the final realisation is complete.
  - They rarely coincide with a fitness jump (0–29%).
- **XOR:** a feasible-class nonlinear unit first appears 15 generations (median) before the
  solve, 0/30 coincident.
- **MONK-2:** in 2 of 3 correct seeds, the largest jump is a single input connection added
  into the final `few_indicators_gate` leaf. This is the hypothesis's clean case, but it is
  rare.
