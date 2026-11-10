# Species leads: explaining the near-optimal alternatives in each run

**Exploratory, not pre-registered.**

Thesis Ch 7. Each run's final population is grouped into **post-hoc structural clusters of
near-optimal genomes (not NEAT species)**. The best genome of each cluster (its lead) is
explained with the 10-seed protocol, and the explanations are compared within each run and
across seeds. Datasets: Heart (40 generations, `10seed-heart-s0..s9`) and churn100
(`churn100-s0..s9`).

## Summary

1. **NEAT kept almost no diversity.** 16 of 20 final populations are a single NEAT species. Post-hoc structural clusters of the near-optimal genomes (train AUC within 0.01 of the final best) are just as homogeneous: 15 of 20 runs have one cluster. This gives 27 leads, 14 already explained and 13 new.
2. **All 13 new leads were explained to full coverage.** Every function was preserved to ≤ 1.2e-15. The 13 took 126 min in total (median 7.9 min for Heart, 10.2 min for churn), against an estimate of 180 min. Each has a lint-clean v5 plain-language draft, waiting for Mike's review.
3. **Within a run, the leads are variations on one design.** Churn100 leads share their useful nonlinear signals (the service-call threshold and the day-usage hinges). Heart leads share a linear core plus small corrections, which sometimes differ between clusters (Heart s2) and are sometimes shared (Heart s5).
4. **Heart's nonlinear corrections are small and test at chance.** None is useful at the 0.02 restore-gain threshold. By the stricter residual match they are mostly specific to one seed, and 20 of 23 are at chance on test. Churn's 22 useful signals all hold on test, and most of them recur across seeds.
5. **The inventory did not build a better churn network.** No lead lacked a required signal, so there was nothing to graft. Pruning "rare" signals hurt, which shows rare does not mean overfit. One of 8 prune-only candidates passed validation, and its test AUC was unchanged.
6. **Within-run agreement is weaker evidence than cross-seed agreement.** Leads of one run share ancestry. Heart s5's leads all share an ischaemia gate that no other seed's best genome reproduces.

## 1. Units and leads

`STEP1_SPECIES.md` recovered NEAT's species by exact replay. All 20 replays were bit-identical
to the DB: Heart 10,000 genomes per run and churn100 25,000 per run, every generation, checked
by key, fitness and md5. That replay showed NEAT's species carry almost no information here:
16 runs ended with one species and 4 with two.

Mike therefore changed the unit (`STEP1B_CLUSTERS.md`):
- **Members:** the final-generation genomes with train AUC ≥ the final best − 0.01.
- **Distance:** NEAT compatibility distance with each run's own coefficients.
- **Clusters:** average-linkage agglomerative clustering, cut at half the compatibility threshold.
- **Leads:** the best train AUC per cluster, ties to the lowest key, at most 5 per run.

Leads come from the final generation only. In 6 runs (Heart s1, s2, s4, s5; churn100 s2, s7) the
run's best genome is from an earlier generation. It is already explained in the 10-seed study,
and it is set beside the leads below for comparison, but it is never a lead.

| Dataset | Runs with 1 / 2 / 3 clusters | Leads | Reused 10-seed records | New explanations |
|---|---|---|---|---|
| Heart | 8 / 0 / 2 (s2, s5) | 14 | 6 | 8 |
| churn100 | 8 / 1 (s2) / 1 (s4) | 13 | 8 | 5 |

## 2. Explanations of the 13 new leads

The protocol was the same as in the 10-seed study (`experiments/10seed/EXPLAIN_PROTOCOL.md` v3):
one agent per genome, a 3 h box, the train split only, and function-preserving refactoring only.
`verify_record.py` checked each result, as `experiments/10seed/verify_record.py` does, and
`recheck_records.py` re-checked all 13 against the live DB after every agent had finished
(`records_recheck.csv`). All 13 reach full coverage: structural and compositional coverage 1.0
and a complete hierarchy. Their operations are unchanged since verification. Heart's 10-seed
records used v2 tooling; these use v3.

| Lead | Key | Cluster (size) | Train AUC | Test AUC (report only) | Hidden | Leaves / compositions | Splits / identity | Function preserved | Closed-form err | Minutes |
|---|---|---|---|---|---|---|---|---|---|---|
| heart-s1-c1 | 9674 | 1 (140) | 0.9392 | 0.8610 | 7 | 5 / 1 | 3 / 0 | 3.3e-16 | 0.005 | 8.0 |
| heart-s2-c1 | 9083 | 1 (88) | 0.9432 | 0.8966 | 6 | 3 / 1 | 5 / 0 | 2.2e-16 | 0.013 | 6.9 |
| heart-s2-c2 | 9685 | 2 (75) | 0.9403 | 0.9009 | 4 | 2 / 1 | 2 / 0 | 2.2e-16 | 0.005 | 5.9 |
| heart-s2-c3 | 9600 | 3 (3) | 0.9381 | 0.8987 | 3 | 3 / 1 | 4 / 0 | 2.2e-16 | 0.007 | 7.0 |
| heart-s4-c1 | 9679 | 1 (175) | 0.9423 | 0.8858 | 4 | 4 / 1 | 5 / 0 | 3.3e-16 | 0.002 | 9.2 |
| heart-s5-c1 | 9259 | 1 (173) | 0.9418 | 0.8869 | 3 | 3 / 1 | 4 / 2 | 2.2e-16 | 0.011 | 9.4 |
| heart-s5-c2 | 9858 | 2 (2) | 0.9389 | 0.8922 | 5 | 5 / 1 | 5 / 2 | 2.2e-16 | 0.017 | 10.4 |
| heart-s5-c3 | 9698 | 3 (1) | 0.9376 | 0.9084 | 4 | 2 / 1 | 1 / 0 | 2.2e-16 | 0.021 | 7.9 |
| churn100-s2-c1 | 24636 | 1 (41) | 0.9009 | 0.8756 | 22 | 8 / 2 | 17 / 2 | 6.7e-16 | 0.031 | 18.6 |
| churn100-s2-c2 | 24564 | 2 (5) | 0.8944 | 0.8883 | 25 | 7 / 2 | 13 / 2 | 5.6e-16 | 0.043 | 10.2 |
| churn100-s4-c2 | 24427 | 2 (6) | 0.8908 | 0.8664 | 13 | 7 / 2 | 5 / 2 | 4.4e-16 | 0.020 | 8.6 |
| churn100-s4-c3 | 24445 | 3 (1) | 0.8884 | 0.8671 | 21 | 5 / 2 | 8 / 2 | 4.4e-16 | 0.028 | 8.9 |
| churn100-s7-c1 | 20708 | 1 (59) | 0.9189 | 0.9007 | 14 | 5 / 2 | 5 / 2 | 1.2e-15 | 0.038 | 15.0 |

**What they compute.** The agents' summaries are in the records.
- **Heart:**
  - Every lead is essentially logistic regression: 13 to 19 direct edges, about 0.93 train AUC on their own. On top sit one to three small ReLU corrections, each worth ≤ 0.01 train AUC.
  - In Heart s2 the three clusters use different corrections. c1, the retrained copy of the 10-seed s2 genome, has an age/heart-rate-reserve gate, a flat-slope gate and oldpeak hinges. c2 and c3 have gates keyed on non-anginal chest pain and maximum heart rate.
  - In Heart s5 all three leads carry the same ST-depression ischaemia gate. c1 and c2 also share the resting-BP offset gate behind it. They differ in the small extra terms (an age gate, an LVH indicator, a damping node).
- **churn100:**
  - Every lead is a sparse linear backbone plus state offsets. Its gain over linear (all-linearised ≈ 0.83 against ≈ 0.89–0.92 for the full network) comes from two thresholds: a customer-service-call threshold, nothing for 0–2 calls and then steep, and day-usage hinges.
  - The leads differ in how these two are gated, by voicemail, evening minutes or individual states, and in their state units.
  - churn100-s7-c1 has the same three-part structure as the 10-seed s7 genome, of which it is the retrained copy.

**Plain-language layers.** Each lead has a v5 read-down draft in `plain_language/<label>.json`.
All 13 are lint-clean, `drafted_by: ai-assisted`, `verified_by: null`. The "worth checking"
flags repeat the 10-seed wording where they apply: high fasting blood sugar lowers risk,
typical angina has the lowest risk, a fixed thallium defect lowers risk, and for churn the
state effects and "voice-mail plan lowers, messages raise". Two new flags appear, both from
heart-s5-c2: "higher resting blood pressure lowers risk in one part" and "undecoded thallium
code 4 shuts the ST-depression signal".

**Verification pack** (`verification/`): one certificate PDF per lead, a combined
`review.pdf` and the combined plain-language review `plain_language/review.pdf`.
`RECORDING.txt` has the verdict commands, which write to the species-leads records. **Mike has
verified 0 of the 13.**

## 3. Signal inventory (`signal_inventory.py`, `signals.csv`, `signal_matches.csv`)

**Method.** I used the cross-seed reproducibility method of churn transplant study C2, with the
same thresholds, generalised to every leaf.
- **Signal:** a nonlinear leaf's exit activation(s) on train.
- **Match:** genome j has the signal if nov = 1 − R² of an OLS fit of the signal on genome j's
  inputs plus all its hidden activations is < 0.5.
- **Useful:** train restore gain ≥ 0.02.
- **Classes within a set:** required = found in every genome, rare = only in its own, common =
  in between.
- **Sets:** the leads of a run, and the 10 seeds' best genomes from the 10-seed records.

On these networks the C2 match turned out to be nearly vacuous. Most exits are well
reproduced by the dataset's inputs alone, so 53 of 62 cross-seed signals are "required". I
therefore also report a stricter **residual match**, added for this study. It matches the part
of each exit that the inputs cannot give linearly (the exit minus its least-squares fit on the
inputs), and only for signals with at least 5% nonlinear variance. The tables give both.

**Test check** (reported only, never used to select): the test restore gain, and the test AUC of
the signal's own logit contribution, each with a bootstrap CI (seed 42). A signal is "at
chance" if that AUC's CI contains 0.5.

### Cross-seed inventory: the 10 seeds' best genomes

| Dataset | Nonlinear signals | Useful (gain ≥ 0.02) | C2 match: required / common / rare | Residual match: required / common / rare / linear | At chance on test |
|---|---|---|---|---|---|
| Heart | 23 | **0** | 21 / 2 / 0 | 0 / 8 / 12 / 3 | 20 of 23 |
| churn100 | 39 | 22 | 32 / 7 / 0 (useful 18 / 4 / 0) | 7 / 22 / 7 / 3 (useful 6 / 12 / 4) | 3 of 39; **0 of 22 useful** |

- **Heart:** the corrections are small (largest train restore gain 0.010) and mostly at chance on
  test. By the residual match most are specific to one seed (median: found in 0 of the 9 other
  seeds). What recurs across Heart seeds is the linear core.
- **churn100:** each useful signal has a positive test restore gain whose CI excludes 0
  (+0.023 to +0.075; the leads' useful signals likewise, up to +0.077). By the residual match the useful signals are found in a median of 6 of
  the 9 other seeds. The required ones are the service-call thresholds and the
  day/evening-usage hinges, as in the C2 transplant study's finding that churn signals recur
  across seeds.

### Within-run inventory: runs with more than one lead

| Run (leads) | Nonlinear signals | Useful | C2 match | Residual match: required / common / rare / linear | At chance on test |
|---|---|---|---|---|---|
| Heart s2 (c1, c2, c3) | 7 | 0 | all required | 0 / 2 / 5 / 0 | 7 of 7 |
| Heart s5 (c1, c2, c3) | 9 | 0 | all required | 6 / 0 / 3 / 0 | 7 of 9 |
| churn100 s2 (c1, c2) | 8 | 2 | all required | 5 / 0 / 2 / 1 (useful: 2 required) | 3 of 8; 0 useful |
| churn100 s4 (c1, c2, c3) | 12 | 7 | 11 required, 1 rare | 7 / 2 / 2 / 1 (useful: 6 required, 1 rare) | 3 of 12; 0 useful |

- **Heart s5:** all three leads carry the ischaemia gate, and two also carry the BP-offset gate.
  These are required within the run, yet no other seed's best genome reproduces their residual
  (0 of 9). That is agreement from shared ancestry, not from the data. Heart s2's three
  clusters do differ in their corrections, so there the clusters found different small fixes.
- **churn100 s4:** six of the seven useful signals are required across its three leads, and
  also found in 7 to 9 of the other 9 seeds. These are the service-call threshold and the
  day-usage hinges. The one rare useful signal is c3's three-stage `day_usage_hinges`, found in
  only 2 other seeds; its test restore gain is +0.050. In churn100 s2 the useful
  service-call × low-day-usage gate is required within the run.
- **Earlier-generation best (comparison only).** The 10-seed explained genome, added to its
  run's leads, agrees with them about as well as they agree with each other. For example, in
  Heart s4 and churn100 s7 every residual signal is required. The final-generation leads are
  therefore not a different design from the explained run best.

**Caveat.** Within-run agreement is weaker evidence than cross-seed agreement. Leads of one run
descend from shared ancestors, and the clusters show they are near-identical in structure.
Agreement inside a run can therefore come from descent alone, as in Heart s5. The cross-seed
inventory is the one to cite for "signals recur".

## 4. Constructive check, churn100 only (`constructive_check.py`, `constructive_check.csv`)

**Setup.**
- **Split:** the committed churn validation split, a stratified 20% of train (`val_split.json`).
  Fits use the other 80%.
- **Candidate:** each seed gets at most one, built on its cluster-1 lead:
  - prune the nonlinear leaves that are rare across seeds by the residual match, or overfit
    (validation restore gain ≤ 0). Pruning replaces the leaf's contribution by its linear fit,
    so the nonlinearity goes and the linear part stays;
  - graft at most one missing required signal, with a readout refit.
- **Acceptance:** on validation AUC against the unmodified lead.
- **Test:** computed once, at the end.

**Revision.** The first run pruned by disabling readouts, which deleted the linear part, and
refit every output weight. The refit alone lowered both validation and test AUC in every seed,
because the base weights had seen the validation rows. That run printed base and refit-only
test AUCs, but no candidate test AUCs. The rules above are the second run's. The first run is
kept as `constructive_check_run1.csv`.

| Seed | Lead | Pruned (reason) | Graft | Val AUC base → candidate | Accepted | Test AUC base → candidate [95% CI of Δ] |
|---|---|---|---|---|---|---|
| 0 | churn100-s0 | minor_hinges (rare) | none | 0.9114 → 0.9113 | no | 0.8951 → 0.8952 [−0.0007, +0.0008] |
| 1 | churn100-s1 | totalChargeChain (rare) | none | 0.9220 → 0.9182 | no | 0.9038 → 0.8906 [−0.0230, −0.0049] |
| 2 | churn100-s2-c1 | intl_state_unit (rare, overfit); eve_minutes_hinge (overfit) | none | 0.9048 → 0.9041 | no | 0.8756 → 0.8778 [−0.0008, +0.0057] |
| 3 | churn100-s3 | csc_extreme_chain (overfit) | none | 0.90486 → 0.90488 | **yes** | 0.8884 → 0.8884 [−0.0001, +0.0002] |
| 4 | churn100-s4 | area415_state47 (rare, overfit) | none | 0.8997 → 0.8996 | no | 0.8833 → 0.8833 [−0.0004, +0.0004] |
| 5 | churn100-s5 | nothing to prune | none | no candidate | – | 0.8811 |
| 6 | churn100-s6 | service_calls_low_day_usage (rare) | none | 0.9120 → 0.9015 | no | 0.8955 → 0.8907 [−0.0112, +0.0015] |
| 7 | churn100-s7-c1 | nothing to prune | none | no candidate | – | 0.9007 |
| 8 | churn100-s8 | usage_hinges (rare) | none | 0.9056 → 0.9050 | no | 0.8735 → 0.8742 [−0.0003, +0.0016] |
| 9 | churn100-s9 | day_eve_charge_cascade (rare) | none | 0.9030 → 0.8889 | no | 0.8750 → 0.8553 [−0.0313, −0.0079] |

**Reading.**
- **No grafts.** Every churn100 lead already carries the required signals in its residual,
  and the ones that are not seed-best genomes (s2-c1, s7-c1) are no exception.
- **Pruning "rare" signals hurts when the signal is useful.** It cost validation AUC in s1, s6
  and s9 (−0.004 to −0.014), and test AUC in s1 and s9 (CIs exclude 0). A signal no other seed
  realises is not therefore overfit: rarity in the residual sense measures how a signal is
  realised, not whether it is real.
- **Pruning small or overfit signals changes almost nothing:** |Δ test AUC| ≤ 0.003.
- **Validation selected only s3**, whose test AUC is unchanged.

The inventory gives no route to a better churn network by surgery. It does confirm, on test,
that the inventory's useful signals are real.

## Caveats

- **Not pre-registered.** The unit (post-hoc clusters), the residual-match variant and the
  constructive-check rules were all set during the analysis. The constructive check was
  revised once, before any candidate's test AUC was seen; this is recorded above.
- **Clusters are not NEAT species.** They depend on the 0.01 margin and the cut (sensitivity in
  `STEP1B_CLUSTERS.md`). Halving the cut splits Heart runs into many small groups, but churn
  stays one family.
- **Shared ancestry.** Within-run comparisons are among relatives (see section 3).
- **The C2 match is weak on these networks.** It counts a signal as present whenever the
  inputs reproduce it linearly, so most signals pass. The residual variant is stricter but its
  threshold is just as arbitrary. Some signals sit near it, for example churn100-s2's
  service-call gate: the s2 run best's version is found in 0 other seeds, the s2-c1 lead's in 6.
- **Leakage on validation.** The base networks were trained on all of train, validation rows
  included. Base and candidate share that advantage, so they are compared with each other only.
- **Test** is used only for reporting: the leads' test AUC, the signal test checks and the
  constructive check's final row. Nothing was selected on it.
- **Unverified by Mike.** All 13 explanations and their plain-language drafts are
  Claude-produced and wait for Mike's verification (`verification/RECORDING.txt`).
- **Heart 10-seed records used v2 tooling**; the new Heart leads use v3. Leaf boundaries can
  differ for that reason as well.

## Files

All under `experiments/species_leads/`:

| What | Files |
|---|---|
| Species, step 1b | `STEP1_SPECIES.md`, `replay_species.py`, `run_species_replays.sh`, `build_species.py`, `species/`, `leads.csv`, `runs.csv`, `verification.csv` |
| Clusters | `STEP1B_CLUSTERS.md`, `build_clusters.py`, `clusters/`, `cluster_leads.csv`, `cluster_runs.csv` |
| Explanations | `STEP2_AGENT_BRIEF.md`, `xn.py`, `verify_record.py`, `recheck_records.py`, `explanations/`, `plain_language/`, `records_recheck.csv`, `make_pack.py`, `verification/` |
| Inventory | `signal_inventory.py`, `signals.csv`, `signal_matches.csv` |
| Constructive check | `constructive_check.py`, `constructive_check.csv`, `constructive_check_run1.csv` |
