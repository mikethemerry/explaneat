# Rashomon-set membership, in AUC terms (task 6)

Written 7 Nov 2026 on branch `thesis/numbers`. This is a report for decision D6. No reported number is changed: the per-seed test AUCs are the unchanged `test_auc` column of the runs CSVs.

**Membership.** A model is inside the ε-set for reference *r* when its test AUC ≥ *r* − ε, for ε ∈ {0.005, 0.01, 0.02}. Strictly, the Rashomon set is defined by loss. Here it is stated in AUC terms, because the thesis uses AUC.

**Scripts:**
- `scripts/thesis/baselines.py` (references ii and iii) → `baselines.json`;
- `scripts/thesis/rashomon_population.py` → `rashomon_population.csv` (15,000 genomes);
- `scripts/thesis/rashomon.py` → `rashomon_seeds.csv`, `rashomon_counts.csv`, `rashomon_population_summary.csv` and `rashomon_summary.json`.

**Run sets:**
- **Pre-registered:** Heart (40 generations), MONK-2 and Adult, seeds 0–9.
- **Supplementary, not part of the pre-registration:** Heart-100, churn100 and churn200, seeds 0–9.

No seed was re-run, dropped or reselected.

## References

All three references are fitted on the **same** train/test split the PropNEAT runs used, with the same z-normalised features.

| Dataset | (i) Best observed seed | (ii) Logistic regression | (iii) Gradient boosting (this report) | GB with default hyperparameters (sensitivity) |
|---|---|---|---|---|
| Heart (n_test = 61) | 0.912 (s8) | **0.919** [0.841, 0.978], C = 0.1. Refit equals the paper's 0.919 | **0.945** [0.885, 0.988] | 0.918 |
| Heart-100 | 0.926 (s2) | 0.919 (same split) | 0.945 (same split) | — |
| MONK-2 (n_test = 87) | 1.000 (s5, s7) | **0.423** [0.306, 0.542], C = 0.1. Refit equals the paper's 0.423 | **0.948** [0.878, 0.997] | 0.889 |
| Adult (n_test = 9,769) | 0.909 (s3) | **0.905** [0.899, 0.912], C = 0.1. Refit equals the paper's 0.905 | **0.928** [0.922, 0.933] | 0.926 |
| churn100 / churn200 (n_test = 1,000) | 0.904 (s1) / 0.911 (s0) | **0.813** from `experiments/churn/baselines.json`, as instructed. The same-protocol CV refit gives 0.819 | **0.934** [0.902, 0.962] | 0.932 (= `baselines.json`) |

CIs are 95% percentile bootstrap intervals (10,000 resamples, seed 42).

**Logistic regression (ii)** follows the paper's protocol (`scripts/aaai27/run_a_logistic.py`): L2, C chosen from 11 values by 5-fold CV on train (ROC AUC), lbfgs, random_state 42. The paper's three values are confirmed exactly.

**Gradient boosting (iii):**
- **Model:** sklearn `HistGradientBoostingClassifier`, random_state 0, no early stopping.
- **Tuning:** hyperparameters chosen by 5-fold stratified CV (shuffle, seed 42) on the **train split only** (ROC AUC), over learning_rate {0.03, 0.1} × max_iter {100, 300} × max_leaf_nodes {7, 15, 31} × min_samples_leaf {5, 20} × l2_regularization {0, 1} (48 configurations). The model is then refit on all of train; n_jobs = 1.
- **Chosen hyperparameters:**

  | Dataset | learning_rate | max_iter | max_leaf_nodes | min_samples_leaf | l2 |
  |---|---|---|---|---|---|
  | Heart | 0.03 | 100 | 7 | 20 | 1 |
  | MONK-2 | 0.1 | 300 | 15 | 20 | 0 |
  | Adult | 0.1 | 300 | 15 | 5 | 1 |
  | churn | 0.1 | 300 | 15 | 20 | 0 |

- **Caveats:**
  - **This is the first gradient-boosting fit on the paper's Heart and MONK-2 splits.**
  - **Adult on the paper's split gives 0.928**, not the 0.930 of the PMLB screen.
  - Heart's GB (0.945) sits well above its CV score (0.888). With n_test = 61 the interval is about 0.10 wide, so this reference is noisy.
  - On MONK-2, GB (0.948) is *below* the three exact seeds (1.000), so "within ε of GB" and "within ε of the best" differ there.

## Per-seed test AUC (best genome, from the runs CSVs)

| Run set | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Heart | 0.875 | 0.880 | 0.897 | 0.874 | 0.874 | 0.893 | 0.865 | 0.886 | 0.912 | 0.872 |
| MONK-2 | 0.638 | 0.998 | 0.767 | 0.850 | 0.862 | 1.000 | 0.754 | 1.000 | 0.870 | 0.682 |
| Adult | 0.903 | 0.904 | 0.903 | 0.909 | 0.907 | 0.907 | 0.905 | 0.907 | 0.909 | 0.905 |
| *Heart-100* | 0.899 | 0.895 | 0.926 | 0.899 | 0.889 | 0.913 | 0.863 | 0.889 | 0.882 | 0.874 |
| *churn100* | 0.895 | 0.904 | 0.876 | 0.888 | 0.883 | 0.881 | 0.895 | 0.900 | 0.874 | 0.875 |
| *churn200* | 0.911 | 0.905 | 0.894 | 0.910 | 0.899 | 0.876 | 0.906 | 0.898 | 0.900 | 0.894 |

`rashomon_seeds.csv` has every seed's gap to each reference and its in-set flags.

## Seeds inside the ε-set (out of 10)

Membership is test AUC ≥ reference − ε.

| Run set | Reference (AUC) | ε = 0.005 | ε = 0.01 | ε = 0.02 |
|---|---|---|---|---|
| **Heart** | (i) best observed 0.912 | 1 | 1 | 3 |
| | (ii) LR 0.919 | 0 | 1 | 1 |
| | (iii) GB 0.945 | 0 | 0 | 0 |
| **MONK-2** | (i) best observed 1.000 | 3 | 3 | 3 |
| | (ii) LR 0.423 | 10 | 10 | 10 |
| | (iii) GB 0.948 | 3 | 3 | 3 |
| **Adult** | (i) best observed 0.909 | 8 | 10 | 10 |
| | (ii) LR 0.905 | 10 | 10 | 10 |
| | (iii) GB 0.928 | 0 | 0 | 2 |
| *Heart-100 (supplementary)* | (i) best observed 0.926 | 1 | 1 | 2 |
| | (ii) LR 0.919 | 1 | 2 | 2 |
| | (iii) GB 0.945 | 0 | 0 | 1 |
| *churn100 (supplementary)* | (i) best observed 0.904 | 2 | 4 | 5 |
| | (ii) LR 0.813 | 10 | 10 | 10 |
| | (iii) GB 0.934 | 0 | 0 | 0 |
| *churn200 (supplementary)* | (i) best observed 0.911 | 2 | 4 | 9 |
| | (ii) LR 0.813 | 10 | 10 | 10 |
| | (iii) GB 0.934 | 0 | 0 | 0 |

**Reading:**
- **Adult** is the clean case. All 10 seeds are within 0.01 of the best observed and at or above LR. None is within 0.01 of GB, and only 2 are within 0.02 of it.
- **Heart.** Most seeds are 0.02–0.04 below both the best observed (s8) and LR. That is one to two seed-standard deviations, all inside the test-set noise at n = 61. Against GB, none is within 0.02.
- **MONK-2** is bimodal. Exactly the 3 exact seeds are in the set for any ε, whether the reference is the best observed or GB. LR is below chance, so it is not a meaningful reference.
- **churn.** All seeds beat LR. None is within 0.02 of GB. Within 0.02 of the best observed seed are 5 of 10 (churn100) and 9 of 10 (churn200).

## Population level, final generation

This section is labelled population-level, final generation. It was cheap: 15,000 NeuralNeat forward passes in a few minutes.
- **Genomes:** all 250 genomes of each run's last evaluated generation (generation 39 for Heart, 99 for the others, 199 for churn200), forward through NeuralNeat on the test split.
- **Pooling:** 2,500 genomes per run set.
- **References:** the same as above, so the shares are comparable with the seed counts.
- Final-generation genomes are not selected on test and are not the run's best genome.
- In 39 of the 60 runs, the best genome is itself in the final generation; its test AUC there reproduces `runs*.csv` exactly.

Per row: test AUC across all 2,500 genomes (min / p10 / median / p90 / max), then the pooled share inside the set (per-run median in brackets).

| Run set | Test AUC: min / p10 / median / p90 / max | (i) ε 0.005 | (i) ε 0.01 | (i) ε 0.02 | (ii) ε 0.01 | (ii) ε 0.02 | (iii) ε 0.02 |
|---|---|---|---|---|---|---|---|
| Heart | 0.702 / 0.848 / 0.878 / 0.904 / 0.934 | 8.2% (2.8%) | 13.4% (5.8%) | 24.9% (12.2%) | 6.4% (2.0%) | 15.2% (7.6%) | 0.7% (0%) |
| MONK-2 | 0.301 / 0.482 / 0.603 / 0.807 / 1.000 | 0.8% (0%) | 0.9% (0%) | 1.2% (0%) | 98.4% | 99.0% | 2.8% (0%) |
| Adult | 0.664 / 0.873 / 0.896 / 0.904 / 0.909 | 7.3% (5.8%) | 31.6% (35.6%) | 70.9% (74.2%) | 51.5% (54.2%) | 78.0% (79.6%) | 0.2% (0%) |
| *Heart-100* | 0.681 / 0.849 / 0.887 / 0.910 / 0.954 | 3.4% | 5.9% | 15.0% | 10.7% | 26.4% | 2.3% |
| *churn100* | 0.613 / 0.793 / 0.858 / 0.888 / 0.907 | 2.2% | 5.1% | 15.2% | 87.2% | 89.9% | 0% |
| *churn200* | 0.580 / 0.781 / 0.867 / 0.900 / 0.917 | 4.3% | 9.2% | 21.1% | 84.4% | 87.4% | 0.1% |

`rashomon_population_summary.csv` has every reference and ε, with the per-run min, median and max shares and the number of runs that have any genome in the set.

**Reading (population-level, final generation):**
- Whole populations at the end of training are **not** inside the set for small ε. The final population is spread well below its best genome: Adult's p10 is 0.873 against a best of 0.909.
- On Adult, about a third of the final generation is within 0.01 of the best seed, and about 70% is within 0.02.
- On Heart and on churn relative to the best seed, the share is a few percent at ε = 0.005 and about 15–25% at ε = 0.02.
- **Some individual final-generation genomes beat the selected best genome on test**: Heart 0.934 (best seed 0.912) and Heart-100 0.954. That is expected when selection is on train AUC and n_test = 61; these genomes were not selected and must not be.
- This answers the open item in `docs/signals-and-realisations/README.md` §3 ("whether whole populations at convergence sit inside the set has not been measured").

## For D6

The choice of reference matters more than the choice of ε:
- Against the best observed seed, or against LR, Adult is fully inside at ε = 0.01, and Heart mostly is not.
- Against same-split gradient boosting, essentially no seed on any dataset is inside at ε ≤ 0.01. The exception is MONK-2, where the 3 exact seeds exceed GB.

If the thesis wants a single statement, "within 0.01 of the best observed model" gives Adult 10/10, Heart 1/10 and MONK-2 3/10. That statement should come with the GB gap (Adult −0.019 to −0.025; Heart −0.03 to −0.08).
