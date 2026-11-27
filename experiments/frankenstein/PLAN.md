# Frankenstein union graft on Churn: plan

> **Future work, not in the thesis.** Exploratory study for Mike's curiosity (requested
> 28 Nov 2026). No thesis claim rests on it. Written and committed before any outcome run.

## Question

If every useful explained feature from all ten churn100 networks is put into one model, how
well does it perform? Does it beat the best single seed and an ensemble of the ten, or does it
only duplicate the shared signals that the novelty screen (study C2) found?

## Feature pool (train only; `results/pool.csv`, written by the `pool` stage)

The C2 donor pool (`experiments/churn/TRANSPLANT2_PLAN.md`), recomputed by
`run_transplant.candidates` and checked against the graftable donors in
`experiments/churn/transplant2_selection.csv`:

- a churn100 leaf annotation with all-input entries, the three collapse preconditions and a
  readout edge to the output;
- train restore gain ≥ 0.02;
- graftable: no input → output edge inside the part (one leaf, s4 `csc_threshold`, is excluded
  for this, as in C2).

**21 features**, from all ten seeds (s0: 3, s1: 3, s2: 1, s3: 2, s4: 1, s5: 3, s6: 3, s7: 1,
s8: 2, s9: 2).

**Signal family** (train only, fixed before evaluation): each feature's donor logit contribution
is projected on the churn signal catalogue's data basis (`experiments/churn_signals/basis.py`).
A block's share is var(component) / var(contribution). The family is the **non-linear block**
(`shape:*` or `int:*`) with the largest share, i.e. what the part adds beyond straight lines.
Pool families: `shape:day` 9, `shape:csc` 6, `int:day*eve` 3, `int:csc*day` 2,
`int:day*vmail` 1. Every feature also carries a large linear share (0.40 to 0.80). The pool CSV
lists both shares, the top blocks and the hypothesis text.

## Builds

All builds use the C2 graft machinery (`genome_surgery.graft`, ext_common), with no new surgery
code.

- **Standalone Frankenstein:** a genome with only the 71 inputs and the output node. It gets
  **all 71 direct input → output edges** (the linear backbone; weight 0 before the refit) and
  every pooled part grafted in with its trained weights and its readout edges into the one
  output. Fine-tune seed 10.
- **Union into seed r** (r = 0..9): churn100-s<r> plus every pooled part whose donor seed is not
  r. Its own structure is unchanged. Fine-tune seed r.
- **Sham:** the same topologies, with every grafted weight and grafted-node bias drawn from
  N(0, 1). The k-th pool part uses seed 100·b + k (b = r, or 10 for the standalone). The direct
  edges are refit anyway.
- **Function check:** before any refit, each non-sham graft must have logit = base logit + the
  sum of its donor signals (to 1e-8).

## Arms per build (graft and sham alike)

- **Readout refit:** L2 logistic regression over *every* connection into the output, plus the
  output bias (unpenalised). For a union this includes the recipient's own readout and direct
  edges.
  - Columns are standardised; the weights are folded back into the genome.
  - Damped Newton (`genome_surgery._logistic_newton`) from zero.
  - Penalty is sklearn's C convention: mean BCE + ‖w‖² / (2Cn).
  - C is chosen from the LR reference grid {0.001, 0.003, …, 100}: fit on the 3,200 fit rows,
    lowest BCE on the 800 validation rows of `experiments/churn/val_split.json`, ties to the
    smaller C.
  - The refit is checked to write back exactly (to 1e-9).
- **Conditions:**
  1. `refit`: refit on the 4,000 train rows with the chosen C.
  2. `refit-ft500`: (1), then masked fine-tune with E = 500 on the train rows (the D2 setting).
  3. `refitfit-ftES`: refit on the 3,200 fit rows with the chosen C, then masked fine-tune with
     early stopping (`EARLY_STOPPING_PLAN.md`: validation BCE, patience 50, min_delta 1e-5,
     max 5,000 epochs, best epoch restored; the D2 ES setting).
- **Optional arm, standalone only (deduplicated Frankenstein):** greedy forward selection on
  validation.
  - Start from the direct edges alone.
  - At each step add the part with the largest validation AUC. Each candidate set is fit on the
    fit rows, with C chosen as above.
  - Stop when the best gain is < 0.001.
  - The kept parts are then built as one genome and refit on the train rows (C chosen as above;
    refit only).
  - Report how many features it keeps.

## Comparators (1,000-row test split)

- each base seed, and the best-by-train seed;
- the **10-seed ensemble**: sigmoid of the mean of the ten networks' logits (the key control);
- the LR (L2) and gradient-boosting references, read from
  `experiments/benchmarks/benchmarks.csv` (`lr_l2`, `gb`) and
  `experiments/references/results/baselines.json`, never typed.

## Outcomes (`RESULTS.md`, CSVs from `frankenstein.py` / `summary.py`)

- Test AUC with 95% percentile bootstrap CIs (10,000 resamples, seed 42, `ec.bootstrap_indices`).
  Paired deltas against the best seed, the ensemble and the sham of the same condition, plus,
  for a union, against its base recipient. For the ten unions: the median and the sign counts.
- Size: hidden units and enabled phenotype connections, against a single seed and the ensemble
  total.
- Refit readout per pooled feature (standalone, every condition):
  - the readout weights;
  - the train sd of the part's logit contribution, and its ratio to the donor's own sd.
- Grouped by family, the **effective number of realisations** (Σ sd)² / Σ sd²:
  - 1 means one realisation carries the family;
  - n means the weight is spread evenly over n aliases.
- For a union: the train sd of the grafted contribution against the recipient's own remaining
  logit.
- One paragraph on explainability.

## Rules

- **Deterministic:**
  - `PYTHONHASHSEED=0`, torch threads 1, BLAS threads 1;
  - fixed fine-tune and sham seeds;
  - Newton refits from zero.
  - Each stage is run twice and the outputs compared byte for byte before committing.
- **Train-only selection:** pool, families, C, early stopping and the greedy choice use train,
  fit or validation rows only. The test split is read only in the evaluation step of `run`.
- **No DB writes or migrations:** read only (`ReadOnlySession`), and no derived genome is
  stored.
- **Compute:** a dry run timed one fine-tune epoch at well under 0.2 s, so the full grid
  (4 fine-tunes × 11 builds) is expected to take under an hour. If it runs much longer, the
  standalone build is reported first.
