# Chapter 7 datasets: what the run records say (20 Nov 2026; preprocessing and MNIST 21 Nov 2026)

Numbers below are quoted from `data.json` (written by `extract.py`) and `datasets.csv`; rerun the
scripts before quoting them elsewhere. The prose compared against is Chapter 7 of the thesis
(`sections/7-ExplaNEAT.tex`) as of 20 Nov 2026.

## Every reported run used the split in the table

`extract.py` asserts it: the ten Adult, ten Heart-100 and ten Churn-100 runs of Experiment 2 (their
experiment ids in `experiments/10seed/runs.csv`, `runs_heart100.csv`, `experiments/churn/runs_churn100.csv`,
with matching row and input counts), and each Experiment 1 network (the genome ids in
`experiments/benchmarks/results/propneat.json`, and the three original networks in
`experiments/10seed/runs_paper_reference.csv`). Heart's original network (Experiment 1) and the Heart-100
runs (Experiment 2) share one split; so do Adult's original network (Experiment 1) and the ten Adult runs
(Experiment 2). WHAS500 / GBSG2: the split file, the loader and the selected run's
record agree on rows, inputs and training events.

## Disagreements with the current Chapter 7 prose

1. **Adult's source.** Experiment 2's prose says "Adult (UCI, one-hot encoded)". The runs used PMLB's copy
   of UCI Adult (`datasets.csv`: source PMLB, `explaneat/db/pmlb_loader.py`). Citing Kohavi (1996) for the
   data is still right; the source word in the table is PMLB.
2. **Adult's positive class agrees with the prose.** The stored class names are `[">50k", "<=50k"]` and
   y = 1 is `<=50k` in both the raw PMLB record and the encoded one; the positive share of the training rows
   is 0.761 (`datasets.csv`), the majority class. So the networks predict *income at most \$50K*, as
   Experiment 2's prose says (any reading of an Adult part as "raises the chance of high income" has the
   sign reversed).
3. **Heart's missing values.** Not in the prose: 4 missing `ca` and 2 missing `thal` values were set to the
   column mean at download (`ucimlrepo` path, `explaneat/api/routes/datasets.py`), and because `thal` is
   then one-hot encoded, the 2 imputed rows form their own indicator column `thal:4` (one of the 26 inputs).
   The mean is over all 303 rows (fixed before the split), a negligible leak.
4. **Splits are not all stratified.** The old Methods paragraph says "80:20 train:test splits with a fixed
   seed (random_state=42)", which is right for Adult, Heart and MONK-2, but only MONK-2 is stratified;
   Adult and Heart are not (`stratified = no` in `datasets.csv`). Churn, Connect-4 and the synthetic sets
   are stratified (Churn's prose says so); WHAS500 / GBSG2 are stratified on the event.
5. **The Methods opening is the old paper framing.** "I demonstrate the test on three models trained on
   ... Adult, Heart, MONK-2; and on a densified twin of the Adult network" no longer matches the tables:
   Experiment 1 reports Heart, Adult and MONK-2 (original networks; Adult back since 21 Nov, see below),
   WHAS500, Connect-4 (cells, lines), XOR and the three synthetic sets. Experiment 1's list ("twelve
   datasets in seven groups") still names Adult and Churn (both reported in Experiment 2), names "U,
   Saddle" but not U + saddle, and does not separate the two Connect-4 encodings.
6. **Numbers that agree.** Adult 107 inputs, 39,073 / 9,769; Churn 71 inputs, 4,000 / 1,000, phone number
   removed, state and area code one-hot; Heart 26 inputs, 242 / 61; MONK-2 17 inputs, 432 configurations,
   345 / 87; synthetic 2,000 rows, 10 inputs, 75:25 stratified, 1,500 / 500, outcome share about 40% (0.403,
   0.388, 0.418 of the training rows): all match `datasets.csv`.

## What the brief asked to check

- **WHAS500**: 320 train / 80 validation / 100 test (64:16:20, stratified on the event, seed 42). The 80
  validation rows chose the model: each seed's generation champion with the lowest validation Cox loss,
  then the seed with the lowest validation loss (`experiments/survival/cox_case/run_case.py`). Event
  (death, `fstat`) share of the training rows: `datasets.csv`. GBSG2 is split the same way.
- **Connect-4 target**: PMLB's three classes (win, loss, draw for the first player, x, to move, with
  perfect play) reduced to 1 = win, 0 = loss or draw (`experiments/complex_real/data.py`; the stored
  target description). Cells: one input per cell, x = +1, o = -1, blank = 0.
- **Connect-4 lines**: 16 counts computed from the 42 cells (codebook in the dataset record, copied into
  `data.json`): open lines of four holding exactly 1, 2 or 3 pieces of one player and none of the other (6),
  each player's threats (empty cells that complete an open three; 2) and their split by odd / even row (4),
  each player's pieces in the centre column (2), and the number of columns whose next playable cell is on
  an odd / even row (2). Same rows and split as the cells (identical training indices:
  `train_indices_sha256_16` in `data.json`).
- **Experiment 3 validation rows** (marked $^a$ in the table): early stopping in the grafting studies held
  out a stratified 20% of the training rows (Churn 800, MONK-2 69; `experiments/churn/val_split.json`,
  `monk2_val_split.json`); the test rows are untouched.

## Preprocessing and splits as the code ran them (21 Nov 2026)

Each phrase in the tables' Preprocessing, Scaling and Split columns, the code it was read from, and the
stored record `make_table.py` asserts it against. One method always gets the same words.

| Phrase in the table | Datasets | Code that did it | Checked against |
|---|---|---|---|
| all inputs z-normalised (training-row mean and standard deviation) | Adult, Heart, MONK-2, Churn, Connect-4 (both), synthetic (3) | `(X - mean) / sd` with the training rows' mean and population SD (constant columns: SD 1), applied to every model input, one-hot columns included: `experiments/10seed/run_seed.py` `load_training_data` (asserts equality with the split's stored `scaler_params`), `experiments/complex_real/data.py` / `c4_line_features/lfdata.py` / `ushape_saddle/data.py` `prepare` | split `scaler_type` = StandardScaler, scaler mean equal to the training rows' mean (`scaler_fit_rows` = train) |
| the same | WHAS500, GBSG2 | `explaneat.survival.data.standardise(X_train, ...)` in `experiments/survival/run_seed.py` | `survival.scaler_fit_rows` |
| *k* categorical attributes one-hot encoded (*k* levels → *k* columns; *n* columns) | Adult (7, 100 columns), Heart (7, 20), MONK-2 (all 6, 17), Churn (state and area code, 54) | `explaneat/db/encoding.py` `prepare_dataset_arrays` (one column per level, no level dropped) | the prepared dataset's `encoding_config`; input count = one-hot columns + passthrough |
| one-hot encoded, first level dropped (*k* levels → *k* − 1 columns) | WHAS500 (8 two-level attributes, 8 columns), GBSG2 (3 attributes, 4 columns) | `explaneat.survival.data.load_builtin`: `sksurv.preprocessing.OneHotEncoder` (drops the first level) | raw categorical levels from scikit-survival; input count |
| binary attribute(s) kept as one 0/1 column | Adult (sex), Churn (international plan, voice mail plan) | `encoding_config.passthrough` type `binary` | the same |
| the unknown value `?` kept as its own level (workclass, occupation, native-country) | Adult | PMLB's `adult` keeps all 48,842 UCI rows and codes `?` as a level | stored level counts are one more than UCI's documented values (8, 14, 41; `adult.names`) |
| missing values mean-imputed (mean of all 303 rows; `ca`: 4 values; `thal`: 2 values, one-hot encoded as their own column `thal:4`) | Heart | `explaneat/api/routes/datasets.py` (ucimlrepo path): `np.nanmean` over all rows before any split; the one-hot step then turns the imputed `thal` value into the level `thal:4` | detected in the raw record (`_imputed_columns`) |
| phone number removed | Churn | `experiments/churn/ingest_churn.py` (`DROP`) | raw record metadata `dropped` |
| duplicate rows removed (each of the 432 attribute configurations once) | MONK-2 | `experiments/aaai27/ingest_monk2.py` (`drop_duplicates`, PMLB's 601 rows → 432) | rows = unique rows = product of the attribute cardinalities |
| each of the 42 cells coded x = +1, o = −1, blank = 0 | Connect-4 cells | `experiments/complex_real/data.py` `load_task("c4")` | dataset metadata `encoding` |
| 16 counts computed from the 42 cells (...) | Connect-4 lines | `experiments/c4_line_features/lfdata.py` | metadata `codebook` (16 entries), base dataset = the cells |
| generated in natural units (binary inputs 0/1) | synthetic | `experiments/ushape_saddle/data.py` | split `preprocessing_steps[0]` = "raw units" |
| none (inputs 0/1); none (used as they are) | XOR | `experiments/xor/run_xor.py` (`X`, no scaling) | `X`, `Y` asserted to be the XOR truth table |
| none (grey-scale pixel intensities 0–255); pixel intensities scaled to [0, 1] (divided by 255) | MNIST (both rows) | `explaneat.conv.data.mnist_splits`: `X / 255.0`; no further normalisation in `explaneat/conv` | raw `uint8` maximum 255; scaled training images span [0, 1] |
| random split 80:20, not stratified, seed 42 | Adult, Heart | `sklearn.model_selection.train_test_split(shuffle=True, random_state=42)` (the split rows in the database) | split `stratify` = false, `test_size`, `random_state` |
| random split 80:20 (75:25), stratified on the outcome, seed 42 | MONK-2, Churn, Connect-4 (80:20); synthetic (75:25) | the same with `stratify=y` | split `stratify` = true |
| random split 64:16:20, stratified on the event: test 20%, then validation 20% of the rest; seed 42 | WHAS500, GBSG2 | `explaneat.survival.data.survival_split_indices` (two `train_test_split` calls stratified on the event) | split file `seed`, `test_size`, `val_size` |
| test: the standard MNIST test set; validation: from the 60,000 standard training images, stratified on the digit, seed 0; training: a stratified subset of the other 50,000 / all other 50,000 | MNIST 10-class | `explaneat.conv.data.mnist_splits(seed, classes=0..9, n_train, n_val)` | recomputed and asserted equal to the run records' saved row indices; the two share validation and test rows |

Not shown in the table: the Experiment 3 validation rows ($^a$) are a stratified 20% of the training
rows (`experiments/churn/val_split.json`, `monk2_val_split.json`); the MNIST 10k run's pruned
continuation trained on rotating chunks of 5,000 of its 10k training rows per generation (`train_chunk`
in its run record).

## MNIST (Experiment 4) and KRK

MNIST is in the tables since 21 Nov 2026 (`tab:ch7-datasets-mnist`): the datasets the lineage of the one
presented ExConNEAT network trained on (the MNIST 10-class network of the 50k warm start;
`experiments/conv_neat/RESULTS.md` section 11b). Two rows, one per training set, each naming the phases
that used it (clearer than one merged row: the phases differ in training rows only, and the 10k subset
is a different split cell from the full 50k):

- **MNIST 10-class, 10k training subset: 10k run, pruning phase**: run records `results/sweep_p50_m4_e5`
  (the selected genome) and `results/mnist10_pruned` (its pruned continuation); the same rows.
- **MNIST 10-class, all 50k training rows: 50k warm start**: `results/mnist10_ws50k_halving` (warm start
  from the pruned population; same validation and test rows).

Dropped on 21 Nov 2026 (branch thesis/run-config-trim): **MNIST 3 vs 8 vs noise**
(`three8noise/results/run_s0`); no network of it is presented. `extract.py` no longer rebuilds it.

Left out: the earlier MNIST runs of `experiments/conv_neat` section 11 (`mnist38_*`, `mnist10_p0.001_*`:
2,000 validation images) and the sklearn 8 x 8 digits runs, and the database's `mnist10_784` /
`mnist38_14x14` experiments (`experiments/mnist/`, the plain-PropNEAT linear check): none is reported in
the Chapter 7 tables.

**KRK** has run records but is not in the Chapter 7 tables, so it is not in the table (`data.json`,
`krk_mnist_run_records`): five experiments in both `explaneat_dev` and `explaneat_v4`
(`complex-krk-default-s0..s2`, `complex-krk-tuned-s0..s1`); study folder `experiments/complex_real/`
(runs, explanations, verification; book card 1.5, a blind verifier run, not signed off).

## "Used in" read from the run checks; Adult in Experiment 1 (21 Nov 2026, branch thesis/adult-exp1)

Mike's decision: the original Adult network is presented in Experiment 1 again. Adult's "Used in" is now
"1, 2". The column is no longer typed for the classification datasets: `make_table.py` reads it from
`data.json`, the experiments whose runs `extract.py` checked to have used the dataset's split (`run_splits`:
Experiment 1's explained networks and the three original networks, Experiment 2's ten-seed runs) plus
Experiment 3 where the grafting studies hold early-stopping rows (`exp3_validation`). The derivation gives the
previously typed value for every other classification dataset. WHAS500, GBSG2, XOR and MNIST (no database
split) keep the curated value. `extract.py` adds the original networks from
`experiments/10seed/runs_paper_reference.csv` when `propneat.json` lacks them (its Experiment 1 Adult entry
is added on branch thesis/wide-grids; when it lands, the genome ids are deduplicated and nothing changes).
Both scripts were run twice: byte-identical `data.json`, `datasets.csv` and `.tex`.
