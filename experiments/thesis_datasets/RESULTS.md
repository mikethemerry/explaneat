# Chapter 7 datasets: what the run records say (20 Nov 2026)

Numbers below are quoted from `data.json` (written by `extract.py`) and `datasets.csv`; rerun the
scripts before quoting them elsewhere. The prose compared against is Chapter 7 of the thesis
(`sections/7-ExplaNEAT.tex`) as of 20 Nov 2026.

## Every reported run used the split in the table

`extract.py` asserts it: the ten Adult, ten Heart-100 and ten Churn-100 runs of Experiment 2 (their
experiment ids in `experiments/10seed/runs.csv`, `runs_heart100.csv`, `experiments/churn/runs_churn100.csv`,
with matching row and input counts), and each Experiment 1 network (the genome ids in
`experiments/benchmarks/results/propneat.json`). Heart's original network (Experiment 1) and the Heart-100
runs (Experiment 2) share one split. WHAS500 / GBSG2: the split file, the loader and the selected run's
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
   Adult is reported only in Experiment 2, and Experiment 1 reports Heart and MONK-2 (original networks),
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

## KRK and MNIST

Both have run records; neither is in the Chapter 7 tables, so neither is in this table (`data.json`,
`krk_mnist_run_records`):

- **KRK** (`krk_draw_ordinal`, PMLB `krkopt`): five experiments in both `explaneat_dev` and `explaneat_v4`
  (`complex-krk-default-s0..s2`, `complex-krk-tuned-s0..s1`); study folder `experiments/complex_real/`
  (runs, explanations, verification; book card 1.5, a blind verifier run, not signed off).
- **MNIST**: `mnist10_784` (three experiments) and `mnist38_14x14` (four), in both databases; study folders
  `experiments/mnist/` (the linear check and linear start, book card 3.4) and `experiments/conv_neat/`
  (ExConNEAT, Experiment 4 in the chapter's introduction). If Experiment 4 stays in the chapter, its
  MNIST data may need a row here.
