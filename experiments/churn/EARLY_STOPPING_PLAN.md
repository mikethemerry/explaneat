# Early stopping for fine-tuning and re-initialisation training (amendment, 2026-11-05)

Requested by Mike: replace fixed-duration training (E = 500 / 2000) with early stopping.
Written before any early-stopped run.

## Rule (applies to every masked fine-tune / from-scratch training in studies B, D2 and E)

- **Validation set:** a fixed, stratified 20% of the churn **train** split, `train_test_split`
  with `random_state = 42`. The **test split is never used**, as before. The remaining 80% is
  "fit". The indices are written once to `experiments/churn/val_split.json` and reused
  everywhere.
- **Training:**
  - Masked backprop on the fit rows: full batch, Adadelta lr 1.5, BCE, as before.
  - After every epoch, compute the validation BCE.
  - Stop when the validation BCE has not improved by more than `min_delta = 1e-5` for
    **patience = 50** epochs, or at **max_epochs = 5000**.
  - **Restore the weights from the best validation epoch.**
  - Record the stopping epoch, the best epoch and the best validation BCE.
- **Readout refit** (studies C, D2) is fitted on the fit rows only, so the validation rows stay
  unseen until early stopping uses them.
- **Unchanged:** evolution itself (PropNEAT with 5 backprop epochs per generation). This rule
  covers only post-hoc fine-tuning and from-scratch training.

## Runs (new rows, `es` tag; the fixed-epoch results stay as recorded and are not replaced)

- **B (repair):** conditions ft-only, prune-ft and flagged-ft, with early stopping, for all 10
  seeds.
- **D2 (rescue):** conditions 3, 5, 6 and 7 with early stopping, and conditions 4, 8 and 9 with
  the refit on the fit rows, for all 35 runnable cases.
- **E (re-initialisation), both levels:** 10 inits each with early stopping. This replaces the
  E = 500/2000 checkpoints in the early-stopped analysis. The fixed-epoch E run, already in
  progress, finishes and is reported as well.

## Outcomes

The same outcomes as each study's plan, computed on the early-stopped rows, plus the stopping
epochs. Results are reported next to the fixed-epoch results; where conclusions differ, both
are stated.
