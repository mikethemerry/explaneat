# Heart: validation selection, phantom masking and weight decay (quick test, 2026-11-20)

**Question.** On UCI Heart (paper split: 242 train / 61 test), does PropNEAT stop underperforming logistic regression if we (1) hold out validation rows and choose the genome by validation AUC, (2) mask phantom weights and (3) add a weight penalty?

**Setup.** The 242 training rows are split, stratified, seed 42, into 193 fit rows and 49 validation rows (`validation_split.json`). The scaler is fit on the 193 fit rows. Otherwise the Heart-100 settings: `experiments/10seed/configs/heart.ini`, pop 250, 100 generations, 5 Adadelta epochs per generation (lr 1.5), AUC fitness on the fit rows. Changes: `mask_phantom=True`, Adadelta `weight_decay` = wd, and the genome returned is the best by validation AUC over all generations (`BackpropPopulation` with validation data). Runs are in memory with no DB writes; seeds 0–4 per arm; PYTHONHASHSEED=0; 1 torch thread. Each run took 1–2 min. The arm is chosen on median validation AUC, with ties broken by mean validation AUC. Test was read for every arm and used for no choice. The original grid was {0, 1e-4, 1e-3, 1e-2}. 0 and 1e-2 tied at the top, at the edge of the grid, so 3e-2 and 1e-1 were added. The choice stayed 1e-2.

| weight decay | train AUC | val AUC | **test AUC** | selected gen | hidden | conns | test AUC if best-by-train (same runs) |
|---|---|---|---|---|---|---|---|
| 0 | 0.917 (0.814–0.925) | 0.960 (0.943–0.966) | **0.867 (0.831–0.883)** | 10 (2–32) | 3 (1–6) | 25 (19–32) | 0.879 (0.859–0.884), gen 96–99 |
| 1e-4 | 0.919 (0.884–0.930) | 0.951 (0.948–0.958) | **0.853 (0.820–0.918)** | 18 (2–29) | 3 (1–7) | 24 (21–32) | 0.873 (0.864–0.904), gen 94–99 |
| 1e-3 | 0.911 (0.902–0.921) | 0.953 (0.951–0.960) | **0.850 (0.816–0.904)** | 13 (9–45) | 2 (2–4) | 21 (16–26) | 0.888 (0.866–0.911), gen 97–99 |
| **1e-2 (chosen)** | 0.924 (0.900–0.935) | 0.960 (0.951–0.966) | **0.875 (0.871–0.880)** | 10 (3–80) | 3 (1–10) | 25 (22–40) | 0.898 (0.862–0.906), gen 89–99 |
| 3e-2 | 0.912 (0.899–0.927) | 0.958 (0.951–0.961) | **0.869 (0.846–0.950)** | 16 (3–75) | 2 (0–9) | 20 (17–30) | 0.894 (0.879–0.906), gen 69–99 |
| 1e-1 | 0.911 (0.863–0.919) | 0.948 (0.944–0.963) | **0.883 (0.850–0.917)** | 7 (3–93) | 1 (0–13) | 22 (15–47) | 0.894 (0.887–0.904), gen 28–91 |

Each cell is the median (min–max) over 5 seeds. "Selected gen" is the generation the validation-selected genome came from. Hidden and connections are phenotype counts (enabled edges).

References: Heart-100 (242 rows, no mask, no decay, best-by-train) has a test median of **0.892** (0.863–0.926). Tuned L2 LR scores 0.919 and L1 LR 0.926 (`experiments/benchmarks/benchmarks.csv`). Untuned LR fit on the same 193 fit rows scores 0.912 val / 0.903 test with C=1, and 0.904 val / 0.927 test with C=0.1.

## Answer

No. Validation selection with masking and weight decay does not close the gap. The arm chosen on validation (wd 1e-2) tests at a median of 0.875. That is below Heart-100's 0.892 and well below LR's 0.919–0.926. Validation selection makes things worse rather than better. The networks score 0.95–0.96 AUC on the 49 validation rows, where LR scores 0.90–0.91. That gap shows that picking the best of ~25,000 genomes on 49 rows overfits the validation set. The selected genomes come from early generations (median gen 7–18) and test lower than the same runs' best-by-train genomes. Weight decay helps a little at most: best-by-train test rises from 0.879 (wd 0) to ~0.89–0.90 (wd ≥ 1e-2), which only reaches the Heart-100 level. Training on 49 fewer rows probably costs part of that.

## Caveats

- The validation set has 49 rows (22 positive). One validation AUC point is about two pairs, and the median validation AUCs of the arms differ only in the third decimal place, so the choice of arm is close to noise.
- There are 5 seeds per arm. The test set has 61 rows, and the test spread within one arm (e.g. 0.846–0.950 at 3e-2) is larger than the differences between arms.
- This is an exploratory test and was not pre-registered. The grid was extended after the first look (still on validation only). The networks train on 193 rows, not 242, so this is not a clean comparison with Heart-100. LR on the same 193 rows is the fair reference.
- Runs are in memory with `BackpropPopulation.run`, not `DatabaseBackpropPopulation.run`.

## Files

`prepare_split.py` (writes `validation_split.json`; reads `explaneat_dev` in a read-only transaction), `run_arm.py <wd> <seed>` (writes `runs/wd<wd>_s<seed>.json` with per-generation best-by-train and best-by-val train/val/test AUC), `launch.py` (grid, ≤ 8 processes), `summarise.py` (writes `summary.json` and prints the table and the LR reference).
