# Heart benchmark on the paper split (exploratory, 2026-11-18)

Extra comparators for Heart (242 train / 61 test, the AAAI paper split, z-scored on train), added
next to the logistic regression and gradient boosting in `../rashomon.md` / `../baselines.json`.
Each model is tuned by 5-fold stratified CV on the training rows only (seed 42); the test set is read
once. 95% intervals: 10,000 bootstrap resamples of the test set, `RandomState(42)`. Data exported read-only
from explaneat_v4 (dataset eea247c2, split 165cebbd). Run with the conv-neat worktree venv (it has
`interpret` for the EBM): `OMP_NUM_THREADS=2 <venv>/python bench.py` with `heart.npz` beside it.
Not pre-registered; with n_test = 61 the intervals are about 0.1 wide, so the ranking is noisy.
