# Reference implementation (2026-08-20 survival brief)

Verbatim copies of the code delivered with the brief (`docs/survival-in-explaneat.md`),
kept for provenance. The originals live untracked in the main checkout at
`~/dev/explaneat/survival/`. The packaged, tested version is `explaneat/survival/`.

Run as delivered (from this directory, needs `uv sync --extra survival`):

    uv run python test_cox_torch.py   # ALL CHECKS PASSED (re-run 2026-11-06)
    uv run python simulate.py         # reproduces the brief's section 7 table exactly

## Comparison with the untracked `survival/` folder (2026-11-07)

Checked read-only for VARIANZ readiness (THESIS_FINAL_WORK task 9 (vi)). The
folder holds six files; nothing in it is missing from the repository.

- The five `.py` files are byte-identical to the copies here, and
  `survival-in-explaneat.md` is byte-identical to `docs/survival-in-explaneat.md`.
- Every function maps to the package:

| Reference | Package | Difference |
|---|---|---|
| `cox_torch.CoxRiskSets` | `explaneat/survival/cox.py` | same sort and tie blocks; adds input validation and precomputed Efron indices |
| `cox_ph_loss` (Breslow, Efron), `_log_cumsum_exp_desc`, `partial_log_likelihood` | `cox.py` | Efron vectorised (scatter-add) instead of a per-block loop; default ties `breslow` -> `efron`; `null_cox_loss` added |
| `BreslowBaseline` | `breslow.py` | vectorised fit; adds `baseline_*`, `risk_at`, `SurvivalPredictor` |
| `concordance` | `metrics.py` | same algorithm (Harrell, Uno, td-AUC, Brier, IBS, calibration added) |
| `explaneat_survival.CoxCriterion` | `criterion.py` | risk-set cache keyed by a SHA-1 of (time, event) |
| `cox_partial_likelihood_fitness`, `cox_concordance_fitness` | `fitness.py` | built by `make_cox_fitness` (adds the parsimony term); `nneat_cls` is a factory argument |
| `fit_survival_predictor(net, xs, ys)` | `breslow.fit_survival_predictor(risk_fn, X, time, event)` | signature takes a risk function |
| `make_survival_targets` | `data.py` | adds `validate_survival_target` |
| `simulate.simulate` / `simulate.run` | `survival/simulate.py` / `experiments/survival/run_seed.py`, `study.py sim` | same data-generating process (identical arrays for the same seed); `run` drives an evolved NEAT genome |
| `benchmark.get_dataset`, `evaluate`, `run` | `data.load_builtin`, `metrics.survival_metrics`, `run_seed.py` / `study.py bench` | `load_builtin` drops FLCHAIN's `chapter` (outcome leakage) and nudges t <= 0 |
| `benchmark.MLP`, `train_nn` | (superseded) | the brief's stand-in for "a NEAT phenotype"; replaced by the evolved network, as the brief intended |
| `test_cox_torch.py` checks 1-6 (`ref_breslow`, `ref_efron`, LBFGS recovery) | `tests/test_survival/test_cox.py` | all ported; a finite-difference gradient check added |

Numeric check on random data (n = 300, with and without tied times):
`cox_ph_loss` Breslow loss and gradient differ by 0.0, Efron loss by 0.0 and
gradient by at most 1.5e-15; the Breslow baseline, `concordance` and
`simulate()` (both modes, two seeds) differ by 0.0.
