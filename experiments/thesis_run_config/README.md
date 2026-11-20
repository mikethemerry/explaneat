# Chapter 7 run-configuration table (21 Nov 2026)

One table giving the full configuration of every run family Chapter 7 reports (`tab:ch7-run-config`),
cited from the chapter's common Methods. Every value is read from the record the runs left behind: the
database rows of the runs (read-only), the run records of in-memory runs, the config files, and the
training code at each run's recorded commit. Findings (where the configuration differs from what the
prose implies) are in `RESULTS.md`.

## Regenerate

```bash
# layer 2: read-only extractor (explaneat_dev + explaneat_v4 in READ ONLY transactions, PGOPTIONS
# default_transaction_read_only=on; `git show` of explaneat/core/backproppop.py at each run's commit; ~10 s)
uv run python experiments/thesis_run_config/extract.py      # -> data.json
# layer 3: formats data.json only
uv run python experiments/thesis_run_config/make_table.py   # -> run_config.csv, ch7-run-config.tex
# preview with the thesis preamble (no errors, no overfull boxes on 21 Nov 2026)
cd experiments/thesis_run_config && latexmk -pdf preview.tex && latexmk -c
# copy into the thesis (thesis side)
uv run python thesis/scripts/sync_ch7_exports.py            # -> thesis/ch7/exports/ch7-run-config.tex
bash thesis/scripts/build_overleaf_upload.sh                # -> sections/tables/ch7-run-config.tex
```

Both scripts were run twice on 21 Nov 2026; `data.json`, `run_config.csv` and `ch7-run-config.tex`
came out byte-identical.

The table emits `\providecommand{\thesistablestyle}{}\thesistablestyle` as the first line inside the
`sidewaystable`, so it takes the thesis-wide table style when the style file defines it and compiles
without it.

## Files

| File | What it is |
|---|---|
| `extract.py` | One record per run family. Database runs: `experiments.config_json`, `hardware_info.run_params`, stored generations, config template, `neat_config_text`; the family's experiments come from the study's own run list (`runs*.csv`, `g300_scores.csv`, `sig_gens.json`) or the explained genome (`propneat.json`). In-memory runs: Cox run JSONs (with their NEAT config text), Connect-4 logs plus configs (capacity study: base config + recorded overrides, checked against the config text stored on import), `run_xor.py`, ExConNEAT `config.json` / `results.json`. Code: optimiser, learning rate and phantom-mask default from `backproppop.py` at each commit; training-path diff between a family's commits. Experiment 3 fine-tuning: constants of `ext_common.py` / `genome_surgery.py` / `reinit.py` (ast), checked against the result CSVs |
| `data.json` | The extracted configurations and checks |
| `make_table.py` | Formats `data.json`. Typed here (curated, not results): row order and names, column labels, short wording of fitness / stop / selection rules (each asserted against the record), labels of NEAT options. The "All NEAT studies" line is built from the options equal in every family; any option that differs between families must have a column or an "Other" label (asserted) |
| `run_config.csv` | One row per family (plus the common line), with commits and sources |
| `ch7-run-config.tex` | The table (`sidewaystable`, panels (a) PropNEAT and (b) ExConNEAT) |
| `preview.tex`, `preview.pdf` | The table with the thesis preamble (as `thesis/ch7/exp1/preview.tex`) |

## Checks the extractor makes (each an assertion)

- Every family's runs share one NEAT config (all options except `num_inputs`) and one set of run parameters
  (except seed and commit).
- The config file each database run names (`run_params.neat_config`) gives every option equal to what the
  run stored; the two original networks equal `experiments/10seed/configs/{heart,monk2}.ini` (as
  `experiments/10seed/PREREG.md` states).
- Experiment 2 configs and run parameters are equal in `explaneat_dev` and `explaneat_v4`.
- The Connect-4 configs equal the config text stored when the selected run was imported into `explaneat_dev`.
- The Cox runs' config text equals `experiments/survival/configs/survival.ini` (inputs aside).
- Selection: each original network is its run's highest-fitness genome; the selected Connect-4 and synthetic
  seeds have the largest training AUC of their runs; the Cox selection rule is the one in `selected.json`.
- Experiment 3: fixed-epoch and early-stopping settings in the result CSVs equal the constants; every
  fine-tune is masked.
