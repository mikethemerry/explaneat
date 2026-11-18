# Thesis write-up

Everything here is write-up: the files the thesis includes, the Chapter 7 experiment book, the
scripts that turn experiment results into thesis tables, the planning notes and the paper sources.
The experiments themselves (code, configs, runs, results, verification) live in
`experiments/<study>/`; the library lives in `explaneat/`.

```
thesis/
  ch7/
    book/       Chapter 7 experiment book (build.py -> ch7_experiments.html; one card per experiment in data/)
    exp2/       Experiment 2 LaTeX tables and vector figures (make_assets.py), preview.tex / preview.pdf
    exports/    files the thesis \input{}s or \includegraphics{}s for Chapter 7 (README.md lists each with its source)
  scripts/      thesis_experiment_tables.py (Experiment 1-3 tables), sync_ch7_exports.py (copies experiment outputs into ch7/exports)
  notes/        THESIS_EXPERIMENTS.md (the experiment manifest), THESIS_FINAL_WORK.md (the final task list and decisions)
  papers/       LaTeX sources: explaneat.tex (the AAAI paper), beyond_intuition.tex + Supplementary.tex
```

## Dependency rule

`thesis/` may read from `experiments/` and the core code. `experiments/` and the core code never
read from or write into `thesis/`. Experiment generators therefore write their thesis tables and
figures into their own study folder, and `thesis/scripts/sync_ch7_exports.py` copies them into
`thesis/ch7/exports/`. Historical files (run records, logs, JSON results) may still name the old
paths listed in the appendix; nothing reads those strings as paths.

## Chapter 7: experiments

The manifest is `notes/THESIS_EXPERIMENTS.md`; the book (`ch7/book/`) has one card per row below,
with the exact source files and commits. Paths are relative to the repository root.

| Experiment | Where the code and data live | Thesis assets |
|---|---|---|
| 1.1 Heart, Adult, MONK-2: the AAAI paper genomes, recomputed | `experiments/aaai27/` (the paper's scripts; `results/`), `experiments/aaai27/recheck/` (recompute with the current code; `results/`), `experiments/reproducibility/` (offline bundle), `experiments/references/` (LR / GB baselines) | rows 1-3 of `ch7/exports/ch7_experiment1_cases.tex` |
| 1.2 Churn case | `experiments/churn_case/` | row of `ch7_experiment1_cases.tex` |
| 1.3 Cox survival case (GBSG2, WHAS500) | `experiments/survival/cox_case/` (`run_case.py`, `tables.py`) | `ch7/exports/ch7_cox_case_table.tex`, rows of `ch7_experiment1_cases.tex` |
| 1.3a Cox reference explanation | `experiments/survival/cox_case/reference_explanation.py`, `REFERENCE.md` | `ch7/exports/ch7_cox_reference_table.tex` |
| 1.4 XOR worked example | `experiments/xor/` (`verify_xor.py`, `make_xor_figures.py` -> `figures/`) | `ch7/exports/xor_ridge_valley.{pdf,png}`, `xor_gates.{pdf,png}` |
| 1.5 KRK boundary case | `experiments/complex_real/` | row of `ch7_experiment1_cases.tex` |
| 1.6 Connect-4 boundary case | `experiments/c4_capacity/` | row of `ch7_experiment1_cases.tex` |
| 1.7 Synthetic U-shape and saddle | `experiments/ushape_saddle/` | three rows of `ch7_experiment1_cases.tex` |
| 2.1 Ten-seed consistency, frozen-method v4 | `experiments/10seed/` (`RESULTS.md`, `PREREG.md`, `V4_RUN.md`, prompts [`PROMPTS_V4.md`](../experiments/10seed/PROMPTS_V4.md), `VERIFY_EXP2.md`, `make_figures.py`), `experiments/churn/`, `experiments/heart100_audit/`, `experiments/churn_signals/` | `ch7/exports/ch7_table.tex`, `structure_*.pdf`, `cost_*.pdf`, `test_auc_*.pdf`, `coverage_time.pdf`, `ch7_experiment2_consistency.tex`; all of `ch7/exp2/tables/exp2/` and `ch7/exp2/figures/exp2/` |
| 2.2 Rashomon counts | `experiments/references/` (`rashomon.py`, `rashomon_population.py`, `results/`) | Rashomon column of `ch7_experiment2_consistency.tex`, `exp2-headline` |
| 2.3 Three levels of explanation | `experiments/explanation_levels/` (`compute_levels.py`, `make_tables.py`) | `ch7/exports/ch7_levels_table.tex` |
| 2.4 Verification pack and signoff-v1 | `experiments/10seed/review/`, `experiments/10seed/verification/` | sign-off columns |
| 3.1 Rashomon proposition | `docs/signals-and-realisations/` | `ch7_experiment3_complexity.tex` |
| 3.2 Explanation complexity ladder | `experiments/explanation_barrier/` | `ch7_experiment3_complexity.tex` |
| 3.3 Three Connect-4 bases | `experiments/c4_line_features/` | `ch7_experiment3_complexity.tex` |
| 3.4 MNIST linear check and linear start | `experiments/mnist/` | `ch7_experiment3_complexity.tex` |
| 3.5 Species leads and MONK-2 grafting | `experiments/species_leads/`, `experiments/churn/` (`EXTENSIONS_RESULTS.md`, `GRAFT_F_G_RESULTS.md`) | book only |
| 3.6 Authoring-skill evaluation and errata | `experiments/authoring_v4/`, `experiments/errata/` | book only |
| 3.7 Verifier benchmark | `experiments/verifier_bench/` | book only |
| 3.8 Jumps and plateaus in lineages | `experiments/lineage_jumps/`, `experiments/lineage/` | book only |
| 3.9 Very long runs | `experiments/long_run/` | book only |
| 4 ExConNEAT (convolutional NEAT) | `experiments/conv_neat/` (and `explaneat/conv/`) | book only (cards `exp4`, `S.1`-`S.3`) |

## Regenerate

```bash
# Experiment outputs, each into its own study folder (see each study's README / RESULTS for runtimes and DBs)
uv run python experiments/10seed/make_figures.py
uv run python experiments/xor/verify_xor.py && uv run python experiments/xor/make_xor_figures.py
uv run python experiments/explanation_levels/make_tables.py
uv run python experiments/survival/cox_case/tables.py
PYTHONHASHSEED=0 uv run python experiments/survival/cox_case/reference_explanation.py
uv run python experiments/references/rashomon.py                  # no DB; baselines.py and rashomon_population.py need the DB

# Thesis side
uv run python thesis/scripts/sync_ch7_exports.py                  # copy the above into ch7/exports (--check to compare only)
uv run python thesis/scripts/thesis_experiment_tables.py          # ch7/exports/ch7_experiment{1,2,3}_*.tex
uv run python thesis/ch7/exp2/make_assets.py                      # ch7/exp2 tables and figures (--refresh: DB caches, read-only)
cd thesis/ch7/exp2 && latexmk -pdf preview.tex && latexmk -c

# The Chapter 7 book (see the docstring of build.py)
cd thesis/ch7/book
python3 extract.py <repo root>        # -> xdata/ (read-only DB; not committed)
python3 graphs.py <repo root> <db>    # -> graphs/ (not committed)
python3 e2_stats.py <repo root> > e2_stats.json
python3 build.py                      # -> ch7_experiments.html (not committed)
```

Figure PDFs written by matplotlib carry a creation date, so a rerun changes those bytes only.
On 18 Nov 2026 the tables `ch7_experiment2_consistency.tex`, `exp2_summary.md` and
`e2_stats.json` regenerate with the sign-off counts recorded since they were last built (the
committed copies show 0/10); everything else regenerated byte-identical after the move.

## Appendix: repository reorganisation (18 Nov 2026)

Moves were made with `git mv` (history follows with `git log --follow`). References in code and
Markdown were updated; JSON / CSV records and logs keep the paths they were written with.

| Old | New |
|---|---|
| `experiments/ch7_book/` | `thesis/ch7/book/` |
| `experiments/exp2_thesis/` | `thesis/ch7/exp2/` |
| `thesis_exports/ch7/` | `thesis/ch7/exports/` |
| `scripts/thesis/thesis_experiment_tables.py` | `thesis/scripts/thesis_experiment_tables.py` |
| `THESIS_EXPERIMENTS.md`, `THESIS_FINAL_WORK.md` | `thesis/notes/` |
| `papers/` | `thesis/papers/` |
| `scripts/thesis/{paper_recompute,paper_shap_summary,paper_coverage_options,heart_paper_trace,heart_formula_check}.py` | `experiments/aaai27/recheck/` |
| `results/thesis/{paper_recompute/,paper_recompute.md,paper_coverage_options.*,paper_corrections.md,heart_paper_genome.*,heart_formula_check.json}` | `experiments/aaai27/recheck/results/` |
| `scripts/thesis/{baselines,rashomon,rashomon_population}.py` | `experiments/references/` |
| `results/thesis/{baselines.*,rashomon*}` | `experiments/references/results/` |
| `results/thesis/heart_benchmark/` | `experiments/references/heart_benchmark/` |
| `scripts/aaai27/*` | `experiments/aaai27/` |
| `results/aaai27-*.md`, `results/monk2-topology.md`, `results/capgain-response.png` | `experiments/aaai27/results/` |
| `results/10seed-consistency.md` | `experiments/10seed/RESULTS.md` |
| `reproducibility/` | `experiments/reproducibility/` |
| `configs/` (Backache experiment configs) | `experiments/backache/configs/` |
| `scripts/{compare_backache_sources,update_backache_features,debug_forward_pass}.py`, `scripts/backache_pmlb.csv` | `experiments/backache/` |
| `API_REFERENCE.md`, `ANCESTRY_TRACKING.md`, `DATABASE_SETUP.md`, `DOCUMENTATION.md`, `EXPLANATIONS.md`, `GENOME_EXPLORER_FEATURES.md`, `GETTING_STARTED.md`, `LIVE_MONITORING.md`, `PLOTTING_FIXES.md`, `RUNNING_EXPERIMENTS.md` | `docs/` |
| `Compositional Explanation.md` | `docs/compositional_explanation.md` |
| `run_*.py`, `quickstart*.py`, `check_database.py`, `quick_db_check.py`, `debug_annotations.py`, root `*.cfg` except `config-file.cfg` | `legacy/` |
| root `test_*.py` | `legacy/tests/` |

Why `scripts/thesis` and `results/thesis` went to `experiments/` rather than `thesis/`: they are
experiments (recomputing the paper's numbers from the database; fitting reference models), and
`experiments/10seed/verify_exp2.py` reads `rashomon_counts.csv`, which the dependency rule
forbids from `thesis/`. `papers/` holds paper sources (write-up), so it sits with the thesis
rather than with the code documentation.

New experiment-side copies (so experiment generators no longer write into the thesis tree):
`experiments/xor/figures/xor_*.{pdf,png}`, `experiments/explanation_levels/ch7_levels_table.tex`,
`experiments/survival/cox_case/ch7_cox_case_table.tex`, `ch7_cox_reference_table.tex`.

Left at the root: `README.md`, `CLAUDE.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `pyproject.toml`,
`uv.lock`, `.python-version`, `alembic.ini`, `genome_explorer_cli.py` (documented root command,
tested by `tests/test_cli/`), `config-file.cfg` (opened from the working directory by
`explaneat/analysis/ancestry_analyzer.py`, `annotation_manager.py` and `coverage.py`),
`requirements.txt` (generated from `uv.lock` for pip users, per `README.md`), `examples/`
(library usage examples).
