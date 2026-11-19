#!/usr/bin/env bash
# Rebuild every Chapter 7 output in the Overleaf upload (thesis/scripts/build_overleaf_upload.sh) from
# committed results. See thesis/ch7/PROVENANCE.md for each output's chain.
#
#   bash thesis/scripts/reproduce_ch7.sh            # layer 3 only: generators from committed files (no DB, ~3 min)
#   bash thesis/scripts/reproduce_ch7.sh --with-db  # also layer 2: analyses and extractors that READ the DB
#                                                   # (read-only transactions; explaneat_v4 / explaneat_dev; ~20 min)
#
# Afterwards `git status` should show no .tex / .csv / .json change. Matplotlib figure PDFs embed a
# creation date, so those bytes change on every run (restore them with `git checkout -- <path>` if unchanged
# otherwise). The 60 appendix network PDFs are rebuilt only when their TikZ source hash changes.
#
# Never runs migrations or `explaneat setup`; never writes to a database.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1

if [[ "${1:-}" == "--with-db" ]]; then
  # ---------------------------------------------------------------- layer 2 (read-only DB)
  # Experiment 2 run metrics (seeded bootstrap, RandomState(42)) from the evolved genomes
  uv run python experiments/10seed/compute_runs.py --skip-paper                  # explaneat_dev -> runs.csv, runs_heart100.csv
  uv run python experiments/churn/compute_churn_runs.py --prefix churn100        # -> experiments/churn/runs_churn100.csv
  # Explaining minutes and annotation counts of the v4 records from the database (read-only explaneat_v4)
  DATABASE_URL=postgresql://localhost/explaneat_v4 uv run python experiments/10seed/explaining_time.py --dir explanations_v4
  # Explanation records (+ explaining_time_v4.csv) -> CSVs (no DB)
  (cd experiments/10seed && uv run python build_explanations_csv.py --dir explanations_v4 --suffix _v4)
  # Signal analyses (read-only explaneat_v4)
  V4="DATABASE_URL=postgresql://localhost/explaneat_v4 EXPLANEAT_RECORDS_DIR=experiments/10seed/explanations_v4"
  env $V4 uv run python experiments/churn_signals/catalogue.py                   # ~3 min
  env $V4 uv run python experiments/adult_signals/catalogue.py --refresh         # ~5 min
  env $V4 uv run python experiments/heart100_audit/audit.py                      # ~5 min (uses committed cache/heart40.*)
  env $V4 uv run python experiments/heart100_audit/summarise.py
  # Benchmark inputs and PropNEAT rows of the comparator suite
  uv run python experiments/benchmarks/export_data.py                            # read-only DB -> cache/*.npz (+ data_manifest.json)
  uv run python experiments/benchmarks/propneat.py                               # read-only DB -> results/propneat.json
  uv run --extra survival python experiments/benchmarks/survival.py             # ~80 s, no DB -> results/survival.json
  uv run python experiments/benchmarks/make_csv.py                               # -> benchmarks.csv
  uv run python experiments/ushape_saddle/references.py                          # -> references.json
  # Thesis-side DB caches
  uv run python thesis/ch7/exp1/extract_db.py                                    # explaneat_dev -> exp1/data/networks.json
                                                                                 # (+ coverage checks, explaining minutes)
  DATABASE_URL=postgresql://localhost/explaneat_v4 uv run python thesis/ch7/exp2/extract_db.py --refresh   # exp2/data/*
fi

# ------------------------------------------------------------------ layer 3 (generators; read files only)
uv run python thesis/ch7/exp1/make_tables.py            # exp1-networks.tex, exp1-benchmarks.tex
uv run python thesis/ch7/appendix/heart_splits_table.py # ch7-heart-splits.tex
uv run python thesis/ch7/exp2/make_assets.py            # exp2-benchmarks, exp2-runs-*, exp2-signals-*, exp2-signalticks-*,
                                                        # appendix_networks.tex + figures/exp2/networks/*.pdf, other exp2 assets
bash thesis/scripts/build_overleaf_upload.sh            # thesis/ch7/overleaf_upload/ + zip (screenshots copied as images)

# ------------------------------------------------------------------ layer 1 (expensive; NOT run here)
# Evolution (PYTHONHASHSEED=0 enforced by run_seed.py; torch threads 1; writes explaneat_dev):
#   for s in 0..9: PYTHONHASHSEED=0 uv run python experiments/10seed/run_seed.py adult $s
#                  PYTHONHASHSEED=0 uv run python experiments/10seed/run_seed.py heart100 $s
#                  PYTHONHASHSEED=0 uv run python experiments/10seed/run_seed.py churn $s --name churn100-s$s
#   (experiments/10seed/README.md, launch.sh, PREREG.md; explaneat_v4 is a createdb -T clone, V4_RUN.md)
# Experiment 1 networks: experiments/survival/cox_case/run_case.py, experiments/c4_capacity/, experiments/c4_line_features/,
#   experiments/ushape_saddle/, experiments/xor/run_xor.py; the AAAI paper genomes (experiments/aaai27/).
# Explanations (agent-authored by method, then Mike's verification in the review UI): V4_RUN.md; then
#   verify_record.py --out-dir explanations_v4 and `plain-language extract --dir plain_language_v4`.
# Comparator suite (~2.5 h; needs `interpret`, not in the uv env):
#   OMP_NUM_THREADS=2 <venv with interpret>/python experiments/benchmarks/classification.py   (skips models already in results/classification/)
#   OMP_NUM_THREADS=2 <venv with interpret>/python experiments/benchmarks/heart_splits.py     (~1 min) -> heart_splits.json
# Heart-40 cache for the Heart audit's recurrence column: experiments/heart100_audit/heart40_signals.py (explaneat_dev)
echo "done: compare with git status / git diff"
