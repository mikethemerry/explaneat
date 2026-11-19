# Chapter 7 figure candidates

A gallery of candidate figures to sit beside the Chapter 7 tables, for Mike to choose from. Nothing
here is in the thesis yet. `out/manifest.json` lists every candidate: `id`, `title`, `group`,
`pairs_with` (table labels), `what_it_shows`, `sources` (files read) and `status` (`new`, built
here, or `existing`, a thesis or experiment figure copied in unchanged so everything is in one place).

## Commands

```bash
# 1. read-only DB caches (data/*.json; ~5 min, explaneat_v4, every session READ ONLY)
PGOPTIONS='-c default_transaction_read_only=on' DATABASE_URL=postgresql://localhost/explaneat_v4 \
EXPLANEAT_RECORDS_DIR=experiments/10seed/explanations_v4 OMP_NUM_THREADS=1 \
uv run python thesis/ch7/figure_candidates/extract_db.py [--refresh]

# 2. the figures (files only; seconds)
uv run python thesis/ch7/figure_candidates/make_candidates.py [id ...]
```

The Heart split figures need `per_split` in `experiments/benchmarks/heart_splits.json`
(`experiments/benchmarks/heart_splits.py`, see that folder's README for the venv with `interpret`).

## Outputs

- `out/<id>.png`: 300 dpi, the thesis style of `thesis/ch7/exp2/make_assets.py` (rcParams, Okabe-Ito).
- `out/plotdata/<id>.json`: every plotted number of a new candidate.
- `out/manifest.json`.

## Provenance

Layer 2: `extract_db.py` (read-only) writes `data/adult_shapes.json` (Adult catalogue projection,
asserted equal to `matrix.csv`), `data/heart_families.json` (Heart-100 family contributions, asserted
equal to `family_presence.csv`) and `data/seed_agreement.json` (pairwise logit correlations). Layer 3:
`make_candidates.py` reads only committed result files and these caches; the literals in it are design
choices (the 0.03 / 0.10 thresholds of `explaneat/analysis/signal_presence.py`, grids, bins, labels,
colours). The tick-map counts are asserted equal to the tick tables' (`thesis/ch7/exp2/data/signalticks.json`).

Determinism (20 Nov 2026): two runs of `make_candidates.py` gave byte-identical PNGs, plot data and
manifest; two runs of `extract_db.py --refresh` gave byte-identical caches.
