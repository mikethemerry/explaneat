# Experiment 2: thesis tables and figures

LaTeX tables and vector figures for Experiment 2 (Adult, Churn, Heart-100; 10 seeds x 100
generations; frozen v4 explanations, all 30 verified by Mike). Every number is computed by
`make_assets.py` from the repository's result files; none is typed. Preview: `preview.pdf`.

## Regenerate

```
uv run python thesis/ch7/exp2/make_assets.py            # reuses the data/ caches
uv run python thesis/ch7/exp2/make_assets.py --refresh  # also recomputes them (read-only, explaneat_v4)
cd thesis/ch7/exp2 && latexmk -pdf preview.tex && latexmk -c
```

`extract_db.py` writes the five caches in `data/` that need the networks themselves. Every session
is a READ ONLY transaction on `explaneat_v4` (about 1 minute); nothing is written to any database.

## Using them in the thesis

Copy `tables/exp2/` to `tables/exp2/` under the thesis root. The figures (since 20 Nov 2026) live with every
other Chapter 7 figure in `thesis/ch7/figures/` (written through `thesis/ch7/figlib.py`, listed in
`thesis/ch7/FIGURES.md`), already in the Overleaf layout that `thesis/scripts/build_overleaf_upload.sh` zips:

```latex
\input{tables/exp2/exp2-headline}               % a table float
\input{sections/figures/used/exp2-auc}          % a figure float; \includegraphics{figures/ch7/exp2-auc.pdf} (figure files flat)
```

Only packages already in `dissertation.tex` are used: booktabs, tabularx, makecell, adjustbox,
rotating (`\rotatebox`), xcolor, amsmath. No siunitx. Figures are PDF (Computer Modern text,
fonts embedded) with a 300 dpi PNG beside each; they are drawn at 6.0 in wide, the print layout's
text width, so `width=\textwidth` keeps the font at about 8.5 pt. Colours are Okabe-Ito.

## Tables (`tables/exp2/`)

| file | label | content | sources |
|---|---|---|---|
| exp2-headline | tab:exp2-headline | per study: test AUC median (range), LR / GB, seeds within 0.01 of the best, hidden units, full coverage, recheck, agent minutes, Mike-verified and his minutes | `thesis/ch7/book/e2_stats.py` (run on this checkout), runs CSVs, v4 CSVs and records, `experiments/references/results/rashomon_summary.json` |
| exp2-heart-benchmark | tab:exp2-heart-benchmark | Heart on the paper split: the paper network, Heart-100 best / median / mean prediction, eight comparators, 95% intervals | `experiments/references/results/baselines.json`, `experiments/references/heart_benchmark/bench.json`, `runs_heart100.csv`, `data/paper_net.json`, `data/agreement.json` |
| exp2-heart-audit | tab:exp2-heart-audit | (a) full vs linearised AUC, train and test; (b) counts of non-linear units / parts | `experiments/heart100_audit/networks.csv`, `features.csv` |
| exp2-heart-families | tab:exp2-heart-families | thal (with ST depression), thal x age, BP, exang, age: seeds, direction, train vs test gain | `experiments/heart100_audit/features.csv` (unit rows) |
| exp2-linear-core | tab:exp2-linear-core | linear core across seeds: sign agreement, average effect and spread, LR | `experiments/heart100_audit/linear_core.csv` |
| exp2-s8-families | tab:exp2-s8-families | heart100-s8 resting-BP families: n, shape, defining condition, units | `experiments/feature_dossiers/heart100-s8_trestbps/dossier.json` |
| exp2-churn-signals | tab:exp2-churn-signals | the signal list by class (core / common / rare / absent; noise marked) | `experiments/churn_signals/signals.csv` |
| exp2-churn-matrix | tab:exp2-churn-matrix | seeds x main signals, SD and realisation (H / G / Sk, /np) | `experiments/churn_signals/matrix.csv` |
| exp2-churn-disagreement | tab:exp2-churn-disagreement | disagreement decomposition and agreement after harmonising | `experiments/churn_signals/disagreement.csv`, `harmonise.csv`, `data/agreement.json` |
| exp2-churn-blindspot | tab:exp2-churn-blindspot | international plan x international use: churn counts, ridge vs networks vs ensemble | `data/churn_blindspot.json`, `experiments/churn_signals/cache/meta.json`, `signals.csv`, `runs_churn100.csv` |
| exp2-agreement | tab:exp2-agreement | prediction agreement Churn vs Heart-100 (and Churn with states at average) | `data/agreement.json` (method of the 18 Nov `agree.py` / `agree2.py`, in `extract_db.py`) |
| exp2-betas-{adult,churn,heart100} | tab:exp2-betas-<set> | linear-term betas per seed (real units), median, LR reference; marks for opposite direction and linear-path disagreement (`linear_betas.py`; landscape) | `experiments/linear_betas/` (`betas.py`) |
| exp2-betas-adult-{household,work,education,country}, exp2-betas-churn-states | tab:exp2-betas-<set>-<group> | every level of the many-level categories, per seed (appendix) | `experiments/linear_betas/` |

## Figures (in `thesis/ch7/figures/figures/ch7/{used,options}/`, each `.pdf` + `.png`, float in `sections/figures/`)

Drawn by `make_assets.py` (below), `worked.py` (the worked example) and `exp2_figures.py` (the signal, completeness,
performance and verification-cost figures that began in `thesis/ch7/figure_candidates`); all are listed with
their captions in `thesis/ch7/FIGURES.md`.

| file | label | content | sources |
|---|---|---|---|
| exp2-auc | fig:exp2-auc | per-seed test AUC with 95% intervals, three panels, LR / GB lines, near-best band | e2_stats, rashomon_summary.json |
| exp2-cost | fig:exp2-cost | (a) agent minutes vs hidden units; (b) verification minutes per network | e2_stats, v4 records (`mike_verification`) |
| exp2-heart-linearised | fig:exp2-heart-linearised | full vs linearised AUC per seed, train and test | heart100_audit/networks.csv |
| exp2-s8-ice | fig:exp2-s8-ice | heart100-s8 resting-BP ICE curves (predicted risk), one panel per family + the family medians | `data/s8_ice.npz` (the dossier rebuilt with the same config; family sizes asserted equal) |
| exp2-heart-pairs | fig:exp2-heart-pairs | heart100-s2 whole-model pairs plot, five inputs, real units | `data/pairs_s2.json` (`viz_data.compute_pairs`, others at the training median) |
| exp2-churn-shapes | fig:exp2-churn-shapes | each signal's realised shape in the ten seeds vs a ridge logistic fit (the data's own) | `data/churn_shapes.json` (as `experiments/churn_signals/figures.py`) |
| exp2-churn-calls-parts | fig:exp2-churn-calls-parts | service calls split into explanation parts, seeds 0, 3, 4 | `experiments/churn_signals/cache/curves.json`, `consolidation.csv` |
| exp2-churn-s5-network | fig:exp2-churn-s5-network | churn100-s5 network, hidden units coloured by signal | `experiments/churn_signals/cache/graphs.json` |

## Notes

- Adult s3 has no verification minutes in its record, so Mike's minutes are over 9 Adult networks
  (marked in Table exp2-headline and exp2-runs-adult).
- Captions hold only a title and the keys needed to read the table or figure (Mike, 20 Nov 2026); the
  method and results sentences they used to carry are in `thesis/ch7/CAPTION_MOVED_TEXT.md`.
- The paper network's 95% interval (0.808 to 0.966) was not stored before; it is computed here with
  the runs CSVs' bootstrap (10,000 resamples, `RandomState(42)`).
- Blind spot: plan holders with neither few calls nor many minutes churn at 33 of 252 (13%) on the
  training rows. `CATALOGUE.md` says "about 25%"; the table prints the recomputed count.
