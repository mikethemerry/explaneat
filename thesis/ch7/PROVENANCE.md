# Chapter 7 provenance audit (20 Nov 2026)

Scope: every file in the Overleaf upload (`thesis/scripts/build_overleaf_upload.sh`) plus
`thesis/ch7/exp2/tables/exp2/exp2-runs-*.tex`. For each output the chain runs from the table back to the
run or database. Rebuild everything with `bash thesis/scripts/reproduce_ch7.sh` (add `--with-db` to also
rerun the read-only database layer).

**Status labels**
- **computed**: written by committed code from committed inputs.
- **extracted (DB, read-only)**: read from `explaneat_dev` or `explaneat_v4` in READ ONLY transactions, then computed.
- **agent-authored, verified by Mike**: written by an explaining agent by method (annotations, part names, hypotheses), then verified by Mike.
- **agent-reported (unverified)**: a number the agent wrote into its own record that no code checks.
- **human-entered**: typed by Mike himself, e.g. his verification minutes in the review UI.
- **curated (judgment)**: a vocabulary, catalogue, label or description chosen by an agent or a human and typed into a script. It is not a result, but it is not computed either.
- **UNKNOWN**: no writer found.

**Result of the check.** I reran layer 3 (the generators) from committed files. Every in-scope `.tex`
came out byte-identical, and no appendix network PDF was recompiled because every TikZ source hash
matched `figures/exp2/networks/sources.json`.

Layer 2 (read-only DB) was also rerun into scratch copies, and these reproduced byte-identical:
- `exp1/data/networks.json`, `exp2/data/structure_metrics.json`
- `benchmarks.csv`, `propneat.json`, `heart_splits.json`, `references.json`, `results/data_manifest.json`
- `runs_churn100.csv`, the three explanation CSVs
- every `churn_signals` and `heart100_audit` output

Three outputs reproduced with small, non-printed differences:
- **`survival.json`**: only the `seconds` timing fields differ.
- **`adult_signals/{signals,matrix}.csv`**: float noise, at most 1.4e-10. Every flag and code is identical.
- **`runs.csv` / `runs_heart100.csv`**: two bookkeeping columns differ (issue 6). Every column the thesis uses is identical.

No result file was found to be hand-made, and every number traced to an origin (no UNKNOWN links). The
non-computed parts are named below: the signal vocabulary and Mike's minutes. The explanation
records' self-reported fields (agent minutes, Experiment 1 coverage, annotation counts) were replaced
by read-only database checks on 20 Nov (issue 3, resolved; see "Computed records" below).


**Explanation records from the database (22 Nov 2026, branch feature/records-db-readers).** Mike's
decision: the database is the source of truth for the explanation records and his verifications
(`explanation_records`, migration `m6n7o8p9q0r1`, loaded with `explaneat records import`; each
verification appended to `explanation_record_events`). The record JSON files are caches written by
`explaneat records export` when wanted, committed only as deliberate milestone snapshots, not after each
verification; the committed ones stay as the historical snapshot. Every reader of a record goes through
`explaneat.analysis.explanation_records` (`load_record_file`, `load_record`, `find_record_for_genome`,
`list_records`): the row of `$EXPLANEAT_RECORDS_DATABASE_URL` (the records' own database), else of the
process's `DATABASE_URL`, when it holds the record, else the file. The readers: `thesis/ch7/exp2/make_assets.py`
(`record()`; `records()` for the headline, cost figure and `per_run.py`'s exp2-runs-* tables, also
`exp2_figures.py`), `thesis/ch7/exp2/worked.py` (the worked example's recheck and minutes row),
`thesis/ch7/exp1/extract_db.py` (`rec_*` and `mike`), `thesis/ch7/exp1/make_figures.py` (shape-C's genome),
`thesis/scripts/thesis_experiment_tables.py`, `thesis/ch7/book/e2_stats.py` and `extract.py`,
`experiments/10seed/explaining_time.py` and `build_explanations_csv.py`; the signal analyses already read
records through the library. So record-derived outputs reproduce from the database, like the other
layer-2 extractors: `reproduce_ch7.sh` passes the records database (default `explaneat_v4`, read only) to
the generators that read records. Without a records table (before the migration) every reader falls back
to the committed files: on 22 Nov the default `reproduce_ch7.sh`, `e2_stats.py` and
`thesis_experiment_tables.py` were byte-identical to the committed outputs, and a throwaway database
holding the imported records gave the same outputs.

**Captions (20 Nov 2026, branch thesis/simple-captions).** Every caption of the in-scope tables and of
the Experiment 2 figure floats now holds only a title and the keys needed to read it (what bold, ✓, w,
letters, daggers, brackets and -- mean; abbreviations). Method and results sentences were removed from
the generators; each table's and figure's old caption, new caption and removed sentences (grouped as
methodology / results) are in `thesis/ch7/CAPTION_MOVED_TEXT.md` for Mike's body text. Key numbers are
read from source: the ✓ / w thresholds from `explaneat/analysis/signal_presence.py` (`USED_SD`,
`WEAK_SD`), the "within 0.01" margin and the pairs plot's colour centre from `make_assets.NEAR_BEST` /
`RISK_CENTRE`, split and seed counts from the data. Everything outside the `\caption{...}` lines is
byte-identical. The appendix network captions were already a title only and are unchanged.
---

## ch7-ex1-networks (`thesis/ch7/exp1/tables/exp1-networks.tex`)

| Link | Writer / command | Status |
|---|---|---|
| `exp1-networks.tex` | `thesis/ch7/exp1/make_tables.py` | computed (byte-identical) |
| `thesis/ch7/exp1/data/networks.json` | `thesis/ch7/exp1/extract_db.py` (explaneat_dev, ~1 min) | extracted (DB, read-only); rerun byte-identical |
| structure: inputs, hidden, conns, depth, skips, density, skippiness | genome in explaneat_dev, `compute_runs.structure` and `ExplaNEAT` measures | extracted (DB, read-only) |
| splits, identity nodes, leaf + comp. counts | explanation operations in explaneat_dev (`explaining_time.annotation_counts`; also equal to `get_coverage`'s counts), asserted equal to the records (paper: to `paper_coverage_options.json`) | extracted (DB, read-only); all agree |
| coverage struct./comp., hierarchy completeness | MCP `get_coverage` (`coverage_report`) on each explanation, in `extract_db.py`, as `experiments/10seed/verify_record.py` does; check_explanation's stricter hierarchy check kept as `hierarchy_complete_check` | extracted (DB, read-only), computed. Agree with the six v4 records and paper Heart. **MONK-2: 1.00** (was printed 0.75): under the current definition (decision D4, 7 Nov, commit 1fb3a8b6) its pruned-input annotation is exempt from the single-root rule; 0.75 is the pre-D4 value in `paper_coverage_options.json` ("current"), now kept as `rec_cov_c` (no longer printed: the caption sentence moved to `CAPTION_MOVED_TEXT.md`). check_explanation's strict check still reports MONK-2's hierarchy incomplete (two roots) |
| agent min | `experiments/10seed/explaining_time.explaining_window` in `extract_db.py`: explanation created → last timestamped write (operations, evidence records, `evidence_log`) | extracted (DB, read-only), computed. 0.6–4.5 min below the record's own start/end (kept as `rec_agent` in `networks.json`); paper Heart / MONK-2 have no evidence log: "--" |
| function-preservation error (not printed since 21 Nov: the column was removed, see "Thesis table style" below) | `check_explanation.check` on the training split, in `extract_db.py`; equals the records and `experiments/explanation_levels/levels/*.json` exactly for all eight | extracted (DB, read-only), computed; internal check only |
| XOR row | `experiments/xor/xor_results.json` ← `experiments/xor/run_xor.py` | computed |
| train / test AUC and CI | `experiments/benchmarks/results/propneat.json` ← `propneat.py` (asserts equality with the run records to 1e-9) | extracted (DB, read-only); rerun byte-identical |
| train / test AUC and CI, original Adult network and Churn case (21 Nov) | `networks.json` `perf` ← `extract_db.py` (compute_runs `load_split` / `bootstrap_indices` / `performance`, the code of their run records), asserted equal to `experiments/10seed/runs_paper_reference.csv` (adult) and `experiments/churn_case/runs.csv`; `make_tables.py` switches to `propneat.json` once it holds them (asserted equal) | extracted (DB, read-only), computed; rerun byte-identical |
| WHAS500 and GBSG2 Harrell's C | `experiments/benchmarks/results/survival.json` ← `survival.py` (`make_tables.py` asserts its tag is the selected seed) | computed |
| genomes | the evolution runs of each study (layer 1) | computed (not rerun) |
| caption (title and keys only since 20 Nov; the old prose is in `CAPTION_MOVED_TEXT.md`) | typed in `make_tables.py` | prose, not results |

## ch7-ex1-benchmarks (`thesis/ch7/exp1/tables/exp1-benchmarks.tex`) and ch7-ex2-benchmarks (`thesis/ch7/exp2/tables/exp2/exp2-benchmarks.tex`)

| Link | Writer / command | Status |
|---|---|---|
| `exp1-benchmarks.tex` | `thesis/ch7/exp1/make_tables.py` via `benchmark_tables.py` | computed (byte-identical) |
| `exp2-benchmarks.tex` | `thesis/ch7/exp2/make_assets.py` via `exp1/benchmark_tables.py` | computed (byte-identical) |
| `experiments/benchmarks/benchmarks.csv` | `make_csv.py` (reads `results/classification/*.json`, `propneat.json`, `survival.json`; then the grid check) | computed; rerun byte-identical. Experiment 1 now also has the original Adult network and the Churn case (`propneat.py`, checked against `runs_paper_reference.csv` and `churn_case/runs.csv` to 1e-9); Adult and Churn comparator rows are experiment "1,2" |
| `results/classification/*.json` | `classification.py` (hours on Adult / Connect-4 cells; needs `interpret`) | computed. Seeds: 5-fold stratified CV seed 42, bootstrap `RandomState(42)`, every estimator seeded. Grids widened 21-22 Nov 2026 (`experiments/benchmarks/README.md`, Grids); the 19 Nov narrower-grid run is in `results/narrow_grids/`. Grid check `results/grid_check.md` (one choice at its edge: Connect-4 cells EBM interactions); before/after `results/before_after.md`; two fresh Heart runs identical (`results/determinism_heart.json`) |
| `cache/*.npz`, `results/data_manifest.json` | `export_data.py` | extracted (DB, read-only); rerun byte-identical |
| `results/propneat.json` | `propneat.py` from the runs CSVs and the DB | extracted (DB, read-only) |
| `results/survival.json` | `survival.py` (~20 min; widened grids) | computed (timings aside) |
| synthetic true-logit AUC (was in the caption; now in `CAPTION_MOVED_TEXT.md` for the body text) | `experiments/ushape_saddle/references.json` ← `references.py` | computed; rerun byte-identical |

All of these files were committed with their scripts in a8512b24.

## ch7-heart-splits (`thesis/ch7/appendix/tables/ch7-heart-splits.tex`)

| Link | Writer / command | Status |
|---|---|---|
| `ch7-heart-splits.tex` | `thesis/ch7/appendix/heart_splits_table.py` | computed (byte-identical); split count, sizes and seed now read from data (fixed) |
| `experiments/benchmarks/heart_splits.json` | `heart_splits.py` (~1 min; needs `interpret`) | computed; rerun byte-identical. The comparator settings were typed in; they are now read from `benchmarks.csv` (fixed) |
| `experiments/10seed/runs_heart100.csv` (PropNEAT median) | see exp2-runs below | extracted (DB, read-only) |
| `cache/heart.npz` | `export_data.py` | extracted (DB, read-only) |

## ch7-ex2-{adult,churn,heart} signal-tick tables (`exp2-signalticks-{adult,churn,heart100}.tex`)

*Updated 20 Nov 2026 (branch thesis/signal-checks): one presence rule for all three sets, every row description checked by code, and a completeness check. See issues 1, 8 and 9.*

| Link | Writer / command | Status |
|---|---|---|
| `exp2-signalticks-*.tex` | `thesis/ch7/exp2/signal_ticks.py` (called by `make_assets.py`) | computed |
| presence rule ("present", "weak", direction) | `explaneat/analysis/signal_presence.py`, called by all three analyses | computed; one rule (below) |
| non-linear rows | parsed from `exp2-signals-*.tex` ← `signal_tables.py` (`w` from the analyses' `presence` column) | computed |
| linear rows, Churn / Adult | `experiments/{churn,adult}_signals/matrix.csv` (`presence`, `direction`) | computed |
| linear rows, Heart | `experiments/heart100_audit/linear_core.csv` (`presence`, `direction`, `proj_coef`, `sd_contribution`) | computed |
| non-linear rows, Heart | `experiments/heart100_audit/family_presence.csv` (per seed and family) and `features.csv` | computed |
| row descriptions, all three | `experiments/signal_descriptions/describe.py` → `descriptions.json` (claims in `claims.csv`, changes in `rewrites.csv`) | **computed, checked**: each claim tested by code; numbers and counts filled from computed values |
| metric rows (AUC, structure, parts) | `per_run.study_rows`, the same sources as exp2-runs | computed / extracted / agent-authored (see exp2-runs) |
| `experiments/churn_signals/*.csv`, `cache/*` | `catalogue.py` (~3 min, explaneat_v4) | extracted (DB, read-only) |
| `experiments/adult_signals/*.csv`, `cache/*` | `catalogue.py --refresh` (~5 min, explaneat_v4) | extracted (DB, read-only) |
| `experiments/heart100_audit/{features,linear_core,family_presence,networks,recurrence}.csv` | `audit.py` (~5 min, explaneat_v4), then `summarise.py` | extracted (DB, read-only). Reads committed `cache/heart40.*` ← `heart40_signals.py` (explaneat_dev, not rerun; feeds only the "recurrence" notes) |
| `thesis/ch7/exp2/data/churn_shapes.json` (the "4 calls" and "300 minutes") | `extract_db.py churn_shapes.json --refresh` | extracted (DB, read-only); `signal_tables.churn_desc` recomputes both numbers from it and asserts they equal `descriptions.json` |
| completeness check (not in the upload) | `experiments/signal_completeness/{load,completeness,report}.py` → `RESULTS.md`; `thesis/ch7/exp2/completeness_table.py` → `exp2-signal-completeness.tex` | computed; run twice, byte-identical |
| inputs of the analyses | `experiments/10seed/runs*.csv`, `churn/runs_churn100.csv`, `explanations_v4/`, `plain_language_v4/` | see exp2-runs |

**One presence rule** (`explaneat/analysis/signal_presence.py`; design choices, the same for every row of every set):
- **Logit contribution of a signal in one network** (training rows):
  - Linear term: weight × input. The weight comes from the least-squares projection of the network's logit on the linear columns. For Adult and Churn this is the catalogue's linear block. For Heart it is a number's column or a category's non-reference levels (`audit.linear_basis`).
  - Adult and Churn non-linear signal: its block's component of the same projection.
  - Heart non-linear family: the full logit minus the logit with the family's switching units replaced by their affine fit on their own inputs.
- **Size**: SD after clipping to the 1st–99th percentiles (the Adult rule, applied to all three). Clipping keeps a term that is large on under 1% of rows from counting: country offsets on Adult, and the two Heart levels with 2 of 242 patients (ECG ST-T abnormality, thallium not recorded).
- **States**:
  - **present** (✓, counted in *n*): SD ≥ 0.10 logit.
  - **weak** (w, shown and not counted): 0.03–0.10, on linear and non-linear rows alike.
- **Read guard**: a network that reads none of a signal's input variables carries no weight on it. Without the guard, the Heart projection put weight on unread correlated inputs.
- **Direction** of a present linear term: `opp` when its contribution correlates ≤ 0 with the mean contribution of the seeds that use it ("opposite direction to most seeds"). On Adult and Churn this gives the same result as the earlier comparison with the data's ridge fit (0 opp either way). On Heart, the data's fit gives the downsloping ST slope a coefficient of −0.002, which would make the direction arbitrary.

**What changed with the shared rule**:
- Adult ticks: unchanged.
- Churn ticks: two non-linear ticks fall below 0.10 after clipping (calls × light day s4, clipped SD 0.095; evening × calls s7, 0.078).
- Heart: the linear rows change wherever the rule differs. The old rule ticked any non-zero average effect; this removes two rare-level rows. The non-linear families are now ticked only when their contribution passes the threshold.
- Weak cells are shown as w instead of ticks with a note.

The full before/after list is in the commit message of 49d0a2cb and the Part A report.

**The row descriptions are checked by code** (`experiments/signal_descriptions/describe.py`). Each description is split into its claims:
- Directions, thresholds and knots, sizes, which group is larger, interaction partners, counts such as "in 3 of 4 seeds", and "mostly".
- A claim about a signal is tested on the data's own fit (partial dependence of the ridge fit on the catalogue basis); how many using seeds agree is reported.
- A claim about Heart units is tested on `features.csv`: drivers, modifiers, and the new `on_profile` column (who each unit is on for).

Failing claims were rewritten to what the code shows (`rewrites.csv`, 21 rows). The texts are no longer typed in the thesis scripts: `FAM_ROWS` keeps only labels, and `churn_desc` reads `descriptions.json`. The Adult catalogue's own `signals.csv` description column is its draft and is no longer printed.

**Completeness, networks → catalogue** (`experiments/signal_completeness/RESULTS.md`). The criterion, fixed before running, has two parts:
- every network's residual logit (logit minus its projection on the catalogue signals) has a clipped SD below 0.10;
- a residual scan finds no candidate signal. The scan covers each variable and each variable pair, with a family-wise permutation test, and flags a structure that recurs in at least 2 networks or has SD ≥ 0.10 in any one.

No data set meets the criterion. The numbers and the candidate list are in RESULTS.md. For Heart, the list is given data blocks in `heart_basis.py`, a design choice of that check.

All randomness is seeded:
- `StratifiedKFold(5, shuffle=True, random_state=0)` and `KFold(5, shuffle=True, random_state=0)`
- bootstrap `RandomState(42)`
- permutations and partial-dependence rows `RandomState(0)`
- regime tree and gradient boosting `random_state=0`
- single threads

The following is **curated (judgment)**: chosen by an agent or by Mike and typed into scripts.
- **The signal catalogues.**
  - Adult: `SIGNALS` in `experiments/adult_signals/catalogue.py` (names and basis blocks).
  - Churn: `basis.py` `NAMES`, `common.py` `GROUPS`, and `catalogue.py` `THEMES` / `BLOCK_VARS` / `RESP_VARS`.
  - Heart, for the completeness check only: `experiments/signal_completeness/heart_basis.py`.
  - The knots and quantiles.

  The interaction candidates were chosen as "interactions the explanations' units read together", so the *candidate list* came from reading the agent-authored explanations. Each signal is then scored on the data alone (cross-validated relevance) and on each network by projection. The completeness check measures what this list misses.
- **Display labels.** `make_assets.SIG_NAME`, `SHORT_TERM`, the Heart `HEART_SHORT` in `signal_ticks.py`, and the `FAM_ROWS` labels.
- **Description templates.** The sentence shapes in `describe.py` are typed. Every factual part is tested, and every number or count in them is filled from a computed value.
- **The Heart family tags and grouping rules** in `audit.py`.
- **Thresholds (design choices).**
  - Presence: 0.10 / 0.03 logit, with clipping at 1%/99%.
  - Carrier: 0.05 SD and 20% of the top carrier.
  - Heart: `MIN_NOV` 0.5, `SMALL_N` 30.
  - "Small" slope in a description: data component SD < 0.25 logit.
  - Completeness: residual SD < 0.10; the scan's 19 permutations and structure SD ≥ 0.03.

## exp2-runs-{adult,churn,heart100} (`thesis/ch7/exp2/tables/exp2/exp2-runs-*.tex`)

| Link | Writer / command | Status |
|---|---|---|
| `exp2-runs-*.tex` | `thesis/ch7/exp2/per_run.py` (via `make_assets.py`) | computed (byte-identical); asserts genome ids agree across sources |
| `experiments/10seed/runs.csv` (Adult), `runs_heart100.csv` | `experiments/10seed/compute_runs.py --skip-paper` (explaneat_dev) | extracted (DB, read-only); bootstrap `RandomState(42)`, 10,000. Every thesis column reproduces; two bookkeeping columns do not (issue 6) |
| `experiments/churn/runs_churn100.csv` | `experiments/churn/compute_churn_runs.py --prefix churn100` | extracted (DB, read-only); rerun byte-identical |
| `explanations_v4.csv`, `explanations_heart100_v4.csv`, `churn/explanations_churn100_v4.csv` | `experiments/10seed/build_explanations_csv.py --dir explanations_v4 --suffix _v4` (no DB) | computed; rerun byte-identical |
| record field `verified_by_orchestrator` (coverage, function-preservation error, splits, identity nodes, `complete_ok`) | `experiments/10seed/verify_record.py` (`check_explanation`, `get_coverage` on explaneat_v4) | computed |
| CSV `n_leaf`, `n_composition` | `experiments/10seed/explaining_time.py` (annotate operations in explaneat_v4) → `explaining_time_v4.csv` → `build_explanations_csv.py`, which asserts them equal to the records' `n_leaf_annotations` / `n_composition_annotations` | extracted (DB, read-only); all 60 agree. The annotations themselves are agent-authored, verified by Mike |
| CSV `agent_minutes_db` → "Agent min" (exp2-runs-*, exp2-headline, exp2-cost) | `explaining_time.py` (read-only explaneat_v4): explanation created → last timestamped write → `explaining_time_v4.csv` → `build_explanations_csv.py` | extracted (DB, read-only), computed. The record's own start/end stays in the CSV as `claude_minutes` (audit only, not printed) |
| record `mike_verification` (verified, minutes) | review UI (`explaneat/api/routes/records.py`), committed ab99b365; since 22 Nov the database's `explanation_records` row is the source (read through `explanation_records.load_record_file`, see "Explanation records from the database" below) | human-entered by Mike. **Exception:** adult-s0 (6 min) and adult-s4 (5 min) were written by an agent "at Mike's instruction in chat on 2026-11-18" (issue 2); adult-s3 has no minutes (footnoted) |
| `thesis/ch7/exp2/data/structure_metrics.json` (density, skippiness, layers) | `extract_db.py structure_metrics.json --refresh` (explaneat_v4) | extracted (DB, read-only); rerun byte-identical |
| genomes | `PYTHONHASHSEED=0 uv run python experiments/10seed/run_seed.py {adult\|heart100\|churn} <seed>` (enforces the hash seed, seeds random/numpy/torch, 1 torch thread, float64) | computed (layer 1, not rerun; prefix reproduction checked by `verify_prefix_reproduction.py`) |

## appendix-exp2-networks (`thesis/ch7/exp2/appendix_networks.tex`) and its 60 PDFs

| Link | Writer / command | Status |
|---|---|---|
| `appendix_networks.tex`, `figures/exp2/networks/*.{tex,pdf}` | `thesis/ch7/exp2/networks.py` (via `make_assets.py`; `--refresh` recompiles with pdflatex) | computed; every PDF stamped with the sha256 of its TikZ source (`sources.json`), all matched |
| `experiments/10seed/plain_language_v4/<label>.json` | `plain-language extract --dir plain_language_v4` (V4_RUN.md step 3; explaneat_v4), commits 656e96de, 26533fe8 | extracted (DB, read-only). Graph and weights computed; part names and grouping agent-authored, verified by Mike. Not re-extracted in this audit |
| hidden units / enabled connections in the raw figure | asserted against the runs CSVs | computed |

## Screenshots (`thesis/ch7/screenshots/ch7-ws-0[0-8]-*.png`)

These are images of the explorer UI. They carry no generated numbers, and nothing regenerates them.

---

## Issues found (most serious first)

1. **Signal vocabulary and descriptions are curated, and part of the text is unchecked.** The tick
   tables' signals, names, Heart families and row descriptions are typed into scripts (see above).
   The interaction candidates came from reading the agent-authored explanations. The Churn
   (`churn_desc`) and Heart (`FAM_ROWS`) descriptions make claims no code checks, e.g. "for women in
   one", "mostly with ST depression". The ticks themselves are computed. *Fixed (20 Nov,
   thesis/signal-checks) for the descriptions:* `experiments/signal_descriptions/describe.py` tests
   every claim in all three data sets' descriptions and fills their numbers and counts. 21 texts were
   rewritten to what the code shows (`rewrites.csv`), e.g. "for women in one" became "on mostly for women
   in 2", and "mostly with ST depression" became "with ST depression in 9 of 11 units". The vocabulary itself stays
   curated; `experiments/signal_completeness/` now measures what it misses (no data set is complete by
   the stated criterion).
2. **Two of Mike's verification-minute values were written by an agent.** adult-s0 (6) and adult-s4
   (5) were recorded from chat. They feed the headline and cost figure, not the in-scope tables, but
   Mike should confirm them.
3. **Resolved 20 Nov (computed records, below).** *Was:* **"Agent min" and Experiment 1 coverage are self-reported by the explaining agent.** No code
   recomputes them. The function-preservation error is cross-checked; coverage in Experiment 1 is not.
   For Experiment 2, coverage comes from `verify_record.py` (computed). A read-only check against
   `evidence_log` first/last timestamps, or `check_explanation` inside `exp1/extract_db.py`, would
   close this.
4. **`lr_l1` (liblinear) had no `random_state`.** *Fixed*: `random_state=42` in
   `experiments/benchmarks/classification.py`. A seeded refit of lr_l1 on all nine sets chose the
   same `C`. Printed AUC (3 dp), interval (2 dp) and accuracy (3 dp) are unchanged; test AUC moved at
   most ~2.4e-5 and CV AUC ~3.6e-4. The committed `results/classification/*.json` lr_l1 entries
   still come from the unseeded fit, because `classification.py` skips existing models.
5. **Typed result literals.** *Fixed*:
   - `heart_splits.py` hyperparameters (C=0.1, GB settings, …) are now read from `benchmarks.csv`, and the printed 0.026 margin is now computed.
   - `heart_splits_table.py` header and caption (100, 303, 242/61, seed 1) are now read from `heart_splits.json` and the runs CSV.
   - The exp1-networks caption's MONK-2 coverage 0.75 is now read from `networks.json`.

   All outputs are byte-identical.
6. **Bookkeeping columns no longer reproduce.** `runs.csv` / `runs_heart100.csv` columns
   `n_rows_at_best_fitness` and `n_keys_tied_in_best_generation` differ on rerun for a few seeds
   (heart s2, monk2 s1/s2/s5/s7, heart100 s6), against both DBs. The committed copies (5 Nov) give
   smaller counts. No thesis table uses them. *Not changed.*
7. **The comparator suite needs a second venv.** `classification.py` and `heart_splits.py` need
   `interpret`, which is not in the uv environment; the README pointed at another worktree's venv.
   *Not changed (dependency decision).*
8. **The tick-table definitions of "present" differ.**
   - Churn and Adult linear rows: component SD ≥ 0.10 logit (Adult after 1st–99th percentile clipping).
   - Heart linear rows: any non-zero average effect.
   - Weak non-linear uses: ticked with a note and counted in *n*.

   The caption says "present" uniformly. *Fixed (20 Nov, thesis/signal-checks):* one rule for all
   three (`explaneat/analysis/signal_presence.py`): clipped contribution SD ≥ 0.10 logit is present;
   0.03–0.10 is shown as w and not counted. The rule also adds a read guard and direction against
   most seeds. See the signal-tick section for what changed.
9. **Input-table labelling bug in `exp2-signals-adult.tex` (not in the upload).** The "w" mark tests
   raw `sd_train`, but "used" is decided on the clipped SD. So four linear rows show w with raw SD
   > 0.10 (s0, s6, s8 country; s2 census weight). The tick tables take linear rows from `matrix.csv`
   and are unaffected. *Fixed:* w now comes from the shared rule's `presence` column (clipped SD).
10. **Minor.**
    - `extract_db.churn_shapes` uses `0.27 * imin` instead of the catalogue's ratio (plot grid only).
    - `basis.py` names a signal "bend at about 4 calls" (overridden for printing).
    - `signal_ticks.py` parses generated LaTeX rather than the CSVs (linear rows and presence now come
      from the CSVs; the non-linear realisation letters are still parsed).
    - `extract_db.ensure` silently reuses stale caches without `--refresh`.
    - `exp1/extract_db.py` takes the latest explanation by `updated_at`.
    - No pickle caches are committed, so the signal analyses always need `explaneat_v4`.
    - Seeds differ by script (0 for GB/GBS and the signal CVs, 42 elsewhere) but are all fixed.

## What was changed in this audit
- `experiments/benchmarks/classification.py`: `lr_l1` seeded.
- `experiments/benchmarks/heart_splits.py`: settings and fixed-split AUCs read from `benchmarks.csv`; margin computed; docstring names the JSON output and the command.
- `thesis/ch7/appendix/heart_splits_table.py`: split count, sizes and seed read from data.
- `thesis/ch7/exp1/make_tables.py`: MONK-2 compositional coverage read from data.
- `experiments/benchmarks/README.md`: `heart_splits.py` command; skip-existing behaviour; lr_l1 seeding and rerun result.
- New: `thesis/scripts/reproduce_ch7.sh`, this file.

## For Mike to decide
- Confirm adult-s0 (6 min) and adult-s4 (5 min).
- How to name the curated signal vocabulary in the thesis, and whether to check or assert the Churn and Heart row descriptions.
- ~~Whether to state "Agent min" and Experiment 1 coverage as agent-reported, or add the read-only recomputes.~~ Recomputed (20 Nov).
- MONK-2's compositional coverage now prints 1.00 under the D4 exemption (the 0.75 without it is in `CAPTION_MOVED_TEXT.md`, no longer in the caption); check the Chapter 7 prose that cites 0.75.
- The book (`thesis/ch7/book/e2_stats.py`, `data/exp2.py`) and `thesis/scripts/thesis_experiment_tables.py` (`ch7_experiment2_consistency.tex`) still use the record minutes (`claude_minutes`); switch them to `agent_minutes_db` if they stay in the thesis.
- Whether to refit and commit the seeded lr_l1 results (printed values would not change).
- Whether to add `interpret` to a uv extra.
- Whether to regenerate and commit `runs.csv` / `runs_heart100.csv` with the current bookkeeping columns.
- Signal checks (thesis/signal-checks):
  - Whether to keep the clipping rule on Churn. It moves two borderline interaction seeds just below 0.10.
  - The projection weight can differ in sign from a network's own average effect when non-linear units shadow an input. Example: Heart s0 maximum heart rate is now "opp".
  - The Heart data blocks used by the completeness check (`heart_basis.py`).
  - The candidate signals in `experiments/signal_completeness/RESULTS.md` (findings; nothing was added to a catalogue).

## Computed records (20 Nov 2026, branch thesis/computed-records)

The explaining agent's self-reported numbers in the Chapter 7 tables are replaced by read-only
database checks (every process runs with `PGOPTIONS=-c default_transaction_read_only=on`).

**Explaining minutes.** Definition (`experiments/10seed/explaining_time.py`): from the explanation's
creation (`explanations.created_at`, the agent's first operation) to the last timestamped write the
explaining tools made: an operation's `created_at`, an evidence record's `timestamp`, `amended_at`,
`history[].replaced_at` or `deleted_at`, or an `evidence_log` entry. Left out as later writers:
`provenance_fixed_at` (`fix_amended_provenance.py`), evidence saved through the explorer API (on churn100-s5,
a week later), `explanations.updated_at` (bumped by later edits on several explanations, hours to days
after the record's end), and `evidence_log` entries before the explanation existed (one runner
`compute_performance` call on WHAS500). It is a lower bound on wall-clock time: reading before the
first write and the closing recheck and record writing are not in it. For all 66 explanations
(60 v4 records + six Experiment 1) no counted write falls after the record's end, and the computed time is
0.6 to 4.5 minutes below the record's start/end (`explaining_time_v4.csv` `difference_minutes`;
`networks.json` `rec_agent` vs `agent`). The paper explanations (Heart, MONK-2) predate the evidence
log, their operations span months, and they keep "--".

**Experiment 1 coverage.** `get_coverage` and `check_explanation.check` (verify_record.py's two checks)
run in `thesis/ch7/exp1/extract_db.py`. Coverage, hierarchy completeness and the function-preservation
error agree with the six v4 records exactly; MONK-2 is the one change (see the ch7-ex1-networks table).

**Annotation counts.** Leaf and composition `annotate` operations in the stored stream agree with every
record (60 v4, six Experiment 1) and with the paper files; the tables print the database counts.

**Cells that changed.** `exp1-networks.tex`: Agent min of the six v4 rows; MONK-2 coverage 1.00/0.75 →
1.00/1.00; the caption (definition of agent min, MONK-2 sentence). `exp2-runs-{adult,churn,heart100}.tex`:
every Agent min cell and its median (range); the caption's definition. `exp2-headline.tex`: the "Agent
explaining minutes" row and a caption sentence. `exp2-cost` figure (panel a) and its caption range. Every
other generated `.tex` is byte-identical; `networks.json`, `explaining_time_v4.csv` and the v4 explanation
CSVs reproduce byte-identically on a second run (old CSV columns unchanged; three columns appended).

## Chapter 7 figures (20 Nov 2026, branch thesis/kept-figures)

Every figure of the candidate gallery is now a thesis figure: a vector PDF (+ 300 dpi PNG) and a generated float, written through `thesis/ch7/figlib.py` into `thesis/ch7/figures/` in the Overleaf layout (`figures/ch7/<id>.pdf`, flat since 21 Nov 2026, Mike's Overleaf; `sections/figures/<group>/<id>.tex`; both in the upload). Group **used** (Mike's picks, `figlib.USED`) or **options**. Index: `thesis/ch7/FIGURES.md`. Layer 3 only reads files; the DB caches are layer 2 (`reproduce_ch7.sh --with-db`). Every PDF is written without a date, so two runs of the generators gave byte-identical PDFs, PNGs, plot data (`thesis/ch7/figures/plotdata/`) and floats (checked 20 Nov 2026, all 59). The Experiment 2 figures that were in `thesis/ch7/exp2/figures/exp2/` moved here (same labels, one file each); their PNGs are pixel-identical to the old ones except exp2-churn-shapes (legend wording) and exp2-s8-ice (new layout). Captions: title and keys only; read thresholds (`signal_presence.USED_SD` / `WEAK_SD`, `bench_figures.XCLIP`, split counts) from code or data.

| Group | Figure (label) | Generator (command) | Reads | Status |
|---|---|---|---|---|
| used | `bench-dots` (`fig:ch7-bench-dots`) | `thesis/ch7/exp1/bench_figures.py` (`uv run python thesis/ch7/exp1/make_figures.py`) | `experiments/benchmarks/benchmarks.csv` | computed |
| used | `exp2-auc` (`fig:exp2-auc`) | `thesis/ch7/exp2/make_assets.py` (`uv run python thesis/ch7/exp2/make_assets.py`) | `thesis/ch7/book/e2_stats.py` (runs CSVs: `experiments/10seed/runs.csv`, `runs_heart100.csv`, `experiments/churn/runs_churn100.csv`); `experiments/references/results/rashomon_summary.json` (LR, GB) | computed |
| used | `exp2-churn-calls-parts` (`fig:exp2-churn-calls-parts`) | `thesis/ch7/exp2/make_assets.py` (`uv run python thesis/ch7/exp2/make_assets.py`) | `experiments/churn_signals/cache/curves.json`, `consolidation.csv` | computed |
| used | `exp2-churn-shapes` (`fig:exp2-churn-shapes`) | `thesis/ch7/exp2/make_assets.py` (`uv run python thesis/ch7/exp2/make_assets.py`) | `thesis/ch7/exp2/data/churn_shapes.json` (`exp2/extract_db.py` from `experiments/churn_signals/figures.load`: basis, network betas and the ridge logistic fit `beta_data` of `cache/betas.npz`, C in `cache/meta.json`) | computed (caches of catalogue.py) |
| used | `exp2-s8-ice` (`fig:exp2-s8-ice`) | `thesis/ch7/exp2/make_assets.py` (`uv run python thesis/ch7/exp2/make_assets.py`) | `thesis/ch7/exp2/data/s8_ice.npz` (`exp2/extract_db.py`, the dossier's ICE and families); `experiments/feature_dossiers/heart100-s8_trestbps/dossier.json` (family sizes, asserted) | extracted (DB, read-only explaneat_v4), computed |
| used | `exp2-worked-ice` (`fig:exp2-worked-ice`) | `thesis/ch7/exp2/worked.py` (`uv run python thesis/ch7/exp2/make_assets.py`) | `thesis/ch7/exp2/data/worked_churn100_s5.json` (`worked.py`, read-only explaneat_v4); `experiments/10seed/plain_language_v4/churn100-s5.json` | extracted (DB, read-only explaneat_v4), computed |
| used | `exp2-worked-net-concept` (`fig:exp2-worked-net-concept`) | `thesis/ch7/exp2/worked.py` (`uv run python thesis/ch7/exp2/make_assets.py`) | `thesis/ch7/exp2/data/worked_churn100_s5.json` (`worked.py`, read-only explaneat_v4); `experiments/10seed/plain_language_v4/churn100-s5.json` | extracted (DB, read-only explaneat_v4), computed |
| used | `exp2-worked-net-exploded` (`fig:exp2-worked-net-exploded`) | `thesis/ch7/exp2/worked.py` (`uv run python thesis/ch7/exp2/make_assets.py`) | `thesis/ch7/exp2/data/worked_churn100_s5.json` (`worked.py`, read-only explaneat_v4); `experiments/10seed/plain_language_v4/churn100-s5.json` | extracted (DB, read-only explaneat_v4), computed |
| used | `exp2-worked-net-unexplained` (`fig:exp2-worked-net-unexplained`) | `thesis/ch7/exp2/worked.py` (`uv run python thesis/ch7/exp2/make_assets.py`) | `thesis/ch7/exp2/data/worked_churn100_s5.json` (`worked.py`, read-only explaneat_v4); `experiments/10seed/plain_language_v4/churn100-s5.json` | extracted (DB, read-only explaneat_v4), computed |
| used | `exp2-worked-pairs` (`fig:exp2-worked-pairs`) | `thesis/ch7/exp2/worked.py` (`uv run python thesis/ch7/exp2/make_assets.py`) | `thesis/ch7/exp2/data/worked_churn100_s5.json` (`worked.py`, read-only explaneat_v4); `experiments/10seed/plain_language_v4/churn100-s5.json` | extracted (DB, read-only explaneat_v4), computed |
| used | `seed-agreement` (`fig:ch7-seed-agreement`) | `thesis/ch7/exp2/exp2_figures.py` (`uv run python thesis/ch7/exp2/make_assets.py`) | thesis/ch7/figure_candidates/data/seed_agreement.json (extract_db.py, read-only explaneat_v4) | extracted (DB, read-only explaneat_v4), computed |
| used | `sig-shapes-adult` (`fig:ch7-sig-shapes-adult`) | `thesis/ch7/exp2/exp2_figures.py` (`uv run python thesis/ch7/exp2/make_assets.py`) | thesis/ch7/figure_candidates/data/adult_shapes.json (extract_db.py: experiments/adult_signals/catalogue.py basis and projection, read-only explaneat_v4); `experiments/adult_signals/matrix.csv`; `experiments/adult_signals/cache/betas.npz` | extracted (DB, read-only explaneat_v4), computed |
| used | `synthetic-shape-c-saddle` (`fig:ch7-synthetic-shape-c-saddle`) | `thesis/ch7/exp1/make_figures.py` (`uv run python thesis/ch7/exp1/make_figures.py synthetic-shape-c-saddle`) | `thesis/ch7/exp1/data/shapes.json` (`thesis/ch7/exp1/extract_shapes.py`, read-only explaneat_dev, via `experiments/ushape_saddle/shape_plots.compute` / `score_gens.surfaces`; summaries asserted equal to `figures/<label>-shape.json` / `sig_gens.json`); truth constants from `experiments/ushape_saddle/data.py`. Row (d), the parts carrying the term: every part's range on the grid from `experiments/ushape_saddle/figures/shape-C-shape.json` (selection rule `term_parts`); weights from the committed layer `experiments/ushape_saddle/plain_language/shape-C.json`, asserted to be genome a1cbdd4a (the panels' genome, `explanations/shape-C.json`, `runs.csv`) and, by `thesis/ch7/exp1/data/network_figs.json` (`extract_networks.py`, read-only), the genome's current explanation with the database's weights; seed from `runs.csv` | extracted (DB, read-only explaneat_dev), computed |
| used | `synthetic-shape-c-sodium` (`fig:ch7-synthetic-shape-c-sodium`) | `thesis/ch7/exp1/make_figures.py` (`uv run python thesis/ch7/exp1/make_figures.py synthetic-shape-c-sodium`) | `thesis/ch7/exp1/data/shapes.json` (`thesis/ch7/exp1/extract_shapes.py`, read-only explaneat_dev, via `experiments/ushape_saddle/shape_plots.compute` / `score_gens.surfaces`; summaries asserted equal to `figures/<label>-shape.json` / `sig_gens.json`); truth constants from `experiments/ushape_saddle/data.py`. Row (c), the parts carrying the term: ranges of the parts' sodium sweeps in `shapes.json` (asserted equal to `figures/shape-C-shape.json`; selection rule `term_parts`); weights from the committed layer `experiments/ushape_saddle/plain_language/shape-C.json`, asserted to be genome a1cbdd4a (the panels' genome) and, by `thesis/ch7/exp1/data/network_figs.json` (`extract_networks.py`, read-only), the genome's current explanation with the database's weights; seed from `runs.csv` | extracted (DB, read-only explaneat_dev), computed |
| used | `tickmap-adult` (`fig:ch7-tickmap-adult`) | `thesis/ch7/exp2/exp2_figures.py` (`uv run python thesis/ch7/exp2/make_assets.py`) | `experiments/adult_signals/matrix.csv`; `experiments/adult_signals/signals.csv`; thesis/ch7/exp2/data/signalticks.json (counts asserted equal) | computed |
| used | `tickmap-churn` (`fig:ch7-tickmap-churn`) | `thesis/ch7/exp2/exp2_figures.py` (`uv run python thesis/ch7/exp2/make_assets.py`) | `experiments/churn_signals/matrix.csv`; `experiments/churn_signals/signals.csv`; thesis/ch7/exp2/data/signalticks.json (counts asserted equal) | computed |
| used | `tickmap-heart100` (`fig:ch7-tickmap-heart100`) | `thesis/ch7/exp2/exp2_figures.py` (`uv run python thesis/ch7/exp2/make_assets.py`) | `experiments/heart100_audit/linear_core.csv`; `experiments/heart100_audit/family_presence.csv`; thesis/ch7/exp2/signal_ticks.py (HEART_SHORT labels); thesis/ch7/exp2/signal_tables.py (FAM_ROWS labels); thesis/ch7/exp2/data/signalticks.json (counts asserted equal) | computed |
| used | `train-vs-test-auc` (`fig:ch7-train-vs-test-auc`) | `thesis/ch7/exp2/exp2_figures.py` (`uv run python thesis/ch7/exp2/make_assets.py`) | `experiments/10seed/runs.csv`; `experiments/churn/runs_churn100.csv`; `experiments/10seed/runs_heart100.csv`; `experiments/benchmarks/benchmarks.csv` | computed |
| used | `xor-ridge-valley` (`fig:ch7-xor-ridge-valley`) | `thesis/ch7/exp1/make_figures.py` (`uv run python thesis/ch7/exp1/make_figures.py xor-ridge-valley`) | `experiments/xor/xor_results.json` (the two minimal networks of the Experiment 1 appendix, picked by `thesis/ch7/exp1/networks.py` `xor_runs`); surfaces drawn by `experiments/xor/make_xor_figures.py` `draw_ridge_valley_column` from exactly the drawn weights and biases (asserted equal to `verify_xor.evolved_minimal`, stored logits reproduced) | computed |
| options | `10seed-cost_prereg` (`fig:ch7-10seed-cost_prereg`) | `thesis/ch7/exp1/make_figures.py` (`uv run python thesis/ch7/exp1/make_figures.py`) | `experiments/10seed/make_figures.py` drawing code and its committed CSVs | computed; **legacy**: its LR / GB reference lines are constants typed in that script |
| options | `10seed-cost_supp` (`fig:ch7-10seed-cost_supp`) | `thesis/ch7/exp1/make_figures.py` (`uv run python thesis/ch7/exp1/make_figures.py`) | `experiments/10seed/make_figures.py` drawing code and its committed CSVs | computed; **legacy**: its LR / GB reference lines are constants typed in that script |
| options | `10seed-coverage_time` (`fig:ch7-10seed-coverage_time`) | `thesis/ch7/exp1/make_figures.py` (`uv run python thesis/ch7/exp1/make_figures.py`) | `experiments/10seed/make_figures.py` drawing code and its committed CSVs | computed; **legacy**: its LR / GB reference lines are constants typed in that script |
| options | `10seed-structure_prereg` (`fig:ch7-10seed-structure_prereg`) | `thesis/ch7/exp1/make_figures.py` (`uv run python thesis/ch7/exp1/make_figures.py`) | `experiments/10seed/make_figures.py` drawing code and its committed CSVs | computed; **legacy**: its LR / GB reference lines are constants typed in that script |
| options | `10seed-structure_supp` (`fig:ch7-10seed-structure_supp`) | `thesis/ch7/exp1/make_figures.py` (`uv run python thesis/ch7/exp1/make_figures.py`) | `experiments/10seed/make_figures.py` drawing code and its committed CSVs | computed; **legacy**: its LR / GB reference lines are constants typed in that script |
| options | `10seed-test_auc_prereg` (`fig:ch7-10seed-test_auc_prereg`) | `thesis/ch7/exp1/make_figures.py` (`uv run python thesis/ch7/exp1/make_figures.py`) | `experiments/10seed/make_figures.py` drawing code and its committed CSVs | computed; **legacy**: its LR / GB reference lines are constants typed in that script |
| options | `10seed-test_auc_supp` (`fig:ch7-10seed-test_auc_supp`) | `thesis/ch7/exp1/make_figures.py` (`uv run python thesis/ch7/exp1/make_figures.py`) | `experiments/10seed/make_figures.py` drawing code and its committed CSVs | computed; **legacy**: its LR / GB reference lines are constants typed in that script |
| options | `bench-gap` (`fig:ch7-bench-gap`) | `thesis/ch7/exp1/bench_figures.py` (`uv run python thesis/ch7/exp1/make_figures.py`) | `experiments/benchmarks/benchmarks.csv` | computed |
| options | `completeness-residual` (`fig:ch7-completeness-residual`) | `thesis/ch7/exp2/exp2_figures.py` (`uv run python thesis/ch7/exp2/make_assets.py`) | `experiments/signal_completeness/output_level.csv`; `experiments/signal_completeness/summary.json` | computed |
| options | `completeness-shares` (`fig:ch7-completeness-shares`) | `thesis/ch7/exp2/exp2_figures.py` (`uv run python thesis/ch7/exp2/make_assets.py`) | `experiments/signal_completeness/output_level.csv`; experiments/signal_completeness/heart_basis.py (Heart basis) | computed |
| options | `cost-minutes-size` (`fig:ch7-cost-minutes-size`) | `thesis/ch7/exp1/bench_figures.py` (`uv run python thesis/ch7/exp1/make_figures.py`) | `thesis/ch7/exp1/data/networks.json`; `experiments/10seed/explanations_v4.csv`; `experiments/churn/explanations_churn100_v4.csv`; `experiments/10seed/explanations_heart100_v4.csv`; `experiments/10seed/runs.csv`; `experiments/churn/runs_churn100.csv`; `experiments/10seed/runs_heart100.csv` | computed |
| options | `cost-verify` (`fig:ch7-cost-verify`) | `thesis/ch7/exp2/exp2_figures.py` (`uv run python thesis/ch7/exp2/make_assets.py`) | experiments/10seed/explanations_v4/*.json (mike_verification); `experiments/10seed/explanations_v4.csv`; `experiments/churn/explanations_churn100_v4.csv`; `experiments/10seed/explanations_heart100_v4.csv` | computed; verification minutes human-entered |
| options | `cox-whas500-forest` (`fig:ch7-cox-whas500-forest`) | `thesis/ch7/exp1/make_figures.py` (`uv run python thesis/ch7/exp1/make_figures.py`) | `experiments/survival/cox_case/reference_explanation.json` (`reference_explanation.py`) | computed |
| options | `cox-whas500-h2d` (`fig:ch7-cox-whas500-h2d`) | `thesis/ch7/exp1/make_figures.py` (`uv run python thesis/ch7/exp1/make_figures.py`) | `experiments/survival/cox_case/reference_explanation.json` (`reference_explanation.py`) | computed |
| options | `cox-whas500-pd` (`fig:ch7-cox-whas500-pd`) | `thesis/ch7/exp1/make_figures.py` (`uv run python thesis/ch7/exp1/make_figures.py`) | `experiments/survival/cox_case/reference_explanation.json` (`reference_explanation.py`) | computed |
| options | `exp2-churn-s5-network` (`fig:exp2-churn-s5-network`) | `thesis/ch7/exp2/make_assets.py` (`uv run python thesis/ch7/exp2/make_assets.py`) | `experiments/churn_signals/cache/graphs.json` | computed |
| options | `exp2-cost` (`fig:exp2-cost`) | `thesis/ch7/exp2/make_assets.py` (`uv run python thesis/ch7/exp2/make_assets.py`) | `e2_stats.py` (hidden units); v4 explanation CSVs `agent_minutes_db`; the records' `mike_verification.minutes` | computed; verification minutes human-entered |
| options | `exp2-heart-linearised` (`fig:exp2-heart-linearised`) | `thesis/ch7/exp2/make_assets.py` (`uv run python thesis/ch7/exp2/make_assets.py`) | `experiments/heart100_audit/networks.csv` | computed |
| options | `exp2-heart-pairs` (`fig:exp2-heart-pairs`) | `thesis/ch7/exp2/make_assets.py` (`uv run python thesis/ch7/exp2/make_assets.py`) | `thesis/ch7/exp2/data/pairs_s2.json` (`exp2/extract_db.py`, `viz_data.compute_pairs`) | extracted (DB, read-only explaneat_v4), computed |
| options | `heart-lincore-vs-data` (`fig:ch7-heart-lincore-vs-data`) | `thesis/ch7/exp2/exp2_figures.py` (`uv run python thesis/ch7/exp2/make_assets.py`) | `experiments/heart100_audit/linear_core.csv`; thesis/ch7/exp2/signal_ticks.py (HEART_SHORT labels) | computed |
| options | `heart-splits-box` (`fig:ch7-heart-splits-box`) | `thesis/ch7/exp1/bench_figures.py` (`uv run python thesis/ch7/exp1/make_figures.py`) | `experiments/benchmarks/heart_splits.json`; `experiments/10seed/runs_heart100.csv` | computed |
| options | `heart-splits-diff` (`fig:ch7-heart-splits-diff`) | `thesis/ch7/exp1/bench_figures.py` (`uv run python thesis/ch7/exp1/make_figures.py`) | `experiments/benchmarks/heart_splits.json` | computed |
| options | `heart-units-gain` (`fig:ch7-heart-units-gain`) | `thesis/ch7/exp2/exp2_figures.py` (`uv run python thesis/ch7/exp2/make_assets.py`) | `experiments/heart100_audit/features.csv`; thesis/ch7/exp2/signal_tables.py (FAM_ROWS labels) | computed |
| options | `lincore-adult` (`fig:ch7-lincore-adult`) | `thesis/ch7/exp2/exp2_figures.py` (`uv run python thesis/ch7/exp2/make_assets.py`) | `experiments/adult_signals/matrix.csv`; `experiments/adult_signals/signals.csv` | computed |
| options | `lincore-churn` (`fig:ch7-lincore-churn`) | `thesis/ch7/exp2/exp2_figures.py` (`uv run python thesis/ch7/exp2/make_assets.py`) | `experiments/churn_signals/matrix.csv`; `experiments/churn_signals/signals.csv` | computed |
| options | `lincore-heart100` (`fig:ch7-lincore-heart100`) | `thesis/ch7/exp2/exp2_figures.py` (`uv run python thesis/ch7/exp2/make_assets.py`) | `experiments/heart100_audit/linear_core.csv`; `experiments/heart100_audit/family_presence.csv`; thesis/ch7/exp2/signal_ticks.py (HEART_SHORT labels); thesis/ch7/exp2/signal_tables.py (FAM_ROWS labels) | computed |
| options | `linearised-auc` (`fig:ch7-linearised-auc`) | `thesis/ch7/exp2/exp2_figures.py` (`uv run python thesis/ch7/exp2/make_assets.py`) | `experiments/10seed/linearity_v2/summary.json`; `experiments/10seed/runs.csv`; `experiments/churn/runs_churn100.csv`; `experiments/10seed/runs_heart100.csv`; experiments/heart100_audit/networks.csv (cross-check) | computed |
| options | `recurrence` (`fig:ch7-recurrence`) | `thesis/ch7/exp2/exp2_figures.py` (`uv run python thesis/ch7/exp2/make_assets.py`) | `experiments/adult_signals/matrix.csv`; `experiments/adult_signals/signals.csv`; `experiments/churn_signals/matrix.csv`; `experiments/churn_signals/signals.csv`; `experiments/heart100_audit/linear_core.csv`; `experiments/heart100_audit/family_presence.csv`; thesis/ch7/exp2/signal_ticks.py (HEART_SHORT labels); thesis/ch7/exp2/signal_tables.py (FAM_ROWS labels) | computed |
| options | `seed-agreement-strip` (`fig:ch7-seed-agreement-strip`) | `thesis/ch7/exp2/exp2_figures.py` (`uv run python thesis/ch7/exp2/make_assets.py`) | thesis/ch7/figure_candidates/data/seed_agreement.json (extract_db.py, read-only explaneat_v4) | extracted (DB, read-only explaneat_v4), computed |
| options | `sig-shapes-heart100` (`fig:ch7-sig-shapes-heart100`) | `thesis/ch7/exp2/exp2_figures.py` (`uv run python thesis/ch7/exp2/make_assets.py`) | thesis/ch7/figure_candidates/data/heart_families.json (extract_db.py: experiments/heart100_audit/audit.py Net.family_presence, read-only explaneat_v4); `experiments/heart100_audit/family_presence.csv`; thesis/ch7/exp2/signal_tables.py (FAM_ROWS labels) | extracted (DB, read-only explaneat_v4), computed |
| options | `size-vs-auc` (`fig:ch7-size-vs-auc`) | `thesis/ch7/exp2/exp2_figures.py` (`uv run python thesis/ch7/exp2/make_assets.py`) | `experiments/10seed/runs.csv`; `experiments/churn/runs_churn100.csv`; `experiments/10seed/runs_heart100.csv` | computed |
| options | `synthetic-shape-a-saddle` (`fig:ch7-synthetic-shape-a-saddle`) | `thesis/ch7/exp1/make_figures.py` (`uv run python thesis/ch7/exp1/make_figures.py`) | `thesis/ch7/exp1/data/shapes.json` (`thesis/ch7/exp1/extract_shapes.py`, read-only explaneat_dev, via `experiments/ushape_saddle/shape_plots.compute` / `score_gens.surfaces`; summaries asserted equal to `figures/<label>-shape.json` / `sig_gens.json`); truth constants from `experiments/ushape_saddle/data.py` | extracted (DB, read-only explaneat_dev), computed |
| options | `synthetic-shape-a-sodium` (`fig:ch7-synthetic-shape-a-sodium`) | `thesis/ch7/exp1/make_figures.py` (`uv run python thesis/ch7/exp1/make_figures.py`) | `thesis/ch7/exp1/data/shapes.json` (`thesis/ch7/exp1/extract_shapes.py`, read-only explaneat_dev, via `experiments/ushape_saddle/shape_plots.compute` / `score_gens.surfaces`; summaries asserted equal to `figures/<label>-shape.json` / `sig_gens.json`); truth constants from `experiments/ushape_saddle/data.py` | extracted (DB, read-only explaneat_dev), computed |
| options | `synthetic-shape-b-saddle` (`fig:ch7-synthetic-shape-b-saddle`) | `thesis/ch7/exp1/make_figures.py` (`uv run python thesis/ch7/exp1/make_figures.py`) | `thesis/ch7/exp1/data/shapes.json` (`thesis/ch7/exp1/extract_shapes.py`, read-only explaneat_dev, via `experiments/ushape_saddle/shape_plots.compute` / `score_gens.surfaces`; summaries asserted equal to `figures/<label>-shape.json` / `sig_gens.json`); truth constants from `experiments/ushape_saddle/data.py` | extracted (DB, read-only explaneat_dev), computed |
| options | `synthetic-shape-b-sodium` (`fig:ch7-synthetic-shape-b-sodium`) | `thesis/ch7/exp1/make_figures.py` (`uv run python thesis/ch7/exp1/make_figures.py`) | `thesis/ch7/exp1/data/shapes.json` (`thesis/ch7/exp1/extract_shapes.py`, read-only explaneat_dev, via `experiments/ushape_saddle/shape_plots.compute` / `score_gens.surfaces`; summaries asserted equal to `figures/<label>-shape.json` / `sig_gens.json`); truth constants from `experiments/ushape_saddle/data.py` | extracted (DB, read-only explaneat_dev), computed |
| options | `synthetic-shape-c-s1-g300-saddle` (`fig:ch7-synthetic-shape-c-s1-g300-saddle`) | `thesis/ch7/exp1/make_figures.py` (`uv run python thesis/ch7/exp1/make_figures.py`) | `thesis/ch7/exp1/data/shapes.json` (`thesis/ch7/exp1/extract_shapes.py`, read-only explaneat_dev, via `experiments/ushape_saddle/shape_plots.compute` / `score_gens.surfaces`; summaries asserted equal to `figures/<label>-shape.json` / `sig_gens.json`); truth constants from `experiments/ushape_saddle/data.py` | extracted (DB, read-only explaneat_dev), computed |
| options | `synthetic-shape-c-s1-g300-sodium` (`fig:ch7-synthetic-shape-c-s1-g300-sodium`) | `thesis/ch7/exp1/make_figures.py` (`uv run python thesis/ch7/exp1/make_figures.py`) | `thesis/ch7/exp1/data/shapes.json` (`thesis/ch7/exp1/extract_shapes.py`, read-only explaneat_dev, via `experiments/ushape_saddle/shape_plots.compute` / `score_gens.surfaces`; summaries asserted equal to `figures/<label>-shape.json` / `sig_gens.json`); truth constants from `experiments/ushape_saddle/data.py` | extracted (DB, read-only explaneat_dev), computed |
| options | `synthetic-shape-c-sig-best-g300-saddle` (`fig:ch7-synthetic-shape-c-sig-best-g300-saddle`) | `thesis/ch7/exp1/make_figures.py` (`uv run python thesis/ch7/exp1/make_figures.py`) | `thesis/ch7/exp1/data/shapes.json` (`thesis/ch7/exp1/extract_shapes.py`, read-only explaneat_dev, via `experiments/ushape_saddle/shape_plots.compute` / `score_gens.surfaces`; summaries asserted equal to `figures/<label>-shape.json` / `sig_gens.json`); truth constants from `experiments/ushape_saddle/data.py` | extracted (DB, read-only explaneat_dev), computed |
| options | `synthetic-shape-c-sig-best-g300-sodium` (`fig:ch7-synthetic-shape-c-sig-best-g300-sodium`) | `thesis/ch7/exp1/make_figures.py` (`uv run python thesis/ch7/exp1/make_figures.py`) | `thesis/ch7/exp1/data/shapes.json` (`thesis/ch7/exp1/extract_shapes.py`, read-only explaneat_dev, via `experiments/ushape_saddle/shape_plots.compute` / `score_gens.surfaces`; summaries asserted equal to `figures/<label>-shape.json` / `sig_gens.json`); truth constants from `experiments/ushape_saddle/data.py` | extracted (DB, read-only explaneat_dev), computed |
| appendix | `xor-gates` (`fig:ch7-xor-gates`) | `thesis/ch7/exp1/make_figures.py` (`uv run python thesis/ch7/exp1/make_figures.py xor-gates`) | since 22 Nov: the two evolved networks of `xor-ridge-valley` (`experiments/xor/xor_results.json`, count seed and difference seed picked by `thesis/ch7/exp1/networks.py` `xor_runs`), drawn by `experiments/xor/make_xor_figures.py` `fig_gates(nets)`; weights asserted equal to `verify_xor.evolved_minimal()`, gates asserted (`make_figures.XOR_GATES`: count, skips OR / hidden flips AND; difference, mirrored: skips x2 AND NOT x1 / hidden flips x1 AND NOT x2) by `verify_xor.describe_solution`. Was the hand-built `verify_xor.REPS` (the study's own `experiments/xor/figures/xor_gates.pdf` still draws those, byte-identical) | computed |

## ch7-datasets (`experiments/thesis_datasets/ch7-datasets.tex`; 20 Nov 2026, branch thesis/datasets-table)

One table describing every dataset Chapter 7 reports (`tab:ch7-datasets`). Copied to
`thesis/ch7/exports/ch7-datasets.tex` by `sync_ch7_exports.py` and into the upload as
`sections/tables/ch7-datasets.tex`. Two runs of both scripts gave byte-identical `data.json`,
`datasets.csv` and `.tex`; the preview (`experiments/thesis_datasets/preview.tex`, thesis preamble)
compiles with no errors or overfull boxes, portrait.

| Link | Writer / command | Status |
|---|---|---|
| `ch7-datasets.tex`, `datasets.csv` | `experiments/thesis_datasets/make_table.py` (reads `data.json` only) | computed |
| `experiments/thesis_datasets/data.json` | `uv run --extra survival python experiments/thesis_datasets/extract.py` (~1 min) | extracted (DB, read-only explaneat_v4 + explaneat_dev), computed |
| rows, inputs, training share, encoding, split settings (classification sets) | each set's database split as registered in `experiments/benchmarks/common.CLASSIFICATION` and its raw parent record; asserted to be the split of the 30 Experiment 2 runs (runs CSVs) and of each Experiment 1 network (`experiments/benchmarks/results/propneat.json` genome ids) | extracted (DB, read-only) |
| Heart imputation | detected in the raw record (one non-integer value per column equal to the mean of the rest), with the one-hot level it formed | extracted (DB, read-only), computed |
| WHAS500, GBSG2 | `explaneat.survival.data.load_builtin` (scikit-survival bundled data), `experiments/survival/cox_case/splits/*.splits.json`, the selected run's record in `cox_case/runs/`; asserted to agree | computed |
| XOR | `experiments/xor/run_xor.py` (`X`, `Y`), `xor_results.json` | computed |
| Experiment 3 validation rows ($^a$) | `experiments/churn/val_split.json`, `monk2_val_split.json` | computed (committed selections) |
| short names, outcome labels, loaders, "used in" | typed in `make_table.py`; each outcome label is asserted against the stored class names or target description; "used in" read off the Chapter 7 tables (Exp. 1: networks and benchmark tables; Exp. 2: ten-seed benchmark table; Exp. 3: the grafting studies on Churn and MONK-2) | curated (judgment) |
| caption (title and keys only; the rest is in `CAPTION_MOVED_TEXT.md`, "Datasets table") | typed in `make_table.py` | prose, not results |

Findings (prose disagreements, KRK / MNIST run records) are in `experiments/thesis_datasets/RESULTS.md`.

## ch7-datasets, landscape with MNIST (21 Nov 2026, branch thesis/table-style)

`ch7-datasets.tex` now holds two floats: `tab:ch7-datasets` (the twelve tabular datasets of Experiments
1 to 3, a `sidewaystable`) and `tab:ch7-datasets-mnist` (the three MNIST datasets of Experiment 4). The
upload name is unchanged (`sections/tables/ch7-datasets.tex`). Every preprocessing, scaling and split
phrase is spelled out, one wording per method, each asserted against the stored record it describes;
`experiments/thesis_datasets/RESULTS.md` ("Preprocessing and splits as the code ran them") gives the code
behind each phrase. Two runs of `extract.py` and `make_table.py` gave byte-identical `data.json`,
`datasets.csv` and `.tex`; the experiment-side preview and `thesis/tablestyle/preview.tex` compile with
no errors or overfull boxes.

| Link | Writer / command | Status |
|---|---|---|
| MNIST rows (10k subset, all 50k): rows, class shares, split | `extract.py` `mnist()`: the run records' saved row indices (`experiments/conv_neat/results/{sweep_p50_m4_e5,mnist10_pruned,mnist10_ws50k_halving}/config.json`), asserted equal to `explaneat.conv.data.mnist_splits` recomputed from the cached OpenML `mnist_784` v1 copy; labels from that copy | computed |
| MNIST 3 vs 8 vs noise: rows, class shares, noise mix | `extract.py` `mnist()`: `experiments/conv_neat/three8noise/data.build(seed)` (seed from `results/run_s0/config.json`), asserted equal to the run record's class sizes; test digits asserted to be every standard-test 3 and 8 | computed |
| preprocessing / scaling / split wording | `PHRASE`, `SPLIT` in `make_table.py`, each asserted against `data.json` | curated (judgment) |
| captions (title and keys only; old and new in `CAPTION_MOVED_TEXT.md`, "Datasets tables, landscape") | typed in `make_table.py` | prose, not results |

## Thesis table style (21 Nov 2026, branch thesis/table-style)

One table style for the whole thesis: `thesis/tablestyle/thesis-tablestyle.tex`, shipped in the upload at
the thesis root as `thesis-tablestyle.tex` and loaded once in the preamble (`\input{thesis-tablestyle}`
after `\usepackage{makecell}`). Every generated Chapter 7 table calls `\thesistablestyle` as its first
line (`exp1/make_tables.py`, `exp2/make_assets.py` `write_table`, `exp2/per_run.py`, `exp2/signal_ticks.py`
with `[1]`, `exp2/signal_tables.py`, `exp2/linear_betas.py`, `exp2/completeness_table.py`,
`appendix/heart_splits_table.py`, `experiments/thesis_datasets/make_table.py`); by default the style also
applies itself to every `table` / `sidewaystable` float, so the hand-written tables of other chapters get
it with no edit. Layer 3 rerun twice: byte-identical; the only change to generated tables is the style
line (plus the removal below). Previews: `thesis/tablestyle/preview.pdf` (every table in the upload) and
`exp1/preview.pdf`, `exp2/preview*.pdf`, `figures/preview.pdf` recompiled with the style (the figure
preview's preamble is copied from `exp1/preview.tex` by `ch7_figures_index.py`).

**Removed: function preservation.** The *Function pres. error* column of `tab:exp1-networks`
(`exp1/make_tables.py`) is gone (Mike, 21 Nov: the operations are identity operations by proof; the
residual is rounding). `exp1/extract_db.py` still computes it into `networks.json` as an internal check.
The same column in `exp2/per_run.py` and two rows of `exp2/worked.py` are still printed (outside the
upload): that edit was refused by the session's permission check and is left for Mike to confirm
(`CAPTION_MOVED_TEXT.md`, "Removed: function preservation").

## ch7-run-config (`experiments/thesis_run_config/ch7-run-config.tex`; 21 Nov 2026, branch thesis/run-config)

The full configuration of every run family Chapter 7 reports (`tab:ch7-run-config`, cited from the common
Methods). Copied to `thesis/ch7/exports/ch7-run-config.tex` by `sync_ch7_exports.py` and into the upload as
`sections/tables/ch7-run-config.tex`. Two runs of both scripts gave byte-identical `data.json`,
`run_config.csv` and `.tex`; the preview (`experiments/thesis_run_config/preview.tex`, thesis preamble)
compiles with no errors or overfull boxes (landscape, `sidewaystable`).

| Link | Writer / command | Status |
|---|---|---|
| `ch7-run-config.tex`, `run_config.csv` | `experiments/thesis_run_config/make_table.py` (reads `data.json` only) | computed |
| `experiments/thesis_run_config/data.json` | `uv run python experiments/thesis_run_config/extract.py` (~10 s) | extracted (DB, read-only explaneat_dev + explaneat_v4), computed |
| NEAT options, generations, epochs, fitness, activations, mask (database runs: original networks, synthetic, Experiment 2, MONK-2 ten seeds) | `experiments.config_json`, `hardware_info.run_params`, stored generations, the launch template (Heart original) or launch-script constants (`experiments/aaai27/ingest_monk2.py`, MONK-2 original); asserted equal across a family's seeds, to the config file each run read, and (Experiment 2) between explaneat_dev and explaneat_v4 | extracted (DB, read-only) |
| Cox, Connect-4, XOR | run records (`cox_case/runs/*.json` with their NEAT config text; `c4_capacity/logs`, `c4_line_features/logs` + configs, checked against the config stored on import; `experiments/xor/run_xor.py`) | computed |
| optimiser, learning rate, phantom-mask default | `explaneat/core/backproppop.py` at each run's recorded commit (`git show`); training-path `git diff` between a family's commits (empty for all) | computed |
| Experiment 3 fine-tuning | constants of `experiments/churn/ext_common.py`, `genome_surgery.py`, `reinit.py` (read with ast), checked against the in-scope result CSVs | computed |
| ExConNEAT | `experiments/conv_neat/results/{sweep_p50_m4_e5,mnist10_pruned,mnist10_ws50k_halving}/` and `three8noise/results/run_s0/` `config.json`, `results.json` | computed |
| row names, column labels, short wording of fitness / stop / selection rules, option labels | typed in `make_table.py`; each short rule asserted against the stored record; every option that differs between families must have a column or label (asserted); the "All NEAT studies" line lists only options equal in every family | curated (judgment) |
| caption (title and keys only) | typed in `make_table.py` | prose, not results |

Findings (three NEAT config families, unmasked runs, Heart's 40 of 100 generations, dirty trees, ExConNEAT
records) are in `experiments/thesis_run_config/RESULTS.md`.

## ch7-run-config and ch7-datasets: presented runs only (21 Nov 2026, branch thesis/run-config-trim)

`tab:ch7-run-config` now shows only the runs that produced the networks Chapter 7 presents, and
`tab:ch7-datasets-mnist` only the data of the presented ExConNEAT network's lineage. Both extractors and
both generators were rerun twice: byte-identical `data.json`, `run_config.csv` / `datasets.csv` and `.tex`;
both experiment-side previews and `thesis/tablestyle/preview.tex` (which now also inputs
`ch7-run-config`, in the thesis table style) compile with no errors or overfull boxes. Supersedes, in the two sections
above, the Cox (GBSG2), synthetic 300-generation / sigmoid-unit and MNIST 3 vs 8 vs noise links.

| Link | Writer / command | Status |
|---|---|---|
| Seed(s) presented | `make_table.py` `presented_runs`: the run each family's record names as selected (`selected` in `data.json`: Cox `selected.json`, Connect-4 the largest training AUC, synthetic the largest training AUC asserted equal to the explained genome in `propneat.json`), or every seed for the families listed in `ALL_SEEDS` | computed (selection), curated (which families present every seed) |
| Cox row (WHAS500 only) | `extract.py` `COX_PRESENTED`: the five WHAS500 run records; GBSG2 left out because `tab:exp1-networks` has WHAS500 only and the survival benchmark table gives GBSG2 as a side note | curated (judgment) |
| synthetic 300-generation and sigmoid-unit rows | removed: their networks are only in the options figures (`FIGURES.md`) | curated (judgment) |
| Heart original network, generations | the generations stored (40); the planned budget stays a provenance note (`experiments/thesis_run_config/RESULTS.md`, finding 3) | extracted (DB, read-only) |
| ExConNEAT panel (b) | `experiments/conv_neat/results/{sweep_p50_m4_e5,mnist10_pruned,mnist10_ws50k_halving}/` `config.json`, `results.json` (the three phases of the presented MNIST 10-class network); MNIST 3 vs 8 vs noise removed | computed |
| MNIST datasets rows | `experiments/thesis_datasets/extract.py` `mnist()`: the two 10-class training sets (10k subset: the 10k run and its pruning phase; all 50k: the warm start); MNIST 3 vs 8 vs noise no longer rebuilt | computed |
| captions (title and keys only; old and new in `CAPTION_MOVED_TEXT.md`, "Run-configuration table, presented runs only") | typed in `make_table.py` | prose, not results |

What was dropped and why: `experiments/thesis_run_config/RESULTS.md` ("Trimmed to the presented runs") and
`experiments/thesis_datasets/RESULTS.md` ("MNIST (Experiment 4) and KRK").


## Experiment 1: the original Adult network, the Churn case and GBSG2 back (21 Nov 2026, branch thesis/adult-exp1)

Mike's decisions of 21 Nov: Experiment 1 presents the original Adult network again beside the original Heart and
MONK-2 networks (Adult also stays the ten-seed study of Experiment 2), the Churn case (`churn-case-s10`) and GBSG2
(as a side note beside WHAS500). Every Experiment 1 asset was extended; every extractor and generator touched was run
twice with byte-identical outputs, and the previews compile with no errors or overfull boxes.

| Asset | Change | Status |
|---|---|---|
| `tab:exp1-networks` rows "Adult (original network)", "Churn case (s10)", "GBSG2 (Cox, side note)" | `thesis/ch7/exp1/extract_db.py` (`NETS`), the same computed checks as every row: structure from the genome, annotation / split / identity counts from the operations, coverage from `get_coverage`, the strict hierarchy check and the function-preservation error from `check_explanation`, explaining minutes from the database writes (Adult: no evidence log, "--"; Churn case and GBSG2 from their v4 records' databases, asserted against the records' counts) | extracted (DB, read-only), computed |
| Adult coverage | 1.00 / 1.00 under the current definition: its `unused_inputs` annotation (53 never-connected inputs) is exempt from the single-root rule (D4), as MONK-2's; 0.9 compositional under the earlier definition (`paper_coverage_options.json` "current", `rec_cov_c`); `check_explanation`'s strict check reports two roots (`hierarchy_complete_check` false) | computed |
| Adult / Churn case AUCs | computed in `extract_db.py` (`perf`) until `experiments/benchmarks/results/propneat.json` holds them (its Adult entry is being added on branch thesis/wide-grids); then read from there, asserted equal | extracted (DB, read-only) |
| `networks.json` `parts` | each stored annotation's name, entry / exit counts, the model inputs among its entries (by column) and children; read by the book's original-networks card | extracted (DB, read-only) |
| `tab:ch7-datasets` "used in" | no longer typed for the classification datasets: read from `data.json` `run_splits` (Experiment 1: `propneat.json` genomes, the three original networks of `runs_paper_reference.csv`, the Churn case's run; Experiment 2: the ten-seed runs) and `exp3_validation`. Adult 2 → 1, 2; Churn 2, 3 → 1, 2, 3; every other classification row unchanged. Survival, XOR and MNIST stay curated (GBSG2 "1") | computed (was curated) |
| `tab:ch7-run-config` | new rows "Adult, original network" (template `high-exploration` + six overrides, asserted to be exactly the options that differ; `adult.ini`; 100 generations; seed n.r.) and "Churn case" (`churn.ini`, seed 10, 200 generations, mask off as recorded); the Cox row is "WHAS500, GBSG2" again (ten run records, one configuration asserted; seeds 0, 1) | extracted (DB, read-only) / run records |
| Book | Experiment 1: the full original-networks card (`data/e1_1.py`, Heart, Adult, MONK-2) with an Adult part-by-part table read in code from `networks.json`, `runs_paper_reference.csv`, `paper_recompute/core.json` and `shap_summary.json`; the Churn case card (`e1_2.py`) and the full Cox cards (`e1_3.py`, `e1_3a.py`, GBSG2 as a side note); `e1_1hm.py` and `e1_3w.py` removed (the full cards replace them); none of these is in the Miscellany any more. Argument, methods, prose methods (`xc_methods_prose.py`), results and discussion updated | prose; numbers read from files or named |

**Inconsistencies found (stored records, not edited; read only).** The stored Adult explanation (`7e3761fe`) predates
the output-activation fix: its root hypothesis says "output = ReLU(...)" (the output is a sigmoid); its capital-gain
hypothesis says the inner knots are "masked by the output unit's clipping" (the output does not clip; the hub, a
hidden ReLU, does; `paper_corrections.md` item 2) and calls the pathway a "SINGLE-KNOT threshold" with "~90%" at zero
(the recompute: two close knots at $3,225 and $4,300, 92%). Its linear-backbone hypothesis ("most of the model's
predictive mass (top SHAP)") still holds after the SHAP fix (87.5% of mean |SHAP|). Four
category-gate part names give the wrong native-country level for the input they read (`bump_native_country_11_33`
reads native-country:1 and :31; `_20` :19; `_34` :32; `_26` :25), read from the column order in `networks.json`
`parts`; the capital-gain core's five country inputs are named correctly. The book prints these as a caveat.


## MONK-2 rule comparison (22 Nov 2026, branch thesis/monk2-rules)

`tab:exp1-monk2-rules` (`thesis/ch7/exp1/tables/exp1-monk2-rules.tex`; Overleaf `sections/tables/ch7-ex1-monk2-rules.tex`)
and the options figures `ebm-monk2-count`, `ebm-monk2-pairs`. Study: `experiments/monk2_rule_comparison/` (README:
protocol and commands). Every script was run twice with byte-identical outputs.

| Link | Writer / command | Status |
|---|---|---|
| `exp1-monk2-rules.tex` | `thesis/ch7/exp1/monk2_rule_table.py` (formats only) | computed |
| `experiments/monk2_rule_comparison/results/monk2_rules.{json,csv}` | `fit_models.py` (InterpretML venv, ~45 s): EBM one-hot (settings read from `experiments/benchmarks/results/classification/monk2.json`, the current file; test AUC asserted equal to the suite's), EBM raw (nominal, all pairs), trees (unlimited depth), tree / EBM on the count; seeds 42 | computed |
| `experiments/monk2_rule_comparison/data/monk2_split.json` | `extract_data.py`: the suite's MONK-2 split in explaneat_v4, READ ONLY (`compute_runs.ReadOnlySession`); the scaled arrays' hash asserted equal to `experiments/benchmarks/results/data_manifest.json` | extracted (DB, read-only) |
| PropNEAT row: test AUC | `experiments/benchmarks/results/propneat.json` (`paper-monk2`) | extracted (DB, read-only) |
| PropNEAT row: parts, "yes" | `experiments/10seed/plain_language/monk2-paper.json` (the explanation's committed plain-language layer): part count by kind; the root's children with nonzero contribution are one linear and one gated unit, each reading all six attributes (the count ramp and the at-least-three gate); counts asserted equal to `thesis/ch7/exp1/data/networks.json` | agent-authored explanation (July 2026), structure read by code |
| "Rule stated?" | operational checks in `fit_models.py` (docstring): count models' leaves / shape equal the rule; EBMs without the count: log-odds by count on training rows peak at 2 and separate the classes; trees without the count: "no" | computed |
| `ebm-monk2-count`, `ebm-monk2-pairs` | `thesis/ch7/exp1/make_figures.py` (`monk2_figures`) from `monk2_rules.json` (`ebm_raw` per-row log-odds; pair terms, value 1 first on both axes); jitter `RandomState(0)` | computed |

The EBM one-hot row follows the suite's file: with the widened-grid rerun (branch thesis/wide-grids) its CV
choice changes, so after that merge rerun `fit_models.py`, `monk2_rule_table.py` and the figures.

## Experiment 1, the Churn case figures and text check (22 Nov 2026, branch thesis/churn-case-figures)

Two used figures and a check of the Chapter 7 Churn paragraph. Every number is a forward pass of the explained
model (`experiments/churn_case/probes.py` `ChurnModel`: genome `9cacd431`'s phenotype with the stored explanation
`ce11259b`'s 95 operations replayed by `ModelStateEngine`; `StructureNetwork` with the trained output and hidden
activations, sigmoid; each pass asserted equal to `PlainModel.logit`; the explanation id asserted equal to the
committed layer's) on constructed customers in real units. Each script was run twice: byte-identical outputs,
figures and plot data.

| Output | Writer / command | Reads | Status |
|---|---|---|---|
| `experiments/churn_case/text_check.json` | `experiments/churn_case/text_check.py` | explaneat_v4, READ ONLY (genome, split data and scaler, explanation operations); `plain_language/churn-case-s10.json` (`reference_patient`, parts, titles) | extracted (DB, read-only), computed |
| `thesis/ch7/exp1/data/churn_case_probes.json` | `thesis/ch7/exp1/extract_churn_case.py` (engine and helpers of `experiments/churn_case/probes.py`, `text_check.py`) | as above | extracted (DB, read-only), computed |
| used `churn-case-probes` (`fig:ch7-churn-case-probes`) | `thesis/ch7/exp1/churn_case_figures.py` (`uv run python thesis/ch7/exp1/make_figures.py churn-case-probes`) | `churn_case_probes.json`; part titles from the layer | computed |
| used `churn-case-probes` panel (c), added 22 Nov 2026 (branch thesis/churn-day-panel; the service-call panels re-lettered (d), (e)) | `extract_churn_case.py` `day_terms` (the direct day-charge edge and the parts `dayMinutesRamp`, `heavyDayUse`, `dayChargeBoost`, `serviceCallChain` along panel (b)'s sweep; asserted to rebuild the linear base's contribution; the ramp's start from `text_check.first_positive`) -> `churn_case_probes.json` `day.terms`; drawn by `churn_case_figures.py` | explaneat_v4, READ ONLY | extracted (DB, read-only), computed; rerun byte-identical |
| `text_check.json` S9 (the ramp sentence: day charge held at the typical value, risk at 130 and 210 day minutes, the ramp's contribution change), added 22 Nov 2026 | `experiments/churn_case/text_check.py` | as above | extracted (DB, read-only), computed; S1-S8 unchanged |
| used `churn-case-net-concept` (`fig:ch7-churn-case-net-concept`) | `thesis/ch7/exp1/churn_case_figures.py` (`uv run python thesis/ch7/exp1/make_figures.py churn-case-net-concept`), drawn with `thesis/ch7/exp2/worked.py`'s collapse and `render_structure.overview` | `experiments/churn_case/plain_language/churn-case-s10.json` (asserted the genome and explanation of `churn_case_probes.json`, same top-level parts) | agent-authored explanation (parts and titles), structure drawn by code |
