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

---

## ch7-ex1-networks (`thesis/ch7/exp1/tables/exp1-networks.tex`)

| Link | Writer / command | Status |
|---|---|---|
| `exp1-networks.tex` | `thesis/ch7/exp1/make_tables.py` | computed (byte-identical) |
| `thesis/ch7/exp1/data/networks.json` | `thesis/ch7/exp1/extract_db.py` (explaneat_dev, ~1 min) | extracted (DB, read-only); rerun byte-identical |
| structure: inputs, hidden, conns, depth, skips, density, skippiness | genome in explaneat_dev, `compute_runs.structure` and `ExplaNEAT` measures | extracted (DB, read-only) |
| splits, identity nodes, leaf + comp. counts | explanation operations in explaneat_dev (`explaining_time.annotation_counts`; also equal to `get_coverage`'s counts), asserted equal to the records (paper: to `paper_coverage_options.json`) | extracted (DB, read-only); all agree |
| coverage struct./comp., hierarchy completeness | MCP `get_coverage` (`coverage_report`) on each explanation, in `extract_db.py`, as `experiments/10seed/verify_record.py` does; check_explanation's stricter hierarchy check kept as `hierarchy_complete_check` | extracted (DB, read-only), computed. Agree with the six v4 records and paper Heart. **MONK-2: 1.00** (was printed 0.75): under the current definition (decision D4, 7 Nov, commit 1fb3a8b6) its pruned-input annotation is exempt from the single-root rule; 0.75 is the pre-D4 value in `paper_coverage_options.json` ("current"), now kept as `rec_cov_c` and quoted in the caption. check_explanation's strict check still reports MONK-2's hierarchy incomplete (two roots) |
| agent min | `experiments/10seed/explaining_time.explaining_window` in `extract_db.py`: explanation created → last timestamped write (operations, evidence records, `evidence_log`) | extracted (DB, read-only), computed. 0.6–4.5 min below the record's own start/end (kept as `rec_agent` in `networks.json`); paper Heart / MONK-2 have no evidence log: "--" |
| function-preservation error | `check_explanation.check` on the training split, in `extract_db.py`; equals the records and `experiments/explanation_levels/levels/*.json` exactly for all eight | extracted (DB, read-only), computed |
| XOR row | `experiments/xor/xor_results.json` ← `experiments/xor/run_xor.py` | computed |
| train / test AUC and CI | `experiments/benchmarks/results/propneat.json` ← `propneat.py` (asserts equality with the run records to 1e-9) | extracted (DB, read-only); rerun byte-identical |
| WHAS500 Harrell's C | `experiments/benchmarks/results/survival.json` ← `survival.py` | computed |
| genomes | the evolution runs of each study (layer 1) | computed (not rerun) |
| caption prose ("July 2026", "short sandbox pause", selection sentences) | typed in `make_tables.py` / `extract_db.py` | prose, not results |

## ch7-ex1-benchmarks (`thesis/ch7/exp1/tables/exp1-benchmarks.tex`) and ch7-ex2-benchmarks (`thesis/ch7/exp2/tables/exp2/exp2-benchmarks.tex`)

| Link | Writer / command | Status |
|---|---|---|
| `exp1-benchmarks.tex` | `thesis/ch7/exp1/make_tables.py` via `benchmark_tables.py` | computed (byte-identical) |
| `exp2-benchmarks.tex` | `thesis/ch7/exp2/make_assets.py` via `exp1/benchmark_tables.py` | computed (byte-identical) |
| `experiments/benchmarks/benchmarks.csv` | `make_csv.py` (reads `results/classification/*.json`, `propneat.json`, `survival.json`) | computed; rerun byte-identical |
| `results/classification/*.json` | `classification.py` (~2.5 h; needs `interpret`) | computed. Seeds: 5-fold stratified CV seed 42, bootstrap `RandomState(42)`. `lr_l1` was **unseeded** until this audit (see issue 4) |
| `cache/*.npz`, `results/data_manifest.json` | `export_data.py` | extracted (DB, read-only); rerun byte-identical |
| `results/propneat.json` | `propneat.py` from the runs CSVs and the DB | extracted (DB, read-only) |
| `results/survival.json` | `survival.py` (~80 s) | computed (timings aside) |
| synthetic true-logit AUC (caption) | `experiments/ushape_saddle/references.json` ← `references.py` | computed; rerun byte-identical |

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
| record `mike_verification` (verified, minutes) | review UI (`explaneat/api/routes/records.py`), committed ab99b365 | human-entered by Mike. **Exception:** adult-s0 (6 min) and adult-s4 (5 min) were written by an agent "at Mike's instruction in chat on 2026-11-18" (issue 2); adult-s3 has no minutes (footnoted) |
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
- MONK-2's compositional coverage now prints 1.00 under the D4 exemption (caption gives the 0.75 without it); check the Chapter 7 prose that cites 0.75.
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
