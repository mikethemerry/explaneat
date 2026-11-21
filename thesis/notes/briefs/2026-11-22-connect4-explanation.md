# Brief: Chapter 7, Experiment 1, "Explanation E: Connect-4" (22 Nov 2026)

You are joining Mike (PhD candidate; owner of this work and research) to write up the Connect-4
case of Chapter 7, Experiment 1. You are the codebase's steward; Mike owns the argument and writes
the prose. Read `CLAUDE.md` first (especially the **Results provenance rule**), then this brief.

## House rules (all settled; follow them without re-litigating)

- **Provenance**: every number in a thesis table, figure or caption comes from committed code Mike
  can rerun (analysis script → results file → generator). Never type a result. Seed everything; run
  new analyses twice and check byte-identical output. Explanation records and Mike's verifications
  are read from the database (`explaneat/analysis/explanation_records.py`, DB first, file fallback).
- **Databases**: everything is in Postgres `explaneat_v4` (Experiment 1 networks were copied in on
  21 Nov). Read-only access only: `PGOPTIONS='-c default_transaction_read_only=on'`. Never run
  `explaneat setup` or migrations against `explaneat_v4` or `explaneat_dev`; give Mike the commands.
- **Git**: another Claude session ("Random Forest ExplaNEAT experiment") works in
  `.claude/worktrees/boosting`. Do NOT switch branches in the main checkout
  (`/Users/mike/dev/explaneat`); use a `git worktree` for any branch work. Commit messages end with
  the Co-Authored-By line your session gives you.
- **Thesis text**: Mike writes the prose (Overleaf; local copy `/Users/mike/dev/ExplaNEAT-Thesis`,
  synced via GitHub). Draft wording in chat for him to adapt; don't write prose into the thesis repo.
  You may do formatting/asset syncing there if he asks (no content changes). Never print "Claude" or
  Mike's name in thesis text, tables or captions. Never say "the paper" (use "the original networks",
  "the fixed split"). No function-preservation numbers anywhere (identity operations hold by proof).
- **Captions**: a short title plus reading keys only (what colours, symbols, bold mean). Method and
  results sentences go to `thesis/ch7/CAPTION_MOVED_TEXT.md` for Mike's body text.
- **Figures/tables**: generated via `thesis/ch7/figlib.py` (figures flat at
  `figures/ch7/<id>.pdf`; floats in `thesis/ch7/figures/sections/figures/{used,options,appendix}/`);
  tables start with `\thesistablestyle` (thesis-wide style). Overleaf bundle:
  `bash thesis/scripts/build_overleaf_upload.sh`; full rebuild: `bash thesis/scripts/reproduce_ch7.sh`.
- **Wording**: "realisation" of a signal; captions/keys plain; NZ spelling; `\cref`.
- **Keep it light**: Mike wants the simplest thing that's correct; don't over-elaborate.

## Where Chapter 7 is (one paragraph)

Experiment 1 (existence) now covers: Adult, Heart, MONK-2 (the original networks, recomputed),
the Churn case, Connect-4 (cells and line features), XOR, and synthetic U / saddle / U+saddle.
Survival (WHAS500, GBSG2) moves to Chapter 8. Experiment 2 (Adult, Churn, Heart-100, ten seeds,
all verified by Mike) is done. Comparators were retuned on widened grids (final 22 Nov); a PropNEAT
sensitivity sidebar exists (tuning changes no conclusion). Mike has just written Explanation D
(Churn) with supporting figures `fig:ch7-churn-case-net-concept` and `fig:ch7-churn-case-probes`;
that is the pattern to follow here: his paragraph, a concept-level network figure, probe plots that
back each sentence, and a text-check script (`experiments/churn_case/text_check.py`) that computes
every number in his paragraph from the network.

## The Connect-4 case: what exists

**Data** (`tab:ch7-datasets`): 67,557 eight-ply positions (PMLB/UCI Connect-4), x to move; target
x wins vs loss-or-draw (65.8%); stratified 80:20 split, seed 42 → 54,045 / 13,512.
- *Cells*: 42 inputs, x = +1, o = −1, blank = 0, z-normalised. Pieces concentrate low on the board
  (share of positions with a piece, bottom→top row: 0.58, 0.31, 0.15, 0.07, 0.02, 0.00). Columns are
  asymmetric because the source keeps one of each mirror pair (left columns hold more pieces).
- *Line features*: 16 counts computed from the cells (open lines with 1/2/3 pieces per side, threats
  per side and their row parity, centre-column pieces, playable cells by row parity);
  `experiments/c4_line_features/lfdata.py`.

**Configuration** (`tab:ch7-run-config`): population 150, 80 generations, 50 backprop epochs per
generation, 10 hidden units at start, weight init SD 0.15, phantom mask on. Chosen by 17 pilots on
held-out training rows: the default setup (5 epochs, no hidden start) collapsed to ~1 hidden unit at
LR-level validation AUC 0.862; 10 hidden + SD 0.15 + 50 epochs reached 0.919
(`experiments/c4_capacity/RESULTS.md`).

**Networks** (explorer on :8000, read-only data in explaneat_v4; focus mode = magnifying glass on an
annotation):
- Cells, capacity champion seed 1 (10 hidden, 472 connections, 12 annotations):
  http://127.0.0.1:8000/genome/f0eb7635-7f85-437b-815b-993c5ad6cc14?experiment=154143d6-dae7-454f-9d6d-f9649ebfad19
- Lines (`c4lf-lines-heavy-s0`, 9 annotations):
  http://127.0.0.1:8000/genome/6ebf7e0c-55ec-4215-9852-0ac982e43c0e?experiment=b4b4e7b6-1bd1-4a5d-a496-7a6bf23fe884
- Lines + cells (14 annotations): …/genome/9e843d20-8e42-47d6-9c94-576c5f06fdf3?experiment=19fd1a9f-69ab-4571-ba83-a3d8c7c86122
- Per-line counts (7 annotations): …/genome/94fb8bdc-784b-4f75-a63c-25b1ad31da15?experiment=fd7c8e00-a9e0-4d3c-858b-4247ee6e4d99

**Performance** (`tab:exp1-benchmarks`, `experiments/benchmarks/benchmarks.csv`, widened grids,
test AUC): cells — PropNEAT 0.915; LR 0.861, tree 0.889, RF 0.955, EBM 0.941, MLP 0.963, GB 0.977.
Lines — PropNEAT 0.879; LR 0.862, tree 0.871, EBM 0.883, RF 0.887, MLP 0.889, GB 0.889.

**Explanations** (records: `experiments/c4_capacity/explanations/record.json`,
`experiments/c4_line_features/explanations/{lines,lines_cells,perline}/`; RESULTS in each folder):
- *Cells*: full coverage, 11 leaves all "described" (0 meaningful); gates on individual squares.
  Replacing switching gates by affine fits: training AUC 0.919 → 0.855 (real non-linearity). A linear
  fit on cells explains 74.7% of logit variance; aggregate line/threat/parity/centre features explain
  6.2% of the residual; largest unit–concept correlation ≈ 0.33 (a unit vs centre-x count).
  Exploratory weight-map reading (not yet a committed script — make one if used): every hidden unit
  and the direct edges weight the bottom rows most (e.g. direct edges, mean |w| by row bottom→top:
  0.78, 0.77, 0.46, 0.20, 0.22, 0.19), mirroring where pieces are; units contrast cells along the
  bottom row (e.g. "x on the bottom-row edges vs o in the middle"); units 4/6/8 lean to centre
  columns.
- *Lines*: "largely a weighted sum of line features with a few gates" (affine fit 0.860 vs 0.882
  training AUC). x's open twos/singles raise x's chance, o's lower it; gates: an x-open-twos cap,
  a single o threat switching the o-lines ramp and cap off (74% → 37% at the typical case), a centre
  column block, a quiet-o gate. 3–6 of 8 leaves meaningful (author, blind verifier, concept probe).
- *Lines + cells*: test AUC 0.929, but the gates read raw cells again (0 of 11 meaningful).
- *Per-line counts*: nearly linear (linearised 0.9119 vs 0.9120); three units read as single lines.
- Mike has NOT verified any of the Connect-4 explanations yet (review screen: `/review`; Experiment 1
  records now resolve there after a restart of :8000).

**Assets already built**:
- Tables: Connect-4 rows in `tab:exp1-networks` and `tab:exp1-benchmarks`
  (`\input{sections/tables/ch7-ex1-networks}`, `\input{sections/tables/ch7-ex1-benchmarks}`).
- Appendix figures (Experiment 1 network appendix, `sections/appendix-exp1-networks.tex`):
  `fig:exp1-net-c4cap-main-h10-e50-s1-explained` (collapsed view; raw view unreadable, 472 edges),
  `fig:exp1-net-c4lf-lines-heavy-s0-raw`, `fig:exp1-net-c4lf-lines-heavy-s0-explained` (collapsed).
- No main-text Connect-4 figure exists. A Connect-4 "boundary" table was considered and dropped by
  Mike. The experiment book (`thesis/ch7/book`, published artifact) has a Connect-4 card with the
  input-bases table.

**Mike's prose**: `sections/7-ExplaNEAT.tex`, `\subsubsection{Explanation E: Connect-4}` (empty;
~line 496). The Experiment 1 dataset list (~line 348) names Connect-4 as "Relational data".

## The argument so far (for strategising with Mike)

Connect-4 is Experiment 1's case of *explained but not understood*: on raw cells the network is
strong and fully, faithfully decomposed, but its parts are square-level gates, not threats, lines or
parity; on line features (same rows, split, target) the parts become named tactical concepts, at an
accuracy cost because 16 counts hold less of the signal; given lines and cells together it reverts
to cell gates. Point: what is *understood* depends on the input vocabulary and on whether the model
builds concept-like intermediates — contrast with MONK-2, where the network built the count itself
(`tab:exp1-monk2-rules`; a literature review on concept recovery is in Mike's hands: TREPAN/M-of-N
prior art, polysemanticity/superposition as analogues of the Connect-4 result).

## Your job with Mike

1. Walk the networks with him in the explorer (focus mode) and the records; answer his questions
   from the records and the network (compute, don't recall).
2. Strategise the explanation and results piece: what the subsection claims, which network(s) carry
   it (cells and lines at minimum), which existing tables/figures to cite, and whether a main-text
   figure is needed (candidates: a cells weight-map board figure — bottom-row concentration with the
   occupancy profile; a lines probe figure like the Churn one; a two-network concept view).
3. Build what he chooses under the provenance rule (generator + committed script; figlib; captions
   title+keys), give him the exact `\input` lines and `\cref` labels, and a text-check script for
   his paragraph's numbers, as for the Churn case.
