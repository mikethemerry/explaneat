# Connect-4 line-features base test

**Exploratory, not pre-registered.** Branch `exp/c4-line-features`, run on 2026-11-12. Thesis
thread "Experiment 3: where does complexity live?".

**Question.** Every Connect-4 network explained so far on the 42 raw cells (PropNEAT at 0.86 and
0.91 test AUC, dense MLPs up to 0.94) was faithful and decomposed but not understood: the leaves
were single-cell positional gates and no leaf matched a threat, line or parity concept
(`../explanation_levels/RESULTS.md`, `../explanation_barrier/RESULTS.md`, `../c4_capacity/RESULTS.md`
on `exp/c4-capacity`). Hypothesis: the complexity is in the realisation and the vocabulary, not in
the signal. Test: same rows, split and target, different **base**: give PropNEAT the line features
as inputs.

**Answer in brief.** On the line-features-only base the leaves became named tactical concepts:
the selected network's eight leaves are gates and ramps on open twos, open singles, threats and
the centre column, and the author, the blind verifier and the concept probe agree that 3-6 of the
8 are meaningful and none is closed-form-only (cells-only: 0 of 35 ladder leaves meaningful, 69%
closed-form-only; 1 of 11 on the evolved champion). But the AUC is not comparable: the 16
aggregate line features cap at 0.885 (GBM) and the network reaches 0.879. On the line features +
cells base the network reaches 0.929 test AUC, above every earlier Connect-4 PropNEAT network,
yet its gates read the raw cells again: the author grades 0 of 11 leaves meaningful, 7 partial,
4 closed-form-only, and names them positional gates. So the change of base moved the leaves to
tactical concepts only where the base forced it, at the cost of AUC; where both vocabularies
were available at comparable AUC, evolution and backprop went back to the cells. The signal
check explains why: at eight ply the aggregate tactical vocabulary (threats, open lines, centre,
parity) carries much less of the signal than the cells, while a *per-line* vocabulary (138 named
open-line counts) carries more than the cells (GBM 0.953, LR 0.917), so a nameable base that
keeps the signal exists but it is positional at the line level, not tactical.

## Data and bases (`lfdata.py`)

Rows, target and split are those of `../complex_real` (PMLB `connect_4`: 67,557 eight-ply
positions, x to move, no forced move; y = 1 iff x wins with perfect play; stratified 80/20 split,
seed 42: 54,045 / 13,512 rows). Inputs are z-scored with a scaler fit on the training rows; the
database keeps raw values and the scaler on the split. Three bases:

| Base | Inputs | What they are |
|---|---|---|
| `lines` | 16 | open lines of 4 with exactly 1, 2 or 3 pieces of one side and none of the other (`x_open_1..3`, `o_open_1..3`); threats per side (`x_thr`, `o_thr`: empty cells completing an open three) and their row parity (`*_thr_odd`, `*_thr_even`); centre column pieces (`centre_x`, `centre_o`); columns whose next playable cell is on an odd / even row (`n_odd_empty_playable`, `n_even_empty_playable`) |
| `lines_cells` | 58 | the 16 line features followed by the 42 cells (`c_a1..c_g6`; +1 x, -1 o, 0 blank) |
| `perline` | 154 | per line of 4 (69) and side, the side's piece count when no opposing piece is on the line, else 0 (`x_line_a1-d1`, ...); then the 16 line features. Added after the signal check, default config only |

The features are those of `c4_features.py` (`exp/c4-capacity`) and the Connect-4 concept library
(`explaneat/analysis/concepts.py`), computed through the library so inputs and probe concepts are
the same functions. The playable-now threats (`x_thr_now`, `o_thr_now`) are constant zero on this
dataset (UCI excludes positions with a forced move) and were left out. Threats are rare at eight
ply: 660 training positions have an x threat, 765 an o threat, 66 and 100 an odd-row threat. Each
dataset carries a codebook (`additional_metadata.codebook`) naming every input, and the concept
library offers every line input as a domain concept (plus x-minus-o differences and, when the
cells are present, the full cell-derived library), so leaves can be named and probed in the same
vocabulary (`tests/test_analysis/test_concepts_c4_lines.py`).

Datasets in `explaneat_dev` (new rows only; no schema change; `explaneat_v4` not touched):
`connect4_lines` `a520a67b-855a-4606-8038-88cb5bc1fb2c` (split `1bdaf11a-…`),
`connect4_lines_cells` `347b3198-2332-4a01-97e2-cb207dfff87e` (split `b9e6a9cf-…`).

## Signal check: where does the signal live? (`signal_check.py`)

GBM (sklearn HistGradientBoosting, defaults) and LR on nameable vocabularies, same split, test AUC:

| Vocabulary | Inputs | LR | GBM |
|---|---|---|---|
| aggregate line features (the `lines` base) | 16 | 0.862 | 0.885 |
| open lines by direction | 24 | 0.880 | 0.915 |
| open lines by lowest row | 36 | 0.885 | 0.918 |
| the three above | 76 | 0.900 | 0.936 |
| per-line open counts | 138 | 0.917 | 0.953 |
| per-line + aggregate (the `perline` base) | 154 | 0.919 | 0.952 |
| the 42 cells | 42 | 0.861 | 0.944 |

The aggregate tactical vocabulary loses 0.06 AUC against the cells; the per-line vocabulary
gains 0.01 and makes the problem far more linear (LR 0.917 against 0.861). This is the central
caveat of the test as posed: "same signal" does not hold for base (a).

## Runs (`run_seed.py`, `run_queue.sh`, `runs.csv`, `tables.md`)

Training path as `../complex_real/run_seed.py` and `../c4_capacity/run_seed.py` (BackpropPopulation,
full-batch Adadelta lr 1.5, BCE, AUC fitness, sigmoid output, `mask_phantom`, `skip_invalid_genomes`,
no complexity penalty, best genome = best training fitness, seeded, `PYTHONHASHSEED=0`, one thread).
Two configurations:

- **heavy**: the capacity study's main config (`p18-h10-w015-e50`: population 150, 10 started
  hidden units with aligned ids, weight SD 0.15, 50 backprop epochs per generation, 80 generations).
  The capacity study's pilots found that the LR warm start did not beat this setting, so no warm
  start was used.
- **default**: `../complex_real` (population 250, no started hidden units, weight SD 1, 5 epochs,
  100 generations).

Three seeds per base and config; at most two runs at once (other jobs shared the machine). The
monitor AUC during a run is on the training rows; test is computed once at the end. Selection
per base: the largest training AUC over all its runs. Test AUC never entered selection.

| Base | Config | Seed | Train AUC | Test AUC | Hidden | Conns | Inputs used | Wall (min) |
|---|---|---|---|---|---|---|---|---|
| lines | heavy | 0 | **0.8815** | 0.8785 | 11 | 127 | 16 | 73 |
| lines | heavy | 1 | 0.8815 | 0.8786 | 11 | 109 | 16 | 71 |
| lines | heavy | 2 | 0.8808 | 0.8776 | 10 | 129 | 16 | 101 |
| lines | default | 0 | 0.8723 | 0.8702 | 13 | 52 | 14 | 19 |
| lines | default | 1 | 0.8681 | 0.8665 | 13 | 41 | 10 | 17 |
| lines | default | 2 | 0.8709 | 0.8694 | 9 | 31 | 10 | 16 |
| lines_cells | heavy | 0 | 0.9344 | 0.9289 | 10 | 648 | 58 | 114 |
| lines_cells | heavy | 1 | 0.9337 | 0.9282 | 10 | 648 | 58 | 127 |
| lines_cells | heavy | 2 | **0.9352** | 0.9293 | 10 | 648 | 58 | 108 |
| lines_cells | default | 0 | 0.8862 | 0.8864 | 7 | 60 | 48 | 24 |
| lines_cells | default | 1 | 0.8872 | 0.8885 | 8 | 59 | 42 | 24 |
| lines_cells | default | 2 | PERLINE_LC2 |
| perline | default | 0-2 | PERLINE_ROWS |

Bold: selected per base. Seed spread is small (heavy test AUC: lines 0.8782, SD 0.0006;
lines_cells 0.9288, SD 0.0005). References on the same split (train / test AUC):

| Base | LR | GBM | PropNEAT |
|---|---|---|---|
| cells (earlier work) | 0.857 / 0.861 | 0.947 / 0.944 | complex_real 0.858 / 0.862; capacity 0.919 / 0.915 (selected; mean test 0.911) |
| lines | 0.865 / 0.862 | 0.894 / 0.885 | heavy 0.882 / 0.879 (selected); default 0.872 / 0.870 |
| lines_cells | 0.888 / 0.888 | 0.949 / 0.943 | heavy 0.935 / 0.929 (selected); default 0.887 / 0.889 |
| perline | 0.917 / 0.919 | 0.959 / 0.952 | default PERLINE_BEST |

On the lines base PropNEAT closes 74% of the LR-to-GBM gap (0.862 to 0.885); on lines+cells 75%
of the gap from LR (0.888) to GBM (0.943), against 60% for the capacity study on cells. The
lines+cells heavy networks (0.929) are the best Connect-4 PropNEAT networks so far, above the
capacity champion (0.915).

The selected genomes were imported with `import_run.py` (experiment `c4lf-<run>`, final population
of 150 and the best genome; the DB genome reproduces the stored train and test AUC through
`StructureNetwork` to six decimals):

- lines: experiment `b4b4e7b6-1bd1-4a5d-a496-7a6bf23fe884`, genome `6ebf7e0c-55ec-4215-9852-0ac982e43c0e`
- lines_cells: experiment `19fd1a9f-69ab-4571-ba83-a3d8c7c86122`, genome `9e843d20-8e42-47d6-9c94-576c5f06fdf3`

## Explanations (`EXPLAIN_TASK.md`, `explanations/<base>/`, `VERIFY_TASK.md`, `verification/<base>/`)

One explaining agent per selected genome, the frozen method (authoring skill v4, strict mode with
provenance, function-preserving operations only, 3-hour box, training split only, no plain-language
layer), then one blind verifier per explanation (the verifier skill; read only; sees the record and
the stored explanation, not the author's scratch work or the other genome). Both explanations
reached full coverage with the recheck clean:

| | lines (`c4lf-lines-heavy-s0`) | lines_cells (`c4lf-lines_cells-heavy-s2`) |
|---|---|---|
| Time used of the 3-hour box | 27 min | 40 min |
| Operations | 17 splits (16 inputs + 1 unit), 20 consolidations, 1 identity node | 58 input splits, 1 identity node |
| Leaves / compositions | 8 / 1 | 11 / 3 |
| Coverage (structural / compositional), hierarchy | 1.0 / 1.0, complete | 1.0 / 1.0, complete |
| Function preservation (max abs err) | 8.9e-16 | 2.7e-15 |
| Rounded closed-form error | 0.020 | 0.030 |
| Numbers stated / provenance-backed / allow-listed | 229 / 202 / 27 | 251 / 219 / 32 |
| Recheck | 0 mismatch, 0 needs-review | 0 mismatch, 0 needs-review |
| Linearisation (AUC, all gates replaced by affine fits vs full) | 0.860 vs 0.882 | 0.885 vs 0.935 |
| Verifier | weakened: 0 refuted / 7 weakened / 27 upheld (+2 flags); ~150 statements, every number reproduced | weakened: 0 refuted / 11 weakened / 27 upheld; 171 statements, every number reproduced |

### What the networks do in game terms

**lines.** The network is close to a weighted sum of the line features through a sigmoid, with a
few gates. x's open twos and open singles raise x's predicted chance and o's open singles and
open twos lower it (whole-model chance 44% to 97% as x's open twos go 0 to 6 at the typical
position; 88% to 8% as o's open twos go 0 to 6). The gates: a cap on x's open twos (the unit
reading them floors near two open twos, after which more twos add nothing); a threat switch (an
o threat turns the o-lines ramp off and the chance roughly halves, 74% to 37%); a centre-column
block in which x pieces in the centre raise the chance while about three or more centre pieces
of either side lower it; a "quiet o" gate scoring positions where o has few open lines (and
usually x holds the centre) as nearly won (13,329 training positions, 93.6% x wins). Playable-cell
parity enters weakly. The verifier upheld every number and weakened five readings for scope
(the quiet-o gate fires without x in the centre on 28% of its firings; the o-threat switch is
general for the ramp but not for the cap; "both units raise x's chance with more open lines"
fails for open twos through one unit; o's open singles stop lowering the chance once the gated
units floor) and graded one hypothesis as a restated closed form.

**lines_cells.** Mostly a linear score over all 58 inputs (affine-fitting the ten gates keeps
AUC 0.885 of 0.935), with the gates adding thresholds. x's open twos and open singles and x's
centre pieces raise the chance, o's open lines lower it, a few single cells (e2, e1, a4, d4, b1,
d1) carry large direct weights, and positions with an o threat get a low predicted chance (31%
against 65%). But the gates read cells: they detect who holds c3, whether x holds g3 or d2, and
bottom-row patterns around c1 (units 2, 4, 6, 8, 9); only the two centre gates (3, 10) and the two
open-two gates (5, 7) read a line feature, and then alongside cells. The author names them
positional gates; units 6, 7, 9 and the direct sum are described, not explained. The verifier
reproduced every number and weakened eleven readings: the single-cell triggers cover only 16-33%
of each unit's active rows (x at c3 for unit 2, x at g3 for unit 4, o at c3 for unit 8; unit 9's
bottom-row rule fails on 23% of its active rows), the open-twos sweep is held at an empty board
rather than an eight-ply position (on real positions the mean chance goes 40% to 91%), and five
hypotheses restate their closed forms. It notes one real concept the record set aside for its
small n: unit 4 fires on every one of the 66 x odd-row-threat positions.

### The three levels against the cells-only networks

Computed read-only by `../explanation_levels/compute_levels.py --set c4lf` (same script and
thresholds as the cells-only rows; `levels/`), plus the author's and verifier's own grades. The
*probe (standard)* column is the script's grade from the best library concept; the *probe
(support-aware)* column (`levels_support.py`) repeats the sweep with binary concepts holding on
fewer than 1% of the training rows left out, because the standard score for a binary concept is
the AUC of the exit as a score for the concept, and the odd-row threat flags (66 and 100 rows)
top several leaves here with AUC near 1 while explaining none of their variance (both authors
noticed and set those aside; the cells-only leaves never reached partial on those flags, so their
rows are unchanged under the filter). Understood = share of leaves graded meaningful; CFO =
closed-form only.

| Network | Test AUC | Faithful | Leaves | Granularity | Probe (standard): meaningful / partial / CFO | Probe (support-aware) | Author | Verifier |
|---|---|---|---|---|---|---|---|---|
| Cells, ladder MLPs (6 networks, 35 leaves; domain concepts) | 0.885-0.942 | yes | 4.5 [1-8] | 1.0 [0-1] | **0%** / 31% / 69% | same | 0% meaningful (explainer grades) | — |
| Cells, evolved capacity champion | 0.915 | 1.7e-15 | 11 | 1.0 | **9%** / 55% / 36% (domain: 0%) | same | 0 / 0 / 11 described | upheld numbers; semantics weakened |
| **Lines (16 features)** | 0.879 | 8.9e-16 | 8 | 1.0 | **50%** / 50% / 0% | **38%** / 62% / 0% | 6 / 2 / 0 | 3 / 4 / 1 |
| **Lines + cells (58)** | 0.929 | 2.7e-15 | 11 | 1.0 | **55%** / 27% / 18% | **9%** / 64% / 27% | 0 / 7 / 4 | 0 / 6 / 5 |

Per-leaf probes are in `levels/*.json` and `levels/support_aware.json`. On the lines base the
leaves that stay meaningful under every grading are the open-twos cap (`x_open_2`, score 0.88),
the centre-column block (`centre_x + centre_o`, 0.81-0.95) and the direct terms (`x_open_2`,
0.86); the quiet-o gate is meaningful to the author and verifier (an ad-hoc concept scores 0.90)
but partial to the library sweep (0.67 on `o_open_1`). On lines+cells the only leaf meaningful
under the support-aware probe is unit 7 (`o_has_thr`, 0.84: it is on when o has a threat), which
the author grades closed-form-only because its level is set by cells; the leaves the author
grades partial are cell rules (x at c3, o at c3, x at g3) or a count plus cells.

## Answer

Did the change of base move the leaves from positional gates to named tactical concepts at
comparable AUC? **Partly, and not at comparable AUC.**

1. **Lines only (0.879 test AUC).** Yes: every leaf reads line features by construction, and
   3-6 of 8 are meaningful by whichever grader is used (0 closed-form-only by the probe). The
   understood fraction rises from 0-9% (cells) to 38-75% (lines). This is the vocabulary effect
   the hypothesis predicted. But the base itself caps at 0.885 (GBM), 0.06 below the cells, so
   the network that is understood is a weaker model: the aggregate tactical vocabulary does not
   carry the eight-ply signal.
2. **Lines + cells (0.929 test AUC).** No: given both vocabularies at a higher AUC than any
   earlier Connect-4 PropNEAT network, the hidden units are positional gates again (who holds
   c3, g3, d2, bottom-row patterns), the line features act mainly through the linear part, and
   the author's grading (0 meaningful / 7 partial / 4 closed-form-only) is about where the cells-only
   champion was (9% / 55% / 36%). The standard probe's 55% is an artefact of the rare threat
   flags; with support required it is 9%.
3. **Where the complexity lives.** The signal check says the eight-ply signal is positional and
   line-local: the per-line vocabulary (which line, how many of whose pieces) carries more of it
   than the cells and most of it linearly, while the aggregate tactical counts lose a third of the
   LR-to-GBM gap. The complexity is therefore in the *realisation and the vocabulary* in the
   specific sense that a nameable base exists on which the function is simpler (per-line counts),
   but the named concepts are not the tactical ones (threats, parity): at eight ply those are too
   rare to matter (1.2-1.4% of positions). When the cells are available, evolution and backprop
   realise the function with cell gates even though a line-feature reading would have been
   available: the base does not steer the realisation unless it forces it.

## Caveats

- Exploratory, three seeds, one split; seed spread is small but the selection reuses the training
  split and test rows were seen in the earlier baselines on this split. No multiple-comparison
  correction. The signal check is a GBM diagnostic, not a model selection.
- "Same signal" does not hold for the lines base (GBM 0.885 vs 0.944), so its understood fraction
  is at a lower AUC. The per-line base (added after the signal check) keeps the signal but was
  only run with the default config and was not explained PERLINE_CAVEAT.
- The heavy config is the capacity study's `p18` main config; the warm-start pilots there did not
  beat it, so no warm start was used.
- The concept probe's grade is not stable to rare binary concepts: both authors and the
  support-aware re-probe disagree with the standard probe on 1-5 leaves per network. The cells-only
  rows are unaffected, so the comparison in the support-aware column is like for like. The
  meaningful / partial / closed-form-only thresholds are the ones fixed in `docs/annotation_coverage_model.md`.
- The author grades and the verifier grades are judgements; the two disagree on 3 of 8 lines
  leaves (quiet-o gate, o-lines ramp, direct terms, x-singles-vs-twos gate) in both directions.
- Hidden units were not split in either explanation except one (lines); a leaf is one unit (or a
  small block), so granularity 1.0 is expected and the decomposed level is not informative here.
- Only the selected genome per base was imported and explained; the other runs' genomes are in
  `~/.explaneat/c4_line_features_runs/<name>/` (not committed).

## Reproduce

```sh
uv run python experiments/c4_line_features/baselines.py      # baselines.json
uv run python experiments/c4_line_features/signal_check.py   # signal_check.json
MAXJ=2 ./experiments/c4_line_features/run_queue.sh            # 15 runs, ~7 h at two at a time
DATABASE_URL=postgresql://localhost/explaneat_dev uv run python experiments/c4_line_features/import_run.py lines-heavy-s0
uv run python experiments/c4_line_features/summarize.py      # runs.csv, tables.md
uv run python experiments/explanation_levels/compute_levels.py --set c4lf --out experiments/c4_line_features/levels
uv run python experiments/c4_line_features/levels_support.py # levels/support_aware.json
```
