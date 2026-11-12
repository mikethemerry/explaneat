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
keeps the signal exists but it is positional at the line level, not tactical. The third base
tests that: on the per-line features the default config reaches the cells champion's AUC
(0.914 test) with three hidden units, and the explanation is short and nearly linear (6 leaves,
linearised AUC 0.9119 of 0.9120). Its three units are meaningful under every probe, but as
single named lines (x on the diagonal b5-e2 against o on the vertical b3-b6, x on d6-g3, x's
even-row threats), and the three direct-term groups that carry the discrimination are weighted
sums over 34-35 lines whose best "concept" restates the formula: 3 of 6 leaves understood in game
terms (50%), 4-6 of 6 by probe (the blind verifier: 1 meaningful, 2 partial, 3 closed-form-only). The change of base therefore moves what
is understood from positional cell gates to positional line terms; it does not reach tactical
concepts at the cells' AUC.

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
`connect4_lines_cells` `347b3198-2332-4a01-97e2-cb207dfff87e` (split `b9e6a9cf-…`),
`connect4_perline` `ed521255-65a3-486f-bc23-23e437636da7` (split `7bb132f0-…`).

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
| lines_cells | default | 2 | 0.8861 | 0.8859 | 4 | 46 | 40 | 23 |
| perline | default | 0 | 0.9115 | 0.9146 | 6 | 98 | 87 | 38 |
| perline | default | 1 | 0.9107 | 0.9137 | 2 | 81 | 79 | 37 |
| perline | default | 2 | **0.9120** | 0.9140 | 3 | 86 | 82 | 35 |

Bold: selected per base. Seed spread is small (test AUC: lines heavy 0.8782, SD 0.0006;
lines_cells heavy 0.9288, SD 0.0005; perline default 0.9141, SD 0.0004). On the per-line base the
*default* config (no started hidden units, 5 epochs, 35 min a run) reaches the capacity champion's
AUC (0.914 vs 0.915) with 2-6 hidden units and about 85 connections, where the cells needed 10
started units, 50 epochs and 470 connections: on this base the function is nearly linear. References on the same split (train / test AUC):

| Base | LR | GBM | PropNEAT |
|---|---|---|---|
| cells (earlier work) | 0.857 / 0.861 | 0.947 / 0.944 | complex_real 0.858 / 0.862; capacity 0.919 / 0.915 (selected; mean test 0.911) |
| lines | 0.865 / 0.862 | 0.894 / 0.885 | heavy 0.882 / 0.879 (selected); default 0.872 / 0.870 |
| lines_cells | 0.888 / 0.888 | 0.949 / 0.943 | heavy 0.935 / 0.929 (selected); default 0.887 / 0.889 |
| perline | 0.917 / 0.919 | 0.959 / 0.952 | default 0.912 / 0.914 (selected) |

On the lines base PropNEAT closes 74% of the LR-to-GBM gap (0.862 to 0.885); on lines+cells 75%
of the gap from LR (0.888) to GBM (0.943), against 60% for the capacity study on cells. The
lines+cells heavy networks (0.929) are the best Connect-4 PropNEAT networks so far, above the
capacity champion (0.915).

The selected genomes were imported with `import_run.py` (experiment `c4lf-<run>`, final population
of 150 and the best genome; the DB genome reproduces the stored train and test AUC through
`StructureNetwork` to six decimals):

- lines: experiment `b4b4e7b6-1bd1-4a5d-a496-7a6bf23fe884`, genome `6ebf7e0c-55ec-4215-9852-0ac982e43c0e`
- lines_cells: experiment `19fd1a9f-69ab-4571-ba83-a3d8c7c86122`, genome `9e843d20-8e42-47d6-9c94-576c5f06fdf3`
- perline: experiment `fd7c8e00-a9e0-4d3c-858b-4247ee6e4d99`, genome `94fb8bdc-784b-4f75-a63c-25b1ad31da15`

## Explanations (`EXPLAIN_TASK.md`, `explanations/<base>/`, `VERIFY_TASK.md`, `verification/<base>/`)

One explaining agent per selected genome, the frozen method (authoring skill v4, strict mode with
provenance, function-preserving operations only, 3-hour box, training split only, no plain-language
layer), then one blind verifier per explanation (the verifier skill; read only; sees the record and
the stored explanation, not the author's scratch work or the other genomes). The per-line
explanation was done after the first two (same prompt and method, `explanations/perline/`,
`verification/perline/`). All three explanations reached full coverage with the recheck clean:

| | lines (`c4lf-lines-heavy-s0`) | lines_cells (`c4lf-lines_cells-heavy-s2`) | perline (`c4lf-perline-default-s2`) |
|---|---|---|---|
| Time used of the 3-hour box | 27 min | 40 min | 50 min |
| Operations | 17 splits (16 inputs + 1 unit), 20 consolidations, 1 identity node | 58 input splits, 1 identity node | 1 input split, 3 identity nodes (one per direct-term group) |
| Leaves / compositions | 8 / 1 | 11 / 3 | 6 / 1 |
| Coverage (structural / compositional), hierarchy | 1.0 / 1.0, complete | 1.0 / 1.0, complete | 1.0 / 1.0, complete |
| Function preservation (max abs err) | 8.9e-16 | 2.7e-15 | 8.9e-16 |
| Rounded closed-form error | 0.020 | 0.030 | 0.009 |
| Numbers stated / provenance-backed / allow-listed | 229 / 202 / 27 | 251 / 219 / 32 | 148 / 148 / 0 |
| Recheck | 0 mismatch, 0 needs-review | 0 mismatch, 0 needs-review | 0 mismatch, 0 needs-review |
| Linearisation (AUC, all gates replaced by affine fits vs full) | 0.860 vs 0.882 | 0.885 vs 0.935 | 0.9119 vs 0.9120 |
| Verifier | weakened: 0 refuted / 7 weakened / 27 upheld (+2 flags); ~150 statements, every number reproduced | weakened: 0 refuted / 11 weakened / 27 upheld; 171 statements, every number reproduced | weakened: 0 refuted / 10 weakened / 15 upheld; 80 statements, every number reproduced |

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

**perline.** A near-linear score with three hidden units. Three summed groups of direct terms
carry the discrimination (training AUC 0.912; 0.841 without x's per-line terms, 0.869 without
o's, 0.871 without the 16 aggregates): x's open counts on lines low on the board and through the
centre (the bottom-row horizontals, row three, the centre column, the low diagonals through the
middle), x's open singles, open twos and centre pieces raise x's chance; o's open horizontals on
rows two to four, o's diagonals into the left-centre, o's open singles and twos and an o
even-row threat lower it. The network uses 82 of the 154 inputs and none of the open-three,
total-threat or playable-parity features. The three units are small: a hinge that is on exactly
when x's open count on the diagonal b5-e2 is at least o's open count on the vertical b3-b6 (off
on 3,115 training rows; AUC 0.910 without it), a hinge on x's open count on the diagonal d6-g3
(a tiny bonus), and a unit on x's even-row threats that is on for every training row and partly
cancels its own direct term; replacing the hinges by straight lines leaves the AUC at 0.9119. The
author's own reading of the two per-line sums is a sign pattern (reward x's low and central open
lines, penalise o's low horizontals) whose best concepts are signed unweighted sums of the lines
the weights favour, which the author flags as partly restatements of the formula; the lines that
pull "the wrong way" (x's diagonals reaching row six, o's centre verticals) are within-model
offsets against the aggregate counts, not data patterns (positions with an x piece on an open
c3-f6 still win 75% against 64%). The verifier reproduced every number, count, sweep, ablation and the gate rule (on every training row and on the full count grid including ties) and weakened ten readings: the two per-line sums and the d6-g3 hinge restate their closed forms (a concept built only from the x reading's board regions, rows one to three and column d minus every line touching row six, scores r2 0.36 against 0.80 for the sign-pattern copy; the o reading alone scores 0.77, so it has substance, but row four is mixed inside the part); "open twos worth about as much as open singles" holds per standard deviation but not per line (one more open single moves the typical chance 49% to 54%, one more open two 49% to 62%, and a game-unit concept with twos at three singles scores r2 0.95 against 0.82); the o even-row threat is "nearly decisive" at the typical case (9%) but 23% on the 665 real rows; three sweep values lie outside the training range; and the game-terms text's claim that the o centre verticals "still have above-average x-win rates" is wrong (o on an open d1-d4: 46% against 70%), though the "not a data pattern" conclusion stands. Its leaf grades are 1 meaningful (the b5-e2-vs-b3-b6 hinge), 2 partial (the o-lines and aggregate sums) and 3 closed-form-only (the x-lines sum, the d6-g3 hinge and the always-on threat unit, where the input itself is the concept).

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
| **Per-line (154)** | 0.914 | 8.9e-16 | 6 | 1.0 | **67%** / 17% / 17% | **50%** / 0% / 50% | 6 / 0 / 0 by probe; 3 / 0 / 3 with the direct-term groups counted as restatements | 1 / 2 / 3 |

On the per-line base the understood fraction is given both ways because the "restatement"
concern on the three direct-term groups is real. The three units (the b5-e2-vs-b3-b6 hinge, the
d6-g3 hinge, the even-row-threat unit) are meaningful under every probe (library scores 0.99-1.0
on single per-line inputs). The three direct-term groups score 0.80-0.83 on the author's ad-hoc
concepts, but those concepts are signed unweighted sums of the lines the weights favour or
penalise, that is, a rounded copy of the formula, and no library concept comes close (standard
sweep: `x_thr_odd` 0.89 on the aggregates, a rare-flag artefact; row-six lines 0.47-0.59 on the
per-line sums; support-aware sweep: 0.35-0.49, all closed-form-only). Counting the three as
"meaningful by probe, restatement by reading" gives 3 of 6 understood (50%) in game terms, and
6 of 6 (100%) by the probe alone; the support-aware library sweep lands on the former (50% / 0%
/ 50%). The verifier goes further: it grades a leaf meaningful only when the reading adds
something to the formula, so the two single-input hinges (x on d6-g3; the always-on threat unit)
are closed-form-only to it and only the two-line hinge is meaningful (1 / 2 / 3, 17%). Under
every grading the understood leaves are per-line positional readings (which line, whose pieces,
how many), not tactical ones.

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
3. **Per-line (0.914 test AUC, the capacity champion's AUC).** At comparable AUC the function
   became nearly linear (three hidden units, linearised AUC 0.9119 of 0.9120) and the explanation
   became short (6 leaves, 148 numbers, every one provenance-backed, closed-form error 0.009).
   The three units are meaningful under every probe, but as per-line positional readings (x's
   count on the diagonal b5-e2 against o's on the vertical b3-b6; x's count on d6-g3; x's
   even-row threats), not tactical ones. The three direct-term groups, which carry nearly all of
   the discrimination, are weighted sums over 34-35 lines each whose best "concept" is a signed
   sum of the favoured lines, a restatement of the formula: understood 50% in game terms (3 of
   6), 67-100% by probe (50% support-aware), and 1 of 6 (17%) to the blind verifier, which also grades the two single-input hinges closed-form-only (the input is the concept) and the o-lines and aggregate sums partial. So the per-line base gives a faithful,
   decomposed and short explanation at the cells' AUC, but what is understood is the sign
   pattern of a linear score over named lines.
4. **Where the complexity lives.** The signal check says the eight-ply signal is positional and
   line-local: the per-line vocabulary (which line, how many of whose pieces) carries more of it
   than the cells and most of it linearly, while the aggregate tactical counts lose a third of the
   LR-to-GBM gap; the per-line network confirms it, reaching the cells' AUC with three small
   hinges on top of a linear score. The complexity is therefore in the *realisation and the
   vocabulary* in the specific sense that a nameable base exists on which the function is simpler
   (per-line counts), but the named concepts are not the tactical ones (threats, parity): at
   eight ply those are too rare to matter (1.2-1.4% of positions), and on the per-line base the
   named parts are single lines while the bulk of the function is a weighted sum that no concept
   names. When the cells are available, evolution and backprop realise the function with cell
   gates even though a line-feature reading would have been available: the base does not steer
   the realisation unless it forces it.

## Caveats

- Exploratory, three seeds, one split; seed spread is small but the selection reuses the training
  split and test rows were seen in the earlier baselines on this split. No multiple-comparison
  correction. The signal check is a GBM diagnostic, not a model selection.
- "Same signal" does not hold for the lines base (GBM 0.885 vs 0.944), so its understood fraction
  is at a lower AUC. The per-line base (added after the signal check) keeps the signal but was
  only run with the default config (no heavy runs: the default already matches the capacity
  champion), and it was explained and verified after the first two arms, with the arms' results
  in hand; its explanation is the smallest of the three (6 leaves, 3 hidden units), so its
  understood fraction rests on how the three direct-term groups are counted (see the levels
  section: 50% in game terms, 67-100% by probe). The per-line "concepts" are positional (a named
  line and a count), which the concept library provides by construction; a reader who does not
  accept a single line's open count as a game concept would grade the three units partial.
- The heavy config is the capacity study's `p18` main config; the warm-start pilots there did not
  beat it, so no warm start was used.
- The concept probe's grade is not stable to rare binary concepts: both authors and the
  support-aware re-probe disagree with the standard probe on 1-5 leaves per network. The cells-only
  rows are unaffected, so the comparison in the support-aware column is like for like. The
  meaningful / partial / closed-form-only thresholds are the ones fixed in `docs/annotation_coverage_model.md`.
- The author grades and the verifier grades are judgements; the two disagree on 3 of 8 lines
  leaves (quiet-o gate, o-lines ramp, direct terms, x-singles-vs-twos gate) in both directions,
  and on 5 of 6 per-line leaves in one direction (the author's "meaningful" is the probe grade;
  the verifier asks whether the reading adds anything to the formula, and for a one-input hinge
  or a weighted sum it mostly does not). The per-line understood fraction is therefore reported
  as a range (17-67%, 50% by the support-aware probe), not a number.
- Hidden units were not split in any explanation except one (lines); a leaf is one unit (or a
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
