# Three levels of explanation: faithful, decomposed, understood

**Probe v2, 2026-11-13, branch `fix/probe-simplicity`.** Every number below was recomputed with
the concept probe's two new conditions (`docs/annotation_coverage_model.md`, "Two conditions
on the concept"): a concept counts only when it is **simpler than the leaf** (a library
concept, or an ad-hoc expression over at most 3 variables that is not a reweighting of the
leaf's own entries; otherwise the grade is `restatement`), and a binary concept's AUC counts
only against a **gate of comparable support** (both minority shares at least 1% of the rows
and within a factor of two; otherwise the score is R², the between-groups share of variance).
The blind verifier's leaf grades, where a `verdict.json` exists (the ladder, the line-features
arms), are reported beside the probe's with the count of leaves on which the two disagree. The
v1 numbers (2026-11-12) are kept in the table at the end of this section for the record.

What moved from v1 to v2 (understood = median share of leaves graded meaningful):

| Study | v1 understood | v2 understood | v1 closed-form only | v2 closed-form only | Why |
|---|---|---|---|---|---|
| Heart | 100% [33-100] | 67% [25-80] | 0% [0-0] | 0% [0-33] | direct-term leaves were "meaningful" on a one-hot pair (`sex:0 and cp:3`, AUC 0.93, R² 0.23); gates on a conjunction three times their size |
| MONK-2 | 33% [0-100] | 33% [0-75] | 0% [0-50] | 33% [0-100] | the conjunction leaves now score R² on a single attribute level |
| Adult | 100% [88-100] | 82% [57-91] | 0% [0-12] | 9% [0-17] | rare one-hot levels with AUC 1 and R² < 0.01 (`native-country:20`, `occupation:2`) no longer count; rare levels that *are* the leaf (R² 1, `italy_branch`) still do |
| Heart-100 | 83% [75-100] | 60% [33-75] | 0% [0-0] | 20% [0-25] | as Heart |
| churn100 | 87% [57-100] | 67% [29-75] | 0% [0-25] | 21% [0-60] | one-hot state levels with AUC 1 on direct-term and state leaves |
| churn200 | 89% [71-100] | 82% [33-90] | 0% [0-10] | 11% [0-33] | as churn100 |
| Ladder, synthetic | 61% [0-75] | 61% [0-75] | 0% [0-17] | 0% [0-17] | one leaf of parity3: the `smoker and female` conjunction scored on R² (0.59, partial) rather than AUC (0.99) |
| Ladder, Connect-4 | 0% | 0% | 69% | 69% | unchanged: no rare flag reached partial here |
| Connect-4 capacity | 9% | 9% | 36% | 45% | one partial (`x_row6`, 0/1 at eight ply, AUC 0.79, R² 0.07) became closed-form only |
| C4 line features: lines | 50% (38% support-aware) | 38% | 0% | 0% | see `../c4_line_features/RESULTS.md` |
| C4 line features: lines + cells | 55% (9%) | 0% | 18% | 64% | `o_thr_odd` (100 rows) no longer counts on unit 7, and `o_has_thr` (1.4% of rows) is not comparable to a gate on 30% of rows |
| C4 line features: per-line | 67% (50%) | 50% | 17% | 50% | the three direct-term groups: library sweep closed-form only; the author's 5-19-term sums `restatement` |

Across the 60 ten-seed explanations 72 of 366 leaves lost the meaningful grade and 4 gained
it. All 72 had been graded on a binary concept's AUC alone: 58 with R² below 0.5 (the rare-flag
artefact, or a one-hot level or pair that separates the exit's extreme rows without
explaining its variance) and 14 with R² between 0.5 and 0.8 (now partial). The tabular
studies therefore moved more than the per-input vocabulary suggested they would: the v1
tabular numbers carried the same artefact as the Connect-4 threat flags, on one-hot levels.

**Read-only re-analysis, 2026-11-12, branch `feat/explanation-levels`.** Nothing in the stored
explanations, records, summaries or earlier CSV columns was changed. Every number here is
recomputed from the stored explanations by `compute_levels.py` (the genome's phenotype with the
stored operations replayed in memory; database read only; the concept probes are run locally
and not stored, so their `query_id` is null) and tabulated by `make_tables.py`
(`tables.md`, `thesis_exports/ch7/ch7_levels_table.tex`). Definitions:
`docs/annotation_coverage_model.md`, "Three levels of explanation".

| Level | Claim | Test |
|---|---|---|
| Faithful | the hierarchy computes the model's function | max abs error of the explained model state against the phenotype on the training rows (the closed-form error of the leaves is the record's) |
| Decomposed | each leaf is a small, separately readable part | leaf count; granularity = share of hidden units whose part is at most 6 units **wide** (depth not penalised); the *coarse decomposition* warning |
| Understood | each leaf maps to a concept a domain reader recognises | the concept probe: the dataset's library swept against every leaf's exits; a leaf is *meaningful* at probe score >= 0.8 (R² >= 0.8 or AUC >= 0.9), *partial* at >= 0.5, else *closed-form only*; for Connect-4 the *domain* column counts only domain concepts (threats, open lines, centre, rows, parity), not raw cells |

## Answer in brief

- **All 77 explanations are faithful** (worst function-preservation error 2.4e-15; 60 ten-seed
  records, 16 ladder networks, the evolved Connect-4 capacity network).
- **The ten-seed explanations are decomposed**: granularity 1.0 in 55 of 60 (median 1.0 in every
  study). The five below 1.0 are churn networks whose "state" leaves gather 6-10 parallel one-hot
  state units (churn100-s0, churn100-s7, churn200-s7/s8/s9); churn200-s7 is the one *coarse*
  flag (its `state_area_offsets` part is 10 units wide and 3 deep and holds half of the 33
  hidden units). Heart, MONK-2, Adult and Heart-100 are 1.0 throughout.
- **The ten-seed explanations are mostly understood in the per-input vocabulary** (probe v2):
  the median share of leaves whose best library concept grades meaningful is 82% (Adult,
  churn200), 67% (Heart, churn100), 60% (Heart-100) and 33% (MONK-2, where the leaves are
  conjunctions of attributes). The median closed-form-only share is 0-21% (33% for MONK-2).
  The v1 figures (100% for Heart and Adult, 83-89% for the rest, 0% closed-form only
  throughout) rested in part on one-hot levels and pairs that separate a leaf's extreme rows
  by AUC without explaining its variance; under v2 those leaves grade partial or closed-form
  only. What remains meaningful is what those explanations claim: per-input readings (a hinge
  on oldpeak, a gate on the international plan, a one-hot level that is the leaf).
- **Probe and verifier disagree on a third to a half of the leaves they both grade** (the
  ladder: 19 of 40 synthetic leaves, 19 of 24 Connect-4 leaves; the line-features arms: 11 of
  25). The verifier asks a different question (does the reading add anything to the formula?)
  and grades the Connect-4 MLP leaves higher than the probe (lad-be3c9: 6 of 8 meaningful to
  the verifier, 0 to the probe) and the per-line single-input hinges lower (the input itself
  is the concept). The two are reported side by side, not reconciled.
- **The ladder's synthetic networks** are faithful; decomposed below two layers (granularity
  1.0 for the one-layer networks, 0.0-0.5 for 8x8, 16x16, 32x16, where a leaf or the root
  swallows a layer); and understood at 33-75% meaningful for the one-layer networks, with the
  ground truth recovered at the root in every case (the grades of `gt_grades.json`).
- **The Connect-4 networks are faithful and (up to width 24) decomposed, but not understood.**
  This is the headline. Across the six ladder MLPs and the evolved capacity network, **no leaf
  reaches *meaningful* on a domain concept**: the best domain concept per leaf scores 0.24-0.62
  (R², or 2(AUC-0.5)). Eight of the 35 Connect-4 leaves reach *partial* on a domain concept
  (`centre_diff` three times, `o_open_1`, `x_row3`, `col_b_o`, `o_row1`, and one multi-exit
  leaf on `centre_diff`, `col_c_x` and `o_row1`); the other 27 are closed-form only in domain
  terms (v1 had nine: `x_row6` on two leaves is 0/1 at eight ply and now scores by R²). The leaves are what their names say, *positional gates*: the single
  best predictor of a leaf is usually one raw cell (`c_c1`, `c_b1`, `c_d1`, `c_c2`, scores
  0.4-0.8), which is a description of the weights, not a concept. The domain library (threat
  counts, open lines, immediate threats, parity, centre occupancy) does not predict what these
  units compute. Either the networks have not learnt those concepts (their AUCs, 0.885-0.942,
  are below GBM's 0.944 and the networks are linear or one-layer) or the units encode
  something the vocabulary does not name. The probe cannot tell these apart; it can only say
  that nothing in this vocabulary predicts the leaf.
- **Width is the barrier at the decomposed level, exactly where the ladder found it:** C4-32 and
  C4-32x16 are one leaf of 32 parallel units (granularity 0.0, the coarse flag), closed-form only.

## Granularity: the definition used here

The ladder (`experiments/explanation_barrier`) used the share of hidden units in a leaf of at
most 6 *units*, which penalises a deep leaf. Following the decision that a 3-4 deep subnetwork
is a perfectly good leaf, the metric here is **width-based**: within each annotation, the
layer of a node is its longest path from an entry node over the annotation's own edges; the
annotation's width is the most hidden units in one layer (split copies counted once); each
hidden unit belongs to the smallest annotation holding a copy of it as a non-entry node;
granularity is the share of hidden units whose part has width <= 6. A chain of four units has
width 1 and depth 4 (fine); a layer of 32 units in one leaf has width 32 (coarse). The block
also reports units that sit only in the root composition and units that appear in parts only
as entry nodes (both lower granularity: nothing explains them), the largest part's share of
the model's parameters, and the max width and depth of any part. For the one-layer networks
the two definitions agree; they differ on 8x8 (0.5 by units, 0.0 by width: each of the three
leaves holds split copies of all 8 first-layer units, so each is 8 wide) and on 16x16 (0.5 by
both, for different reasons).

## Caveats

- The probe scores *prediction* of a leaf's exit by a single concept. A leaf that genuinely
  combines several concepts (a long direct-edge sum, a conjunction of three attributes) can grade
  partial or closed-form only while being perfectly explained in words; MONK-2's conjunction
  leaves are the clear case. Pairwise products and conjunctions of a leaf's own entry inputs are
  in the tabular library, higher-order combinations are not. An explainer who names the
  combination can test it as an ad-hoc expression (`probe_concept(concept="a1:1 and a2:1 and
  a3:1")`).
- The vocabulary is per-input and feature-based; relational domains need a library, and a low
  grade means "nothing in this vocabulary predicts the leaf", not "the leaf means nothing".
- `lad-e47f6` and `lad-f01d4` (the C4-32 and C4-32x16 re-runs on `exp/explanation-barrier-2`,
  past the 26-copy split limit) have explanations in `explaneat_dev` with no annotations yet;
  they are computed (faithful 0, no leaves) and left out of the tables.
- The ladder's own explainer and verifier grades (`leaf_meaningful` etc. in `ladder.csv`) were
  judgements; the grades here are the probe's. They agree in direction (the one-layer synthetic
  networks high, the two-layer ones and Connect-4 low) but are not the same quantity.

## The three AAAI paper explanations (added 13 Nov; concept probe v1)

`compute_levels.py --set paper` (part of `--set all`): the paper's Heart, MONK-2 and Adult
explanations (`results/thesis/paper_recompute.md`; genomes `9022a9eb`, `23dde0fa`, `4bc8fa07`,
one explanation each, `explaneat_dev`, read only), written to `levels/paper-{heart,monk2,adult}.json`.
These rows were computed with the current (v1) concept probe; the probe v2 recompute will
replace them, as it will every other set. No record file exists for them, so the closed-form
error is null. They feed rows 1-3 of `thesis_exports/ch7/ch7_experiment1_cases.tex`
(`scripts/thesis/thesis_experiment_tables.py`) and are not part of the per-study table above.

| Explanation | Hidden (with split copies) | Faithful: max abs err | Leaves | Granularity | Max width / depth | Meaningful / partial / closed-form only | Leaves below meaningful |
|---|---|---|---|---|---|---|---|
| paper-heart (`a03ebc86`) | 5 | 2.2e-16 | 4 | 1.0 | 1 / 2 | 4 / 0 / 0 (100%) | none (`restbp_chol_interaction` scores 0.80, best concept `trestbps > 130`) |
| paper-monk2 (`d6ff7ca0`) | 3 | 3.3e-16 | 4 | 1.0 | 1 / 1 | 1 / 2 / 1 (25%) | `count_gate_atleast3` and `linear_count_ramp` partial (0.67, 0.73; best single concept `a2:1 and a3:1`, the leaves are counts over several attributes, as in the ten-seed MONK-2 runs); `inert_unit_a23` closed-form only (score 0, a dead unit) |
| paper-adult (`7e3761fe`) | 16 | 6.7e-16 | 10 | 1.0 | 3 / 5 | 10 / 0 / 0 (100%) | none (every leaf is a per-input term or a one-hot bump, score 1.0) |

The paper explanations are faithful and decomposed like the ten-seed ones, and understood in the
per-input vocabulary where their leaves are per-input terms (Heart, Adult); MONK-2's two count
leaves grade partial for the same reason as in the ten-seed study (the probe tests one concept at
a time; a count over three attributes is not in the library). The levels per leaf are in the JSON.

## Reproduce

```bash
uv run python experiments/explanation_levels/compute_levels.py --set v4       # explaneat_v4, 60 records, ~25 min
uv run python experiments/explanation_levels/compute_levels.py --set ladder   # explaneat_dev, 16 explanations
uv run python experiments/explanation_levels/compute_levels.py --set c4cap    # explaneat_dev, 1 explanation
uv run python experiments/explanation_levels/compute_levels.py --set c4lf --out experiments/c4_line_features/levels   # 3 explanations
uv run python experiments/explanation_levels/compute_levels.py --set paper    # explaneat_dev, the 3 AAAI paper explanations
uv run python experiments/explanation_levels/compute_levels.py --set all      # every set, one process each
uv run python experiments/explanation_levels/make_tables.py                   # tables.md, thesis_exports/ch7/ch7_levels_table.tex
uv run python experiments/10seed/build_explanations_csv.py --dir explanations_v4 --suffix _v4   # the levels columns (+ understood_fraction_verifier)
```
`--max-terms` and `--min-support` change the probe v2 conditions (defaults 3 and 0.01).

## Tables (generated: `tables.md`)
### Three levels per study

| Study | n | Faithful: max abs err (worst) | Leaves | Granularity | Coarse (n) | Understood (probe): meaningful share | Partial | Closed-form only | Domain-concept share | Author's concept (v2) | Understood (verifier) | Probe vs verifier disagree |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Heart | 10 | 4.4e-16 | 3.5 [3-6] | 1.00 [1.00-1.00] | 0 | 67% [25%-80%] | 33% [0%-75%] | 0% [0%-33%] | n/a | n/a | n/a | n/a |
| MONK-2 | 10 | 2.4e-15 | 3 [2-6] | 1.00 [1.00-1.00] | 0 | 33% [0%-75%] | 29% [0%-40%] | 33% [0%-100%] | n/a | n/a | n/a | n/a |
| Adult | 10 | 7.8e-16 | 8 [6-12] | 1.00 [1.00-1.00] | 0 | 82% [57%-91%] | 10% [0%-43%] | 9% [0%-17%] | n/a | n/a | n/a | n/a |
| Heart-100 | 10 | 4.4e-16 | 4 [4-7] | 1.00 [1.00-1.00] | 0 | 60% [33%-75%] | 22% [0%-50%] | 20% [0%-25%] | n/a | n/a | n/a | n/a |
| churn100 | 10 | 1.0e-15 | 7 [5-12] | 1.00 [0.57-1.00] | 0 | 67% [29%-75%] | 17% [0%-43%] | 21% [0%-60%] | n/a | n/a | n/a | n/a |
| churn200 | 10 | 1.3e-15 | 8.5 [6-10] | 1.00 [0.45-1.00] | 1 | 82% [33%-90%] | 11% [0%-33%] | 11% [0%-33%] | n/a | n/a | n/a | n/a |
| Ladder, synthetic (MLP) | 8 | 3.3e-16 | 5 [3-7] | 1.00 [0.00-1.00] | 2 | 61% [0%-75%] | 39% [25%-100%] | 0% [0%-17%] | n/a | n/a | 67% [0%-75%] | 19/40 |
| Ladder, Connect-4 (MLP) | 6 | 0.0e+00 | 4.5 [1-8] | 1.00 [0.00-1.00] | 2 | 0% [0%-0%] | 31% [0%-80%] | 69% [20%-100%] | 0% [0%-0%] | n/a | 20% [0%-75%] | 19/24 |
| Connect-4 capacity (evolved) | 1 | 1.7e-15 | 11 | 1.00 | 0 | 9% | 45% | 45% | 0% | n/a | n/a | n/a |

### The ladder and the Connect-4 capacity network

| Label | Family | Task | Arch | Hidden | Faithful | Leaves | Granularity | Max width | Max depth | Understood (probe) | Partial | CFO | Domain share | Author (v2) | Verifier | Disagree | Ground truth |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lad-6eb46 | c4 | c4 | 4 | 4 | 0.0e+00 | 4 | 1.0 | 1 | 1 | 0% | 25% | 75% | 0% | n/a | 50% | 3/4 |  |
| lad-be3c9 | c4 | c4 | 8 | 8 | 0.0e+00 | 8 | 1.0 | 1 | 1 | 0% | 38% | 62% | 0% | n/a | 75% | 8/8 |  |
| c4cap-main-h10-e50-s1 | c4 |  | evolved | 10 | 1.7e-15 | 11 | 1.0 | 1 | 1 | 9% | 45% | 45% | 0% | n/a | n/a | n/a |  |
| lad-bcc3b | c4 | c4 | 16 | 16 | 0.0e+00 | 5 | 1.0 | 6 | 1 | 0% | 80% | 20% | 0% | n/a | 20% | 3/5 |  |
| lad-97ea3 | c4 | c4 | 24 | 24 | 0.0e+00 | 5 | 1.0 | 6 | 1 | 0% | 60% | 40% | 0% | n/a | 20% | 3/5 |  |
| lad-da6de | c4 | c4 | 32 | 32 | 0.0e+00 | 1 | 0.0 | 32 | 1 | 0% | 0% | 100% | 0% | n/a | 0% | 1/1 |  |
| lad-61e79 | c4 | c4 | 32x16 | 48 | 0.0e+00 | 1 | 0.0 | 32 | 1 | 0% | 0% | 100% | 0% | n/a | 0% | 1/1 |  |
| lad-6f6c3 | synthetic | hinges3 | 4 | 4 | 0.0e+00 | 4 | 1.0 | 1 | 1 | 75% | 25% | 0% | n/a | n/a | 75% | 2/4 | recovered (4/4) |
| lad-b58a2 | synthetic | additive | 4 | 4 | 0.0e+00 | 4 | 1.0 | 1 | 1 | 75% | 25% | 0% | n/a | n/a | 75% | 0/4 | recovered (5/5) |
| lad-12979 | synthetic | threeway | 8 | 8 | 0.0e+00 | 4 | 1.0 | 3 | 1 | 75% | 25% | 0% | n/a | n/a | 75% | 2/4 | recovered (5/5) |
| lad-cb415 | synthetic | hinges6 | 8 | 8 | 0.0e+00 | 6 | 1.0 | 2 | 1 | 50% | 50% | 0% | n/a | n/a | 67% | 3/6 | mostly recovered (5 of 6 hinges as their own leaves) (6/7) |
| lad-f6c4c | synthetic | parity3 | 8 | 8 | 0.0e+00 | 6 | 1.0 | 2 | 1 | 33% | 67% | 0% | n/a | n/a | 67% | 6/6 | recovered (at the root; the leaves are not the true terms) (3/3) |
| lad-6e63b | synthetic | pairwise | 8x8 | 16 | 0.0e+00 | 3 | 0.0 | 8 | 2 | 0% | 100% | 0% | n/a | n/a | 0% | 0/3 | mostly recovered (at the root; the leaves entangle the two true interactions) (4/5) |
| lad-b29f2 | synthetic | rule | 16x16 | 32 | 0.0e+00 | 7 | 0.5 | 16 | 1 | 71% | 29% | 0% | n/a | n/a | 57% | 3/7 | recovered (the rule list at the root; the second layer is not annotated) (5.5/6) |
| lad-ae239 | synthetic | hinges6 | 32x16 | 48 | 3.3e-16 | 6 | 0.3333 | 6 | 1 | 0% | 83% | 17% | n/a | n/a | 33% | 3/6 | recovered at the root and in second-layer leaves; the 32 first-layer units are described only (6.5/7) |

### Connect-4 and KRK leaves against the concept library

| Label | Arch | Leaf | Exits | Best concept (any) | Score | Grade | Best domain concept | Domain score | Domain grade | Author's concept (v2 grade) | Verifier |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lad-6eb46 | 4 | xRaisedInCentre | 1 | o_row1 | 0.46 | closed_form_only | o_row1 | 0.46 | closed_form_only |  | meaningful |
| lad-6eb46 | 4 | xOwnsCentreColumn | 1 | centre_diff | 0.53 | partial | centre_diff | 0.53 | partial |  | partial |
| lad-6eb46 | 4 | leftXRightO | 1 | c_f1 | 0.41 | closed_form_only | col_c_o | 0.24 | closed_form_only |  | partial |
| lad-6eb46 | 4 | columnCOverBottomLeft | 1 | c_a1 | 0.42 | closed_form_only | o_open_1 | 0.29 | closed_form_only |  | meaningful |
| lad-be3c9 | 8 | xCentreAboveOBottom | 1 | o_open_1 | 0.54 | partial | o_open_1 | 0.54 | partial |  | meaningful |
| lad-be3c9 | 8 | oBottomCentre | 1 | c_c1 | 0.53 | partial | col_c_x | 0.28 | closed_form_only |  | meaningful |
| lad-be3c9 | 8 | xCentreThirdRow | 1 | c_d1 | 0.30 | closed_form_only | centre_diff | 0.30 | closed_form_only |  | meaningful |
| lad-be3c9 | 8 | xLowColumnD | 1 | centre_diff | 0.44 | closed_form_only | centre_diff | 0.44 | closed_form_only |  | meaningful |
| lad-be3c9 | 8 | xOnC2 | 1 | c_c2 | 0.65 | partial | col_c_x | 0.38 | closed_form_only |  | meaningful |
| lad-be3c9 | 8 | xRightOCentre | 1 | o_row2 | 0.33 | closed_form_only | o_row2 | 0.33 | closed_form_only |  | partial |
| lad-be3c9 | 8 | xBottomLeftRow | 1 | c_b1 | 0.42 | closed_form_only | o_row1 | 0.35 | closed_form_only |  | meaningful |
| lad-be3c9 | 8 | xHighColumnD | 1 | c_c2 | 0.39 | closed_form_only | o_row2 | 0.31 | closed_form_only |  | partial |
| c4cap-main-h10-e50-s1 | evolved | direct_square_values | 1 | c_c1 | 0.23 | closed_form_only | centre_diff | 0.06 | closed_form_only |  |  |
| c4cap-main-h10-e50-s1 | evolved | positional_gate_1 | 1 | c_c1 | 0.71 | partial | centre_diff | 0.57 | partial |  |  |
| c4cap-main-h10-e50-s1 | evolved | positional_gate_7 | 1 | c_b1 | 0.52 | partial | o_row1 | 0.38 | closed_form_only |  |  |
| c4cap-main-h10-e50-s1 | evolved | positional_gate_4 | 1 | c_d1 | 0.57 | partial | centre_diff | 0.44 | closed_form_only |  |  |
| c4cap-main-h10-e50-s1 | evolved | positional_gate_10 | 1 | c_a1 | 0.52 | partial | col_e_o | 0.34 | closed_form_only |  |  |
| c4cap-main-h10-e50-s1 | evolved | positional_gate_9 | 1 | o_row2 | 0.46 | closed_form_only | o_row2 | 0.46 | closed_form_only |  |  |
| c4cap-main-h10-e50-s1 | evolved | positional_gate_6 | 1 | c_d2 | 0.29 | closed_form_only | o_row1 | 0.26 | closed_form_only |  |  |
| c4cap-main-h10-e50-s1 | evolved | positional_gate_8 | 1 | x_row1 | 0.44 | closed_form_only | x_row1 | 0.44 | closed_form_only |  |  |
| c4cap-main-h10-e50-s1 | evolved | positional_gate_5 | 1 | c_b1 | 0.81 | meaningful | col_b_o | 0.62 | partial |  |  |
| c4cap-main-h10-e50-s1 | evolved | positional_gate_3 | 1 | c_c3 | 0.47 | closed_form_only | o_row2 | 0.32 | closed_form_only |  |  |
| c4cap-main-h10-e50-s1 | evolved | positional_gate_2 | 1 | x_row3 | 0.59 | partial | x_row3 | 0.59 | partial |  |  |
| lad-bcc3b | 16 | bottomRowVsUpper | 1 | o_row1 | 0.55 | partial | o_row1 | 0.55 | partial |  | meaningful |
| lad-bcc3b | 16 | sideUnits | 6 | c_b1, c_c2, c_d1 | 0.51 | partial | centre_diff, col_b_x, col_c_x | 0.36 | closed_form_only |  | closed_form_only |
| lad-bcc3b | 16 | oCentreBottom | 3 | c_b1, c_c1, c_d1 | 0.62 | partial | centre_diff, col_b_x | 0.42 | closed_form_only |  | partial |
| lad-bcc3b | 16 | upperRowUnits | 3 | c_c3, o_row2, x_row2 | 0.35 | closed_form_only | o_row2, x_row2 | 0.32 | closed_form_only |  | partial |
| lad-bcc3b | 16 | xCentreBottom | 3 | c_d1, centre_diff | 0.64 | partial | centre_diff | 0.62 | partial |  | partial |
| lad-97ea3 | 24 | bottom_row_centre_A | 5 | c_c1, c_d1 | 0.61 | partial | centre_diff, col_c_x | 0.43 | closed_form_only |  | meaningful |
| lad-97ea3 | 24 | bottom_row_centre_B | 6 | c_c1, c_d1, o_row1 | 0.56 | partial | centre_diff, col_c_x, o_row1 | 0.51 | partial |  | partial |
| lad-97ea3 | 24 | second_row_centre | 5 | c_a2, c_b2, c_c2 | 0.51 | partial | centre_diff, col_b_o, col_b_x | 0.27 | closed_form_only |  | partial |
| lad-97ea3 | 24 | upper_rows_mixed | 3 | c_b3, c_d1, x_mean_row | 0.41 | closed_form_only | x_mean_row, x_open_1 | 0.30 | closed_form_only |  | partial |
| lad-97ea3 | 24 | low_board_mixed | 5 | c_e1, c_f2, c_g1 | 0.45 | closed_form_only | centre_diff, col_e_o, col_f_o | 0.29 | closed_form_only |  | partial |
| lad-da6de | 32 | board_detectors | 32 | c_a1, c_a3, c_b1 | 0.49 | closed_form_only | centre_diff, centre_o, col_a_x | 0.38 | closed_form_only |  | partial |
| lad-61e79 | 32x16 | board_square_features | 32 | c_a1, c_a2, c_b1 | 0.49 | closed_form_only | centre_diff, col_a_x, col_b_x | 0.35 | closed_form_only |  | partial |
