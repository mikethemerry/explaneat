# XOR realisations: theory vs evolved PropNEAT networks

The script is `run_xor.py` and the raw output is `xor_results.json` and `run.log`. Evolution runs
are 30 seeds, PropNEAT (ReLU hidden, sigmoid output, masked backprop, 5 epochs per generation),
pop 150, stopping at the first generation whose best genome classifies all four rows.

## 1. Theory: minimal topology (1 hidden node, skip connections allowed), LP enumeration

*Corrected 7 Nov 2026 after `verify_xor.py` (report: `verify_xor_report.md`, claim (e)). The
earlier version of this table labelled the 2-class row "threshold / sigmoid". The LP it comes
from fixes h = 1 on the active set and h = 0 off it, which is a step unit. Sigmoid units have 6
classes under the same group. The step and ReLU numbers are unchanged.*

A realisation is (P, u):
- **P:** the hidden unit's active set, i.e. the rows on which its pre-activation is positive
  (which Boolean function of (x1, x2) it fires on);
- **u:** the sign of the hidden-to-output weight.

Feasibility is checked by linear programming with a margin of 1. Classes are counted under two
symmetry groups:
- the input swap x1 ↔ x2 alone;
- the *full* group: the swap, plus XOR's own input complement x → 1 − x, plus (step and sigmoid
  units only) the twin symmetry h → 1 − h, i.e. σ(−z) = 1 − σ(z). ReLU has no twin.

| Hidden unit | Feasible (P, u) | Classes under swap | Classes under full group |
|---|---|---|---|
| Step (threshold) | 8 | 6 | **2** |
| Sigmoid (from `verify_xor.py`) | 28 | 18 | **6** |
| ReLU | 16 | 10 | **6** |

For sigmoid units a realisation is keyed by (family, P, u), because P and u alone do not fix the
family; 12 classes under swap and complement, 6 once the twin is added.

**Families and cells.** Every solution belongs to one of two **families**, fixed by the sign of
w1·w2 (the hidden unit's two input weights). Within a family, a **cell** is the set of rows on the
active side of the hidden unit's fold, counted by how many rows it fires on. The gates below are
for ReLU units in canonical orientation (count: w1, w2 > 0; difference: w1 < 0 < w2). The input
complement turns OR ∧ ¬AND into NAND ∧ ¬NOR.
- **Count family** (w1·w2 > 0, u < 0 in canonical form): the skips alone compute OR, the hidden
  unit flips exactly AND, so XOR = OR ∧ ¬AND. Cells: 1 row (AND), 2 rows (single input), 3 rows (OR).
- **Difference family** (w1·w2 < 0, u > 0): the skips alone compute x1 ∧ ¬x2, the hidden unit
  flips exactly x2 ∧ ¬x1, so XOR = (x1 ∧ ¬x2) ∨ (x2 ∧ ¬x1). Cells: 1 row (x2 ∧ ¬x1), 2 rows
  (single input), 3 rows (implication, x2 ∨ ¬x1).

The two step-unit classes are the two families, each with a single cell:
- **Count family:** the hidden unit is a symmetric gate. {AND (u−), OR (u+), NOR (u−), NAND (u+)}.
  Example: h = AND(x1, x2), y = step(x1 + x2 − 2h − ½). The 1-row (AND) and 3-row (OR) cells are
  one class: complementing the inputs and applying the twin h → 1 − h maps one onto the other
  (the "up to −1").
- **Difference family:** the hidden unit computes one asymmetric half of XOR, and the output adds
  the other half linearly. {x1 ∧ ¬x2 (u+), x2 ∧ ¬x1 (u+), x1 ∨ ¬x2 (u−), x2 ∨ ¬x1 (u−)}.
  Example: h = x1 ∧ ¬x2, y = step(x2 − x1 + 2h − ½).

ReLU has more classes for two reasons:
- **It has no twin symmetry**, so the 1-row and 3-row cells, which the twin σ(−z) = 1 − σ(z)
  merges for step units, stay distinct.
- **Its output is graded:** a ReLU unit active on two rows passes different values on them. A
  unit active on "x1" (rows 10 and 11) can still separate those rows, which a step unit can't.
  This adds the 2-row (single-input) cell to each family.

So ReLU has 2 families × 3 cells = 6 classes. Sigmoid units also have 2 families × 3 cells, but
different cells. They are graded, so they have the 2-row cell. They are never exactly 0, so they
have a cell with no row on the active side. The twin merges the 1-row cell with the 3-row cell,
and the 0-row cell with the 4-row cell.

**The number of distinct realisations is relative to the activation's symmetry group.** The
number of families is not: it is 2 for step, sigmoid and ReLU units.

## 2. What PropNEAT actually evolves

- **Solved:** 30/30, median generation 16.5.
- **Raw networks:** 2–8 hidden units (median 6.5), of which 1–6 are nonlinear on the data.

**Reduction to a minimal form.** Two steps alternate to a fixed point:
- *Exact relay collapse:* a hidden unit active on all four rows is linear there, so it is folded
  into direct edges. This preserves the function on the data exactly; checked.
- *Greedy pruning:* disable one connection, retrain for 300 epochs, and keep the deletion if XOR
  is still solved with a 0.1 margin.

This is a simplification of the solution, not part of evolution. Greedy pruning finds a local
minimum, so it can miss smaller realisations.

**Hidden units after reduction:**

| 1 | 2 | 3 | 4 |
|---|---|---|---|
| 6 | 14 | 9 | 1 |

**The 6 minimal solutions fall into two cells, one in each family:**

| Family and cell (full group) | Seeds | Form |
|---|---|---|
| Count family, 1-row (AND) cell: ReLU {AND, NOR} (u−) | 2, 6, 25, 27 | h = ReLU(AND hinge), y = σ(≈11·x1 + ≈11·x2 − big·h + c) |
| Difference family, the graded single-input (2-row) cell: ReLU {x1, x2, ¬x1, ¬x2} (u+) | 12 (¬x2), 21 (x1) | one-input hinge plus opposite-signed skips |

*Flag from `verify_xor.py`, for Mike: the classification above is confirmed, but every one of the
six sits at a cell boundary.* Seeds 12 and 21 are in the 2-row cell by a hair: their second active
row has h = 0.018 (seed 12, row 00) and h = 0.002 (seed 21, row 11). These values are taken from
the 3-decimal weights in `xor_results.json`, so they are good to about ±0.002. Functionally, both
seeds sit on the boundary with the 1-row difference cell. In the four count-family seeds, rows 10
and 01 have pre-activations between −0.002 and −0.009, on the boundary of the AND cell. Evolution
put the hinge's fold through data points.

The other 24 solutions keep 2–4 nonlinear units after reduction:
- **Duplicates:** two units with the same active set (seeds 1, 19, 23 and 26). The same hinge is
  realised twice inside one network, which is *within-network* aliasing.
- **Mixed decompositions:** e.g. OR + x1 ∧ ¬x2 (seed 4).

Across all solutions, 13 of the 14 non-constant Boolean active sets occur somewhere. Evolution
realises XOR in many ways.

## 3. Identifiability, made concrete

The minimal solutions were compared pairwise. The weight vector has the hidden unit's incoming
vector normalised to unit length and its outgoing weight rescaled, which removes ReLU scaling.
Logit r is computed over a 21 × 21 grid on [0, 1]².

| Pairs | n | Logit r over [0, 1]² | Weight cosine |
|---|---|---|---|
| Same cell (count family, 1-row AND cell, 4 seeds) | 6 | **0.995** | **0.9996** |
| Same cell (difference family, graded single-input cell: seeds 12 and 21) | 1 | 0.956 | 0.996 |
| Between families (count vs difference) | 8 | 0.50 | −0.84 |
| All 30 solutions (any size) | 435 | 0.51 (median) | — |

- **Within a cell, the weights are identifiable.** Four independent seeds found the same weights
  up to symmetry, with cosine 0.9996.
- **Across families they aren't.** Two solutions compute identical outputs on all four data
  points, with weights pointing in nearly opposite directions.
- **They also disagree *off* the data:** their logits correlate only 0.50 over [0, 1]². The
  signal XOR is pinned down on the data's support. Realisations are free to differ everywhere
  else. So a signal is identifiable *on the support of the data*, and two realisations can be
  equivalent there while extrapolating differently.
- **Restated 7 Nov, with Mike's agreement.** Before, this table had a row "Different classes (as labelled by
  `run_xor.py`)", with n = 9, logit r 0.51 and weight cosine −0.83. That row mixed the 8 between-family pairs
  with the pair of seeds 12 and 21.
  - Seeds 12 and 21 are in the same family and the same cell: cosine 0.996, logit r 0.956.
  - `run_xor.py` labelled them differently (¬x2, x1) only because each has one row within 0.02 of
    the fold (see §2).
  - The earlier caveat's reason, that the comparison doesn't apply the input-complement map, was
    wrong.
  - The conclusion is unchanged.

## 4. What this shows

XOR is the base case of the signals / realisations framing:
- **The signal is unique.** On the data, XOR is identified exactly.
- **The realisations are not.** Counted modulo the symmetry group, there are 2 classes for step
  units, 6 for sigmoid units and 6 for ReLU units, always in 2 families: the count family and the
  difference family. PropNEAT finds both families.
- **Within a cell, independent runs agree to within symmetry.** Across families, the weights
  are unrelated. Within a family, any two solutions are joined by a path of solutions; across
  families no such path exists (`verify_xor_report.md`, claim (c)).
- **Evolution mostly finds non-minimal realisations:** duplicate hinges and mixed
  decompositions. Reducing them to a minimal form is itself a recomposition step, the kind of
  simplification ExplaNEAT's annotation operations make explicit.

## Related work

- **Méloux et al. (ICLR 2025)** show that mechanistic explanations of small XOR MLPs are not
  identifiable: they find 85 circuits for one small XOR MLP. Cite them here. What this study adds:
  - an exact count of realisation classes under the symmetry group;
  - the comparison with what evolution actually finds;
  - the split between behaviour on the data and off it.
- **The background for the step and sigmoid cases** is the classical identifiability result for minimal
  tanh networks: Sussmann (1992), unique up to permutation and sign flips.
