# XOR realisations: theory vs evolved PropNEAT networks

The script is `run_xor.py` and the raw output is `xor_results.json` and `run.log`. Evolution runs
are 30 seeds, PropNEAT (ReLU hidden, sigmoid output, masked backprop, 5 epochs per generation),
pop 150, stopping at the first generation whose best genome classifies all four rows.

## 1. Theory: minimal topology (1 hidden node, skip connections allowed), LP enumeration

A realisation is (P, u):
- **P:** the hidden unit's active set, i.e. which Boolean function of (x1, x2) it fires on;
- **u:** the sign of the hidden-to-output weight.

Feasibility is checked by linear programming with a margin of 1. Classes are counted under two
symmetry groups:
- the input swap x1 ↔ x2 alone;
- the *full* group: the swap, plus XOR's own input complement x → 1 − x, plus (threshold units
  only) the sign-flip symmetry σ(−z) = 1 − σ(z).

| Hidden unit | Feasible (P, u) | Classes under swap | Classes under full group |
|---|---|---|---|
| Threshold / sigmoid | 8 | 6 | **2** |
| ReLU | 16 | 10 | **6** |

The two threshold classes:
- **A: the hidden unit is a symmetric gate.** {AND (u−), OR (u+), NOR (u−), NAND (u+)}.
  Example: h = AND(x1, x2), y = step(x1 + x2 − 2h − ½). The AND and OR forms are one class:
  complementing the inputs and flipping the unit's sign maps one onto the other (the "up to −1").
- **B: the hidden unit computes one asymmetric half of XOR**, and the output adds the other half
  linearly. {x1 ∧ ¬x2 (u+), x2 ∧ ¬x1 (u+), x1 ∨ ¬x2 (u−), x2 ∨ ¬x1 (u−)}.
  Example: h = x1 ∧ ¬x2, y = step(x2 − x1 + 2h − ½).

ReLU has more classes for two reasons:
- **It has no sign-flip symmetry**, so the class-A equivalence through σ(−z) = 1 − σ(z) is
  lost.
- **Its output is graded:** a ReLU unit active on two rows passes different values on them. A
  unit active on "x1" (rows 10 and 11) can still separate those rows, which a threshold unit
  can't.

**The number of distinct realisations is relative to the activation's symmetry group.**

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

**The 6 minimal solutions fall into exactly two theoretical classes:**

| Class (full group) | Seeds | Form |
|---|---|---|
| ReLU {AND, NOR} (u−), the class-A analogue | 2, 6, 25, 27 | h = ReLU(AND hinge), y = σ(≈11·x1 + ≈11·x2 − big·h + c) |
| ReLU {x1, x2, ¬x1, ¬x2} (u+), the class-B analogue | 12 (¬x2), 21 (x1) | one-input hinge plus opposite-signed skips |

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
| Same realisation class (the AND class, 4 seeds) | 6 | **0.995** | **0.9996** |
| Different classes | 9 | 0.51 | −0.83 |
| All 30 solutions (any size) | 435 | 0.51 (median) | — |

- **Within a realisation class, the weights are identifiable.** Four independent seeds found the
  same weights up to symmetry, with cosine 0.9996.
- **Across classes they aren't.** Two solutions compute identical outputs on all four data
  points, with weights pointing in nearly opposite directions.
- **They also disagree *off* the data:** their logits correlate only 0.51 over [0, 1]². The
  signal XOR is pinned down on the data's support. Realisations are free to differ everywhere
  else. So a signal is identifiable *on the support of the data*, and two realisations can be
  equivalent there while extrapolating differently.
- One caveat: the x1 (seed 21) and ¬x2 (seed 12) solutions belong to the same full-group class,
  but are counted under "different classes" here, because the comparison doesn't apply the
  input-complement map to their weights.

## 4. What this shows

XOR is the base case of the signals / realisations framing:
- **The signal is unique.** On the data, XOR is identified exactly.
- **The realisations are not.** There are 2 classes for threshold units and 6 for ReLU,
  counted modulo the symmetry group. PropNEAT finds the ReLU analogues of both threshold classes.
- **Within a class, independent runs agree to within symmetry.** Across classes, the weights
  are unrelated.
- **Evolution mostly finds non-minimal realisations:** duplicate hinges and mixed
  decompositions. Reducing them to a minimal form is itself a recomposition step, the kind of
  simplification ExplaNEAT's annotation operations make explicit.
