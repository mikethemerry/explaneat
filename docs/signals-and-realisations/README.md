# Signals and realisations

Working argument for the reproducibility section of thesis Chapter 7 (November 2026). The companion figure sheet is `index.html` in this folder: open it in a browser. Its 3D plots load plotly.js from a CDN; everything else works offline.

Status: argument settled in discussion with Mike, 7 Nov 2026. The XOR claims in §4 were first checked in a working session against hand-chosen example weights. On 7 Nov they were verified by script: `experiments/xor/verify_xor.py` (report: `experiments/xor/verify_xor_report.md`). The *(verify)* marks have been removed for the claims that passed; §4 notes the one that passed only with a condition. Citations marked *(check)* were written from memory and must be checked in Mendeley.

## 0. Context

The thesis's central claim is that **explainability is a property of individual models, not of model classes**. A model is explainable when a verifiable explanation of it can be exhibited, whatever class it belongs to. ExplaNEAT is a demonstration of that claim: it produces verified explanations of sparse evolved neural networks, a class conventionally treated as unexplainable. This document covers one part of that demonstration, reproducibility.

## 1. The question

The thesis works with two mappings:

```
World --A--> Model --B--> Explanation
```

- **A** fits a model to data drawn from the world. Trust in A is the goal: is this model a good model of the world?
- **B** produces an explanation of the model. We do not need the process behind B to be explainable (Claude can draft the annotations). We need its output to be **verifiable**: the explanation must be checkably true to the model.
- Knowing the algorithm behind A (algorithmic understanding) does not by itself tell us whether to trust the particular model it produced. A verified explanation is how we inspect what A produced.

Reproducibility asks a question about A: **if we run World → Model ten times, do we get ten equivalent models?**

- **Cox proportional hazards:** yes. The partial likelihood is convex, so on the same data the fit converges to the same model, up to numerical tolerance.
- **PropNEAT:** no. Different seeds give different topologies and weights, even when their predictions agree (Heart: decisions agree on 94.6% of patients across 10 seeds).

So "equivalent" needs a definition that is finer than "same predictions" and coarser than "same weights". Signals and realisations provide it.

## 2. Definitions (working)

- **Support:** the points the model was trained on. Equivalence is only ever established there.
- **Signal:** a function of the inputs that the model uses, identified on the support. In practice, the contribution of one leaf annotation.
- **Realisation:** a subgraph, with its weights and activation function, that computes a signal. Two realisations of the same signal agree on the support. They can disagree everywhere else: **between** training points (interpolation) as well as **beyond** them (extrapolation).
- **Alias:** two realisations of the same signal, in different runs or within one network.
- **Family:** a set of realisations of a signal that you can move between continuously without the signal changing on the support. Counted up to symmetries that keep the mechanism: permuting hidden units, rescaling a ReLU unit, relabelling inputs.
- **Cell:** within a family, which training points fall on the active side of each unit's fold. Explanations tend to name cells ("AND gate", "single-input hinge"), so two annotations with different names can describe the same family.

Levels of equivalence, weakest to strongest:

| Level | Two models are equivalent if they have the same... |
|---|---|
| E0 | decisions on the support |
| E1 | scores on the support |
| E2 | family, for each signal |
| E3 | function everywhere |

## 3. Relation to the Rashomon set

**The concept.** Breiman (2001, "Statistical Modeling: The Two Cultures") observed that many different models fit a dataset about equally well. The **ε-Rashomon set** is the set of all models in a hypothesis class whose loss is within ε of the best achievable. Xin et al. (2022, "Exploring the Whole Rashomon Set of Sparse Decision Trees", NeurIPS) enumerate that set exactly for sparse decision trees.

**The neighbouring argument.** Semenova, Rudin and Parr (2022, "On the Existence of Simpler Machine Learning Models", FAccT) argue that when the Rashomon set is large it probably contains a simple, interpretable model; Rudin (2019, Nature Machine Intelligence) draws the practical conclusion: use interpretable models instead of explaining black boxes. *(check both in Mendeley)*

**Our position.**

- We do not need the best model, and we do not need every model to be explainable. **We need a model inside the Rashomon set, for a stated ε, that we can explain and verify.**
- The difference from Rudin and colleagues is the starting position. They start from model classes being explainable or not, and restrict the search to a class assumed interpretable. We start from **some models** being explainable: we search a neural hypothesis space and **certify each model found** by exhibiting a verified explanation. Explainability is a property shown for the model, not assumed from its class, and that opens up different methods.
- What that requires is a method that **reliably, though not necessarily quickly,** finds a model in the set that can be explained. In the 10-seed experiment, PropNEAT plus Claude-drafted explanations reached full coverage for 30 of 30 best genomes, in a median of 12 to 44 minutes per network; Mike's verification of them is still to come. A "train until explainable" loop is acceptable if each pass is reliable.
- The models PropNEAT reaches across seeds are a **sample** from (or near) the Rashomon set of its hypothesis class. Signals, families and realisations partition that sample **by mechanism**, not by loss: which signals a model carries, and which realisation of each it uses, decides how it can be explained and how it behaves between and beyond the training points. Structuring and navigating this is the contribution, not recovering the whole set.

**Expectation (with supporting evidence, not proof):** to be inside the Rashomon set, a model will probably need realisations of the **major** signals, without carrying overfit ones. How many signals it needs depends on ε: relaxing ε lets a model drop the smaller signals, which also makes it simpler to explain. Heart is an example of the trade-off: its seed-specific nonlinear corrections are each worth only 0.005 to 0.022 train AUC, so a slightly larger ε admits models without them.
- Churn: all 23 named nonlinear features add test AUC, while the overfit state offsets do not (`experiments/churn/EXTENSIONS_RESULTS.md`, the held-out check of the named features and the explanation-guided repair study).
- MONK-2: the 7 seeds that miss the window's lower edge sit far outside the set (test AUC down to 0.64). Their explanations say exactly what is missing, and grafting that one piece from a correct seed brings all 7 to test AUC 0.996-1.0 (`GRAFT_F_G_RESULTS.md`, MONK-2 grafting). That is navigating *into* the set using explanations.
  - **Caveats:** a random-weight graft of the same shape gets close (the real graft leads by only +0.007 after fine-tuning: 7 better, 1 worse, 6 tied); the test set has only 87 rows.
  - **Scope:** in the thesis as a light example (Mike, 7 Nov).

**Measured vs not yet measured.**
- Best genome per seed, test AUC spread: Adult 0.903-0.909 (all inside for ε of about 0.006); Heart 0.865-0.912; MONK-2 0.639-1.000 (not all inside for any small ε). Source: `experiments/10seed/summary.md`. The thesis needs to state ε and the reference "best" (best observed model, or a strong baseline such as gradient boosting).
- Whether whole **populations** at convergence sit inside the set has not been measured. It can be checked from the final-generation fitness distribution; do not claim it until it is.

## 4. Worked example: XOR (`index.html`)

- **Signal:** XOR on the four points 00, 10, 01, 11.
- **Minimal network:** one hidden unit plus skip connections. With no hidden unit, XOR is impossible (not linearly separable); without skips the minimum is two hidden units.
- **Two families**, each a composition of two logic gates on the data:
  - **Count:** skips alone compute OR; the hidden unit flips exactly AND; XOR = OR ∧ ¬AND. Over the plane this is a **ridge** along x₁ + x₂, with 10 and 01 on the crest.
  - **Difference:** skips alone compute x₁ ∧ ¬x₂; the hidden unit flips exactly x₂ ∧ ¬x₁; XOR = (x₁ ∧ ¬x₂) ∨ (x₂ ∧ ¬x₁). This is a **valley** along x₁ − x₂, with 00 and 11 in the trough.
- **Four realisation shapes:** each family with a ReLU or a sigmoid hidden unit. All four are identical on the four points and differ between them (for example at (0.5, 0.5)) and beyond.
- **Three cells per family:** as the fold slides past data points the hidden unit's name changes (AND, single input, OR), but the gates it composes do not, provided the weights stay in proportion.
- **What decides the family:** XOR is solvable with one hidden unit plus skips exactly when I = h(00) + h(11) − h(10) − h(01) ≠ 0. The output weight u must have the opposite sign to I and a size of at least 4m / |I|, where m is the margin every row must clear (t·z ≥ m); for unit margin, |u| ≥ 4 / |I|. The bound is exact, not approximate: summing the four margins gives −u·I, and equality is reached. Verified by LP and random sampling for ReLU, sigmoid and step units: 0 mismatches in about 13,000 LP optima and 16,000 random-u checks.
  - For ReLU, sign(I) = sign(w₁ · w₂) whenever the unit is non-degenerate on the four points, i.e. some row active and some inactive. Same-sign hidden weights give the count family; opposite signs give the difference family. 0 exceptions in 208,906 non-degenerate units.
  - For sigmoid, each family appears twice (u > 0 and u < 0), but the two are exact twins under h → 1 − h, so they are one realisation counted twice. The twin map (w, b) → (−w, −b), u → −u, c → c + u gives the same z everywhere, not just on the four points: the largest difference over 6,000 solutions and points up to 10³ away is at float rounding (2 × 10⁻¹³ relative).
  - Class counts depend on the unit. Under input swap, input complement and (where it exists) the twin: step units have 2 classes, sigmoid units 6 and ReLU units 6, always in the same 2 families. The earlier "threshold / sigmoid: 2 classes" holds for step units only; `experiments/xor/RESULTS.md` is corrected.
  - The gate reading above (skips compute one gate; the hidden unit flips exactly one row) holds for every sampled ReLU solution. For sigmoid units it often does not: a sigmoid is never exactly 0, so the skips alone are wrong on two rows in about 63% of sampled sigmoid solutions.
- **What PropNEAT found** (`experiments/xor/RESULTS.md`): 30/30 seeds solved XOR. 6 reached the minimal form, in **both** families: 4 count, in the 1-row (AND) cell; 2 difference, in the graded single-input cell. The other 24 kept redundant units (duplicate hinges, mixed decompositions) above the minimum. All six minimal solutions put the hinge's fold within 0.01 of a data point, so they sit at cell boundaries. The two difference-family seeds are single-input by a margin of h = 0.018 and 0.002.
- **Moving between realisations:** within a family, straight paths between cells keep XOR throughout, **with one condition**. Both solutions must be aligned: put in the family's orientation by input swap or complement, and rescaled so that |u| = 1 (ReLU rescaling). That is then proved, and confirmed on about 40,000 sampled pairs of ReLU solutions, 2,000 per cell. With a different rescaling (hidden weights of unit length) about a third of straight paths break XOR. For sigmoid units, which have no rescaling symmetry, about half do. A piecewise path that re-fits the readout as the hidden weights move works for every pair, ReLU and sigmoid. Between families no path exists: along any path of solutions u·I stays negative, so u would have to pass through 0. 0 of 12,000 sampled between-family straight paths work.
- **The h/z transposition:** swapping the hidden unit's weights with the skips' weights works only for the narrowest (1-row) difference cell, and then for every solution in it. It never works in the count family or in the wider difference cells (0 of 10,000 sampled solutions). The new u must exceed −a(r⁺)/s(r⁺), where r⁺ is the positive row on which the hidden unit is silent. This threshold depends on the solution's weights. It is 6 for the figure sheet's example (so 6.25, its first slider step above 6), and ranges from 0.24 to 16.6 over sampled solutions. It is a ratio for each solution, not a constant of the cell.

## 5. Evidence on real data (this repo)

| Level | Finding | Source |
|---|---|---|
| E0 decisions | Heart, 10 seeds: decisions agree on 94.6% of patients; training logits correlate at a median of 0.983 | `experiments/10seed/compare_heart.json` |
| Shape | Heart: 3–6 hidden nodes per seed | `experiments/10seed/summary.md` |
| Signals recur | Churn: every useful feature in one seed is about 90% reproducible from any other seed's own units | `experiments/churn/EXTENSIONS_RESULTS.md`, cross-seed signal reproducibility: the novelty screen, measured on train before any transplant |
| Realisations are interchangeable | Knock out a feature, graft another seed's alias, refit the readout: about 94% of the lost function is recovered; beats a same-topology random graft in 35/35 cases and a wrong-family feature in 32/35 | same file, knockout-and-rescue with algorithmic aliases |
| Topology does not fix the realisation | Re-initialising the evolved churn topologies: 0/100 networks come back equivalent to the evolved weights | same file, re-initialisation |

The 10-seed explanations were drafted by Claude and have not yet been verified by Mike (0/30 as of 6 Nov).

## 6. Why it matters

- **Equivalent on the data is not equivalent in use.** Realisations differ between and beyond the training points. For a given use, one family or realisation may be preferable: for example, one whose behaviour between known cases is monotone, or clinically plausible.
- Two steps make this workable:
  1. **Hypothesis:** the explanations assert that two annotations, in different models or the same one, compute the same signal.
  2. **Empirical test:** manipulate the models to swap or morph between realisations (knockout and rescue, grafting, paths within a family) and check that the signal is preserved.
- With both, the choice among Rashomon-equivalent models can be made on **mechanism**, not only on loss.
- **Shown so far:** both steps on XOR, and the empirical step on churn. **Not yet shown:** deliberately choosing a realisation for a use case. That is future work.

## 7. Open items before the thesis cites this

- ~~Confirm the XOR claims in section 4 with a script in `experiments/xor/`.~~ Done 7 Nov: `experiments/xor/verify_xor.py`, report `verify_xor_report.md`. All passed except straight-line connectivity, which holds only with the alignment stated in §4 (a piecewise path always works). The transposition threshold is a ratio for each solution, not the constant 6.25.
- ~~Correct `experiments/xor/RESULTS.md`.~~ Done 7 Nov. "Threshold / sigmoid: 2 classes" is now a step-unit result (sigmoid units have 6), and seeds 12 and 21 are the graded single-input cell of the difference family. Two flags for Mike are in that file: the six minimal seeds sit at cell boundaries, and seeds 12 and 21 are counted in its "different classes" row although they share a cell.
- PropNEAT uses ReLU hidden units, so the sigmoid realisations are a theoretical illustration only.
- Check every citation in section 3 in Mendeley.
- Measure the population-level Rashomon claim (final-generation fitness spread) before making it.
- State ε and the reference best model for each dataset.
