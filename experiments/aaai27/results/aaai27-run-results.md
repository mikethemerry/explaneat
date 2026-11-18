# AAAI-27 paper results — [RUN] placeholder fills

Generated 2026-07-10 on branch `analysis/aaai27-paper-results` (off `main`).
All numbers computed on the stored splits with stored scalers (seed 42
everywhere, no re-splitting). Scripts under `scripts/aaai27/`; every number
below is reproducible by re-running the named script.

## Ground truth (scripts/aaai27/verify_ground_truth.py)

**⚠ Experiment-attribution correction (the one discrepancy found).** The task
doc attributed the Adult model to experiment "Adult - OHE - 250-100-5"
(`5c059459`). That experiment's best fitness over 100 generations is 0.8965
and its only annotated genome (8 nodes / 125 genes, 5 annotations) carries
`prune_node` operations — it cannot be the paper model. The unique genome
matching **every** paper claim is:

- **Adult**: genome `4bc8fa07-bbf5-48e4-96a8-77be8b577ba3`, generation 99 of
  experiment **"No pressure"** (`dbab685c-e9c8-4118-bb4e-0fbdfab4c4c9`).
  Verified: 26 nodes, 171 connection genes, 92 enabled edges, 54
  phenotype-connected inputs, 14 hidden ReLU units, train AUC 0.9085
  (n=39,073), test AUC 0.9055 (n=9,769), 11 annotations, exactly 9
  identity-node insertions and zero function-changing operations.
  The experiment's split `1df50169` is **index- and scaler-identical** to the
  task-specified split `7c025383`, so all split-based instructions hold.
  Confirmed by Mike 2026-07-10.
- **Heart**: genome `9022a9eb-ecf9-41ba-b138-28ed4d131c9d`, experiment "Heart
  Binary" (`c8f3805e`), split `165cebbd` — reconciles exactly as specified
  (8 nodes, 42 genes, 19 connected inputs, 5 hidden units, 0.9455/0.8966,
  6 annotations, 8 node splits, nothing function-changing).

## fitness-metric

**AUC-ROC evaluated on the full training split** (the runner's `auc`
evaluator); no validation slice. Stored genome fitness equals train-split AUC
to 4 decimal places for both models (Adult 0.9085, Heart 0.9455). **No
structural penalty**: configs contain no penalty terms and both runs predate
the parsimony feature.

Suggested .tex text for `[RUN: confirm exact fitness metric]`:
> Fitness was AUC-ROC on the training split

## adult-logistic, heart-logistic (scripts/aaai27/run_a_logistic.py)

L2 LogisticRegression, C tuned by 5-fold CV on the train split only
(roc_auc scoring), same one-hot features and stored z-normalisation.

- `adult-logistic`: **0.905** — 95% bootstrap CI [0.899, 0.912] (C=0.1)
- `heart-logistic`: **0.919** — 95% bootstrap CI [0.841, 0.978] (C=0.1)

**⚠ Flags for the prose around these numbers:**
1. On Adult, the logistic baseline **ties the evolved network** (0.905 vs
   0.9055). The paper's "for reference" framing survives, but any implication
   that the evolved net outperforms LR on *these* datasets does not.
2. On Heart, the logistic baseline **beats the evolved network's 0.897 point
   estimate** (0.919), though n=61 makes the CI very wide and it comfortably
   contains 0.897.

## adult-ablation-CI (scripts/aaai27/run_b_ablations.py)

Full model vs the 5 native-country edges into the capital-gain core disabled
(edges `-73→655`, `-81→655`, `-82→655`, `-89→2129`, `-94→2129`; the paper's
quoted weights +1.10 / −1.39 are `-73→655` = +1.0964 and `-81→655` = −1.3883).
Direct-to-output linear terms of `-73`/`-82` left enabled. Measurement-only
genome copies; nothing persisted.

Point estimates reproduce the paper exactly: test AUC 0.9055 → 0.9054,
accuracy 0.851 → 0.852 (0.8514 → 0.8520 at 4dp).

- `adult-ablation-CI`: **ΔAUC = −0.0001, 95% CI [−0.0003, +0.0002]**
  (paired bootstrap, 10,000 resamples, seed 42, 0 skipped; n=9,769)

## heart-ablation-CIs (scripts/aaai27/run_b_ablations.py)

Ablations disable the named unit's readout edge(s) into output 0: benign-
profile gate = node 155, resting-BP interaction = node 2379, risk-aggregation
gate = node 534; linear core = all three. All point estimates reproduce the
paper's table within 0.001. Full model test AUC 0.8966, n=61.

| Configuration | Test AUC | Δ | 95% CI |
|---|---|---|---|
| Full model | 0.897 | — | — |
| Linear core only | 0.860 | −0.037 | **[−0.079, −0.005]** |
| − benign-profile gate | 0.869 | −0.028 | **[−0.069, +0.001]** |
| − resting-BP interaction | 0.885 | −0.012 | **[−0.031, +0.002]** |
| − risk-aggregation gate | 0.899 | +0.002 | **[−0.011, +0.016]** |

(paired bootstrap, 10,000 resamples each, seed 42, 0 skipped)

**⚠ Flag:** only the "linear core only" delta excludes zero at 95%. The
benign-profile (−0.028, CI touching +0.001) and resting-BP (−0.012) deltas
straddle zero — expected with n=61, but the paper's "the nonlinear units
genuinely help" / "costs 0.028" prose should be read against these intervals.
The risk-aggregation row supports the paper's claim (at worst unchanged).

## dense-width, dense-auc, mean-fanin, weight-spread (scripts/aaai27/run_c_dense_twin.py)

MLP on Adult's 54 connected features, same split/scaling: 54–16–16–1, ReLU,
Adam (lr 1e-3), weight decay 1e-4, full-batch, seed 42. No L1, no pruning.
The starting width (16) landed inside the target band immediately.

- `dense-width`: **16** (two hidden layers of 16)
- `dense-auc`: **0.909** (0.9089; target 0.905 ± 0.005 ✓, evolved 0.9055)
- `mean-fanin`: **31** (mean 31.0 of 54 inputs; median 32, min 10, max 40)
- weight-spread: |w| ≥ 10% of the unit's max |w| for **57.4%** of inputs on
  average (per-unit range 18.5%–74.1%); first-layer |w| mean 0.086,
  median 0.054, max 0.959
- Per-unit effective fan-in: [10, 25, 26, 27, 28, 29, 31, 32, 33, 33, 35, 36, 36, 37, 38, 40]

Suggested fill for the .tex sentence "each first-layer unit mixes weight from
[RUN: mean-fanin] features (mean $|w| >$ [RUN] for [RUN]\% of inputs)":
> each first-layer unit mixes weight from 31 of the 54 features on average
> ($|w| \geq 10\%$ of the unit's largest weight for 57\% of inputs)

**No stop condition**: mean effective fan-in is high (31 ≫ evolved 2–6); the
units are thoroughly mixed and not nameable. Section 5.4's contrast stands.

Caveat: training ran to the 2,000-epoch cap without the train-loss plateau
(patience 50) triggering; loss was still improving slowly. Test AUC was
inside the target band, so the width search stopped at 16 as specified.

## fig:capgain (scripts/aaai27/run_d_capgain_figure.py)

No original plotting code exists in the repo; regenerated from the annotated
subgraph (weights/biases read from the stored genome at runtime; manual
forward cross-checked against the phenotype `StructureNetwork` to 1e-9).
Reference: five native-country inputs at raw-0 (z-scaled); external node
10307 is bias-only ReLU(−0.736) ≡ 0 (permanently inert).

Output: **results/capgain-response.png** (300 dpi; title, dual raw/z x-axes,
threshold marker, inset zoom of $0–$10,000).

Exact `\caption` text:

> Response of the capital-gain pathway (the depth-3 ReLU cascade, hub, and
> four output readouts of annotation \textit{capital\_gain\_core}) over the
> observed range of capital-gain, holding the five native-country inputs at
> zero; the vertical axis is the pathway's contribution to the output logit
> of $P(\text{income} \leq 50\text{K})$. Despite comprising seven neurons the
> pathway is flat for the 92\% of individuals with zero capital gain and then
> a steady downward ramp -- functionally a single threshold at
> $\approx$\$3,225 (exactly: two knots at \$3,225 and \$4,300,
> indistinguishable at data scale) -- pushing predictions towards $>$50K as
> capital gain grows.

**⚠ Nuance vs the paper text:** the exact piecewise-linear response has **two
knots** in the observed range ($3,225 and $4,300, i.e. ~1% of the x-range
apart; slope −3.37 between them, −6.24 after). At data scale they read as a
single threshold, so "functionally a single ReLU" holds, but "one knot" is
strictly imprecise — the caption above words this honestly. The positive
class is `<=50k` (`class_names = ['>50k', '<=50k']`), hence the *downward*
ramp; the paper's Section 5.2 prose ("threshold... then a linear ramp") is
direction-agnostic and unaffected.

## Did not reproduce / needs author attention

1. **Adult experiment attribution** (ground truth section above) — the only
   outright discrepancy; genome and all its stats verified in "No pressure".
2. **"One knot"** in the capital-gain pathway is exactly two close knots (see
   fig:capgain section); functionally-single-threshold claim survives.
3. **Heart ablation CIs straddle zero** for the two "helpful" gates (n=61);
   point estimates all reproduce, but significance language should be
   avoided in tab:heart-ablation prose.
4. **Baselines are not flattering**: LR ties the Adult model and beats the
   Heart point estimate (see Run A flags). Nothing to reconcile — the paper's
   argument is about explainability, not accuracy — but the sentence
   introducing [RUN: adult-logistic]/[RUN: heart-logistic] should be worded
   with these values in mind.

Everything else reproduced within the 0.002 tolerance: all five ablation
point estimates, both models' train/test AUCs, node/gene/input counts,
annotation and operation counts, and the ±1.10/−1.39 country weights.
