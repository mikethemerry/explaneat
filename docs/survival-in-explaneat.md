# Survival analysis in ExplaNEAT: a Cox-equivalent benchmark

**Research brief, 20 August 2026.** Prepared with a verified reference implementation (`cox_torch.py`), an ExplaNEAT adapter (`explaneat_survival.py`), and a runnable benchmark (`benchmark.py`, `simulate.py`).

---

## The short version

Train the evolved network to output a scalar **log-risk** `g(x)` under the **Cox partial-likelihood loss**, then fit a **Breslow baseline hazard** on top. That gives you `S(t|x) = S₀(t)^exp(g(x))`, which is structurally identical to a Cox model with the linear predictor `βᵀx` replaced by the evolved network. Nothing else about the comparison changes: same data, same baseline-hazard machinery, same metrics, same folds. This is what DeepSurv is, and Cox-nnet before it.

One structural fact works strongly in your favour. The Cox partial likelihood couples every event to its entire risk set, so it cannot be computed correctly in mini-batches. The whole Cox-CC / sampled-risk-set literature (Kvamme et al. 2019) exists to work around that. ExplaNEAT trains **full-batch**, which is the one regime where the partial likelihood is exact. You get the correct objective for free, and you can say so in a paper.

Three things in the current codebase block this, and all three are small fixes. Details in §3.

The honest expectation, which I measured rather than assumed: on small tabular clinical data you should expect **parity with Cox, or slightly worse**. That is the state of the art, not a failure. §7 has the numbers.

---

## 1. What "as close to equivalent as possible" actually means

A Cox model asserts

```
h(t|x) = h₀(t) · exp(g(x)),   g(x) = βᵀx
```

The hazard splits into an arbitrary baseline over time and a time-constant multiplicative risk score. The partial likelihood estimates `g` without ever estimating `h₀`, which is the trick that makes Cox semi-parametric.

Replace `βᵀx` with an evolved network and *nothing else about the structure changes*. The model is still proportional hazards. Hazard ratios between any two subjects remain constant over time, `h(t|x₁)/h(t|x₂) = exp(g(x₁) − g(x₂))`, and predicted survival curves never cross. The only difference between your model and Cox is the functional form of the index. That is exactly the comparison you want: it isolates "does a non-linear index help?" from every other modelling choice.

This has a consequence worth being explicit about. If the real failure of Cox on your data is non-proportionality (crossing curves, time-varying effects), a Cox-loss network will not fix it, because it inherits the same constraint. Fixing that needs Cox-Time (`g(t,x)`) or a discrete-time model, and at that point you are no longer running a structurally equivalent comparison. Keep those as a separate question.

**Three levels of equivalence, and what each costs:**

| Level | What you compare | Cost |
|---|---|---|
| Risk ranking only | `g(x)` from both models, C-index | Cheapest. No calibration story. |
| **Risk ranking + curves** | `g(x)` plus Breslow baseline → `S(t|x)` | **Recommended.** ~20 lines. Unlocks Brier score and calibration. |
| Native survival curves | Discrete-time head, hazard per time-bin | Not Cox-equivalent. Imports grid and interpolation choices. Multi-output head fights NEAT topology evolution. |

Go with the middle row. The Breslow step is cheap and it is the difference between "my model ranks patients slightly differently" and "my model produces a calibrated survival curve".

---

## 2. The objective

With observed time `tᵢ`, event indicator `δᵢ` (1 = event, 0 = right-censored), and risk set `R(t) = {j : tⱼ ≥ t}`:

```
pll(g) = Σ_{i: δᵢ=1} [ g(xᵢ) − log Σ_{j ∈ R(tᵢ)} exp(g(xⱼ)) ]
```

Censoring enters in exactly two places. A censored subject contributes **no** outer term, because we never observed their event. But they **do** sit in the risk-set denominator of every event that happened before they were censored, because up to that point they were genuinely at risk. That asymmetry is the whole of survival analysis, and it is why you cannot just throw censored rows away or treat them as non-events.

Two practical consequences follow immediately.

**The loss is not decomposable over samples.** Every existing ExplaNEAT loss is a mean of per-row terms. This one is not. The gradient with respect to one subject's output depends on every other subject in its risk sets. This is the single most important thing to internalise about the implementation.

**The effective sample size is the event count, not the row count.** Only events contribute outer terms. On a 70%-censored dataset, 6,500 rows give you 1,950 usable terms. Expect noisier fitness and flatter differences between models.

### Ties

Real clinical data records time in days or months, so tied event times are the norm, not the exception. Rossi has 432 rows across 49 distinct times. Two corrections exist:

- **Breslow**: uses the full risk set for every tied event. Simpler, slightly biased with many ties. This is scikit-survival's default.
- **Efron**: subtracts an increasing fraction of the tied events' own risk mass from the denominator. More accurate. This is what lifelines and R's `survival` use.

Pick one and use it for **both** models, or your NN and your Cox comparator are literally optimising different objectives. I default to Efron in the implementation and match lifelines to 1.1e-13.

Worth knowing: pycox's Cox loss, which most DeepSurv reimplementations use, **ignores ties entirely**. Its own docstring calls this "a limitation, but simple and fast". If you benchmark against published DeepSurv numbers, that is part of why they are not exactly reproducible.

### Identifiability

The partial likelihood is invariant to `g → g + c`. Adding a constant cancels in the numerator and denominator; equivalently, it is absorbed into the baseline hazard. Three implications:

1. The output node needs **no bias term**. It is unidentifiable. (In the reference MLP I set `bias=False` on the output layer.)
2. Never compare raw risk scores across two fitted models. Centre first, or compare ranks.
3. Always pair a fitted `ĝ` with **its own** Breslow baseline. Mixing a network's risk scores with a Cox model's baseline is meaningless.

The loss as implemented is shift-invariant by construction, verified to 8.9e-16.

---

## 3. What has to change in ExplaNEAT

Three blockers, in order of severity.

**(a) The output activation is hardcoded to sigmoid.** `neuralneat.py` line ~413: every output layer gets `torch.sigmoid(...)`. A Cox log-risk needs an **identity** output. Squashed into (0,1), the largest hazard ratio your model can express is `e¹ ≈ 2.72`, which is not enough to represent clinical risk. Real Cox linear predictors on the datasets I tested span roughly ±3, so you need ~e⁶ of range.

The fix is to add an `output_activation` argument to `NeuralNeat.__init__`, default `torch.sigmoid` so nothing existing changes, and use `self.output_activation(...)` at that line. This is the only change that touches core inference code, and it is worth doing carefully.

**(b) `BackpropPopulation.backpropagate` ignores its own `criterion`.** It accepts `criterion=nn.BCELoss()` in `__init__` and stores it as `self.criterion`, then hardcodes `nn.BCELoss()` twice in the backprop loop (lines 152 and 158). Use `self.criterion` in both places. This is a latent bug regardless of survival work: any experiment passing a non-default criterion today is silently running BCE.

**(c) The criterion interface assumes per-row losses.** The Cox criterion needs the full `(time, event)` target, not a per-row target. `CoxCriterion` in the adapter has a standard `forward(pred, target)` signature with `target` as an `[N, 2]` tensor, so it drops into the existing call site once (b) is fixed. `BackpropPopulation`'s tensor conversion already handles a 2-D target correctly (its `unsqueeze(1)` only fires on 1-D input).

**Minor, but check it:** the existing fitness convention is `genome.fitness = 1.0 / loss`. Do not reuse that for Cox. The Cox loss is a log-likelihood on a different scale, `1/loss` is badly behaved near zero, and it makes `fitness_threshold` uninterpretable. Use `fitness = −loss` and set the config threshold relative to the null model. For reference, a constant-output genome scores exactly the mean of `log|R(tᵢ)|` over events:

| Dataset | Null loss (g = 0) |
|---|---|
| WHAS500 | 5.709 |
| GBSG2 | 5.980 |
| Veterans | 3.949 |

So a fitness of −5.71 on WHAS500 means "no signal found". That is your floor, and a much better diagnostic than an absolute threshold.

**One performance note.** The sort permutation and tie structure depend only on the dataset, never on the genome. Build the `CoxRiskSets` object once per dataset or fold and reuse it across every genome and every generation. The loss itself is then one sort-free cumulative sum, O(n), evaluated once per genome per epoch. With a population of 250 over 100 generations you are doing this ~25,000 times, so it matters.

---

## 4. Evaluation, and the trap to avoid

Use three metrics, and pre-specify them before you run anything.

**Harrell's C** is the default concordance measure and it is **biased upward under heavy censoring**. scikit-survival's own documentation shows substantial overestimation around 50% censoring. Report it, because everyone else does and you need comparability, but do not lead with it.

**Uno's IPCW C** reweights by the inverse Kaplan-Meier censoring distribution and is the one to lead with. Your candidate datasets sit at 42-70% censoring, squarely in the range where the difference matters.

**Integrated Brier score** requires `S(t|x)`, which is what the Breslow step buys you. It measures discrimination *and* calibration together. This is where a ranking-optimised model gets caught: DeepHit wins C-index and loses IBS, consistently, across independent evaluations.

**The trap** is documented as "C-hacking" (Sonabend et al., *Bioinformatics* 2022). Reviewing nine ML-vs-Cox comparison papers, they found that *without exception* none stated how they converted distributional predictions into a risk score for concordance. In their worked example the same random survival forest scored anywhere from C = 0.104 to C = 0.897 depending on that unstated choice.

You are structurally immune to this, and should say so. Both your models produce a scalar `g(x)` directly, so there is no transformation to cherry-pick. State in the methods that concordance is computed on `g(x)` for both models. That single sentence puts you ahead of most of the published deep-survival literature.

**Protocol requirements**, none of which are optional:

- Identical folds for both models. Not "the same random seed", the same index arrays.
- Tune both models or neither. The most common unfairness in this literature is a heavily-tuned network against an out-of-the-box `coxph()`. DeepSurv's own paper reports Cox baselines of 0.576 on SUPPORT where properly tuned Cox reaches 0.598.
- Give Cox the same preprocessing the network sees.
- Repeated splits with reported spread. Expected differences are ~0.01 C-index and single-split noise is larger than that. My five-repeat runs below show per-repeat swings of ±0.07 on n=137.

---

## 5. Fitness function choice

Use the **partial likelihood** as the NEAT fitness, not the C-index.

Two reasons. First, selection and backprop then optimise the same objective, so the evolutionary search and the gradient step pull in the same direction rather than fighting. Second, the DeepHit result generalises: optimising rank alone degrades calibration. If your fitness is concordance, you will evolve genomes that rank well and predict badly, and your Brier scores will show it.

`cox_concordance_fitness` is in the adapter as the direct analogue of the existing `auc_fitness`, for the ablation. It should not be the primary.

If you want a parsimony objective as well, the closest prior art is Rovito & Virgolin (GECCO 2025), who evolve symbolic features inside a Cox model with fitness = IPCW C-index plus a feature-count penalty. Multi-objective (likelihood, complexity) is the natural fit for the ExplaNEAT story and I would treat it as the interesting second experiment.

---

## 6. Where this sits in the literature

The lineage is Faraggi-Simon (1995) → Cox-nnet (2018) → DeepSurv (2018) → Cox-Time (2019). Cox-nnet is your closest structural precedent: full-batch training, a single linear output node, Cox loss plus regularisation. Same shape as what you are building.

**No prior work applies NEAT or topology-evolving neuroevolution to a survival loss.** I searched for it specifically and found nothing. The nearest neighbours are all symbolic or gradient-based:

- **CoxKAN** (*Bioinformatics* 2025) is the most important comparator. A Kolmogorov-Arnold network trained with exactly this loss, pruned, then converted to symbolic formulas. Same objective, same "extract an interpretable non-linear index" ambition, different search mechanism. Beat Cox by ≤4% C-index on five clinical datasets. Cite it and position against it.
- **Rovito & Virgolin** (GECCO 2025) evolve symbolic features inside a Cox model with NSGA-II.
- **Schlender et al.** (2026) evolve survival trees with GP-GOMEA, matching or beating DeepSurv on GBSG and METABRIC.

The gap is real and the contribution claim is safe. The transferable lesson from the GP papers is that parsimony pressure matters and censoring-aware fitness matters.

---

## 7. What to expect, measured

I ran the full protocol: five repeats, stratified splits, ridge tuned on a validation fold for Cox, early stopping on the same fold for the network, identical preprocessing. The network is a 2×32 MLP standing in for an evolved genome.

| Dataset | n / events | Model | Harrell C | Uno C | IBS |
|---|---|---|---|---|---|
| WHAS500 | 500 / 215 (57% cens.) | Cox PH | 0.751 ± 0.034 | 0.779 ± 0.079 | 0.165 ± 0.011 |
| | | NN (Cox loss) | 0.730 ± 0.030 | 0.777 ± 0.077 | 0.165 ± 0.009 |
| GBSG2 | 686 / 299 (56% cens.) | Cox PH | 0.648 ± 0.012 | 0.634 ± 0.022 | 0.136 ± 0.009 |
| | | NN (Cox loss) | 0.639 ± 0.015 | 0.630 ± 0.027 | 0.137 ± 0.008 |
| Veterans | 137 / 128 (7% cens.) | Cox PH | 0.718 ± 0.038 | 0.711 ± 0.039 | 0.171 ± 0.012 |
| | | NN (Cox loss) | 0.679 ± 0.080 | 0.673 ± 0.080 | 0.190 ± 0.022 |

Paired deltas (NN − Cox, Harrell C): −0.021, −0.009, −0.039. The network loses slightly everywhere, and the spread across repeats exceeds the mean difference in every case.

This is the expected result, not a bug. Three independent lines of evidence agree:

- **Burk et al.** (*Bioinformatics* 2025), 32 datasets, 19 models, nested CV, Bayesian tuning: "**No method significantly outperforms the commonly used Cox proportional hazards model.**" They excluded deep models entirely because they "could not be evaluated reliably due to a lack of stable implementations".
- **Meta-analysis of 21 oncology studies** (*BMC Med Res Methodol* 2025): pooled difference in C = **0.01 (95% CI −0.01 to 0.03)**.
- **Kvamme et al.**, properly tuned: gains of 0.01-0.03 on SUPPORT/METABRIC/GBSG, **zero** on FLCHAIN, and a large gain only on KKBox at n ≈ 2.6M.

The pattern is consistent. Neural survival models win on very large n, on genuinely non-linear structure, and on unstructured inputs. On small tabular clinical data they do not.

### The sanity check that matters

Because "no difference" is the expected result, you cannot use real-data parity to tell a correct implementation from a broken one. You need a case with known headroom. I simulated exponential survival with a Gaussian interaction log-risk over two coordinates, which has zero linear main effect by symmetry and so is invisible to Cox:

| Truth | Cox C | NN C | Oracle C | Spearman vs true g: Cox / NN |
|---|---|---|---|---|
| Linear | 0.826 | 0.824 | 0.828 | +0.998 / +0.993 |
| Linear | 0.788 | 0.786 | 0.787 | +0.999 / +0.991 |
| Linear | 0.682 | 0.677 | 0.683 | +0.996 / +0.971 |
| **Non-linear** | **0.494** | **0.611** | 0.624 | −0.008 / **+0.843** |
| **Non-linear** | **0.487** | **0.624** | 0.633 | −0.029 / **+0.880** |
| **Non-linear** | **0.510** | **0.598** | 0.631 | −0.015 / **+0.831** |

When the truth is linear the network matches Cox and both reach the oracle ceiling. When it is not, Cox sits at chance and the network recovers most of the available signal. **Run this first.** If your NEAT implementation cannot reproduce the bottom three rows, the problem is in your plumbing, not in the data.

### What this means for framing

Do not build the paper around beating Cox on C-index. You probably will not, and per Burk et al. nobody else has either. Build it around what NEAT adds that DeepSurv does not: a parsimonious, inspectable evolved topology at Cox-parity performance, with a decomposition story neither Cox nor a dense MLP can offer. "Matches Cox within noise while producing an interpretable non-linear index" is both defensible and, given the CoxKAN framing, exactly where the field is.

Include the non-linear simulation as a positive result in its own right. It demonstrates the method can find structure Cox cannot, on data where you know the ground truth.

---

## 8. Dataset recommendation

Start with **GBSG2** (n=686, 56% censored) and **WHAS500** (n=500, 43% events). Both ship inside scikit-survival, so no download and no preprocessing ambiguity, and both are small enough to iterate quickly.

Add **SUPPORT** (n=8,873) and **METABRIC** (n=1,904) when you want comparability with published DeepSurv and Cox-Time numbers. Both are in pycox, but note they are the *DeepSurv-preprocessed* versions, which is what you want for comparability and not what you want if you care about the underlying data.

Add **FLCHAIN** (n=6,524, 70% censored) as the heavy-censoring stress case. Every method in the literature scores 0.790 on it, so it is a good test of whether your pipeline reports "no difference" honestly.

Avoid **Veterans** as a headline dataset. At n=137 with 93% events, the repeat-to-repeat spread (±0.08) swamps any model difference. It is useful only as a small-n stress test.

One sizing caution: WHAS500 has 215 events. Classical prediction-model sample-size guidance is events-per-parameter based, and a large evolved network is simply unidentifiable at that scale. Strong parsimony pressure is not optional on these datasets, it is a statistical requirement.

---

## 9. Tooling

For the loss, **use the implementation delivered here** or **torchsurv** (Novartis, actively maintained, v0.2.0 released June 2026). torchsurv handles Efron and Breslow ties, supports strata, and is loss-only so it drops into your evaluation loop. pycox is semi-dormant (last release September 2024) and its Cox loss ignores ties.

For metrics and the Cox comparator, **scikit-survival 0.28**. It has the reference implementations of Uno's C, time-dependent AUC and IPCW Brier, with the best documentation of their censoring caveats, and you need `CoxPHSurvivalAnalysis` anyway. **lifelines 0.30** if you want Efron ties matching R and easy access to `baseline_hazard_`.

Neither torchsurv nor pycox gives you a Breslow baseline estimator you can apply to arbitrary risk scores, which is why `BreslowBaseline` is in the delivered code.

---

## 10. Files delivered

| File | Contents |
|---|---|
| `cox_torch.py` | `CoxRiskSets`, `cox_ph_loss` (Breslow + Efron), `BreslowBaseline`, `concordance`. Torch + numpy only. |
| `explaneat_survival.py` | `CoxCriterion` and two fitness functions in ExplaNEAT's `(genomes, config, xs, ys, device)` shape, plus `fit_survival_predictor`. Patch notes for the three blockers in the module docstring. |
| `benchmark.py` | The full protocol: repeated splits, tuned Cox, early-stopped NN, three metrics, paired deltas. |
| `simulate.py` | Linear and non-linear simulation. Run this first. |
| `test_cox_torch.py` | Verification suite. |

### Verification status

Everything in `cox_torch.py` is checked against reference implementations on the Rossi dataset (432 rows, 49 distinct times, so ties are exercised):

| Check | Result |
|---|---|
| Breslow partial log-likelihood vs naive reference | agrees to 1.1e-13 |
| Efron partial log-likelihood vs **lifelines** | agrees to 1.1e-13 |
| `S(t\|x)` vs **scikit-survival** `predict_survival_function` | agrees to 1.1e-16 |
| Shift invariance, `g → g + 7.3` | unchanged to 8.9e-16 |
| LBFGS on a linear model recovers the Cox MLE | max coefficient difference 6.5e-06 |
| C-index vs **scikit-survival** `concordance_index_censored` | exact |

The MLE recovery test is the one that matters most. It proves the loss, the gradient and the tie handling are jointly correct: full-batch gradient descent on a linear network reproduces lifelines' Cox coefficients.

---

## 11. Suggested order of work

1. Run `simulate.py` and `test_cox_torch.py` as delivered. Confirm the reference implementation behaves on your machine.
2. Patch `neuralneat.py` for a configurable output activation, and `backproppop.py` to honour its own criterion. Both are small and both are independently worth doing.
3. Add the `[N, 2]` survival target path through dataset loading and the DB layer. Your `Dataset` model already carries `task_type` in `additional_metadata`, so `"survival"` slots in beside `"classification"` and `"regression"`.
4. Reproduce the non-linear simulation result *through ExplaNEAT* rather than through the reference MLP. This is the real integration test.
5. Only then run GBSG2 and WHAS500 against Cox.

Step 4 is the one to not skip. It is the only test that distinguishes a working NEAT survival implementation from a broken one, because on real data both look like "no difference".

---

## Sources

Formulas and implementation choices verified against:

- DeepSurv — https://link.springer.com/article/10.1186/s12874-018-0482-1
- Cox-nnet — https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1006076
- Kvamme, Borgan & Scheel (Cox-Time, Breslow, batch-risk-set problem) — https://jmlr.org/papers/v20/18-424.html
- Burk et al., neutral comparison study — https://arxiv.org/pdf/2406.04098v2
- Wiegrebe et al., deep learning for survival analysis review — https://link.springer.com/article/10.1007/s10462-023-10681-3
- ML vs Cox meta-analysis — https://link.springer.com/article/10.1186/s12874-025-02694-z
- Sonabend et al., C-hacking — https://academic.oup.com/bioinformatics/article/38/17/4178/6640155
- CoxKAN — https://academic.oup.com/bioinformatics/article/41/8/btaf413/8209483
- Rovito & Virgolin, evolutionary symbolic survival — https://arxiv.org/html/2504.05756
- scikit-survival metrics guide — https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html
- torchsurv — https://opensource.nibr.com/torchsurv/
- pycox loss (the no-ties caveat) — https://github.com/havakv/pycox/blob/master/pycox/models/loss.py
