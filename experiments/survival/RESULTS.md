# ExplaNEAT-Cox: simulation, benchmark and explanation results

2026-11-06, branch `exp/cox-survival`. Public and simulated data only. No
PREDICT/VAREANZ data was used or accessed. User guide: `docs/survival.md`.
Brief: `docs/survival-in-explaneat.md`.

## 1. Summary

- **Through ExplaNEAT, the non-linear simulation result reproduces, and with
  margin** (brief §11 step 4). Where the true log-risk is a Gaussian bump with
  no linear main effect, linear Cox scores C = 0.49–0.51. ExplaNEAT-Cox scores
  0.59–0.63, within 0.005 of the oracle on average. Its Spearman correlation
  with the true log-risk is 0.92–0.97 (the brief's reference MLP: 0.83–0.88).
  Where the truth is linear, ExplaNEAT matches Cox to ≤ 0.001 C. The evolved
  networks drop the noise inputs: the reading in §5.2 uses only x0 and x1.
- **On real data the result is parity, as expected.** Over 10 paired
  repeated splits, ExplaNEAT-Cox minus tuned ridge Cox in Harrell's C (95%
  t-interval):

  | Dataset | ΔC |
  |---|---|
  | WHAS500 | −0.001 [−0.007, +0.005] |
  | FLCHAIN | +0.000 [−0.000, +0.000] |
  | AIDS | +0.000 [−0.024, +0.024] |
  | Veterans | +0.005 [−0.012, +0.022] |
  | GBSG2 | −0.013 [−0.026, −0.001] |

  GBSG2 is the one dataset with a small loss: its interval excludes zero.
  Uno's C, IBS and calibration (E/O) show the same picture. The networks are
  parsimonious: 4 to 17 parameters, and 18–81 events per parameter (except
  AIDS, at 7.7).
- **Getting there took two protocol revisions,** both kept on record (§4).
  The first two versions lost 0.026–0.043 C on GBSG2/WHAS500. The cause was
  selection noise and under-training, not the Cox machinery.
- **Explainability carries over.** The pre-selected GBSG2 genome is explained
  end to end with the MCP tools: full structural and compositional coverage,
  a closed-form log-risk, hazard ratios per clinical unit, and a Schoenfeld PH
  check (§5).

## 2. Setup (protocol v3, the reported one)

**Model:**

- identity output, so the network output is the log-risk g(x);
- full-batch backprop on the Efron Cox loss (`CoxCriterion`), 5 Adadelta
  epochs per generation (the PropNEAT default), phantom mask on;
- population 150, 60 generations, `weight_init_stdev = 0.1`
  (`configs/survival.ini`);
- fitness = −loss − k/D on the training fold (per-event AIC; k = enabled
  connections + hidden nodes, D = training events);
- the final genome is the generation champion (by training fitness) with the
  lowest plain validation Cox loss;
- Breslow baseline on the training rows.

**Comparator:** scikit-survival `CoxPHSurvivalAnalysis` (Efron ties), on the
same standardised features and folds, with the ridge α ∈ {1e-4 … 100} tuned
on the same validation fold by validation Cox loss.

**Folds:** event-stratified, 25% test, then 20% of the rest for validation.
Repeat r uses split seed r and run seed r. The index arrays are persisted
next to each run (`bench/*.splits.json`), and both models read the same
arrays.

**Metrics:** all computed on g(x) for both models, on the test fold:

- Harrell's C;
- Uno's C, truncated at τ inside both folds' event support;
- cumulative/dynamic AUC at the 25/50/75% quantiles of the training event
  times (mean reported);
- IBS over a 25-point grid up to the 80th test event percentile;
- test Cox loss;
- E/O, the expected/observed 1 − S ratio at the median training event time,
  by decile (Kaplan-Meier observed).

Test rows later than the last training time (0–3 per fold) are excluded from
the IPCW metrics only.

**Intervals:** 95% t-intervals over the 10 repeats. Repeated random splits
share rows, so these intervals are optimistic. Read them as "within noise or
not", not as exact coverage.

**Environment:** one torch/BLAS thread, `PYTHONHASHSEED=0`, deterministic
torch. A re-run of GBSG2 repeat 0 at the final code reproduced every metric
and the split file bit for bit. Each run takes 11–32 s.

Reproduce:

```bash
uv sync --extra survival
PYTHONHASHSEED=0 uv run python experiments/survival/study.py sim --seeds 0-4
PYTHONHASHSEED=0 uv run python experiments/survival/study.py bench --datasets gbsg2,whas500,aids,veterans,flchain --repeats 10
PYTHONHASHSEED=0 uv run python experiments/survival/study.py bench --datasets gbsg2,whas500 --repeats 10 --complexity-penalty 0 --suffix nopenalty
uv run python experiments/survival/study.py aggregate && uv run python experiments/survival/report.py
```

`tests/test_survival/test_integration_simulation.py` (marked slow, about
40 s) re-checks the simulation claims end to end on every test run.

## 3. Results

### 3.1 Simulation (`simulation.csv`, `sim/`)

n = 4000, 10 inputs uniform on [−1, 1], about 37% censored. Non-linear truth:
$g = \log 5 \cdot e^{-(x_0^2+x_1^2)/0.5}$. Linear truth: 5 active inputs.

| Truth | Seed | Cox C | ExplaNEAT C | Oracle C | Uno C Cox / NEAT | IBS Cox / NEAT | Spearman vs true g: Cox / NEAT | NEAT hidden / k |
|---|---|---|---|---|---|---|---|---|
| linear | 0 | 0.833 | **0.833** | 0.833 | 0.827 / 0.827 | 0.107 / 0.107 | +0.999 / +0.999 | 3 / 12 |
| linear | 1 | 0.811 | **0.811** | 0.811 | 0.805 / 0.804 | 0.123 / 0.123 | +0.998 / +0.999 | 2 / 11 |
| linear | 2 | 0.692 | **0.693** | 0.695 | 0.687 / 0.688 | 0.167 / 0.167 | +0.992 / +0.993 | 0 / 8 |
| linear | 3 | 0.723 | **0.724** | 0.723 | 0.712 / 0.713 | 0.165 / 0.164 | +0.993 / +0.994 | 1 / 10 |
| linear | 4 | 0.722 | **0.723** | 0.723 | 0.718 / 0.719 | 0.156 / 0.156 | +0.999 / +1.000 | 0 / 5 |
| non-linear | 0 | 0.495 | **0.601** | 0.609 | 0.498 / 0.598 | 0.210 / 0.199 | +0.045 / +0.942 | 5 / 18 |
| non-linear | 1 | 0.495 | **0.633** | 0.636 | 0.492 / 0.629 | 0.205 / 0.188 | −0.021 / +0.966 | 4 / 15 |
| non-linear | 2 | 0.501 | **0.591** | 0.593 | 0.497 / 0.590 | 0.205 / 0.197 | −0.042 / +0.953 | 3 / 13 |
| non-linear | 3 | 0.495 | **0.606** | 0.615 | 0.491 / 0.609 | 0.199 / 0.188 | −0.022 / +0.920 | 5 / 18 |
| non-linear | 4 | 0.513 | **0.623** | 0.629 | 0.510 / 0.616 | 0.200 / 0.186 | +0.005 / +0.935 | 5 / 19 |

- **Linear truth:** mean ΔC (NEAT − Cox) +0.001 (range −0.000 to +0.001).
- **Non-linear truth:** mean ΔC +0.111 (range +0.090 to +0.137). The mean
  gap from NEAT to the oracle is 0.005, and IBS is better in every seed
  (0.186–0.199 against Cox's 0.199–0.210).

The brief's reference MLP (2×32, Adam, early stopping) reached C
0.598–0.624, Spearman 0.83–0.88, on its own 60/20/20 split of the same
simulation seeds. The ExplaNEAT runs use about 18 parameters.

### 3.2 Benchmark (`benchmark.csv`, `benchmark_summary.csv`, `bench/`)

Mean ± SD over 10 repeats:

| Dataset | n / p | Train events | Censored | Model | Harrell C | Uno C | td-AUC | IBS | Test Cox loss | E/O |
|---|---|---|---|---|---|---|---|---|---|---|
| GBSG2 | 686 / 9 | 179 | 56% | Cox | 0.661 ± 0.022 | 0.644 ± 0.021 | 0.713 ± 0.034 | 0.133 ± 0.010 | 4.510 ± 0.073 | 1.04 ± 0.11 |
| | | | | ExplaNEAT | 0.648 ± 0.019 | 0.632 ± 0.019 | 0.700 ± 0.029 | 0.135 ± 0.009 | 4.521 ± 0.065 | 1.03 ± 0.12 |
| WHAS500 | 500 / 14 | 129 | 57% | Cox | 0.762 ± 0.031 | 0.772 ± 0.065 | 0.779 ± 0.051 | 0.156 ± 0.015 | 3.933 ± 0.122 | 1.05 ± 0.16 |
| | | | | ExplaNEAT | 0.761 ± 0.029 | 0.765 ± 0.057 | 0.778 ± 0.044 | 0.156 ± 0.016 | 3.922 ± 0.111 | 1.06 ± 0.18 |
| FLCHAIN | 7874 / 23 | 1302 | 72% | Cox | 0.794 ± 0.005 | 0.791 ± 0.006 | 0.820 ± 0.011 | 0.078 ± 0.002 | 6.668 ± 0.046 | 0.98 ± 0.05 |
| | | | | ExplaNEAT | 0.794 ± 0.005 | 0.791 ± 0.006 | 0.821 ± 0.011 | 0.078 ± 0.002 | 6.672 ± 0.050 | 0.97 ± 0.06 |
| AIDS | 1151 / 19 | 58 | 92% | Cox | 0.733 ± 0.053 | 0.726 ± 0.053 | 0.727 ± 0.059 | 0.039 ± 0.005 | 5.274 ± 0.391 | 1.00 ± 0.30 |
| | | | | ExplaNEAT | 0.734 ± 0.038 | 0.730 ± 0.037 | 0.711 ± 0.071 | 0.039 ± 0.005 | 5.147 ± 0.138 | 1.03 ± 0.30 |
| Veterans | 137 / 8 | 75 | 7% | Cox | 0.712 ± 0.040 | 0.702 ± 0.040 | 0.792 ± 0.047 | 0.177 ± 0.014 | 2.454 ± 0.077 | 1.14 ± 0.22 |
| | | | | ExplaNEAT | 0.717 ± 0.050 | 0.707 ± 0.049 | 0.793 ± 0.065 | 0.178 ± 0.013 | 2.473 ± 0.070 | 1.16 ± 0.20 |

Paired differences, ExplaNEAT − Cox, mean [95% t-interval]. For IBS and test
loss, negative favours ExplaNEAT.

| Dataset | ΔHarrell C | ΔUno C | ΔIBS | ΔTest Cox loss | NEAT hidden / k / inputs used | Events per parameter |
|---|---|---|---|---|---|---|
| GBSG2 | −0.013 [−0.026, −0.001] | −0.012 [−0.024, +0.001] | +0.001 [−0.000, +0.003] | +0.011 [−0.007, +0.028] | 0.0 / 6.8 / 6.8 | 27.4 |
| WHAS500 | −0.001 [−0.007, +0.005] | −0.006 [−0.019, +0.006] | −0.000 [−0.002, +0.002] | −0.010 [−0.029, +0.008] | 0.0 / 7.7 / 7.7 | 17.8 |
| FLCHAIN | +0.000 [−0.000, +0.000] | +0.000 [−0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.004 [−0.002, +0.009] | 1.6 / 16.7 / 11.7 | 81.0 |
| AIDS | +0.000 [−0.024, +0.024] | +0.004 [−0.022, +0.031] | +0.000 [−0.000, +0.001] | −0.128 [−0.350, +0.095] | 0.0 / 9.8 / 9.8 | 7.7 |
| Veterans | +0.005 [−0.012, +0.022] | +0.005 [−0.011, +0.021] | +0.000 [−0.006, +0.006] | +0.020 [−0.013, +0.052] | 0.0 / 4.0 / 4.0 | 20.9 |

**Reading:**

- **Parity, within noise, on four of five datasets.** GBSG2 shows a small
  but consistent loss (−0.013 C, interval excluding zero; Uno's C and IBS
  point the same way without excluding zero). The brief's MLP lost −0.009 on
  GBSG2 and −0.021 on WHAS500 over 5 repeats. This matches Burk et al. and
  the 0.01 meta-analytic difference. It is the expected result, not a
  failure.
- **Real data shows no non-linearity worth paying for.** On every dataset
  except FLCHAIN, the selected networks have no hidden nodes: evolution
  returns a sparse linear Cox model on a subset of the inputs. On FLCHAIN
  (1302 training events) it keeps 1–2 hidden units and gains nothing. That
  is itself a finding: under an AIC pressure, the data does not support a
  non-linear index. The simulation shows the same machinery finds
  non-linearity where it exists.
- **Calibration** at the median event time is equivalent for both models
  (E/O 0.97–1.16).
- **AIDS** has only 58 training events. Both models are fragile there:
  Cox's test-loss SD is 0.39, and one ridge fold failed badly. Its events per
  parameter (7.7) is below the usual 10–20 guidance, so do not read anything
  into its deltas.

### 3.3 Parsimony ablation (λ = 0, `*-nopenalty`)

| Dataset | ΔHarrell C | ΔIBS | ΔTest Cox loss | NEAT hidden / k | Events per parameter |
|---|---|---|---|---|---|
| GBSG2, λ = 0 | −0.003 [−0.015, +0.009] | −0.001 [−0.002, +0.001] | +0.009 [−0.035, +0.052] | 5.3 / 24.8 | 10.7 |
| WHAS500, λ = 0 | −0.013 [−0.022, −0.004] | +0.007 [+0.002, +0.013] | +0.072 [+0.018, +0.126] | 4.5 / 26.2 | 5.9 |

Without the penalty the networks grow to about 25 parameters with about 5
hidden units. On WHAS500 (5.9 events per parameter) they overfit: worse C,
IBS and test loss, all with intervals excluding zero. On GBSG2 the larger
network recovers some C (−0.003 rather than −0.013), but its test loss is no
better and its spread is wider. Keep the AIC pressure, as brief §8 says. The
GBSG2 λ = 0 result suggests λ could be tuned, but tuning it on these test
folds would be C-hacking, so it was not done.

## 4. Protocol history (honest record)

| Version | Selection | Validation score | `weight_init_stdev` | GBSG2 ΔC | WHAS500 ΔC | Files |
|---|---|---|---|---|---|---|
| v1 | best validation fitness over the whole population | penalised training fitness, with the validation event count | 1.0 | −0.026 [−0.047, −0.004] | −0.043 [−0.071, −0.015] | `bench_v1/`, `sim_v1/`, `benchmark_v1*.csv`, `simulation_v1.csv` |
| v2 | same | plain Cox loss | 1.0 | −0.027 [−0.048, −0.006] | −0.036 [−0.060, −0.013] | `bench_v2/`, `benchmark_v2*.csv` |
| v3 | generation champions only | plain Cox loss | 0.1 | −0.013 [−0.026, −0.001] | −0.001 [−0.007, +0.005] | `bench/`, `sim/`, `benchmark*.csv`, `simulation.csv` |

**v1 → v2 (design error).** v1 applied the AIC term λk/D to the validation
fold using the validation event count (about 45 on GBSG2, four times the
intended weight). That double-counts an in-sample correction on held-out
data. Selection collapsed to linear models with 2–6 inputs.

**v2 → v3 (diagnosed on training/validation losses only).** The selected
v2 genomes were under-trained. On gbsg2-r8 the training Cox loss was 5.49,
worse than the null 5.47, against 5.31 when the same topology is refitted to
convergence. They won selection by luck: about 45 validation events were
mined over roughly 9,000 genomes. v3 scores only one candidate per
generation, and starts new connections at Cox-coefficient scale.

The v3 settings were chosen on the training/validation losses of three
development folds (gbsg2-r3, gbsg2-r8, whas500-r0), not on test metrics.
Twenty backprop epochs per generation gave nothing extra at three times the
cost and was not adopted.

The test folds of GBSG2/WHAS500 had already been seen under v1/v2, so
v3's GBSG2/WHAS500 numbers are not a clean confirmatory result. AIDS,
Veterans and FLCHAIN were first run under v3. The comparator never changed.
v1 also exposed a metrics bug, now fixed: test rows past the last training
time made scikit-survival raise, which gave a NaN IBS on 3/10 WHAS500 folds.

For the cohort study, pre-register v3 (or its successor) before any run.

## 5. Explanation

### 5.1 GBSG2: the pre-selected genome, end to end via the MCP tools

The pre-stated rule picks GBSG2 repeat 0 of the reported protocol, with a
fallback to the first repeat with a hidden node if it had none. No v3 GBSG2
repeat has a hidden node, so repeat 0 it is.

`explain_setup.py` loads the genome into a local SQLite DB: the dataset is
tagged `task_type: "survival"` with y = [N, 2], the split is the run's own,
and `config_json.training` records the identity output and the Cox
criterion. The genome rebuilt from the DB reproduces the run's test C exactly
(0.65874).

`explain_run.py` then calls the MCP tool functions in-process, 23 calls in
0.4 s:

- `rename_node` ×9;
- `add_identity_node` ×2;
- leaf annotations `tumour_biology` and `nodal_burden`;
- root composition `log_risk` over the leaves, the hormone-therapy edge and
  the four zero-effect inputs;
- `get_formula`, `get_viz_summary` (sensitivity, train split), `add_evidence`,
  `get_coverage`, `compute_performance` (survival bundle, train split).

Output: `explain/gbsg2-r0.explanation.json`.

**Coverage:** structural 1.0, compositional 1.0, hierarchy complete with a
single root containing the output. Function preservation error 4.4e-16.
Closed form at 3 significant figures: maximum log-risk error 0.0043.

**Closed form of the log-risk** (standardised inputs; the bias −0.732 is
absorbed by the Breslow baseline):

$$g(x) = -0.220\,x_{\text{horTh}} + 0.312\,x_{\text{pnodes}} - 0.558\,x_{\text{progrec}} + 0.307\,x_{\text{tgrade=II}} + 0.283\,x_{\text{tgrade=III}}$$

**Leaf hypotheses in hazard-ratio terms** (per raw unit; training rows):

- `tumour_biology`:
  - progesterone receptor: HR 0.9975 per fmol/l (0.57 per SD; the p95 vs
    p5 patient HR is 2.9);
  - tumour grade II and grade III vs grade I: HR 1.92 and 1.97.
- `nodal_burden`: each positive lymph node multiplies the hazard by 1.060
  (1.37 per SD; p95 vs p5 HR 2.4).
- Direct term: hormone therapy, HR 0.63.
- Pruned to zero effect: age, oestrogen receptor, menopausal status and
  tumour size.

These directions match the GBSG2 literature: nodes and grade raise the
hazard, while progesterone receptor and tamoxifen lower it.

**PH caveat (stated in every hypothesis).** The model can only represent
time-constant hazard ratios. A Schoenfeld test (lifelines, rank transform,
training rows) on a Cox model with the same five inputs finds no violation
at 0.05, but progesterone receptor is borderline (p = 0.058). Its effect is
known to wane over follow-up. Read its HR as an average over follow-up.

Training-split survival performance of the explained model: C 0.696, Uno
0.664, td-AUC at 1/3/5 years 0.78/0.75/0.73, IBS 0.126, 5-year E/O 1.008.

**What this does and does not show.** It shows that the whole explanation
path (DB, MCP tools, formulas, evidence, coverage) works unchanged on a Cox
model. Once the output activation is right, the explanation is a sparse
linear Cox model with HRs. It does not exercise splits or hinges, because
the selected genome has none.

### 5.2 Non-linear simulation: reading hinges against a known truth

`explain_sim.py sim/sim-nonlinear-s0` (analysis classes directly, no DB).
Output: `explain/sim-nonlinear-s0.reading.json`.

**Inputs used:** exactly x0 and x1. All eight noise inputs were pruned.

**The log-risk is five ReLU hinges plus one linear term:**

$$g = 0.231 x_0 - 0.292\,\mathrm{relu}(1.15x_0+0.04) - 0.291\,\mathrm{relu}(1.24x_1-0.03) - 0.28\,\mathrm{relu}(1.1\,\mathrm{relu}(1.24x_1-0.03)+0.67) - 0.545\,\mathrm{relu}(\dots) - 0.638\,\mathrm{relu}(-0.54x_0-0.47x_1+\mathrm{relu}(1.15x_0+0.04)-0.10) + 0.52$$

**Hinge readings,** for example: node 112 adds log-HR −0.589 per unit of x0
above x0 ≈ 0, and nothing below. Combined with the +0.231·x0 term, this gives
a tent: the hazard rises towards x0 = 0 and falls beyond it. Node 1581 is the
same shape for x1.

**The shape along each axis matches the truth,** with the other inputs at
their mean:

| Axis | Model log-HR range | True log-HR range |
|---|---|---|
| x0 | 1.36 | 1.39 |
| x1 | 1.29 | 1.39 |
| x2 (noise) | 0 | 0 |

Test Spearman correlation with the true g: 0.94.

This is the explanation story the brief anticipated: the evolved network
names the two interacting covariates and the knot locations, in units a
clinician can read. A linear Cox model cannot even rank these patients
(C = 0.495).

## 6. Caveats

- The repeated-split intervals are optimistic, because the splits overlap.
  Ten repeats detect differences of about 0.01 C on GBSG2/WHAS500.
- Only WHAS500, GBSG2, FLCHAIN, AIDS and Veterans were run. SUPPORT and
  METABRIC (pycox, a download) were not.
- The comparator is a plain ridge Cox on the network's inputs. A cohort
  study also needs the published-equation comparator (splines, interactions,
  strata), see `docs/survival.md` §6.
- The horizon for td-AUC and E/O here is data-driven (training event-time
  quartiles). A cohort study pre-registers 5 years.
- PH is inherited by both models. On GBSG2, progesterone receptor is
  borderline non-PH.
