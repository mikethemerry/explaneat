# Churn signals and their realisations: the ten churn100 networks

**Exploratory, not pre-registered** (2026-11-18, branch `analysis/churn-signals`). The ten networks
are Experiment 2's churn100-s0..s9, with their v4 explanations (`experiments/10seed/explanations_v4`,
layers `plain_language_v4`). The data were read only from explaneat_v4 (READ ONLY), and no v4 record or layer was changed.
Everything is fitted and described on the **4,000 training customers**. **TEST** numbers (1,000
customers) are reported only and labelled, and nothing was selected or tuned on them.

Terms (Mike's): a **signal** is a churn-relevant function of the inputs, defined in data terms and
independent of any model. A **realisation** is the subgraph and weights that compute it in one network.
An **alias** is a different realisation of the same signal; this is not signal-processing aliasing.
Σ(M) is the set of signals network M uses. A signal is **decodable** in M when it can be read off
M's hidden activations (novelty = 1 − R² < 0.5), and **used** when M's score actually moves with it.

## In plain words

All ten networks are built from the same short list of churn signals:

1. **Customer-service calls.** Risk is flat up to 3 calls and then climbs steeply.
2. **Heavy daytime use.** Risk rises with day minutes (or day charge, which is the same thing) and bends upward above about 220 minutes.
3. **The international plan** raises risk.
4. **The voice-mail plan** lowers risk.
5. **Evening use** raises risk slowly.
6. **Per-state offsets.**

These are in **every** seed.

Two usage interactions are **common**. Many service calls matter much more for light daytime users,
in 9 seeds. Day and evening use act together as one combined bill, also in 9 seeds. A third,
"heavy daytime use only matters without voice mail", appears in 4 seeds. What differs between
the seeds is **how** each signal is built:

- by one hinge, by a gate on another input, or spread over 2 to 8 units in 1 to 4 parts;
- whether a part reads day minutes, day charge or both;
- how the bend in service calls is assembled from several pieces that are each nearly straight.

The **state offsets** are in every network but are **noise**:

- the data do not support them (adding states makes the cross-validated ridge fit worse);
- keeping them never helps the TEST AUC with confidence (0 of 10 seeds), and in s8 removing them raises it (CI below 0);
- they are the largest single source of between-seed disagreement (22% of it).

The biggest surprise is a signal that **none** of the ten networks has. Customers with an
international plan churn almost certainly when they make 2 or fewer international calls (69 of 69
in train) or use more than 13 international minutes (66 of 66). Elsewhere plan holders churn at
about 25%. This is the strongest interaction in the data. All ten networks treat the plan as a
flat offset. A ridge logistic fit on the same signal list (with this interaction) reaches TEST
AUC 0.922, against 0.873–0.904 for the networks and 0.899 for their ensemble.

So Mike's working view mostly holds. The networks are close to one **signal-equivalence class**:
the same core signals, realised differently, with seed-specific state adjustments. Two refinements:

- The class is not quite one Σ. The usage interactions are partly interchangeable: seed 5 has no combined-bill or no-voice-mail term but the strongest calls × light-use gate, and seed 9 has no calls × light-use gate at all.
- The class shares a blind spot (international plan × international usage).

Most between-seed disagreement is **realisation** (different shapes for the same usage and service-call signals), not state noise:

- If every network's realisation of every signal is replaced by the ten-seed average, the median TEST logit correlation goes from 0.890 to 0.990. Doing this for every signal except state gives 0.962.
- Replacing the state offsets alone gives 0.912.

## 1. The signal list Σ

Signals are blocks of one fixed data basis (`basis.py`):

- **linear**: one per data variable, with aliases merged;
- **shape**: a bend in one variable;
- **interaction**: a two-variable term.

The interaction pairs were taken from the inputs that the ten explanations' units read together.
The international-plan × international-usage pair was added from a train-only look at churn rates.
The **data's own version** of each signal comes from a ridge logistic fit of churn on the whole
basis (C = 0.1 by 5-fold CV on train; CV AUC 0.909; TEST AUC 0.922).

The columns in the table are:

- **Relevance**: the cross-validated log-loss gain when the block is added (train CV);
- **Used**: the number of seeds whose own component has SD ≥ 0.10 logit;
- **Gain**: the median over using seeds of AUC(full) − AUC(score with that component removed), on train and on TEST;
- **TEST CI>0**: the number of seeds whose TEST gain has a 95% bootstrap CI above 0 (10,000 resamples, seed 42).

Full table: `signals.csv`.

| class | signal | in data terms | relevance (CV) | used | decodable | gain train / TEST | TEST CI>0 |
|---|---|---|---|---|---|---|---|
| **core** | service calls, slope (`lin:csc`) | risk rises with calls | +0.027 | 10 | 10 | 0.025 / 0.034 | 10 |
| **core** | service-call threshold (`shape:csc`) | flat 0–3 calls, steep from 4 (train churn 10.6% at 0–3, 44% at 4, 57% at 5) | +0.022 | 10 | 10 | 0.010 / 0.009 | 3 |
| **core** | day usage, slope (`lin:day`) | minutes ≡ charge | +0.016 | 10 | 10 | 0.010 / 0.007 | 3 |
| **core** | heavy-day hinge (`shape:day`) | bends up above about 220–250 min (churn 18% at 225–250, 38% at 250–275, 58% at 275–300) | +0.017 | 10 | 8 | 0.014 / 0.018 | 7 |
| **core** | international plan (`lin:intlplan`) | offset (42% vs 11% churn) | +0.040 | 10 | 10 | 0.064 / 0.075 | 10 |
| **core** | voice mail (`lin:vmail`) | plan lowers, messages give a little back | +0.010 | 10 | 10 | 0.005 / 0.006 | 2 |
| **core** | evening usage, slope (`lin:eve`) | minutes ≡ charge | +0.012 | 10 | 10 | 0.003 / 0.009 | 8 |
| **core** | night usage, slope (`lin:night`) | small | −0.003 | 10 | 10 | 0.001 / 0.004 | 5 |
| **core, noise** | **state offsets** (`lin:state`) | 51 per-state offsets | **−0.001** (CV says drop) | 10 | 10 | 0.009 / **−0.005** | 0 (1 CI < 0) |
| **common** | **calls × light day use** (`int:csc*day`) | with 4+ calls: light users 90–100% churn, 200–225 min 5% | +0.021 | 9 | 1 (data's shape) | 0.008 / 0.010 | **9** |
| **common** | **day + evening bill** (`int:day*eve`) | top 1/12 of the total bill churn 74% | +0.024 | 9 | 6 | 0.008 / 0.008 | 3 |
| **common** | **heavy day use without voice mail** (`int:day*vmail`) | 275–300 min: 76% churn without plan, 4% with | +0.020 | 4 (s0, s1, s6, s7) | 3 | 0.006 / 0.006 | 3 |
| common | international minutes, slope | | +0.010 | 9 | 10 | 0.002 / 0.002 | 0 |
| common | account length, slope | | +0.001 | 5 | 10 | 0.001 / −0.002 | 0 |
| common | international calls, slope | | +0.001 | 4 | 10 | 0.001 / 0.001 | 0 |
| common, noise | evening bend (`shape:eve`) | | −0.006 | 7 | 1 | −0.001 / 0.002 | 0 |
| common, noise | day calls, slope | | +0.002 | 6 | 10 | 0.001 / −0.003 | 0 (3 CI < 0) |
| rare | international-minutes bend; day × night; evening × calls; calls × voice mail | | ≤ +0.008 | 1–2 | 0 | ≤ 0.001 | 0 |
| **absent from all ten** | **international plan × intl usage** (`int:intlplan*intl`) | plan holders with ≤ 2 intl calls (69/69) or > 13 intl minutes (66/66) churn | **+0.047 (the largest)** | **0** | 0 | data's own: TEST +0.015 [+0.003, +0.026] | – |
| absent | international-calls bend | 0–2 calls (driven by plan holders) | +0.003 | 0 | 0 | | |
| absent | area code, the other call counts and bends, calls × international plan | | ≤ +0.001 | 0 | | | |

**Notes.**

- "decodable" uses the *data's* version of the signal against each network's inputs plus all hidden
  activations.
- For calls × light day use the data's version is much steeper than any network's (figure
  `signal_shapes.png`), so only seed 5's realisation is close enough (novelty 0.37).
- By the species-leads method (each network's own realisation matched against the other seeds)
  the same signal is found in a median of 4 of the 9 other seeds.
- The service-call threshold and the heavy-day hinge are found in 7–9 of 9 by that method.

## 2. Aliases in the data (`aliases.csv`)

| pair | relation | after z-scoring | how the networks read them |
|---|---|---|---|
| total day / eve / night / intl **charge** vs **minutes** | exact multiples (×0.170, ×0.085, ×0.045, ×0.270, rounded to cents) | identical up to 5e-4, 1.1e-3, 2.3e-3, 6.2e-3 (train and test) | day: **both** columns in 8 seeds, minutes only in s2 and s3. Units mix them freely (e.g. s5's gate reads day charge, its hinge day minutes) |
| voice-mail **plan** vs **messages** | near alias: messages > 0 only for plan holders (0 exceptions); 1 plan holder has 0 messages; corr 0.954 | not identical | every network reads both. All give the plan a negative offset and the messages a small positive slope, which gives back part of it |

A realisation reading charge where another reads minutes is the same signal. All analyses below
move each alias pair together.

## 3. Realisations ρ(s) per seed

Every hidden node of every network (214 in all, with the h-labels of the plain-language figures)
is mapped to a signal in `units.csv`:

- 178 are assigned by projection (unit knock-out effect onto the signal blocks);
- 13 are state-gated units open for at most 100 customers, counted as state offsets;
- 9 are dead and 12 inert (fixed effect below 1e-6);
- 2 are unassigned / residual: s3's `8417` (an always-on state:6 unit, R² 0.44) and `16043` (a night-calls ramp feeding the state:32 gate).

`realisations.csv` lists one row per seed × signal × unit. It includes the unit's share of its own
effect, its role (primary or secondary), the inputs it reads, and its linearisation gain with a TEST CI.

### The matrix (`matrix.csv`, `figures/matrix.png`)

Each cell gives the SD of the network's own component (logit, train). For the non-linear signals
it also gives the realisation:

- H = single hinge;
- G = single gate, i.e. one unit that also reads another input;
- Sk = split across k units;
- /np = spread over n explanation parts.

A dash means not used, w means weak (SD 0.03–0.10).

| seed | calls slope | calls threshold | calls × light day | day slope | day hinge | day+eve bill | day w/o vmail | eve | intl plan | vmail | state | intl plan × intl use |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0 | 0.73 | 0.53 S4/3p | 0.33 S3/2p | 0.53 | 0.86 S4/3p | 0.40 S2/1p | 0.24 H/1p | 0.32 | 0.82 | 0.50 | 0.51 | – |
| s1 | 0.72 | 0.57 S2/1p | 0.33 H/1p | 0.66 | 0.99 S4/2p | 0.40 S2/1p | 0.38 H/1p | 0.45 | 0.85 | 0.45 | 0.51 | – |
| s2 | 0.85 | 0.60 S8/4p | 0.41 S6/3p | 0.56 | 0.80 S4/2p | 0.38 S4/2p | – | 0.37 | 0.79 | 0.74 | 0.60 | – |
| s3 | 0.74 | 0.40 G/1p | 0.29 S3/2p | 0.68 | 0.74 S3/1p | 0.40 S2/1p | – | 0.29 | 0.78 | 0.55 | 0.53 | – |
| s4 | 0.69 | 0.43 G/1p | 0.10 G/1p | 0.87 | 0.64 G/1p | 0.44 S2/1p | – | 0.38 | 0.77 | 0.58 | 0.63 | – |
| s5 | 0.81 | 0.56 S4/1p | **0.69** S5/1p | 0.68 | 0.71 S4/2p | – | – | 0.48 | 0.78 | 0.60 | 0.47 | – |
| s6 | 0.88 | 0.50 S4/1p | 0.47 S5/2p | 0.55 | 0.86 S2/2p | 0.39 G/1p | 0.49 G/1p | 0.33 | 0.92 | 0.52 | 0.69 | – |
| s7 | 0.74 | 0.52 S2/2p | 0.25 S3/2p | 0.84 | 0.84 S5/3p | 0.50 S5/3p | 0.63 S5/3p | 0.25 | 0.88 | 0.41 | 0.55 | – |
| s8 | 0.68 | 0.53 S4/2p | 0.37 S4/2p | 0.51 | 0.64 S2/2p | 0.27 G/1p | – | 0.32 | 0.70 | 0.57 | 0.59 | – |
| s9 | 0.61 | 0.48 H/1p | **–** | 0.67 | 0.58 S5/2p | 0.30 S4/1p | – | 0.29 | 0.76 | 0.47 | 0.54 | – |

How each part of the class is realised:

- **Linear signals.** The international plan is a direct edge, or linear on the data through
  one-step units, in 9 seeds. In s3 it goes only through a switching gate that is also keyed to
  state:6. Voice mail is a direct edge in all ten. The day slope goes only through switching units
  in 6 seeds; it has no straight edge at all.
- **Shapes.** `figures/signal_shapes.png` overlays the ten realised shapes in real units, with the
  data's own as a black dashed line:
  - the **service-call** curves agree closely (corr with the consensus 0.99);
  - the **day hinge** agrees (0.98), with a 2-fold spread in steepness above 300 minutes;
  - **calls × light use** has one shape in nine seeds (0.97 with the consensus) but is much weaker than the data's (corr 0.77);
  - every network's **international plan** is a flat offset of about +2.6 logit, while the data's rises steeply above 13 minutes.
- **Strength on TEST.** Removing the calls × light-use component lowers TEST AUC with a CI above 0
  in 9 of 9 seeds that have it. The heavy-day hinge does so in 7 of 10. The state offsets: in 0 of
  10, and s8's TEST AUC goes *up* (CI below 0).

## 4. Distributed vs consolidated: how a shape is assembled

This answers Mike's second question. For each input, the score's response is split into the
explanation parts whose exits feed the output; these sum exactly to the logit (checked to 1e-9).
The response is also split into the units that feed the score, with identity nodes expanded. The
figures `figures/parts_<variable>.png` and `figures/s5_units_response.png` show this
(`consolidation.csv`, `part_shares.csv`). The effective number of parts is 1/Σ share², where
share is each part's share of the variance of the response over the training customers.

| input | parts carrying it (median, range) | effective parts | units carrying it | one-variable feature: R² of the summed response | consolidation ΔAUC train / TEST (median) |
|---|---|---|---|---|---|
| service calls | 2 (1–3) | 1.45 (1.0–3.0) | 2.5 (1–4) | 0.82 (0.63–1.00) | −0.011 / −0.011 |
| day usage | 2 (1–3) | 1.82 (1.06–2.59) | 3 (1–4) | 0.72 (0.58–0.87) | −0.018 / −0.024 |
| evening usage | 1.5 (1–3) | 1.21 | 1.5 | 0.43 | −0.009 / −0.011 |
| international plan | 1 (all seeds) | 1.0 | 1 | 1.00 (exact in 8) | 0 / 0 |
| voice mail | 1 (1–4) | 1.04 (1.0–3.1) | 1 | 1.00 exact in 6; 0.15–0.75 in s0, s1, s6, s7 | 0 / 0 |

**Service calls** is assembled differently in almost every seed:

- **s1, s5, s7, s9**: in one part. Even so, s5 builds it from four units (h1, h11, h12, h13 carry 31/20/31/18%).
- **s2**: two parts.
- **s0, s3, s6**: three parts. In s3 these are a direct slope (39%), a threshold hinge (37%) and the day-minutes ramp (21%).
- **s4**: a direct linear slope plus a gate (`1255`) that cancels it for 0–2 calls. The two parts point in opposite directions: the variance of the sum is 0.19 of the summed part variances.

**Consolidation test.** The summed response was replaced by one function of calls alone, a lookup
per call count fitted on train. This is exact only where the response is additive. Everywhere
except s9 it **breaks the calls × day-use interaction**:

- R² 0.63–0.93;
- up to 8.6 logit change for single customers;
- TEST AUC −0.011 (median).

A consolidated feature of (calls, day use) reproduces 0.87–0.995 of it. It still costs TEST AUC
−0.002 to −0.009, because of the remaining state and evening terms. Swapping each seed's shape for
the ten-seed consensus shape costs about the same as its own best one-variable fit (median TEST
−0.012). So the seeds' shapes are equivalent, and what is lost is the interaction.

**Can ExplaNEAT express the consolidation?**

- **Grouping** the carrying parts into one composition annotation is always possible and function-preserving. But for service calls (9 seeds) and day use (10) that composition also carries day use, evening use, voice mail or states.
- **Isolating** a pure service-call feature needs a rewrite that is not function-preserving, because the ReLU units read calls and day charge together. No seed is structurally separable in calls.
- **International plan** (6 seeds) and **voice mail** (6 seeds) enter additively. Identity nodes or splits can isolate them, function-preserving and exactly on the data.
- In s1 and s3 the international plan passes through a state-keyed gate. In s0, s1, s6 and s7 voice mail is inside a day-use gate.

## 5. Seed 5 worked example (`figures/network_churn100-s5.png`, `s5_units_response.png`)

**The tangle.** churn100-s5 has 21 hidden nodes, h1..h21: 16 original units and 5 identity nodes
from the explanation. The state one-hots, day charge and day minutes, service calls, plans and
usage all wire into a few of them. The explanation's five parts are:

- service calls × day charge (h1, h3, h4, h6, h8, h11, h12, h13 → exit h17);
- daytime minutes (h9 → h18);
- account length (h15 → h19);
- state offsets (h2, h5, h7, h10, h14, h16 → h20);
- plans and usage, linear (h21).

The h-labels map to node ids as follows (`units.csv`, seed 5):

| h | node | kind | realises (signal) | note |
|---|---|---|---|---|
| h1 | 104 | switching | calls × light day use (+ day hinge, calls slope) | the main gate: opens with calls, closes with day charge (also NH/CT indicators) |
| h3 | 1290 | switching | service-call threshold | calls hinge from about 1 call, feeding h1 |
| h4 | 2039 | switching | state (Florida-gated) | opens on 3 training customers |
| h6 | 2963 | switching | calls × light day use | day-charge ramp shaping h1; train gain 0.014, TEST 0.018 [0.009, 0.029] |
| h8 | 4834 | switching | day hinge (inside the gate) | day-charge ramp feeding h1; TEST 0.013 [0.006, 0.021] |
| h11 | 7506 | switching | service-call threshold | second calls threshold, switched off by high day charge; TEST 0.013 [0.005, 0.024] |
| h12 | 7664 | switching | calls × light day use | follower of h1 (relu(0.98·h1 − 0.62)): steepens the gate |
| h13 | 9029 | always on | calls slope/threshold | affine follower of h11 |
| h17 | id_cs | identity | exit of the service-call part | |
| h9 | 6238 | switching | heavy-day hinge | above about 210 min (also a NY indicator) |
| h18 | id_day | identity | day slope + hinge | |
| h15 | 11575 | switching | account-length bend | inert on AUC (gain 0.000) |
| h19 | id_acct | identity | account length | |
| h2, h14 | 1084, 11236 | always on | state offsets | |
| h7, h16 | 4253, 11955 | binary one-step | state offsets (MD, …) | |
| h5, h10 | 2730, 6958 | dead | – | Colorado, Indiana |
| h20 | id_state | identity | state offsets | 25 state inputs |
| h21 | id_plan_usage | identity | international plan, voice mail, evening, intl, night, call counts (all linear) | |

**Seed 5's signal set.**

| status | signals |
|---|---|
| has | calls slope + threshold, calls × light day use (the strongest of all ten, SD 0.69), day slope + hinge, international plan, voice mail, evening slope, night / intl / intl-calls / day-calls slopes, account length, state offsets |
| lacks | day + evening bill, day without voice mail, international plan × intl use |

**How the service-call shape is assembled.** The whole response sits in one part, but it is
summed from four units feeding the score:

- h12 (31%) and h1 (31%), the gate and its follower;
- h11 (20%) and h13 (18%), the second threshold and its affine follower.

Each is close to a ramp from 3–4 calls. Together they make the steep curve: +0.7 logit at 3 calls,
+3.7 at 5, +12.6 at 9 (mean over training customers). Every one of them is also keyed to day
charge. That is why a one-variable consolidation fits only R² 0.63 here, the lowest of the ten,
while (calls, day use) fits 0.99.

So s5 is the case Mike described: "not explainable without work". 16 units are wired through
5 identity exits, and a third of the inputs reach the score only through gates. But it reduces to
10 signals, each with a clear realisation, and to one compact shape for calls × day.

## 6. How much of the between-seed disagreement is realisation vs state noise

On TEST, the median pairwise logit correlation is 0.890 and the median overlap of the 141 riskiest
customers is 79%. The median AUC is 0.886 and the ensemble AUC 0.899. These match the measurements
quoted in the brief.

`disagreement.csv` decomposes each pair's logit difference into signal blocks plus the residual.
A block's share is Cov(block difference, difference) / Var(difference), so the shares sum to 1;
the table gives the mean over the 45 pairs on TEST (train is similar). `harmonise.csv` replaces
one signal family's realisation in every network by the ten-seed average.

| source | share of disagreement (TEST) | agreement after harmonising it (median corr / top-141 overlap) |
|---|---|---|
| (none) | – | 0.890 / 79% |
| state offsets | **22%** | 0.912 / 83% |
| day slope + hinge | 15% | 0.904 / 82% |
| day w/o voice mail | 13% | 0.900 / 80% |
| calls × light day use | 12% | 0.904 / 84% |
| residual (not in Σ: 3-way gates, state × usage) | 11% | – |
| day + evening bill | 8% | 0.904 / 81% |
| night / intl usage | 6% | 0.897 / 82% |
| service calls slope + threshold | 3% | 0.891 / 81% |
| voice mail, evening, account / area / calls, minor interactions, intl plan | ≤ 3% each | ≤ 0.894 |
| heavy-usage complex (day, day+eve, day w/o vmail, eve) | 38% combined | 0.935 / 83% |
| both complexes (usage + service calls) | 53% combined | 0.947 / 85% |
| all signals except state | | 0.962 / 90% |
| all signals | | **0.990 / 94%** |
| state inputs held at average | | 0.909 / 80% (median TEST AUC 0.886 → 0.894) |

What this table shows:

- About **half** of the disagreement is different realisations of the heavy-usage and service-call
  signals.
- The usage interactions substitute for one another. Harmonising the four usage blocks *together*
  gains more agreement than any one of them alone.
- State offsets are the biggest single item, at 22%. Removing them raises TEST AUC, which is the
  overfitting reading. But they do not explain most of the disagreement.

## Method (`catalogue.py`, `basis.py`, `common.py`, `fastboot.py`, `figures.py`)

1. **Basis Φ** (201 columns, all from train data). The blocks are:
   - linear: per data variable, with aliases averaged;
   - shape: hinges at train quantiles (integer knots for counts), orthogonalised on all linear columns;
   - interaction: tensor products of coarse hinges for 8 pairs, orthogonalised on all main effects.

   Each network's logit is projected on Φ (least squares, train). R² is 0.985–0.998 (TEST
   0.978–0.998), except s7 at 0.943 (TEST 0.930); s7 has 3–4-input usage units. The projection keeps the AUC (±0.008).
2. **Data signals.** A ridge logistic fit of the labels on Φ, with C chosen by 5-fold CV on train.
   Relevance is the CV log-loss change when the block is dropped.
3. **Realisations.**
   - The functional realisation is the network's own component Φ_s β_s.
   - Used means SD ≥ 0.10 logit.
   - Strength is the AUC drop when the component is removed (train; TEST with a paired bootstrap CI).
   - Decodable means the novelty of the data's component on the network's inputs plus all hidden activations (`genome_surgery.signal_basis`) is < 0.5. Non-linear blocks are already orthogonal to the inputs, so this is the species-leads residual match.
   - The cross-seed residual match of each seed's own realisation is in `found_in_other_seeds`.
4. **Units.** Each hidden node is fixed at its training mean (knock-out), and the effect is
   projected on Φ:
   - a switching unit's primary signal is its largest non-linear block when that block carries ≥ 25% of its top block's SD, otherwise its largest block;
   - a unit carries a signal when that block's SD ≥ 0.05 and ≥ 20% of the top carrier's;
   - the realisation type is read from the carriers (one-variable hinge, gate reading other inputs, chain, split) and their parts;
   - per-unit linearisation gains with TEST CIs follow the heart100 audit (`experiments/heart100_audit/audit.py`, Net class reused).
5. **Distributed vs consolidated.**
   - Partial dependence of each leaf part's output-feeding contribution (exact sum to the logit), and of each output-feeding unit with identity nodes expanded.
   - Effective parts = 1/Σ share².
   - A one-variable lookup or piecewise-linear fit is made to the summed response, plus a (calls, day) fit and the cross-seed consensus. The result is labelled exact (max |Δ| < 1e-9), approximate (R² ≥ 0.9) or breaks interactions.
   - A variable is called structurally separable when every switching unit downstream of it reads only that variable.
6. **Disagreement.** Covariance shares of pairwise logit differences, and harmonisation to the
   consensus coefficient vector.

## Caveats

- **The basis is a choice.** The knots, the eight interaction pairs and the 0.10 / 0.05 / 20% cut-offs were all set by me during the analysis:
  - the pairs came from the explanations, and the international-plan pair from a train-only look at churn rates;
  - "present / absent" depends on the 0.10 SD cut;
  - three-way gates (s7) and state × usage gates fall in the residual (11% of disagreement).
- **Interactions overlap.** The blocks are orthogonalised hierarchically, but the interaction blocks
  are correlated with each other (they all contain day hinges). Attribution among day × eve,
  day × voice mail and calls × day is therefore somewhat arbitrary. That is one reason the
  complexes are also reported jointly.
- **The data's own signals are a ridge fit**, and in sparse regions its curves extrapolate wildly:
  above 300 day minutes, above 15 international minutes. Its TEST AUC of 0.922 is a single labelled
  number. No network was trained with the international-plan interaction, so its value to a network is untested.
  A graft or retrain on train/validation would test it.
- **Decodability uses the data's version of the signal.** When a network uses a weaker shape (calls × light day use) the novelty is high even though the signal is clearly used. The model-to-model match is in `found_in_other_seeds`.
- **Knock-out attribution.** It fixes one node at its mean, so for chains it counts a unit's effect through every downstream unit. Shares are of each unit's own effect, not additive across units.
- **Unverified.** These are Claude's analyses of v4 explanations, which Mike has not verified. The unit-to-signal mapping is automatic; the s5 table was checked against the v4 hypotheses.

## Files

| file | content |
|---|---|
| `signals.csv` | one row per signal: definition, data relevance, class, use, decodability, strength, TEST checks |
| `units.csv` | every hidden node of every network (h-label, node id, kind, part, primary signal, carried signals, linearisation gains) |
| `realisations.csv` | one row per seed × signal × unit (plus every unit's own row) |
| `matrix.csv` | one row per seed × signal: SD, ablation gain train/TEST CI, decodability, cross-seed match, realisation type, layout, carriers |
| `consolidation.csv`, `part_shares.csv` | distributed-vs-consolidated analysis per seed × input |
| `disagreement.csv`, `harmonise.csv` | pairwise disagreement shares; agreement after harmonising |
| `aliases.csv` | the data aliases |
| `figures/` | `network_<seed>.png` (all ten, coloured by signal), `signal_shapes.png`, `parts_*.png`, `s5_units_response.png`, `matrix.png`, `disagreement.png` |
| `cache/` | betas, curves, graphs, meta (projection R², GAM CV) |

To reproduce (read-only, about 10 minutes):

```
DATABASE_URL=postgresql://localhost/explaneat_v4 EXPLANEAT_RECORDS_DIR=experiments/10seed/explanations_v4 \
OMP_NUM_THREADS=1 uv run python experiments/churn_signals/catalogue.py && uv run python experiments/churn_signals/figures.py
```
