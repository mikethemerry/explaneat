# Churn case (Experiment 1.2): one explained PropNEAT model on telecom churn

Written 13 Nov 2026 on branch `exp/churn-case`. One training run, one explanation, one blind
verification. Everything new is in `explaneat_dev` (experiment `churn-case-s10`) and in this
directory; nothing already reported was changed.

## 1. Answer in brief

- **Run.** Seed 10, 200 generations, the churn configuration and split of the 10-seed churn
  sets. The best genome by training fitness (generation 197, NEAT key 31452, genome
  `9cacd431`) has 18 hidden units, 75 enabled phenotype connections and 42 of 71 inputs
  connected. Train AUC 0.9077, **test AUC 0.880 [0.844, 0.913]**, between L2 logistic
  regression (0.813) and gradient boosting (0.932) on the same split, and below the churn200
  ten-seed median (0.900; range 0.876–0.911).
- **Explanation.** Stored explanation `ce11259b` (95 operations), authoring skill v4, strict mode with
  provenance, 31 minutes: 10 leaves and 4 compositions, structural and compositional coverage 1.0,
  function preservation 1.4e-15, recheck 0 / 0. The model is a sigmoid of a bias plus a linear sum
  (international plan up, voice-mail plan down) and two nonlinear rules: a deep service-call gate
  (4 or more calls, suppressed for heavy daytime users, counted again inside a day-charge hinge) and
  a heavy-daytime-use hinge above about 220 day minutes; four small state flags; 29 inputs ignored.
- **Levels and verifier.** Faithful 1.4e-15; 10 leaves, granularity 1.0; understood 10 / 0 / 0 by the
  probe. Blind verifier: **weakened** (25 upheld, 9 weakened, 0 refuted, 2 domain flags): nothing
  refuted, four wording weakenings and a double-rounding digit; see section 6.
- **Plain-language draft** stored (lint clean, 5 worth-checking flags), unverified.

## 2. Design: matching the other Experiment 1 cases

The other Experiment 1 tabular cases are the AAAI paper genomes (Heart, Adult, MONK-2;
`results/thesis/heart_paper_genome.md`, `results/thesis/paper_recompute.md`). Their protocol,
read from the experiments' `config_json` and `hardware_info` in `explaneat_dev`
(`c8f3805e`, `dbab685c`, `45e998d7`) and from `experiments/10seed/PREREG.md`:

| Item | Paper runs (Heart / Adult / MONK-2) | Churn case |
|---|---|---|
| Runs | one unseeded UI run per dataset (`random_seed` is null on all three experiment rows; `runs_paper_reference.csv` records the seed as "paper") | one seeded run, `run_seed.py churn 10 --generations 200 --name churn-case-s10` |
| Seed | none recorded | **10**: the first integer after the ten study seeds 0–9. The paper runs recorded no seed, so no number can be copied from them; a seed in 0–9 would reproduce a churn200 genome bit for bit (the harness is deterministic, checked below), not an independent model. Fixed before the run; not changed afterwards. |
| NEAT config | Heart: template `bb576521`; Adult: the same template with conn add/delete 0.68/0.15, node add/delete 0.7/0.1, compatibility 5.0/2.0; MONK-2 defaults | `experiments/churn/configs/churn.ini`: the Adult paper config with `num_inputs` 107 → 71 (the only difference) |
| Population / epochs / fitness | 250 / 5 backprop epochs per generation / AUC on the training split | the same |
| Output activation, phantom mask | sigmoid; unmasked (the mask did not exist) | sigmoid (`output_activation` absent = default); unmasked (`mask_phantom` false), as the paper and 10-seed runs |
| Generations | Heart 40 (cut), Adult 100, MONK-2 100 | **200**, the churn200 budget: churn is harder (the pilot and churn100 left about 0.03 of headroom to gradient boosting, and churn200 gained 0.014 of median test AUC over churn100), so the longer of the two churn budgets was used, matched to `churn200` exactly |
| Selection | best genome by training fitness within the run; the test split is never used | the same rule, with the 10-seed tie-break (earliest generation, then lowest NEAT key); no tie occurred (1 row at the best fitness) |
| Split and encoding | the paper splits | the 10-seed churn split `45962ea6` (stratified 80:20, seed 42, 4,000 / 1,000; StandardScaler on train), dataset `churn_onehot` `802129c2` (state and area-code one-hots, 71 inputs; the codebook of `plain_language_codebooks.py`) |
| References | L2 logistic regression; gradient boosting (`results/thesis/baselines.json`) | the same two models refitted on this split (`references.py` → `references.json`; identical to `experiments/churn/baselines.json` to 4 dp) |
| Ancestry | not recorded | `track_ancestry` on: 49,750 of 50,000 genome rows carry `parent1_id` (all but generation 0) |
| Hardware | CPU | Apple M4 Pro, CPU, float64, 1 torch thread; 0.40 h wall clock |

**Harness check.** The training path has changed since the churn200 runs (phantom mask, output
activation, ancestry, criterion commits). Before the case run, `run_seed.py churn 0
--generations 3` (`churn-case-dryrun-s0`, kept in `explaneat_dev`) reproduced churn200-s0's
population statistics for generations 1 and 2 to full precision (best 0.6594633585162715 /
0.6894323240264164, mean and standard deviation equal to the last digit); the early
generations' genome rows of churn200 are not retained, so the comparison is at the population
level. Parent tracking does not touch the RNG.

**What is not matched.** The paper runs were not seeded, so "the same seed number" could not be
honoured literally; the 200-generation budget is twice Adult's and five times Heart's, by
design. Everything else (population, epochs, fitness, selection rule, config family, output
activation, mask, references) is the same.

## 3. Performance (test split, n = 1,000; CIs are 10,000 paired bootstrap resamples, seed 42)

| Model | Train AUC | Test AUC (95% CI) | Test accuracy (0.5) | Source |
|---|---|---|---|---|
| **Churn case, `churn-case-s10`** (seed 10, 200 generations) | 0.9077 | **0.880 [0.844, 0.913]** | 0.900 | `runs.csv` |
| L2 logistic regression | 0.843 | 0.813 [0.775, 0.848] | 0.862 | `references.json` |
| Gradient boosting (HistGB, defaults) | 1.000 | 0.932 [0.898, 0.962] | 0.961 | `references.json` |
| churn200, ten-seed median | 0.918 | 0.900 [0.895, 0.906] (range 0.876–0.911) | 0.911 | `experiments/10seed/summary.md` (reported, unchanged) |
| churn200, best seed by training fitness (s1) | 0.930 | 0.905 [0.868, 0.937] | — | `experiments/churn/runs_churn200.csv` |
| churn200, best test AUC (s0; shown for scale, never a selection rule) | 0.924 | 0.911 | — | `experiments/churn/runs_churn200.csv` |

The case model closes about 56% of the gap from logistic regression to gradient boosting
((0.880 − 0.813) / (0.932 − 0.813)); the churn200 median closes about 73%. Seed 10 lands at the
lower end of the churn200 range: an eleventh draw from the same distribution, reported as it
came.

Structure (`runs.csv`, the definitions of `experiments/10seed/compute_runs.py`): 18 hidden
units, 75 enabled connections in the phenotype, 35 direct input→output edges, 20 skip
connections, depth 8, mean hidden fan-in 1.67 (max 6); 42 inputs connected, 29 unused; genome
37 node genes and 161 connection genes (96 enabled).

## 4. Method for the explanation

- Authoring skill v4 (`.claude/skills/explanation-authoring/SKILL.md`), protocol v4, strict mode
  with provenance, function-preserving operations only, training split only, one explaining
  agent, 3-hour box. The agent's prompt is `EXPLAIN_TASK.md` (written before the best genome
  was inspected). It did not see `experiments/churn/`, the 10-seed records or the levels.
- Closing recheck (`xn.py --recheck`, recheck v3), then `verify_record.py --out-dir
  explanations` for the orchestrator's independent re-check (coverage, hierarchy, function
  preservation, closed form, recheck) → `explanations/churn-case-s10.json`.
- One blind verifier (`.claude/skills/explanation-verifier/SKILL.md`), read only, levels L1–L4,
  → `verdicts/churn-case-s10/`.
- The three levels: `experiments/explanation_levels/compute_levels.py --set churn_case` (a new
  set: `explaneat_dev`, records in this directory) → `levels/churn-case-s10.json`.
- Plain-language draft: `plain-language extract` / `draft` with the plain-language skill, into
  `plain_language/churn-case-s10.json` (an AI-assisted draft, unverified).

## 5. The explanation (`explanations/churn-case-s10.json`)

Explaining time 31 minutes of the 3-hour box (11:16–11:47 NZDT, 13 Nov). Operations: 71 input
renames, 6 node splits, 1 consolidation, 3 identity nodes, 14 annotations (10 leaves, 4
compositions including the root). Structural and compositional coverage 1.0, hierarchy
complete, all 18 hidden units in exactly one leaf without a split; function preservation
1.4e-15; the typed closed forms (3 s.f.) are off by at most 0.052 on a training row. 426
numbers in the stored text: 234 are references to 72 stored queries and 192 are allow-listed
literals (probe settings, codes, node ids); the closing recheck found 0 mismatches and 0
needs-review items. All 14 parts have a stored self-probe; none was refuted.

**What the model does** (every number below is from the stored, provenance-backed hypotheses;
training split, n = 4,000, churn rate 14.1%). The churn score is a sigmoid of a bias plus four
additive blocks, so the parts' effects multiply in odds:

| Part (hidden units) | What the model does | Ablation (train AUC 0.908 with everything on) |
|---|---|---|
| `directTerms` (0; 35 direct edges) | A linear sum led by an international plan (+0.79 per sd; the typical customer's risk goes from 2% to 26%; 380 plan holders, churn 42.4% vs 11.2%) and a voice-mail plan (−1.13, against +0.55 for the message count), then the direct day-charge edge (+0.67), evening minutes (+0.42), international charge (+0.40) and a few state codes | 0.786 with the direct terms fixed at their mean: the largest part |
| `serviceCallChain` (7, a 6-deep chain) | A gate on the service-call count whose threshold rises with daytime use: for the typical customer off at 3 calls, on at 4 (1.07), largest at 5 (1.50), easing by 9 calls (0.42); on from 3 calls at 100 day minutes, shut at 300 day minutes. Acts on 351 customers (churn 41.0% vs 11.6%), adding 3.19 to the score on average for them; its falling branch beyond 5 calls rests on 42 rows | 0.813 without it |
| `dayChargeBoost` (2) | A hinge on day charge from about 32 dollars (off at 30, 0.21 at 35, 2.23 at 60 dollars) that also carries the chain's output with a weight near one, so the complaint signal is counted a second time when it is on. Acts on 1,921 customers (churn 23.9% vs 5.1%), adding 1.11 on average | with the chain: 0.815 when both are off |
| `serviceCallSpike` (1) | A linear hinge from 4 service calls, independent of day use (0.072 at 4 calls, 1.21 at 9); keeps the score rising where the chain falls back. Acts on 318 customers (churn 50.0%) | 0.907 |
| `heavyDayUse` (2, read out twice) | A hinge from about 220 day minutes (0.025 at 220, 0.78 at 300, 1.25 at 350); acts on 964 customers (churn 29.3% vs 9.4%), adding 1.07 on average | with the ramp: 0.904 when both are off |
| `dayMinutesRamp` (2) | A negative near-linear ramp above about 130 day minutes, on for 4,000 of 4,000 rows, that offsets part of the direct day-charge term for moderate users (typical customer: 5% at 130 minutes to 1% at 210, day charge held) | 0.898 (replaced by its mean) |
| `twoStateFlag`, `state38Hinge`, `state18Hinge`, `state24Hinge` (1 each) | Flags for states 45 and 49 (−0.48 / −0.42 on average for their 91 / 122 customers), 38 (−1.02; 56 customers), 18 (+0.20; 65) and 24 (−0.33; 80): small regional adjustments fitted to few customers | 0.907–0.908 each |

Compositions: `serviceCallEffects` (chain, day-charge boost, spike: the typical customer's risk
is 2% at 3 service calls, 31% at 4, 68% at 5, then 28% at 9 as the chain's falling branch
outweighs the spike), `dayUseShape` (ramp, heavy-use hinge, with the direct day-charge edge:
2% at 100 day minutes, 2% at 180, 4% at 220, 17% at 250, 73% at 300, 97% at 350, day charge set
consistently; 919 customers above 220 minutes, churn 30.4% vs 9.3%), `stateAdjustments` (the
four flags) and the root `churnScore`. 29 inputs are unconnected: the remaining state codes,
area code 510, evening calls, night minutes and calls and international minutes.

In customer terms: the model raises the predicted chance of leaving for customers with an
international plan, for customers who have called customer service four or more times (three
times for light daytime users) unless they are heavy daytime users, and for heavy daytime
users (above about 220 day minutes); a voice-mail plan lowers it. The two risk gates are partly
exclusive: the service-call gate is suppressed for heavy daytime users, who already score high
on usage alone. This is the same pair of rules that the churn200 ten-seed summaries describe
(heavy daytime use without a voice-mail plan; many customer-service calls at low day charge),
found again on an independent seed.

Honest limits stated in the record (`resisted`): the chain's falling branch beyond 5 calls
rests on 42 training rows; the day-charge unit's two small state terms and the two-state
flag's hand-off from the heavy-use unit are described by their formulas without a reading of
their own; the typed closed forms are rounded to three significant figures. The explainer's
tool notes: the probe and subgroup tools take the dataset's raw column names, not the renamed
node names; a leaf's exit cannot serve as another leaf's entry, so the two serial hand-offs
needed identity nodes; and an exit counts as covered only when every outgoing edge sits in some
annotation, so the hand-off edges had to be listed in the root.

## 6. The three levels and the verifier

| Label | Faithful: max abs err | Leaves | Granularity (largest part) | Understood: meaningful / partial / closed-form only | Recheck | Verifier |
|---|---|---|---|---|---|---|
| churn-case-s10 | 1.4e-15 (closed form 0.052) | 10 (4 compositions) | 1.00 (`serviceCallChain`: 7 units, width 2, depth 6; no coarse-decomposition warning) | 10 / 0 / 0 | 0 mismatch, 0 needs review | **weakened**: 36 findings, 25 upheld, 9 weakened, 0 refuted, 2 domain flags |

Levels: `levels/churn-case-s10.json`, computed read-only by
`experiments/explanation_levels/compute_levels.py --set churn_case`. Probe scores: the direct
terms 0.88 (`international plan`), the chain 0.91 (`number customer service calls`), the
day-charge boost 0.89 (`total day minutes`), the ramp 1.00 (`total day minutes > 180.5`), the
heavy-use hinge 1.00 (`total day minutes`), the spike 1.00, the four state flags 1.00 (their
state). As on the Cox case, a single-input leaf's best concept is its input, so the four state
flags' grade is a description of a weight; the substantive "understood" readings are the chain,
the day-charge boost and the two day-minute units. For the ten churn200 explanations the
median is 8.5 leaves and 89% meaningful (`experiments/explanation_levels/tables.md`); this case
has 10 leaves, all meaningful by the probe.

### Verifier findings (`verdicts/churn-case-s10/verdict.json`, `verdict.md`, `recheck.json`)

Blind, read-only, one agent, levels L1–L4 all completed, 24 minutes. Its recheck: 0 mismatch,
0 needs review, 184 of 184 provenance claims reproduce. Every closed form, node value,
threshold, count, churn rate, contribution, AUC and concept score it re-ran on the training
split reproduces; nothing is refuted. **Overall: weakened**, on nine findings in four groups:

| Location | Level | Stated | What the model does | Check |
|---|---|---|---|---|
| `churnScore` hypothesis and evidence[0] | L3 / L2 | for the typical customer "the gated parts are off"; the score is the bias plus the direct terms (0.33) and the ramp (0.09) | the typical customer is in state 49, so the two-state flag is on: exit 2471 = 4.25, contribution −0.42; score −3.77 = bias + 0.33 + 0.09 − 0.42; without the flag the typical risk is 3.4%, not 2.3% | `verify probe` on the typical customer, flag ablation |
| `record.summary`, `dayUseShape` hypothesis | L3 | the response to daytime use is "flat to falling" for light and moderate users, then steeply rising | on a tariff-consistent sweep (0.17 dollars per minute) the risk rises from 0.74% at 25 minutes to 2.64% at 130, falls only over 140–190 minutes (to 2.18%), then 4.0% (220), 16.7% (250), 73.3% (300), 97.4% (350); the quoted 2 / 2 / 4 / 17 / 73 / 97% reproduce, the "flat" light-use range is rising | `verify probe --sweep` |
| `directTerms` hypothesis | L2 | "day charge appears here as well as in two hidden units" | the split copy `-61_b` feeds only unit 486; unit 1197 reads day minutes: one hidden unit, not two | operations dump |
| `serviceCallEffects` hypothesis, `record.customer_terms` | L3 | the gate opens at "3 calls for light daytime users" / "one call earlier" | on at 3 calls below about 140 day minutes, at 2 calls below about 75 (exit 1792 = 0.59 at 50 minutes), at 1 call below about 25; 21 training customers with 2 or fewer calls are inside the gate (0 of 21 churned), 90 with exactly 3 | `verify probe`, `verify subgroup` |
| `state24Hinge`, `stateAdjustments` hypotheses | L1 | "AUC 0.908 to 0.907" with the state-24 flag off | unrounded 0.9075469 = 0.908 at 3 dp; the query stored 0.9075 and the template printed 0.907 (double rounding); the change is 0.00017, one rank flip | `verify ablate` |

Flags (L4, domain only; the data agree with the model): the "complaint signal" chain also fires
for 111 customers with 3 or fewer calls (churn 7.2%; the 21 with 2 or fewer: 0 of 21), and its
output falls for 6 or more calls (1.50 → 0.42) although those 42 churn at 66.7% (the whole
model still gives them a mean risk of 72% through the boost and the spike); and the state-18
flag raises scores (+0.20) although that state's training churn (9.2%, 6 of 65) is below the
rest (14.2%). The hypotheses state both honestly.

Verifier tool notes: `verify linearise` at parts level fails on this model ("model state (with
overrides) is not a DAG"), so the root's linearised AUC was checked only against the stored
query (0.8338, n = 4,000) and a node-level run (0.8348); `--edge 1792->486` is refused because
the hand-off now runs through `identity_1792_486` (that edge works: 0.8896); `query_ablate`
stores AUCs at 4 dp, which is the source of the 0.907 / 0.908 digit; `verify recheck` gives no
`target_hash` for a record-JSON target, so the verdict records the record file's sha256.

The stored explanation was not edited after verification: the record stands as verified, and
the weakenings are listed here for the reader. All four substantive weakenings are wording that
understates or mis-scopes a correct number (the flag's −0.42 for the typical customer; "flat"
for a 0.7–2.6% rise; "two" for one unit; "3 calls" for a threshold that reaches 2 calls on 21
rows); none changes the model reading in section 5.


## 7. Plain-language draft (`plain_language/churn-case-s10.json`)

Extracted read-only from the stored explanation: 184 claims, 175 verified, 9 not supported
(a whole-model state claim of the wrong sign on 39% of customers, two interactions below 0.1
log-odds, three composition structure claims, one chain threshold whose folds spread, one
subgroup whose effect goes both ways, one "barely changes" flag claim). The claim generator
handled the 0/1 target without change. The draft (readdown v5, `drafted_by: ai-assisted`,
`verified_by: null`, lint clean after three rounds) has 17 summary sentences (a 2-sentence lead
and four blocks), 79 read-down sentences over the 14 parts, 59 detail sentences, 5 "worth
checking" flags and 14 proposed hypothesis revisions; every number is a claim slot filled by
code. Its lead: for a typical customer (West Virginia, no plans, one service call, 180 daytime
minutes) the predicted chance of leaving is about 2%; the model adds up a weighted sum of
inputs, a customer-service-calls group, a daytime-use group and a set of small state flags.

Worth-checking flags, for Mike's review: (1) inside the service-call chain, more daytime
minutes take an acted-on typical customer from about 98% to about 4%; (2) daytime use enters
the model four times (charge directly and through the boost; minutes through the ramp and the
heavy-use part), and charge is a fixed per-minute rate; (3) the state effects rest on groups
of 41–122 customers and the state names rest on one decoding check; (4) a voice-mail plan
lowers the typical customer's risk from about 2% to about 1%; (5) through the state
adjustments a customer outside the five flagged states comes out higher than one in West
Virginia. Two round-once renderings read oddly ("from about 3% to about 3%" for 2.75% → 3.34%;
"from about 2% to about 2%" for 2.26% → 1.63%) and may want rewording.

## 8. Caveats

- **One seed.** Seed 10 is a single draw; its test AUC (0.880) sits at the lower end of the
  churn200 range (0.876–0.911). The case shows one model explained end to end; the ten-seed
  sets carry the distributional claims. The seed was fixed before the run and nothing was
  re-run.
- **Seed matching is by rule, not by copying.** The paper runs were unseeded (no `random_seed`
  on their experiment rows), so the "same seed number" could not be used; the rule (the next
  integer after the study's seeds) is stated in section 2.
- **Budget.** 200 generations is the churn200 budget, longer than the paper's (40 / 100 / 100),
  chosen because churn is harder; it is the only deliberate departure from the paper protocol.
- **Code drift.** The run used the current training code (commit `959c5f1`), not the churn200
  commits; the 3-generation dry run reproduced churn200-s0's population statistics exactly,
  so the unmasked path is unchanged, but the comparison is at the population level because
  the early-generation genome rows of churn200 were not retained.
- **Not signed off.** The explanation is agent-written and agent-verified; `mike_verification`
  is absent. The plain-language draft is unverified.
- **Probe grades.** The "understood" level grades a single-input leaf by its own input (score
  1.0), which is a description of a weight, not a finding; four of the ten meaningful leaves
  are such state flags.
- **Collinear inputs.** Day charge and day minutes are the same column after z-scoring; the
  model reads "daytime use" through both names, which the explanation states and the draft
  flags.
- **Dry-run row.** `churn-case-dryrun-s0` (3 generations, seed 0) was left in `explaneat_dev`
  as the record of the harness check.

## 9. Reproduce

```bash
cd /Users/mike/dev/explaneat
PYTHONHASHSEED=0 DATABASE_URL=postgresql://localhost/explaneat_dev \
  uv run python experiments/10seed/run_seed.py churn 10 --generations 200 --name churn-case-s10
uv run python experiments/churn_case/compute_run.py            # runs.csv (read only)
uv run python experiments/churn_case/references.py             # references.json (read only)
# explanation: experiments/churn_case/EXPLAIN_TASK.md with the authoring skill; then
uv run python experiments/10seed/verify_record.py <report.json> --out-dir experiments/churn_case/explanations
uv run python experiments/explanation_levels/compute_levels.py --set churn_case --out experiments/churn_case/levels
EXPLANEAT_RECORDS_DIR=experiments/churn_case/explanations EXPLANEAT_LAYERS_DIR=experiments/churn_case/plain_language \
  uv run python -m explaneat plain-language extract --label churn-case-s10 --dir experiments/churn_case/plain_language
# verifier: the explanation-verifier skill on explanations/churn-case-s10.json -> verdicts/churn-case-s10/
uv run python scripts/thesis/thesis_experiment_tables.py       # thesis_exports/ch7/ch7_experiment1_cases.tex
```

