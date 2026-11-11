# MNIST-10 linear-start series

**Interim report, 12 November 2026: eight of nine runs complete.** `ls-struct-s2`
is still running unchanged; it is excluded from completed-run averages and model
comparisons below. `logs/linear_start/summary.json` records the exact snapshot.

Starting from a dense linear readout closes most of the earlier sparse-network
accuracy gap. LR warm starts reproduce the linear training trajectory. The two
completed structural runs do not improve validation accuracy over LR, but this
**does not establish a limit of structural evolution**: their species controller
failed to prevent fragmentation, and reproduction stopped after generations 6–7.
Most of the nominal 100-generation growth experiment is continued training of
fixed elite structures.

## Design and selection

This is an exploratory continuation of [LINEAR_CHECK.md](LINEAR_CHECK.md), not a
pre-registered comparison. All runs use `data.prepare("mnist10", 10000)`: 10,000
standard-training MNIST rows, 1,000 per digit, subset seed 12345; 784 pixel values
scaled by 255; the same full 10,000-row standard test set. Validation comprises
5,000 seeded rows from the other 50,000 standard-training rows, seed 12346, with
an explicit train/validation disjointness assertion.

The common training path uses identity outputs, hidden ReLUs, softmax cross
entropy, masked phantom connections, and full-batch Adadelta(lr=1.5), restarted
at each generation. The population target is 250, with two elites per species.
Weights and biases do not undergo numerical genetic mutation: mutation rates
and replacement rates are zero. Backpropagation changes their values.

| arm | seeds | initialization | generations × epochs | fitness / other changes |
|---|---|---|---|---|
| full | 0, 1, 2 | `full_direct`, N(0,1) weights/biases | 100 × 5 | negative train CE, original config |
| lr | 0, 1, 2 | every genome receives the same sklearn LR coefficients | 100 × 5 | negative train CE, original config |
| struct | 0, 1, 2 (last pending) | same LR warm start | 100 × 25 | negative train CE − 0.0001 × enabled non-backbone connections |

The structural arm also sets enabled mutation to zero, node-add probability to
0.8, disjoint-distance coefficient to 5, initial compatibility threshold to 0.5,
and targets 5–10 species by multiplying the threshold by 0.8 or 1.25. Add-node
chooses an enabled edge in proportion to absolute weight. Direct input→output
edges are exempt from the penalty; hidden edges are charged. Connection deletion
remains 0.3, connection addition 0.5 and node deletion 0.2.

**Selection actually performed:** reproduction and the saved champion use
training fitness, including the structural penalty. Validation is logged only
for each generation's train champion; it does not select genomes or early-stop
these runs. The final summary is the saved train-fitness champion. The promise
to compare configurations using train/validation should not be read as
within-run validation selection. No retrospective test-based selection is used
in this report. The hidden-unit case study below is the better-validation of
the two completed structural champions; it is provisional until seed 2 finishes.

Test accuracy/CE were logged every generation, and test values from the preceding
MNIST studies were already visible while this series was designed. The test set
therefore is not an untouched confirmatory evaluation. All test columns are
exploratory descriptions. Three evolution seeds share one training/validation
split; they are not three independent data experiments. LR warm starts are
identical deterministic starts, explaining their identical winners.

## Completed results

Accuracies are fractions; CE is mean cross entropy. Every saved champion is from
generation 99 (zero based). Counts refer to enabled connections and hidden nodes
in the saved genome. Raw per-run JSON/CSV files and the generated `table.csv`
retain unrounded values.

| run | train acc | train CE | validation acc | validation CE | test acc | hidden | connections |
|---|---:|---:|---:|---:|---:|---:|---:|
| full s0 | .8988 | .396469 | .8810 | .503363 | .8791 | 0 | 6,772 |
| full s1 | .8952 | .401485 | .8840 | .468998 | .8792 | 0 | 6,500 |
| full s2 | .8977 | .398910 | .8804 | .495692 | .8774 | 0 | 6,794 |
| lr s0 | .9698 | .122109 | .9110 | .324603 | .9087 | 0 | 7,840 |
| lr s1 | .9698 | .122109 | .9110 | .324603 | .9087 | 0 | 7,840 |
| lr s2 | .9698 | .122109 | .9110 | .324603 | .9087 | 0 | 7,840 |
| struct s0 | .9751 | .105804 | .9060 | .352147 | .9056 | 0 | 7,840 |
| struct s1 | .9754 | .105172 | .9076 | .351365 | .9058 | 1 | 7,842 |
| struct s2 | pending | pending | pending | pending | pending | pending | pending |

The full-start arm has mean validation accuracy .8818 and mean test accuracy
.878567. All three LR-start champions follow exactly the 500-epoch LR-start
linear control. They preserve all 7,840 backbone edges and acquire no hidden
units. Thus LR initialization supplies the improvement; no extra nonlinear
capacity is demonstrated in that arm.

### Linear controls

`controls.json` contains LR plus linear models at 500 and 2,500 epochs, resetting
Adadelta every five epochs. Random linear controls use PyTorch random initial
weights, whereas the evolutionary populations use NEAT/Python random draws;
they match budgets and distributions but are not paired identical starts.
`control_reset25.json` adds the exact structural-arm reset schedule from the
same cached LR coefficients. No model is selected in that control.

| fixed linear control | epochs | train CE | validation acc | validation CE | test acc |
|---|---:|---:|---:|---:|---:|
| sklearn LR, C=1 | fitted | .128227 | .9112 | .316517 | .9097 |
| LR start, reset every 5 | 500 | .122109 | .9110 | .324603 | .9087 |
| LR start, reset every 5 | 2,500 | .105805 | .9060 | .352144 | .9056 |
| LR start, reset every 25 | 2,500 | .105804 | .9060 | .352147 | .9056 |

The structural s0 champion is effectively the matched linear control: same
validation/test accuracies and train CE within 4.1e-7. Struct s1 improves train CE
by about .000632 and validation accuracy by .0016 over that matched control,
but remains .0036 below the original LR's validation accuracy. Train CE falls
while held-out CE rises during unregularized continuation; this is consistent
with overfitting, not a broken multi-output training path. The small two-seed
structural differences do not support a significance claim.

## Why the structural search stopped

`summary.json` audits `parents.csv.gz`: a newborn has a new key and
`is_elite_copy=0`; generation zero is excluded. The parent file also contains
generation 100, produced after the final evaluation, so its newborns must not
be confused with evaluated offspring. The structural freeze predates that edge
case by more than 90 generations.

| run | species from gen 2 onward | maximum / final population | last generation containing newborns | total newborns |
|---|---:|---:|---:|---:|
| struct s0 | 163 | 354 / 326 | 6 | 733 |
| struct s1 | 153 | 339 / 306 | 7 | 736 |
| struct s2, checkpoint at gen 50 | 165 | 358 / 330 at logged gen 54 | 6 | 740 through checkpoint |

The installed dependency is `neat-python==0.92`. Its `DefaultSpeciesSet.speciate`
first assigns a representative to **every surviving existing species**, without
checking the threshold. The threshold is used only to assign remaining genomes.
Consequently raising it cannot merge the already-created species. The low target
band is not an effective upper bound. The controller lowers the threshold from
.5 to .4 after the first generation, and fragmentation appears at generation 2.
The completed runs reach a threshold of approximately 1.257e9 without reducing
the species count.

`DefaultReproduction.reproduce` enforces at least two members per species because
`min_species_size` is at least `elitism=2`. This forces populations above the
nominal target. Eventually every species contains exactly two elites. Both are
copied and no allocation remains for offspring, so mutation and crossover stop.
Continuing backpropagation improves elite fitness, preventing normal stagnation
removal from rescuing the search. This mechanism is demonstrated by both the
species-size CSVs and the independently recorded parent events.

This is a failed controller/reproduction configuration, not a valid 100-generation
test of sustained nonlinear growth. The live seed is left unchanged to preserve
comparability. No corrected evolution series or pruning study has been launched.

A follow-up fix should explicitly consolidate/rebuild species when over the
upper band, and enforce a population/offspring budget with room for births.
Simply increasing the threshold faster cannot fix the preserved-representative
behavior. Before another expensive series, a tiny deterministic regression
should assert that an oversplit population can return below the species cap,
stay within the population budget, and still create offspring. Rebuilding also
changes species identity/stagnation history and must be specified and tested;
a silent mid-run change would invalidate the comparison. Such a fix and new
three-seed growth series are deferred to a separately defined next experiment.

`linear_start_species_probe.py` checks the mechanism without training: a six-genome
target fragments into six species, grows to 12 members and produces no new keys
on subsequent reproductions despite threshold 1e9. Starting a fresh species set
at that threshold yields one species, and its next reproduction returns six
members with four newborns. The probe's assertions passed; `species_probe.json`
records the result. This demonstrates a candidate consolidation mechanism only,
not a validated production controller or a result on MNIST.

## Mutation-damage diagnostic

`mutation_damage.json` records 30 untrained mutations of the LR backbone for
each of three configurations, using random seed 0:

| mutation configuration | mean train CE | minimum–maximum CE | mean disabled edges |
|---|---:|---:|---:|
| original enabled mutation .1, connection deletion .3 | .174955 | .156757–.201848 | 387.33 |
| enabled mutation 0 | .128278 | .128227–.129650 | .267 |
| enabled mutation 0, connection deletion 0 | .128255 | .128227–.128803 | .367 |

The unmutated LR CE is .128227. The original enabled-gene mutation damages
hundreds of backbone edges in one reproduction step; removing it largely
preserves the fit. Add-node also disables the edge it splits, explaining why
some disabled edges remain even with connection deletion disabled. Deletion
removes genes rather than counting them as disabled. This diagnostic is small,
untrained, and not paired mutation by mutation across configurations: changing
mutation settings changes RNG consumption. It supports a mechanism for mutation
fragility, not a quantitative estimate of post-training evolutionary success.

## The surviving nonlinear unit: pixels and digits

The provisional best-validation completed structural champion, `ls-struct-s1`,
has one reachable hidden unit, node 292. On the training rows it switches on for
49.92% of images; it is neither dead nor always active. With coefficients rounded
to three decimals, its computation is approximately:

`h = ReLU(1.432 × pixel[r10,c14] + 1.073 × pixel[r10,c11] − .963)`

It adds approximately `−1.763 × h` to the digit-4 logit, with no other outgoing
edge. Pixel coordinates are zero based in the original 28×28 image and input
intensities lie in [0,1]. This is evidence of a two-pixel gate suppressing the
4 score, not evidence of a recognized stroke or semantic digit detector. Its
per-digit training activity is stored in `hidden_ls-struct-s1.json`.

The NumPy per-node forward pass agrees with `NeuralNeat` to maximum absolute
logit error 3.6e-14 on the checked first 2,000 training rows. Mean ablation fixes
the unit at its training mean output (.264453), preserving its average logit
contribution; removal deletes it and its connections. Neither variant is
retrained, so these measure dependence of the fitted network on this unit.
Removal leaves 7,839 direct backbone edges; it does not restore the edge that
was originally split to create the hidden path.

| split | full acc / CE | mean-ablated acc / CE | removed acc / CE |
|---|---|---|---|
| train | .9754 / .105172 | .9741 / .108064 | .9738 / .109076 |
| validation | .9076 / .351365 | .9054 / .355796 | .9070 / .354569 |
| test | .9058 / .381187 | .9056 / .383431 | .9052 / .383882 |

On validation, mean ablation changes 25 predictions: the full model is correct
on 17 of them and the ablated model on six (net +11/5,000 for the full model).
Fifteen changed examples have true digit 4, and four have true digit 9. The
per-digit accuracy gains are about +1.04 percentage points for 4 and +.81 for 9.
On test it changes 42 predictions, with only a net two additional correct
predictions (+.02 percentage points); digit-4 accuracy decreases by about .41
points while digit-9 accuracy rises by .30 points. These are small, mixed
exploratory effects. All per-class scores and counts are saved; this unit does
not establish an improvement over LR or an interpretable high-level feature.

## Artifacts, verification, and finalization

- `logs/linear_start/ls-*.{csv,json}`: source trajectories and saved-champion scores.
- `logs/linear_start/runs.json`: reachable hidden-unit activity, species summaries,
  and forward-parity checks for each completed saved champion.
- `logs/linear_start/summary.json`, `table.csv`: reproducible consolidation,
  explicit pending status, source-log SHA-256 hashes, and parent-event audit.
- `controls.json`, `control_reset25.json`, `mutation_damage.json`: controls above.
- `hidden_ls-struct-s1.json`: pixel weights, per-digit activity and ablations.
- Saved configurations, genomes, final populations, checkpoints and parent files
  remain outside git in `~/.explaneat/mnist_runs/<name>/` (override with
  `EXPLANEAT_MNIST_RUNS`). These are needed to recompute genome analyses; the
  committed scalar logs are sufficient for the score tables.

The legacy weighted genome was serialized as `__main__.WeightedAddNodeGenome`.
The analysis-only `RunUnpickler` maps that known class to the imported definition
when reading trusted local artifacts; training and saved runs are unchanged.
Two regression tests exercise that legacy load and the exact requested optimizer
reset schedule. The targeted analysis and existing multiclass-linear tests pass
(four tests). The full repository suite reports 1,131 passed, 92 failed, 32
skipped and one error in 241 seconds; `validation.json` names every failing/error
node. These failures are outside the edited MNIST analysis files and were not
repaired in this task. The complete repository suite is not clean.

The original core training-selection path stores `best_genome` by reference;
subsequent elite backpropagation can mutate that historical champion. The root
integration is addressing this separately with snapshots and an explicit best
generation. For all eight completed runs here, the saved fitness equals the
maximum logged generation-champion fitness (all at generation 99), so this
specific historical-best issue does not alter their reported final scores. The
live seed and this worktree's training code remain unchanged. Future snapshot
integration must update the reporter's identity-based best-generation detection
and checkpoint restoration together; otherwise generation attribution breaks.

Recompute the existing analyses (no evolutionary training):

```bash
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export UV_CACHE_DIR=/private/tmp/explaneat-codex-uv-cache
uv run --no-sync python experiments/mnist/linear_start_analyze.py runs
uv run --no-sync python experiments/mnist/linear_start_analyze.py hidden ls-struct-s1
uv run --no-sync python experiments/mnist/linear_start_analyze.py matched-control
uv run --no-sync python experiments/mnist/linear_start_summary.py
uv run --no-sync python experiments/mnist/linear_start_species_probe.py
uv run --no-sync pytest tests/test_experiments/test_linear_start_analysis.py tests/test_core/test_multiclass_linear.py
```

To finish after `ls-struct-s2.json` appears, rerun `runs` and `linear_start_summary.py`.
Check 100 consecutive CSV generations, a final JSON and the saved final/best/parent
files; all nine forward-parity errors should be below 1e-9. Add seed 2 to the
table and replace this interim status. Revisit the provisional structural case
study using validation only; if seed 2 has a better-validation hidden champion,
run its hidden analysis as well. Preserve s1 as an informative surviving-unit
case either way. Update the final parent audit and three-seed descriptive means,
then commit the final artifacts. Do not restart training or change the active
controller to complete this report.
