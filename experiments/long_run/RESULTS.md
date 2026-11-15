# Very long runs: are PropNEAT models explainable only because the runs stop early?

Exploratory, not pre-registered. Branch `exp/long-run`. Section 2 was written and committed before
any long run started; results are added below it as the checkpoints complete.

**Question (Mike).** Every result so far ran 40–300 generations. Perhaps PropNEAT models are
explainable only because evolution is stopped early. Over thousands of generations, do the networks
keep growing structure, overfit, accumulate marginal capacity and spurious terms, and become hard to
explain?

## 1. Answer in brief

**No: on these two datasets the models are not explainable only because of the generation budget.**

- **Every run stopped on its own long before 10,000 generations.** Each one stopped 1,000
  generations after its monitoring AUC peaked:

  | Run | Monitoring peak | Stop |
  |---|---|---|
  | shape-C seed 2 | 304 | 1,304 |
  | shape-C seed 0 | 510 | 1,510 |
  | churn seed 0 | 694 | 1,694 |
  | churn seed 1 | 779 | 1,779 |
  | shape-C seed 1 | 1,650 | 2,650 |

  So we observe each model 1,000 generations past its useful life, at 4 to 33 times the earlier
  budgets.
- **The phenotype grows slowly, then stops.** From generation 300 to the stop, shape-C goes from
  14–15 to 19–21 hidden units. Churn seed 0 goes from 18 to 26. Churn seed 1 is the exception: 33
  units at generation 300 (16 at 200), 49 at the stop.
- **The genotype grows much more**: 48 / 27 / 30 for shape-C and 82 / 187 for churn, almost all
  of it disconnected from the output. Only the phenotype is explained.
- **No overfitting collapse.** The best-by-train genome's training AUC creeps up by about 0.01, and
  its test AUC stays flat:
  - shape-C: 0.857–0.871, against 0.86–0.87 at generation 100–300;
  - churn: 0.876 → 0.890 (seed 0) and 0.904 → 0.912 (seed 1), from generation 200 to the stop.

  The monitoring AUC of the best-by-train genome sags from its peak by up to 0.024 (0.000 for shape-C seed 1), so selecting
  by training fitness slightly over-fits.
- **Signal kept, some spurious structure gained late (shape-C).** Seed 0 is flat:
  - U r 0.97 → 0.94, saddle r 0.93 → 0.93, spurious share 0.15 → 0.16.

  Seed 1 loses some of the U (U r 0.97 at 100 → 0.86 at the stop). Seed 2 holds its best shape
  until generation 1,000 (U r 0.98, spurious 0.10), then degrades in its last 300 generations: U r
  0.76, spurious share 0.27. The noise input never carries more than 1% of the logit's variance.
- **Marginal capacity does not accumulate on shape-C.** The marginal share stays at 0.2–0.5 with
  no trend. On churn it is high at every age (0.7–0.9): most churn units are small per-state or
  per-segment offsets. It rises to 0.94 on churn seed 1, the run that grew most.
- **Every later network was explained, as fast as the earlier ones.** Seven explanations at
  generations 200–2,650, frozen method:
  - all reached full coverage, exact function preservation (≤ 1.8e-15) and a clean recheck;
  - all finished in **17–28 minutes** of the 3-hour box;
  - granularity was 1.0 everywhere.

  The 49-unit churn network took 21 minutes, against 23 for the same run's 16-unit
  generation-200 network. It needed 23 splits instead of 1, and its largest leaf held 15 units
  against 7 at generation 200.
- **Verifier results do not get worse with age.** The table below shows refuted / weakened per 100
  statements, with the explainer's time:

  | Model | Early | Late |
  |---|---|---|
  | shape-C seed 1 (generation 300 → stop) | 0.0 / 2.8 (17 min) | 0.0 / 9.8 (27 min) |
  | churn seed 0 (generation 200 → stop) | 0.6 / 3.6 | 1.3 / 2.5 |
  | churn seed 1 (generation 200 → stop) | 1.2 / 3.6 | 0.0 / 5.1 |

  Every number in every record reproduced; what the verifiers challenged was wording and shape
  words, as in the earlier studies. The automated "understood" level moved both ways
  (probe 0.86 → 0.43 on shape-C seed 1, 0.56 → 0.92 on churn seed 1).
- **Ground truth was recovered at every point (shape-C).** The U and the saddle were named in all
  three shape-C explanations. What the longer run changed is **what there is to explain**:
  - **generation 300:** 7.5 of 9 truth terms recovered;
  - **final model of seed 1:** 6 of 9. It added a spurious sex × diabetes × age × BMI gate, so
    that diabetes lowers the predicted risk for older, lean men, a BMI dip and a larger noise
    effect. The explainer described all of it, and the verifier confirmed it.

**So, plainly:** running far longer does not make PropNEAT networks unexplainable. The monitoring
plateau ends the runs within about 1,000–2,600 generations. The networks plateau in size, and the
method explains a 49-unit late network as fully and as fast as a 15-unit early one. What the extra
generations cost is model quality, not explainability: a little test-set drift, a slowly growing
genotype and, in some runs, late spurious interactions. The explanation then faithfully reports
those interactions. The generation budget matters for which model you get, not for whether you
can explain it. **Caveats:**
- two datasets and five runs;
- the stopping rule means 10,000-generation models were never produced, which is itself the
  finding for this config;
- the "understood" screens are automated, with no sign-off.

## 2. Design (fixed before any run)

### 2.1 Datasets

- **Primary: the synthetic `both` dataset (shape-C, `experiments/ushape_saddle/`).** The backbone
  plus a U on sodium (minimum 137 mmol/L) and a saddle −1.5 zh·zp between heart rate and systolic
  pressure, with a `noise` column that no term uses. 1,500 train / 500 test rows; oracle test AUC
  0.884, L2 logistic regression 0.699, gradient boosting 0.854. The known truth lets us measure the
  signal kept and the spurious structure gained.
- **Secondary: churn** (the churn200 dataset, split and config: `experiments/10seed/run_seed.py`
  `DATASETS["churn"]`, `experiments/churn/configs/churn.ini`; 4,000 train / 1,000 test, 71
  inputs), a real dataset on which PropNEAT overfits easily.

### 2.2 Runs (`run_long.py`, `launch.sh`)

| Item | Setting |
|---|---|
| Runs | shape-C seeds 0, 1, 2 (`longrun-shape-C-s<seed>`); churn seeds 0, 1 (`longrun-churn-s<seed>`) |
| Config | unchanged: `experiments/ushape_saddle/configs/syn10.ini` and `experiments/churn/configs/churn.ini` (population 250, ReLU hidden units, sigmoid output, `full_direct` start) |
| Training | unchanged: the 10-seed harness (`seed_everything`, 1 torch thread, `PYTHONHASHSEED=0`, `DatabaseBackpropPopulation.run` with `auc_fitness`, 5 backprop epochs, unmasked). No new penalty |
| Monitoring split | 20% of the training rows, stratified, `random_state` 20261115, the same rows for every seed of a dataset (`monitor_split_<task>.json`). Evolution and backprop see the other 80% (shape-C 1,200 rows, churn 3,200); the fitness is the training AUC on those rows |
| Budget | **up to 10,000 generations** |
| Stopping rule | after each generation, the monitoring AUC of that generation's best-by-training-fitness genome (ties: lowest key) is computed. The reference is the best monitoring AUC so far, updated only by an improvement of at least **0.001**. The run stops when **1,000 generations** pass without such an improvement, or at 10,000 generations, or if free disk falls below 30 GB. The reason is stored (`logs/<name>.stop.json` and the experiment's `run_params.stop`) |
| Test split | never read during a run, and never used for stopping or selection; read only afterwards, for reporting |
| Processes | at most 3 training processes at a time; explanations after the shape-C runs finish, so that no more than 3 heavy processes run together |

These runs are not continuations of the earlier shape-C / churn200 runs: they train on 80% of the
same training rows, so their trajectories differ from the 100-, 200- and 300-generation runs from
generation 0.

### 2.3 Storage (light mode; `DatabaseBackpropPopulation(genome_storage="best")`)

Postgres `explaneat_dev`, new experiments only. 250 genomes × 10,000 generations would be 2.5 million
genome rows per run, so the runs use a light storage option added for this study (default behaviour
unchanged; `tests/test_db/test_population_light_storage.py` shows that the evolution is
bit-identical):

- every generation: its population row (summary statistics) and **its best genome** by training
  fitness;
- **full populations** at generations 100, 200, 300, 500, 1,000, 2,000, 5,000 and 10,000 (generation
  indices 99, 199, ...) and at the stop generation;
- **parents** of every genome in a sidecar CSV, `experiments/lineage/parents/<name>.csv.gz` (the
  replay-sidecar format read by `explaneat/analysis/lineage.py`; no parent ids in the database);
- a per-generation summary, `logs/<name>.gen.csv`: best and mean fitness, species, genotype size of
  the best and mean genome, monitoring AUC, the stopping reference and the generations since an
  improvement.

### 2.4 Checkpoint measurements (`score_checkpoints.py`, read-only)

Checkpoints: generations 100, 200, 500, 1,000, 2,000, 5,000, 10,000 (those reached) and the stop
generation. At each, the **best-by-train genome** is the stored genome of highest training fitness
in generations up to the checkpoint (ties: earliest generation, lowest key; the 10-seed rule). For it:

- training AUC (the 80% fit rows), monitoring AUC, test AUC;
- hidden units, enabled phenotype connections, depth (`experiments/10seed/compute_runs.py`'s
  `structure`);
- **shape-C only:**
  - U r and saddle r: `experiments/ushape_saddle/score_g300.py`'s scoring (logit over sodium
    128–150 mmol/L at the typical case vs the true U; mean-removed logit on a 41 × 41 heart-rate ×
    SBP grid vs −zh·zp);
  - **spurious-structure share:** 1 − R² of the model logit against the oracle (true) logit, on
    20,000 fresh rows drawn from the generating distribution (`data.inputs`, seed 20261116; the
    true function is known everywhere, so no data row is spent on this);
  - **noise-input variance:** on the same rows, the variance of the change in the model logit when
    `noise` is set to its training mean, as a share of the variance of the model logit;
- **marginal-capacity share:** the share of the phenotype's hidden nodes whose mean-ablation
  (`explaneat.verify.probe.Ablation(to="mean")`, as `experiments/lineage_jumps/signal_nodes.py`)
  lowers the training AUC (fit rows) by less than 0.01;
- **decomposition proxy:** the mechanical one-leaf-per-unit annotation is not cheap (it needs the
  splits and identity nodes of a real refactoring), so it is skipped at the checkpoints; the
  decomposition block (`coverage_report`, max part width and granularity) is reported for the
  explained genomes from their real explanations (`compute_levels.py`).

Curves (`make_figures.py`, PNG, log generation axis): training / monitoring / test AUC, size, U r and
saddle r, spurious share and marginal share, against generation. AUCs and size come from every
stored per-generation best; the other quantities from a log grid of generations (10, 20, 50, 100,
200, 300, 500, 1,000, 1,500, 2,000, 3,000, 5,000, 7,000, 10,000, and the stop).

### 2.5 Explanations at three points

For the shape-C seed with the **longest run** (most generations before stopping; ties: lowest seed),
the best-by-train genome at generation **300**, at generation **2,000** (if the run stops before
2,000: the generation midway on a log scale between 300 and the stop) and at the **final stop**.
If two points select the same stored genome, it is explained once. For churn seed 0, if time allows:
generation 200 and the final stop.

- **Method (frozen):** the authoring skill (`.claude/skills/explanation-authoring/SKILL.md`, v4),
  protocol v4 (`experiments/10seed/EXPLAIN_PROTOCOL.md`), strict mode with provenance,
  function-preserving operations only, a **3-hour box**; one explaining agent per genome, on the
  same model as the earlier studies; the task text is `EXPLAIN_TASK.md`, as in the U-shape study,
  with the network description changed. The agent is not told the generation or the study.
- **Recorded per explanation:** explaining time; leaves, coverage (structural / compositional) and
  function preservation; the closing recheck (`verify_record.py`); the three levels
  (`experiments/explanation_levels/compute_levels.py --set longrun`); one blind verifier
  (`VERIFY_TASK.md`; refuted and weakened statements per 100 checked); ground-truth recovery (is the
  U on sodium named, with both ends raising risk; is the heart-rate × pressure saddle named as an
  opposing-direction interaction), graded recovered / partial / not.
- **If a later network cannot be explained within the box, that is the result:** the record says
  where it got stuck.

### 2.6 What would answer the question

"Explainable only because of the budget" would show as: test or monitoring AUC falling while
training AUC rises; size growing without bound; signal r falling or the spurious and marginal shares
rising with generation; and the later explanations taking longer, covering less, decomposing more
coarsely, or failing verification more often than the generation-300 one. The opposite pattern
(plateaus in AUC and size, flat shares, comparable explanations) would say the generation budget is
not what makes these models explainable.

## 3. Why and when each run stopped

Every run stopped by the plateau rule (no improvement of 0.001 in monitoring AUC for 1,000
generations). None reached the 10,000 cap, and free disk never fell below 76 GB (`logs/<name>.stop.json`).

| Run | Generations | Monitoring AUC peak (generation) | Wall clock | Best genome at the stop: genotype / phenotype hidden units |
|---|---|---|---|---|
| shape-C seed 0 | 1,510 | 0.8929 (510) | 0.8 h | 48 / 21 |
| shape-C seed 1 | **2,650** (the longest) | 0.8988 (1,650) | 1.2 h | 27 / 19 |
| shape-C seed 2 | 1,304 | 0.9108 (304) | 0.5 h | 30 / 19 |
| churn seed 0 | 1,694 | 0.9154 (694) | 1.5 h | 82 / 26 |
| churn seed 1 | 1,779 | 0.9235 (779) | 2.6 h | 187 / 49 |

The genotype counts are the generation's best genome in the run log; the phenotype counts are the
best-by-train genome scored at the stop. Most genotype nodes are not connected to the output.
Population mean genotype size at the stop: shape-C 49 / 30 / 27 hidden nodes, churn 89 / 194.

## 4. Checkpoint tables (`checkpoints.csv`, `checkpoints.json`)

- **Best-by-train genome:** the best up to each generation (ties to the earliest generation).
- **AUCs:** train is on the 80% fit rows, monitor on the 20% monitoring rows, test on the test split.
- **Hidden / conns / depth:** of the phenotype.
- **Shape scores (shape-C):** U r and saddle r as in `score_g300.py`.
- **Spurious share:** 1 − R² of the model logit against the true logit, on 20,000 fresh rows.
- **Noise share:** the logit variance carried by `noise`.
- **Marginal share:** hidden units whose mean-ablation costs < 0.01 training AUC.

For comparison, the 100-generation runs on the full training rows had test AUC 0.840–0.873
(shape-C) and 0.894–0.911 (churn200, 200 generations).

**shape-C seed 0** (`longrun-shape-C-s0`)

| Gen. | Best from gen. | Train | Monitor | Test | Hidden / conns / depth | U r | Saddle r | Spurious share | Noise share | Marginal share |
|---|---|---|---|---|---|---|---|---|---|---|
| 100 | 100 | 0.888 | 0.884 | 0.866 | 10 / 34 / 5 | 0.97 | 0.93 | 0.159 | 0.003 | 0.10 (1/10) |
| 200 | 196 | 0.892 | 0.888 | 0.864 | 10 / 34 / 5 | 0.97 | 0.94 | 0.149 | 0.003 | 0.10 (1/10) |
| 300 | 299 | 0.898 | 0.888 | 0.869 | 15 / 45 / 6 | 0.97 | 0.94 | 0.146 | 0.003 | 0.53 (8/15) |
| 500 | 499 | 0.902 | 0.892 | 0.869 | 16 / 47 / 6 | 0.97 | 0.95 | 0.135 | 0.004 | 0.50 (8/16) |
| 1,000 | 997 | 0.903 | 0.886 | 0.869 | 18 / 50 / 8 | 0.95 | 0.94 | 0.145 | 0.004 | 0.39 (7/18) |
| 1,510 (stop) | 1,509 | 0.905 | 0.887 | 0.868 | 21 / 58 / 8 | 0.94 | 0.93 | 0.155 | 0.003 | 0.48 (10/21) |

**shape-C seed 1** (`longrun-shape-C-s1`)

| Gen. | Best from gen. | Train | Monitor | Test | Hidden / conns / depth | U r | Saddle r | Spurious share | Noise share | Marginal share |
|---|---|---|---|---|---|---|---|---|---|---|
| 100 | 100 | 0.880 | 0.882 | 0.860 | 12 / 34 / 4 | 0.97 | 0.88 | 0.163 | 0.005 | 0.33 (4/12) |
| 200 | 200 | 0.887 | 0.877 | 0.859 | 15 / 42 / 7 | 0.91 | 0.85 | 0.224 | 0.006 | 0.27 (4/15) |
| 300 | 300 | 0.891 | 0.878 | 0.857 | 14 / 41 / 6 | 0.91 | 0.88 | 0.216 | 0.011 | 0.21 (3/14) |
| 500 | 484 | 0.894 | 0.878 | 0.855 | 18 / 50 / 6 | 0.85 | 0.90 | 0.201 | 0.011 | 0.39 (7/18) |
| 1,000 | 999 | 0.900 | 0.890 | 0.862 | 21 / 58 / 9 | 0.84 | 0.93 | 0.170 | 0.009 | 0.43 (9/21) |
| 2,000 | 1,922 | 0.906 | 0.898 | 0.859 | 19 / 54 / 9 | 0.86 | 0.94 | 0.177 | 0.011 | 0.32 (6/19) |
| 2,650 (stop) | 2,064 | 0.906 | 0.898 | 0.860 | 19 / 54 / 9 | 0.86 | 0.94 | 0.177 | 0.011 | 0.32 (6/19) |

**shape-C seed 2** (`longrun-shape-C-s2`)

| Gen. | Best from gen. | Train | Monitor | Test | Hidden / conns / depth | U r | Saddle r | Spurious share | Noise share | Marginal share |
|---|---|---|---|---|---|---|---|---|---|---|
| 100 | 100 | 0.883 | 0.882 | 0.861 | 11 / 32 / 4 | 0.98 | 0.93 | 0.154 | 0.000 | 0.18 (2/11) |
| 200 | 200 | 0.893 | 0.908 | 0.868 | 13 / 37 / 4 | 0.98 | 0.96 | 0.137 | 0.000 | 0.38 (5/13) |
| 300 | 299 | 0.898 | 0.910 | 0.869 | 14 / 40 / 4 | 0.98 | 0.97 | 0.113 | 0.000 | 0.50 (7/14) |
| 500 | 500 | 0.901 | 0.909 | 0.870 | 15 / 42 / 4 | 0.99 | 0.97 | 0.097 | 0.000 | 0.33 (5/15) |
| 1,000 | 734 | 0.901 | 0.909 | 0.871 | 15 / 41 / 4 | 0.98 | 0.97 | 0.096 | 0.000 | 0.33 (5/15) |
| 1,304 (stop) | 1,276 | 0.904 | 0.886 | 0.866 | 19 / 51 / 8 | 0.76 | 0.94 | 0.268 | 0.000 | 0.42 (8/19) |

**churn seed 0** (`longrun-churn-s0`)

| Gen. | Best from gen. | Train | Monitor | Test | Hidden / conns / depth | Marginal share |
|---|---|---|---|---|---|---|
| 100 | 100 | 0.897 | 0.885 | 0.862 | 16 / 70 / 3 | 0.81 (13/16) |
| 200 | 182 | 0.903 | 0.880 | 0.876 | 18 / 66 / 4 | 0.72 (13/18) |
| 300 | 300 | 0.905 | 0.879 | 0.874 | 18 / 72 / 6 | 0.72 (13/18) |
| 500 | 497 | 0.917 | 0.908 | 0.903 | 22 / 79 / 4 | 0.73 (16/22) |
| 1,000 | 984 | 0.919 | 0.898 | 0.890 | 28 / 82 / 6 | 0.79 (22/28) |
| 1,694 (stop) | 1,530 | 0.921 | 0.895 | 0.890 | 26 / 81 / 7 | 0.77 (20/26) |

**churn seed 1** (`longrun-churn-s1`)

| Gen. | Best from gen. | Train | Monitor | Test | Hidden / conns / depth | Marginal share |
|---|---|---|---|---|---|---|
| 100 | 99 | 0.921 | 0.906 | 0.911 | 18 / 80 / 4 | 0.83 (15/18) |
| 200 | 190 | 0.924 | 0.906 | 0.904 | 16 / 78 / 4 | 0.81 (13/16) |
| 300 | 293 | 0.926 | 0.912 | 0.910 | 33 / 110 / 7 | 0.91 (30/33) |
| 500 | 408 | 0.931 | 0.921 | 0.909 | 23 / 90 / 4 | 0.87 (20/23) |
| 1,000 | 408 | 0.931 | 0.921 | 0.909 | 23 / 90 / 4 | 0.87 (20/23) |
| 1,779 (stop) | 1,571 | 0.933 | 0.908 | 0.912 | 49 / 137 / 9 | 0.94 (46/49) |


**Figures** (log generation axis; the AUC and genotype size come from every generation's stored
best genome, the other panels from the log grid):

- `figures/shape-C_curves.png`: training / monitoring / test AUC, hidden units, marginal share,
  U r and saddle r, and the spurious and noise shares;
- `figures/churn_curves.png`: AUC, hidden units, marginal share.

Per-generation data: `curves/<name>.csv.gz` and `logs/<name>.gen.csv`.

## 5. Explanations

### 5.1 Points explained (`explain_targets.json`)

**The longest shape-C run is seed 1 (2,650 generations).** Its best-by-train genomes:
- **generation 300:** `lr-b52d3`, from generation 300;
- **generation 2,000:** from generation 1,922;
- **final:** from generation 2,064, `lr-9475d`.

The 2,000 and final genomes are **the same network**: the same NEAT key (440594), identical
topology, connection weights within 0.095, logits correlated 0.99998. Explaining it twice would
add nothing, so it was explained once, at the final point. This departs from section 2.5, which
treated them as different stored rows. The run's best genome did not change structurally over its
last 730 generations.

**Two points were added after the runs, as exploratory additions, not part of the fixed design:**
- **shape-C seed 2 final (`lr-ae3cb`):** the clearest late degradation (U r 0.98 → 0.76, spurious
  0.10 → 0.27 in the last 300 generations). It replaces the duplicated mid point.
- **churn seed 1 at generation 200 and final (`lr-72c47`, `lr-8a025`):** the run that grew most
  (16 → 49 phenotype units).

Churn seed 0 at 200 and final (`lr-10a59`, `lr-84498`) are as designed.

**Method:** the frozen method, `EXPLAIN_TASK.md` / `EXPLAIN_TASK_CHURN.md`. One explaining agent
per genome, on the same model as the earlier studies. Then `verify_record.py`, one blind verifier
each (`VERIFY_TASK.md` / `VERIFY_TASK_CHURN.md`), and `compute_levels.py --set longrun`.

The explainers were not told the generation. The database experiment names (`longrun-...`) are
visible to the tools, so an explainer could have looked. Evidence uses the split's full training
rows (1,500 / 4,000), which include the 20% monitoring rows that evolution did not see.

### 5.2 Results

**Columns:**
- **Time:** the explainer's start-to-end time.
- **Levels:** the probe v2 understood fraction / the blind verifier's leaf grades.
- **Verifier:** the overall grade, then refuted / weakened / upheld findings, then refuted and
  weakened per 100 statements checked.
- **GT:** truth terms recovered (`gt_grades.json`).

| Model | Run, point | Hidden units | Time | Leaves / comps | Splits / identity | Coverage | Function preservation | Closed-form error | Recheck | Levels: probe / verifier | Verifier | Per 100 statements | GT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lr-b52d3` | shape-C s1, gen. 300 | 14 | 17 min | 7 / 3 | 6 / 4 | 1.0 / 1.0 | 1.1e-15 | 0.020 | 0 / 0 | 0.86 / 0.71 | weakened: 0 / 4 / 19 | 0.0 / 2.8 (142) | 7.5 / 9 |
| `lr-9475d` | shape-C s1, final (2,650; = gen. 2,000) | 19 | 27 min | 7 / 3 | 10 / 9 | 1.0 / 1.0 | 7.2e-16 | 0.016 | 0 / 0 | 0.43 / 0.71 | weakened: 0 / 11 / 24 | 0.0 / 9.8 (112) | 6 / 9 |
| `lr-ae3cb` | shape-C s2, final (1,304), added | 19 | 24 min | 7 / 3 | 11 / 1 | 1.0 / 1.0 | 1.8e-15 | 0.036 | 0 / 0 | 0.71 / 0.57 | weakened: 0 / 7 / 17 | 0.0 / 6.3 (112) | 7 / 9 |
| `lr-10a59` | churn s0, gen. 200 | 18 | 18 min | 10 / 3 | 6 / 6 | 1.0 / 1.0 | 4.4e-16 | 0.030 | 0 / 0 | 0.60 / 0.60 | refuted: 1 / 6 / 19 | 0.6 / 3.6 (168) | – |
| `lr-84498` | churn s0, final (1,694) | 26 | 20 min | 9 / 4 | 13 / 4 | 1.0 / 1.0 | 8.9e-16 | 0.138 | 0 / 0 | 0.78 / 0.44 | refuted: 2 / 4 / 23 | 1.3 / 2.5 (160) | – |
| `lr-72c47` | churn s1, gen. 200, added | 16 (the record says 15) | 23 min | 9 / 2 | 1 / 10 | 1.0 / 1.0 | 4.4e-16 | 0.032 | 0 / 0 | 0.56 / 0.67 | refuted: 2 / 6 / 22 | 1.2 / 3.6 (165) | – |
| `lr-8a025` | churn s1, final (1,779), added | 49 | 21 min | 13 / 1 | 23 / 2 | 1.0 / 1.0 | 6.7e-16 | 0.027 | 0 / 0 | 0.92 / 0.31 | weakened: 0 / 6 / 22 | 0.0 / 5.1 (118) | – |

- **Decomposed level** (`levels/`): granularity 1.0 and no coarse-decomposition warning in all
  seven. The largest part grows with the network:
  - shape-C: 4 units → 8;
  - churn seed 1: 7 → 15 (the 49-unit network's `night_calls_tangle`, a part with negligible
    effect, described rather than explained).
- **Where the later explanations got stuck:** nowhere within the box. The explainers' `stuck`
  fields list:
  - restructuring cost: more identity ports and splits for cross-leaf edges (9 identity nodes in
    `lr-9475d`, 23 splits in `lr-8a025`);
  - tool friction: `query_linearise` at parts level fails with "not a DAG" on two networks;
  - dead or constant units that the concept probe matches to arbitrary state indicators.

**Refuted statements (all on churn, all wording):**
- **`lr-10a59`:** evening minutes called a minor partner of the heavy-use unit. They move risk from
  5% to 93% at heavy day use.
- **`lr-84498`:** a state part's list of states omits Louisiana (one part, two places).
- **`lr-72c47`:** "most states have no edge". 31 of 51 have one (one part, two places).

The shape-C verifiers refuted nothing. Their weakenings are shape words:
- the side of the U said to be steeper;
- the sodium minimum given as 138 when it is 139;
- "steady" age effects that step for one subgroup;
- "fast heart rate with low or normal pressure" when it holds only below about 125 mmHg.

### 5.3 Ground truth across the three shape-C points (`gt_grades.json`)

| Term | gen. 300 (`lr-b52d3`) | final, seed 1 (`lr-9475d`) | final, seed 2 (`lr-ae3cb`) |
|---|---|---|---|
| Sodium U | recovered, minimum near 139 | recovered, minimum 138.5–139 | recovered, flat bottom 134–139 |
| Saddle | recovered: crossing interaction, crossover 120–140 mmHg | recovered: discordant pressure and pulse raise risk | recovered: saddle from an sbp/hr arm, a tachycardia gate and a brake |
| Backbone | age, creatinine, smoker, diabetes, BMI recovered; male missed | creatinine and smoker recovered; age, diabetes and BMI partial (spurious gate); male missed | age, creatinine and smoker recovered; diabetes and male only through the gate; BMI barely used |
| Noise | small fitted edge, flagged | fitted effect, larger at high sodium | unconnected |
| Score | 7.5 / 9 | 6 / 9 | 7 / 9 |

Both shapes were named at every point. The later models add spurious couplings around them:
- **seed 1:** a sex × diabetes × age × BMI gate feeding a unit that low sodium also drives;
- **seed 2:** a shock gate that low sodium also opens, and sodium thresholds moved by pressure.

The explanations report these couplings, and the verifiers confirmed them. The whole-model
probes agree: U r 0.91 → 0.86 (seed 1, 300 → final) and 0.98 → 0.76 (seed 2, 1,000 → final).

## 6. Deviations and notes

- **The generation-2,000 point is not separate.** It is the same network as the final point of
  shape-C seed 1, so it was explained once (section 5.1). Points were added for shape-C seed 2 and
  churn seed 1.
- **The full-population checkpoints include generation 300**, beyond the list in section 2.3. This
  was harmless: it only added storage.
- **`DatabaseBackpropPopulation` has a new light storage option.** `genome_storage="best"`,
  `full_save_generations` and `run(on_generation=...)`; the default is unchanged
  (`tests/test_db/test_population_light_storage.py`). Database rows per run: one population row
  per generation, one genome per generation, plus 250 at each full-save generation. The parent
  sidecars are 1.7–2.3 MB each.
- **A 5-generation dry run is left in `explaneat_dev`** as the harness check
  (`longrun-dryrun-both-s0`).
- **Tool issues reported by the explainers:**
  - `annotate` refuses a leaf whose entry is another leaf's exit, so identity ports are needed;
  - `query_linearise(level="parts")` fails with "not a DAG" on two networks;
  - `get_formula` renders one composition argument as 0.0;
  - `query_ablate`'s `switched_off` list starts with a header entry, so the k-th edge is at
    index k + 1;
  - one batched `add_identity_node` was silently dropped;
  - the concept probe matches constant exits to arbitrary indicators.
- **Not done:** Mike's sign-off and plain-language drafts.
