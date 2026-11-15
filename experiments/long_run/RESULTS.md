# Very long runs: are PropNEAT models explainable only because the runs stop early?

Exploratory, not pre-registered. Branch `exp/long-run`. Section 2 was written and committed before
any long run started; results are added below it as the checkpoints complete.

**Question (Mike).** Every result so far ran 40–300 generations. Perhaps PropNEAT models are
explainable only because evolution is stopped early. Over thousands of generations, do the networks
keep growing structure, overfit, accumulate marginal capacity and spurious terms, and become hard to
explain?

## 1. Answer in brief

(Filled in when the runs and the explanations are complete.)

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
