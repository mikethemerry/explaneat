# Modular crossover: explained chunks as heritable modules (thesis Ch 7)

Branch `exp/modular-crossover` (from `exp/10seed-consistency`). Code in
`experiments/modular/`, tests in `tests/test_experiments/test_modular.py`.
This plan is committed before any outcome run. Two pilots were run for
timing and debugging only (`results/pilot`, `results/pilot2`, logs under
`logs/pilot*`; not committed); they are not analysed for outcomes and are not reported as
results.

## Question

Every K generations, find explained chunks (subgraphs with a clean
boundary) in island elites, put them in a shared module library with a
module id, and use them in insertion, crossover and replacement. Does this
higher-order sharing help, compared with (1) sharing whole elite genomes,
(3) sharing the same module topologies with random weights, and (4)
sharing the topologies only, letting backprop train them?

Prior results this design respects (`experiments/churn/EXTENSIONS_RESULTS.md`):
chunks carry their weights; every graft gets a readout refit (raw grafts
double-count); MONK-2 diverges across runs (3/10 learned the rule in
`experiments/10seed`), unlike churn; a sham arm is required.

## Tasks

Both tasks use fixed train/test splits; NEAT trains and selects on train
only; outcomes are on test.

**MONK-2** (`tasks.load_monk2`): the `experiments/10seed` split
(split `63257a45-…`, dataset `c2340904-…`), exported once from Postgres by
`export_monk2.py` to `data/monk2.npz` with the split's stored
StandardScaler: 345 train / 87 test rows, 17 inputs. Config
`configs/monk2.ini` = `experiments/10seed/configs/monk2.ini` unchanged
(pop_size overridden per island, below). One stage.

**MUX-11 curriculum** (`tasks.load_mux11`): 11-input multiplexer, address
bits a0..a2 and data bits d0..d7, inputs encoded −1/+1;
y = d_k with k = a0 + 2·a1 + 4·a2 (bit 1 ⇔ input +1). All 2^11 = 2048
inputs occur once; a stratified 50/50 split with
`numpy.random.default_rng(20261106)` gives 1024 train / 1024 test rows.
Noise-free (no label noise).
- Stage 1: rows with a2 = 0 (495 train / 529 test): a 6-multiplexer on
  a0, a1, d0..d3, with the a2 input constant at −1.
- Stage 2: all rows (1024 / 1024).
- **Stage switch rule (pre-registered): fixed generation**, the first
  generation of the second half of the run (generation n_gen/2, see Budget).
  At the switch every island's train data becomes the stage-2 train rows.
- **Ground-truth features**: g_k(x) = [address = k]·[d_k = 1], k = 0..7
  (so y = Σ_k g_k exactly), and the address decoders D_k(x) = [address = k].
- Config `configs/mux11.ini` = the MONK-2 config with `num_inputs = 11`.

Fitness (both tasks): `auc_fitness` (train AUC on the active stage), as in
`experiments/10seed`.

## Islands and evolution

- 4 islands per run, each an `IslandPopulation` (subclass of
  `BackpropPopulation`, masked backprop = the default `mask_phantom=True`):
  per generation 5 epochs of the PropNEAT backprop primitive on the active
  train rows, then fitness, then reproduction (neat-python
  DefaultReproduction / DefaultSpeciesSet / DefaultStagnation). The loop
  body is BackpropPopulation.run's, split so islands step in lockstep (no
  reporters, no validation data). In memory; no DB writes.
- The 4 islands share one neat Config, hence one node indexer: node ids are
  unique across the run, so grafted / migrated genes never collide with
  later add-node mutations. Genome keys are per island; migrants are
  re-keyed.
- On complete extinction an island is re-initialised (neat
  `reset_on_extinction` semantics) and the event logged (none expected).
- Seeding as `experiments/10seed/run_seed.py`: `random`, `numpy`, `torch`
  seeded with the run seed, `torch.set_num_threads(1)`,
  `torch.use_deterministic_algorithms(True)`, `PYTHONHASHSEED=0`
  (enforced), BLAS limited to 1 thread. Operator randomness uses
  `numpy.random.default_rng([seed, generation, island, tag])`.
  Same seed ⇒ bit-identical run (tested).
- **Checkpoints** every K = 25 generations: at the end of generations
  24, 49, 74, … (after evaluation, before reproduction).
  Genome used per island: that generation's best (highest train fitness;
  ties: first in population order).

## Checkpoint chunk finder (`chunks.py`, algorithmic, no LLM)

The D2/C2 candidate criteria adapted to genomes without a human
explanation. Live nodes = the hidden nodes NeuralNeat computes; only
enabled edges among live nodes, inputs and the output count.
1. Seeds: live hidden nodes with an enabled readout edge to the output.
2. Closure: C = seed ∪ its live hidden ancestors; while a node of C has an
   enabled edge to a live hidden node outside C, add that node and its
   ancestors. C is the smallest node set containing the seed whose ingress
   is only from inputs (**all-input entries**; input entries may be split,
   an identity operation) and whose egress is only readout edges
   (**exit-only egress**; exits are **pure** because every predecessor of a
   C node is in C or an input). So C satisfies the collapse preconditions
   and is **graftable** along its boundary, and has **a readout edge**.
3. Contribution s_C(x) = Σ over C's readout edges of weight × exit
   activation (`genome_surgery.readout_contribution` semantics).
4. **Train restore gain** (as in C/D2, for the genome partitioned into
   chunks): with H the summed contribution of every hidden readout and P
   the least-squares projection onto span{1, inputs} on the active train
   rows, gain = AUC(z − H + s_C + P(H − s_C)) − AUC(z − H + P H), i.e. the
   AUC from restoring this chunk's nonlinearity alone onto the all-linear
   network. A linear chunk has gain 0.
5. **Candidate iff gain ≥ 0.02** (pre-registered; = D2's MIN_GAIN) and
   |C| ≤ 8 hidden nodes (a chunk, not the whole network).
Identical node sets are found once.

## Module library (`registry.py`)

Each candidate is added as a *variant*: entry input ids, internal nodes
(bias, response, activation, aggregation), internal connections and
weights, exits, readout weights, gain, source island / generation / genome,
and its contribution on the task's probe set (the final stage's train
rows, so stage-1 and stage-2 chunks are compared on the same inputs).
- **Alignment**: the variant joins the module whose variant has the
  highest Pearson r with it on the probe set if r ≥ 0.9 (signed; ties →
  lowest id); otherwise a new module id. Library keeps the latest variant
  per (module id, island).
- After alignment the island best's chunk nodes are (re)tagged with the
  module id (re-derived boundaries); `module_tags` live on the genome
  (`ModuleGenome`), are inherited through crossover (from the parent that
  supplied the node gene) and survive mutation. A genome carries module m
  iff it has a node tagged m. For every variant the log records whether its
  nodes already carried that id (descended from a graft) or not (an
  independent alias).
- For MUX-11 every variant's r with each ground-truth feature is logged.

**Primary arm uses open chunks**: module genes keep mutating under normal
NEAT; boundaries are re-derived at each checkpoint. Encapsulated (frozen)
chunks are **not run** in this study (stated now, before outcomes).

## Operators (`operators.py`, `islands.Study._operator_hook`)

Applied once per checkpoint, in the reproduction that follows it (the
offspring of the next generation), to **20% of each island's non-elite
offspring** (round(0.2·n), chosen with the operator rng), assigned round
robin insertion, crossover, replacement, insertion, …
Ordinary NEAT reproduction produces all other offspring.
- **Insertion**: a module id from the library that has a variant from
  another island and that the child does not carry (uniform); its
  highest-gain variant from another island is grafted into the child with
  fresh node ids along its boundary (entries = the shared input ids), then
  readout refit.
- **Crossover**: parent a uniform from the island's top 20% (by fitness,
  previous generation); parent b uniform from the top 20% whose module
  content differs from a's (else from the whole previous generation);
  child = NEAT crossover (fitter parent first) + NEAT mutation; then for
  each module id carried by b with node ids a lacks, the child takes b's
  version (keeping b's node ids, replacing a's version if any) with
  probability 1/2; readout refit of the inherited readouts if any. Only edges from inputs or
  other tagged nodes into b's tagged nodes, and their readouts, are copied.
- **Replacement**: a module id the child carries that has a variant from
  another island (uniform): delete the child's genes tagged with it (KO),
  graft that variant (highest gain, other island), readout refit.
- Not applicable (no such module / pair) → insertion; still not applicable
  → the child is left as NEAT made it (logged). An operator that raises an
  error is treated like an invalid result (logged). A result that is not a
  valid feed-forward genome (acyclic, NeuralNeat builds it and reaches the
  output) is discarded and the NEAT child kept (logged).
- **Readout refit** = `genome_surgery.refit_readout`: logistic regression
  of the grafted readout weights plus the output bias on the active train
  rows, all other weights fixed, L2 1e-6 on readout weights only; readouts
  NeuralNeat does not compute (a dead exit) are left out of the refit. Then the
  normal per-generation backprop (5 epochs) as for every genome.

**Speciation fix (pre-registered)**: compatibility distance counts
module-tagged genes as one unit per module id (`ModuleGenome.distance`): a
module carried by only one genome is one disjoint node gene and one
disjoint connection gene; for a shared module id, tagged genes with the
same key are compared as homologous genes and alias variants add nothing;
genome size counts each module as one gene. Untagged genes as in
DefaultGenome; with no tags it *is* DefaultGenome.distance. Same genome
class and rule in every arm.

## Arms (same seeds, same islands, same budget)

1. **plain**: PropNEAT islands; at each checkpoint every island receives a
   deep copy of each other island's best genome (3 migrants), replacing
   randomly chosen non-elite offspring. No library. The control for
   sharing.
2. **modules**: library + operators, grafted genes carry the variant's
   explained weights and biases, plus readout refit.
3. **sham**: as 2 (same library procedure on this arm's own elites, same
   operators), but every grafted/inherited module weight and node bias is
   drawn fresh from N(0, 1), readout weights then refit.
4. **topology-only**: as 3 but readout weights are set to 0 and **not
   refit**; internal weights and biases N(0, 1). The child then computes
   exactly its pre-graft function and the normal backprop trains the new
   structure. Justification: "3 without refit" with N(0, 1) readouts would
   be a raw random graft, which earlier studies showed damages the
   recipient (a strawman); a zero readout is the standard
   function-preserving way to add capacity, and keeps 3 and 4 distinct
   (3 = random features + optimal readout immediately; 4 = structure only,
   trained by evolution's own backprop). In arms 3 and 4 the library is
   still built from the arm's own elites with their real weights (for
   alignment and gain); only the transplanted weights differ.

Arms 2–4 do not migrate whole genomes. Generations 0–24 are identical
across the four arms for a given seed.

## Outcomes

Final genome of a run: among all genomes of the last generation on all
islands, the highest train fitness (AUC on the final stage's train rows),
ties → lower train BCE → lower island → lower genome key.

**Primary** (per task):
- test AUC of the final genome;
- generations-to-solve: the first generation (0-based) at which the run's
  best train fitness on the final stage is 1.0 (train AUC = 1, the rule is
  separated: the definition under which 3/10 `experiments/10seed` MONK-2
  runs "learned the rule"); unsolved runs are censored at n_gen. For
  MUX-11 only stage-2 generations can solve.
- Comparisons arm 2 vs 1, 2 vs 3, 2 vs 4, paired by seed: number of seeds
  where 2 is better / worse / tied, two-sided sign test (ties dropped),
  median and mean paired difference with percentile bootstrap 95% CIs over
  seeds (10,000 resamples, `numpy.random.default_rng(42)`). Holm
  adjustment over the 12 primary tests (2 tasks × 3 comparisons × 2
  outcomes) is reported next to raw p. Test accuracy at p > 0.5 is
  reported the same way as a co-primary descriptive (AUC fitness does not
  calibrate the threshold, so accuracy is not the headline).

**Secondary**: operator counts (applied per type, not applicable);
module reuse — number of library modules and variants, independent
aliases; module-carrying lineages — genomes in the final generation
carrying any module and carrying a module first found on another island,
island bests carrying modules; MUX-11: library variants matching a
ground-truth feature (r ≥ 0.9 with g_k or D_k on the probe set) and which
features; final network size (live hidden nodes, live enabled
connections); wall time. Secondary pairs 3 vs 1, 4 vs 1, 3 vs 4 are
reported without adjustment.

## Budget

Pilots (timing only): 150 generations × 100 per island took ≈ 2 min per
run; 300 generations × 150 per island ≈ 8.5–11 min per run (4 runs in parallel; networks grow, so later generations are slower).
CPU limit: at most 4 concurrent processes, 1 torch/BLAS thread each.

**Pre-registered**: seeds 0–9 for every (task, arm) = 80 runs;
pop_size 150 per island (600 per run), 400 generations,
MUX-11 stage switch at generation 200 (stage 1 = generations 0–199); K = 25; operator fraction 0.2;
backprop 5 epochs per generation. Expected wall time ≈ 5–6 h on 4
processes (limit 12 h). Run order: seed-major (all 8 task × arm
combinations of seed 0, then seed 1, …), so a partial study is balanced.
If the study had to stop early, only complete seeds (all 8 runs) would be
analysed and this would be reported.

## Files

- `results/main/<task>-<arm>-s<seed>.json` (summary),
  `.detail.json` (per-generation log, checkpoints, operator records,
  library with weights), `.elites.pkl` (final best and island bests).
- `results/runs.csv`, `arm_summary.csv`, `comparisons.csv`, `library.csv`
  (`analyze.py`).
- `logs/main/*.log` progress logs.
- `RESULTS.md` write-up, deviations from this plan listed as such.
