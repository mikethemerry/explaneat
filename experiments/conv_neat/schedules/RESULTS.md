# ConvNEAT search schedules: exploration, exploitation and scarcity over time

Branch `conv-neat-schedules` (local side branch of `conv-neat-prototype`). Code: `explaneat/conv/schedule.py`.
Tests: `tests/test_conv/test_schedule.py`. Runner: `run.py`, `run_all.sh`. Table and figures: `summarise.py`.
No database. CPU only, 2 processes at a time.

## In plain language

We tried four ways of running the same ConvNEAT search for the same 15 minutes of computer time on the
10-class handwritten-digits task (sklearn digits, 8×8 pixels):

1. **Fixed settings** (today's MNIST winner): a lot of structural mutation, 5 training epochs per generation.
2. **A repeating cycle**: explore (grow structure, train briefly), then exploit (no new structure, train long),
   then *scarcity* (only deletions, five times the size penalty).
3. **Self-adaptive**: each network carries its own mutation rate and training length, and evolution tunes them.
4. **Successive halving**: every network gets a short training run. Only the most promising get more.

What we found, with only three seeds per arm:

* **Successive halving was the best use of the time.** It was the most accurate arm on every seed, with
  validation 0.951 against 0.920 for fixed settings and test 0.959 against 0.938. Its networks were also somewhat smaller
  (748 against 891 parameters). Spending training time on the promising networks, rather than equally on all, simply pays.
* **The cycle with scarcity gave networks about half the size, with almost no dead weight.** They had 497 against 891 parameters
  and 11 against 19 hidden maps. Only 6 % of their hidden maps were marginal (removing one changes accuracy by less than 1 point),
  against 26 % for fixed settings. **But this was not free here.** Accuracy was about 2 to 3 points lower on average, mostly because one
  seed lagged (test 0.858). So the answer to "smaller and cleaner at no accuracy cost?" is "smaller and cleaner, yes; no cost, not
  shown". The scarcity windows themselves cost nothing measurable. The monitored accuracy held or rose through each of them, and the
  population's mean size fell by 10 to 20 % in each window. Much of the size gap, though, comes from the cycle simply spending only
  half its generations growing.
* **Self-adaptation discovered "train more".** Each network's epochs gene drifted upward, from about 5 to about 13, and its mutation scale rose
  a little, from 4 to between 4.4 and 7.8. Training longer always raises fitness, and fitness does not charge for time, so the
  networks chose it. The result was fewer generations (16 against 26) and no gain in accuracy (validation 0.917, test 0.919).
  Without a time cost in fitness, this arm is just a slow way to raise the epoch count.

**Recommendation for the rig run:** use **successive halving** for the training budget. It is the only arm that beat fixed
settings at equal time, on all three seeds, and it costs nothing to add. If network size and explainability matter as much as
accuracy, the next experiment should be **halving plus a periodic scarcity window** (e.g. 2 scarcity generations every 10).
Run it at a longer budget, because the cycle arm finished fewer than 3 cycles here. Do not use the self-adaptive epochs gene
unless fitness charges for training time.

## Setup

* **Task:** sklearn `load_digits`, 10 classes, 8×8, pixels / 16. Split 60/20/20, stratified, seeded per run seed:
  1,078 training, 359 validation and 360 test images.
* **All arms:** population 50, λ = 0.001 (×5 in scarcity), max level 2, Adam lr 0.01 and batch 128 (the prototype's
  defaults). Fitness = training accuracy − λ × (hidden maps + enabled edges); validation is only monitored.
  The selected genome is the best by validation among each generation's best-by-fitness genome (run_long.py's rule).
  **Test is read once, for that genome, at the end.** The test curve in the figures is for display only; it is never used to select.
* **Budget:** 15 minutes of evolution wall time per run (evaluate + speciate + reproduce, checked after each generation, so
  runs overshoot by up to one generation: 15.4 to 16.8 min). Torch 1 thread, CPU. The machine was shared with other
  jobs (load average 8 to 14 on 14 cores), so all arms ran under similar contention, scheduled two at a time in the
  order fixed/phase, selfadapt/halving for each seed.
* **Seeds:** 0, 1 and 2 per arm. The same seed means the same data split and the same initial population draw.
* **Marginal nodes:** mean-ablation of each hidden map of the selected genome on the validation set
  (`explaneat/conv/explain/importance.py`, map held at its per-pixel mean). Marginal = accuracy drop < 0.01.
  Hidden maps off every input → output path count as marginal too. That happened once: one map in fixed seed 1, which is why its count reads 8 of 21 here and 7 of 20 functional.

### Arms, exactly

| arm | structural mutation (add node, add connection, kernel size) | toggle / level | weight perturbation | epochs | λ | deletions |
|---|---|---|---|---|---|---|
| 1 fixed | ×4 of default (1.0, 1.4, 0.4 expected per offspring) | default (0.05, 0.05) | 0.6 | 5 | 0.001 | none |
| 2 cycle, explore (5 gens) | ×4 | default | 0.6 | 2 | 0.001 | none |
| 2 cycle, exploit (3 gens) | 0 | 0 | 0.3 (×0.5) | 20 | 0.001 | none |
| 2 cycle, scarcity (2 gens) | 0 | 0 | 0 | 5 | 0.005 | 1 connection + 0.5 nodes expected per offspring |
| 3 self-adaptive | default × the genome's `mut_scale` | default | 0.6 | round(the genome's `epochs`) | 0.001 | none |
| 4 successive halving | ×4 | default | 0.6 | 2 for all; the top 20 % by fitness get +8; species leaders brought to 20 | 0.001 | none |

* **Cycle:** a 10-generation cycle (`schedule.PHASE_CYCLE`). Generations 0–4 explore, 5–7 exploit, 8–9 scarcity, 10–14
  explore, and so on. The phase of generation g sets the mutation that *creates* generation g and the epochs and λ that
  *evaluate* it (switching is tested).
* **Deletion mutations:** `delete_conn` removes an enabled connection, chosen uniformly among the 25 % with the smallest kernel
  norm, and never removes a node's last enabled input. `delete_node` removes a hidden map and all its genes, chosen among
  the 25 % with the smallest outgoing kernel norm, and never when that would leave a node with no input. Afterwards,
  `prune_dead_ends` removes hidden maps with no enabled output; that step is exact.
* **Self-adaptive:** initial `mut_scale` = 4·e^N(0, 0.25) and `epochs` = 5·e^N(0, 0.25). The child inherits from the fitter parent,
  then each value is ×0.8 or ×1.25 with probability 0.2, clipped to [0.25, 16] and [1, 30]. Elites keep their values.
* **Successive halving:** after the 2-epoch evaluation, the top 20 % (10 genomes) train 8 more epochs and are re-scored. Then the
  best genome of each species (assigned to last generation's representatives, as speciation does) is brought to 20
  epochs in total. Each extra round is a fresh Adam run from the written-back weights.

## Results (3 seeds per arm, mean ± sd)

| arm | generations | val (selected) | test (selected) | hidden maps | edges | params | marginal share | final population mean complexity |
|---|---|---|---|---|---|---|---|---|
| 1 fixed (4× mutation, 5 epochs) | 26 ± 1 | 0.920 ± 0.013 | 0.938 ± 0.013 | 19.0 ± 1.6 | 59.0 ± 6.5 | 891 ± 78 | 0.256 ± 0.089 (4/19, 8/21, 3/17) | 81.8 ± 7.3 |
| 2 phase cycle | 27 ± 0 | 0.903 ± 0.025 | 0.907 ± 0.037 | 10.7 ± 1.2 | 30.7 ± 3.3 | 497 ± 60 | **0.056 ± 0.079** (2/12, 0/11, 0/9) | **46.0 ± 2.1** |
| 3 self-adaptive | 16 ± 2 | 0.917 ± 0.003 | 0.919 ± 0.017 | 12.7 ± 3.3 | 45.3 ± 13.0 | 679 ± 111 | 0.163 ± 0.094 (1/12, 5/17, 1/9) | 70.2 ± 13.9 |
| 4 successive halving | 29 ± 1 | **0.951 ± 0.014** | **0.959 ± 0.006** | 17.3 ± 2.9 | 53.0 ± 7.5 | 748 ± 112 | 0.352 ± 0.052 (5/14, 6/21, 7/17) | 77.4 ± 4.7 |

Per seed (val / test / params / marginal):

| seed | fixed | phase | self-adaptive | halving |
|---|---|---|---|---|
| 0 | 0.911 / 0.919 / 840 / 4 of 19 | 0.916 / 0.947 / 567 / 2 of 12 | 0.916 / 0.897 / 637 / 1 of 12 | 0.936 / 0.953 / 635 / 5 of 14 |
| 1 | 0.939 / 0.947 / 1001 / 8 of 21 | 0.925 / 0.917 / 505 / 0 of 11 | 0.914 / 0.919 / 832 / 5 of 17 | 0.969 / 0.967 / 900 / 6 of 21 |
| 2 | 0.911 / 0.947 / 833 / 3 of 17 | 0.869 / 0.858 / 420 / 0 of 9 | 0.922 / 0.939 / 569 / 1 of 9 | 0.947 / 0.958 / 708 / 7 of 17 |

* **Halving vs fixed:** halving is higher on every seed in both validation and test. Paired test differences are +0.034, +0.020 and +0.011.
* **Cycle vs fixed:** the cycle's genomes are smaller on every seed (by 32 to 50 % of parameters). Its accuracy is higher on seed 0 (test),
  lower on seed 1 and much lower on seed 2.
* **The scarcity windows themselves:** in the cycle runs, the monitored validation accuracy across each 2-generation scarcity window
  went 0.889 → 0.897, 0.883 → 0.894 and 0.864 → 0.855 in the second cycle, and population mean complexity fell 39.7 → 35.7,
  37.7 → 30.4 and 34.5 → 29.9. Deletions: about 180 connections and 85 nodes per run.
  The cycle explores only 5 of every 10 generations, though, so part of its size advantage is simply less growth, not scarcity.
  This experiment does not separate the two effects.
* **Exploit phases do the accuracy work in the cycle:** the jump into the first exploit phase is large (validation 0.42 → 0.69 in
  one generation, seed 0). Explore generations train only 2 epochs, so the cycle's networks are under-trained during the
  explore phases.
* **Halving has the most marginal maps (35 %).** A plausible reading is that longer training on the leaders lets them keep
  maps that the head can do without, and nothing in its fitness removes them (λ is unchanged).
* All runs were still improving at the end (see `figures/val_vs_time.png`), so these are results about the first 15 minutes
  of search, not about converged runs.

## Figures

* `figures/val_vs_time.png`: validation accuracy of the run's current selection against evolution wall time (12 runs).
* `figures/test_vs_time.png`: the same for test (display only, never used to select).
* `figures/size_vs_time.png`: parameters of the current selection against wall time.
* `figures/popsize_vs_time.png`: population mean parameters against wall time.
* `figures/selfadapt_trajectories.png`: the self-adaptive arm's population mean and 10–90 % band of `mut_scale` and
  `epochs` per generation (dashed lines: the initial centres 4 and 5).

## Caveats

* **Small budget, few seeds:** 15 minutes, about 26 generations and 3 seeds per arm. The ± values are standard deviations over 3 runs.
  Only the halving-vs-fixed difference is consistent on every seed. Treat the rest as directions to test, not findings.
* **Small validation set:** 359 images, so one image is 0.28 points. The marginal threshold (0.01) is about 4 images.
  The marginal analysis uses the same validation set that picked the genome.
* **Shared, noisy CPU:** other jobs were running, so generation times varied. All arms ran in the same period, two at a time,
  but wall-time equality is approximate (runs stop after the generation that crosses 15 min, at 15.4 to 16.8 min).
* **Cycle not isolated:** the cycle changes three things at once: the explore epochs (2), the long exploit training and scarcity. It
  completed fewer than 3 cycles, and the runs ended inside an explore or exploit phase. A clean test of scarcity needs fixed settings with
  scarcity windows inserted.
* **Self-adaptive with no time cost:** the epochs gene is selected only on fitness, so it drifts upward. That is a property of the design,
  not of self-adaptation in general.
* **Digits, not MNIST:** 8×8 and about 1k training images. The rig's MNIST run has 50k images and much longer runs. The relative
  value of halving (which saves epochs on unpromising genomes) should, if anything, grow with dataset size, but this is
  untested.

## Reproduce

```
uv run pytest tests/test_conv/test_schedule.py -q
experiments/conv_neat/schedules/run_all.sh            # 12 runs, 2 at a time, about 100 min (MINUTES=15 by default)
uv run python experiments/conv_neat/schedules/summarise.py
```
