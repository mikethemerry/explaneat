# Chapter 7 run configuration: what the run records say (21 Nov 2026)

Values below are quoted from `data.json` / `run_config.csv` (written by `extract.py` and `make_table.py`);
rerun the scripts before quoting them elsewhere. The prose compared against is Chapter 7 of the thesis
(`sections/7-ExplaNEAT.tex`) and the new common-Methods sentence that cites `tab:ch7-run-config` ("Where a
study departs from this configuration, for example in the number of generations, the population or the
starting topology, it says so in its methods").

## The configuration is not one config with a few departures

1. **Three NEAT config families, not one.** Structural mutation rates and speciation differ between studies,
   and the prose names none of these departures:
   - Heart-style (`experiments/10seed/configs/heart.ini`; Heart original and Heart-100, synthetic, Cox,
     Connect-4, XOR): node add/delete 0.4/0.2, conn add/delete 0.5/0.3, compatibility threshold 3
     (disjoint coefficient 1). XOR has node add 0.3.
   - Adult / Churn (`adult.ini`, `churn.ini`; Adult's original network and the ten Experiment 2 Adult seeds,
     Churn): node 0.7/0.1, conn 0.68/0.15, threshold 5 (disjoint 2).
   - MONK-2 (original network and the ten Experiment 3 seeds; neat-python's default template values): node
     0.15/0.05, conn 0.3/0.1, enable toggle 0.01, and **weight and bias mutation switched on**
     (0.8/0.5/0.1 and 0.7/0.5/0.1, rate/power/replace). Every other family has weight and bias mutation 0:
     only backprop changes their weights.
   Experiment 2's bullet list ("a population of 250 for 100 generations; five epochs ...") is right, but
   Adult and Churn ran a different NEAT config from Heart-100.
2. **Phantom-weight mask.** Experiment 2, the three original networks, the synthetic runs and the ten
   MONK-2 seeds trained **unmasked**: the code at their commits has no mask (or the
   run recorded `mask_phantom: false`), so dense phantom weights trained and were discarded at write-back.
   Cox, Connect-4 and XOR trained masked; Experiment 3's fine-tuning is masked. The prose does not say.
3. **Heart's original network: 40 of 100 generations.** The run was launched from template
   `high-exploration` with 100 generations and is stored `interrupted` after 40; "the Heart model for 40"
   in the prose reads as a design choice (`experiments/10seed/PREREG.md` already notes this). Provenance
   note: the table prints the generations stored (40, Mike's decision of 21 Nov 2026); the planned budget
   is the launch template's `n_generations` (`data.json`, `families.paper_heart.template.training`), and
   `make_table.py` asserts that a run stopped short of its budget is stored `interrupted`.
4. **Departures by study** (all in the table): Cox — population 150, 60 generations, identity output,
   fitness −Cox loss − λk/D with λ = 1, initial weight SD 0.1, selection on validation; Connect-4 — population
   150, 80 generations, **50** epochs per generation, 10 hidden units at the start, initial weight SD 0.15;
   XOR — population 150, fitness 1/BCE, run until the best genome solves the four rows (cap 150), fitness
   threshold 10^9, reset on extinction. Fitness is training AUC only for the BCE-trained classification
   studies.
5. **Experiment 3 has its own evolution family**: the ten MONK-2 seeds (`10seed-monk2-s0..9`, 100
   generations, the MONK-2 config) are not in any Experiment 2 table. Fine-tuning is Adadelta 1.5, BCE,
   masked, E = 500 (also 50 in the repair study and 2,000 in re-initialisation) or early stopping
   (patience 50, min. change 1e-5, at most 5,000 epochs, 20% of training rows held out).

## Code commits

- Optimiser: every NEAT run's `backproppop.py` (at its recorded commit) is full-batch Adadelta, lr 1.5,
  no weight decay; the Experiment 3 fine-tune uses the same (`genome_surgery.FT_*`).
- Seeds of one family ran at different commits for Adult (2), Churn (2), Connect-4 cells (2) and Connect-4
  lines (3). `git diff` of the training path
  (`backproppop.py`, `neuralneat.py`, `db/population.py`, `evaluators/`) between them is empty in every case.
- **Dirty trees**: the WHAS500 Cox runs and Connect-4 cells seeds 0 and 1 recorded `code_dirty: true`, so
  their commit does not fully pin the code they ran. Both presented networks of these families (the selected
  WHAS500 seed and Connect-4 cells seed 1) are among them. `run_config.csv` `commits` lists the commits of the
  presented runs only.
- XOR's runs record no commit; the code read is the one of the commit that last wrote `xor_results.json`.
- The original networks record no seed (`random_seed` null): "1 (n.r.)" in the table.

## ExConNEAT records

- Phase 1 (`results/sweep_p50_m4_e5`): `config.json` holds the arguments of the original 22-minute sweep
  (`patience` 100000, `max_minutes` 22). The run was then resumed to generation 67 and stopped by decision;
  the resumed run's plateau setting is not recorded. The table gives 68 generations, "stopped by decision".
- Phases 2 and 3 record `args.pop` 100, but a warm start takes the checkpoint's population; the config they
  ran with has population 50 (as the book says). The table prints the config's value.
- The structural rates are 4x the code defaults (checked: run rate / `ConvNeatConfig` default = `mut_scale`).

## Trimmed to the presented runs (21 Nov 2026, branch thesis/run-config-trim)

Mike's decision: the table shows only the runs that produced the networks Chapter 7 presents.

- **Seed(s) presented** (was "Seeds"): the seed of each presented network, read from the selection the
  records state (`selected` in `data.json`; the synthetic ones asserted equal to the explained genomes in
  `experiments/benchmarks/results/propneat.json`). All seeds where all are presented: Experiment 2 (ten
  networks per dataset), XOR (all seeds analysed), and Experiment 3's ten MONK-2 seeds (kept unchanged until
  Experiment 3's scope is decided). The original networks record no seed: "n.r." (was "1 (n.r.)").
- **Cox: WHAS500 only.** GBSG2's network is not presented: `tab:exp1-networks` has WHAS500 only and the
  survival benchmark table gives GBSG2 as a side note. `extract.py` now reads only the five WHAS500 runs
  (`COX_PRESENTED`); every check (one config, config file, selection rule) still holds on them.
- **Dropped: the synthetic 300-generation and sigmoid-unit families.** Their networks appear only in the
  options figures (`thesis/ch7/FIGURES.md`, "Options": `synthetic-shape-c-s1-g300-*`,
  `synthetic-shape-c-sig-best-g300-*`), not in any used figure or upload table, and none is explained. Their
  run lists (`experiments/ushape_saddle/g300_scores.csv`, `sig_gens.json`) are no longer read; finding 2
  above still holds for them (they trained unmasked) if they are ever presented.
- **Heart's original network: "40"** (was "40 of 100"; see finding 3 for the budget).
- **ExConNEAT panel (b): the presented network's lineage only.** The MNIST 10-class network of the 50k warm
  start, as its three phases (the 10k run, its pruning phase, the 50k warm start), each read from its phase
  records as before; the Classes column is gone (the panel header gives the class count, asserted equal in
  the three phases). **Dropped: MNIST 3 vs 8 vs noise** (`three8noise/results/run_s0`), not presented.

## Author's account of the config families (21 Nov 2026)

Not in the run records; recorded here from Mike's recollection so the methods can cite it.
The Adult and Churn family (the original Adult run "No pressure", 24 Jun 2026, and Churn, which
copied Adult's NEAT settings) came from a series of exploratory runs that tried to force
non-explainable models by easing the downward pressure on size: higher node and connection
add rates, lower delete rates and a wider speciation threshold than the Heart template. The
values themselves are in the table and in `experiments/10seed/PREREG.md` ("Configs"). The
MONK-2 original run used neat-python's default template (weight and bias mutation on); nothing
records that as intended. The Heart original run stopped after 39 evaluated generations of a
100-generation budget (stored as interrupted).

## The original Adult network back in Experiment 1 (21 Nov 2026, branch thesis/adult-exp1)

Mike's decision: Experiment 1 presents the original Adult network again, beside the original Heart and MONK-2
networks (Adult stays the ten-seed study of Experiment 2). New row "Adult, original network" (`paper_adult`),
read like the other original networks:

- **Which run.** The explained genome `4bc8fa07` is read from `experiments/benchmarks/results/propneat.json`
  when that file holds an Experiment 1 Adult entry (added on branch thesis/wide-grids), else from
  `experiments/10seed/runs_paper_reference.csv` (the three original networks); where both hold it they must
  agree. Its experiment is "No pressure" (`dbab685c`), 100 generations stored, `completed`, no seed recorded
  ("n.r."), commit `0bd2e54`; the genome has the run's highest training fitness (asserted).
- **Config.** `config_json` equals `experiments/10seed/configs/adult.ini` in every option (asserted, as for
  Heart and MONK-2). The run was launched from template `high-exploration` (`bb576521`, the Heart run's
  template) with six NEAT options changed (`PREREG.md`: "template `bb576521` + overrides"). The extractor
  asserts that exactly these six differ and records both values (`data.json`,
  `families.paper_adult.template.overrides`): node add/delete 0.4/0.2 -> 0.7/0.1, conn add/delete
  0.5/0.3 -> 0.68/0.15, compatibility threshold 3 -> 5, disjoint coefficient 1 -> 2. This is the Adult /
  Churn family of finding 1, the run the author's account below describes.
- **Training.** The run stored no `run_params`; generations, epochs and fitness come from the template's
  training block (100, 5, AUC), as for Heart's original network; population 250 equals the stored config.
  No phantom mask: the code at `0bd2e54` has no mask parameter.

Both scripts were run twice after the change: byte-identical `data.json`, `run_config.csv` and `.tex`.
