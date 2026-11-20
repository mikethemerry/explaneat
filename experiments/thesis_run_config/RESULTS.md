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
   - Adult / Churn (`adult.ini`, `churn.ini`): node 0.7/0.1, conn 0.68/0.15, threshold 5 (disjoint 2).
   - MONK-2 (original network and the ten Experiment 3 seeds; neat-python's default template values): node
     0.15/0.05, conn 0.3/0.1, enable toggle 0.01, and **weight and bias mutation switched on**
     (0.8/0.5/0.1 and 0.7/0.5/0.1, rate/power/replace). Every other family has weight and bias mutation 0:
     only backprop changes their weights.
   Experiment 2's bullet list ("a population of 250 for 100 generations; five epochs ...") is right, but
   Adult and Churn ran a different NEAT config from Heart-100.
2. **Phantom-weight mask.** Experiment 2, the two original networks, the synthetic runs (all three
   variants) and the ten MONK-2 seeds trained **unmasked**: the code at their commits has no mask (or the
   run recorded `mask_phantom: false`), so dense phantom weights trained and were discarded at write-back.
   Cox, Connect-4 and XOR trained masked; Experiment 3's fine-tuning is masked. The prose does not say.
3. **Heart's original network: 40 of 100 generations.** The run was launched from template
   `high-exploration` with 100 generations and is stored `interrupted` after 40; "the Heart model for 40"
   in the prose reads as a design choice (`experiments/10seed/PREREG.md` already notes this).
4. **Departures by study** (all in the table): Cox — population 150, 60 generations, identity output,
   fitness −Cox loss − λk/D with λ = 1, initial weight SD 0.1, selection on validation; Connect-4 — population
   150, 80 generations, **50** epochs per generation, 10 hidden units at the start, initial weight SD 0.15;
   XOR — population 150, fitness 1/BCE, run until the best genome solves the four rows (cap 150), fitness
   threshold 10^9, reset on extinction; synthetic 300-generation and sigmoid-unit variants (option figures
   only). Fitness is training AUC only for the BCE-trained classification studies.
5. **Experiment 3 has its own evolution family**: the ten MONK-2 seeds (`10seed-monk2-s0..9`, 100
   generations, the MONK-2 config) are not in any Experiment 2 table. Fine-tuning is Adadelta 1.5, BCE,
   masked, E = 500 (also 50 in the repair study and 2,000 in re-initialisation) or early stopping
   (patience 50, min. change 1e-5, at most 5,000 epochs, 20% of training rows held out).

## Code commits

- Optimiser: every NEAT run's `backproppop.py` (at its recorded commit) is full-batch Adadelta, lr 1.5,
  no weight decay; the Experiment 3 fine-tune uses the same (`genome_surgery.FT_*`).
- Seeds of one family ran at different commits for Adult (2), Churn (2), Connect-4 cells (2), Connect-4
  lines (3) and the sigmoid synthetic runs (2). `git diff` of the training path
  (`backproppop.py`, `neuralneat.py`, `db/population.py`, `evaluators/`) between them is empty in every case.
- **Dirty trees**: the Cox runs and Connect-4 cells seeds 0 and 1 recorded `code_dirty: true`, so their
  commit does not fully pin the code they ran.
- XOR's runs record no commit; the code read is the one of the commit that last wrote `xor_results.json`.
- The original networks record no seed (`random_seed` null): "1 (n.r.)" in the table.

## ExConNEAT records

- Phase 1 (`results/sweep_p50_m4_e5`): `config.json` holds the arguments of the original 22-minute sweep
  (`patience` 100000, `max_minutes` 22). The run was then resumed to generation 67 and stopped by decision;
  the resumed run's plateau setting is not recorded. The table gives 68 generations, "stopped by decision".
- Phases 2 and 3 record `args.pop` 100, but a warm start takes the checkpoint's population; the config they
  ran with has population 50 (as the book says). The table prints the config's value.
- The structural rates are 4x the code defaults (checked: run rate / `ConvNeatConfig` default = `mut_scale`).
