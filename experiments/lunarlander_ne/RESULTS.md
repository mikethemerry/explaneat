# LunarLander by pure neuroevolution (scouting spike)

Exploratory, not pre-registered. Branch `exp/lunarlander-ne`. Question: do NEAT-evolved LunarLander policies come out
small enough to explain with ExplaNEAT's tools, and does the lineage show stark jumps? No backprop yet (blending in a
value or policy-gradient loss is a later step).

## Setup

- Environment: Gymnasium 1.4 `LunarLander-v3` (optional extra `control`: `gymnasium[box2d]`, `imageio`).
  - 8 inputs: x, y, vx, vy, angle, angular velocity, left leg, right leg.
  - 4 outputs; the action is the argmax: noop, left engine, main engine, right engine.
- NEAT: neat-python 0.92 with `configs/lander.ini`.
  - Population 150, starting fully connected input→output with no hidden nodes.
  - Weights and biases evolve by mutation (rate 0.8, power 0.5).
  - Add connection 0.3, delete connection 0.2, add node 0.1, delete node 0.05.
- Network semantics (`policy.py`) are PropNEAT's: hidden nodes are relu, outputs are linear scores, and only
  enabled connections on an input→output path count.
  - Checked against ExplaNEAT's `NeuralNeat` (sigmoid outputs): it picks the same action on all 1,331 states of 5 test
    episodes.
- Fitness is the mean return over 8 episodes.
  - Generation g uses env seeds 1,000,000 + 8g + i, so every genome in a generation sees the same seeds.
  - Validation seeds start at 2,000,000 and test seeds at 3,000,000; evolution never sees either set.
- Champion: the per-generation best genomes (by training fitness) are rescored on 50 validation seeds, and the best is
  tested on 100 test seeds. "Solved" = test mean ≥ 200.
- Ancestry: `parents.csv` lists every genome's parents (neat-python's reproduction record, fitter parent first).
  - `genomes/gen_XXXX.pkl.gz` holds every generation's whole population (18 MB, not committed; a rerun regenerates it
    exactly).
- Deterministic: `PYTHONHASHSEED=0`, python `random` seeded, seeded env resets, a numpy forward pass and one torch
  thread for the parity check.
  - A 3-generation run with 4 workers and one with 2 workers gave identical `generations.csv` and `parents.csv`.

## Commands (from this folder)

```bash
uv sync --extra control
PYTHONHASHSEED=0 uv run python run.py --seed 0 --generations 400 --episodes 8 --workers 4 > results/seed0/log.txt
PYTHONHASHSEED=0 uv run python evaluate.py results/seed0 --last 400 --val 50 --test 100
PYTHONHASHSEED=0 uv run python lineage.py results/seed0
```

Timing: evolution took 35 min of evaluation on 4 workers with the machine heavily loaded, at 3–15 s per generation.

## Pilot result (seed 0): solved, by a linear policy

Values are from `results/seed0/champion.json`, `test_returns.csv`, `generations.csv`, `val_curve.csv` and
`lineage.csv`.

- Champion: generation 155, genome 22754.
  - Validation mean 274.8 (50 seeds).
  - **Test mean 266.7, sd 31.6, median 269.3, with 99/100 test episodes ≥ 200.**
- Size: **0 hidden nodes and 14 connections**. Angular velocity is not used at all.
- Rollout: `results/seed0/rollout_seed3000000.gif` (return 238).

### What the champion does

It is four linear scores, and the largest one picks the action (weights rounded):

| action | bias | x | y | vx | vy | angle | leg L | leg R |
|---|---|---|---|---|---|---|---|---|
| noop | 3.79 | 1.30 | −1.14 | 3.73 | 9.47 | | 0.07 | |
| main engine | −1.79 | | −4.78 | | −18.17 | | 2.32 | |
| left engine | 0.19 | 7.89 | | 7.33 | | −11.46 | | −3.74 |
| right engine | −0.01 | −5.74 | | | | 18.22 | | |

- **Main engine vs noop is a descent-rate controller.** Main minus noop ≈ −5.6 − 27.6·vy − 3.6·y − 3.7·vx + …
  - So it fires the main engine when the lander is sinking faster than about 0.2–0.3 (in the env's units).
  - The sink rate it tolerates rises with height (about 0.20 at the ground, 0.33 at y = 1).
- **The side engines are an attitude controller with a target angle set by position and drift.**
  - Left engine ≈ −11.5·(angle − 0.69·x − 0.64·vx).
  - Right engine ≈ 18.2·angle − 5.7·x.
  - This has the same form as Gymnasium's hand-written heuristic, where the target angle is a function of x and vx. The
    champion has no angular-velocity damping term.
- **The legs turn it off.**
  - The right leg suppresses the left engine (−3.7).
  - With both legs down, noop wins 1,003 of 1,018 steps (20 test episodes).
  - The left engine never fires once either leg is down.

This looks directly explainable in the explorer: four output units, each a readable linear rule. The explanation is
mostly the pairwise competitions between them.

### Fitness curve

- Training best fitness (8 noisy seeds per generation):
  - −120 at generation 0, then 82 (generation 10), 224 (25), 260 (40), 275 (50).
  - Then a plateau at about 260–280 until generation 399.
  - The largest generation-to-generation rises (+142, +118, +116 at generations 16, 14, 10) all fall in the first 25
    generations.
- The population mean climbs more slowly: −407 at generation 0, −62 at 40, about +40 by 200.
- Champion size shrinks while performance holds: 32 connections at the start, 18 at generation 50, 10 at generation
  399.
  - Selection on return alone favours pruning here, possibly because fewer weights means fewer harmful mutations.
- The validation-scored curve (`val_curve.csv`, every 10th generation's best on 20 seeds) is noisy after generation 50,
  between 137 and 281.
  - So a generation's training-best is a weak pick; rescoring candidates on validation seeds matters.

### Lineage: jumps exist, but each step is not one clean edit

- The champion has 130 ancestors.
- 19 steps raise the 20-seed validation mean by more than 50 over the parent.
- Steps that came down to one edit:
  - Generation 17: deleting one connection (vy→left engine) took the validation mean from −61 to +92.
    - Re-adding it in the next generation dropped it back to −63.
  - Generation 34: adding y→right engine, while dropping a hidden node and six connections, took it from 129 to 244.
  - The last two steps add one connection each: vx→main (+84) and then leg L→main (+51), the final champion.
- But every step also perturbs all the weights (RMS weight change about 0.4–2.8, because the weight mutation rate is
  0.8).
  - The lineage also oscillates: many ancestors fall back by 50–150.
  - Some "jumps" are recoveries, and 20 seeds are themselves noisy.
- So the c4 problem is partly back: structural events are confounded with weight changes.

## Next steps worth trying

1. Lower the weight mutation rate (e.g. 0.1–0.3 per connection, or mutate one connection per child).
   - Lineage steps would then become clean structural or weight events, and jumps attributable.
2. Start from no connections (`initial_connection = unconnected`) to see the minimal policy grow.
   - Optionally add a small size penalty.
3. Run a few more seeds to see whether the linear solution is typical, and whether any seed needs hidden units.
   - Hidden units would make a more interesting explanation case.
4. Load the champion into the explorer.
   - It would need an Experiment/Genome row; nothing was written to any database here.
