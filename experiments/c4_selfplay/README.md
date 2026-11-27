# Connect-4 self-play (AlphaZero-style, value only)

Exploratory, not pre-registered. Branch `exp/c4-selfplay`. Design: `docs/plans/2026-11-25-c4-selfplay-design.md`;
implementation plan: `docs/plans/2026-11-25-c4-selfplay.md`.

**Question.** Can a PropNEAT network taught only by self-play become a strong Connect-4 player? And where do the
jumps in its playing strength come from along the champion's lineage? Strength first, explainability a close
second: the network stays small (42 cells in, one sigmoid out) so it can go through the explorer.

## How it works

- **Network:** 42 cells from the side to move's view (+1 mine, -1 theirs) map to P(side to move wins), with draws
  as 0.5. The NEAT config `configs/selfplay.ini` is `c4_capacity`'s base with `pop_size = 150`.
- **Round loop (`run.py`):**
  1. The champion plays N games against itself with batched value-only MCTS (`mcts.py`, `selfplay.py`).
     - The first 8 moves are sampled from the visit counts, then the most-visited move is played.
     - The search has two built-in rules: take an immediate win; block a single immediate threat.
  2. Each position is labelled with the game's result from its side to move.
  3. The replay buffer holds the last K rounds, deduplicated by position with the label averaged.
  4. The **same** population is trained for G generations on the buffer. Fitness is -BCE. An invalid or
     non-finite genome scores just below the worst valid one.
  5. Stagnation history is reset each round, because fitness across buffers isn't comparable.
  6. The round's best genome is the next champion. Round 0 plays with a constant-0.5 value.
- **Saved for lineage analysis:** every generation's genomes, the parent rows and the champions (with their
  generation).
- **Determinism:** runs are deterministic and resume exactly; tests check both.

## Fixed yardstick (`yardstick.py`; test only, never trained on)

- **Perfect play, value:** AUC, win/draw/loss accuracy and Brier on all 67,557 UCI 8-ply positions.
  - Every position was solved with Pascal Pons' solver (AGPL v3, built locally at
    `~/.explaneat/tools/pons_connect4`, commit `d6ba50d`, not vendored).
  - The solver agrees with every UCI label, in both the binary and the 3-class version (`results/yardstick/uci_check.json`).
- **Perfect play, moves:** on 2,000 fixed solved positions (plies 8-30, from seeded games among random,
  alpha-beta 2 and alpha-beta 4), the share of moves that keep the game-theoretic result, the blunder rate,
  and the exactly-optimal share.
  - Measured for the **net alone** (one ply, greedy on its value: what the network knows) and for
    **net + MCTS** (playing strength).
- **Ladder:** match score against random play and alpha-beta at depths 2, 4 and 6, over 21 two-ply openings,
  each played with both colours.
- **Solver rung:** final champions only. The net plays the perfect player from 50 fixed 8-ply openings,
  reporting preservation of the game-theoretic result per outcome class.
- **Ply 8 limit:** the solver has no opening book, so it can't play or label positions before ply 8.

**Yardstick v2 (optional, less noisy; v1 stays the default so existing scores remain comparable):**
- `--positions 10k`: move accuracy on 10,000 solved positions (`positions_10k.csv`, same rules as v1, seed 1,
  built independently of v1 and deduplicated within; SE of a share ~0.004 against ~0.009 on 2,000).
- `--openings 3ply`: the ladder over all 147 three-ply openings with a central first move, both colours = 294
  games per opponent (42 in v1; SE of a match score ~0.025 against ~0.07).
- Every score has a standard error `<metric>_se`, and every row records `positions_set`, `openings_set`,
  `move_n` and `ladder_n_games`. Non-default settings write to their own files (suffix `_<positions>_<openings>`).
- About 85 s per genome at 200 simulations (constant-0.5 value, one core): ~45 s moves, ~40 s ladder.
- `jumps.py --ladder-every k` scores the ladder only every k-th generation (plus round champions and the final
  champion); the other rows get NaN ladder metrics, and ladder jumps are taken between ladder-scored rows.

Reference points, using a constant-0.5 value (all knowledge then comes from the search and its two rules):
- move accuracy is 0.61 with the net alone and 0.80 with MCTS;
- with MCTS it scores 0.57 against alpha-beta depth 4.

## Commands, in order

```bash
# 0. Solver (once): clone PascalPons/connect4 into ~/.explaneat/tools/pons_connect4 and `make c4solver`
# 1. Yardstick data (committed in results/yardstick/; each build is cached and deterministic)
PYTHONHASHSEED=0 uv run python experiments/c4_selfplay/yardstick.py build-uci              # ~50 min, 12 workers
PYTHONHASHSEED=0 uv run python experiments/c4_selfplay/yardstick.py build-positions --n 2000 --seed 0
PYTHONHASHSEED=0 uv run python experiments/c4_selfplay/yardstick.py build-solver-openings --n 50 --seed 0
PYTHONHASHSEED=0 uv run python experiments/c4_selfplay/yardstick.py build-positions --n 10000 --seed 1 --tag 10k --workers 4  # v2, ~10 min
# 2. Pilot (5 rounds, ~6 min)
PYTHONHASHSEED=0 uv run python experiments/c4_selfplay/run.py --name pilot --seed 0 --rounds 5 \
    --games 200 --sims 100 --gens 5 --epochs 20 --buffer 4 --fresh
# 3. Main runs: 3 seeds x 100 rounds in parallel (resumable)
bash experiments/c4_selfplay/run_main.sh
# 4. Score every round champion (add --final-solver for the solver rung on the last one)
PYTHONHASHSEED=0 uv run python experiments/c4_selfplay/yardstick.py champions --name main-s0
# 5. Lineage of the final champion: yardstick per generation (resumable), top-5 jumps per metric
PYTHONHASHSEED=0 uv run python experiments/c4_selfplay/jumps.py --name main-s0 --sims 100 --workers 6
#    v2 yardstick (own output files): add --positions 10k --openings 3ply [--ladder-every 5] (also on step 4)
# Tests
uv run pytest experiments/c4_selfplay/tests -q
```

## Outputs

- `results/<name>/`: `rounds.csv` (per round: buffer, champion key, generation, fitness and size, times,
  game stats), `meta.json`, `config.ini` and `champions_yardstick.csv`.
- `results/<name>/` from `jumps.py`: `lineage_yardstick.csv` (one row per lineage genome: generation, key, round,
  training fitness, size, flags, yardstick metrics), `jumps.json` / `jumps.md` (top-5 rises of `move_acc_net`,
  `uci_auc`, `ladder_ab4_mcts`, `move_acc_mcts`, with the gene diff and boundary vs within round) and
  `lineage_summary.json`. The chain follows the fitter parent at each crossover and the same key across an
  elite copy (as `explaneat.analysis.lineage.get_lineage`, which needs database ids).
- `~/.explaneat/c4_selfplay_runs/<name>/` (not in git): `genomes_rNNN.pkl.gz` (each generation's population),
  `parents.csv.gz`, `champions.pkl.gz` (round -> {genome, generation, key}), `ckpt.pkl.gz`. Logs are in `logs/`.

## Main-run budget (design choices)

- **Per round:** 400 games × 200 simulations, buffer of 5 rounds, 5 generations × 20 epochs.
- **Length:** 100 rounds, so about 500 generations of continuous lineage.
- **Parallelism:** 3 seeds at once, one core each.
- **Why these numbers:** the pilot showed training dominates the time (about 110 s per round at an 18k-position
  buffer, against 7 s for self-play), so the extra budget goes into more and better games per round.
- **Pilot observation:** NEAT collapsed to one species. A species-band threshold controller (as in
  `c4_capacity`) is the fix if the main run stalls.
