# Connect-4 self-play (AlphaZero-style, value only): design

Exploratory, not pre-registered. Study folder `experiments/c4_selfplay/`, branch `exp/c4-selfplay`.
Runs on Mike's Mac (M4 Pro, 14 cores).

## Aim

A strong Connect-4 player taught only by self-play, kept small enough to explain. Strength first,
explainability a close second. The research interest is the **jumps**: steps in the champion's
lineage where playing strength rises sharply, and what structural change caused each one.

Earlier Connect-4 networks were supervised on the UCI 8-ply positions (0.86-0.93 test AUC). Their
leaves read as positional cell gates, not tactics (`experiments/c4_capacity/`,
`experiments/c4_line_features/`). Self-play changes both the data (the network makes it) and the
target (playing strength).

## Network and encoding

- 42 cell inputs from the **side to move's** perspective (+1 mine, -1 theirs, 0 empty), so one
  network plays both colours. Cell order `c_a1 .. c_g6` as in `experiments/complex_real/data.py`.
- One sigmoid output: P(side to move wins), with a draw as 0.5.
- Same PropNEAT path as `c4_capacity`: `BackpropPopulation`, ReLU hidden units, full-batch backprop,
  ancestry tracked (`track_ancestry`).

## Training loop (pure zero: no bootstrap, no external data)

One round:

1. **Self-play.** The current champion plus MCTS plays N games against itself. Variety comes from
   sampling the first few moves in proportion to visit counts (temperature). Each position is labelled
   with the final result from its side to move's point of view (1 win, 0.5 draw, 0 loss).
2. **Replay buffer.** The positions of the last K rounds, deduplicated by position (mean label).
3. **Train.** Continue the **same** NEAT population for G generations on the buffer. Fitness is the
   value fit on the buffer, as in the supervised runs. The population is never restarted, so the
   lineage is continuous across rounds.
4. The new champion plays the next round.

Round 0 uses a blank network: effectively random play guided only by terminal wins in the search.

## MCTS

Value-only, uniform move priors (no policy head). Each simulation: select by UCB, expand one node,
evaluate with the network (a terminal position takes its exact value), back up the value with its sign
flipped at each level. Play the most-visited move (or sample it in the opening during self-play).

To make this fast in Python: a numba bitboard engine, and **batched** MCTS. Hundreds of games run at
once, their leaf positions are collected, and one numpy forward pass evaluates them all. The starting
budget is about 200 simulations per move, tuned from a pilot's timings.

## Fixed yardstick (test only, never trained on)

1. **Perfect play: move accuracy.** A fixed set of positions from every stage of the game (ply 6
   onward), solved once by Pascal Pons' solver (AGPL v3, built locally at
   `~/.explaneat/tools/pons_connect4`, commit `d6ba50d`, not vendored into the repo). The metric is the
   share of the player's moves that keep the game-theoretic result, plus the blunder rate. This is the
   headline strength number.
2. **Perfect play: value.** Value AUC (and win/draw/loss agreement) on **all 67,557 UCI 8-ply
   positions**. Nothing trains on them, so no split is needed. Solver scores are cached to a file; the
   solver agreed with the UCI label on a 50-position check.
3. **Ladder.** Win/draw/loss against random play, alpha-beta at depths 2, 4 and 6, and the perfect
   solver. Both colours, a fixed set of openings, deterministic. Against the solver: can it win as
   first player, and how many moves does it survive as second?

Each champion is scored two ways:

- **net alone**: greedy on its own value, one ply, no search. This measures what the network knows,
  which is what gets explained;
- **net plus MCTS** at a fixed simulation count: playing strength.

Solver labels and yardstick scores are written to csv/json in the study folder.

## Jumps

After a run:

1. Take the final champion's lineage (`explaneat/analysis/lineage.py`).
2. Score every ancestor on the yardstick.
3. Run `detect_jumps` on the yardstick metrics (move accuracy first).
4. At each jump, use `diff_genomes` and `map_genes_to_annotations` to find the structural change.
5. Import the genomes at each jump into the database (as `experiments/c4_capacity/import_run.py`
   does) and explain them in the explorer.

## Provenance and tests

- Fixed seeds everywhere, `PYTHONHASHSEED=0`, fixed torch threads. Every result goes to a file. The
  README lists the regenerating commands in order. New analyses are run twice and checked
  byte-identical.
- Tests:
  - the engine against known positions (wins in every direction, full columns, draws);
  - board-to-move-sequence conversion round-trips;
  - MCTS finds a win in 1 and blocks a loss in 1 with a constant-value net;
  - alpha-beta depth 6 beats random about 100% of the time;
  - the solver wrapper agrees with the UCI labels on a sample;
  - a tiny round runs end to end.

## Out of scope for now

Policy head, ExConNEAT (later: the same loop and yardstick, as a strength comparison), the 5080
rig, the solver's opening book.

## Success

- With search, the champion beats alpha-beta depth 4 more often than not.
- The net alone clearly beats random play, and its move accuracy rises over the run.
- At least one lineage jump can be tied to a structural change we can annotate.
