"""The fixed yardstick: test-only measures of playing strength (never trained on).

Builds (deterministic; outputs under ``<results-dir>/yardstick/``, default
``experiments/c4_selfplay/results``):

    uv run python experiments/c4_selfplay/yardstick.py build-uci            # ~35-60 min
    uv run python experiments/c4_selfplay/yardstick.py build-positions --n 2000 --seed 0
    uv run python experiments/c4_selfplay/yardstick.py build-solver-openings --n 50 --seed 0
    uv run python experiments/c4_selfplay/yardstick.py champions --name s0 [--final-solver]

``champions`` reads ``<runs-dir>/<name>/champions.pkl.gz`` ({round: {"genome", "generation",
"key"}}, written by run.py) and writes ``<results-dir>/<name>/champions_yardstick.csv``: one
row per round, columns round, generation, key, then the scores in sorted order.

``build-uci``  -> ``uci_solved.csv`` (``row,score``) and ``uci_check.json``. Every UCI
    Connect-4 position (PMLB ``connect_4``, 67,557 8-ply positions, x = first player to
    move) solved by Pons' solver; the score is from the side to move (= x). Checked row by
    row against the dataset's 3-class label (win <-> score > 0, draw <-> 0, loss <-> < 0);
    any mismatch fails the build. ``--limit N`` takes the first N rows of a fixed seeded
    permutation (testing only; recorded in ``uci_check.json``). Cache:
    ``uci_solve_cache.csv`` (seq-keyed).
``build-positions`` -> ``positions.csv`` (``seq,ply,s0..s6``): positions from games between
    RandomPlayer, AlphaBeta(2), AlphaBeta(4) (all 9 ordered pairings, round-robin over
    games, from the empty board). One position per game, its ply drawn uniformly from
    8..min(30, game length - 1) (non-terminal by construction), deduplicated by position
    (bitboard pair; the first move order seen is kept), until ``n``. Analyse scores (side
    to move) from ``solve_cached`` (cache ``positions_solve_cache.csv``).
``build-solver-openings`` -> ``solver_openings.csv`` (``row,seq,label,score``): 8-ply
    openings for games against the perfect player, sampled from the UCI rows balanced over
    the three outcome classes. Design choice: the solver cannot play before ply 8 without
    an opening book (ply 7 ~1.4 s a position, ply 5 ~31 s, ply <= 4 impractical).

Scores (``score_value_fn`` / ``score_genome``):
    uci_auc, uci_wdl_acc, uci_brier, uci_n     value on every row of uci_solved.csv vs the
        solver (AUC vs score > 0; 3-way accuracy with thresholds v < 1/3 loss, v > 2/3 win;
        Brier vs targets 1 / 0.5 / 0).
    move_acc_<m>, blunder_<m>, move_opt_<m>    on positions.csv for m in net (NetGreedy),
        mcts (NetMCTS at ``sims``): correct iff sign(s_chosen) == sign(s_best) (a lost
        position: any legal move is correct), blunder iff sign(s_chosen) < sign(s_best),
        opt iff s_chosen == s_best (exactly optimal, secondary).
    ladder_<opp>_<m> (+ _w, _d, _l)            ``match_score`` of the net player against
        random, ab2, ab4, ab6 over OPENINGS_2PLY, each opening with both colours.
    solver rung ("solver" in parts)            see ``solver_rung``.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import netfn  # noqa: E402  (first: forces EXPLANEAT_DEVICE=cpu)

import argparse  # noqa: E402
import csv  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import engine as E  # noqa: E402
import opponents as O  # noqa: E402
import solver as S  # noqa: E402

DEFAULT_RESULTS = HERE / "results"
N_UCI = 67557
UCI_PERM_SEED = 0            # design choice: the permutation behind --limit
POS_PLY_MIN, POS_PLY_MAX = 8, 30
POS_BATCH = 90               # games per play_games call (a constant: keeps builds deterministic)
LADDER_OPPS = ("random", "ab2", "ab4", "ab6")
MODES = ("net", "mcts")
CLASS_NAMES = ("loss", "draw", "win")   # index = sign(score) + 1


def ydir(results_dir) -> Path:
    return Path(results_dir) / "yardstick"


def _sign(x):
    return np.sign(np.asarray(x, dtype=np.int64)).astype(np.int64)


def _chunk(n, workers):
    """Chunk size that keeps every worker busy on small builds (results do not depend on it)."""
    return int(max(1, min(64, -(-n // max(1, workers)))))


def _write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    tmp.replace(path)


def _read_csv(path):
    with Path(path).open(newline="") as f:
        r = csv.reader(f)
        header = next(r)
        return header, [row for row in r if row]


# --- UCI ------------------------------------------------------------------------------

def load_uci():
    """(X [N, 42] x=+1 first player to move, y [N] 1 iff x wins, cls [N] 0 loss / 1 draw /
    2 win) from PMLB ``connect_4`` (target codes 2 = win, 0 = draw, 1 = loss)."""
    sys.path.insert(0, str(HERE.parent / "complex_real"))
    import data as D  # experiments/complex_real/data.py

    X, y, _, _ = D.load_task("c4")
    t = D._fetch("connect_4")["target"].to_numpy()
    cls = np.select([t == 2, t == 0, t == 1], [2, 1, 0], -1).astype(np.int64)
    if (cls < 0).any():
        raise ValueError(f"unexpected PMLB connect_4 target codes {sorted(set(t.tolist()))}")
    if not np.array_equal(y, (cls == 2).astype(np.int64)):
        raise ValueError("load_task's binary label disagrees with the raw PMLB target")
    return X, y, cls


def uci_rows(n, limit=None):
    if limit is None:
        return np.arange(n)
    perm = np.random.default_rng(UCI_PERM_SEED).permutation(n)
    return np.sort(perm[:int(limit)])


def build_uci(results_dir=DEFAULT_RESULTS, limit=None, workers=12, log=print):
    t0 = time.time()
    out = ydir(results_dir)
    X, y, cls = load_uci()
    rows = uci_rows(len(X), limit)
    seqs = [S.board_to_seq(X[i]) for i in rows]
    if any(len(s) != 8 for s in seqs):
        raise ValueError("a UCI position is not 8-ply")
    log(f"build-uci: {len(rows)} positions -> sequences in {time.time() - t0:.1f} s; solving")
    sc = S.solve_cached(seqs, out / "uci_solve_cache.csv", workers=workers,
                        chunk=_chunk(len(seqs), workers))
    pred = _sign(sc) + 1
    counts = {"n": int(len(rows)), "limit": None if limit is None else int(limit),
              "perm_seed": UCI_PERM_SEED if limit is not None else None}
    bin_mismatch = int(((sc > 0) != (y[rows] == 1)).sum())
    counts["binary_mismatch"] = bin_mismatch
    confusion = {}
    for a in range(3):
        for b in range(3):
            confusion[f"label_{CLASS_NAMES[a]}_solver_{CLASS_NAMES[b]}"] = int(
                ((cls[rows] == a) & (pred == b)).sum())
    counts["confusion"] = confusion
    counts["class_mismatch"] = int((cls[rows] != pred).sum())
    counts["label_counts"] = {CLASS_NAMES[a]: int((cls[rows] == a).sum()) for a in range(3)}
    _write_csv(out / "uci_solved.csv", ["row", "score"],
               [[int(i), int(s)] for i, s in zip(rows, sc)])
    (out / "uci_check.json").write_text(json.dumps(counts, indent=2, sort_keys=True) + "\n")
    if bin_mismatch or counts["class_mismatch"]:
        raise RuntimeError(f"UCI labels disagree with the solver: {bin_mismatch} binary and "
                           f"{counts['class_mismatch']} 3-class mismatches; see "
                           f"{out / 'uci_check.json'}")
    log(f"build-uci: {len(rows)} rows agree with the labels ({time.time() - t0:.1f} s)")
    return counts


def load_uci_solved(results_dir=DEFAULT_RESULTS):
    """(rows, X [n, 42] side-to-move encoding, scores [n]) for the rows in uci_solved.csv."""
    _, body = _read_csv(ydir(results_dir) / "uci_solved.csv")
    rows = np.array([int(r[0]) for r in body], dtype=np.int64)
    sc = np.array([int(r[1]) for r in body], dtype=np.int64)
    X, _, _ = load_uci()
    return rows, X[rows], sc


# --- positions ------------------------------------------------------------------------

def _pos_players():
    return [O.RandomPlayer(), O.AlphaBeta(2), O.AlphaBeta(4)]


def generate_positions(n, seed=0):
    """[(seq, ply)] for ``n`` unique positions (see the module docstring)."""
    players = _pos_players()
    pairings = [(a, b) for a in players for b in players]   # 9 ordered pairings
    seen = set()
    out = []
    g = 0
    batch = 0
    while len(out) < n:
        games = [([], *pairings[(g + k) % len(pairings)]) for k in range(POS_BATCH)]
        recs = O.play_games(games, np.random.default_rng([seed, 0, batch]))
        for r in recs:
            gi = g + r["game"]
            moves = r["moves"]
            hi = min(POS_PLY_MAX, len(moves) - 1)
            if hi < POS_PLY_MIN:
                continue
            ply = int(np.random.default_rng([seed, 1, gi]).integers(POS_PLY_MIN, hi + 1))
            st = O.GameState.from_moves(moves[:ply])
            key = (st.pos, st.mask)
            if key in seen:
                continue
            seen.add(key)
            out.append((S.seq_of_game(moves[:ply]), ply))
            if len(out) == n:
                break
        g += POS_BATCH
        batch += 1
    return out


def build_positions(results_dir=DEFAULT_RESULTS, n=2000, seed=0, workers=12, log=print):
    t0 = time.time()
    out = ydir(results_dir)
    pos = generate_positions(n, seed)
    log(f"build-positions: {len(pos)} positions generated in {time.time() - t0:.1f} s; solving")
    seqs = [s for s, _ in pos]
    a = S.solve_cached(seqs, out / "positions_solve_cache.csv", analyse=True, workers=workers,
                       chunk=_chunk(len(seqs), workers))
    _write_csv(out / "positions.csv", ["seq", "ply"] + [f"s{k}" for k in range(7)],
               [[s, p] + [int(x) for x in row] for (s, p), row in zip(pos, a)])
    log(f"build-positions: wrote {out / 'positions.csv'} ({time.time() - t0:.1f} s)")
    return out / "positions.csv"


def load_positions(results_dir=DEFAULT_RESULTS):
    """(states [GameState], ply int[n], scores int[n, 7]) from positions.csv."""
    _, body = _read_csv(ydir(results_dir) / "positions.csv")
    states = [O.GameState.from_moves([int(ch) - 1 for ch in r[0]]) for r in body]
    ply = np.array([int(r[1]) for r in body], dtype=np.int64)
    sc = np.array([[int(x) for x in r[2:9]] for r in body], dtype=np.int64)
    return states, ply, sc


# --- solver openings ------------------------------------------------------------------

def build_solver_openings(results_dir=DEFAULT_RESULTS, n=50, seed=0, workers=12, log=print):
    """Balanced over loss / draw / win (labels from PMLB, checked by the solver)."""
    out = ydir(results_dir)
    X, _, cls = load_uci()
    rng = np.random.default_rng(seed)
    want = [n // 3 + (1 if k < n % 3 else 0) for k in range(3)]   # win, draw, loss order
    order = (2, 1, 0)
    picked = []
    spare = 0
    for k, c in enumerate(order):
        pool = np.flatnonzero(cls == c)
        m = min(len(pool), want[k] + spare)
        spare = want[k] + spare - m
        picked.extend(int(i) for i in rng.choice(pool, m, replace=False))
    if spare:
        raise ValueError(f"not enough UCI rows for {n} openings")
    picked.sort()
    seqs = [S.board_to_seq(X[i]) for i in picked]
    sc = S.solve_cached(seqs, out / "uci_solve_cache.csv", workers=workers,
                        chunk=_chunk(len(seqs), workers))
    bad = [i for i, s in zip(picked, sc) if int(np.sign(s)) + 1 != cls[i]]
    if bad:
        raise RuntimeError(f"solver disagrees with the UCI label on rows {bad}")
    _write_csv(out / "solver_openings.csv", ["row", "seq", "label", "score"],
               [[i, q, CLASS_NAMES[cls[i]], int(s)] for i, q, s in zip(picked, seqs, sc)])
    log(f"build-solver-openings: wrote {out / 'solver_openings.csv'} ({len(picked)} openings)")
    return out / "solver_openings.csv"


def load_solver_openings(results_dir=DEFAULT_RESULTS):
    _, body = _read_csv(ydir(results_dir) / "solver_openings.csv")
    return [([int(ch) - 1 for ch in r[1]], int(r[3])) for r in body]


# --- scoring --------------------------------------------------------------------------

def make_player(value_fn, mode, sims):
    if mode == "net":
        return O.NetGreedy(value_fn)
    if mode == "mcts":
        return O.NetMCTS(value_fn, sims)
    raise ValueError(mode)


def make_opponent(name):
    if name == "random":
        return O.RandomPlayer()
    if name.startswith("ab"):
        return O.AlphaBeta(int(name[2:]))
    if name == "solver":
        return O.SolverPlayer()
    raise ValueError(name)


def uci_metrics(value_fn, X, scores):
    from sklearn.metrics import roc_auc_score

    v = np.asarray(value_fn(X), dtype=np.float64).reshape(-1)
    if v.shape[0] != len(X) or not np.all(np.isfinite(v)):
        raise ValueError("value_fn returned non-finite or wrongly shaped values")
    win = scores > 0
    true = _sign(scores) + 1                       # 0 loss, 1 draw, 2 win
    pred = np.where(v < 1 / 3, 0, np.where(v > 2 / 3, 2, 1))
    target = true / 2.0
    auc = float(roc_auc_score(win, v)) if 0 < win.sum() < len(win) else float("nan")
    return {"uci_auc": auc, "uci_wdl_acc": float((pred == true).mean()),
            "uci_brier": float(np.mean((v - target) ** 2)), "uci_n": int(len(X))}


def move_metrics(player, states, scores, suffix):
    """Move accuracy of ``player`` (any player: choose_many) on solved positions."""
    cols = np.asarray(player.choose_many(states, [np.random.default_rng([0, i])
                                                  for i in range(len(states))]))
    sc = scores[np.arange(len(states)), cols]
    if (sc == -1000).any():
        raise ValueError(f"{player.name} chose a full column")
    best = np.where(scores == -1000, np.iinfo(np.int64).min, scores).max(axis=1)
    s_c, s_b = _sign(sc), _sign(best)
    return {f"move_acc_{suffix}": float((s_c == s_b).mean()),
            f"blunder_{suffix}": float((s_c < s_b).mean()),
            f"move_opt_{suffix}": float((sc == best).mean())}


def ladder_metrics(player, opp_name, mode, seed):
    recs = O.play_match(player, make_opponent(opp_name), O.OPENINGS_2PLY,
                        rng=np.random.default_rng(seed))
    res = np.array([r["result_a"] for r in recs])
    k = f"ladder_{opp_name}_{mode}"
    return {k: O.match_score(recs), f"{k}_w": int((res == 1.0).sum()),
            f"{k}_d": int((res == 0.5).sum()), f"{k}_l": int((res == 0.0).sum())}


def solver_rung(player, openings, seed=0):
    """Net vs the perfect player from 8-ply openings (first player to move at ply 8).

    Each opening twice: the net as first player (moves at ply 8), and the solver as first
    player (net second). Theory = the opening's game-theoretic result for the net's side
    (1 / 0.5 / 0 from the solver score's sign). Reports
      solver_first_score    net's mean result as first player
      solver_first_theory   mean theoretical result for the first player (the ceiling)
      solver_first_win_kept share of first-player-won openings the net wins (nan if none)
      solver_second_moves   mean number of moves the net makes as second player
      solver_preserve       share of all games where the net reaches its side's theory
    """
    solver = O.SolverPlayer()
    games, theory, net_first = [], [], []
    for moves, score in openings:
        if len(moves) != 8:
            raise ValueError("solver openings must be 8-ply (first player to move)")
        t_first = (np.sign(score) + 1) / 2.0
        games.append((moves, player, solver))
        theory.append(t_first)
        net_first.append(True)
        games.append((moves, solver, player))
        theory.append(1.0 - t_first)
        net_first.append(False)
    recs = O.play_games(games, np.random.default_rng(seed))
    if len(recs) != len(games):
        raise RuntimeError("a solver opening was terminal")
    res = np.array([r["result_first"] if net_first[r["game"]] else 1.0 - r["result_first"]
                    for r in recs])
    th = np.array(theory)
    nf = np.array(net_first)
    won_first = nf & (th == 1.0)
    second_moves = [sum(1 for k in range(8, r["n_moves"]) if k % 2 == 1)
                    for r in recs if not net_first[r["game"]]]
    return {"solver_first_score": float(res[nf].mean()),
            "solver_first_theory": float(th[nf].mean()),
            "solver_first_win_kept": (float(res[won_first].mean()) if won_first.any()
                                      else float("nan")),
            "solver_second_moves": float(np.mean(second_moves)),
            "solver_preserve": float((res == th).mean()),
            "solver_n_openings": int(len(openings))}


class _Data:
    """Yardstick files, loaded once per results dir."""
    _cache = {}

    @classmethod
    def get(cls, results_dir, what):
        key = (str(Path(results_dir).resolve()), what)
        if key not in cls._cache:
            cls._cache[key] = {"uci": lambda: load_uci_solved(results_dir),
                               "moves": lambda: load_positions(results_dir),
                               "solver": lambda: load_solver_openings(results_dir)}[what]()
        return cls._cache[key]


def score_value_fn(value_fn, sims=200, parts=("uci", "moves", "ladder"), seed=0,
                   results_dir=DEFAULT_RESULTS, ladder_opps=LADDER_OPPS, modes=MODES):
    """All yardstick numbers for one value function (dict, sorted keys)."""
    out = {}
    for p in parts:
        if p not in ("uci", "moves", "ladder", "solver"):
            raise ValueError(f"unknown part {p!r}")
    if "uci" in parts:
        _, X, sc = _Data.get(results_dir, "uci")
        out.update(uci_metrics(value_fn, X, sc))
    if "moves" in parts:
        states, _, sc = _Data.get(results_dir, "moves")
        for m in modes:
            out.update(move_metrics(make_player(value_fn, m, sims), states, sc, m))
    if "ladder" in parts:
        for m in modes:
            pl = make_player(value_fn, m, sims)
            for opp in ladder_opps:
                out.update(ladder_metrics(pl, opp, m, seed))
    if "solver" in parts:
        out.update(solver_rung(make_player(value_fn, "mcts", sims),
                               _Data.get(results_dir, "solver"), seed))
    return dict(sorted(out.items()))


def score_genome(genome, config, sims=200, parts=("uci", "moves", "ladder"), seed=0, **kw):
    return score_value_fn(netfn.ValueFn(genome, config), sims=sims, parts=parts, seed=seed, **kw)


# --- champions ------------------------------------------------------------------------

def default_runs_dir():
    home = Path(os.environ.get("EXPLANEAT_HOME", Path.home() / ".explaneat")).expanduser()
    return home / "c4_selfplay_runs"


def score_champions(name, results_dir=DEFAULT_RESULTS, runs_dir=None, sims=200, seed=0,
                    final_solver=False, allow_partial=False, log=print):
    import torch

    torch.set_num_threads(1)
    runs_dir = Path(runs_dir) if runs_dir else default_runs_dir()
    check = json.loads((ydir(results_dir) / "uci_check.json").read_text())
    if check.get("limit") is not None and not allow_partial:
        raise RuntimeError("uci_solved.csv is a --limit build; run the full build-uci "
                           "(or pass --allow-partial for testing)")
    with gzip.open(runs_dir / name / "champions.pkl.gz", "rb") as f:
        champions = pickle.load(f)
    cfg_path = Path(results_dir) / name / "config.ini"
    config = netfn.load_config(cfg_path if cfg_path.exists() else netfn.DEFAULT_CONFIG)
    rounds = sorted(champions)
    rows = []
    for r in rounds:
        t0 = time.time()
        parts = ("uci", "moves", "ladder") + (("solver",) if final_solver and r == rounds[-1]
                                              else ())
        entry = champions[r]   # run.py: {"genome": DefaultGenome, "generation": int, "key": int}
        d = score_genome(entry["genome"], config, sims=sims, parts=parts, seed=seed,
                         results_dir=results_dir)
        d["round"] = int(r)
        d["generation"] = int(entry["generation"])
        d["key"] = int(entry["key"])
        rows.append(d)
        log(f"champions {name}: round {r} scored in {time.time() - t0:.1f} s")
    ids = ["round", "generation", "key"]
    keys = sorted({k for d in rows for k in d} - set(ids))
    header = ids + keys
    path = Path(results_dir) / name / "champions_yardstick.csv"
    _write_csv(path, header, [[d.get(k, "") for k in header] for d in rows])
    log(f"wrote {path}")
    return path


# --- CLI ------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("build-uci")
    a.add_argument("--limit", type=int, default=None)
    a.add_argument("--workers", type=int, default=12)
    a = sub.add_parser("build-positions")
    a.add_argument("--n", type=int, default=2000)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--workers", type=int, default=12)
    a = sub.add_parser("build-solver-openings")
    a.add_argument("--n", type=int, default=50)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--workers", type=int, default=12)
    a = sub.add_parser("champions")
    a.add_argument("--name", required=True)
    a.add_argument("--runs-dir", default=None)
    a.add_argument("--sims", type=int, default=200)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--final-solver", action="store_true")
    a.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args(argv)
    rd = args.results_dir
    if args.cmd == "build-uci":
        build_uci(rd, limit=args.limit, workers=args.workers)
    elif args.cmd == "build-positions":
        build_positions(rd, n=args.n, seed=args.seed, workers=args.workers)
    elif args.cmd == "build-solver-openings":
        build_solver_openings(rd, n=args.n, seed=args.seed, workers=args.workers)
    elif args.cmd == "champions":
        score_champions(args.name, rd, args.runs_dir, sims=args.sims, seed=args.seed,
                        final_solver=args.final_solver, allow_partial=args.allow_partial)


if __name__ == "__main__":
    main()
