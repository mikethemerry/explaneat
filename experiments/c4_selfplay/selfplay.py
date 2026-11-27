"""Self-play games and the replay buffer (value-only, pure zero).

``play_games`` plays ``n_games`` games of the current value function against itself in
lock step. Each step every live game gets a FRESH root (no tree reuse) and one
``mcts.run_batch`` call searches all of them, so the value function sees one batch per
simulation step across all games. For a game's first ``temp_moves`` moves the move is
drawn from the visit-count policy at temperature 1 (``rng``); afterwards it is the
argmax (temperature 0, ties lowest column). With ``temp_moves=0`` and no Dirichlet noise
the whole run is deterministic and every game is the same game.

Training data: one row per move chosen, the position BEFORE the move, encoded from the
side to move's view (``engine.encode``); terminal positions are never recorded. Labels
are the game's outcome z from that row's side to move: 1 win, 0 loss, 0.5 draw
(``label_positions``).

Search value targets: for every recorded row ``play_games`` also records q, the MCTS root
value after that step's simulations, from the row's side to move's view: P(side to move
wins). ``mcts`` stores a node's W from its PARENT's mover's view (``_backup`` adds
``1 - v`` where v is the node's own side-to-move value), so the root's own value is
``q = 1 - root.W / root.N`` (an immediate win at the root gives exactly 1.0). The label is
``value_mix * z + (1 - value_mix) * q``; the default ``value_mix=1.0`` is the pure outcome
label (bit-identical to before: q is read from the tree, no RNG is drawn for it).
``return_q=True`` also returns the q vector, aligned with the rows.

``split_holdout`` splits rows into train / held-out by unique position (every copy of a
held-out position is held out), e.g. to pick a champion on positions from the newest
round that it was not trained on.

``ReplayBuffer`` keeps the last ``max_rounds`` rounds added and serves them deduplicated:
identical encoded positions become one row whose label is the mean over all their
occurrences in those rounds. Rows come out sorted by the position's int8 bytes
(``np.unique`` order), so the order does not depend on the order positions were added.
"""
from collections import deque

import numpy as np

import engine as E
import mcts
from opponents import GameState

W = E.W


def label_positions(n_positions, result_first) -> np.ndarray:
    """Labels for positions 0..n-1 of one game; position i has the first player to move
    iff i is even. ``result_first``: 1 / 0.5 / 0 for the first player."""
    r = float(result_first)
    lab = np.where(np.arange(n_positions) % 2 == 0, r, 1.0 - r)
    return lab.astype(np.float64)


def play_games(value_fn, n_games, n_sims, rng, temp_moves=8, dirichlet=None, c_puct=1.5,
               value_mix=1.0, return_q=False):
    """Self-play ``n_games`` games in lock step from the empty board.

    Returns ``(X [m, 42] float64, y [m] float64, games)``; rows are grouped by game (in
    game order) and in move order within a game. ``games[i]``: ``moves`` (0-based),
    ``n_moves``, ``result_first`` (1 / 0.5 / 0). Deterministic given the ``rng`` state.

    ``y = value_mix * z + (1 - value_mix) * q`` (z outcome, q root search value, both
    from the row's side to move; see the module docstring). ``value_mix=1.0`` (default):
    outcome labels exactly as before. ``return_q=True``: returns ``(X, y, games, Q)``
    with ``Q [m] float64`` aligned with the rows.
    """
    if n_sims < 2:
        raise ValueError("n_sims must be >= 2 (the first simulation evaluates the root)")
    value_mix = float(value_mix)
    if not 0.0 <= value_mix <= 1.0:
        raise ValueError("value_mix must be in [0, 1]")
    states = [GameState(0, 0, []) for _ in range(n_games)]
    rows = [[] for _ in range(n_games)]
    qs = [[] for _ in range(n_games)]
    results = [None] * n_games
    live = list(range(n_games))
    while live:
        roots = [mcts.new_root(states[g].pos, states[g].mask) for g in live]
        mcts.run_batch(roots, value_fn, n_sims, rng=rng, dirichlet=dirichlet, c_puct=c_puct)
        still = []
        for g, root in zip(live, roots):
            s = states[g]
            rows[g].append(E.encode(np.uint64(s.pos), np.uint64(s.mask)))
            qs[g].append(root_value(root))
            if len(s.moves) < temp_moves:
                p = mcts.policy(root, 1.0)
                c = int(rng.choice(W, p=p))
            else:
                c = int(np.argmax(mcts.policy(root, 0)))
            s = s.after(c)
            states[g] = s
            if s.last_mover_won():
                results[g] = 1.0 if len(s.moves) % 2 == 1 else 0.0
            elif s.is_full():
                results[g] = 0.5
            else:
                still.append(g)
        live = still

    games, Xs, ys = [], [], []
    for g in range(n_games):
        mv = states[g].moves
        games.append({"moves": list(mv), "n_moves": len(mv), "result_first": results[g]})
        Xs.append(np.asarray(rows[g], dtype=np.float64))
        ys.append(label_positions(len(mv), results[g]))
    X = np.vstack(Xs) if Xs else np.empty((0, E.N_CELLS), np.float64)
    y = np.concatenate(ys) if ys else np.empty(0, np.float64)
    Q = np.asarray([v for q in qs for v in q], dtype=np.float64)
    if value_mix != 1.0:
        y = value_mix * y + (1.0 - value_mix) * Q
    if return_q:
        return X, y, games, Q
    return X, y, games


def root_value(root) -> float:
    """The searched root's value for ITS side to move: 1 - W/N (``mcts`` keeps W from the
    parent's mover's view). Needs at least one simulation."""
    if root.N <= 0:
        raise ValueError("root has no visits")
    return 1.0 - root.W / root.N


def split_holdout(X, y, frac, rng):
    """Split rows by unique position: ``round(frac * u)`` of the u unique positions (and
    every row holding one) are held out. Returns ``(X_train, y_train, X_hold, y_hold)``,
    each part in the input's row order. Which positions are held out depends only on the
    set of positions and the ``rng`` state (unique positions are taken in int8-byte order,
    then permuted by ``rng``), not on the row order."""
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    if X.ndim != 2 or X.shape[0] != y.shape[0]:
        raise ValueError("X must be [n, d] with len(y) == n")
    if not 0.0 <= frac <= 1.0:
        raise ValueError("frac must be in [0, 1]")
    Xi = np.ascontiguousarray(X, dtype=np.int8)
    keys = Xi.view(np.dtype((np.void, Xi.shape[1]))).reshape(-1)
    _, inv = np.unique(keys, return_inverse=True)
    inv = inv.reshape(-1)
    u = int(inv.max()) + 1 if inv.size else 0
    n_hold = int(round(frac * u))
    hold_ids = np.zeros(u, bool)
    hold_ids[rng.permutation(u)[:n_hold]] = True
    hold = hold_ids[inv]
    return X[~hold], y[~hold], X[hold], y[hold]


class ReplayBuffer:
    """Positions of the last ``max_rounds`` rounds added (oldest dropped first)."""

    def __init__(self, max_rounds):
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        self.max_rounds = int(max_rounds)
        self._rounds = deque(maxlen=self.max_rounds)  # (round_idx, X int8, y float64)
        self._cache = None

    def add(self, round_idx, X, y):
        X = np.asarray(X)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if X.ndim != 2 or X.shape[1] != E.N_CELLS or X.shape[0] != y.shape[0]:
            raise ValueError(f"X must be [n, {E.N_CELLS}] with len(y) == n")
        self._rounds.append((int(round_idx), np.ascontiguousarray(X, dtype=np.int8), y))
        self._cache = None

    def _dedup(self):
        if self._cache is None:
            if not self._rounds:
                self._cache = (np.empty((0, E.N_CELLS), np.float64), np.empty(0, np.float64))
                return self._cache
            Xi = np.ascontiguousarray(np.vstack([r[1] for r in self._rounds]))
            y = np.concatenate([r[2] for r in self._rounds])
            keys = Xi.view(np.dtype((np.void, E.N_CELLS))).reshape(-1)
            _, first, inv = np.unique(keys, return_index=True, return_inverse=True)
            inv = inv.reshape(-1)
            sums = np.bincount(inv, weights=y, minlength=len(first))
            counts = np.bincount(inv, minlength=len(first))
            self._cache = (Xi[first].astype(np.float64), sums / counts)
        return self._cache

    def data(self):
        """(X [u, 42] float64, y [u] float64): unique positions sorted by their int8
        bytes, label = mean over occurrences in the rounds held."""
        X, y = self._dedup()
        return X.copy(), y.copy()

    def __len__(self):
        return self._dedup()[0].shape[0]

    def stats(self):
        return {"rounds": len(self._rounds),
                "raw": int(sum(r[1].shape[0] for r in self._rounds)),
                "unique": len(self)}
