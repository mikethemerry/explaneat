"""Fixed opponents and the match runner: the yardstick for playing strength.

Every player has ``name`` and ``choose_many(states, rngs) -> int64[n]`` (one column per
state; ``rngs`` is a list of ``np.random.Generator`` aligned with ``states``), plus
``choose(state, rng)`` (one generator, required). A state is a ``GameState`` (pos, mask,
move history), the same object for every player, so the match runner treats all players
alike. States passed to a player are non-terminal with the player to move.

Determinism: players draw randomness for a state only from that state's generator
(RandomPlayer, AlphaBeta's tie-break among equal root scores); SolverPlayer, NetGreedy and
NetMCTS are deterministic and ignore the generators. ``play_games`` draws one seed from the
caller's ``rng`` and spawns one generator per game (``SeedSequence(seed).spawn``, in game
order), so with a seeded ``rng`` the records are identical whatever ``n_parallel`` or
batching is used.

Players
    RandomPlayer     uniform over legal moves.
    AlphaBeta(d)     numba negamax with alpha-beta to ``d`` plies (counting the root move).
                     A win for the side to move scores ``43 - n``, ``n`` = stones on the
                     board after the winning move (faster wins score higher, slower losses
                     less negative); a draw and a depth-limit leaf score 0. Children are
                     searched centre first (3,2,4,1,5,0,6). Every root move gets its exact
                     score (full window), and the move is drawn uniformly among the best.
    SolverPlayer     Pons' perfect solver (``solver.solve(analyse=True)``); the best-scoring
                     column, ties centre-most. Empty history -> column 3 (the known perfect
                     first move; the solver rejects the empty sequence and is very slow on
                     the empty board without an opening book). Scores are cached by
                     sequence in memory. Slow before ~ply 8 (seconds to minutes a move).
    NetGreedy(vf)    one ply: an immediate win (lowest column) if there is one, else the
                     argmax of ``1 - vf(child)`` (ties lowest column). Its only hard-coded
                     rule is the immediate win. A non-finite value raises ValueError.
    NetMCTS(vf, n)   ``mcts.run_batch`` on fresh roots with ``shortcuts=True``, no noise,
                     temperature-0 policy. The shortcuts are hard-coded rules applied at
                     expansion: an immediate win, and a forced block when the opponent has
                     a single immediate threat. NetGreedy has only the immediate-win rule.
"""
import warnings
from dataclasses import dataclass, field

import numpy as np
from numba import njit

import engine as E
import mcts
import solver as S

W = E.W
CENTRE_ORDER = (3, 2, 4, 1, 5, 0, 6)
_ORDER = np.array(CENTRE_ORDER, dtype=np.int64)
_FULL = sum(((1 << E.H) - 1) << (7 * c) for c in range(W))

# Design choice: two-ply openings whose first move is in a central column (0-based 2, 3, 4),
# any reply: 3 x 7 = 21 openings, each played with both colours.
OPENINGS_2PLY = [[a, b] for a in (2, 3, 4) for b in range(W)]


# --- state ----------------------------------------------------------------------------

def _can_play(mask, c):
    return not (mask >> (7 * c + 5)) & 1


def _has_won(p):
    for s in (1, 7, 6, 8):
        m = p & (p >> s)
        if m & (m >> (2 * s)):
            return True
    return False


@dataclass
class GameState:
    """Position (Pons bitboard, Python ints) plus the full move history (0-based)."""
    pos: int
    mask: int
    moves: list = field(default_factory=list)

    @classmethod
    def from_moves(cls, moves):
        s = cls(0, 0, [])
        for c in moves:
            s = s.after(int(c))
        return s

    def legal(self):
        return [c for c in range(W) if _can_play(self.mask, c)]

    def after(self, c):
        c = int(c)
        if not 0 <= c < W or not _can_play(self.mask, c):
            raise ValueError(f"illegal move {c} after {self.moves}")
        return GameState(self.pos ^ self.mask, self.mask | (self.mask + (1 << (7 * c))),
                         self.moves + [c])

    def is_win_move(self, c):
        return _has_won(self.pos | ((self.mask + (1 << (7 * c))) & (63 << (7 * c))))

    def last_mover_won(self):
        return _has_won(self.pos ^ self.mask)

    def is_full(self):
        return self.mask == _FULL


def _check_rngs(states, rngs):
    if rngs is None or len(rngs) != len(states) or not all(
            isinstance(g, np.random.Generator) for g in rngs):
        raise ValueError("rngs must be a list of np.random.Generator aligned with states")


class _Player:
    name = "player"

    def choose(self, state, rng):
        return int(self.choose_many([state], [rng])[0])

    def __repr__(self):
        return self.name


# --- random ---------------------------------------------------------------------------

class RandomPlayer(_Player):
    name = "random"

    def choose_many(self, states, rngs):
        _check_rngs(states, rngs)
        return np.array([int(g.choice(s.legal())) for s, g in zip(states, rngs)],
                        dtype=np.int64)


# --- alpha-beta -----------------------------------------------------------------------

NEG_INF = -1_000_000
ILLEGAL = -2_000_000


# Not cached: numba's on-disk cache of a recursive function segfaulted when reloaded
# (compiled fine in the first process, crashed in the next). Compiling costs ~1 s a process.
@njit
def _negamax(pos, mask, depth, alpha, beta):
    """Score for the side to move, searching ``depth`` more plies (depth >= 1 here)."""
    n = E.n_moves(mask)
    if n == 42:
        return 0
    for i in range(7):
        c = _ORDER[i]
        if E.can_play(mask, c) and E.is_winning_move(pos, mask, c):
            return 43 - (n + 1)
    if depth == 1:
        return 0  # every remaining move is a non-winning leaf
    best = NEG_INF
    for i in range(7):
        c = _ORDER[i]
        if E.can_play(mask, c):
            p2, m2 = E.play(pos, mask, c)
            v = -_negamax(p2, m2, depth - 1, -beta, -alpha)
            if v > best:
                best = v
                if v > alpha:
                    alpha = v
                    if alpha >= beta:
                        break
    return best


@njit
def _root_scores(pos, mask, depth):
    out = np.full(7, ILLEGAL, np.int64)
    n = E.n_moves(mask)
    for c in range(7):
        if not E.can_play(mask, c):
            continue
        if E.is_winning_move(pos, mask, c):
            out[c] = 43 - (n + 1)
        elif depth <= 1 or n + 1 == 42:
            out[c] = 0
        else:
            p2, m2 = E.play(pos, mask, c)
            out[c] = -_negamax(p2, m2, depth - 1, NEG_INF, -NEG_INF)  # full window: exact
    return out


def alphabeta_root_scores(state, depth) -> np.ndarray:
    """Exact depth-limited negamax score of every root move (``ILLEGAL`` for full columns)."""
    return _root_scores(np.uint64(state.pos), np.uint64(state.mask), int(depth))


class AlphaBeta(_Player):
    def __init__(self, depth):
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.depth = int(depth)
        self.name = f"ab{self.depth}"

    def choose_many(self, states, rngs):
        _check_rngs(states, rngs)
        out = np.empty(len(states), np.int64)
        for i, (s, g) in enumerate(zip(states, rngs)):
            sc = alphabeta_root_scores(s, self.depth)
            best = np.flatnonzero(sc == sc.max())
            out[i] = int(best[0]) if len(best) == 1 else int(g.choice(best))
        return out


# --- perfect solver -------------------------------------------------------------------

class SolverPlayer(_Player):
    name = "solver"

    def __init__(self):
        self.cache = {}

    def scores(self, states):
        """Per state, the solver's 7 analyse scores (None for the empty history)."""
        seqs = [S.seq_of_game(s.moves) for s in states]
        todo = list(dict.fromkeys(q for q in seqs if q and q not in self.cache))
        if todo:
            for q, row in zip(todo, S.solve(todo, analyse=True)):
                self.cache[q] = row
        return [self.cache.get(q) for q in seqs]

    def choose_many(self, states, rngs=None):
        out = np.empty(len(states), np.int64)
        for i, (s, row) in enumerate(zip(states, self.scores(states))):
            if row is None:  # empty history
                out[i] = 3
                continue
            legal = [c for c in CENTRE_ORDER if row[c] != -1000]
            out[i] = max(legal, key=lambda c: row[c])  # max keeps the first (centre-most)
        return out


# --- network players ------------------------------------------------------------------

class NetGreedy(_Player):
    def __init__(self, value_fn, name="net_greedy"):
        self.value_fn = value_fn
        self.name = name

    def choose_many(self, states, rngs=None):
        out = np.full(len(states), -1, np.int64)
        need = []  # (state index, column, child)
        for i, s in enumerate(states):
            legal = s.legal()
            wins = [c for c in legal if s.is_win_move(c)]
            if wins:
                out[i] = wins[0]
            else:
                need.extend((i, c, s.after(c)) for c in legal)
        if need:
            X = np.empty((len(need), E.N_CELLS), np.float64)
            for k, (_, _, ch) in enumerate(need):
                E.encode_into(np.uint64(ch.pos), np.uint64(ch.mask), X[k])
            v = np.asarray(self.value_fn(X), dtype=np.float64).reshape(-1)
            if v.shape[0] != len(need) or not np.all(np.isfinite(v)):
                raise ValueError(f"{self.name}: value_fn returned non-finite or wrongly "
                                 f"shaped values (shape {v.shape}, expected {len(need)})")
            score = 1.0 - v
            best = {}
            for (i, c, _), v in zip(need, score):  # columns ascend: strict > keeps lowest
                if i not in best or v > best[i][1]:
                    best[i] = (c, v)
            for i, (c, _) in best.items():
                out[i] = c
        return out


class NetMCTS(_Player):
    def __init__(self, value_fn, n_sims, c_puct=1.5, name=None):
        if n_sims < 2:
            raise ValueError("n_sims must be >= 2 (the first simulation evaluates the root)")
        self.value_fn = value_fn
        self.n_sims = int(n_sims)
        self.c_puct = float(c_puct)
        self.name = name or f"net_mcts{self.n_sims}"

    def choose_many(self, states, rngs=None):
        roots = [mcts.new_root(s.pos, s.mask) for s in states]
        mcts.run_batch(roots, self.value_fn, self.n_sims, rng=None, dirichlet=None,
                       c_puct=self.c_puct, shortcuts=True)
        return np.array([int(np.argmax(mcts.policy(r, 0))) for r in roots], dtype=np.int64)


# --- match runner ---------------------------------------------------------------------

def play_games(games, rng, n_parallel=None):
    """Play ``games`` = [(opening, first_player, second_player)] in lock step.

    Up to ``n_parallel`` games (default all) are live at once; each step every live game
    gets one move, and each player gets one ``choose_many`` call over its games (first
    players before second players, games in input order), with the list of those games'
    generators. One seed is drawn from ``rng`` at the start and spawned into one generator
    per entry of ``games`` (in input order), so outcomes do not depend on ``n_parallel``
    or on how games are batched. Openings that are illegal raise;
    openings already terminal are skipped with a warning. Returns one record per played
    game in input order: ``game`` (index into ``games``), ``opening``, ``moves``, ``n_moves``,
    ``result_first`` (1 / 0.5 / 0).
    """
    seed = int(rng.integers(0, 2**63 - 1))
    game_rngs = [np.random.default_rng(ss)
                 for ss in np.random.SeedSequence(seed).spawn(len(games))]
    todo = []
    for gi, (opening, p1, p2) in enumerate(games):
        s = GameState.from_moves(opening)
        if s.last_mover_won() or s.is_full():
            warnings.warn(f"opening {list(opening)} is already terminal; skipped")
            continue
        todo.append((gi, list(opening), p1, p2, s))
    n_parallel = len(todo) if n_parallel is None else max(1, int(n_parallel))
    records = {}
    queue = list(todo)
    live = []
    while queue or live:
        while queue and len(live) < n_parallel:
            gi, opening, p1, p2, s = queue.pop(0)
            live.append({"gi": gi, "opening": opening, "players": (p1, p2), "state": s})
        # group by player object, first-player slots before second, input order
        groups = {}
        order = []
        for slot in (0, 1):
            for g in live:
                if len(g["state"].moves) % 2 != slot:
                    continue
                pl = g["players"][slot]
                key = id(pl)
                if key not in groups:
                    groups[key] = (pl, [])
                    order.append(key)
                groups[key][1].append(g)
        for key in order:
            pl, gs = groups[key]
            mv = pl.choose_many([g["state"] for g in gs], [game_rngs[g["gi"]] for g in gs])
            for g, c in zip(gs, mv):
                g["state"] = g["state"].after(int(c))
        still = []
        for g in live:
            s = g["state"]
            if s.last_mover_won():
                first_won = len(s.moves) % 2 == 1
                res = 1.0 if first_won else 0.0
            elif s.is_full():
                res = 0.5
            else:
                still.append(g)
                continue
            records[g["gi"]] = {"game": g["gi"], "opening": g["opening"], "moves": list(s.moves),
                                "n_moves": len(s.moves), "result_first": res}
        live = still
    return [records[k] for k in sorted(records)]


def play_match(a, b, openings, rng, n_parallel=None):
    """Each opening twice: a first, then b first. Records add ``a_first`` and
    ``result_a`` (1 / 0.5 / 0 for player a)."""
    games, flags = [], []
    for o in openings:
        for a_first in (True, False):
            games.append((list(o), a, b) if a_first else (list(o), b, a))
            flags.append((tuple(o), a_first))
    out = []
    for r in play_games(games, rng, n_parallel=n_parallel):
        o, a_first = flags[r["game"]]
        res = r["result_first"] if a_first else 1.0 - r["result_first"]
        out.append({"opening": list(o), "a_first": a_first, "result_a": res,
                    "n_moves": r["n_moves"], "moves": r["moves"]})
    return out


def match_score(records) -> float:
    """Mean of ``result_a`` (wins 1, draws 0.5)."""
    return float(np.mean([r["result_a"] for r in records]))
