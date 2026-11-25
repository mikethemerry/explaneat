import time

import numpy as np
import pytest

import engine as E
import opponents as O
import solver as S
from netfn import ConstValue


def state_of(moves):
    return O.GameState.from_moves(moves)


# --- GameState and openings -----------------------------------------------------------

def test_gamestate_from_moves_matches_engine():
    pos, mask = np.uint64(0), np.uint64(0)
    for c in [3, 3, 2, 4]:
        pos, mask = E.play(pos, mask, c)
    s = state_of([3, 3, 2, 4])
    assert (s.pos, s.mask) == (int(pos), int(mask))
    assert s.moves == [3, 3, 2, 4]


def test_openings_2ply():
    assert len(O.OPENINGS_2PLY) == 21
    assert all(len(o) == 2 and o[0] in (2, 3, 4) for o in O.OPENINGS_2PLY)
    assert len({tuple(o) for o in O.OPENINGS_2PLY}) == 21


# --- AlphaBeta ------------------------------------------------------------------------

def test_alphabeta_takes_immediate_win():
    # x: col 0 three high, o: col 1 three high; x to move wins in col 0.
    s = state_of([0, 1, 0, 1, 0, 1])
    rng = np.random.default_rng(0)
    for _ in range(5):
        assert O.AlphaBeta(2).choose(s, rng) == 0


@pytest.mark.parametrize("moves,block", [
    ([0, 6, 0, 6, 0], 0),   # x threatens col 0 vertically; o (two in col 6) must block
    ([0, 0, 1, 1, 2], 3),   # x holds a1 b1 c1, d1 is the only threat (a is the edge)
])
def test_alphabeta_blocks_immediate_loss(moves, block):
    s = state_of(moves)
    rng = np.random.default_rng(1)
    for _ in range(5):
        assert O.AlphaBeta(2).choose(s, rng) == block


def test_alphabeta_root_scores_prefer_fast_win():
    sc = O.alphabeta_root_scores(state_of([0, 1, 0, 1, 0, 1]), 4)
    assert sc[0] == 43 - 7  # wins at once: 7 stones after the move
    assert all(sc[c] < sc[0] for c in range(1, 7))


def _brute(s, depth):
    """Plain negamax without pruning (same scoring) for checking the jitted search."""
    n = len(s.moves)
    if n == 42 or depth == 0:
        return 0
    for c in s.legal():
        if s.is_win_move(c):
            return 43 - (n + 1)
    if depth == 1:
        return 0
    return max(-_brute(s.after(c), depth - 1) for c in s.legal())


def test_alphabeta_root_scores_match_brute_force():
    rng = np.random.default_rng(2)
    checked = 0
    for _ in range(40):
        s = state_of([])
        for _ in range(int(rng.integers(6, 20))):
            legal = [c for c in s.legal() if not s.is_win_move(c)]
            if not legal:
                break
            s = s.after(int(rng.choice(legal)))
        if s.last_mover_won() or not s.legal():
            continue
        sc = O.alphabeta_root_scores(s, 4)
        for c in range(7):
            if c not in s.legal():
                assert sc[c] == O.ILLEGAL
            elif s.is_win_move(c):
                assert sc[c] == 43 - (len(s.moves) + 1)
            else:
                assert sc[c] == -_brute(s.after(c), 3)
        checked += 1
    assert checked >= 20


def test_alphabeta_vs_random():
    recs = O.play_match(O.AlphaBeta(4), O.RandomPlayer(), O.OPENINGS_2PLY,
                        np.random.default_rng(0))
    assert len(recs) == 42
    assert O.match_score(recs) >= 0.95


def test_alphabeta_depth6_beats_depth2():
    recs = O.play_match(O.AlphaBeta(6), O.AlphaBeta(2), O.OPENINGS_2PLY,
                        np.random.default_rng(0))
    assert O.match_score(recs) > 0.5


def test_alphabeta_depth6_timing():
    ab = O.AlphaBeta(6)
    rng = np.random.default_rng(0)
    ab.choose(state_of([3]), rng)  # compile
    for moves in ([], [3], [3, 3], [3, 2, 4]):
        t = time.perf_counter()
        ab.choose(state_of(moves), rng)
        assert time.perf_counter() - t < 0.1


# --- match runner ---------------------------------------------------------------------

def test_play_match_deterministic_and_seed_dependent():
    a, b = O.RandomPlayer(), O.RandomPlayer()
    r1 = O.play_match(a, b, O.OPENINGS_2PLY[:5], np.random.default_rng(7))
    r2 = O.play_match(a, b, O.OPENINGS_2PLY[:5], np.random.default_rng(7))
    r3 = O.play_match(a, b, O.OPENINGS_2PLY[:5], np.random.default_rng(8))
    assert r1 == r2
    assert [r["moves"] for r in r1] != [r["moves"] for r in r3]


@pytest.mark.parametrize("n_parallel", [1, 5])
def test_play_match_independent_of_n_parallel(n_parallel):
    a, b = O.RandomPlayer(), O.AlphaBeta(2)
    ref = O.play_match(a, b, O.OPENINGS_2PLY[:6], np.random.default_rng(3))
    got = O.play_match(a, b, O.OPENINGS_2PLY[:6], np.random.default_rng(3),
                       n_parallel=n_parallel)
    assert len(ref) == 12
    assert got == ref  # moves, results and bookkeeping identical


def test_choose_requires_generator():
    s = state_of([3])
    with pytest.raises(TypeError):
        O.RandomPlayer().choose(s)
    with pytest.raises(ValueError):
        O.RandomPlayer().choose(s, None)
    with pytest.raises(ValueError):
        O.AlphaBeta(2).choose(s, None)
    with pytest.raises(ValueError):
        O.RandomPlayer().choose_many([s, s], [np.random.default_rng(0)])


class ScriptPlayer:
    """Plays script[len(history)]: both sides read the same script."""

    def __init__(self, script, name="script"):
        self.script = script
        self.name = name

    def choose_many(self, states, rng):
        return np.array([self.script[len(s.moves)] for s in states], dtype=np.int64)


def test_result_bookkeeping_second_player_win():
    # First mover plays 6,6,5,5; second mover plays col 0 four times and wins at ply 8.
    script = [6, 0, 6, 0, 5, 0, 5, 0]
    a, b = ScriptPlayer(script, "a"), ScriptPlayer(script, "b")
    recs = O.play_match(a, b, [[]], np.random.default_rng(0))
    by_first = {r["a_first"]: r for r in recs}
    assert by_first[False]["result_a"] == 1  # b first, a second: a wins
    assert by_first[True]["result_a"] == 0
    for r in recs:
        assert r["n_moves"] == 8
        assert r["moves"] == script
        assert r["opening"] == []


# Full-board games found by random play that avoids every winning move until ply 41
# (a scratch search, seed 0); both are re-checked against the engine below.
FULL_WIN_ON_42 = [4, 1, 4, 4, 6, 4, 6, 6, 0, 0, 3, 3, 3, 5, 2, 1, 4, 3, 6, 0, 4, 5, 1, 0,
                  1, 3, 6, 6, 0, 1, 2, 5, 0, 3, 5, 5, 1, 2, 5, 2, 2, 2]
FULL_DRAW = [3, 3, 6, 1, 3, 0, 2, 6, 4, 2, 6, 4, 6, 0, 2, 6, 4, 3, 2, 3, 3, 2, 4, 0, 2, 4,
             4, 6, 1, 5, 5, 1, 0, 0, 1, 5, 5, 0, 5, 5, 1, 1]


def _engine_wins(moves):
    """Per ply, whether that move wins, from the jitted engine."""
    pos, mask = np.uint64(0), np.uint64(0)
    out = []
    for c in moves:
        assert E.can_play(mask, c)
        out.append(bool(E.is_winning_move(pos, mask, c)))
        pos, mask = E.play(pos, mask, c)
    assert E.n_moves(mask) == 42
    return out


@pytest.mark.parametrize("script,result_first", [(FULL_WIN_ON_42, 0.0), (FULL_DRAW, 0.5)])
def test_game_ending_on_move_42(script, result_first):
    wins = _engine_wins(script)
    assert not any(wins[:41])
    assert wins[41] == (result_first == 0.0)  # the 42nd move is the second player's
    a, b = ScriptPlayer(script, "a"), ScriptPlayer(script, "b")
    recs = O.play_games([([], a, b), (script[:2], b, a)], np.random.default_rng(0),
                        n_parallel=2)
    assert len(recs) == 2
    for r in recs:
        assert r["n_moves"] == 42
        assert r["moves"] == script
        assert r["result_first"] == result_first


def test_terminal_opening_is_skipped():
    with pytest.warns(UserWarning):
        recs = O.play_match(O.RandomPlayer(), O.RandomPlayer(),
                            [[0, 1, 0, 1, 0, 1, 0], [3, 3]], np.random.default_rng(0))
    assert len(recs) == 2
    assert all(r["opening"] == [3, 3] for r in recs)


def test_draw_scores_half():
    recs = [{"result_a": 0.5}, {"result_a": 1}, {"result_a": 0}]
    assert O.match_score(recs) == pytest.approx(0.5)


# --- net players ----------------------------------------------------------------------

def test_netgreedy_takes_immediate_win():
    s = state_of([0, 1, 0, 1, 0, 1])
    assert O.NetGreedy(ConstValue(0.5)).choose(s, None) == 0


def test_netgreedy_argmax_of_one_minus_child_value():
    class ByCol:  # value of the child = 0.1 * (column of the last stone), read from cells
        def __call__(self, X):
            X = np.atleast_2d(X)
            out = []
            for row in X:
                # the child is encoded from the opponent's view: our stones are -1
                n_ours = [(row[c * 6:(c + 1) * 6] == -1).sum() for c in range(7)]
                out.append(0.1 * int(np.argmax(n_ours)))
            return np.array(out)

    # empty board: child after col c has our single stone in col c -> value 0.1c
    # 1 - value is largest for c = 0
    assert O.NetGreedy(ByCol()).choose(state_of([]), None) == 0


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_netgreedy_rejects_non_finite_values(bad):
    class Bad:
        def __call__(self, X):
            v = np.full(np.atleast_2d(X).shape[0], 0.5)
            v[-1] = bad
            return v

    with pytest.raises(ValueError):
        O.NetGreedy(Bad()).choose(state_of([3]), None)


def test_netmcts_takes_immediate_win_and_batches():
    calls = []

    class Counting(ConstValue):
        def __call__(self, X):
            calls.append(np.atleast_2d(X).shape[0])
            return super().__call__(X)

    states = [state_of([0, 1, 0, 1, 0, 1]), state_of([3]), state_of([2, 2])]
    mv = O.NetMCTS(Counting(0.5), n_sims=20).choose_many(states, None)
    assert mv[0] == 0
    assert max(calls) > 1  # one value_fn call covers several trees


# --- SolverPlayer ---------------------------------------------------------------------

needs_solver = pytest.mark.skipif(not S.SOLVER.exists(), reason="Pons solver not built")


def _random_opening(rng, n):
    """A random n-ply legal non-terminal opening (rejection sampling)."""
    while True:
        s = O.GameState.from_moves([])
        ok = True
        for _ in range(n):
            legal = [c for c in range(7) if E.can_play(s.mask, c)]
            c = int(rng.choice(legal))
            if E.is_winning_move(s.pos, s.mask, c):
                ok = False
                break
            s = s.after(c)
        if ok:
            return s.moves


def test_solver_player_empty_history_is_centre():
    assert O.SolverPlayer().choose(state_of([]), None) == 3


@needs_solver
@pytest.mark.slow
def test_solver_player_picks_max_score():
    rng = np.random.default_rng(5)
    sp = O.SolverPlayer()
    states = [state_of(_random_opening(rng, 12)) for _ in range(3)]
    mv = sp.choose_many(states, None)
    sc = S.solve([S.seq_of_game(s.moves) for s in states], analyse=True)
    for m, row in zip(mv, sc):
        assert row[m] == row[row != -1000].max()


@needs_solver
@pytest.mark.slow
def test_solver_player_keeps_first_player_wins():
    rng = np.random.default_rng(11)
    openings = []
    while len(openings) < 2:
        o = _random_opening(rng, 12)  # even ply: the first player is to move
        if S.solve([S.seq_of_game(o)])[0] > 0:
            openings.append(o)
    sp, ab = O.SolverPlayer(), O.AlphaBeta(4)
    recs = O.play_games([(o, sp, ab) for o in openings], np.random.default_rng(0))
    for r in recs:
        assert r["result_first"] == 1
