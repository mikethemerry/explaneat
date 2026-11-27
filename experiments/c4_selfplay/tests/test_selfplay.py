import time

import numpy as np
import pytest

import engine as E
import opponents as O
import selfplay as SP
from netfn import ConstValue


def run(seed=0, n_games=6, n_sims=20, temp_moves=8, dirichlet=None):
    rng = np.random.default_rng(seed)
    return SP.play_games(ConstValue(0.5), n_games, n_sims, rng, temp_moves=temp_moves,
                         dirichlet=dirichlet)


# --- labelling ------------------------------------------------------------------------

def test_label_positions_first_wins():
    np.testing.assert_array_equal(SP.label_positions(5, 1.0), [1, 0, 1, 0, 1])


def test_label_positions_second_wins():
    np.testing.assert_array_equal(SP.label_positions(4, 0.0), [0, 1, 0, 1])


def test_label_positions_draw():
    np.testing.assert_array_equal(SP.label_positions(42, 0.5), np.full(42, 0.5))


# --- play_games -----------------------------------------------------------------------

def test_shapes_and_records():
    X, y, games = run()
    assert X.dtype == np.float64 and y.dtype == np.float64
    assert X.shape == (sum(g["n_moves"] for g in games), E.N_CELLS)
    assert y.shape == (X.shape[0],)
    assert len(games) == 6
    for g in games:
        assert g["n_moves"] == len(g["moves"])
        s = O.GameState.from_moves(g["moves"])
        if g["result_first"] == 0.5:
            assert s.is_full() and not s.last_mover_won()
        else:
            assert s.last_mover_won()
            assert g["result_first"] == (1.0 if g["n_moves"] % 2 == 1 else 0.0)


def test_rows_are_encoded_positions_before_each_move():
    X, y, games = run()
    k = 0
    for g in games:
        s = O.GameState(0, 0, [])
        for c in g["moves"]:
            np.testing.assert_array_equal(X[k], E.encode(np.uint64(s.pos), np.uint64(s.mask)))
            s = s.after(c)
            k += 1
    assert k == X.shape[0]


def test_labels_alternate_and_winner_last_is_one():
    X, y, games = run(n_games=8)
    k = 0
    decisive = 0
    for g in games:
        lab = y[k:k + g["n_moves"]]
        k += g["n_moves"]
        if g["result_first"] == 0.5:
            assert np.all(lab == 0.5)
            continue
        decisive += 1
        assert set(np.unique(lab)) == {0.0, 1.0}
        assert np.all(lab[1:] + lab[:-1] == 1.0)
        assert lab[-1] == 1.0
        assert lab[0] == g["result_first"]
    assert decisive > 0


def test_deterministic_given_seed():
    a, b = run(seed=3), run(seed=3)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert a[2] == b[2]


def test_different_seed_different_games():
    a, b = run(seed=1), run(seed=2)
    assert [g["moves"] for g in a[2]] != [g["moves"] for g in b[2]]


def test_temp0_no_noise_all_games_identical():
    _, _, games = run(temp_moves=0)
    assert len({tuple(g["moves"]) for g in games}) == 1


def test_dirichlet_runs_and_is_deterministic():
    a = run(seed=5, temp_moves=0, dirichlet=(1.0, 0.25))
    b = run(seed=5, temp_moves=0, dirichlet=(1.0, 0.25))
    assert a[2] == b[2]


def test_timing_64_games_50_sims():
    rng = np.random.default_rng(0)
    SP.play_games(ConstValue(0.5), 2, 5, rng)  # warm up numba
    t = time.perf_counter()
    X, y, games = SP.play_games(ConstValue(0.5), 64, 50, np.random.default_rng(0))
    dt = time.perf_counter() - t
    print(f"\nselfplay timing: 64 games x 50 sims: {dt:.2f} s, {X.shape[0]} positions, "
          f"mean length {np.mean([g['n_moves'] for g in games]):.1f}")


# --- ReplayBuffer ---------------------------------------------------------------------

def _rows(*vals):
    return np.array([[v] * E.N_CELLS for v in vals], dtype=np.float64)


def test_buffer_dedup_averages_labels():
    buf = SP.ReplayBuffer(max_rounds=3)
    buf.add(0, _rows(1, 0, 1), np.array([1.0, 0.5, 0.0]))
    buf.add(1, _rows(1, -1), np.array([1.0, 0.0]))
    X, y = buf.data()
    assert len(buf) == 3 and X.shape == (3, E.N_CELLS)
    lab = {float(r[0]): v for r, v in zip(X, y)}
    assert lab == pytest.approx({1.0: 2 / 3, 0.0: 0.5, -1.0: 0.0})
    assert buf.stats() == {"rounds": 2, "raw": 5, "unique": 3}


def test_buffer_drops_old_rounds():
    buf = SP.ReplayBuffer(max_rounds=2)
    buf.add(0, _rows(1), np.array([1.0]))
    buf.add(1, _rows(0), np.array([0.0]))
    buf.add(2, _rows(-1), np.array([0.5]))
    X, y = buf.data()
    assert sorted(X[:, 0].tolist()) == [-1.0, 0.0]
    assert buf.stats()["rounds"] == 2 and buf.stats()["raw"] == 2


def test_buffer_order_deterministic_and_sorted_by_bytes():
    rng = np.random.default_rng(0)
    Xr = rng.integers(-1, 2, size=(500, E.N_CELLS)).astype(np.float64)
    Xr = np.vstack([Xr, Xr[:100]])
    yr = rng.random(600)
    b1, b2 = SP.ReplayBuffer(4), SP.ReplayBuffer(4)
    b1.add(0, Xr, yr)
    perm = rng.permutation(600)
    b2.add(0, Xr[perm], yr[perm])
    X1, y1 = b1.data()
    X2, y2 = b2.data()
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_allclose(y1, y2)
    keys = [r.astype(np.int8).tobytes() for r in X1]
    assert keys == sorted(keys)


def test_buffer_on_selfplay_data():
    X, y, _ = run(temp_moves=0)  # identical games: every position repeats
    buf = SP.ReplayBuffer(1)
    buf.add(0, X, y)
    Xu, yu = buf.data()
    assert len(buf) == X.shape[0] // 6
    assert set(np.unique(yu)) <= {0.0, 0.5, 1.0}


# --- root value q, mixed targets, holdout split ----------------------------------------

import hashlib

import mcts

# play_games fingerprint computed from the committed code BEFORE value_mix / return_q
# existed (commit 6105196d's play_games): default output and RNG consumption unchanged.
_FINGERPRINT_BEFORE_Q = "c0434bbfc51274cbf0f16662b1431981d35ffe036689ffebd85f16c12ca77220"


def _tanh_value(X):
    w = np.linspace(-0.3, 0.3, X.shape[1])
    return 0.5 + 0.4 * np.tanh(X @ w)


def _fingerprint(**kw):
    rng = np.random.default_rng(11)
    X, y, games = SP.play_games(_tanh_value, 5, 12, rng, temp_moves=6, dirichlet=(1.0, 0.25),
                                **kw)
    h = hashlib.sha256()
    h.update(X.tobytes())
    h.update(y.tobytes())
    h.update(repr([g["moves"] for g in games]).encode())
    h.update(repr([g["result_first"] for g in games]).encode())
    h.update(np.float64(rng.random()).tobytes())  # rng consumption
    return h.hexdigest()


def test_default_output_identical_to_before():
    assert _fingerprint() == _FINGERPRINT_BEFORE_Q
    assert _fingerprint(value_mix=1.0, return_q=False) == _FINGERPRINT_BEFORE_Q


def test_root_value_after_one_sim_is_value_fn_of_root():
    s = O.GameState(0, 0, [])
    root = mcts.new_root(s.pos, s.mask)
    mcts.run_batch([root], ConstValue(0.8), 1)
    assert SP.root_value(root) == pytest.approx(0.8)  # own view, not the parent's 0.2


def test_root_value_immediate_win_is_one():
    s = O.GameState.from_moves([0, 1, 0, 1, 0, 1])  # first player to move, wins in col 0
    root = mcts.new_root(s.pos, s.mask)
    mcts.run_batch([root], ConstValue(0.2), 30)
    assert SP.root_value(root) == pytest.approx(1.0)


def test_root_value_double_threat_is_low():
    # second player to move; first player has two open threats on row 0 (cols 0 and 4)
    s = O.GameState.from_moves([1, 1, 2, 2, 3])
    root = mcts.new_root(s.pos, s.mask)
    mcts.run_batch([root], ConstValue(0.5), 200)
    assert SP.root_value(root) < 0.1


def test_return_q_aligned_and_winning_move_rows_are_one():
    X, y, games, Q = SP.play_games(_tanh_value, 6, 20, np.random.default_rng(4),
                                   return_q=True)
    assert Q.shape == y.shape and Q.dtype == np.float64
    assert np.all((Q >= 0) & (Q <= 1))
    k = 0
    for g in games:
        k += g["n_moves"]
        if g["result_first"] != 0.5:
            assert Q[k - 1] == pytest.approx(1.0)  # the winner had an immediate win
    X0, y0, games0 = SP.play_games(_tanh_value, 6, 20, np.random.default_rng(4))
    np.testing.assert_array_equal(X, X0)
    np.testing.assert_array_equal(y, y0)
    assert games == games0


def test_value_mix_labels():
    kw = dict(n_games=5, n_sims=12, temp_moves=6, dirichlet=(1.0, 0.25))
    X1, z, g1, Q = SP.play_games(_tanh_value, rng=np.random.default_rng(2), return_q=True, **kw)
    Xm, ym, gm, Qm = SP.play_games(_tanh_value, rng=np.random.default_rng(2), value_mix=0.25,
                                   return_q=True, **kw)
    X0, y0, g0 = SP.play_games(_tanh_value, rng=np.random.default_rng(2), value_mix=0.0, **kw)
    np.testing.assert_array_equal(X1, Xm)          # labels only: games unchanged
    np.testing.assert_array_equal(X1, X0)
    assert g1 == gm == g0
    np.testing.assert_array_equal(Q, Qm)
    np.testing.assert_allclose(ym, 0.25 * z + 0.75 * Q)
    np.testing.assert_allclose(y0, Q)


def test_value_mix_out_of_range():
    with pytest.raises(ValueError):
        SP.play_games(ConstValue(0.5), 1, 4, np.random.default_rng(0), value_mix=1.5)


def test_buffer_averages_mixed_labels():
    buf = SP.ReplayBuffer(max_rounds=2)
    z = np.array([1.0, 0.0, 0.5])
    q = np.array([0.8, 0.3, 0.6])
    lab = 0.5 * z + 0.5 * q  # 0.9, 0.15, 0.55
    buf.add(0, _rows(1, 1, 0), lab)
    buf.add(1, _rows(1), np.array([0.3]))
    X, y = buf.data()
    got = {float(r[0]): v for r, v in zip(X, y)}
    assert got == pytest.approx({1.0: (0.9 + 0.15 + 0.3) / 3, 0.0: 0.55})


def test_split_holdout_by_position():
    rng = np.random.default_rng(0)
    base = rng.integers(-1, 2, size=(200, E.N_CELLS)).astype(np.float64)
    X = np.vstack([base, base[:80], base[:30]])  # duplicates
    y = rng.random(X.shape[0])
    Xt, yt, Xh, yh = SP.split_holdout(X, y, 0.25, np.random.default_rng(7))
    assert Xt.shape[0] + Xh.shape[0] == X.shape[0]
    kt = {r.astype(np.int8).tobytes() for r in Xt}
    kh = {r.astype(np.int8).tobytes() for r in Xh}
    assert not kt & kh                       # no position on both sides
    assert len(kh) == round(0.25 * 200)
    # rows keep their labels
    lab = {}
    for r, v in zip(X, y):
        lab.setdefault(r.astype(np.int8).tobytes(), []).append(v)
    for r, v in zip(Xh, yh):
        assert v in lab[r.astype(np.int8).tobytes()]
    # deterministic, and independent of row order
    a = SP.split_holdout(X, y, 0.25, np.random.default_rng(7))
    perm = rng.permutation(X.shape[0])
    b = SP.split_holdout(X[perm], y[perm], 0.25, np.random.default_rng(7))
    np.testing.assert_array_equal(a[2], Xh)
    assert {r.astype(np.int8).tobytes() for r in b[2]} == kh
    # another seed, another held-out set
    c = SP.split_holdout(X, y, 0.25, np.random.default_rng(8))
    assert {r.astype(np.int8).tobytes() for r in c[2]} != kh


def test_split_holdout_edges():
    X = _rows(1, 0, 1)
    y = np.array([1.0, 0.0, 0.5])
    Xt, yt, Xh, yh = SP.split_holdout(X, y, 0.0, np.random.default_rng(0))
    assert Xh.shape == (0, E.N_CELLS) and Xt.shape[0] == 3
    Xt, yt, Xh, yh = SP.split_holdout(X, y, 1.0, np.random.default_rng(0))
    assert Xt.shape[0] == 0 and Xh.shape[0] == 3
    with pytest.raises(ValueError):
        SP.split_holdout(X, y, 1.2, np.random.default_rng(0))
