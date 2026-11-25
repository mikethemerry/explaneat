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
