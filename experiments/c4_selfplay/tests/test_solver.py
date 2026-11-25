import sys
from pathlib import Path

import numpy as np
import pytest

import engine as E
import solver as S

pytestmark = pytest.mark.skipif(not S.SOLVER.exists(), reason="Pons solver not built")


def replay(s):
    pos, mask = np.uint64(0), np.uint64(0)
    for ch in s:
        c = int(ch) - 1
        assert E.can_play(mask, c)
        pos, mask = E.play(pos, mask, c)
    return int(pos), int(mask)


def first_player_cells(s):
    pos, mask = replay(s)
    v = E.encode(pos, mask)
    return v if len(s) % 2 == 0 else -v


def test_win_in_one_positive():
    assert S.solve(["121212"])[0] > 0


def test_analyse_full_column():
    a = S.solve(["111111"], analyse=True)
    assert a.shape == (1, 7)
    assert a[0, 0] == -1000
    assert (a[0, 1:] != -1000).all()


def test_board_to_seq_round_trip():
    s0 = "4453621"  # 7 ply: second player to move
    cells = first_player_cells(s0)
    s1 = S.board_to_seq(cells)
    assert len(s1) == len(s0)
    assert replay(s1) == replay(s0)
    np.testing.assert_array_equal(first_player_cells(s1), cells)


def test_board_to_seq_rejects_bad_boards():
    floating = np.zeros(42)
    floating[1] = 1.0  # column a, row 2 with row 1 empty
    with pytest.raises(ValueError):
        S.board_to_seq(floating)
    counts = np.zeros(42)
    counts[0] = -1.0  # second player has more stones
    with pytest.raises(ValueError):
        S.board_to_seq(counts)


def test_seq_of_game():
    assert S.seq_of_game([3, 3, 0, 6]) == "4417"


def test_invalid_sequence_raises():
    with pytest.raises(RuntimeError):
        S.solve(["121212", "1212121"])


def test_empty_input():
    assert S.solve([]).shape == (0,)
    assert S.solve([], analyse=True).shape == (0, 7)


def test_order_preserved_across_chunks():
    seqs = ["4453621", "121212", "44444411", "33333", "2121213", "4455", "776655",
            "1234567", "44536213", "443322"]
    one = np.array([S.solve([s])[0] for s in seqs])
    many = S.solve(seqs, workers=4, chunk=3)
    assert many.dtype == np.int64
    np.testing.assert_array_equal(many, one)


def test_solve_cached(tmp_path, monkeypatch):
    path = tmp_path / "cache.csv"
    seqs = ["4453621", "121212", "2121213"]
    first = S.solve_cached(seqs, path)
    calls = []
    real = S.solve

    def counting(ss, **kw):
        calls.append(list(ss))
        return real(ss, **kw)

    monkeypatch.setattr(S, "solve", counting)
    again = S.solve_cached(seqs, path)
    np.testing.assert_array_equal(again, first)
    assert sum(len(c) for c in calls) == 0
    more = S.solve_cached(["121212", "33333"], path)
    assert calls == [["33333"]]
    assert more[0] == first[1]
    text = path.read_text()
    assert text.splitlines()[0] == "seq,score"
    assert len(text.splitlines()) == 5


def test_solve_cached_analyse(tmp_path):
    path = tmp_path / "a.csv"
    a = S.solve_cached(["111111"], path, analyse=True)
    b = S.solve_cached(["111111"], path, analyse=True)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (1, 7)
    assert path.read_text().splitlines()[0] == "seq,s0,s1,s2,s3,s4,s5,s6"


@pytest.mark.slow
def test_uci_sample_agrees():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "complex_real"))
    from data import load_task

    X, y, _, _ = load_task("c4")  # x = +1 (first player, to move), o = -1; y = 1 iff x wins
    idx = np.random.default_rng(0).choice(len(X), 40, replace=False)
    sc = S.solve([S.board_to_seq(X[i]) for i in idx])
    assert ((sc > 0).astype(int) == y[idx]).all()
