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
    real = S._run_chunk

    def counting(ss, *a, **kw):
        calls.append(list(ss))
        return real(ss, *a, **kw)

    monkeypatch.setattr(S, "_run_chunk", counting)
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


def test_first_player_scores():
    seqs = ["121212", "4453621"]
    sc = S.solve(seqs)
    fp = S.first_player_scores(seqs, sc)
    assert fp[0] == sc[0] and fp[1] == -sc[1]
    a = np.array([[-1000, 1, -2, 0, 3, 4, 5], [1, -1000, 2, 0, -3, 4, 5]])
    fa = S.first_player_scores(["11", "111"], a)
    np.testing.assert_array_equal(fa[0], a[0])
    np.testing.assert_array_equal(fa[1], [-1, -1000, -2, 0, 3, -4, -5])


def test_solve_prevalidates(monkeypatch):
    calls = []
    monkeypatch.setattr(S, "_run_chunk", lambda *a, **k: calls.append(a))
    with pytest.raises(ValueError, match="input 2"):
        S.solve(["121212", "4453621", "1280"])
    assert calls == []


def test_solve_timeout_names_chunk():
    with pytest.raises(RuntimeError, match="chunk 0"):
        S.solve(["4"], timeout=1e-4)  # ply 1: takes far longer than the limit


def _write(path, text):
    path.write_bytes(text.encode())


@pytest.mark.parametrize("text,match", [
    ("seq,score\n121212,5,1\n", "line 2"),            # too many fields
    ("seq,score\n12x212,5\n", "line 2"),              # bad key
    ("seq,score\n121212,5\n4455,abc\n", "line 3"),    # non-integer
    ("seq,score\n121212,5\n121212,4\n", "line 3"),    # conflicting duplicate
    ("seq,s0\n121212,5\n", "header"),
])
def test_solve_cached_rejects_bad_cache(tmp_path, text, match):
    path = tmp_path / "c.csv"
    _write(path, text)
    with pytest.raises(ValueError, match=match):
        S.solve_cached(["121212"], path)


@pytest.mark.parametrize("text,kept", [
    ("seq,score\n121212,18\n4455,12", "seq,score\n121212,18\n"),   # cut mid-row
    ("seq,score\n121212,18\n4455,", "seq,score\n121212,18\n"),
    ("seq,sc", ""),                                                   # cut in the header
])
def test_solve_cached_repairs_truncated_tail(tmp_path, text, kept):
    path = tmp_path / "c.csv"
    _write(path, text)
    with pytest.warns(UserWarning, match=f"dropped the last {len(text) - len(kept)} bytes"):
        got = S.solve_cached(["121212", "4455"], path)
    np.testing.assert_array_equal(got, S.solve(["121212", "4455"]))
    lines = path.read_text().splitlines()
    assert lines[0] == "seq,score"
    assert sorted(ln.split(",")[0] for ln in lines[1:]) == ["121212", "4455"]
    assert path.read_text().endswith("\n")


def test_solve_cached_repair_keeps_validation(tmp_path):
    path = tmp_path / "c.csv"
    _write(path, "seq,score\n12x212,5\n4455,1")   # bad row before the cut tail
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="line 2"):
        S.solve_cached(["121212"], path)


def test_solve_cached_consistent_duplicate_ok(tmp_path):
    path = tmp_path / "c.csv"
    _write(path, "seq,score\n121212,18\n121212,18\n")
    assert S.solve_cached(["121212"], path)[0] == 18


def test_solve_cached_keeps_progress_on_crash(tmp_path, monkeypatch):
    path = tmp_path / "c.csv"
    seqs = ["4453621", "121212", "2121213", "4455", "776655", "443322"]
    real = S._run_chunk

    def failing(ss, analyse, timeout=None, index=0):
        if index == 1:
            raise RuntimeError("boom")
        return real(ss, analyse, timeout, index)

    monkeypatch.setattr(S, "_run_chunk", failing)
    with pytest.raises(RuntimeError, match="boom"):
        S.solve_cached(seqs, path, workers=1, chunk=2)
    kept = [ln.split(",")[0] for ln in path.read_text().splitlines()[1:]]
    assert kept[:2] == seqs[:2] and "121212" in kept and seqs[2] not in kept
    calls = []

    def counting(ss, *a, **kw):
        calls.append(list(ss))
        return real(ss, *a, **kw)

    monkeypatch.setattr(S, "_run_chunk", counting)
    got = S.solve_cached(seqs, path, workers=1, chunk=2)
    solved = [s for c in calls for s in c]
    assert set(solved) == set(seqs) - set(kept)
    np.testing.assert_array_equal(got, S.solve(seqs))
    lines = path.read_text().splitlines()[1:]
    assert sorted(ln.split(",")[0] for ln in lines) == sorted(seqs)


@pytest.mark.slow
def test_uci_sample_agrees():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "complex_real"))
    from data import load_task

    X, y, _, _ = load_task("c4")  # x = +1 (first player, to move), o = -1; y = 1 iff x wins
    idx = np.random.default_rng(0).choice(len(X), 40, replace=False)
    sc = S.solve([S.board_to_seq(X[i]) for i in idx])
    assert ((sc > 0).astype(int) == y[idx]).all()
