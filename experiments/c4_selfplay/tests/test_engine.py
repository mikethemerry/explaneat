import numpy as np
import pytest

import engine as E


def seq(s, pos=np.uint64(0), mask=np.uint64(0)):
    """Play a 1-based column-char sequence from (pos, mask)."""
    for ch in s:
        c = int(ch) - 1
        assert E.can_play(mask, c), f"column {ch} full in {s}"
        pos, mask = E.play(pos, mask, c)
    return pos, mask


def brute_win(cells, who):
    """4 in a row of value ``who`` on a 42-cell array, cell index c*6+r."""
    g = np.asarray(cells).reshape(7, 6)  # g[c, r]
    for c in range(7):
        for r in range(6):
            for dc, dr in ((1, 0), (0, 1), (1, 1), (1, -1)):
                ok = True
                for k in range(4):
                    cc, rr = c + dc * k, r + dr * k
                    if not (0 <= cc < 7 and 0 <= rr < 6) or g[cc, rr] != who:
                        ok = False
                        break
                if ok:
                    return True
    return False


def test_play_values_and_njit_callable():
    # numba boxes uint64 returns as Python int at the Python boundary; values are exact
    pos, mask = E.play(np.uint64(0), np.uint64(0), 3)
    assert mask == 1 << 21 and pos == 0
    pos, mask = E.play(pos, mask, 3)
    assert mask == (1 << 21) | (1 << 22) and pos == 1 << 21

    from numba import njit

    @njit
    def drop_all(c):
        pos, mask = np.uint64(0), np.uint64(0)
        for _ in range(6):
            pos, mask = E.play(pos, mask, c)
        return E.can_play(mask, c), E.n_moves(mask)

    assert drop_all(2) == (False, 6)


def test_vertical_win():
    pos, mask = seq("1212121")
    assert E.has_won(pos ^ mask)
    assert not E.has_won(pos)


def test_no_win_after_three_vertical():
    pos, mask = seq("121212")
    assert not E.has_won(pos) and not E.has_won(pos ^ mask)


def test_horizontal_win():
    pos, mask = seq("1122334")
    assert E.has_won(pos ^ mask)
    assert not E.has_won(pos)


def test_diagonal_up_right_win():
    # x: a1 b2 c3 d4 (plus fillers d1, g1)
    s = "12234334744"
    pos, mask = seq(s[:-1])
    assert not E.has_won(pos) and not E.has_won(pos ^ mask)
    pos, mask = seq(s)
    assert E.has_won(pos ^ mask)
    assert not E.has_won(pos)
    assert brute_win(E.encode(pos, mask), -1)


def test_diagonal_up_left_win():
    # mirror: x: g1 f2 e3 d4
    s = "76654554144"
    pos, mask = seq(s[:-1])
    assert not E.has_won(pos) and not E.has_won(pos ^ mask)
    pos, mask = seq(s)
    assert E.has_won(pos ^ mask)
    assert not E.has_won(pos)
    assert brute_win(E.encode(pos, mask), -1)


def test_raised_diagonal_win():
    # diagonals off the bottom row, built from cells
    v = np.zeros(42)
    for c, r in ((0, 5), (1, 4), (2, 3), (3, 2)):  # a6 b5 c4 d3
        v[c * 6 + r] = 1
    pos, _ = E.from_cells(v)
    assert E.has_won(pos)
    v2 = np.zeros(42)
    for c, r in ((3, 2), (4, 3), (5, 4), (6, 5)):  # d3 e4 f5 g6
        v2[c * 6 + r] = 1
    assert E.has_won(E.from_cells(v2)[0])


def test_no_wraparound():
    # a6-equivalent edge: stones a4 a5 a6 b1 must not count as vertical through the sentinel
    v = np.zeros(42)
    for c, r in ((0, 3), (0, 4), (0, 5), (1, 0)):
        v[c * 6 + r] = 1
    assert not E.has_won(E.from_cells(v)[0])
    # horizontal e1 f1 g1 + a2 (wrap across rows) must not count
    v = np.zeros(42)
    for c, r in ((4, 0), (5, 0), (6, 0), (0, 1)):
        v[c * 6 + r] = 1
    assert not E.has_won(E.from_cells(v)[0])


def test_can_play_and_legal_moves():
    pos, mask = seq("111111")
    assert not E.can_play(mask, 0)
    assert all(E.can_play(mask, c) for c in range(1, 7))
    lm = E.legal_moves(mask)
    assert lm.dtype == np.int64
    assert list(lm) == [1, 2, 3, 4, 5, 6]
    assert list(E.legal_moves(np.uint64(0))) == list(range(7))


def test_is_winning_move():
    pos, mask = seq("121212")  # x to move, three in column 1
    assert E.is_winning_move(pos, mask, 0)
    assert not E.is_winning_move(pos, mask, 1)
    assert not E.is_winning_move(pos, mask, 2)
    pos, mask = seq("112233")  # x to move with a1 b1 c1
    assert E.is_winning_move(pos, mask, 3)
    assert not E.is_winning_move(pos, mask, 4)
    pos, mask = seq("1122337")  # o to move with a2 b2 c2; column d lands on d1, no win
    assert not E.is_winning_move(pos, mask, 3)


def test_is_winning_move_for_o():
    # x g1, o a1, x a2, o b1, x b2, o c1, x c2: o to move with a1 b1 c1
    pos, mask = seq("7112233")
    assert E.is_winning_move(pos, mask, 3)
    pos2, mask2 = E.play(pos, mask, 3)
    assert E.has_won(pos2 ^ mask2)


def test_encode_side_to_move():
    pos, mask = seq("4")  # x d1, o to move
    v = E.encode(pos, mask)
    assert v.shape == (42,) and v.dtype == np.float64
    assert v[3 * 6 + 0] == -1 and np.count_nonzero(v) == 1
    pos, mask = seq("44")  # x to move; x d1, o d2
    v = E.encode(pos, mask)
    assert v[18] == 1 and v[19] == -1 and np.count_nonzero(v) == 2


def test_encode_into_matches_encode():
    pos, mask = seq("4433251")
    out = np.zeros((2, 42))
    E.encode_into(pos, mask, out[1])
    assert np.array_equal(out[1], E.encode(pos, mask))
    assert not out[0].any()


@pytest.mark.parametrize("s", ["", "4", "44", "4433251", "12234334744"[:-1], "1212345671"])
def test_from_cells_roundtrip(s):
    pos, mask = seq(s)
    p2, m2 = E.from_cells(E.encode(pos, mask))
    assert p2 == pos and m2 == mask
    assert isinstance(p2, np.uint64) and isinstance(m2, np.uint64)


def test_n_moves():
    assert E.n_moves(np.uint64(0)) == 0
    for s in ["4", "4433251", "1111112222223"]:
        assert E.n_moves(seq(s)[1]) == len(s)


def test_random_games_match_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(200):
        pos, mask = np.uint64(0), np.uint64(0)
        for ply in range(42):
            # before the move: side to move holds pos
            v = E.encode(pos, mask)
            assert E.n_moves(mask) == ply
            assert E.has_won(pos) == brute_win(v, 1)
            assert E.has_won(pos ^ mask) == brute_win(v, -1)
            assert np.array_equal(
                [E.from_cells(v)[0], E.from_cells(v)[1]], [pos, mask])
            if E.has_won(pos ^ mask):
                break
            moves = E.legal_moves(mask)
            assert list(moves) == [c for c in range(7) if v[c * 6 + 5] == 0]
            c = int(rng.choice(moves))
            wins = E.is_winning_move(pos, mask, c)
            pos, mask = E.play(pos, mask, c)
            assert wins == E.has_won(pos ^ mask)
