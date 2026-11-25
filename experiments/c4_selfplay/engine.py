"""Connect-4 bitboard engine (Pons' layout), numba-jitted.

7 columns x 6 rows; column c holds bits c*7..c*7+5 (bit c*7+6 is a sentinel that keeps
shifts from wrapping between columns). ``pos`` = stones of the side to move, ``mask`` = all
stones, both np.uint64. After ``play`` the new ``pos`` is the stones of the new side to move,
so the player who just moved holds ``pos ^ mask``. Columns are 0-based.

The jitted functions are callable from other njit code (MCTS). Called from Python, numba
boxes their uint64 results as Python ints; the values are exact (boards use 49 bits).

Cell index c*6+r (UCI order a1..a6, b1..g6; row 0 = bottom) is bit c*7+r. ``encode`` gives
the 42 cells from the side to move's view: +1 mine, -1 theirs, 0 empty.
"""
import numpy as np
from numba import njit

W, H = 7, 6
N_CELLS = W * H

_ZERO = np.uint64(0)
_ONE = np.uint64(1)


@njit(cache=True)
def bottom_mask(c):
    return np.uint64(1) << np.uint64(c * 7)


@njit(cache=True)
def top_mask(c):
    return np.uint64(1) << np.uint64(c * 7 + 5)


@njit(cache=True)
def column_mask(c):
    return np.uint64(63) << np.uint64(c * 7)


@njit(cache=True)
def can_play(mask, c):
    return (np.uint64(mask) & top_mask(c)) == np.uint64(0)


@njit(cache=True)
def play(pos, mask, c):
    """Drop a stone for the side to move in column c; returns the new (pos, mask).

    No bounds check: the caller must check ``can_play(mask, c)`` first. Playing a full
    column corrupts the state (the carry spills into the sentinel and the next column).
    """
    pos, mask = np.uint64(pos), np.uint64(mask)
    new_pos = pos ^ mask  # the opponent's stones: they move next
    new_mask = mask | (mask + bottom_mask(c))
    return new_pos, new_mask


@njit(cache=True)
def has_won(p):
    """Does the stone set p contain four in a row?"""
    p = np.uint64(p)
    for s in (1, 7, 6, 8):  # vertical, horizontal, diagonal \, diagonal /
        sh = np.uint64(s)
        m = p & (p >> sh)
        if (m & (m >> np.uint64(2 * s))) != np.uint64(0):
            return True
    return False


@njit(cache=True)
def is_winning_move(pos, mask, c):
    """Would the side to move win by playing column c (assumed playable)?"""
    pos, mask = np.uint64(pos), np.uint64(mask)
    p = pos | ((mask + bottom_mask(c)) & column_mask(c))
    return has_won(p)


@njit(cache=True)
def n_moves(mask):
    m = np.uint64(mask)
    n = 0
    while m != np.uint64(0):
        m &= m - np.uint64(1)
        n += 1
    return n


@njit(cache=True)
def legal_moves(mask):
    out = np.empty(W, np.int64)
    k = 0
    for c in range(W):
        if can_play(mask, c):
            out[k] = c
            k += 1
    return out[:k]


@njit(cache=True)
def encode_into(pos, mask, out):
    """Write the 42-cell side-to-move encoding into ``out`` (float array of length 42)."""
    pos, mask = np.uint64(pos), np.uint64(mask)
    for c in range(W):
        for r in range(H):
            b = np.uint64(1) << np.uint64(c * 7 + r)
            if (mask & b) == np.uint64(0):
                out[c * H + r] = 0.0
            elif (pos & b) != np.uint64(0):
                out[c * H + r] = 1.0
            else:
                out[c * H + r] = -1.0


@njit(cache=True)
def encode(pos, mask):
    out = np.empty(N_CELLS, np.float64)
    encode_into(pos, mask, out)
    return out


@njit(cache=True)
def _from_cells(v):
    pos = np.uint64(0)
    mask = np.uint64(0)
    for c in range(W):
        for r in range(H):
            x = v[c * H + r]
            if x != 0:
                b = np.uint64(1) << np.uint64(c * 7 + r)
                mask |= b
                if x > 0:
                    pos |= b
    return pos, mask


def from_cells(v):
    """Inverse of ``encode``: (pos, mask) from a 42-cell side-to-move array.

    Does no validation (gravity, stone counts, fours are not checked). A Python wrapper,
    not callable from njit code; jitted callers use ``_from_cells``.
    """
    pos, mask = _from_cells(np.ascontiguousarray(v, dtype=np.float64))
    return np.uint64(pos), np.uint64(mask)
