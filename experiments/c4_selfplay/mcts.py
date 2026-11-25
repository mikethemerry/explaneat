"""Batched value-only MCTS for Connect-4 (no policy head: uniform priors).

Many independent trees (one per game) are searched in lock step: each simulation step
descends every root to one leaf, evaluates all new non-terminal leaves with ONE
``value_fn`` call on a [k, 42] batch (``engine.encode_into``), expands them and backs up.

Value convention
    A node's value is P(win) for the side to move AT THAT NODE. ``Node.W`` sums backed-up
    values from the PARENT's mover's view (how good the node was for the player who chose
    it), so ``Q = W / N`` is read directly by the parent's selection. Back-up flips the
    value at each level (v -> 1 - v). The root's W is kept but never used.

Selection (PUCT, uniform prior p = 1/k over the k children)
    score = Q + c_puct * p * sqrt(N_parent) / (1 + N_child); ties -> lowest column.
    FPU: an unvisited child has Q = 0.5 ("neutral": no information, a draw-ish guess).
    Because a visited child's exploration term is at most half an unvisited one's, the
    unvisited siblings are tried before a visited child is revisited unless its Q beats
    0.5 by c_puct * p * sqrt(N) * n / (1 + n) (>= 0.107 at c_puct 1.5, k 7): a clearly
    good move can be deepened early, every child is still reached quickly. A parent-Q FPU
    (AlphaZero variants) would tie FPU to the network's calibration; 0.5 keeps the search
    independent of it and makes ConstValue(0.5) a pure uniform search.

Shortcuts (``shortcuts=True``, applied at expansion)
    * Immediate win: if the side to move can win at once, the node gets only that child
      (lowest such column). The child is terminal (value 0.0 for the loser to move).
    * Forced block: else, if the opponent has exactly one playable winning cell, the only
      child is the block. With two or more threats all children are kept (lost anyway).

Root noise (``dirichlet=(alpha, eps)``): each root gets priors
    p = (1 - eps) / k + eps * Dir(alpha), drawn from ``rng`` once, when the root is first
    expanded (or at the start of ``run_batch`` for an already expanded root without noise).
    With ``dirichlet=None`` no randomness is used anywhere.

The first simulation on a fresh root evaluates the root itself, so after ``n_sims`` the
root has N == n_sims and its children's visits sum to n_sims - 1.

Bitboard helpers are pure-Python re-implementations of the engine's (same Pons layout,
Python ints): calling jitted functions one at a time from Python costs more than the bit
twiddling itself. ``tests/test_mcts.py`` checks them against ``engine``.
"""
from math import sqrt

import numpy as np
from numba import njit

import engine as E

W, H = E.W, E.H
_FULL = sum(((1 << H) - 1) << (7 * c) for c in range(W))  # all 42 cells
FPU = 0.5


# --- pure-Python bitboard helpers (Pons layout, Python ints) ------------------------------

def _can_play(mask, c):
    return not (mask >> (7 * c + 5)) & 1


def _play(pos, mask, c):
    return pos ^ mask, mask | (mask + (1 << (7 * c)))


def _has_won(p):
    m = p & (p >> 1)
    if m & (m >> 2):
        return True
    m = p & (p >> 7)
    if m & (m >> 14):
        return True
    m = p & (p >> 6)
    if m & (m >> 12):
        return True
    m = p & (p >> 8)
    return bool(m & (m >> 16))


def _is_winning_move(pos, mask, c):
    return _has_won(pos | ((mask + (1 << (7 * c))) & (63 << (7 * c))))


@njit(cache=True)
def _encode_batch(pos, mask, k, out):
    for i in range(k):
        E.encode_into(pos[i], mask[i], out[i])


# --- tree -----------------------------------------------------------------------------

class Node:
    __slots__ = ("pos", "mask", "terminal_value", "children", "N", "W", "expanded", "prior")

    def __init__(self, pos, mask):
        self.pos = pos
        self.mask = mask
        if _has_won(pos ^ mask):
            self.terminal_value = 0.0  # the move into this node won: side to move lost
        elif mask == _FULL:
            self.terminal_value = 0.5
        else:
            self.terminal_value = None
        self.children = {}  # col -> Node, ascending columns
        self.N = 0
        self.W = 0.0
        self.expanded = False
        self.prior = None  # root Dirichlet priors, aligned with children order

    def __repr__(self):
        return (f"Node(N={self.N}, Q={self.W / self.N if self.N else None}, "
                f"terminal={self.terminal_value}, children={list(self.children)})")


def new_root(pos, mask) -> Node:
    return Node(int(pos), int(mask))


def _expand(node, shortcuts):
    """Create the node's children. Returns True when the side to move has an immediate win
    (found only with ``shortcuts``): the caller then backs up the exact value 1.0."""
    pos, mask = node.pos, node.mask
    legal = [c for c in range(W) if not (mask >> (7 * c + 5)) & 1]
    won = False
    if shortcuts:
        wins = [c for c in legal if _is_winning_move(pos, mask, c)]
        if wins:
            legal = wins[:1]
            won = True
        else:
            opp = pos ^ mask
            threats = [c for c in legal if _is_winning_move(opp, mask, c)]
            if len(threats) == 1:
                legal = threats
    ch = node.children
    for c in legal:
        ch[c] = Node(pos ^ mask, mask | (mask + (1 << (7 * c))))
    node.expanded = True
    return won


def _select(node, c_puct, prior):
    """Child of an expanded node maximising PUCT; ties -> lowest column."""
    sq = c_puct * sqrt(node.N)
    best = None
    best_s = -1.0
    if prior is None:
        u = sq / len(node.children)
        for child in node.children.values():
            n = child.N
            s = (child.W / n if n else FPU) + u / (1 + n)
            if s > best_s:
                best_s, best = s, child
    else:
        for p, child in zip(prior, node.children.values()):
            n = child.N
            s = (child.W / n if n else FPU) + sq * p / (1 + n)
            if s > best_s:
                best_s, best = s, child
    return best


def _backup(path, v):
    """v: value for the side to move at the leaf (path[-1])."""
    for node in reversed(path):
        node.N += 1
        node.W += 1.0 - v
        v = 1.0 - v


def _add_noise(root, rng, dirichlet):
    alpha, eps = dirichlet
    k = len(root.children)
    noise = rng.dirichlet([alpha] * k)
    root.prior = [(1.0 - eps) / k + eps * float(x) for x in noise]


def run_batch(roots, value_fn, n_sims, rng=None, dirichlet=None, c_puct=1.5, shortcuts=True):
    """Run ``n_sims`` simulations on every root (terminal roots are skipped).

    One ``value_fn`` call per simulation step, on all new non-terminal leaves at once.
    Trees are grown in place; returns ``roots``.
    """
    if dirichlet is not None and rng is None:
        raise ValueError("dirichlet noise needs an explicit rng (np.random.Generator)")
    if len({id(r) for r in roots}) != len(roots):
        raise ValueError("the same Node is passed twice in roots; trees must be distinct")
    live = [r for r in roots if r.terminal_value is None]
    m = len(live)
    if m == 0 or n_sims <= 0:
        return roots
    buf = np.empty((m, E.N_CELLS), np.float64)
    pos_a = np.empty(m, np.uint64)
    mask_a = np.empty(m, np.uint64)
    if dirichlet is not None:
        for r in live:
            if r.expanded and r.prior is None:
                _add_noise(r, rng, dirichlet)

    for _ in range(n_sims):
        pending = []
        for root in live:
            path = [root]
            node = root
            if node.expanded:
                node = _select(node, c_puct, root.prior if dirichlet is not None else None)
                path.append(node)
                while node.expanded and node.terminal_value is None:
                    node = _select(node, c_puct, None)
                    path.append(node)
            tv = node.terminal_value
            if tv is not None:
                _backup(path, tv)
            else:
                k = len(pending)
                pos_a[k] = node.pos
                mask_a[k] = node.mask
                pending.append(path)
        k = len(pending)
        if k:
            _encode_batch(pos_a, mask_a, k, buf)
            values = np.asarray(value_fn(buf[:k]), dtype=np.float64).reshape(-1).tolist()
            for path, v in zip(pending, values):
                leaf = path[-1]
                if _expand(leaf, shortcuts):
                    v = 1.0          # immediate win for the side to move: exact, not the net's guess
                if dirichlet is not None and len(path) == 1 and leaf.prior is None:
                    _add_noise(leaf, rng, dirichlet)
                _backup(path, v)
    return roots


def visit_counts(root) -> np.ndarray:
    vc = np.zeros(W, np.int64)
    for c, child in root.children.items():
        vc[c] = child.N
    return vc


def policy(root, temperature) -> np.ndarray:
    """Move distribution from root visits. T = 0: one-hot argmax (ties -> lowest column);
    else counts^(1/T) normalised. All zeros for a root with no visited children."""
    vc = visit_counts(root).astype(np.float64)
    out = np.zeros(W, np.float64)
    if vc.max() <= 0:
        return out
    if temperature == 0:
        out[int(np.argmax(vc))] = 1.0
        return out
    p = (vc / vc.max()) ** (1.0 / temperature)
    return p / p.sum()
