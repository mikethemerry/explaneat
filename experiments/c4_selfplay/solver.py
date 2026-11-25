"""Wrapper around Pascal Pons' perfect Connect-4 solver (``c4solver``).

The solver reads one move sequence per line (columns '1'..'7', first player first) and prints
``<seq> <score> ...`` per line; with ``-a`` it prints ``<seq> s0 ... s6`` (one score per
column, -1000 for a full column). Scores are from the side to move: >0 win, 0 draw, <0 loss.
An invalid sequence (a move that overfills a column or makes four in a row) produces no
stdout line, only a "Line N: Invalid move" message on stderr; ``solve`` detects the gap and
raises rather than misaligning results.

Board cells use the engine's index c*6+r (row 0 = bottom).
"""
import csv
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import engine as E

SOLVER = Path(os.environ.get("C4_SOLVER", "~/.explaneat/tools/pons_connect4/c4solver")).expanduser()

W, H = 7, 6


def seq_of_game(moves) -> str:
    """0-based column ints -> solver string ('1'..'7')."""
    out = []
    for c in moves:
        c = int(c)
        if not 0 <= c < W:
            raise ValueError(f"column {c} out of range")
        out.append(str(c + 1))
    return "".join(out)


def board_to_seq(cells) -> str:
    """A legal move order reaching ``cells`` (float[42], +1 = first player's stones,
    -1 = second's, 0 empty, index c*6+r).

    Legal means gravity per column, alternation starting with the first player, and no move
    that makes four in a row (the solver rejects such sequences). Found by backtracking;
    raises ValueError if the board is not gravity-consistent, the stone counts are wrong, or
    no legal order exists.
    """
    v = np.asarray(cells, dtype=np.float64).reshape(-1)
    if v.shape != (W * H,):
        raise ValueError(f"expected 42 cells, got {v.shape}")
    if not np.isin(v, (-1.0, 0.0, 1.0)).all():
        raise ValueError("cells must be -1, 0 or +1")
    g = v.reshape(W, H)
    cols = []
    for c in range(W):
        col = [int(x) for x in g[c]]
        k = next((r for r in range(H) if col[r] == 0), H)
        if any(col[k:]):
            raise ValueError(f"column {c} has a stone above an empty cell")
        cols.append(col[:k])
    n1 = sum(col.count(1) for col in cols)
    n2 = sum(col.count(-1) for col in cols)
    if n1 - n2 not in (0, 1):
        raise ValueError(f"stone counts {n1} (first) and {n2} (second) are not a legal position")
    n = n1 + n2
    h = [0] * W
    seq = []

    def rec(pos, mask):
        if len(seq) == n:
            return True
        colour = 1 if len(seq) % 2 == 0 else -1
        for c in range(W):
            if h[c] < len(cols[c]) and cols[c][h[c]] == colour:
                if E.is_winning_move(pos, mask, c):
                    continue
                npos, nmask = E.play(pos, mask, c)
                h[c] += 1
                seq.append(str(c + 1))
                if rec(npos, nmask):
                    return True
                h[c] -= 1
                seq.pop()
        return False

    if not rec(np.uint64(0), np.uint64(0)):
        raise ValueError("no legal move order reaches this board")
    return "".join(seq)


def _run_chunk(seqs, analyse):
    for s in seqs:
        if not s or any(ch not in "1234567" for ch in s):
            raise ValueError(f"bad sequence {s!r}")
    cmd = [str(SOLVER)] + (["-a"] if analyse else [])
    p = subprocess.run(cmd, input="".join(s + "\n" for s in seqs), capture_output=True,
                       text=True, check=False)
    lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
    ncol = 7 if analyse else 1
    out = np.empty((len(seqs), ncol), dtype=np.int64)
    it = iter(lines)
    for i, s in enumerate(seqs):
        ln = next(it, None)
        parts = ln.split() if ln is not None else []
        if not parts or parts[0] != s or len(parts) < 1 + ncol:
            raise RuntimeError(
                f"c4solver gave no result for {s!r} (input {i} of chunk; got {ln!r}; "
                f"exit {p.returncode}); stderr:\n{p.stderr}")
        out[i] = [int(x) for x in parts[1:1 + ncol]]
    if next(it, None) is not None:
        raise RuntimeError(f"c4solver printed extra lines; stderr:\n{p.stderr}")
    return out


def solve(seqs, analyse=False, workers=12, chunk=64) -> np.ndarray:
    """Solve move sequences. int64 ``[n]`` scores, or ``[n, 7]`` with ``analyse``.

    One ``c4solver`` subprocess per chunk of ``chunk`` sequences, ``workers`` at a time;
    results are in input order. Raises RuntimeError (with the solver's stderr) if any
    sequence gets no result, e.g. an invalid sequence.
    """
    seqs = [str(s) for s in seqs]
    if not seqs:
        return np.empty((0, 7) if analyse else (0,), dtype=np.int64)
    if not SOLVER.exists():
        raise FileNotFoundError(f"c4solver not found at {SOLVER} (set C4_SOLVER)")
    chunks = [seqs[i:i + chunk] for i in range(0, len(seqs), chunk)]
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(chunks)))) as ex:
        parts = list(ex.map(lambda ch: _run_chunk(ch, analyse), chunks))
    out = np.concatenate(parts, axis=0)
    return out if analyse else out[:, 0]


def _header(analyse):
    return ["seq"] + ([f"s{k}" for k in range(7)] if analyse else ["score"])


def solve_cached(seqs, path, analyse=False, **kw) -> np.ndarray:
    """``solve`` with a csv cache keyed by sequence (``seq,score`` or ``seq,s0,...,s6``).

    Only sequences missing from the cache are solved; they are appended in first-seen
    order. Returns results in input order, like ``solve``.
    """
    path = Path(path)
    seqs = [str(s) for s in seqs]
    header = _header(analyse)
    cache = {}
    if path.exists() and path.stat().st_size > 0:
        with path.open(newline="") as f:
            r = csv.reader(f)
            got = next(r)
            if got != header:
                raise ValueError(f"{path}: header {got} does not match {header}")
            for row in r:
                if row:
                    cache[row[0]] = [int(x) for x in row[1:]]
    missing = list(dict.fromkeys(s for s in seqs if s not in cache))
    if missing:
        res = solve(missing, analyse=analyse, **kw)
        res = res.reshape(len(missing), -1)
        new = not path.exists() or path.stat().st_size == 0
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            if new:
                w.writerow(header)
            for s, row in zip(missing, res):
                vals = [int(x) for x in row]
                w.writerow([s] + vals)
                cache[s] = vals
    ncol = 7 if analyse else 1
    out = np.array([cache[s] for s in seqs], dtype=np.int64).reshape(len(seqs), ncol)
    return out if analyse else out[:, 0]
