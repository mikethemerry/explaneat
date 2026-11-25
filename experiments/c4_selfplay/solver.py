"""Wrapper around Pascal Pons' perfect Connect-4 solver (``c4solver``).

The solver reads one move sequence per line (columns '1'..'7', first player first) and prints
``<seq> <score> ...`` per line; with ``-a`` it prints ``<seq> s0 ... s6`` (one score per
column, -1000 for a full column). Scores are from the side to move: >0 win, 0 draw, <0 loss.
An invalid sequence (a move that overfills a column or makes four in a row) produces no
stdout line, only a "Line N: Invalid move" message on stderr; ``solve`` detects the gap and
raises rather than misaligning results.

Board cells use the engine's index c*6+r (row 0 = bottom).
"""
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
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


_VALID = frozenset("1234567")


def _check_seq(s, where=""):
    if not s or any(ch not in _VALID for ch in s):
        raise ValueError(f"bad sequence {s!r}{where}: must be non-empty, columns '1'..'7'")


def _run_chunk(seqs, analyse, timeout=None, index=0):
    for s in seqs:
        _check_seq(s, f" (chunk {index})")
    cmd = [str(SOLVER)] + (["-a"] if analyse else [])
    try:
        p = subprocess.run(cmd, input="".join(s + "\n" for s in seqs), capture_output=True,
                           text=True, check=False, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"c4solver timed out after {timeout} s on chunk {index} "
                           f"({len(seqs)} sequences, first {seqs[0]!r})") from e
    lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
    ncol = 7 if analyse else 1
    out = np.empty((len(seqs), ncol), dtype=np.int64)
    it = iter(lines)
    for i, s in enumerate(seqs):
        ln = next(it, None)
        parts = ln.split() if ln is not None else []
        if not parts or parts[0] != s or len(parts) < 1 + ncol:
            raise RuntimeError(
                f"c4solver gave no result for {s!r} (chunk {index}, input {i} of chunk; "
                f"got {ln!r}; exit {p.returncode}); stderr:\n{p.stderr}")
        out[i] = [int(x) for x in parts[1:1 + ncol]]
    if next(it, None) is not None:
        raise RuntimeError(f"c4solver printed extra lines on chunk {index}; stderr:\n{p.stderr}")
    return out


DEFAULT_TIMEOUT = 600.0  # seconds per chunk: generous for ply >= 8 (~0.05 s a position)


def _solve_chunks(seqs, analyse, workers, chunk, timeout, on_chunk=None):
    """Solve in chunks on a thread pool. Returns the list of per-chunk arrays (input order).

    ``on_chunk(k, seqs_k, res_k)`` is called in chunk order for every chunk that finished;
    if a chunk fails, the chunks that did finish are still passed to ``on_chunk`` (in
    order) before the first error is re-raised.
    """
    for i, s in enumerate(seqs):
        _check_seq(s, f" (input {i})")
    if not SOLVER.exists():
        raise FileNotFoundError(f"c4solver not found at {SOLVER} (set C4_SOLVER)")
    chunks = [seqs[i:i + chunk] for i in range(0, len(seqs), chunk)]
    results = [None] * len(chunks)
    errors = {}
    nxt = 0
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(chunks)))) as ex:
        futs = {ex.submit(_run_chunk, ch, analyse, timeout, k): k for k, ch in enumerate(chunks)}
        for f in as_completed(futs):
            k = futs[f]
            try:
                results[k] = f.result()
            except Exception as e:  # noqa: BLE001  re-raised below
                errors[k] = e
                for g in futs:
                    g.cancel()
                continue
            while not errors and nxt < len(chunks) and results[nxt] is not None:
                if on_chunk is not None:
                    on_chunk(nxt, chunks[nxt], results[nxt])
                nxt += 1
    if errors:
        if on_chunk is not None:  # keep progress: every finished chunk, in order
            for k in range(nxt, len(chunks)):
                if results[k] is not None:
                    on_chunk(k, chunks[k], results[k])
        k = min(errors)
        raise errors[k]
    return results


def solve(seqs, analyse=False, workers=12, chunk=64, timeout=DEFAULT_TIMEOUT) -> np.ndarray:
    """Solve move sequences. int64 ``[n]`` scores, or ``[n, 7]`` with ``analyse``.

    Sign convention: every score is from the view of the SIDE TO MOVE after the sequence
    (>0 the side to move wins, 0 draw, <0 it loses; Pons' magnitude counts how early).
    The side to move is the first player for even-length sequences and the second player
    for odd-length ones; ``first_player_scores`` converts to the first player's view. With
    ``analyse`` column k is the score of playing column k (0-based) for the side to move
    *before* that move, -1000 for a full column.

    All sequences are validated before anything is scheduled. One ``c4solver`` subprocess
    per chunk of ``chunk`` sequences, ``workers`` at a time, each limited to ``timeout``
    seconds; results are in input order. Raises RuntimeError naming the chunk if any
    sequence gets no result (e.g. an invalid sequence) or a chunk times out.
    """
    seqs = [str(s) for s in seqs]
    if not seqs:
        return np.empty((0, 7) if analyse else (0,), dtype=np.int64)
    parts = _solve_chunks(seqs, analyse, workers, chunk, timeout)
    out = np.concatenate(parts, axis=0)
    return out if analyse else out[:, 0]


def first_player_scores(seqs, scores) -> np.ndarray:
    """Side-to-move scores (from ``solve``) -> the first player's view: the sign flips for
    odd-length sequences (second player to move). Works for ``[n]`` and ``[n, 7]`` (analyse:
    the -1000 'unplayable' marker is kept as is)."""
    sc = np.array(scores, dtype=np.int64, copy=True)
    odd = np.array([len(str(s)) % 2 == 1 for s in seqs], dtype=bool)
    if sc.shape[0] != len(odd):
        raise ValueError(f"{len(odd)} sequences but {sc.shape[0]} scores")
    if sc.ndim == 1:
        sc[odd] = -sc[odd]
    else:
        rows = sc[odd]
        rows = np.where(rows == -1000, -1000, -rows)
        sc[odd] = rows
    return sc


def _header(analyse):
    return ["seq"] + ([f"s{k}" for k in range(7)] if analyse else ["score"])


def _read_cache(path, header):
    """Validated cache dict seq -> list[int]. Raises ValueError naming the line on a bad
    row, a missing trailing newline or conflicting duplicate keys."""
    cache = {}
    raw = path.read_bytes()
    if not raw:
        return cache
    if not raw.endswith(b"\n"):
        raise ValueError(f"{path}: does not end in a newline (truncated write?); "
                         f"inspect and repair the last line")
    lines = raw.decode("utf-8").split("\n")[:-1]
    got = lines[0].split(",")
    if got != header:
        raise ValueError(f"{path}: header {got} does not match {header}")
    width = len(header)
    for ln_no, line in enumerate(lines[1:], start=2):
        row = line.split(",")
        if len(row) != width:
            raise ValueError(f"{path} line {ln_no}: {len(row)} fields, expected {width}: {line!r}")
        key = row[0]
        if not key or any(ch not in _VALID for ch in key):
            raise ValueError(f"{path} line {ln_no}: bad sequence key {key!r}")
        try:
            vals = [int(x) for x in row[1:]]
        except ValueError:
            raise ValueError(f"{path} line {ln_no}: non-integer field in {line!r}") from None
        if any(str(v) != x for v, x in zip(vals, row[1:])):
            raise ValueError(f"{path} line {ln_no}: non-canonical integer in {line!r}")
        if key in cache and cache[key] != vals:
            raise ValueError(f"{path} line {ln_no}: key {key!r} repeats with different "
                             f"values {cache[key]} vs {vals}")
        cache[key] = vals
    return cache


def solve_cached(seqs, path, analyse=False, workers=12, chunk=64,
                 timeout=DEFAULT_TIMEOUT) -> np.ndarray:
    """``solve`` with a csv cache keyed by sequence (``seq,score`` or ``seq,s0,...,s6``).

    Every cache row is validated on read (see ``_read_cache``). Only sequences missing
    from the cache are solved; each finished chunk is appended (one write, then flush and
    fsync) in chunk order, so a crash keeps the progress made. Returns results in input
    order, like ``solve`` (same side-to-move convention).
    """
    path = Path(path)
    seqs = [str(s) for s in seqs]
    for i, s in enumerate(seqs):
        _check_seq(s, f" (input {i})")
    header = _header(analyse)
    cache = _read_cache(path, header) if path.exists() else {}
    missing = list(dict.fromkeys(s for s in seqs if s not in cache))
    if missing:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.stat().st_size == 0:
            with path.open("w", newline="") as f:
                f.write(",".join(header) + "\n")
                f.flush()
                os.fsync(f.fileno())

        def append(k, ch, res):
            res = res.reshape(len(ch), -1)
            text = "".join(",".join([s] + [str(int(x)) for x in row]) + "\n"
                           for s, row in zip(ch, res))
            with path.open("a", newline="") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            for s, row in zip(ch, res):
                cache[s] = [int(x) for x in row]

        _solve_chunks(missing, analyse, workers, chunk, timeout, on_chunk=append)
    ncol = 7 if analyse else 1
    out = np.array([cache[s] for s in seqs], dtype=np.int64).reshape(len(seqs), ncol)
    return out if analyse else out[:, 0]
