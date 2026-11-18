"""Rule-based stroke concepts for handwritten digits (sklearn digits 8x8, MNIST 28x28).

Each concept is a yes / no property computed from the image alone (never from
its label), so a node can be probed against it exactly like a planted concept.
All rules work on the **ink mask** ``ink = x >= INK`` (pixels in [0, 1]) and are
measured relative to the ink's bounding box (rows ``r0..r1``, columns ``c0..c1``,
height ``h``, width ``w``), so they do not depend on where the digit sits or on
the image size. Fractions below are fractions of ``h`` (rows) or ``w`` (columns).

* ``upper_loop`` - **closed upper loop**: a hole (a background region, 4-connected,
  not connected to the image border, of at least ``HOLE_MIN`` of the image's pixels and
  at least 1 pixel) whose centre lies in the upper half of the box.
* ``lower_loop`` - **closed lower loop**: the same, centre in the lower half.
* ``open_left`` - **open left side** (the left side of a 3): in the rows from 25 % to 75 %
  of the box, at most 35 % of the inked rows have ink in the left third of the box.
* ``top_bar`` - **top horizontal stroke**: some row in the top 25 % of the box has a
  run of consecutive ink at least 60 % of the box width long.
* ``waist`` - **central crossing (waist)**: some inked row in the 40 % to 60 % band is
  narrow (ink extent at most 40 % of the box width) and central (the middle of its
  extent lies between 30 % and 60 % of the box width).
* ``left_heavy`` - **more ink on the left**: the left half of the box holds more ink
  (sum of pixel values) than the right half.

Two more for 10-class MNIST (``STROKES10``; the binary reports keep ``STROKES``):

* ``centre_vstroke`` - **tall vertical stem** (meant for the stem of a 1, 4 or 7): only digits at least
  1.6 times taller than wide; starting from a column between 20 % and 80 % of the box width, a path of
  ink followed down the box (moving at most one column per row, gaps of one row allowed) covers at
  least 85 % of its rows. On MNIST it fires on most 1s but also on a third of 6s and 9s and few 4s
  and 7s (``validate``): a "tall, stem-like digit" concept rather than a clean 4/7 stroke.
* ``any_loop`` - **a closed loop anywhere**: ``upper_loop`` or ``lower_loop``.
* ``heavy_ink`` - **a lot of ink**: more ink pixels than ``INK_MEDIAN`` (101, the median over the
  60,000 standard MNIST training images). Not a stroke, but the obvious confound for detectors that
  fire on every digit except 1s (1s have the least ink: median 58 pixels vs 88 to 136).

``validate`` reports how often each concept is found per class, which is the check
that the rules mean what their names say (e.g. loops are common in 8s and rare in 3s).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy import ndimage

INK = 0.5
HOLE_MIN = 0.004

STROKES = ("upper_loop", "lower_loop", "open_left", "top_bar", "waist", "left_heavy")
STROKES10 = STROKES + ("centre_vstroke", "any_loop", "heavy_ink")
INK_MEDIAN = 101  # median ink-pixel count (x >= 0.5) of the 60,000 standard MNIST training images
STROKE_WORDS = {
    "upper_loop": "a closed upper loop",
    "lower_loop": "a closed lower loop",
    "open_left": "an open left side",
    "top_bar": "a horizontal stroke along the top",
    "waist": "a narrow central crossing (the waist of an 8)",
    "left_heavy": "more ink on the left half than the right",
    "centre_vstroke": "a tall vertical stem",
    "any_loop": "a closed loop",
    "heavy_ink": "a lot of ink",
}
STROKE_SHORT = {"upper_loop": "upper loop", "lower_loop": "lower loop", "open_left": "open left side",
                "top_bar": "top stroke", "waist": "waist", "left_heavy": "left-heavy",
                "centre_vstroke": "stem", "any_loop": "loop", "heavy_ink": "heavy ink"}
STROKE_RULES = {
    "upper_loop": "a hole (enclosed background, 4-connected) centred in the upper half of the digit's bounding box",
    "lower_loop": "a hole centred in the lower half of the bounding box",
    "open_left": "in the middle half of the box (rows 25-75 %), at most 35 % of inked rows reach the left third",
    "top_bar": "a run of ink at least 60 % of the box width in the top 25 % of rows",
    "waist": "an inked row in the 40-60 % band whose ink is at most 40 % of the box width, centred between 30 % and 60 % of it",
    "left_heavy": "the left half of the bounding box holds more ink than the right half",
    "centre_vstroke": "a digit at least 1.6 times taller than wide with an ink path from top to bottom (85 % of rows, "
                      "drifting at most one column per row) starting in the middle 60 % of its width",
    "any_loop": "a closed upper or lower loop (a hole anywhere)",
    "heavy_ink": f"more than {INK_MEDIAN} ink pixels (x >= 0.5; the median of the MNIST training images)",
}


def _one(x: np.ndarray) -> Dict[str, int]:
    ink = x >= INK
    out = {s: 0 for s in STROKES10}
    if ink.sum() < 2:
        return out
    rows, cols = np.flatnonzero(ink.any(1)), np.flatnonzero(ink.any(0))
    r0, r1, c0, c1 = rows[0], rows[-1], cols[0], cols[-1]
    h, w = r1 - r0 + 1, c1 - c0 + 1
    H, W = x.shape
    # holes: background components not touching the border
    lab, n = ndimage.label(~ink)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    amin = max(1, int(round(HOLE_MIN * H * W)))
    for j in range(1, n + 1):
        if j in border:
            continue
        yy, _ = np.nonzero(lab == j)
        if len(yy) < amin:
            continue
        cy = (yy.mean() - r0 + 0.5) / h
        out["upper_loop" if cy < 0.5 else "lower_loop"] = 1
    box = ink[r0:r1 + 1, c0:c1 + 1]
    xb = x[r0:r1 + 1, c0:c1 + 1]
    rr = np.arange(h)
    # open left side
    band = box[(rr >= 0.25 * h) & (rr < 0.75 * h)]
    inked = band[band.any(1)]
    if len(inked):
        left = inked[:, : max(1, int(np.ceil(w / 3)))].any(1)
        out["open_left"] = int(left.mean() <= 0.35)
    # top horizontal stroke
    for row in box[rr < max(1, 0.25 * h)]:
        run = best = 0
        for v in row:
            run = run + 1 if v else 0
            best = max(best, run)
        if best >= 0.6 * w:
            out["top_bar"] = 1
            break
    # waist
    for i in rr[(rr >= 0.4 * h - 0.5) & (rr <= 0.6 * h - 0.5)]:
        cs = np.flatnonzero(box[i])
        if len(cs) == 0:
            continue
        ext = cs[-1] - cs[0] + 1
        mid = (cs[0] + cs[-1] + 1) / 2 / w
        if ext <= 0.4 * w and 0.3 <= mid <= 0.6:
            out["waist"] = 1
            break
    # left vs right mass
    half = w / 2
    cc = np.arange(w) + 0.5
    out["left_heavy"] = int(xb[:, cc < half].sum() > xb[:, cc > half].sum())
    # tall vertical stem through the middle (followed down the digit, drifting <= 1 column per row)
    if h >= 1.6 * w:
        for c in range(w):
            if not 0.2 <= (c + 0.5) / w <= 0.8:
                continue
            cur, run, gap, best = c, 0, 0, 0
            for r in range(h):
                lo_ = max(0, cur - 1)
                seg = np.flatnonzero(box[r, lo_:cur + 3]) + lo_
                if len(seg):
                    cur = int(seg[np.argmin(np.abs(seg - cur))])
                    run, gap = run + 1 + gap, 0
                elif run and gap == 0:
                    gap = 1
                else:
                    run, gap = 0, 0
                best = max(best, run)
            if best >= 0.85 * h:
                out["centre_vstroke"] = 1
                break
    out["any_loop"] = int(out["upper_loop"] or out["lower_loop"])
    out["heavy_ink"] = int(ink.sum() > INK_MEDIAN)
    return out


def stroke_concepts(X: np.ndarray, names=STROKES) -> np.ndarray:
    """X [N, 1, H, W] (or [N, H, W]) in [0, 1] -> int matrix [N, len(names)]."""
    X = X[:, 0] if X.ndim == 4 else X
    return np.array([[_one(x)[s] for s in names] for x in X], dtype=np.int64)


def validate(C: np.ndarray, y: np.ndarray, class_names: List[str], names=STROKES) -> Dict[str, Dict[str, float]]:
    """Share of images of each class in which each concept is found."""
    return {s: {class_names[c]: float(C[y == c, j].mean()) if (y == c).any() else float("nan")
                for c in range(len(class_names))} for j, s in enumerate(names)}
