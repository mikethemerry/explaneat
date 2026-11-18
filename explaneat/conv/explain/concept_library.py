"""One concept library for handwritten digits, used everywhere a part or a map is checked against
named concepts (annotation cards, the per-map inspector, the behaviour-first investigation).

It merges the three earlier vocabularies, without duplicates:

* ``mechanism.py``: holes (``loop``, ``two_loops``, ``upper_loop``, ``lower_loop``), the stem, the top
  bar, an open left side, stroke ends, junctions, diagonal strokes, the open curves (pockets);
* ``strokes.py``: the ``waist`` (narrow central crossing) and ``left_heavy`` (more ink on the left);
  its loops and stem are the same concepts as mechanism.py's and are not repeated;
* ``shape_concepts.py``: ``one_piece``, ``coherent``, ``filled``;

plus three geometry concepts of the ink's bounding box, so that location and size are named too:
``wide`` (box at least 0.8 as wide as tall), ``slanted`` (the ink's principal axis leans right
by more than 15 degrees from vertical), ``low_ink_centre`` (the ink's centre of mass sits below
the box centre by more than 5 % of the box height).

Every concept is computed from the image alone (never from the label). ``heavy_ink`` (more ink
pixels than the MNIST training median) is in the table too, but it is the *nuisance*: concept
tests are run with the ink count held fixed (partial correlation), and a part whose response is
explained by ink alone is an ink meter, not a shape detector.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from . import mechanism as MC
from .shape_concepts import shape_table
from .strokes import INK, STROKES10, stroke_concepts

ORDER = ("loop", "two_loops", "upper_loop", "lower_loop", "stem", "top_bar", "waist", "open_left",
         "open_upper_left", "open_upper_right", "open_lower_left", "two_left_openings", "open_top",
         "ends_3plus", "junction", "diagonal", "slanted", "wide", "low_ink_centre", "left_heavy",
         "one_piece", "coherent", "filled", "heavy_ink")
NUISANCE = ("heavy_ink",)
_FROM_MECH = {"loop": "hole_any", "two_loops": "two_holes", "upper_loop": "hole_upper", "lower_loop": "hole_lower",
              "stem": "stem", "top_bar": "top_bar", "open_left": "open_left", "open_upper_left": "open_upper_left",
              "open_upper_right": "open_upper_right", "open_lower_left": "open_lower_left",
              "two_left_openings": "two_left_openings", "open_top": "open_top", "ends_3plus": "ends_3plus",
              "junction": "junction", "diagonal": "diagonal", "heavy_ink": "heavy_ink"}
WORDS = {**{u: MC.WORDS[m] for u, m in _FROM_MECH.items()},
         "waist": "a narrow central crossing (the waist of an 8)", "left_heavy": "more ink on the left half than the right",
         "one_piece": "one connected stroke", "coherent": "one coherent shape", "filled": "a filled, blob-like shape",
         "wide": "a wide digit (box at least 0.8 as wide as tall)", "slanted": "a right-leaning slant",
         "low_ink_centre": "ink concentrated in the lower half"}
SHORT = {"loop": "loop", "two_loops": "two loops", "upper_loop": "upper loop", "lower_loop": "lower loop",
         "stem": "tall stem", "top_bar": "top bar", "waist": "waist", "open_left": "open left side",
         "open_upper_left": "upper curve open left", "open_upper_right": "upper curve open right",
         "open_lower_left": "lower curve open left", "two_left_openings": "two left openings", "open_top": "open top",
         "ends_3plus": "3+ stroke ends", "junction": "junction", "diagonal": "diagonal strokes", "slanted": "slant",
         "wide": "wide", "low_ink_centre": "bottom-heavy", "left_heavy": "left-heavy", "one_piece": "one piece",
         "coherent": "coherent", "filled": "filled", "heavy_ink": "a lot of ink"}
RULES = {**{u: MC.COMPUTES[m] for u, m in _FROM_MECH.items()},
         "waist": "an inked row in the 40-60 % band whose ink is at most 40 % of the box width, centred between 30 % and 60 % of it",
         "left_heavy": "the left half of the ink's bounding box holds more ink than the right half",
         "one_piece": MC.COMPUTES["one_piece"], "coherent": MC.COMPUTES["coherent"], "filled": MC.COMPUTES["filled"],
         "wide": "the ink's bounding box is at least 0.8 times as wide as it is tall",
         "slanted": "the principal axis of the ink pixels leans right by more than 15 degrees from vertical",
         "low_ink_centre": "the ink's centre of mass is below the box centre by more than 5 % of the box height"}
# the earlier libraries' names for the same concepts (so stored evidence can be read in one vocabulary)
ALIASES = {**{m: u for u, m in _FROM_MECH.items()}, "any_loop": "loop", "has_loop": "loop", "two_loops": "two_loops",
           "centre_vstroke": "stem", "upper_loop": "upper_loop", "lower_loop": "lower_loop"}


def _geometry(X: np.ndarray) -> Dict[str, np.ndarray]:
    N = len(X)
    wide, slant, low = np.zeros(N, bool), np.zeros(N, bool), np.zeros(N, bool)
    for i, x in enumerate(X):
        ink = x >= INK
        if ink.sum() < 4:
            continue
        rr, cc = np.nonzero(ink)
        r0, r1, c0, c1 = rr.min(), rr.max(), cc.min(), cc.max()
        h, w = r1 - r0 + 1, c1 - c0 + 1
        wide[i] = w >= 0.8 * h
        low[i] = (rr.mean() - (r0 + r1) / 2) > 0.05 * h
        dy, dx = rr - rr.mean(), cc - cc.mean()
        cov = np.array([[np.mean(dy * dy), np.mean(dy * dx)], [np.mean(dy * dx), np.mean(dx * dx)]])
        ev, evec = np.linalg.eigh(cov)
        v = evec[:, 1]  # principal axis (row, col)
        if v[0] < 0:
            v = -v
        # leaning right = going down the axis moves left (col decreases as row increases)
        ang = np.degrees(np.arctan2(-v[1], v[0]))
        slant[i] = ang > 15
    return {"wide": wide, "slanted": slant, "low_ink_centre": low}


def unified_concepts(X: np.ndarray) -> dict:
    """X [N, 1, H, W] or [N, H, W] in [0, 1] -> {"names", "C" int [N, K], "words", "short", "rules",
    "ink" (ink-pixel counts, the nuisance), "nuisance"}."""
    X3 = X[:, 0] if X.ndim == 4 else X
    M = MC.mechanism_concepts(X3)
    jm = {c: i for i, c in enumerate(M["names"])}
    st = stroke_concepts(X3[:, None], STROKES10)
    js = {c: i for i, c in enumerate(STROKES10)}
    sh = shape_table(X3)
    geo = _geometry(X3)
    cols = {}
    for u in ORDER:
        if u in _FROM_MECH:
            cols[u] = M["C"][:, jm[_FROM_MECH[u]]]
        elif u in ("waist", "left_heavy"):
            cols[u] = st[:, js[u]]
        elif u in ("one_piece", "coherent", "filled"):
            cols[u] = np.asarray(sh[u])
        else:
            cols[u] = geo[u]
    C = np.column_stack([np.asarray(cols[u]).astype(np.int64) for u in ORDER])
    return {"names": list(ORDER), "C": C, "words": dict(WORDS), "short": dict(SHORT), "rules": dict(RULES),
            "ink": M["counts"]["ink"].astype(np.float64), "nuisance": list(NUISANCE), "counts": M["counts"]}
