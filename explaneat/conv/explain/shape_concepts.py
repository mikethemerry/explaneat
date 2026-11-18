"""Shape-level concepts for "is there a digit here at all" (3 vs 8 vs noise), from the image alone.

Additive to ``strokes.py`` (whose loop rules look for holes relative to the bounding box). These are
counts and fractions on the ink mask ``x >= INK`` (0.5, as strokes.py):

* ``n_holes``        closed loops: background regions (4-connected) not touching the border, of at
                     least ``HOLE_MIN_PX`` pixels (flood fill). A 3 has 0, an 8 has 2 (or 1).
* ``n_components``   separate strokes: ink regions (8-connected) of at least ``COMP_MIN_PX`` pixels.
                     A handwritten digit is usually one; shuffled pixels are dozens.
* ``largest_frac``   global coherence: the share of ink pixels in the largest ink region (1 = one
                     connected shape). Low for shuffled or patch-scrambled digits.
* ``compactness``    ink pixels / bounding-box area. A blob fills its box; a digit's strokes do not.

Binary versions (for the s = 2 AUC - 1 probes): ``has_loop`` (n_holes >= 1), ``two_loops``
(n_holes >= 2), ``one_piece`` (n_components == 1), ``coherent`` (largest_frac >= 0.9),
``filled`` (compactness >= 0.6).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import ndimage

INK = 0.5
HOLE_MIN_PX = 3
COMP_MIN_PX = 3
NUMERIC = ("n_holes", "n_components", "largest_frac", "compactness")
BINARY = ("has_loop", "two_loops", "one_piece", "coherent", "filled")
WORDS = {"has_loop": "a closed loop", "two_loops": "two closed loops", "one_piece": "one connected stroke",
         "coherent": "one coherent shape (>= 90% of ink in the largest piece)",
         "filled": "a filled shape (ink covers >= 60% of its box)"}
_EIGHT = np.ones((3, 3), bool)


def _one(x: np.ndarray) -> Dict[str, float]:
    ink = x >= INK
    if not ink.any():
        return {"n_holes": 0, "n_components": 0, "largest_frac": 0.0, "compactness": 0.0}
    bg, nb = ndimage.label(~ink)
    border = set(np.unique(np.concatenate([bg[0], bg[-1], bg[:, 0], bg[:, -1]]))) - {0}
    sizes = ndimage.sum(np.ones_like(bg), bg, index=np.arange(1, nb + 1))
    n_holes = sum(1 for i, s in enumerate(sizes, 1) if i not in border and s >= HOLE_MIN_PX)
    lab, nc = ndimage.label(ink, structure=_EIGHT)
    cs = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, nc + 1))
    ys, xs = np.nonzero(ink)
    box = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
    return {"n_holes": int(n_holes), "n_components": int(np.sum(cs >= COMP_MIN_PX)),
            "largest_frac": float(cs.max() / ink.sum()), "compactness": float(ink.sum() / box)}


def shape_table(X: np.ndarray) -> Dict[str, np.ndarray]:
    """X: [N, 1, H, W] or [N, H, W] in [0, 1] -> dict of numeric and binary concept arrays."""
    X = X.reshape(len(X), *X.shape[-2:])
    rows = [_one(x) for x in X]
    out = {k: np.array([r[k] for r in rows]) for k in NUMERIC}
    out["has_loop"] = (out["n_holes"] >= 1).astype(np.int64)
    out["two_loops"] = (out["n_holes"] >= 2).astype(np.int64)
    out["one_piece"] = (out["n_components"] == 1).astype(np.int64)
    out["coherent"] = (out["largest_frac"] >= 0.9).astype(np.int64)
    out["filled"] = (out["compactness"] >= 0.6).astype(np.int64)
    return out
