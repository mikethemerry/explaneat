"""Mechanism concepts for digits, computed from the image alone (never from the label), and
cheap image interventions that add or remove them.

Concepts (on the ink mask ``x >= INK``, relative to the ink's bounding box):

* **holes**: background components (4-connected) not connected to the image border, of at least
  ``HOLE_MIN_PX`` pixels; ``hole_any`` (at least one), ``two_holes`` (at least two),
  ``hole_upper`` / ``hole_lower`` (a hole centred in the upper / lower half of the box);
* **stem**: a tall vertical stem (``strokes.py`` ``centre_vstroke``);
* **top_bar**: a horizontal stroke along the top (``strokes.py``);
* **open_left**: an open left side in the middle rows (``strokes.py``);
* **skeleton**: the ink mask thinned to one pixel (Zhang-Suen, numpy), spurs of up to 3 pixels
  pruned; ``ends_3plus``: at least
  three stroke ends (skeleton pixels with one neighbour, clusters counted once);
  ``junction``: at least one junction (skeleton pixels whose crossing number, the 0 -> 1
  transitions around them, is at least 3; clustered);
* **diagonal**: the share of strong-gradient ink-edge pixels whose gradient points within 22.5
  degrees of a diagonal is above the median of the explained images;
* **open curves** (pockets): background pixels inside the ink box, not in a hole, with ink on three
  of their four sides (rays up, down, left, right) and none on the fourth. ``open_upper_left``: at
  least ``POCKET_MIN`` of the box area such pixels opening to the left in the upper half (the
  upper curve of a 2, 3 or 7); ``open_upper_right``: opening to the right in the upper half (5,
  6); ``open_lower_left``: opening to the left in the lower half (3, 5, 9); ``two_left_openings``:
  both left openings, one above the other (a 3); ``open_top``: opening upwards anywhere (a 4);
* **ink** (a nuisance, never a mechanism on its own): the number of ink pixels, and
  ``heavy_ink`` = more than the MNIST training median (``strokes.INK_MEDIAN``).

Interventions (each with a matched control edit, so the edit itself is not mistaken for the
concept; the effect is the difference of the two changes, in SDs of the response):

* ``close``: grey dilation (3x3 max); test images = those in which dilation *creates* a hole,
  control = those in which it does not (same edit, no new hole);
* ``fill``: holes filled with ink; control = the same number of pixels inked just outside the
  digit's outer edge;
* ``add_stem``: a 2-pixel vertical line down the middle of the box; control = a horizontal line
  of the same length through the box centre;
* ``add_top_bar``: a 2-pixel horizontal line along the top of the box; control = a vertical line
  of the same length down the box's left edge;
* ``close_upper``: a 2-pixel vertical line across the left mouth of the upper left-open curve,
  from the ink above it to the ink below it (the curve becomes a loop); control = the same number
  of pixels as a vertical line just right of the box, not touching the digit;
* ``open_upper``: the ink left of an upper hole is erased over the hole's middle rows (the loop
  becomes a curve open to the left); control = the same number of ink pixels erased from the
  bottom rows of the digit;
* ``shuffle_ink``: the pixels inside the ink's bounding box randomly permuted (same ink, same box,
  no strokes); control = the digit mirrored left-right (same ink and box, strokes kept);
* ``patch_box``: the ink's bounding box cut into 3 x 3 tiles that are permuted (strokes broken at
  the tile edges, ink position kept); control = the digit mirrored left-right;
* ``bridge_loop``: two free stroke ends (skeleton end points) joined by a 2-pixel line where that
  closes a new loop; control = a line of the same length from one of the ends, pointing away from
  the other (no new loop);
* ``bridge_left``: on images with a left-open curve, a 2-pixel vertical line along the left edge of
  the box over the middle half of its rows (closes the left side); control = the same line down
  the box's centre column (same ink, inside the box, the left side stays open).

``edit_sign(concept, kind)`` is the direction in which an edit moves a concept (+1 adds it, -1
removes it).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy import ndimage

from .strokes import INK, INK_MEDIAN, STROKES10, stroke_concepts

HOLE_MIN_PX = 3

MECH = ("hole_any", "two_holes", "hole_upper", "hole_lower", "stem", "top_bar", "open_left", "ends_3plus", "junction",
        "diagonal", "heavy_ink", "open_upper_left", "open_upper_right", "open_lower_left", "two_left_openings", "open_top")
POCKET_MIN = 0.03
# shape concepts (``shape_concepts.py``), added to a library on request (``library_mech(shape=True)``)
SHAPE = ("one_piece", "coherent", "filled")
NUISANCE = ("heavy_ink",)
WORDS = {"hole_any": "a closed loop", "two_holes": "two closed loops", "hole_upper": "a closed upper loop",
         "hole_lower": "a closed lower loop", "stem": "a tall vertical stem", "top_bar": "a horizontal bar along the top",
         "ends_3plus": "three or more free stroke ends", "junction": "a stroke junction (three strokes meeting)",
         "diagonal": "mostly diagonal strokes", "heavy_ink": "a lot of ink", "open_left": "an open left side",
         "open_upper_left": "an upper curve open to the left", "open_upper_right": "an upper curve open to the right",
         "open_lower_left": "a lower curve open to the left", "two_left_openings": "two curves open to the left, one above the other",
         "open_top": "an opening at the top", "one_piece": "one connected stroke",
         "coherent": "one coherent shape", "filled": "a filled shape"}
SHORT = {"hole_any": "loop", "two_holes": "two-loop", "hole_upper": "upper-loop", "hole_lower": "lower-loop",
         "stem": "stem", "top_bar": "top-bar", "ends_3plus": "stroke-end", "junction": "junction",
         "diagonal": "diagonal-stroke", "heavy_ink": "ink", "open_left": "open-left",
         "open_upper_left": "left-open upper curve", "open_upper_right": "right-open upper curve",
         "open_lower_left": "left-open lower curve", "two_left_openings": "double left opening", "open_top": "open-top",
         "one_piece": "one-piece", "coherent": "coherent-shape", "filled": "filled-shape"}
NAME_POS = {"hole_any": "Closed-loop detector", "two_holes": "Two-loop detector", "hole_upper": "Upper-loop detector",
            "hole_lower": "Lower-loop detector", "stem": "Stem detector", "top_bar": "Top-bar detector",
            "ends_3plus": "Stroke-end detector", "junction": "Junction detector", "diagonal": "Diagonal-stroke detector",
            "heavy_ink": "Ink-amount meter", "open_left": "Open-left-side detector",
            "open_upper_left": "Left-open upper-curve detector", "open_upper_right": "Right-open upper-curve detector",
            "open_lower_left": "Left-open lower-curve detector", "two_left_openings": "Double-left-opening (3-shape) detector",
            "open_top": "Open-top (4-shape) detector", "one_piece": "Connected-stroke detector",
            "coherent": "Coherent-shape (digit strokes) detector", "filled": "Filled-blob detector"}
NAME_NEG = {"hole_any": "Open-shape detector", "two_holes": "Single-loop detector", "hole_upper": "No-upper-loop detector",
            "hole_lower": "Open-bottom detector", "stem": "No-stem detector", "top_bar": "No-top-bar detector",
            "ends_3plus": "Few-ends detector", "junction": "No-junction detector", "diagonal": "Straight-stroke detector",
            "heavy_ink": "Light-ink detector", "open_left": "Closed-left-side detector",
            "open_upper_left": "No-left-open-upper-curve detector", "open_upper_right": "No-right-open-upper-curve detector",
            "open_lower_left": "No-left-open-lower-curve detector", "two_left_openings": "No-double-left-opening detector",
            "open_top": "Closed-top detector", "one_piece": "Broken-stroke detector",
            "coherent": "Scattered-ink detector", "filled": "Thin-stroke detector"}
COMPUTES = {"hole_any": "enclosed background, paper fully surrounded by ink",
            "two_holes": "digits with two separate enclosed regions",
            "hole_upper": "an enclosed region in the upper half of the digit",
            "hole_lower": "an enclosed region in the lower half of the digit",
            "stem": "a long, near-vertical stroke running most of the digit's height",
            "top_bar": "a horizontal stroke across the top of the digit",
            "ends_3plus": "free stroke ends (three or more)",
            "junction": "points where three or more strokes meet",
            "diagonal": "strokes that run diagonally rather than straight up or across",
            "heavy_ink": "the amount of ink in the image",
            "open_left": "a digit whose left side is open in the middle rows (as in 3, 7 and many 2s and 1s)",
            "open_upper_left": "an upper curve that opens to the left, ink above, below and to its right (the top of a 2, 3 or 7)",
            "open_upper_right": "an upper curve that opens to the right (the top of a 5 or 6)",
            "open_lower_left": "a lower curve that opens to the left (the bottom of a 3, 5 or 9)",
            "two_left_openings": "two curves opening to the left, one above the other (the shape of a 3)",
            "open_top": "a gap that opens upwards between two strokes (the top of a 4)",
            "one_piece": "ink that forms one connected stroke (8-connected, pieces of at least 3 pixels)",
            "coherent": "ink that forms one coherent shape (at least 90% of the ink in its largest piece)",
            "filled": "ink that fills its bounding box (at least 60% of the box), as a blob does"}
INTERVENTION = {"hole_any": ["close", "fill", "bridge_loop"], "two_holes": ["close", "fill"], "hole_upper": ["fill", "open_upper"],
                "hole_lower": ["fill"], "stem": ["add_stem"], "top_bar": ["add_top_bar"],
                "open_upper_left": ["close_upper", "open_upper"], "open_left": ["bridge_left"],
                "open_lower_left": ["bridge_left"], "two_left_openings": ["bridge_left"],
                "coherent": ["patch_box", "shuffle_ink"], "one_piece": ["patch_box", "shuffle_ink"]}
_SIGN = {"fill": -1, "close_upper": -1, "bridge_left": -1, "patch_box": -1, "shuffle_ink": -1, ("open_upper", "hole_upper"): -1,
         ("open_upper", "open_upper_left"): 1}


def edit_sign(concept: str, kind: str) -> int:
    """+1 if the edit adds ``concept`` to an image, -1 if it removes it."""
    return _SIGN.get((kind, concept), _SIGN.get(kind, 1))


def _rays(ink: np.ndarray):
    """For a batch of masks [N, H, W]: is there ink strictly above / below / left / right of each pixel."""
    N, H, W = ink.shape
    z_r, z_c = np.zeros((N, 1, W), bool), np.zeros((N, H, 1), bool)
    up = np.concatenate([z_r, np.maximum.accumulate(ink, axis=1)[:, :-1]], 1)
    dn = np.concatenate([np.maximum.accumulate(ink[:, ::-1], axis=1)[:, ::-1][:, 1:], z_r], 1)
    lf = np.concatenate([z_c, np.maximum.accumulate(ink, axis=2)[:, :, :-1]], 2)
    rt = np.concatenate([np.maximum.accumulate(ink[:, :, ::-1], axis=2)[:, :, ::-1][:, :, 1:], z_c], 2)
    return up, dn, lf, rt


def pocket_masks(m: np.ndarray, rays=None) -> Dict[str, np.ndarray]:
    """Open-curve (pocket) pixels of one ink mask, by opening direction and half of the ink box."""
    U, D, L, R = rays if rays is not None else (r[0] for r in _rays(m[None]))
    H, W = m.shape
    rows, cols = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
    box = np.zeros_like(m)
    if len(rows) == 0:
        out = {k: box.copy() for k in ("ul", "ll", "ur", "lr", "top")}
        out["area"] = 1
        return out
    r0, r1 = rows[0], rows[-1]
    box[r0:r1 + 1, cols[0]:cols[-1] + 1] = True
    bg = ~m & ~hole_masks(m) & box
    upper = (np.arange(H)[:, None] < (r0 + r1) / 2) & np.ones((1, W), bool)
    oL, oR = bg & U & D & R & ~L, bg & U & D & L & ~R
    return {"ul": oL & upper, "ll": oL & ~upper, "ur": oR & upper, "lr": oR & ~upper,
            "top": bg & D & L & R & ~U, "area": int(box.sum())}


def pocket_shares(ink: np.ndarray) -> Dict[str, np.ndarray]:
    U, D, L, R = _rays(ink)
    out = {k: np.zeros(len(ink)) for k in ("ul", "ll", "ur", "lr", "top")}
    for i, m in enumerate(ink):
        if m.sum() < 4:
            continue
        pm = pocket_masks(m, (U[i], D[i], L[i], R[i]))
        for k in out:
            out[k][i] = pm[k].sum() / pm["area"]
    return out


def _holes(ink: np.ndarray) -> List[float]:
    """Centre rows (relative to the ink box, 0 = top, 1 = bottom) of the holes of one ink mask."""
    if ink.sum() < 2:
        return []
    lab, n = ndimage.label(~ink)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    rows = np.flatnonzero(ink.any(1))
    r0, h = rows[0], rows[-1] - rows[0] + 1
    out = []
    for j in range(1, n + 1):
        if j in border:
            continue
        yy, _ = np.nonzero(lab == j)
        if len(yy) >= HOLE_MIN_PX:
            out.append((yy.mean() - r0 + 0.5) / h)
    return out


def hole_masks(ink: np.ndarray) -> np.ndarray:
    """Boolean mask of all hole pixels of one ink mask."""
    lab, n = ndimage.label(~ink)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    m = np.zeros_like(ink)
    for j in range(1, n + 1):
        if j not in border and (lab == j).sum() >= HOLE_MIN_PX:
            m |= lab == j
    return m


def thin(ink: np.ndarray, max_iter: int = 40) -> np.ndarray:
    """Zhang-Suen thinning, vectorised over a batch: ink [N, H, W] bool -> skeleton bool."""
    img = np.pad(ink.astype(np.uint8), ((0, 0), (1, 1), (1, 1)))
    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            P = img
            p2, p3, p4 = P[:, :-2, 1:-1], P[:, :-2, 2:], P[:, 1:-1, 2:]
            p5, p6, p7 = P[:, 2:, 2:], P[:, 2:, 1:-1], P[:, 2:, :-2]
            p8, p9 = P[:, 1:-1, :-2], P[:, :-2, :-2]
            c = P[:, 1:-1, 1:-1]
            nb = [p2, p3, p4, p5, p6, p7, p8, p9]
            B = sum(x.astype(np.int16) for x in nb)
            seq = nb + [p2]
            A = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.int16) for i in range(8))
            if step == 0:
                m = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                m = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            rm = (c == 1) & (B >= 2) & (B <= 6) & (A == 1) & m
            if rm.any():
                changed = True
                img = img.copy()
                img[:, 1:-1, 1:-1][rm] = 0
        if not changed:
            break
    return img[:, 1:-1, 1:-1].astype(bool)


def _neighbours(sk: np.ndarray) -> np.ndarray:
    P = np.pad(sk.astype(np.int16), ((0, 0), (1, 1), (1, 1)))
    s = np.zeros(sk.shape, np.int16)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                s += P[:, 1 + dy:P.shape[1] - 1 + dy, 1 + dx:P.shape[2] - 1 + dx]
    return s


def crossing_number(sk: np.ndarray) -> np.ndarray:
    """Number of 0 -> 1 transitions around each pixel's 8-neighbourhood (circular order)."""
    P = np.pad(sk.astype(np.int16), ((0, 0), (1, 1), (1, 1)))
    H, W = sk.shape[1:]
    order = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    nb = [P[:, 1 + dy:1 + dy + H, 1 + dx:1 + dx + W] for dy, dx in order]
    return sum(((nb[i] == 0) & (nb[(i + 1) % 8] == 1)).astype(np.int16) for i in range(8))


def prune_spurs(sk: np.ndarray, n: int) -> np.ndarray:
    """Remove short skeleton spurs: strip end pixels n times, then regrow the surviving branches
    back along the original skeleton n times (so real strokes keep their length)."""
    cur = sk.copy()
    for _ in range(n):
        cur = cur & ~(_neighbours(cur) == 1)
    for _ in range(n):
        grow = (_neighbours(cur) >= 1) & sk & ~cur
        ends = cur & (_neighbours(cur) <= 1)
        near = ndimage.binary_dilation(ends, structure=np.ones((1, 3, 3), bool))
        cur = cur | (grow & near)
    return cur


def _diag_share(x: np.ndarray) -> np.ndarray:
    gy = ndimage.sobel(x, axis=1)
    gx = ndimage.sobel(x, axis=2)
    mag = np.hypot(gx, gy)
    ang = np.degrees(np.arctan2(gy, gx)) % 180
    strong = mag > 0.5 * mag.reshape(len(x), -1).max(1)[:, None, None]
    diag = (np.abs(ang - 45) < 22.5) | (np.abs(ang - 135) < 22.5)
    return (diag & strong).reshape(len(x), -1).sum(1) / np.maximum(1, strong.reshape(len(x), -1).sum(1))


def mechanism_concepts(X: np.ndarray, diag_median: float = None) -> dict:
    """X [N, 1, H, W] or [N, H, W] -> {"C": int [N, len(MECH)], "names": MECH, "counts": {...}}."""
    X = X[:, 0] if X.ndim == 4 else X
    ink = X >= INK
    holes = [_holes(m) for m in ink]
    nh = np.array([len(h) for h in holes])
    st = stroke_concepts(X[:, None], STROKES10)
    jst = {s: i for i, s in enumerate(STROKES10)}
    sk = prune_spurs(thin(ink), 3)
    nb = _neighbours(sk)
    cn = crossing_number(sk)
    ends = np.array([ndimage.label(e)[1] for e in (sk & (nb == 1))])
    junc = np.array([ndimage.label(j)[1] for j in (sk & (cn >= 3))])
    dshare = _diag_share(X.astype(np.float64))
    dm = float(np.median(dshare)) if diag_median is None else diag_median
    n_ink = ink.reshape(len(X), -1).sum(1)
    pk = pocket_shares(ink)
    cols = {"open_upper_left": pk["ul"] >= POCKET_MIN, "open_upper_right": pk["ur"] >= POCKET_MIN,
            "open_lower_left": pk["ll"] >= POCKET_MIN,
            "two_left_openings": (pk["ul"] >= POCKET_MIN) & (pk["ll"] >= POCKET_MIN), "open_top": pk["top"] >= POCKET_MIN,
            "hole_any": nh >= 1, "two_holes": nh >= 2,
            "hole_upper": np.array([any(c < 0.5 for c in h) for h in holes]),
            "hole_lower": np.array([any(c >= 0.5 for c in h) for h in holes]),
            "stem": st[:, jst["centre_vstroke"]] == 1, "top_bar": st[:, jst["top_bar"]] == 1,
            "open_left": st[:, jst["open_left"]] == 1,
            "ends_3plus": ends >= 3, "junction": junc >= 1, "diagonal": dshare > dm,
            "heavy_ink": n_ink > INK_MEDIAN}
    C = np.column_stack([cols[c].astype(np.int64) for c in MECH])
    return {"C": C, "names": list(MECH), "counts": {"holes": nh, "ends": ends, "junctions": junc, "ink": n_ink,
                                                   "diag_share": dshare, **{f"pocket_{k}": v for k, v in pk.items()}},
            "diag_median": dm}


# ---------------------------------------------------------------------------- interventions
def _box(ink):
    rows, cols = np.flatnonzero(ink.any(1)), np.flatnonzero(ink.any(0))
    return rows[0], rows[-1], cols[0], cols[-1]


def edit(X: np.ndarray, kind: str, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (edited test images, edited control images, mask of images the test applies to).
    X [N, 1, H, W]. For ``close`` the control is the same edit; the mask splits test vs control."""
    x = X[:, 0]
    N, H, W = x.shape
    T, Cc = x.copy(), x.copy()
    ok = np.zeros(N, bool)
    if kind == "close":
        D = ndimage.grey_dilation(x, size=(1, 3, 3))
        before = np.array([len(_holes(m)) for m in x >= INK])
        after = np.array([len(_holes(m)) for m in D >= INK])
        T, Cc = D, D
        ok = after > before
        return T[:, None], Cc[:, None], ok
    for i in range(N):
        ink = x[i] >= INK
        if ink.sum() < 4:
            continue
        r0, r1, c0, c1 = _box(ink)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if kind == "fill":
            hm = hole_masks(ink)
            n = int(hm.sum())
            if n == 0:
                continue
            T[i][hm] = 1.0
            ring = ndimage.binary_dilation(ink | hm) & ~(ink | hm)
            idx = np.flatnonzero(ring.ravel())
            if len(idx) == 0:
                continue
            pick = rng.choice(idx, min(n, len(idx)), replace=False)
            Cc[i].ravel()[pick] = 1.0
            ok[i] = True
        elif kind == "add_stem":
            cm = (c0 + c1) // 2
            T[i, r0:r1 + 1, cm:cm + 2] = 1.0
            rm = (r0 + r1) // 2
            half = (r1 - r0 + 1) // 2
            Cc[i, rm:rm + 2, max(0, (c0 + c1) // 2 - half):min(W, (c0 + c1) // 2 + half + 1)] = 1.0
            ok[i] = True
        elif kind == "close_upper":
            pm = pocket_masks(ink)
            if pm["ul"].sum() < POCKET_MIN * pm["area"]:
                continue
            rr, cc = np.nonzero(pm["ul"])
            cl = cc.min()
            col = ink[:, cl]
            above = np.flatnonzero(col[:rr.min()])
            below = np.flatnonzero(col[rr.max() + 1:])
            if not len(above) or not len(below):
                continue
            a, b = above[-1], rr.max() + 1 + below[0]
            T[i, a:b + 1, cl:cl + 2] = 1.0
            n = int((T[i] != x[i]).sum())
            cc0 = min(W - 2, c1 + 3)
            if cc0 <= c1 + 1:
                continue
            Cc[i, max(0, a):max(0, a) + max(1, n // 2), cc0:cc0 + 2] = 1.0
            ok[i] = True
        elif kind == "open_upper":
            hm = hole_masks(ink)
            lab, nlab = ndimage.label(hm)
            best = None
            for j in range(1, nlab + 1):
                yy, xx = np.nonzero(lab == j)
                if (yy.mean() - r0 + 0.5) / h < 0.5:
                    best = (yy, xx)
            if best is None:
                continue
            yy, xx = best
            ya, yb = int(np.percentile(yy, 33)), int(np.percentile(yy, 67))
            erase = np.zeros_like(ink)
            for r_ in range(ya, yb + 1):
                left = xx[yy == r_].min() if (yy == r_).any() else xx.min()
                erase[r_, c0:left] = ink[r_, c0:left]
            n = int(erase.sum())
            if n == 0:
                continue
            T[i][erase] = 0.0
            idx = np.flatnonzero(ink.ravel())
            low = idx[np.argsort(-(idx // W), kind="stable")][:n]
            Cc[i].ravel()[low] = 0.0
            ok[i] = True
        elif kind == "bridge_left":
            pm = pocket_masks(ink)
            if (pm["ul"] | pm["ll"]).sum() < POCKET_MIN * pm["area"]:
                continue  # nothing open on the left to bridge
            ra, rb = r0 + h // 4, r0 + (3 * h) // 4
            cm = (c0 + c1) // 2
            T[i, ra:rb + 1, c0:c0 + 2] = 1.0
            Cc[i, ra:rb + 1, cm:cm + 2] = 1.0
            ok[i] = True
        elif kind in ("shuffle_ink", "patch_box"):
            box = x[i, r0:r1 + 1, c0:c1 + 1]
            if kind == "shuffle_ink":
                T[i, r0:r1 + 1, c0:c1 + 1] = rng.permutation(box.ravel()).reshape(box.shape)
            else:
                hs, ws = -(-h // 3), -(-w // 3)
                pad = np.zeros((3 * hs, 3 * ws), x.dtype)
                pad[:h, :w] = box
                tiles = pad.reshape(3, hs, 3, ws).transpose(0, 2, 1, 3).reshape(9, hs, ws)
                perm = rng.permutation(9)
                while (perm == np.arange(9)).all():
                    perm = rng.permutation(9)
                out = tiles[perm].reshape(3, 3, hs, ws).transpose(0, 2, 1, 3).reshape(3 * hs, 3 * ws)
                T[i, r0:r1 + 1, c0:c1 + 1] = out[:h, :w]
            Cc[i, r0:r1 + 1, c0:c1 + 1] = box[:, ::-1]
            ok[i] = True
        elif kind == "bridge_loop":
            sk = prune_spurs(thin(ink[None]), 3)[0]
            ends = np.argwhere(sk & (_neighbours(sk[None])[0] == 1))
            if len(ends) < 2:
                continue
            n0 = len(_holes(ink))
            done = False
            for a_ in range(len(ends)):
                for b_ in range(a_ + 1, len(ends)):
                    p, q = ends[a_], ends[b_]
                    L = int(np.ceil(np.hypot(*(q - p)))) + 1
                    if L < 3:
                        continue
                    t_ = np.linspace(0, 1, 2 * L)
                    pts = np.round(p[None] + t_[:, None] * (q - p)[None]).astype(int)
                    Ti = x[i].copy()
                    for (yy, xx) in pts:
                        Ti[max(0, yy):yy + 2, max(0, xx):xx + 2] = 1.0
                    if len(_holes(Ti >= INK)) <= n0:
                        continue
                    d = (p - q) / max(1e-9, np.hypot(*(p - q)))
                    cpts = np.round(p[None] + t_[:, None] * (d * (L - 1))[None]).astype(int)
                    if (cpts < 0).any() or (cpts[:, 0] >= H - 1).any() or (cpts[:, 1] >= W - 1).any():
                        continue
                    Ci = x[i].copy()
                    for (yy, xx) in cpts:
                        Ci[yy:yy + 2, xx:xx + 2] = 1.0
                    if len(_holes(Ci >= INK)) > n0:
                        continue
                    T[i], Cc[i], ok[i], done = Ti, Ci, True, True
                    break
                if done:
                    break
        elif kind == "add_top_bar":
            T[i, r0:r0 + 2, c0:c1 + 1] = 1.0
            Cc[i, r0:min(H, r0 + w), c0:c0 + 2] = 1.0
            ok[i] = True
        else:
            raise ValueError(kind)
    return T[:, None], Cc[:, None], ok


def intervention_effect(pool_fn, X: np.ndarray, kind: str, sd: float, seed: int = 0, n_max: int = 600) -> Dict:
    """pool_fn(images) -> pooled responses (np array). Effect in SDs of the response:
    mean(test change) - mean(control change). For ``close``: images where dilation creates a
    hole vs images where it does not."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    if kind == "close":
        n_max = 2000  # dilation creates a hole in only a few % of images
    if len(idx) > n_max:
        idx = np.sort(rng.choice(idx, n_max, replace=False))
    X = X[idx]
    base = pool_fn(X)
    T, Cc, ok = edit(X, kind, rng)
    if ok.sum() < 10 or (kind == "close" and (~ok).sum() < 10):
        return {"kind": kind, "n": int(ok.sum()), "effect_sd": None}
    dt = pool_fn(T) - base
    if kind == "close":
        d_test, d_ctrl = dt[ok], dt[~ok]
        n_ctrl = int((~ok).sum())
    else:
        dc = pool_fn(Cc) - base
        d_test, d_ctrl = dt[ok], dc[ok]
        n_ctrl = int(ok.sum())
    eff = (float(d_test.mean()) - float(d_ctrl.mean())) / (sd if sd > 0 else 1.0)
    j = int(np.flatnonzero(ok)[np.argmax(np.abs(dt[ok]))]) if ok.any() else None
    return {"kind": kind, "n": int(ok.sum()), "n_control": n_ctrl, "effect_sd": eff,
            "mean_change_test": float(d_test.mean()), "mean_change_control": float(d_ctrl.mean()),
            "example": None if j is None else {"index": int(idx[j]), "before": X[j, 0], "after": T[j, 0],
                                               "resp_before": float(base[j]), "resp_after": float(base[j] + dt[j])}}
