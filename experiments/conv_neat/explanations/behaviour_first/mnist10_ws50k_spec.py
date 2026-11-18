"""Behaviour-first hypotheses for mnist10_ws50k (read from the behaviour panels in ``mnist10_ws50k/look/``,
the kernels of each part and scratch probes on the 2,000 discovery images, written before the concept
check; see BEHAVIOUR_FIRST.md). Each part was read afresh: 34 of the 38 maps share their keys with
mnist10_pruned, but the weights were retrained on 50k images, so no reading was copied.

Every number in the texts is a placeholder filled by ``behaviour_first.py write`` from the computed
results: ``{m[...]}`` the hypothesis measure (discovery images), ``{p[<probe id>][...]}`` a self-probe
(probe images, disjoint), ``{c[...]}`` composition statistics. Probe specs are
``explaneat.conv.explain.investigate.edit_one`` specs. Literal numbers in the texts are edit sizes and
image coordinates (design constants), not results.

The probe bar (a directional probe agrees only at 0.3 SD or more; 0.2 to 0.3 SD is "too small") is the
shared rule of ``investigate.probe_verdict``.
"""

import numpy as np
from scipy import ndimage

from explaneat.conv.explain import investigate as IV


# ------------------------------------------------------------------ edits and measures
def bar(at, angle, length=10, frame="box"):
    return {"kind": "bar", "at": at, "angle": angle, "length": length, "frame": frame}


H_MID, V_MID = bar((0.5, 0.5), 0), bar((0.5, 0.5), 90)
RISE, FALL = bar((0.5, 0.5), 45), bar((0.5, 0.5), -45)
SHIFT1 = {"kind": "shift", "dc": 1}
DILATE = {"kind": "dilate", "size": 2}
WIDER, TALLER = {"kind": "stretch", "fx": 1.3}, {"kind": "stretch", "fy": 1.3}
LEAN_RIGHT, LEAN_LEFT = {"kind": "shear", "s": -0.3}, {"kind": "shear", "s": 0.3}
FILL, SAME = {"kind": "fill_holes"}, {"kind": "identity"}
HOLE_DIGITS = [0, 2, 4, 6, 8, 9]


def ink(X):
    return IV.ink_count(X)


def edges_in(X, kind, r0=0, r1=28, c0=0, c1=28):
    """Edge energy of strokes of one orientation (as ``IV.orientation_energy``) in an image region."""
    X3 = (X[:, 0] if X.ndim == 4 else X).astype(np.float64)
    gy = ndimage.sobel(X3, axis=1)
    gx = ndimage.sobel(X3, axis=2)
    g = {"horizontal": gy, "vertical": gx, "diag_up": (gx + gy) / np.sqrt(2), "diag_down": (gx - gy) / np.sqrt(2)}[kind]
    return (g[:, r0:r1, c0:c1] ** 2).reshape(len(X3), -1).sum(1)


def rows_with_ink(X):
    X3 = (X[:, 0] if X.ndim == 4 else X) >= 0.5
    return X3.any(2).sum(1).astype(np.float64)


def rising_minus_falling(X):
    e = IV.orientation_energy(X)
    return e["diag_up"] - e["diag_down"]


# ------------------------------------------------------------------ leaves
PARTS = {
    "S1.1": {
        "name": "Right-hand stroke-edge detector",
        "mechanism": "It reads the image through one 3x3 filter whose centre and right-hand weights are negative and whose "
                     "left, top and bottom weights are positive, with a threshold that only dark ink clears. So it fires on a "
                     "blank pixel that has dark ink on its left (and above and below it): the right-hand edge of a stroke, "
                     "including the inside edge of a loop. Its average measures how much upright stroke edge the digit has, "
                     "not how much ink: a thick stroke has no more right-hand edge than a thin one, and filling a loop removes "
                     "the loop's inside edge.",
        "prediction": "So an upright stroke should raise it more than the same stroke drawn horizontally, making the digit "
                      "30% taller should raise it more than making it 30% wider, and filling a digit's enclosed holes with "
                      "ink should lower it, the opposite of what an ink meter would do.",
        "measure": {"label": "the amount of upright-stroke edge", "fn": lambda X: IV.orientation_energy(X)["vertical"], "predict": 1},
        "probes": [
            {"id": "v_vs_h", "words": "a 10-pixel upright stroke through the centre of the digit vs the same stroke horizontal",
             "test": V_MID, "control": H_MID, "predict": 1},
            {"id": "taller", "words": "the digit stretched 30% taller vs 30% wider", "test": TALLER, "control": WIDER, "predict": 1},
            {"id": "fill", "words": "the digit's enclosed holes filled with ink vs the digit unchanged (digits with holes)",
             "test": FILL, "control": SAME, "predict": -1, "digits": HOLE_DIGITS},
        ],
    },
    "S1.2": {
        "name": "Ink meter tilted to rising strokes",
        "mechanism": "It reads the image through a 3x3 filter weighted along a line from lower left to upper right ('/'), "
                     "minus S1.1's right-hand-edge map. Its average follows the amount of ink closely (Spearman r = {m[within]:.2f} "
                     "within digits), and beyond the ink it is higher for digits whose strokes lean right ('/') and lower "
                     "where S1.1 finds solid right-hand edges.",
        "prediction": "So thickening the strokes should raise it strongly, and leaning the whole digit to the right should "
                      "raise it more than leaning it to the left by the same shear (the same ink either way).",
        "measure": {"label": "the amount of ink (ink pixels)", "fn": ink, "predict": 1, "is_ink": True},
        "probes": [
            {"id": "thicken", "words": "thicken every stroke by one pixel (dilation) vs move the digit one pixel right",
             "test": DILATE, "control": SHIFT1, "predict": 1},
            {"id": "lean_right", "words": "the digit sheared to lean right vs to lean left (0.3 pixels per row)",
             "test": LEAN_RIGHT, "control": LEAN_LEFT, "predict": 1},
        ],
    },
    "S1.3": {
        "name": "Ink-amount meter",
        "mechanism": "It reads the image through two filters (3x3, then 5x5 at half resolution) and its 14x14 map is a "
                     "blurred copy of the ink: every dark pixel raises the map around it, whatever the stroke's direction or place. Its average "
                     "is an ink meter (Spearman r = {m[within]:.3f} with the ink count within digits).",
        "prediction": "So thickening the strokes should raise it strongly, while a stroke of the same length should add about "
                      "the same wherever it is placed and whichever way it runs, and stretching the digit taller or wider by "
                      "the same factor should make no difference.",
        "measure": {"label": "the amount of ink (ink pixels)", "fn": ink, "predict": 1, "is_ink": True},
        "probes": [
            {"id": "thicken", "words": "thicken every stroke by one pixel (dilation) vs move the digit one pixel right",
             "test": DILATE, "control": SHIFT1, "predict": 1},
            {"id": "place", "words": "a 10-pixel horizontal stroke along the top of the digit vs the same stroke upright along its right edge",
             "test": bar((0.0, 0.5), 0), "control": bar((0.5, 1.0), 90), "predict": 0},
            {"id": "taller", "words": "the digit stretched 30% taller vs 30% wider", "test": TALLER, "control": WIDER, "predict": 0},
        ],
    },
    "S1.4": {
        "name": "Left-boundary detector (coarse)",
        "mechanism": "At a coarse 7x7 scale it reads S1.2's ink map with positive weights on the cells to the right and "
                     "negative weights on the cells to the left. So it fires on empty cells that have ink "
                     "just to their right: the left boundary of the digit (and of any stroke with space on its left), most "
                     "strongly in the left half of the image and the lower rows. The longer the digit's left side and the "
                     "more empty space to its left, the higher it is; ink placed to the left of the digit takes that space away.",
        "prediction": "So an upright stroke drawn just right of the digit (a new left boundary with space before it) should "
                      "raise it more than the same stroke drawn horizontally in the same place; the same upright stroke "
                      "should raise it more right of the digit than left of it; and moving the digit 4 pixels right should "
                      "raise it compared with moving it 4 pixels left.",
        "measure": {"label": "the number of image rows the digit spans (the length of its left side)", "fn": rows_with_ink, "predict": 1},
        "probes": [
            {"id": "upright_vs_flat", "words": "a 10-pixel stroke drawn 3 pixels right of the digit, upright vs horizontal",
             "test": bar((0.5, 1.3), 90), "control": bar((0.5, 1.3), 0), "predict": 1},
            {"id": "right_vs_left", "words": "a 10-pixel upright stroke drawn 3 pixels right of the digit vs 3 pixels left of it",
             "test": bar((0.5, 1.3), 90), "control": bar((0.5, -0.3), 90), "predict": 1},
            {"id": "shift_right", "words": "the digit moved 4 pixels right vs 4 pixels left",
             "test": {"kind": "shift", "dc": 4}, "control": {"kind": "shift", "dc": -4}, "predict": 1},
        ],
    },
    "S2": {
        "name": "Horizontal-stroke detector",
        "mechanism": "It reads S1.3's blurred ink map through a 3x3 filter with a positive middle row and negative rows above "
                     "and below: it fires along horizontal strokes (the bars and flat curves at the top, middle and bottom of "
                     "a digit); upright strokes give it little.",
        "prediction": "So a horizontal stroke should raise it more than the same stroke drawn upright, and making the digit "
                      "30% wider should raise it more than making it 30% taller.",
        "measure": {"label": "the amount of horizontal-stroke edge", "fn": lambda X: IV.orientation_energy(X)["horizontal"], "predict": 1},
        "probes": [
            {"id": "h_vs_v", "words": "a 10-pixel horizontal stroke through the centre of the digit vs the same stroke upright",
             "test": H_MID, "control": V_MID, "predict": 1},
            {"id": "wider", "words": "the digit stretched 30% wider vs 30% taller", "test": WIDER, "control": TALLER, "predict": 1},
        ],
    },
    "S3": {
        "name": "Near-silent bottom-border map",
        "mechanism": "It re-reads S1.4's coarse 7x7 map at full resolution through a 3x3 filter whose weights almost cancel "
                     "on a flat block, after a negative bias. So it is zero almost everywhere and fires, very weakly, only "
                     "where S1.4's map meets the image's bottom border or changes between blocks. Its only edge, into S5, has "
                     "a near-zero 1x1 weight, so it does nothing measurable to the scores: it is a vestige, not a feature.",
        "prediction": "So moving the digit down towards the bottom border should wake it more than moving it up, and a "
                      "horizontal stroke along the image's bottom row should raise it more than the same stroke along the "
                      "top row (in units of its own tiny spread).",
        "measure": {"label": "the ink in the bottom 4 image rows", "fn": lambda X: IV.ink_in(X, (24 / 28, 1), (0, 1), frame="image"), "predict": 1},
        "probes": [
            {"id": "shift_down", "words": "the digit moved 4 pixels down vs 4 pixels up",
             "test": {"kind": "shift", "dr": 4}, "control": {"kind": "shift", "dr": -4}, "predict": 1},
            {"id": "bottom_row", "words": "a 10-pixel horizontal stroke along the image's bottom row vs along its top row",
             "test": bar((1.0, 0.5), 0, frame="image"), "control": bar((0.0, 0.5), 0, frame="image"), "predict": 1},
        ],
    },
    "S4": {
        "name": "Upright-left-side detector",
        "mechanism": "Its 7x7 map is high down the left side of the image even for a blank image, and the digit mostly holds "
                     "it back: S2's horizontal-stroke map enters with large negative weights above and to the left of each "
                     "cell, so a horizontal stroke reaching into the upper left of the digit (the top of a 2, 3 or 7) switches "
                     "it off, while an upright stroke on the digit's left side (as in 5, 6 or 0) keeps it high.",
        "prediction": "So an upright stroke along the digit's left edge should raise it more than the same stroke along its "
                      "right edge; in the upper left of the digit a horizontal stroke should lower it compared with an "
                      "upright one; and a horizontal stroke in the upper left should lower it more than the same stroke in "
                      "the lower left.",
        "measure": {"label": "the ink in the left third of the digit's box", "fn": lambda X: IV.ink_in(X, (0, 1), (0, 1 / 3)), "predict": 1},
        "probes": [
            {"id": "left_vs_right", "words": "a 10-pixel upright stroke along the left edge of the digit vs along its right edge",
             "test": bar((0.5, 0.0), 90), "control": bar((0.5, 1.0), 90), "predict": 1},
            {"id": "flat_vs_upright_ul", "words": "at the top left of the digit, a 10-pixel horizontal stroke vs the same stroke upright",
             "test": bar((0.0, 0.3), 0), "control": bar((0.0, 0.3), 90), "predict": -1},
            {"id": "upper_vs_lower", "words": "a 10-pixel horizontal stroke in the upper left of the digit vs in its lower left",
             "test": bar((0.15, 0.2), 0), "control": bar((0.85, 0.2), 0), "predict": -1},
        ],
    },
    "S5": {
        "name": "Top-bar detector",
        "mechanism": "Its 14x14 map fires along a horizontal stroke across the top of the digit (image rows about 2 to 10, left "
                     "of and at the centre): S1.3's ink map enters with a large negative row above each cell, so it needs ink "
                     "with nothing above it, and an upright stroke tip at the top (as in 1, 4 or 6) gives it little.",
        "prediction": "So a horizontal stroke along the top of the digit should raise it far more than the same stroke drawn "
                      "upright there, and more than the same horizontal stroke along the bottom of the digit.",
        "measure": {"label": "the ink in the top quarter of the digit's box", "fn": lambda X: IV.ink_in(X, (0, 0.25), (0, 1)), "predict": 1},
        "probes": [
            {"id": "flat_vs_upright_top", "words": "at the top of the digit, a 10-pixel horizontal stroke vs the same stroke upright",
             "test": bar((0.0, 0.5), 0), "control": bar((0.0, 0.5), 90), "predict": 1},
            {"id": "top_vs_bottom", "words": "a 10-pixel horizontal stroke along the top of the digit vs the same stroke along its bottom",
             "test": bar((0.0, 0.5), 0), "control": bar((1.0, 0.5), 0), "predict": 1},
        ],
    },
    "S6": {
        "name": "Horizontal-stroke detector (second, top bar discounted)",
        "mechanism": "It reads S2's horizontal-stroke map (positive) and S5's top-bar map (negative), so it again fires along "
                     "horizontal strokes, with the top bar counted less than strokes lower down. Its main exit and S2's agree "
                     "closely (r = {c[r_other]:.2f} over the discovery images).",
        "prediction": "So a horizontal stroke should raise it more than the same stroke drawn upright, making the digit wider "
                      "should raise it more than making it taller, and a long horizontal stroke along the top of the digit "
                      "should raise it less than the same stroke along the bottom.",
        "measure": {"label": "the amount of horizontal-stroke edge", "fn": lambda X: IV.orientation_energy(X)["horizontal"], "predict": 1},
        "probes": [
            {"id": "h_vs_v", "words": "a 10-pixel horizontal stroke through the centre of the digit vs the same stroke upright",
             "test": H_MID, "control": V_MID, "predict": 1},
            {"id": "wider", "words": "the digit stretched 30% wider vs 30% taller", "test": WIDER, "control": TALLER, "predict": 1},
            {"id": "top_vs_bottom", "words": "a 14-pixel horizontal stroke along the top of the digit vs along its bottom",
             "test": bar((0.0, 0.5), 0, length=14), "control": bar((1.0, 0.5), 0, length=14), "predict": -1},
        ],
        "corr_with": "11_abdefg",
    },
    "S7": {
        "name": "Low-right horizontal-stroke detector",
        "mechanism": "It reads S2's horizontal-stroke map (and S6's, and the image a little) by position: a horizontal stroke "
                     "low in the image and right of centre (about image rows 15 to 24, columns 14 to 24) raises it, the same "
                     "stroke higher up, further left or drawn upright gives little, and horizontal ink near the top rows "
                     "holds it back.",
        "prediction": "So a horizontal stroke at image row 20, right of centre, should raise it far more than the same stroke "
                      "at row 5; at row 20 a horizontal stroke should raise it more than an upright one; and at row 20 the "
                      "stroke should raise it more right of centre (column 19) than left of centre (column 8).",
        "measure": {"label": "the horizontal-stroke edge low on the right (rows 14 to 25, columns 14 to 27) minus that in the top 8 rows",
                    "fn": lambda X: edges_in(X, "horizontal", 14, 26, 14, 28) - edges_in(X, "horizontal", 0, 8, 0, 28), "predict": 1},
        "probes": [
            {"id": "low_vs_top", "words": "a 10-pixel horizontal stroke at image row 20, column 17, vs the same stroke at row 5",
             "test": bar((20 / 27, 17 / 27), 0, frame="image"), "control": bar((5 / 27, 17 / 27), 0, frame="image"), "predict": 1},
            {"id": "flat_vs_upright", "words": "at image row 20, column 17, a 10-pixel horizontal stroke vs the same stroke upright",
             "test": bar((20 / 27, 17 / 27), 0, frame="image"), "control": bar((20 / 27, 17 / 27), 90, frame="image"), "predict": 1},
            {"id": "right_vs_left", "words": "a 10-pixel horizontal stroke at image row 20, column 19, vs at column 8",
             "test": bar((20 / 27, 19 / 27), 0, frame="image"), "control": bar((20 / 27, 8 / 27), 0, frame="image"), "predict": 1},
        ],
    },
    "S8.1": {
        "name": "Rising-stroke ('/') detector",
        "mechanism": "It reads the image through a 3x3 filter with positive weights along a line from lower left to upper "
                     "right and negative weights on either side of it, so it responds to strokes running '/'; strokes the other "
                     "way ('\\') are held back. Every dark pixel on such a line adds, so it also grows with the amount of ink.",
        "prediction": "So leaning the digit to the right should raise it more than leaning it to the left by the same shear, a "
                      "'/' stroke should raise it more than the same stroke '\\', and more than the same stroke upright.",
        "measure": {"label": "the '/' stroke edge minus the '\\' stroke edge", "fn": rising_minus_falling, "predict": 1},
        "probes": [
            {"id": "lean_right", "words": "the digit sheared to lean right vs to lean left (0.3 pixels per row)",
             "test": LEAN_RIGHT, "control": LEAN_LEFT, "predict": 1},
            {"id": "rise_vs_fall", "words": "a 10-pixel '/' stroke through the centre of the digit vs the same stroke '\\'",
             "test": RISE, "control": FALL, "predict": 1},
            {"id": "rise_vs_upright", "words": "a 10-pixel '/' stroke through the centre vs the same stroke upright",
             "test": RISE, "control": V_MID, "predict": 1},
        ],
    },
    "S8.2": {
        "name": "Base-stroke detector",
        "mechanism": "Its full-resolution map fires along a horizontal stroke at the bottom of the digit, its base (the flat "
                     "base of a 2, the bottom curve of a 0, 3, 5 or 6), wherever the digit sits in the image. It reads S2's "
                     "horizontal-stroke map through an inner node whose large negative weights lie below each cell (and "
                     "S8.1's map), so a horizontal stroke with ink below it counts less: it keeps the lowest one.",
        "prediction": "So a horizontal stroke along the bottom of the digit should raise it more than the same stroke along "
                      "the top, and more than the same stroke drawn upright at the bottom; moving the whole digit up or down "
                      "should make no difference; a stroke drawn just below a digit that already has a base (0, 2, 3, 5) "
                      "should lower it compared with the same stroke just above (the old base now has ink below it), and the "
                      "same stroke below a digit that ends in a stem (1, 4, 7, 9) should raise it (it becomes the base).",
        "measure": {"label": "the ink in the bottom quarter of the digit's box", "fn": lambda X: IV.ink_in(X, (0.75, 1), (0, 1)), "predict": 1},
        "probes": [
            {"id": "bottom_vs_top", "words": "a 10-pixel horizontal stroke along the bottom of the digit vs the same stroke along its top",
             "test": bar((1.0, 0.5), 0), "control": bar((0.0, 0.5), 0), "predict": 1},
            {"id": "flat_vs_upright_bottom", "words": "at the bottom of the digit, a 10-pixel horizontal stroke vs the same stroke upright",
             "test": bar((1.0, 0.5), 0), "control": bar((1.0, 0.5), 90), "predict": 1},
            {"id": "shift", "words": "the digit moved 4 pixels down vs 4 pixels up", "test": {"kind": "shift", "dr": 4},
             "control": {"kind": "shift", "dr": -4}, "predict": 0},
            {"id": "below_based", "words": "on 0s, 2s, 3s and 5s, a 10-pixel horizontal stroke just below the digit vs the same stroke just above it",
             "test": bar((1.15, 0.5), 0), "control": bar((-0.15, 0.5), 0), "predict": -1, "digits": [0, 2, 3, 5]},
            {"id": "below_stem", "words": "on 1s, 4s, 7s and 9s, the same stroke below vs above the digit",
             "test": bar((1.15, 0.5), 0), "control": bar((-0.15, 0.5), 0), "predict": 1, "digits": [1, 4, 7, 9]},
        ],
    },
    "S8.3": {
        "name": "Lower rising-stroke detector",
        "mechanism": "A coarse 7x7 map that reads S8.1's '/' map and S8.2's inner no-stroke-below map (and S2, S6 and S7 a little): it fires along strokes running '/' and "
                     "along the underside of the digit's lowest strokes, mostly in the lower half and the left. The underside "
                     "part needs empty rows below the digit, so a digit sitting high in the image raises it; whether a stroke "
                     "is upright or horizontal matters little.",
        "prediction": "So leaning the digit to the right should raise it more than leaning it to the left, a '/' stroke should "
                      "raise it more than the same stroke '\\', and moving the digit 4 pixels up should raise it compared with "
                      "moving it 4 pixels down.",
        "measure": {"label": "the '/' stroke edge minus the '\\' stroke edge", "fn": rising_minus_falling, "predict": 1},
        "probes": [
            {"id": "lean_right", "words": "the digit sheared to lean right vs to lean left (0.3 pixels per row)",
             "test": LEAN_RIGHT, "control": LEAN_LEFT, "predict": 1},
            {"id": "rise_vs_fall", "words": "a 10-pixel '/' stroke through the centre of the digit vs the same stroke '\\'",
             "test": RISE, "control": FALL, "predict": 1},
            {"id": "shift_up", "words": "the digit moved 4 pixels up vs 4 pixels down",
             "test": {"kind": "shift", "dr": -4}, "control": {"kind": "shift", "dr": 4}, "predict": 1},
        ],
    },
    "S8.4": {
        "name": "Falling-stroke ('\\') edge detector",
        "mechanism": "It reads the image through a 3x3 filter that needs ink at a pixel and up-left of it but no ink directly "
                     "left of it: the left-hand edge of a stroke falling from upper left to lower right ('\\'), such as the "
                     "inner right wall of a loop. Thick strokes and filled loops leave fewer such "
                     "edges, so it is not an ink meter.",
        "prediction": "So a '\\' stroke should raise it more than the same stroke '/', leaning the digit to the left should "
                      "raise it more than leaning it to the right, and filling a digit's enclosed holes with ink should "
                      "lower it.",
        "measure": {"label": "the amount of '\\' stroke edge", "fn": lambda X: IV.orientation_energy(X)["diag_down"], "predict": 1},
        "probes": [
            {"id": "fall_vs_rise", "words": "a 10-pixel '\\' stroke through the centre of the digit vs the same stroke '/'",
             "test": FALL, "control": RISE, "predict": 1},
            {"id": "lean_left", "words": "the digit sheared to lean left vs to lean right (0.3 pixels per row)",
             "test": LEAN_LEFT, "control": LEAN_RIGHT, "predict": 1},
            {"id": "fill", "words": "the digit's enclosed holes filled with ink vs the digit unchanged (digits with holes)",
             "test": FILL, "control": SAME, "predict": -1, "digits": HOLE_DIGITS},
        ],
    },
}

COMPOSITIONS = {
    "S1": {
        "name": "Edges, ink and the left boundary from the image",
        "combine": "S1.1 (right-hand stroke edges) reads the image and feeds S1.2 (ink tilted to '/' strokes), which subtracts "
                   "it; S1.2 feeds S1.3 (the ink meter) and S1.4 (the left boundary). So the four run in series from one "
                   "image filter to a coarse map, and each hands a different reading to later structures: the ink amount "
                   "(S1.3, to S2, S4 and S5), the right-hand edges (S1.1, to S8.3 and the scores) and the left boundary "
                   "(S1.4, to S3, S4, S5 and the scores).",
        "adds": "What the combination adds is that the same image is read three different ways: filling a digit's holes "
                "moves S1.1 by {p[S1.1:fill][effect_sd]:+.2f} SD and S1.3 by {p[S1.3:fill][effect_sd]:+.2f} SD, and moving "
                "the digit right instead of left moves S1.4 by {p[S1.4:shift_right][effect_sd]:+.2f} SD and S1.3 by only "
                "{p[S1.3:shift_right][effect_sd]:+.2f} SD. A 10-class classifier on all four children's pooled exits reaches "
                "{c[acc_both]:.0%} (cross-validated), against {c[acc_best]:.0%} for the best child alone ({c[best_child]}).",
        "probes": [
            {"id": "fill", "child": "S1.1", "predict": -1, "digits": HOLE_DIGITS,
             "words": "the digit's holes filled with ink vs unchanged (S1.1)", "test": FILL, "control": SAME},
            {"id": "fill", "child": "S1.3", "predict": 1, "digits": HOLE_DIGITS,
             "words": "the same edit (S1.3)", "test": FILL, "control": SAME},
            {"id": "shift_right", "child": "S1.4", "predict": 1, "words": "the digit moved 4 pixels right vs 4 pixels left (S1.4)",
             "test": {"kind": "shift", "dc": 4}, "control": {"kind": "shift", "dc": -4}},
            {"id": "shift_right", "child": "S1.3", "predict": 0, "words": "the same edit (S1.3)",
             "test": {"kind": "shift", "dc": 4}, "control": {"kind": "shift", "dc": -4}},
        ],
        "probe_note": "Filling holes adds ink, so the ink meter rises while the edge detector falls; moving the digit leaves the "
                      "ink meter unchanged while the left-boundary detector follows the space on the digit's left.",
    },
    "S8": {
        "name": "Diagonal strokes and the base",
        "combine": "S8.1 ('/' strokes) reads the image and feeds S8.2 (the base stroke), S8.3 (lower '/' strokes) and S8.4 "
                   "('\\' edges); S8.2 also feeds S8.3. S8.4 reads the image again with the opposite diagonal (and S8.1's "
                   "path through an inner node). Every child except S8.1 feeds the scores directly.",
        "adds": "What the combination adds is both diagonals and the base: a '\\' stroke instead of a '/' moves S8.4 by "
                "{p[S8.4:fall_vs_rise][effect_sd]:+.2f} SD and S8.1 by {p[S8.1:fall_vs_rise][effect_sd]:+.2f} SD, and a "
                "horizontal stroke along the bottom instead of the top moves S8.2 by {p[S8.2:bottom_vs_top][effect_sd]:+.2f} SD "
                "and S8.1 by {p[S8.1:bottom_vs_top][effect_sd]:+.2f} SD. A 10-class classifier on all four children's pooled "
                "exits reaches {c[acc_both]:.0%} (cross-validated), against {c[acc_best]:.0%} for the best child alone "
                "({c[best_child]}).",
        "probes": [
            {"id": "fall_vs_rise", "child": "S8.4", "predict": 1, "words": "a '\\' stroke through the centre vs the same stroke '/' (S8.4)",
             "test": FALL, "control": RISE},
            {"id": "fall_vs_rise", "child": "S8.1", "predict": -1, "words": "the same edit (S8.1)", "test": FALL, "control": RISE},
            {"id": "bottom_vs_top", "child": "S8.2", "predict": 1, "words": "a horizontal stroke along the bottom vs the top of the digit (S8.2)",
             "test": bar((1.0, 0.5), 0), "control": bar((0.0, 0.5), 0)},
            {"id": "bottom_vs_top", "child": "S8.1", "predict": 0, "words": "the same edit (S8.1)",
             "test": bar((1.0, 0.5), 0), "control": bar((0.0, 0.5), 0)},
        ],
        "probe_note": "The two diagonal detectors move in opposite directions under the same edit, and the base detector "
                      "responds to where a horizontal stroke is while S8.1 does not.",
    },
}

ROOT = {
    "probes": [
        {"id": "four_topbar", "digit": 4, "words": "on 4s, a 10-pixel horizontal stroke along the top of the digit vs the same stroke upright there (the '4' score margin)",
         "test": bar((0.0, 0.5), 0), "control": bar((0.0, 0.5), 90), "predict": -1, "wakes": "S5"},
        {"id": "six_topbar", "digit": 6, "words": "on 6s, a 10-pixel horizontal stroke along the top of the digit vs the same stroke upright there (the '6' score margin)",
         "test": bar((0.0, 0.5), 0), "control": bar((0.0, 0.5), 90), "predict": -1, "wakes": "S5"},
        {"id": "one_fall", "digit": 1, "words": "on 1s, a 10-pixel '\\' stroke through the centre vs the same stroke upright, along the 1 (the '1' score margin; wakes S8.4)",
         "test": FALL, "control": V_MID, "predict": -1, "wakes": "S8"},
        {"id": "one_base", "digit": 1, "words": "on 1s, a 10-pixel horizontal stroke along the bottom of the digit vs the same stroke upright there (the '1' score margin; wakes S8.2)",
         "test": bar((1.0, 0.5), 0), "control": bar((1.0, 0.5), 90), "predict": -1, "wakes": "S8"},
        {"id": "seven_base", "digit": 7, "words": "on 7s, a 10-pixel horizontal stroke along the bottom of the digit vs the same stroke upright there (the '7' score margin; wakes S8.2)",
         "test": bar((1.0, 0.5), 0), "control": bar((1.0, 0.5), 90), "predict": -1, "wakes": "S8"},
    ],
}
