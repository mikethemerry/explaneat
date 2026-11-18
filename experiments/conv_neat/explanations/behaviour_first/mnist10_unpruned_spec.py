"""Behaviour-first hypotheses for mnist10_unpruned (read from the behaviour panels in
``mnist10_unpruned/look/`` and from scratch probes on the 2,000 discovery images, written before the
concept check; see BEHAVIOUR_FIRST.md). Each part was probed afresh; no reading was copied from
mnist10_pruned.

Every number in the texts is a placeholder filled by ``behaviour_first.py write`` from the computed
results: ``{m[...]}`` the hypothesis measure (discovery images), ``{p[<probe id>][...]}`` a self-probe
(probe images, disjoint), ``{c[...]}`` composition statistics. Probe specs are
``explaneat.conv.explain.investigate.edit_one`` specs.
"""

import numpy as np
from scipy import ndimage

from explaneat.conv.explain import investigate as IV
from explaneat.conv.explain import mechanism as MC


def bar(at, angle, length=10, frame="box", width=2):
    return {"kind": "bar", "at": at, "angle": angle, "length": length, "frame": frame, "width": width}


H_MID, V_MID = bar((0.5, 0.5), 0), bar((0.5, 0.5), 90)
SHIFT1 = {"kind": "shift", "dc": 1}
DILATE = {"kind": "dilate", "size": 2}
WIDER, TALLER = {"kind": "stretch", "fx": 1.3}, {"kind": "stretch", "fy": 1.3}
DOWN4, UP4 = {"kind": "shift", "dr": 4}, {"kind": "shift", "dr": -4}
TOP_BAR, BOTTOM_BAR = bar((0.0, 0.5), 0), bar((1.0, 0.5), 0)
LEFT_UP, RIGHT_UP = bar((0.5, 0.0), 90), bar((0.5, 1.0), 90)
LEAN_LEFT, LEAN_RIGHT = {"kind": "shear", "s": 0.3}, {"kind": "shear", "s": -0.3}
IDENT = {"kind": "identity"}


# ------------------------------------------------------------------------------- measures
def _grads(X):
    X3 = (X[:, 0] if X.ndim == 4 else X).astype(np.float64)
    gy = ndimage.sobel(X3, axis=1)
    gx = ndimage.sobel(X3, axis=2)
    return X3, {"horizontal": gy ** 2, "vertical": gx ** 2}


def orient_in(X, kind, rows=(0, 1), cols=(0, 1)):
    """Edge energy of strokes of one orientation (as ``IV.orientation_energy``) inside a region given as
    fractions of the ink's bounding box."""
    X3, G = _grads(X)
    out = np.zeros(len(X3))
    for i, x in enumerate(X3):
        b = IV.ink_box(x)
        if b is None:
            continue
        rs, cs = IV._rel(b, rows, cols)
        out[i] = G[kind][i][rs, cs].sum()
    return out


def hole_area(X):
    X3 = X[:, 0] if X.ndim == 4 else X
    return np.array([MC.hole_masks(x >= 0.5).sum() for x in X3], float)


def ink(X):
    return IV.ink_count(X)


# ------------------------------------------------------------------------------- leaves
PARTS = {
    "S1": {
        "name": "Inverse ink meter",
        "mechanism": "Its map is lowered by every inked pixel, wherever it is and whichever way the stroke runs, so its "
                     "average falls as the digit gets more ink: it is highest for thin 1s and lowest for heavy 0s and 8s.",
        "prediction": "So thickening the strokes should lower it strongly, while a stroke of the same length should take "
                      "away about the same wherever it is placed and whichever way it runs.",
        "measure": {"label": "the amount of ink (ink pixels)", "fn": ink, "predict": -1, "is_ink": True},
        "probes": [
            {"id": "thicken", "words": "thicken every stroke by one pixel (dilation) vs move the digit one pixel right",
             "test": DILATE, "control": SHIFT1, "predict": -1},
            {"id": "angle", "words": "a 10-pixel '\\' stroke through the centre of the digit vs the same stroke '/'",
             "test": bar((0.5, 0.5), -45), "control": bar((0.5, 0.5), 45), "predict": 0},
            {"id": "place", "words": "a 10-pixel horizontal stroke along the top of the digit vs the same stroke along its bottom",
             "test": TOP_BAR, "control": BOTTOM_BAR, "predict": 0},
        ],
    },
    "S2.1": {
        "name": "Horizontal-stroke detector",
        "mechanism": "It responds along horizontal strokes (the bars and flat curves at the top, middle and bottom of a "
                     "digit) about equally wherever they are; upright strokes give it little. Wide digits, with long "
                     "horizontal strokes, raise it most; 1s, with none, leave it lowest.",
        "prediction": "So a horizontal stroke should raise it more than the same stroke drawn upright, making the digit "
                      "wider should raise it more than making it taller by the same factor, and moving a horizontal "
                      "stroke from the top of the digit to its bottom should change little.",
        "measure": {"label": "the amount of horizontal-stroke edge", "fn": lambda X: IV.orientation_energy(X)["horizontal"], "predict": 1},
        "probes": [
            {"id": "h_vs_v", "words": "a 10-pixel horizontal stroke through the centre of the digit vs the same stroke upright",
             "test": H_MID, "control": V_MID, "predict": 1},
            {"id": "wider", "words": "the digit stretched 30% wider vs 30% taller", "test": WIDER, "control": TALLER, "predict": 1},
            {"id": "top_vs_bottom", "words": "a 10-pixel horizontal stroke along the top of the digit vs the same stroke along its bottom",
             "test": TOP_BAR, "control": BOTTOM_BAR, "predict": 0},
        ],
    },
    "S2.2": {
        "name": "Upper horizontal-stroke detector",
        "mechanism": "It reads S2.1's horizontal-stroke map (and S1's) and keeps the horizontal strokes in the upper part "
                     "of the digit and of the image, such as the top bar of a 5 or 7 and the top curve of a 2 or 3. Upright "
                     "strokes give it little, and horizontal ink near the bottom of the image lowers it, so the same digit "
                     "drawn lower in the frame gives less.",
        "prediction": "So a horizontal stroke along the top of the digit should raise it far more than the same stroke along "
                      "its bottom, the same stroke along the top should count far more horizontal than upright, and moving "
                      "the whole digit 4 pixels down should lower it compared with moving it 4 pixels up.",
        "measure": {"label": "the horizontal-stroke edge in the upper half of the digit's box",
                    "fn": lambda X: orient_in(X, "horizontal", (0, 0.5)), "predict": 1},
        "probes": [
            {"id": "top_vs_bottom", "words": "a 10-pixel horizontal stroke along the top of the digit vs the same stroke along its bottom",
             "test": TOP_BAR, "control": BOTTOM_BAR, "predict": 1},
            {"id": "h_vs_v_top", "words": "a 10-pixel horizontal stroke along the top of the digit vs the same stroke upright at the same place",
             "test": TOP_BAR, "control": bar((0.0, 0.5), 90), "predict": 1},
            {"id": "shift_down", "words": "the digit moved 4 pixels down vs 4 pixels up", "test": DOWN4, "control": UP4, "predict": -1},
        ],
    },
    "S2.3": {
        "name": "Upper horizontal-stroke detector (coarse copy)",
        "mechanism": "It reads S2.2's map at the coarsest level and behaves like S2.2, more strongly: horizontal strokes in "
                     "the upper part of the digit raise it, upright strokes give little, and ink low in the image lowers it.",
        "prediction": "So a horizontal stroke along the top of the digit should raise it far more than the same stroke along "
                      "its bottom, and moving the digit 4 pixels down should lower it compared with moving it 4 pixels up.",
        "measure": {"label": "the horizontal-stroke edge in the upper half of the digit's box",
                    "fn": lambda X: orient_in(X, "horizontal", (0, 0.5)), "predict": 1},
        "probes": [
            {"id": "top_vs_bottom", "words": "a 10-pixel horizontal stroke along the top of the digit vs the same stroke along its bottom",
             "test": TOP_BAR, "control": BOTTOM_BAR, "predict": 1},
            {"id": "h_vs_v_top", "words": "a 10-pixel horizontal stroke along the top of the digit vs the same stroke upright at the same place",
             "test": TOP_BAR, "control": bar((0.0, 0.5), 90), "predict": 1},
            {"id": "shift_down", "words": "the digit moved 4 pixels down vs 4 pixels up", "test": DOWN4, "control": UP4, "predict": -1},
        ],
        "corr_with": "178",
    },
    "S3": {
        "name": "Ink-amount meter",
        "mechanism": "Its maps are a blurred copy of the digit's ink: every inked pixel raises the map around it, wherever it "
                     "is and whichever way the stroke runs, so its average measures how much ink the digit has.",
        "prediction": "So thickening the strokes should raise it strongly, while a stroke of the same length should add about "
                      "the same wherever it is placed and whichever way it runs.",
        "measure": {"label": "the amount of ink (ink pixels)", "fn": ink, "predict": 1, "is_ink": True},
        "probes": [
            {"id": "thicken", "words": "thicken every stroke by one pixel (dilation) vs move the digit one pixel right",
             "test": DILATE, "control": SHIFT1, "predict": 1},
            {"id": "angle", "words": "a 10-pixel horizontal stroke through the centre of the digit vs the same stroke upright",
             "test": H_MID, "control": V_MID, "predict": 0},
            {"id": "place", "words": "a 10-pixel horizontal stroke along the top of the digit vs the same stroke along its bottom",
             "test": TOP_BAR, "control": BOTTOM_BAR, "predict": 0},
        ],
    },
    "S4.1": {
        "name": "Right-edge upright-stroke detector",
        "mechanism": "It responds to upright strokes along the right edge of the digit, such as the right side of a 0 and "
                     "the stems of 4, 7 and 9; an upright stroke through the middle of the digit gives it little, and a "
                     "horizontal stroke along the bottom holds it back. Narrow, upright digits suit it; digits that lean "
                     "right give less.",
        "prediction": "So an upright stroke along the right edge of the digit should raise it far more than the same stroke "
                      "through its centre or the same stroke lying horizontal at the right edge, and a horizontal stroke along the bottom of the digit should lower it compared "
                      "with the same stroke along the top.",
        "measure": {"label": "the upright-stroke edge in the right third of the digit's box",
                    "fn": lambda X: orient_in(X, "vertical", (0, 1), (2 / 3, 1)), "predict": 1},
        "probes": [
            {"id": "right_vs_centre", "words": "a 10-pixel upright stroke along the right edge of the digit vs the same stroke through its centre",
             "test": RIGHT_UP, "control": V_MID, "predict": 1},
            {"id": "right_v_vs_h", "words": "a 10-pixel upright stroke along the right edge of the digit vs the same stroke horizontal at the same place",
             "test": RIGHT_UP, "control": bar((0.5, 1.0), 0), "predict": 1},
            {"id": "bottom_vs_top", "words": "a 10-pixel horizontal stroke along the bottom of the digit vs the same stroke along its top",
             "test": BOTTOM_BAR, "control": TOP_BAR, "predict": -1},
        ],
    },
    "S4.2": {
        "name": "Left-side upright-stroke detector",
        "mechanism": "It responds to an upright stroke on the left side of the digit, as the left side of a 0, 4, 6 or 8 "
                     "has, and stays low when the left side is open, as in a 3 or 7. It goes by the digit's own left side, "
                     "not by where the digit sits in the image.",
        "prediction": "So an upright stroke along the left edge of the digit should raise it far more than the same stroke "
                      "along its right edge or the same stroke lying horizontal at the left edge; mirroring a 3 or 7 left to right (which puts its closed side on the left) "
                      "should raise it; and moving the whole digit 4 pixels left instead of right should change nothing.",
        "measure": {"label": "the upright-stroke edge in the left half of the digit's box",
                    "fn": lambda X: orient_in(X, "vertical", (0, 1), (0, 0.5)), "predict": 1},
        "probes": [
            {"id": "left_vs_right", "words": "a 10-pixel upright stroke along the left edge of the digit vs the same stroke along its right edge",
             "test": LEFT_UP, "control": RIGHT_UP, "predict": 1},
            {"id": "left_v_vs_h", "words": "a 10-pixel upright stroke along the left edge of the digit vs the same stroke horizontal at the same place",
             "test": LEFT_UP, "control": bar((0.5, 0.0), 0), "predict": 1},
            {"id": "mirror_open_left", "words": "on 3s and 7s, the digit mirrored left to right vs left as it is",
             "test": {"kind": "mirror_lr"}, "control": IDENT, "predict": 1, "digits": [3, 7]},
            {"id": "shift_left", "words": "the digit moved 4 pixels left vs 4 pixels right",
             "test": {"kind": "shift", "dc": -4}, "control": {"kind": "shift", "dc": 4}, "predict": 0},
        ],
    },
    "S4.3": {
        "name": "Upper-left upright-stroke detector",
        "mechanism": "It reads the upper-left corner of the digit: an upright stroke there (the left side of a 0 or 6 "
                     "rising to the top) raises it, and a horizontal stroke across the upper left (a top bar starting at "
                     "the left, as in a 7) holds it back.",
        "prediction": "So an upright stroke at the upper left of the digit should raise it far more than the same stroke at "
                      "the upper right or the same stroke lying horizontal there, and a horizontal stroke across the top left should lower it compared with the same "
                      "stroke across the bottom left.",
        "measure": {"label": "the upright-stroke edge in the upper-left quarter of the digit's box",
                    "fn": lambda X: orient_in(X, "vertical", (0, 0.5), (0, 0.5)), "predict": 1},
        "probes": [
            {"id": "ul_vs_ur", "words": "a 10-pixel upright stroke at the upper left of the digit vs the same stroke at its upper right",
             "test": bar((0.25, 0.0), 90), "control": bar((0.25, 1.0), 90), "predict": 1},
            {"id": "ul_v_vs_h", "words": "a 10-pixel upright stroke at the upper left of the digit vs the same stroke horizontal at the same place",
             "test": bar((0.25, 0.0), 90), "control": bar((0.25, 0.0), 0), "predict": 1},
            {"id": "tl_vs_bl", "words": "a 10-pixel horizontal stroke across the top left of the digit vs the same stroke across its bottom left",
             "test": bar((0.0, 0.25), 0), "control": bar((1.0, 0.25), 0), "predict": -1},
        ],
    },
    "S4.4": {
        "name": "Left-lean detector",
        "mechanism": "It fires when the digit's strokes lean left ('\\', top to the left of the bottom) or stand upright, "
                     "and drops as they lean right ('/'), which most handwritten digits do. Mirroring a digit left to "
                     "right turns a right lean into a left lean and raises it.",
        "prediction": "So slanting the digit to lean left should raise it compared with slanting it to lean right by the "
                      "same amount, and a '\\' stroke through the centre should raise it more than the same stroke '/'.",
        "measure": {"label": "the digit's slant (degrees leaning right)", "fn": IV.slant, "predict": -1},
        "probes": [
            {"id": "lean_left", "words": "the digit sheared to lean left vs to lean right (by 0.3 pixels per row)",
             "test": LEAN_LEFT, "control": LEAN_RIGHT, "predict": 1},
            {"id": "falling_vs_rising", "words": "a 10-pixel '\\' stroke through the centre of the digit vs the same stroke '/'",
             "test": bar((0.5, 0.5), -45), "control": bar((0.5, 0.5), 45), "predict": 1},
        ],
    },
    "S5.1": {
        "name": "Wide-base detector",
        "mechanism": "It responds to a wide, horizontal bottom to the digit: the bottom curve of a 0 or 6, the flat base of "
                     "a 2, the lower curve of a 3 or 5. An upright stroke ending at the bottom (the stem of a 1, 4, 7 or 9) "
                     "gives it much less, and wide digits suit it.",
        "prediction": "So a horizontal stroke along the bottom of the digit should raise it far more than the same stroke "
                      "upright at the same place, and making the digit 30% wider should raise it more than making it 30% taller.",
        "measure": {"label": "the horizontal-stroke edge in the lower half of the digit's box",
                    "fn": lambda X: orient_in(X, "horizontal", (0.5, 1)), "predict": 1},
        "probes": [
            {"id": "bottom_h_vs_v", "words": "a 10-pixel horizontal stroke along the bottom of the digit vs the same stroke upright at the same place",
             "test": BOTTOM_BAR, "control": bar((1.0, 0.5), 90), "predict": 1},
            {"id": "wider", "words": "the digit stretched 30% wider vs 30% taller", "test": WIDER, "control": TALLER, "predict": 1},
        ],
    },
    "S5.2": {
        "name": "High horizontal-stroke detector",
        "mechanism": "It reads S5.1's base map among others and favours the opposite end: horizontal strokes in the upper "
                     "part of the digit and wide digits raise it, while ink at the bottom of the digit, and above all near "
                     "the bottom edge of the image, holds it back, so the same digit drawn lower in the frame gives much less.",
        "prediction": "So moving the digit 4 pixels down should lower it compared with moving it 4 pixels up, and a horizontal "
                      "stroke along the top of the digit should raise it more than the same stroke along its bottom or the same "
                      "stroke upright at the top.",
        "measure": {"label": "the horizontal-stroke edge in the upper half of the digit's box",
                    "fn": lambda X: orient_in(X, "horizontal", (0, 0.5)), "predict": 1},
        "probes": [
            {"id": "shift_down", "words": "the digit moved 4 pixels down vs 4 pixels up", "test": DOWN4, "control": UP4, "predict": -1},
            {"id": "top_vs_bottom", "words": "a 10-pixel horizontal stroke along the top of the digit vs the same stroke along its bottom",
             "test": TOP_BAR, "control": BOTTOM_BAR, "predict": 1},
            {"id": "top_h_vs_v", "words": "a 10-pixel horizontal stroke along the top of the digit vs the same stroke upright at the same place",
             "test": TOP_BAR, "control": bar((0.0, 0.5), 90), "predict": 1},
        ],
    },
    "S6": {
        "name": "Enclosed-space detector",
        "mechanism": "It responds to empty space enclosed by strokes, the holes of 0, 6, 8 and 9: filling a hole takes "
                     "most of its response away. Thicker strokes lower it only where they narrow a hole; on digits without "
                     "a hole, thickness does not matter. Mirroring the digit or cutting it into shuffled tiles also raises it, "
                     "and 3s (no hole) fire it nearly as often as 0s; this reading does not explain those.",
        "prediction": "So filling the digit's holes should lower it far more than adding the same number of ink pixels "
                      "just outside the digit, while on 1s and 7s (no hole) thickening the strokes should change nothing "
                      "compared with moving the digit one pixel.",
        "measure": {"label": "the area of the digit's holes (pixels)", "fn": hole_area, "predict": 1},
        "probes": [
            {"id": "fill", "words": "fill the digit's holes vs add as many ink pixels on a ring just outside the digit",
             "test": {"kind": "mech", "edit": "fill"}, "control": IDENT, "predict": -1},
            {"id": "thicken_no_hole", "words": "on 1s and 7s, thicken every stroke by one pixel vs move the digit one pixel right",
             "test": DILATE, "control": SHIFT1, "predict": 0, "digits": [1, 7]},
        ],
    },
    "S7": {
        "name": "Top-and-bottom horizontal-stroke detector",
        "mechanism": "It responds to horizontal strokes at the top and at the bottom of the digit, such as the top and "
                     "bottom curves of a 3, 5 or 8, more than to one across the middle; upright strokes give little. 1s and "
                     "4s, with neither, leave it lowest.",
        "prediction": "So a horizontal stroke along the bottom of the digit should raise it more than the same stroke across "
                      "its middle or the same stroke upright at the bottom, and a stroke along its top should raise it more than "
                      "the same stroke across its middle.",
        "measure": {"label": "the horizontal-stroke edge in the top and bottom quarters of the digit's box",
                    "fn": lambda X: orient_in(X, "horizontal", (0, 0.25)) + orient_in(X, "horizontal", (0.75, 1)), "predict": 1},
        "probes": [
            {"id": "bottom_vs_mid", "words": "a 10-pixel horizontal stroke along the bottom of the digit vs the same stroke across its middle",
             "test": BOTTOM_BAR, "control": H_MID, "predict": 1},
            {"id": "bottom_h_vs_v", "words": "a 10-pixel horizontal stroke along the bottom of the digit vs the same stroke upright at the same place",
             "test": BOTTOM_BAR, "control": bar((1.0, 0.5), 90), "predict": 1},
            {"id": "top_vs_mid", "words": "a 10-pixel horizontal stroke along the top of the digit vs the same stroke across its middle",
             "test": TOP_BAR, "control": H_MID, "predict": 1},
        ],
    },
}

# The fill probe's control: mechanism.edit('fill') returns its own matched control (as many pixels on a ring
# just outside the digit); run_probe uses it when the test kind is 'mech' and ignores ``control``.

COMPOSITIONS = {
    "S2": {
        "name": "Horizontal strokes, then the upper ones",
        "combine": "S2.1 finds horizontal strokes anywhere in the digit; S2.2 reads S2.1's map and keeps the upper ones, and "
                   "S2.3 re-reads S2.2's map at the coarsest level (S2.1 feeds S2.2, which feeds S2.3).",
        "adds": "What the chain adds is height: a horizontal stroke along the top of the digit instead of its bottom moves "
                "S2.2 by {p[S2.2:top_vs_bottom][effect_sd]:+.2f} SD but S2.1 by only {p[S2.1:top_vs_bottom][effect_sd]:+.2f} SD. "
                "A 10-class classifier on the children's pooled exits together reaches {c[acc_both]:.0%} (cross-validated), "
                "against {c[acc_best]:.0%} for the best child alone.",
        "probes": [{"id": "top_vs_bottom", "child": "S2.2", "predict": 1, "words": "a horizontal stroke along the top vs the bottom of the digit (S2.2)",
                    "test": TOP_BAR, "control": BOTTOM_BAR},
                   {"id": "top_vs_bottom", "child": "S2.1", "predict": 0, "words": "the same edit (S2.1)",
                    "test": TOP_BAR, "control": BOTTOM_BAR}],
    },
    "S4": {
        "name": "Which side the upright strokes are on, and the lean",
        "combine": "S4.1 (upright strokes on the right edge) feeds S4.2 and S4.3; S4.2 (an upright stroke on the left side) "
                   "feeds S4.3 (the upper-left corner) and S4.4 (the lean). Together they say on which side of the digit "
                   "its upright strokes stand and whether it leans.",
        "adds": "The same edit moves the children in opposite directions: an upright stroke along the left edge of the digit "
                "instead of its right edge moves S4.2 by {p[S4.2:left_vs_right][effect_sd]:+.2f} SD and S4.1 by "
                "{p[S4.1:left_vs_right][effect_sd]:+.2f} SD. A 10-class classifier on the children's pooled exits together "
                "reaches {c[acc_both]:.0%} (cross-validated), against {c[acc_best]:.0%} for the best child alone.",
        "probes": [{"id": "left_vs_right", "child": "S4.2", "predict": 1, "words": "an upright stroke along the left vs the right edge of the digit (S4.2)",
                    "test": LEFT_UP, "control": RIGHT_UP},
                   {"id": "left_vs_right", "child": "S4.1", "predict": -1, "words": "the same edit (S4.1)",
                    "test": LEFT_UP, "control": RIGHT_UP}],
    },
    "S5": {
        "name": "Base against top",
        "combine": "S5.1 reads the digit's base; S5.2 reads S5.1's map (with others) and favours the top of the digit "
                   "(S5.1 feeds S5.2). Both favour wide digits.",
        "adds": "What the pair adds is a contrast between the bottom and the top of the digit: a horizontal stroke along the "
                "top instead of the bottom moves S5.2 by {p[S5.2:top_vs_bottom][effect_sd]:+.2f} SD and S5.1 by "
                "{p[S5.1:top_vs_bottom][effect_sd]:+.2f} SD. A 10-class classifier on both children's pooled exits reaches "
                "{c[acc_both]:.0%} (cross-validated), against {c[acc_best]:.0%} for the better child alone.",
        "probes": [{"id": "top_vs_bottom", "child": "S5.2", "predict": 1, "words": "a horizontal stroke along the top vs the bottom of the digit (S5.2)",
                    "test": TOP_BAR, "control": BOTTOM_BAR},
                   {"id": "top_vs_bottom", "child": "S5.1", "predict": -1, "words": "the same edit (S5.1)",
                    "test": TOP_BAR, "control": BOTTOM_BAR}],
    },
}

ROOT = {
    "probes": [
        {"id": "three_leftstroke", "digit": 3, "words": "on 3s, a 10-pixel upright stroke along the left edge vs along the right edge (the '3' score margin)",
         "test": LEFT_UP, "control": RIGHT_UP, "predict": -1, "wakes": "S4"},
        {"id": "seven_leftstroke", "digit": 7, "words": "on 7s, a 10-pixel upright stroke along the left edge vs along the right edge (the '7' score margin)",
         "test": LEFT_UP, "control": RIGHT_UP, "predict": -1, "wakes": "S4"},
        {"id": "nine_base", "digit": 9, "words": "on 9s, a 10-pixel horizontal stroke along the bottom vs the same stroke upright there (the '9' score margin)",
         "test": BOTTOM_BAR, "control": bar((1.0, 0.5), 90), "predict": -1, "wakes": "S5"},
        {"id": "four_base", "digit": 4, "words": "on 4s, a 10-pixel horizontal stroke along the bottom vs the same stroke upright there (the '4' score margin)",
         "test": BOTTOM_BAR, "control": bar((1.0, 0.5), 90), "predict": -1, "wakes": "S5"},
        {"id": "zero_erase_bottom", "digit": 0, "words": "on 0s, erase the bottom quarter of the digit vs as many ink pixels from its top quarter (the '0' score margin)",
         "test": {"kind": "erase", "rows": (0.75, 1.0)}, "control": {"kind": "erase_matched", "rows": (0.0, 0.25), "from": "start"},
         "predict": -1, "wakes": "S5 (removes the base it needs firing)"},
    ],
}
