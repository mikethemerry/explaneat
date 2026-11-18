"""Behaviour-first hypotheses for three8noise_run_s0, the 3 vs 8 vs noise network (classes: 0 = "3",
1 = "8", 2 = "noise"; read from the behaviour panels in ``three8noise_run_s0/look/`` and from scratch
probes on the 2,000 discovery images, written before the concept check; see BEHAVIOUR_FIRST.md).
Every part was probed afresh; no reading was copied from the MNIST networks.

``digits: [...]`` in a probe means classes here (0 = 3s, 1 = 8s, 2 = noise images). The noise images
have the ink histogram of a real 3 or 8 (shuffled pixels, scribbles, blobs, permuted patches), so the
amount of ink cannot tell noise from digits; ``mech: shuffle_ink`` (a digit's own ink pixels moved to
random places in its box, against the box mirrored left to right) asks whether a part needs connected
strokes.

Every number in the texts is a placeholder filled by ``behaviour_first.py write`` from the computed
results: ``{m[...]}`` the hypothesis measure (discovery images), ``{p[<probe id>][...]}`` a self-probe
(probe images, disjoint), ``{c[...]}`` composition statistics.

The root's replication bar (80 % of the classes), the class wording ("class" instead of "digit") and the
0.3 SD probe bar are the shared rules of ``behaviour_first.py`` / ``investigate.py``.
"""

import numpy as np
from scipy import ndimage

from explaneat.conv.explain import investigate as IV


def bar(at, angle, length=10, frame="box", width=2):
    return {"kind": "bar", "at": at, "angle": angle, "length": length, "frame": frame, "width": width}


H_MID, V_MID = bar((0.5, 0.5), 0), bar((0.5, 0.5), 90)
SHIFT1 = {"kind": "shift", "dc": 1}
DILATE = {"kind": "dilate", "size": 2}
LEFT_UP, RIGHT_UP = bar((0.5, 0.0), 90), bar((0.5, 1.0), 90)
IDENT = {"kind": "identity"}
SHUFFLE = {"kind": "mech", "edit": "shuffle_ink"}  # its own control: the box mirrored left to right
THREE, EIGHT, DIGITS = [0], [1], [0, 1]


# ------------------------------------------------------------------------------- measures
def _grads(X):
    X3 = (X[:, 0] if X.ndim == 4 else X).astype(np.float64)
    return X3, {"horizontal": ndimage.sobel(X3, axis=1) ** 2, "vertical": ndimage.sobel(X3, axis=2) ** 2}


def orient_in(X, kind, rows=(0, 1), cols=(0, 1)):
    """Edge energy of strokes of one orientation inside a region given as fractions of the ink's box."""
    X3, G = _grads(X)
    out = np.zeros(len(X3))
    for i, x in enumerate(X3):
        b = IV.ink_box(x)
        if b is None:
            continue
        rs, cs = IV._rel(b, rows, cols)
        out[i] = G[kind][i][rs, cs].sum()
    return out


def _labels(X):
    X3 = X[:, 0] if X.ndim == 4 else X
    return [ndimage.label(x >= 0.5, structure=np.ones((3, 3))) for x in X3]


def n_fragments(X):
    return np.array([n for _, n in _labels(X)], float)


def largest_piece(X):
    return np.array([np.bincount(lab.ravel())[1:].max() if n else 0 for lab, n in _labels(X)], float)


def ink(X):
    return IV.ink_count(X)


# ------------------------------------------------------------------------------- leaves
PARTS = {
    "S1.1": {
        "name": "Connected horizontal-stroke detector",
        "mechanism": "It responds along connected strokes, horizontal ones most: the three arms of a 3 and the top, "
                     "waist and bottom of an 8. The same ink scattered as separate pixels gives it little, which is why "
                     "shuffled-pixel noise leaves it lowest and 3s raise it most.",
        "prediction": "So moving a 3's or 8's own ink pixels to random places in its box should lower it far more than "
                      "mirroring the box, and a horizontal stroke should raise it more than the same stroke upright.",
        "measure": {"label": "the amount of horizontal-stroke edge", "fn": lambda X: IV.orientation_energy(X)["horizontal"], "predict": 1},
        "probes": [
            {"id": "shuffle", "words": "on 3s and 8s, the digit's ink pixels moved to random places in its box vs the box mirrored left to right",
             "test": SHUFFLE, "control": IDENT, "predict": -1, "digits": DIGITS},
            {"id": "h_vs_v", "words": "a 10-pixel horizontal stroke through the centre of the image's ink vs the same stroke upright",
             "test": H_MID, "control": V_MID, "predict": 1},
        ],
    },
    "S1.2": {
        "name": "Scattered-ink detector",
        "mechanism": "It responds to ink broken into many small separate pieces, as in the shuffled-pixel noise images, "
                     "and less to smooth connected strokes or solid blobs; upright strokes give it more than horizontal "
                     "ones. Its main exit runs against S1.1's connected-stroke exits.",
        "prediction": "So moving a 3's or 8's own ink pixels to random places in its box should raise it far more than "
                      "mirroring the box, and an upright stroke should raise it more than the same stroke horizontal.",
        "measure": {"label": "the number of separate ink pieces", "fn": n_fragments, "predict": 1},
        "probes": [
            {"id": "shuffle", "words": "on 3s and 8s, the digit's ink pixels moved to random places in its box vs the box mirrored left to right",
             "test": SHUFFLE, "control": IDENT, "predict": 1, "digits": DIGITS},
            {"id": "v_vs_h", "words": "a 10-pixel upright stroke through the centre of the image's ink vs the same stroke horizontal",
             "test": V_MID, "control": H_MID, "predict": 1},
        ],
    },
    "S1.3": {
        "name": "Left-side upright-stroke detector",
        "mechanism": "It responds to an upright stroke on the left side of the shape, as the left sides of an 8's loops "
                     "have, and stays near zero for 3s, whose left side is open. It goes by the shape's own left side, "
                     "not by where it sits in the image.",
        "prediction": "So an upright stroke along the left edge should raise it far more than the same stroke along the "
                      "right edge or the same stroke lying horizontal at the left edge; mirroring a 3 left to right (its "
                      "closed side then on the left) should raise it; and moving the whole image content 4 pixels left "
                      "instead of right should change nothing.",
        "measure": {"label": "the upright-stroke edge in the left half of the ink's box",
                    "fn": lambda X: orient_in(X, "vertical", (0, 1), (0, 0.5)), "predict": 1},
        "probes": [
            {"id": "left_vs_right", "words": "a 10-pixel upright stroke along the left edge of the ink vs the same stroke along its right edge",
             "test": LEFT_UP, "control": RIGHT_UP, "predict": 1},
            {"id": "left_v_vs_h", "words": "a 10-pixel upright stroke along the left edge vs the same stroke horizontal at the same place",
             "test": LEFT_UP, "control": bar((0.5, 0.0), 0), "predict": 1},
            {"id": "mirror_3", "words": "on 3s, the digit mirrored left to right vs left as it is",
             "test": {"kind": "mirror_lr"}, "control": IDENT, "predict": 1, "digits": THREE},
            {"id": "shift_left", "words": "the image content moved 4 pixels left vs 4 pixels right",
             "test": {"kind": "shift", "dc": -4}, "control": {"kind": "shift", "dc": 4}, "predict": 0},
        ],
    },
    "S2.1": {
        "name": "Connected strokes clear of the left margin",
        "mechanism": "It reads S1.1's connected-stroke map at the coarsest level: connected strokes in the middle and "
                     "right of the image raise it, scattered pixels give little, and ink in the leftmost columns of the "
                     "image holds it back, so it is highest for 3s, whose strokes lie right of an open left side.",
        "prediction": "So moving a 3's or 8's own ink pixels to random places in its box should lower it far more than "
                      "mirroring the box, and an upright stroke at image column 4 should lower it compared with the same "
                      "stroke at column 14.",
        "measure": {"label": "the size of the largest connected ink piece", "fn": largest_piece, "predict": 1},
        "probes": [
            {"id": "shuffle", "words": "on 3s and 8s, the digit's ink pixels moved to random places in its box vs the box mirrored left to right",
             "test": SHUFFLE, "control": IDENT, "predict": -1, "digits": DIGITS},
            {"id": "left_margin", "words": "a 10-pixel upright stroke at image column 4 vs the same stroke at column 14 (mid-height)",
             "test": bar((14 / 27, 4 / 27), 90, frame="image"), "control": bar((14 / 27, 14 / 27), 90, frame="image"), "predict": -1},
        ],
    },
    "S2.2": {
        "name": "Open-left horizontal-stroke detector",
        "mechanism": "It reads S2.1's map and S1.3's left-side map and responds to horizontal strokes whose left side is "
                     "open, the arms of a 3; an upright stroke closing the left side, as an 8's loops do, takes it away. "
                     "It fires on 3s and stays low on 8s and noise alike.",
        "prediction": "So on 3s, a stroke closing the left side should lower it more than the same stroke drawn through the "
                      "middle, mirroring the 3 left to right should lower it, and a horizontal stroke should raise it more "
                      "than the same stroke upright.",
        "measure": {"label": "the horizontal-stroke edge minus the upright-stroke edge in the left third of the ink's box",
                    "fn": lambda X: IV.orientation_energy(X)["horizontal"] - 3 * orient_in(X, "vertical", (0, 1), (0, 1 / 3)),
                    "predict": 1},
        "probes": [
            {"id": "close_left_3", "words": "on 3s, an upright stroke closing the left side (bridge) vs the same stroke through the middle of the box",
             "test": {"kind": "mech", "edit": "bridge_left"}, "control": IDENT, "predict": -1, "digits": THREE},
            {"id": "mirror_3", "words": "on 3s, the digit mirrored left to right vs left as it is",
             "test": {"kind": "mirror_lr"}, "control": IDENT, "predict": -1, "digits": THREE},
            {"id": "h_vs_v", "words": "a 10-pixel horizontal stroke through the centre of the image's ink vs the same stroke upright",
             "test": H_MID, "control": V_MID, "predict": 1},
        ],
    },
    "S3": {
        "name": "Ink meter, upright strokes counting more",
        "mechanism": "It adds up the ink, with upright strokes counting more than horizontal ones: the amount of ink "
                     "moves it within every class, and turning a 3 or 8 on its side (its arms become upright) raises it. "
                     "Scattered ink and blobs raise it too, so noise images fire it more than 8s, and 8s more than 3s.",
        "prediction": "So thickening the strokes should raise it strongly, an upright stroke should raise it more than the "
                      "same stroke horizontal, and turning the image a quarter turn should raise it more than mirroring it.",
        "measure": {"label": "the amount of ink (ink pixels)", "fn": ink, "predict": 1, "is_ink": True},
        "probes": [
            {"id": "thicken", "words": "thicken every stroke by one pixel (dilation) vs move the image one pixel right",
             "test": DILATE, "control": SHIFT1, "predict": 1},
            {"id": "v_vs_h", "words": "a 10-pixel upright stroke through the centre of the image's ink vs the same stroke horizontal",
             "test": V_MID, "control": H_MID, "predict": 1},
            {"id": "on_side", "words": "the image turned 90 degrees vs mirrored left to right",
             "test": {"kind": "rotate90"}, "control": {"kind": "mirror_lr"}, "predict": 1},
        ],
    },
}

COMPOSITIONS = {
    "S1": {
        "name": "Connected strokes against scattered ink",
        "combine": "S1.1 (connected, mostly horizontal strokes) feeds S1.2 and S1.3; S1.2 (scattered ink) feeds S1.3 "
                   "(an upright stroke on the left). Together they say whether the ink forms strokes, and whether the "
                   "left side is closed.",
        "adds": "The same edit moves the first two children in opposite directions: scattering a digit's own ink pixels "
                "moves S1.1 by {p[S1.1:shuffle][effect_sd]:+.2f} SD and S1.2 by {p[S1.2:shuffle][effect_sd]:+.2f} SD. "
                "A 3-class classifier on the children's pooled exits together reaches {c[acc_both]:.0%} (cross-validated), "
                "against {c[acc_best]:.0%} for the best child alone.",
        "probes": [{"id": "shuffle", "child": "S1.1", "predict": -1, "words": "on 3s and 8s, the ink pixels scattered in the box vs the box mirrored (S1.1)",
                    "test": SHUFFLE, "control": IDENT, "digits": DIGITS},
                   {"id": "shuffle", "child": "S1.2", "predict": 1, "words": "the same edit (S1.2)",
                    "test": SHUFFLE, "control": IDENT, "digits": DIGITS}],
    },
    "S2": {
        "name": "Connected strokes, then the open-left horizontal ones",
        "combine": "S2.1 keeps connected strokes clear of the left margin; S2.2 reads S2.1's map (with S1.3's left-side "
                   "map) and keeps the horizontal ones with an open left side (S2.1 feeds S2.2).",
        "adds": "What the chain adds is orientation on top of connectedness: on 3s a horizontal stroke instead of an "
                "upright one moves S2.2 by {p[S2.2:h_vs_v_3][effect_sd]:+.2f} SD and S2.1 by {p[S2.1:h_vs_v_3][effect_sd]:+.2f} SD, "
                "while scattering a digit's ink pixels moves S2.1 by {p[S2.1:shuffle][effect_sd]:+.2f} SD. A 3-class "
                "classifier on both children's pooled exits reaches {c[acc_both]:.0%} (cross-validated), against "
                "{c[acc_best]:.0%} for the better child alone.",
        "probes": [{"id": "shuffle", "child": "S2.1", "predict": -1, "words": "on 3s and 8s, the ink pixels scattered in the box vs the box mirrored (S2.1)",
                    "test": SHUFFLE, "control": IDENT, "digits": DIGITS},
                   {"id": "h_vs_v_3", "child": "S2.2", "predict": 1, "words": "on 3s, a horizontal stroke through the centre vs the same stroke upright (S2.2)",
                    "test": H_MID, "control": V_MID, "digits": THREE},
                   {"id": "h_vs_v_3", "child": "S2.1", "predict": 1, "words": "the same edit (S2.1)",
                    "test": H_MID, "control": V_MID, "digits": THREE}],
    },
}

ROOT = {
    "probes": [
        {"id": "three_scatter", "digit": 0, "words": "on 3s, the digit's ink pixels moved to random places in its box vs the box mirrored left to right (the '3' score margin)",
         "test": SHUFFLE, "control": IDENT, "predict": -1, "wakes": "S1 (removes the connected strokes it needs firing)"},
        {"id": "eight_open_left", "digit": 1, "words": "on 8s, erase the left third of the digit vs as many ink pixels from its right third (the '8' score margin)",
         "test": {"kind": "erase", "cols": (0.0, 1 / 3)}, "control": {"kind": "erase_matched", "cols": (2 / 3, 1.0), "from": "end"},
         "predict": -1, "wakes": "S2"},
        {"id": "noise_top_bar", "digit": 2, "words": "on noise images, a 10-pixel horizontal stroke along the top of the ink vs the same stroke upright there (the 'noise' score margin)",
         "test": bar((0.0, 0.5), 0), "control": bar((0.0, 0.5), 90), "predict": -1, "wakes": "S1"},
    ],
}
