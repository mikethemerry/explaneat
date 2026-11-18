"""Behaviour-first hypotheses for mnist10_pruned (read from the behaviour panels in
``mnist10_pruned/look/``, written before the concept check; see BEHAVIOUR_FIRST.md).

Every number in the texts is a placeholder filled by ``behaviour_first.py write`` from the computed
results: ``{m[...]}`` the hypothesis measure (discovery images), ``{p[<probe id>][...]}`` a self-probe
(probe images, disjoint), ``{c[...]}`` composition statistics. Probe specs are
``explaneat.conv.explain.investigate.edit_one`` specs.
"""

import numpy as np

from explaneat.conv.explain import investigate as IV


def bar(at, angle, length=10, frame="box"):
    return {"kind": "bar", "at": at, "angle": angle, "length": length, "frame": frame}


H_MID, V_MID = bar((0.5, 0.5), 0), bar((0.5, 0.5), 90)
SHIFT1 = {"kind": "shift", "dc": 1}
DILATE = {"kind": "dilate", "size": 2}
WIDER, TALLER = {"kind": "stretch", "fx": 1.3}, {"kind": "stretch", "fy": 1.3}

INK_PROBES = [
    {"id": "thicken", "words": "thicken every stroke by one pixel (dilation) vs move the digit one pixel right",
     "test": DILATE, "control": SHIFT1, "predict": 1},
    {"id": "place", "words": "a 10-pixel horizontal stroke along the top of the digit vs the same stroke upright along its right edge",
     "test": bar((0.0, 0.5), 0), "control": bar((0.5, 1.0), 90), "predict": 0},
]


def ink(X):
    return IV.ink_count(X)


def horizontal_edges(X):
    return IV.orientation_energy(X)["horizontal"]


def upright_balance(X):
    e = IV.orientation_energy(X)
    return e["vertical"] + e["diag_up"] - e["horizontal"] - e["diag_down"]


def low_minus_top(X):
    return IV.ink_in(X, (15 / 28, 25 / 28), (14 / 28, 1), frame="image") - IV.ink_in(X, (0, 7 / 28), (0, 1), frame="image")


PARTS = {
    "S1.1": {
        "name": "Ink-amount meter",
        "mechanism": "Its maps are a blurred copy of the digit's ink: every inked pixel raises the map around it, "
                     "wherever it is and whichever way the stroke runs, so its average response measures how much ink the digit has.",
        "prediction": "So thickening the strokes should raise it strongly, while a stroke of the same length should add "
                      "about the same wherever it is placed and whichever way it runs.",
        "measure": {"label": "the amount of ink (ink pixels)", "fn": ink, "predict": 1, "is_ink": True},
        "probes": INK_PROBES,
    },
    "S1.2": {
        "name": "Ink-amount meter (second copy)",
        "mechanism": "Like S1.1, its maps are a blurred copy of the digit's ink, and its average measures how much ink "
                     "the digit has; its main exit and S1.1's agree almost exactly (r = {c[r_other]:.3f} over the discovery images).",
        "prediction": "So thickening the strokes should raise it strongly, and a stroke of the same length should add about "
                      "the same wherever it is placed and whichever way it runs.",
        "measure": {"label": "the amount of ink (ink pixels)", "fn": ink, "predict": 1, "is_ink": True},
        "probes": INK_PROBES, "corr_with": "119",
    },
    "S2.1": {
        "name": "Horizontal-stroke detector",
        "mechanism": "It reads S1.2's blurred ink map and responds along horizontal strokes: the bars and flat curves at "
                     "the top, middle and bottom of a digit. Upright strokes give it little.",
        "prediction": "So a horizontal stroke should raise it more than the same stroke drawn upright, and making the "
                      "digit wider should raise it more than making it taller by the same factor.",
        "measure": {"label": "the amount of horizontal-stroke edge", "fn": horizontal_edges, "predict": 1},
        "probes": [
            {"id": "h_vs_v", "words": "a 10-pixel horizontal stroke through the centre of the digit vs the same stroke upright",
             "test": H_MID, "control": V_MID, "predict": 1},
            {"id": "wider", "words": "the digit stretched 30% wider vs 30% taller", "test": WIDER, "control": TALLER, "predict": 1},
        ],
    },
    "S2.2": {
        "name": "Lowest-stroke detector",
        "mechanism": "It reads S2.1's horizontal-stroke map (and S1.1's ink map) and keeps the digit's lowest horizontal "
                     "stroke, a horizontal stroke with no ink below it: the flat base of a 2, the bottom curve of a 0, 3, 5 "
                     "or 6. It does not depend on where the digit sits in the image; ink below the stroke takes it away.",
        "prediction": "So a horizontal stroke along the bottom of the digit should raise it far more than the same stroke "
                      "along its top. A detached stroke drawn just below a digit that already ends in a horizontal stroke "
                      "(0, 2, 3, 5) should lower it, because the old base now has ink below it, while the same stroke below "
                      "a digit that ends in a stem (1, 4, 7, 9) should raise it, because it becomes the lowest stroke; the "
                      "control is the same stroke drawn just above the digit.",
        "history": "An earlier prediction, that erasing the bottom quarter of the digit lowers it more than erasing as many "
                   "ink pixels from the top quarter, failed ({p[erase_bottom][effect_sd]:+.2f} SD): erasing the bottom of a 4 "
                   "or 9 leaves its crossbar or loop as the new lowest stroke, which raises it. The reading was rewritten "
                   "from 'a stroke along the bottom' to 'the lowest horizontal stroke' and probed again.",
        "measure": {"label": "the amount of ink in the bottom quarter of the digit's box",
                    "fn": lambda X: IV.ink_in(X, (0.75, 1), (0, 1)), "predict": 1},
        "probes": [
            {"id": "bottom_vs_top", "words": "a 10-pixel horizontal stroke along the bottom of the digit vs the same stroke along its top",
             "test": bar((1.0, 0.5), 0), "control": bar((0.0, 0.5), 0), "predict": 1},
            {"id": "below_based", "words": "on 0s, 2s, 3s and 5s, a 10-pixel horizontal stroke 3 pixels below the digit vs the same stroke 3 pixels above it",
             "test": bar((1.2, 0.5), 0), "control": bar((-0.2, 0.5), 0), "predict": -1, "digits": [0, 2, 3, 5]},
            {"id": "below_stem", "words": "on 1s, 4s, 7s and 9s, the same stroke below vs above the digit",
             "test": bar((1.2, 0.5), 0), "control": bar((-0.2, 0.5), 0), "predict": 1, "digits": [1, 4, 7, 9]},
            {"id": "erase_bottom", "words": "(retired prediction) erase the bottom quarter of the digit vs erase as many ink pixels from its top quarter",
             "test": {"kind": "erase", "rows": (0.75, 1.0)}, "control": {"kind": "erase_matched", "rows": (0.0, 0.25), "from": "start"},
             "predict": -1, "retired": True},
        ],
    },
    "S3.1": {
        "name": "Central-band ink meter",
        "mechanism": "It adds up the ink in a central upright band of the image (about columns 7 to 19, top to bottom); "
                     "ink further out to the left or right counts little. So for the same amount of ink, a narrow upright "
                     "digit fires it more than a wide one.",
        "prediction": "So making the digit 30% wider should lower it compared with making it 30% taller (both add the same "
                      "ink), and turning the digit on its side should lower it compared with mirroring it.",
        "measure": {"label": "the amount of ink in image columns 8 to 19", "fn": lambda X: IV.ink_in(X, (0, 1), (8 / 28, 20 / 28), frame="image"),
                    "predict": 1},
        "probes": [
            {"id": "wider", "words": "the digit stretched 30% wider vs 30% taller", "test": WIDER, "control": TALLER, "predict": -1},
            {"id": "on_side", "words": "the digit turned 90 degrees vs mirrored left to right",
             "test": {"kind": "rotate90"}, "control": {"kind": "mirror_lr"}, "predict": -1},
        ],
    },
    "S3.2": {
        "name": "Ink-amount meter (coarse, from S3.1)",
        "mechanism": "It reads only S3.1's map, at the coarsest level, and its average again tracks how much ink there is, "
                     "but less tightly than S3.1 does. What it adds beyond the amount of ink has no reading we could find.",
        "prediction": "So thickening the strokes should raise it and thinning them should lower it.",
        "measure": {"label": "the amount of ink (ink pixels)", "fn": ink, "predict": 1, "is_ink": True},
        "probes": [INK_PROBES[0],
                   {"id": "thin", "words": "thin every stroke by one pixel (erosion) vs move the digit one pixel right",
                    "test": {"kind": "erode", "size": 2}, "control": SHIFT1, "predict": -1}],
    },
    "S4": {
        "name": "Left-of-centre stroke detector",
        "mechanism": "It counts ink in an upright band just left of the image centre (about columns 8 to 11, mid-height) "
                     "and counts ink further left (columns 0 to 6) against it. It fires when the digit has an upright "
                     "stroke on its left side near the middle, as the left side of a 0 or 6 does, and stays low when the "
                     "left side is open, as in a 3 or 7.",
        "prediction": "So an upright stroke along the left edge of the digit should raise it far more than the same stroke "
                      "along the right edge, and moving the whole digit 4 pixels left should lower it compared with moving it 4 pixels right.",
        "measure": {"label": "the amount of ink in image columns 8 to 11, rows 6 to 21",
                    "fn": lambda X: IV.ink_in(X, (6 / 28, 22 / 28), (8 / 28, 12 / 28), frame="image"), "predict": 1},
        "probes": [
            {"id": "left_vs_right", "words": "a 10-pixel upright stroke along the left edge of the digit vs the same stroke along its right edge",
             "test": bar((0.5, 0.0), 90), "control": bar((0.5, 1.0), 90), "predict": 1},
            {"id": "shift_left", "words": "the digit moved 4 pixels left vs 4 pixels right",
             "test": {"kind": "shift", "dc": -4}, "control": {"kind": "shift", "dc": 4}, "predict": -1},
        ],
    },
    "S5": {
        "name": "Top-bar detector",
        "mechanism": "It responds to a horizontal stroke across the top of the digit (image rows about 4 to 9, central "
                     "columns), as in a 2, 3, 5 or 7. Ink across the middle rows holds it back, and a stroke at the bottom "
                     "raises it a little.",
        "prediction": "So a horizontal stroke along the top of the digit should raise it far more than the same stroke along "
                      "the bottom, and erasing the top quarter of the digit should lower it more than erasing as many ink "
                      "pixels from the bottom quarter.",
        "measure": {"label": "the amount of ink in the top quarter of the digit's box", "fn": lambda X: IV.ink_in(X, (0, 0.25), (0, 1)),
                    "predict": 1},
        "probes": [
            {"id": "top_vs_bottom", "words": "a 10-pixel horizontal stroke along the top of the digit vs the same stroke along its bottom",
             "test": bar((0.0, 0.5), 0), "control": bar((1.0, 0.5), 0), "predict": 1},
            {"id": "erase_top", "words": "erase the top quarter of the digit vs erase as many ink pixels from its bottom quarter",
             "test": {"kind": "erase", "rows": (0.0, 0.25)}, "control": {"kind": "erase_matched", "rows": (0.75, 1.0), "from": "end"},
             "predict": -1},
        ],
    },
    "S6.1": {
        "name": "Falling-diagonal stroke detector",
        "mechanism": "A full-resolution map that responds to strokes running from upper left to lower right ('\\'); "
                     "strokes the other way ('/') leave it unchanged. Wide digits, with long strokes, suit it.",
        "prediction": "So a '\\' stroke should raise it far more than the same stroke drawn '/', and slanting the digit to "
                      "the left should raise it compared with slanting it to the right by the same amount.",
        "measure": {"label": "the amount of '\\' stroke edge", "fn": lambda X: IV.orientation_energy(X)["diag_down"], "predict": 1},
        "probes": [
            {"id": "falling_vs_rising", "words": "a 10-pixel '\\' stroke through the centre of the digit vs the same stroke '/'",
             "test": bar((0.5, 0.5), -45), "control": bar((0.5, 0.5), 45), "predict": 1},
            {"id": "lean_left", "words": "the digit sheared to lean left vs to lean right (by 0.3 pixels per row)",
             "test": {"kind": "shear", "s": 0.3}, "control": {"kind": "shear", "s": -0.3}, "predict": 1},
        ],
    },
    "S6.2": {
        "name": "Horizontal-stroke detector (second)",
        "mechanism": "It reads S2.1's horizontal-stroke map and S5's top-bar map and, like S2.1, responds along "
                     "horizontal strokes, most strongly along the bottom of the digit; upright strokes give it little.",
        "prediction": "So a horizontal stroke should raise it more than the same stroke drawn upright, and making the "
                      "digit wider should raise it more than making it taller by the same factor.",
        "measure": {"label": "the amount of horizontal-stroke edge", "fn": horizontal_edges, "predict": 1},
        "probes": [
            {"id": "h_vs_v", "words": "a 10-pixel horizontal stroke through the centre of the digit vs the same stroke upright",
             "test": H_MID, "control": V_MID, "predict": 1},
            {"id": "wider", "words": "the digit stretched 30% wider vs 30% taller", "test": WIDER, "control": TALLER, "predict": 1},
        ],
    },
    "S7": {
        "name": "Low-crossbar detector",
        "mechanism": "It reads S2.1's and S6.2's horizontal-stroke maps and weighs them by height: horizontal ink in the "
                     "lower half of the image, right of centre, raises it, and ink near the very top of the image (above "
                     "about row 7) lowers it. It fires for a crossbar or flat stroke sitting low with little ink at the top.",
        "prediction": "So a horizontal stroke in the lower half of the image (row 20, right of centre) should raise it far "
                      "more than the same stroke near the top (row 5), and moving the digit 4 pixels down should raise it "
                      "compared with moving it 4 pixels up.",
        "measure": {"label": "the ink low on the right (rows 15 to 24, right half) minus the ink in the top rows (0 to 6)",
                    "fn": low_minus_top, "predict": 1},
        "probes": [
            {"id": "low_vs_top", "words": "a 10-pixel horizontal stroke at image row 20, right of centre, vs the same stroke at row 5",
             "test": bar((20 / 27, 17 / 27), 0, frame="image"), "control": bar((5 / 27, 17 / 27), 0, frame="image"), "predict": 1},
            {"id": "shift_down", "words": "the digit moved 4 pixels down vs 4 pixels up",
             "test": {"kind": "shift", "dr": 4}, "control": {"kind": "shift", "dr": -4}, "predict": 1},
        ],
    },
    "S8": {
        "name": "Upright-stroke detector",
        "mechanism": "It responds to tall upright strokes, straight up or leaning right ('/'), as in "
                     "a 1, the stem of a 4 or 7 and the sides of a 0; horizontal strokes across the middle hold it back.",
        "prediction": "So an upright stroke should raise it more than the same stroke drawn horizontally, a '/' stroke more "
                      "than a '\\' stroke, and making the digit 30% taller more than making it 30% wider.",
        "measure": {"label": "the upright and '/' stroke edge minus the horizontal and '\\' stroke edge",
                    "fn": upright_balance, "predict": 1},
        "probes": [
            {"id": "v_vs_h", "words": "a 10-pixel upright stroke through the centre of the digit vs the same stroke horizontal",
             "test": V_MID, "control": H_MID, "predict": 1},
            {"id": "rising_vs_falling", "words": "a 10-pixel '/' stroke through the centre vs the same stroke '\\'",
             "test": bar((0.5, 0.5), 45), "control": bar((0.5, 0.5), -45), "predict": 1},
            {"id": "taller", "words": "the digit stretched 30% taller vs 30% wider", "test": TALLER, "control": WIDER, "predict": 1},
        ],
    },
}

COMPOSITIONS = {
    "S1": {
        "name": "Shared ink measure",
        "combine": "S1.1 and S1.2 both read the image and run in parallel. Their main exits agree almost exactly "
                   "(r = {c[r_children]:.3f}), so the pair hands one measure, how much ink the digit has, to most later "
                   "structures (S2, S3, S4, S5, S6 and S8) through different exits.",
        "adds": "The combination adds no new behaviour: a 10-class classifier on both children's pooled exits reaches "
                "{c[acc_both]:.0%} (cross-validated), against {c[acc_best]:.0%} for the better child alone.",
        "probes": [{"id": "thicken", "child": "S1.1", **INK_PROBES[0]}, {"id": "thicken", "child": "S1.2", **INK_PROBES[0]}],
        "probe_note": "Both children respond to thicker strokes in the same way, as one ink measure would.",
    },
    "S2": {
        "name": "Horizontal strokes, then the lowest one",
        "combine": "S2.1 finds horizontal strokes anywhere in the digit; S2.2 reads S2.1's map (with S1.1's ink map) and "
                   "keeps the lowest one, the stroke with no ink below it (S2.1 feeds S2.2).",
        "adds": "What the combination adds is location: a stroke along the bottom instead of the top moves S2.2 by "
                "{p[S2.2:bottom_vs_top][effect_sd]:+.2f} SD but S2.1 by only {p[S2.1:bottom_vs_top][effect_sd]:+.2f} SD. "
                "A 10-class classifier on both children's pooled exits reaches {c[acc_both]:.0%} (cross-validated), "
                "against {c[acc_best]:.0%} for the better child alone.",
        "probes": [{"id": "bottom_vs_top", "child": "S2.2", "predict": 1, "words": "a horizontal stroke along the bottom vs the top (S2.2)",
                    "test": bar((1.0, 0.5), 0), "control": bar((0.0, 0.5), 0)},
                   {"id": "bottom_vs_top", "child": "S2.1", "predict": 0, "words": "the same edit (S2.1)",
                    "test": bar((1.0, 0.5), 0), "control": bar((0.0, 0.5), 0)}],
    },
    "S3": {
        "name": "Central ink, re-read coarsely",
        "combine": "S3.1 adds up the ink in a central upright band; S3.2 reads only S3.1's map and re-weights it at the "
                   "coarsest level (S3.1 feeds S3.2).",
        "adds": "What the combination adds: S3.2 keeps the amount of ink but drops the band's preference for narrow "
                "digits; making the digit wider instead of taller moves S3.1 by {p[S3.1:wider][effect_sd]:+.2f} SD and "
                "S3.2 by {p[S3.2:wider][effect_sd]:+.2f} SD. A 10-class classifier on both children's pooled exits reaches "
                "{c[acc_both]:.0%} (cross-validated), against {c[acc_best]:.0%} for the better child alone.",
        "probes": [{"id": "wider", "child": "S3.1", "predict": -1, "words": "the digit 30% wider vs 30% taller (S3.1)",
                    "test": WIDER, "control": TALLER},
                   {"id": "wider", "child": "S3.2", "predict": 0, "words": "the same edit (S3.2)", "test": WIDER, "control": TALLER}],
    },
    "S6": {
        "name": "Flat and falling strokes of wide digits",
        "combine": "S6.1 (strokes falling left to right, '\\') and S6.2 (horizontal strokes) run in parallel and both feed "
                   "the scores. Both favour wide digits.",
        "adds": "Making the digit wider instead of taller raises S6.1 by {p[S6.1:wider][effect_sd]:+.2f} SD and S6.2 by "
                "{p[S6.2:wider][effect_sd]:+.2f} SD. A 10-class classifier on both children's pooled exits reaches "
                "{c[acc_both]:.0%} (cross-validated), against {c[acc_best]:.0%} for the better child alone.",
        "probes": [{"id": "wider", "child": "S6.1", "predict": 1, "words": "the digit 30% wider vs 30% taller (S6.1)",
                    "test": WIDER, "control": TALLER},
                   {"id": "wider", "child": "S6.2", "predict": 1, "words": "the same edit (S6.2)", "test": WIDER, "control": TALLER}],
    },
}

ROOT = {
    "probes": [
        {"id": "one_crossbar", "digit": 1, "words": "on 1s, a 10-pixel horizontal stroke across the middle vs the same stroke drawn along the 1 (the '1' score margin)",
         "test": H_MID, "control": V_MID, "predict": -1, "wakes": "S2 and S6"},
        {"id": "six_topbar", "digit": 6, "words": "on 6s, a horizontal stroke along the top vs along the bottom (the '6' score margin)",
         "test": bar((0.0, 0.5), 0), "control": bar((1.0, 0.5), 0), "predict": -1, "wakes": "S5"},
        {"id": "four_topbar", "digit": 4, "words": "on 4s, a horizontal stroke along the top vs along the bottom (the '4' score margin)",
         "test": bar((0.0, 0.5), 0), "control": bar((1.0, 0.5), 0), "predict": -1, "wakes": "S5"},
        {"id": "three_leftstroke", "digit": 3, "words": "on 3s, an upright stroke along the left edge vs along the right edge (the '3' score margin)",
         "test": bar((0.5, 0.0), 90), "control": bar((0.5, 1.0), 90), "predict": -1, "wakes": "S4"},
    ],
}
