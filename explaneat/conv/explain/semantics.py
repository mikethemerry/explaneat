"""Plain-language readings of ConvNEAT nodes, anchored on real concepts.

**Concept libraries** (one yes / no value per image, never derived from the network):

* synthetic: the four planted shapes (horizontal bar, blob, vertical bar, diagonal)
  and the label itself, "bar AND blob", so a node computing the conjunction is recognised;
* digits / MNIST: the rule-based stroke concepts of ``strokes.py`` (closed upper loop,
  closed lower loop, open left side, top stroke, waist, left-heavy) and the class itself
  ("is an 8").

**Evidence per node** (test split). The node's map is pooled to one number per image:
its **peak** (spatial max: "does it fire anywhere") for hidden nodes, its **average** for
the output map (the head averages it). For each concept, ``s = 2 AUC - 1`` of that number
for concept present vs absent (-1 to +1). The node **fires** on an image when the pooled
value reaches a threshold ``t``: the cut that best separates the node's top concept
(maximum |TPR - FPR|, Youden). The plain number is then "fires on a % of images with the
concept and b % without", at that one threshold, for every concept.

**Reading rule** (deterministic):

* inactive: pooled value <= 0 on every image;
* top concept c1 = argmax |s|. |s| < 0.3: *unexplained* ("no clear concept");
  0.3 <= |s| < 0.5: *weak*; |s| >= 0.5: *concept* (the strong threshold of the probe);
* a second concept c2 is named only if it is also strong and not confounded with c1
  (|phi| < 0.3 between the two concept indicators on the test images). So "bar AND
  blob" is not repeated next to "horizontal bar", and on digits the loops are not
  repeated next to "is an 8";
* on digits, for the strongest stroke concept the same rates are also computed **within
  each class**, which separates "tracks the stroke" from "tracks the class".
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from .strokes import STROKE_RULES, STROKE_SHORT, STROKE_WORDS, STROKES, stroke_concepts

STRONG = 0.5
WEAK = 0.3
PHI_MAX = 0.3

SYN_WORDS = {"hbar": "a horizontal bar", "blob": "a blob", "vbar": "a vertical bar", "diag": "a diagonal line",
             "both": "both a horizontal bar and a blob"}
SYN_PLURAL = {"hbar": "horizontal bars", "blob": "blobs", "vbar": "vertical bars", "diag": "diagonal lines",
              "both": "the bar-AND-blob combination"}
SYN_SHORT = {"hbar": "bar", "blob": "blob", "vbar": "vertical-bar", "diag": "diagonal", "both": "bar-AND-blob"}


def concept_library(dataset: str, data: dict, class_names) -> dict:
    """Concept names, words and the test-split indicator matrix."""
    if dataset == "synthetic":
        Cp = data["c_test"]
        names = ["hbar", "blob", "vbar", "diag", "both"]
        C = np.column_stack([Cp, Cp[:, 0] & Cp[:, 1]]).astype(np.int64)
        words = dict(SYN_WORDS)
        plural = dict(SYN_PLURAL)
        short = dict(SYN_SHORT)
        rules = {"hbar": "planted horizontal bar", "blob": "planted blob", "vbar": "planted vertical bar (distractor)",
                 "diag": "planted diagonal (distractor)", "both": "the label: bar AND blob both planted"}
        neg = {"hbar": "Fires when there is no horizontal bar", "blob": "Fires when there is no blob",
               "vbar": "Fires when there is no vertical bar", "diag": "Fires when there is no diagonal line",
               "both": "Fires unless the image has both a bar and a blob"}
        neg_short = {"hbar": "no-bar", "blob": "no-blob", "vbar": "no-vertical-bar", "diag": "no-diagonal",
                     "both": "not-both"}
        return {"names": names, "C": C, "words": words, "plural": plural, "short": short, "rules": rules,
                "neg": neg, "neg_short": neg_short, "class_concept": "both", "kind": "planted"}
    a, b = class_names
    Cs = stroke_concepts(data["X_test"])
    names = list(STROKES) + ["class1"]
    C = np.column_stack([Cs, data["y_test"]]).astype(np.int64)
    words = dict(STROKE_WORDS)
    words["class1"] = art(b)
    plural = {k: (v.split(" ", 1)[1] if v.startswith(("a ", "an ")) else v) for k, v in STROKE_WORDS.items()}
    plural["class1"] = f"{b}s"
    short = {k: v.replace(" ", "-") for k, v in STROKE_SHORT.items()}
    short["class1"] = b
    rules = dict(STROKE_RULES)
    rules["class1"] = f"the true class is {b} (not {a})"
    neg = {"upper_loop": "Fires when there is no closed upper loop", "lower_loop": "Fires when there is no closed lower loop",
           "open_left": "Fires when the left side is closed (not open as in a 3)",
           "top_bar": "Fires when there is no horizontal stroke along the top",
           "waist": "Fires when there is no narrow central crossing",
           "left_heavy": "Fires on digits with more ink on the right half", "class1": f"Fires on {a}s rather than {b}s"}
    neg_short = {"upper_loop": "no-upper-loop", "lower_loop": "no-lower-loop", "open_left": "closed-left-side",
                 "top_bar": "no-top-stroke", "waist": "no-waist", "left_heavy": "right-heavy", "class1": a}
    return {"names": names, "C": C, "words": words, "plural": plural, "short": short, "rules": rules,
            "neg": neg, "neg_short": neg_short, "class_concept": "class1", "kind": "strokes", "class_names": [a, b]}


def _auc(a, y):
    if np.ptp(a) < 1e-12 or y.min() == y.max():
        return 0.5
    return float(roc_auc_score(y, a))


def _phi(u, v):
    if u.std() == 0 or v.std() == 0:
        return 0.0
    return float(np.corrcoef(u, v)[0, 1])


def node_evidence(pooled: np.ndarray, lib: dict, y: np.ndarray) -> dict:
    """Concept evidence for one node's pooled values on the test split."""
    names, C = lib["names"], lib["C"]
    s = {c: 2 * _auc(pooled, C[:, j]) - 1 for j, c in enumerate(names)}
    inactive = bool(pooled.max() <= 0)
    order = sorted(names, key=lambda c: -abs(s[c]))
    c1 = order[0]
    j1 = names.index(c1)
    if inactive or np.ptp(pooled) < 1e-12:
        t = float(pooled.max()) + 1.0 if inactive else float(pooled.max())
    else:
        fpr, tpr, thr = roc_curve(C[:, j1], pooled)
        k = int(np.argmax(np.abs(tpr - fpr)))
        t = float(min(thr[k], pooled.max()))
    fires = pooled >= t
    rates = {c: {"with": float(fires[C[:, j] == 1].mean()) if (C[:, j] == 1).any() else None,
                 "without": float(fires[C[:, j] == 0].mean()) if (C[:, j] == 0).any() else None,
                 "n_with": int((C[:, j] == 1).sum()), "n_without": int((C[:, j] == 0).sum())}
             for j, c in enumerate(names)}
    top = abs(s[c1])
    status = "inactive" if inactive else ("concept" if top >= STRONG else "weak" if top >= WEAK else "unexplained")
    chosen = [c1] if status in ("concept", "weak") else []
    if status == "concept":
        for c in order[1:]:
            if abs(s[c]) < STRONG:
                break
            if all(abs(_phi(C[:, names.index(c)], C[:, names.index(d)])) < PHI_MAX for d in chosen):
                chosen.append(c)
                break
    secondary = None
    if status == "concept":
        for c in order[1:]:
            if c in chosen or abs(s[c]) < WEAK:
                continue
            if abs(s[c]) >= STRONG:
                continue
            if all(abs(_phi(C[:, names.index(c)], C[:, names.index(d)])) < PHI_MAX for d in chosen):
                secondary = c
            break
    within = None
    if lib["kind"] == "strokes" and not inactive:
        strokes = [c for c in order if c != "class1"]
        cs = strokes[0]
        js = names.index(cs)
        within = {"concept": cs, "s_overall": s[cs], "by_class": {}}
        for cl in (0, 1):
            m = y == cl
            ci = C[m, js]
            if ci.sum() >= 15 and (1 - ci).sum() >= 15:
                within["by_class"][str(cl)] = {
                    "s": 2 * _auc(pooled[m], ci) - 1,
                    "with": float(fires[m][ci == 1].mean()), "without": float(fires[m][ci == 0].mean()),
                    "n_with": int(ci.sum()), "n_without": int((1 - ci).sum())}
    return {"s": s, "order": order, "status": status, "concepts": chosen, "secondary": secondary, "threshold": t,
            "fires": fires, "fire_rate": float(fires.mean()), "rates": rates, "within": within,
            "peak_pctl": None}


def concept_key(ev: dict) -> Optional[tuple]:
    if ev["status"] != "concept":
        return None
    return tuple(sorted((c, int(np.sign(ev["s"][c]))) for c in ev["concepts"]))


def short_name(ev: dict, lib: dict, class_names=("0", "1")) -> str:
    st = ev["status"]
    if st == "inactive":
        return "inactive detector"
    if st == "unexplained":
        return "unexplained"
    sh = lib["short"]
    cs = ev["concepts"]
    pos = [c for c in cs if ev["s"][c] > 0]
    neg = [c for c in cs if ev["s"][c] < 0]
    if lib["kind"] == "strokes" and "class1" in cs:
        nm = (class_names[1] if ev["s"]["class1"] > 0 else class_names[0]) + " detector"
    elif pos:
        nm = " + ".join(sh[c] for c in pos) + " detector"
    else:
        nm = lib["neg_short"][neg[0]] + " detector"
    if st == "weak":
        c1 = ev["order"][0]
        nm = (sh[c1] if ev["s"][c1] > 0 else lib["neg_short"][c1]) + " detector"
        if lib["kind"] == "strokes" and c1 == "class1":
            nm = (class_names[1] if ev["s"][c1] > 0 else class_names[0]) + " detector"
        return "possible " + nm
    return nm


def _pct(v):
    return "–" if v is None else f"{100 * v:.0f}%"


def concept_phrase(ev: dict, lib: dict, class_names) -> str:
    """The headline reading (no node ids, no jargon)."""
    st = ev["status"]
    if st == "inactive":
        return "Inactive: it never fires on any test image"
    W, P = lib["words"], lib["plural"]
    c1 = ev["order"][0]
    if st == "unexplained":
        return (f"No clear concept: it fires on {100 * ev['fire_rate']:.0f}% of all test images and does not track any of the "
                f"{len(lib['names'])} concepts (the closest is {W[c1]}, and only weakly)")
    cs = ev["concepts"]
    pos = [c for c in cs if ev["s"][c] > 0]
    neg = [c for c in cs if ev["s"][c] < 0]
    a, b = class_names
    bits = []
    if lib["kind"] == "strokes" and "class1" in cs:
        bits.append(f"Fires on {b}s rather than {a}s" if ev["s"]["class1"] > 0 else f"Fires on {a}s rather than {b}s")
        pos = [c for c in pos if c != "class1"]
        neg = [c for c in neg if c != "class1"]
    if pos:
        bits.append(("Detects " if not bits else "and on ") + " and ".join(P[c] if lib["kind"] == "planted" else W[c] for c in pos))
    if neg:
        if not bits:
            bits.append(lib["neg"][neg[0]])
            neg = neg[1:]
        if neg:
            bits.append("but is silenced by " + " and ".join(P[c] for c in neg))
    s = ", ".join(bits) if len(bits) > 1 and bits[1].startswith("but") else " ".join(bits)
    if st == "weak":
        s = (f"Only weakly linked to a concept: it fires {'more' if ev['s'][c1] > 0 else 'less'} often "
             f"on images with {W[c1]}")
    elif ev.get("secondary"):
        c2 = ev["secondary"]
        s += (", and more weakly responds to " + P[c2] if ev["s"][c2] > 0 else
              ", and is more weakly held back by " + P[c2])
    return s


def rate_sentence(ev: dict, lib: dict, pooling: str = "max") -> Optional[str]:
    if ev["status"] in ("inactive",):
        return None
    c1 = ev["order"][0]
    r = ev["rates"][c1]
    W = lib["words"]
    if pooling == "mean":
        return (f"Its response, averaged over the whole image, is high on {_pct(r['with'])} of test images with "
                f"{W[c1]} and on {_pct(r['without'])} of those without (its peak alone says less).")
    return (f"It fires on {_pct(r['with'])} of test images with {W[c1]} "
            f"and on {_pct(r['without'])} of those without.")


def art(word: str) -> str:
    return ("an " if word[:1] in "8aeiouAEIOU" or word in ("1", "11", "18") else "a ") + word


def within_sentence(ev: dict, lib: dict, class_names) -> Optional[str]:
    w = ev.get("within")
    if not w or not w["by_class"]:
        return None
    bits = []
    for cl, r in sorted(w["by_class"].items()):
        if abs(r["s"]) >= WEAK:
            bits.append(f"among {class_names[int(cl)]}s it fires on {_pct(r['with'])} with {lib['words'][w['concept']]} "
                        f"and {_pct(r['without'])} without")
    if not bits:
        nm = lib["words"][w["concept"]]
        return (f"Within one class its firing hardly depends on {nm}, so the link to {nm} comes through the class.")
    return "Within one class: " + "; ".join(bits) + "."


def effect_sentence(im: Optional[dict], base_acc: float) -> Optional[str]:
    if not im:
        return None
    a0, a1 = 100 * base_acc, 100 * im["acc"]
    if im["d_acc"] >= 0.01:
        return f"Switching it off drops accuracy from {a0:.1f}% to {a1:.1f}%."
    if im["d_logloss"] >= 0.01:
        return (f"Switching it off barely changes accuracy ({a0:.1f}% to {a1:.1f}%) "
                f"but makes the network less sure of its answers.")
    return f"Switching it off changes nothing that matters ({a0:.1f}% to {a1:.1f}% accuracy)."


def image_tags(lib: dict, i: int, class_names) -> str:
    """Short concept tags for test image i, e.g. 'bar+blob+diag' or '8 · upper loop, lower loop'."""
    names, C = lib["names"], lib["C"]
    if lib["kind"] == "planted":
        tag = {"hbar": "bar", "blob": "blob", "vbar": "vbar", "diag": "diag"}
        on = [tag[c] for c in names[:4] if C[i, names.index(c)]]
        return "+".join(on) if on else "nothing planted"
    cl = class_names[int(C[i, names.index("class1")])]
    on = [STROKE_SHORT[c] for c in STROKES if C[i, names.index(c)]]
    return f"{cl} · " + (", ".join(on) if on else "no stroke found")


def image_facts(lib: dict, i: int, class_names) -> str:
    names, C = lib["names"], lib["C"]
    W = lib["words"]
    if lib["kind"] == "planted":
        has = [W[c] for c in names[:4] if C[i, names.index(c)]]
        hasnt = [W[c].split(" ", 1)[1] for c in ("hbar", "blob") if not C[i, names.index(c)]]
        s = "This image has " + (_join(has) if has else "no planted shape")
        if hasnt and has:
            s += ", and no " + " or ".join(hasnt)
        return s + "."
    cl = class_names[int(C[i, names.index("class1")])]
    on = [W[c] for c in STROKES if C[i, names.index(c)]]
    return f"This is {art(cl)}. The stroke rules find " + (_join(on) if on else "none of their strokes") + "."


def _join(xs: List[str]) -> str:
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]
