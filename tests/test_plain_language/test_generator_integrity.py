"""Claim-generator fixes from the 2026-11-11 errata (fix 5 and errata v2):

* a shape exception is attributed to a group only when it holds within the group (heart-s2 C30, E7);
* a fold that depends on a category is reported per group, not as a pooled median no patient has,
  and an "only if" interaction holds only within its window (heart100-s7 ann_38, E15);
* cross-claim consistency checks;
* the percentage lint (heart100-s0, E13 withdrawn).

The heart-s2 / heart100-s7 / heart100-s0 regressions re-extract from the research database (read
only, through the verify cache) and are skipped where it is not available."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from explaneat.analysis import plain_language_shapes as S
from explaneat.analysis.plain_language import folds_by_group
from explaneat.analysis.plain_language_store import (context_probabilities, lint_draft_v5, lint_percents,
                                                     lint_reference_risk, lint_window)

REPO = Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------------------- exception rule
def test_exception_named_only_when_it_holds_within_the_group():
    classes = ["hinge"] * 150 + ["step"] * 21 + ["flat"] * 69 + ["step"] * 2
    in_g = np.zeros(242, bool)
    in_g[150:240] = True                    # 90 patients: 21 step, 69 flat (unaffected)
    ex = S._exception_within("step", [{"group": "patients with X", "in_group": in_g}], np.arange(242), classes, 0.14)
    assert ex["class"] == S.MIXED_WITHIN and ex["alt_class"] == "step"
    assert (ex["n_alt"], ex["n_group"]) == (21, 90)
    classes2 = ["hinge"] * 150 + ["step"] * 88 + ["flat"] * 2 + ["hinge"] * 2
    ex2 = S._exception_within("step", [{"group": "patients with X", "in_group": in_g}], np.arange(242), classes2, 0.37)
    assert ex2["class"] == "step"           # 88 of 90 >= 90%


def test_exception_phrase_and_lint_for_a_mixed_subgroup():
    from explaneat.analysis.plain_language_templates import exception_phrase
    e = {"class": S.MIXED_WITHIN, "alt_class": "step", "who": ["patients with a reversible defect"],
         "n_alt": 21, "n_group": 90}
    layer = {"population": {"plural": "patients"}, "codebook": {"patient": "patients"}}
    assert "for some patients with a reversible defect (21 of the 90 sampled)" in exception_phrase(layer, e)
    shape = {"class": "hinge", "status": "mixed", "exceptions": [e]}
    bad = ("older age raises it up to about 56 years, then no further for most patients, but for patients with "
           "a reversible defect it rises in a jump")
    good = ("older age raises it up to about 56 years, then no further for most patients, but for some patients "
            "with a reversible defect it rises in a jump")
    assert not S.shape_phrase_ok(bad, shape, "direction")[0]
    assert S.shape_phrase_ok(good, shape, "direction")[0]


def test_cross_claim_conflict_between_an_exception_and_an_interaction():
    group = "patients with a reversible defect on the thallium scan"
    shape_claim = {"id": "C30", "type": "direction", "annotation": {"id": "ann_31"},
                   "shape": {"exceptions": [{"class": "step", "who": [group]}]}}
    inter = {"id": "C33", "type": "interaction", "annotation": {"id": "ann_31"}, "statement": {},
             "groups": [{"group": group, "risk_from": 0.42, "risk_to": 0.42, "part_effect": 0.0}]}
    conf = S.cross_claim_conflicts([shape_claim, inter])
    assert conf and conf[0]["claim"] == "C30" and conf[0]["other"] == "C33"
    shape_claim["shape"]["exceptions"][0]["class"] = S.MIXED_WITHIN
    assert S.cross_claim_conflicts([copy.deepcopy(shape_claim), inter]) == []


# ----------------------------------------------------------------------------- folds and windows
def test_folds_by_group_when_the_pooled_median_is_no_groups_fold():
    rng = np.random.RandomState(0)
    t = np.concatenate([160 + rng.normal(0, 1, 121), 176 + rng.normal(0, 1, 121)])
    rows = np.ones(242, bool)
    a = np.r_[np.ones(121, bool), np.zeros(121, bool)]
    fb = folds_by_group(t, rows, [({"group": "normal ECG"}, a), ({"group": "other ECG"}, ~a)])
    assert fb and [round(g["value"]) for g in fb] == [160, 176]
    same = np.concatenate([168 + rng.normal(0, 3, 242)])
    assert folds_by_group(same, rows, [({"group": "normal ECG"}, a), ({"group": "other ECG"}, ~a)]) is None


def test_window_lint_and_cross_claim_window_check():
    c82 = {"id": "C82", "type": "interaction", "annotation": {"id": "ann_38"},
           "statement": {"variable": "thalach", "only_within_window": True,
                         "change": {"from": {"text": "118 beats per minute", "value": 118.0},
                                    "to": {"text": "175 beats per minute", "value": 175.0}},
                         "weaker_group": {"group": "other ECG", "risk_from": 0.38, "risk_to": 0.38,
                                          "beyond_window": {"above": {"to_text": "202 beats per minute"}}}}}
    assert lint_window("with any other resting ECG the risk stays at about 38%", [c82])
    assert not lint_window("with any other resting ECG it stays at about 38% between 118 and 175 beats per "
                           "minute", [c82])
    c83 = {"id": "C83", "type": "threshold", "annotation": {"id": "ann_38"}, "statement": {"variable": "thalach"},
           "estimate": {"threshold": {"by_group": [{"group": "other ECG", "value": 176.1}]}}}
    conf = S.cross_claim_conflicts([c82, c83])
    assert conf and conf[0]["other"] == "C83" and "176.1" in conf[0]["why"]


# ----------------------------------------------------------------------------- percentage lint (E13)
def test_lint_percents_and_the_typical_patients_risk():
    claims = [{"evidence": {"who": {"categories": [{"share_out": 0.192982}]}}, "shape": {"share_x": 0.193}}]
    layer = {"reference_patient": {"predicted_risk": 0.197958}}
    text = "For a typical patient the model predicts a risk of heart disease of about 20%"
    assert lint_percents(text, claims, context_probabilities(layer)) == []
    assert lint_reference_risk(text, layer) == []
    wrong = text.replace("20%", "19%")
    assert lint_reference_risk(wrong, layer)            # the withdrawn E13 "correction" is caught
    assert lint_reference_risk("for the typical patient it acts on, the predicted risk is about 42%", layer) == []


def test_heart100_s0_drafts():
    """Regression (E13): the pre-errata "about 20%" lints clean; "about 19%" is flagged."""
    snap = REPO / "experiments/errata/2026-11-11/snapshot/heart100-s0/plain_language.json"
    if not snap.is_file():
        pytest.skip("snapshot not in this checkout")
    L = json.loads(snap.read_text())
    assert not [e for e in lint_draft_v5(L, L["drafted"]) if e.startswith("lead[0]")]
    L2 = copy.deepcopy(L)
    L2["drafted"]["lead"][0]["text"] = L2["drafted"]["lead"][0]["text"].replace("about 20%", "about 19%")
    assert [e for e in lint_draft_v5(L2, L2["drafted"]) if e.startswith("lead[0]")]


# ----------------------------------------------------------------------------- research genomes
def _claims_or_skip(label):
    from explaneat.verify import labelled
    from explaneat.verify.model import load
    from explaneat.verify.recheck import recompute_claims
    t = labelled.current_target(label)
    try:
        vm = load(t.genome_id, None)
    except Exception as e:
        pytest.skip(f"research genomes not available: {e}")
    return t, recompute_claims(vm, cache=True)


@pytest.mark.slow
@pytest.mark.db
def test_heart_s2_reextraction_e7():
    from explaneat.verify import labelled
    from explaneat.verify.recheck import lint_prose
    t, claims = _claims_or_skip("heart-s2")
    c30 = next(c for c in claims if c["key"] == "ann_31:direction:age")
    ex = c30["shape"]["exceptions"][0]
    assert ex["class"] == S.MIXED_WITHIN and ex["alt_class"] == "step" and ex["n_group"] == 90
    assert ex["n_alt"] / ex["n_group"] < S.CONSISTENT_SHARE
    e7 = {"blocks[1][2]", "readdown.ann_31.evidence[1]"}
    assert not [f for f in lint_prose(t.layer, claims) if f["location"] in e7]   # the E7 correction aligns
    before = lint_prose(labelled.snapshot_target("heart-s2").layer, claims)
    assert {f["location"] for f in before} >= {"blocks[1][2]", "readdown.ann_31.evidence[1]"}


@pytest.mark.slow
@pytest.mark.db
def test_heart100_s7_reextraction_per_group_folds():
    from explaneat.verify.recheck import lint_prose
    t, claims = _claims_or_skip("heart100-s7")
    th = next(c for c in claims if c["key"] == "ann_38:threshold:thalach")
    groups = th["statement"]["threshold"]["by_group"]
    assert [round(g["value"]) for g in groups] == [160, 176]
    inter = next(c for c in claims if c["key"] == "ann_38:interaction:thalachxrestecg")
    assert inter["statement"]["only_within_window"]
    locs = {f["location"] for f in lint_prose(t.layer, claims) if "C82" in f["note"] or "do not change" in f["note"]}
    assert {"blocks[2][1]", "readdown.ann_38.evidence[2]"} <= locs
    # the stored (legacy) claims: the cross-claim check flags C82 against C83
    assert any(c["claim"] == "C82" and c["other"] == "C83"
               for c in S.cross_claim_conflicts(copy.deepcopy(t.layer["claims"])))
