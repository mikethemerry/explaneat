"""The concept probe (the "understood" level): a synthetic leaf with a known concept scores
high, an unrelated concept scores low, grades follow the stated thresholds, the libraries
compute, and an ad-hoc expression works."""

import numpy as np
import pytest

from explaneat.analysis import concept_probe as CP
from explaneat.analysis import concepts as C
from tests.test_plain_language import helpers as H


@pytest.fixture(scope="module")
def world():
    """The additive synthetic network: gate 5 = relu(1.5 z(dose) - 1), group B straight to the output."""
    raw = H.raw_data(400, seed=0)
    mean, scale = raw.mean(axis=0), raw.std(axis=0)
    X = (raw - mean) / scale
    s = H.structure("additive")
    gate = H.operations("additive")[0]["params"]
    gate = {"id": "ann_1", **gate}
    return {"raw": raw, "X": X, "structure": s, "gate": gate, "names": list(H.FEATURES)}


def test_known_concept_scores_high_and_unrelated_low(world):
    r = CP.probe_leaf(world["structure"], world["X"], world["raw"], world["names"], "synthetic", world["gate"],
                      concept="dose")
    assert r["exit_nodes"] == ["5"]
    assert r["per_exit"]["5"]["leaf_gated"] is True
    assert r["auc"] is not None and r["auc"] >= 0.99          # the gate opens with the dose
    assert r["grade"] == "meaningful"
    assert r["meaning"]["grade"] == "meaningful" and r["meaning"]["concept"] == "dose"
    assert r["meaning"]["probe"]["n"] == 400 and r["meaning"]["probe"]["query_id"] is None
    u = CP.probe_leaf(world["structure"], world["X"], world["raw"], world["names"], "synthetic", world["gate"],
                      concept="noise")
    assert u["score"] is not None and u["score"] < 0.2
    assert u["grade"] == "closed_form_only"
    assert u["mi"] < r["mi"]


def test_library_sweep_finds_the_dose(world):
    r = CP.probe_leaf(world["structure"], world["X"], world["raw"], world["names"], "synthetic", world["gate"])
    assert r["concept"] is None
    best = r["per_exit"]["5"]["best"]
    assert best["concept"].startswith("dose")                 # "dose" or "dose > median"
    assert r["grade"] == "meaningful"
    assert r["meaning"]["concept"].startswith("dose")
    assert r["n_library_concepts"] >= 5


def test_adhoc_expression_and_threshold(world):
    r = CP.probe_leaf(world["structure"], world["X"], world["raw"], world["names"], "synthetic", world["gate"],
                      concept="dose > 6.5")
    assert r["concept_source"] == "expression" and r["concept_kind"] == "binary"
    assert r["auc"] is not None and r["auc"] > 0.9
    with pytest.raises(ValueError):
        CP.probe_leaf(world["structure"], world["X"], world["raw"], world["names"], "synthetic", world["gate"],
                      concept="no_such_column > 1")


def test_grade_thresholds():
    assert CP.meaning_grade(None, concept_given=False) == "unnamed"
    assert CP.meaning_grade(0.95) == "meaningful"
    assert CP.meaning_grade(0.8) == "meaningful"
    assert CP.meaning_grade(0.79) == "partial"
    assert CP.meaning_grade(0.5) == "partial"
    assert CP.meaning_grade(0.49) == "closed_form_only"
    assert CP.meaning_grade(None) == "closed_form_only"
    # an AUC of 0.9 is the meaningful line, 0.75 the partial line
    assert CP.probe_score({"r2": 0.1, "auc": 0.9}) == 0.8
    assert CP.probe_score({"r2": 0.1, "auc": 0.75}) == 0.5
    assert CP.grade_from_meaning({"concept": "x", "probe": {"r2": 0.6, "auc": None}}) == "partial"
    assert CP.grade_from_meaning({"concept": "x", "probe": {"score": 0.85}}) == "meaningful"
    assert CP.grade_from_meaning({"concept": None, "probe": {"score": 0.85}}) == "unnamed"
    assert CP.grade_from_meaning(None) == "unnamed"


def test_r2_and_mi_basics():
    rng = np.random.default_rng(0)
    c = rng.normal(size=500)
    assert CP.r_squared(c, 3 * c + 1) == pytest.approx(1.0)
    assert CP.r_squared(c, rng.normal(size=500)) < 0.05
    mi = CP.mutual_information(c, 3 * c + 1)["mi"]
    assert mi > 2.0 and CP.mutual_information(c, rng.normal(size=500))["mi"] < 0.3


def test_connect4_library():
    rng = np.random.default_rng(1)
    X = rng.choice([-1.0, 0.0, 1.0], size=(50, 42))
    X[0] = 0.0
    X[0, [0, 1, 2]] = 1.0          # x on a1, a2, a3: one threat at a4 (playable now)
    cs = {c.name: c for c in C.c4_concepts(X)}
    assert cs["x_thr"].values[0] == 1 and cs["x_thr_now"].values[0] == 1 and cs["x_open_3"].values[0] == 1
    assert cs["o_thr"].values[0] == 0 and cs["x_thr_even"].values[0] == 1 and cs["centre_x"].values[0] == 0
    assert cs["n_pieces"].values[0] == 3 and cs["x_mean_row"].values[0] == 2.0
    lib = C.library_concepts("connect4_win_ordinal", X, C.C4_FEATURES)
    assert any(c.source == "domain" for c in lib) and any(c.source == "input" for c in lib)
    with pytest.raises(ValueError):
        C.library_concepts("connect4_win_ordinal", X[:, :10], C.C4_FEATURES[:10])


def test_krk_library():
    # white king a1, rook b2, black king c3: rook adjacent to both kings -> protected, no capture
    X = np.array([[1, 1, 2, 2, 3, 3], [1, 1, 5, 5, 6, 6], [1, 1, 2, 8, 1, 5]])
    cs = {c.name: c for c in C.krk_concepts(X)}
    assert cs["rook_adjacent"].values.tolist() == [1, 1, 0]
    assert cs["capture"].values.tolist() == [0, 1, 0]
    assert cs["rule_draw"].values[1] == 1
    assert C.library_for("krk_draw_ordinal") == "krk" and C.library_for("Heart Disease") == "tabular"
