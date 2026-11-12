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


# ============================================================================ probe v2 conditions
def test_simplicity_condition(world):
    """A concept counts only when it is simpler than the leaf: a library concept, or an ad-hoc
    expression over at most MAX_CONCEPT_TERMS variables that is not a reweighting of the leaf's
    own entries. Otherwise the grade is `restatement`, whatever the score."""
    entries = ["dose", "grp", "noise"]
    # a library concept is one term; a pairwise product two
    lib = C.library_concepts("synthetic", world["raw"], world["names"], extra_pairs=["dose", "noise"])
    by = {c.name: c for c in lib}
    assert CP.concept_simplicity(by["dose"], entries)["concept_complexity"] == 1
    pair = CP.concept_simplicity(by["dose * noise"], entries)
    assert pair["concept_complexity"] == 2 and pair["simpler_than_leaf"] is True
    # an expression over 4 variables all among the entries is a reweighting of the leaf
    cols = ["dose", "grp:0", "grp:1", "noise"]
    rew = CP.concept_simplicity(C.Concept("dose + `grp:0` - `grp:1` + noise", "", "continuous", "expression",
                                          np.zeros(4)), entries, cols)
    assert rew == {"concept_complexity": 4, "concept_variables": ["dose", "grp:0", "grp:1", "noise"], "max_terms": 3,
                   "simpler_than_leaf": False, "reweighting_of_entries": True}
    # four variables not all in the entries: still too complex, but not a reweighting
    other = CP.concept_simplicity(C.Concept("dose + `grp:0` - `grp:1` + noise", "", "continuous", "expression",
                                            np.zeros(4)), ["dose"], cols)
    assert other["simpler_than_leaf"] is False and other["reweighting_of_entries"] is False
    # the limit is configurable
    assert CP.concept_simplicity(C.Concept("dose + noise", "", "continuous", "expression", np.zeros(4)),
                                 entries, cols, max_terms=1)["simpler_than_leaf"] is False
    assert C.expression_variables("`a b` > 2 and not (c_1 + d) <= abs(e)", ["a b", "c_1", "d", "e", "f"]) \
        == ["a b", "c_1", "d", "e"]
    # on the probe: the gate reads dose; a 4-term signed sum of every input restates it
    r = CP.probe_leaf(world["structure"], world["X"], world["raw"], world["names"], "synthetic", world["gate"],
                      concept="1.5 * dose + 0 * `grp:0` + 0 * `grp:1` + 0 * noise", entry_inputs=entries)
    assert r["r2"] is not None and r["r2"] > 0.5                 # it predicts the leaf
    assert r["grade"] == "restatement" and r["simpler_than_leaf"] is False and r["concept_complexity"] == 4
    assert r["meaning"]["grade"] == "restatement" and r["meaning"]["simpler_than_leaf"] is False
    assert r["meaning"]["concept_complexity"] == 4
    ok = CP.probe_leaf(world["structure"], world["X"], world["raw"], world["names"], "synthetic", world["gate"],
                       concept="dose > 6.5", entry_inputs=entries)
    assert ok["grade"] == "meaningful" and ok["simpler_than_leaf"] is True and ok["concept_complexity"] == 1
    assert CP.meaning_grade(0.95, simpler=False) == "restatement"
    assert CP.grade_from_meaning({"concept": "x", "simpler_than_leaf": False, "probe": {"score": 0.95}}) == "restatement"
    assert CP.grade_from_meaning({"concept": "x", "simpler_than_leaf": True, "probe": {"score": 0.95}}) == "meaningful"
    assert "restatement" in CP.GRADES


def test_support_condition():
    """A binary concept's AUC counts only against a gate of comparable support; otherwise the score
    is r2, so a rare flag cannot grade a leaf meaningful while explaining none of its variance."""
    rng = np.random.default_rng(3)
    n = 2000
    y = rng.normal(size=n)                                      # a continuous (ungated) exit
    rare = np.zeros(n)
    rare[np.argsort(y)[-6:]] = 1                                # 0.3% of rows, the six largest exits
    c = C.Concept("rare_flag", "", "binary", "domain", rare)
    s = CP.score_concept(c, y)
    assert s["auc"] == 1.0 and s["support"] == pytest.approx(0.003)
    assert s["low_support"] is True and s["auc_in_score"] is False and s["score_basis"] == "r2"
    assert s["score"] == s["r2"] and s["score"] < 0.1
    assert CP.meaning_grade(s["score"]) == "closed_form_only"
    # a rare flag that is the leaf still counts: r2 is 1
    s1 = CP.score_concept(c, rare * 3.0 + 1.0)
    assert s1["low_support"] is True and s1["auc_in_score"] is False and s1["score"] == 1.0
    # a common binary concept on an ungated exit: r2 (the between-groups variance share), not the AUC
    half = (y > np.median(y)).astype(float)
    s2 = CP.score_concept(C.Concept("top_half", "", "binary", "domain", half), y)
    assert s2["auc"] == 1.0 and s2["auc_in_score"] is False and s2["score"] == s2["r2"] and s2["low_support"] is False
    # a gate with comparable support: the AUC counts and the concept that is the gate scores 1
    z = rng.normal(size=n)
    gate = np.maximum(z - 0.5, 0.0)                             # on for about 31% of rows
    on = (gate > 0).astype(float)
    s3 = CP.score_concept(C.Concept("gate_on", "", "binary", "domain", on), gate)
    assert s3["leaf_gated"] is True and s3["gate_share"] == pytest.approx(on.mean(), abs=1e-6)
    assert s3["auc_in_score"] is True and s3["score"] == 1.0 and s3["score_basis"].startswith("max")
    # a rare flag matched to a rare gate: comparable, but both are below the 1% floor, so r2 only
    # (the floor is the floor); with a lower floor the AUC counts
    rare_gate = np.maximum(z - 2.6, 0.0)                        # on for about 0.5% of rows
    rare_on = C.Concept("rare_on", "", "binary", "domain", (rare_gate > 0).astype(float))
    s4 = CP.score_concept(rare_on, rare_gate)
    assert s4["auc_in_score"] is False and s4["low_support"] is True and s4["score"] == s4["r2"]
    s4b = CP.score_concept(rare_on, rare_gate, min_support=0.001)
    assert s4b["auc_in_score"] is True and s4b["low_support"] is False and s4b["score"] == 1.0
    # the same rare flag against a common gate: not comparable, r2 only
    s5 = CP.score_concept(C.Concept("rare_on", "", "binary", "domain", (rare_gate > 0).astype(float)), gate)
    assert s5["auc_in_score"] is False and s5["low_support"] is True
    assert CP.comparable_support(0.3, 0.1) is False and CP.comparable_support(0.3, 0.2) is True
    assert CP.comparable_support(0.005, 0.005) is False and CP.comparable_support(0.005, 0.005, min_support=0.001)
    # the floor is configurable
    assert CP.score_concept(c, y, min_support=0.001)["low_support"] is False
    # probe_score honours a score dict's own auc_in_score and defaults to counting the AUC for old blocks
    assert CP.probe_score({"r2": 0.1, "auc": 0.95, "auc_in_score": False}) == 0.1
    assert CP.probe_score({"r2": 0.1, "auc": 0.95}) == 0.9


def test_sweep_drops_low_support_and_records_conditions(world):
    r = CP.probe_leaf(world["structure"], world["X"], world["raw"], world["names"], "synthetic", world["gate"],
                      entry_inputs=["dose"])
    assert r["probe_version"] == 2 and r["conditions"] == {"max_terms": 3, "min_support": 0.01, "comparable_ratio": 0.5}
    assert r["low_support_concepts"] == [] and r["low_support"] is False
    assert r["simpler_than_leaf"] is True and r["concept_complexity"] == 1
    assert r["meaning"]["probe"]["support"] is None or 0 <= r["meaning"]["probe"]["support"] <= 0.5
    assert r["grade"] == "meaningful"
