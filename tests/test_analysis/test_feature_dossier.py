"""Feature dossier on a toy network with known thresholds and two shape families.

Inputs (raw units): x0 in [85, 115] (scaler mean 100, scale 10), x1 noise, x2 a 0/1 flag.
    u1 = relu(z0)                      -> threshold x0* = 100 for everyone (curve starts rising)
    u2 = relu(z0 + 5*z2 - 5.5)         -> threshold x0* = 155 - 50*x2: 105 when x2 = 1, out of range when x2 = 0
    logit = 2*u1 - 4*u2 - 1
Family "flat -> rises" for x2 = 0; "flat -> rises -> falls" for x2 = 1.
x0 enters u1 through copy -1_a and u2 through copy -1_b; the annotation "dip" holds u2.
"""

import json

import numpy as np
import pytest

from explaneat.analysis.feature_dossier import DossierConfig, build_dossier
from explaneat.analysis.plain_language import PlainModel
from explaneat.analysis.plain_language_codebooks import generic_codebook
from explaneat.core.genome_network import NetworkConnection, NetworkNode, NetworkStructure, NodeType


def _toy(n=80, seed=1):
    rng = np.random.default_rng(seed)
    raw = np.column_stack([rng.uniform(85, 115, n), rng.normal(0, 1, n), (np.arange(n) % 2).astype(float)])
    mean = np.array([100.0, 0.0, 0.0])
    scale = np.array([10.0, 1.0, 1.0])
    X = (raw - mean) / scale
    nodes = [NetworkNode("-1_a", NodeType.INPUT), NetworkNode("-1_b", NodeType.INPUT),
             NetworkNode("-2", NodeType.INPUT), NetworkNode("-3", NodeType.INPUT),
             NetworkNode("10", NodeType.HIDDEN, bias=0.0, activation="relu"),
             NetworkNode("11", NodeType.HIDDEN, bias=-5.5, activation="relu"),
             NetworkNode("0", NodeType.OUTPUT, bias=-1.0, activation="sigmoid")]
    conns = [NetworkConnection("-1_a", "10", 1.0, True), NetworkConnection("-1_b", "11", 1.0, True),
             NetworkConnection("-3", "11", 5.0, True), NetworkConnection("10", "0", 2.0, True),
             NetworkConnection("11", "0", -4.0, True), NetworkConnection("-2", "0", 0.0, True)]
    s = NetworkStructure(nodes, conns, input_node_ids=["-1_a", "-1_b", "-2", "-3"], output_node_ids=["0"])
    ops = [{"type": "annotate", "seq": 1,
            "params": {"name": "dip", "hypothesis": "Lowers the score above 105 when x2 is on. More.",
                       "entry_nodes": ["-1_b", "-3"], "exit_nodes": ["11"],
                       "subgraph_nodes": ["-1_b", "-3", "11"],
                       "subgraph_connections": [["-1_b", "11"], ["-3", "11"]]}}]
    y = (rng.uniform(size=n) < 0.4).astype(float)
    names = ["x0", "x1", "x2"]
    m = PlainModel(s, ops, X, y, X[:20], y[:20], names, mean, scale, generic_codebook(names))
    return m, raw


@pytest.fixture(scope="module")
def dossier():
    m, raw = _toy()
    return build_dossier(m, "x0", cfg=DossierConfig(grid_points=121, min_family=3)), m, raw


def test_units_and_closed_form_thresholds(dossier):
    d, m, raw = dossier
    units = {u["unit"]: u for u in d["units"]}
    assert set(units) == {"10", "11"}
    f10 = units["10"]["patterns"][0]["formula"]
    assert f10["const"] == pytest.approx(100.0) and f10["terms"] == {}
    f11 = units["11"]["patterns"][0]["formula"]
    assert f11["const"] == pytest.approx(155.0)
    assert f11["terms"] == {"x2": pytest.approx(-50.0)}
    assert units["11"]["part"] == "dip"
    assert units["11"]["parts"][0]["hypothesis_first_sentence"].startswith("Lowers the score")
    assert units["10"]["n_patients_switching_in_range"] == len(raw)
    assert units["11"]["n_patients_switching_in_range"] == int((raw[:, 2] == 1).sum())
    assert d["checks"]["kinks_unattributed"] == 0
    assert d["checks"]["threshold_formula_max_abs_err"] < 1e-8
    assert d["checks"]["structure_network_median_curve_max_abs_err"] < 1e-9


def test_crossings_located_exactly(dossier):
    d, m, raw = dossier
    bends = {b["bend"]: b for f in d["families"] for b in f["bends"]}
    assert bends["10+"]["x_median"] == pytest.approx(100.0, abs=1e-6)
    assert bends["11+"]["x_median"] == pytest.approx(105.0, abs=1e-6)
    assert bends["11+"]["dslope_median"] == pytest.approx(-0.4, rel=1e-6)   # -4 * 1/10 per raw unit
    assert bends["10+"]["dslope_median"] == pytest.approx(0.2, rel=1e-6)


def test_two_families_with_conditions(dossier):
    d, m, raw = dossier
    fams = {tuple(f["shape"]): f for f in d["families"]}
    assert set(fams) == {("flat", "up"), ("flat", "up", "down")}
    assert fams[("flat", "up")]["n"] == int((raw[:, 2] == 0).sum())
    dip = fams[("flat", "up", "down")]
    assert dip["n"] == int((raw[:, 2] == 1).sum())
    assert dip["conditions"]["rules"][0]["if"] == ["x2 > 0.50"]
    assert dip["conditions"]["rules"][0]["precision"] == 1.0
    assert d["family_tree"]["accuracy_train"] == 1.0
    assert d["kmeans_check"]["adjusted_rand_vs_families"] == pytest.approx(1.0)
    p = dip["pieces"]
    assert [x["word"] for x in p] == ["flat", "up", "down"]
    assert list(p[0]["ends_with"]) == ["10+"] and list(p[1]["ends_with"]) == ["11+"]
    # 105 -> the observed maximum at 0.2 - 0.4 = -0.2 log-odds per raw unit
    assert p[2]["logodds_change_median"] == pytest.approx(-0.2 * (raw[:, 0].max() - 105), rel=1e-2)
    # support counts: patients with their own x0 in the bend region
    b = next(b for b in dip["bends"] if b["bend"] == "11+")
    assert b["family_patients_with_value_in_region"] == int(
        np.sum((raw[:, 2] == 1) & (raw[:, 0] >= b["region"][0]) & (raw[:, 0] <= b["region"][1])))
    assert d["families"][0]["n_test"] is not None


def test_scope_varies_only_the_annotations_copy():
    m, raw = _toy()
    d = build_dossier(m, "x0", annotation="dip", cfg=DossierConfig(grid_points=121, min_family=3))
    assert d["input"]["copies_varied"] == ["-1_b"]
    assert [u["unit"] for u in d["units"]] == ["11"]


def test_write_dossier(tmp_path, dossier):
    from explaneat.analysis.feature_dossier_report import write_dossier
    m, raw = _toy()
    d = build_dossier(m, "x0", cfg=DossierConfig(grid_points=121, min_family=3))
    files = write_dossier(d, tmp_path, notes="## Reading\n\nhello")
    j = json.loads((tmp_path / "dossier.json").read_text())
    assert j["kind"] == "feature_dossier" and "_arrays" not in j
    md = (tmp_path / "report.md").read_text()
    assert "## Reading" in md and "unit 11 (dip) switches on" in md
    for k in ("ice_families", "families", "median_curve", "unit_thresholds", "outcome_bands"):
        assert (tmp_path / "figures" / f"{k}.png").exists(), k
