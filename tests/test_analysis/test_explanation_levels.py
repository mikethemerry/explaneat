"""The decomposed level: a deep leaf is not penalised, a swallowed wide layer is flagged; the
three levels assemble from a coverage report, the checks and the meaning blocks."""

from types import SimpleNamespace

import pytest

from explaneat.analysis.coverage_report import coverage_report, decomposition_report, part_shape
from explaneat.analysis.explanation_levels import (explanation_levels, leaf_meanings_from_operations,
                                                   levels_summary_row, mike_leaf_grades, normalise_grade,
                                                   understood_block, verifier_leaf_grades)
from explaneat.core.genome_network import NetworkConnection, NetworkNode, NetworkStructure, NodeType


def _n(i, t, act="relu"):
    return NetworkNode(id=str(i), type=t, bias=0.0, activation=act, response=1.0, aggregation="sum")


def _ann(name, entries, exits, nodes, conns, parent=None):
    return SimpleNamespace(name=name, entry_nodes=entries, exit_nodes=exits, subgraph_nodes=nodes,
                           subgraph_connections=[tuple(c) for c in conns], parent_annotation_id=parent)


def chain_net(k=4):
    """-1 -> 1 -> 2 -> ... -> k -> 0: a chain k deep."""
    nodes = [_n(-1, NodeType.INPUT, "identity")] + [_n(i, NodeType.HIDDEN) for i in range(1, k + 1)] \
        + [_n(0, NodeType.OUTPUT, "sigmoid")]
    conns = [NetworkConnection(from_node="-1", to_node="1", weight=1.0, enabled=True)]
    conns += [NetworkConnection(from_node=str(i), to_node=str(i + 1), weight=1.0, enabled=True) for i in range(1, k)]
    conns.append(NetworkConnection(from_node=str(k), to_node="0", weight=1.0, enabled=True))
    return NetworkStructure(nodes=nodes, connections=conns, input_node_ids=["-1"], output_node_ids=["0"])


def wide_net(w=8):
    """-1 -> each of 1..w -> 0: one layer w wide."""
    nodes = [_n(-1, NodeType.INPUT, "identity")] + [_n(i, NodeType.HIDDEN) for i in range(1, w + 1)] \
        + [_n(0, NodeType.OUTPUT, "sigmoid")]
    conns = [NetworkConnection(from_node="-1", to_node=str(i), weight=1.0, enabled=True) for i in range(1, w + 1)]
    conns += [NetworkConnection(from_node=str(i), to_node="0", weight=1.0, enabled=True) for i in range(1, w + 1)]
    return NetworkStructure(nodes=nodes, connections=conns, input_node_ids=["-1"], output_node_ids=["0"])


def test_deep_leaf_is_not_penalised():
    k = 4
    s = chain_net(k)
    hidden = [str(i) for i in range(1, k + 1)]
    leaf = _ann("chain", ["-1"], [str(k)], ["-1"] + hidden,
                [["-1", "1"]] + [[str(i), str(i + 1)] for i in range(1, k)], parent="root")
    root = _ann("root", [str(k)], ["0"], [str(k), "0"], [[str(k), "0"]])
    shape = part_shape(leaf, set(hidden))
    assert shape == {"units": 4, "width": 1, "depth": 4, "unit_ids": hidden}
    d = decomposition_report(s, [leaf, root])
    assert d["granularity"] == 1.0 and d["warnings"] == []
    assert d["largest_part"]["name"] == "chain" and d["largest_part"]["width"] == 1 and d["largest_part"]["depth"] == 4
    assert d["max_depth"] == 4 and d["max_width"] == 1


def test_swallowed_layer_is_flagged():
    w = 8
    s = wide_net(w)
    hidden = [str(i) for i in range(1, w + 1)]
    leaf = _ann("layer", ["-1"], hidden, ["-1"] + hidden, [["-1", h] for h in hidden], parent="root")
    root = _ann("root", hidden, ["0"], hidden + ["0"], [[h, "0"] for h in hidden])
    d = decomposition_report(s, [leaf, root])
    assert d["granularity"] == 0.0
    assert d["largest_part"] == {"name": "layer", "hidden_units": 8, "width": 8, "depth": 1, "leaf": True,
                                 "param_share": d["largest_part"]["param_share"]}
    assert any(wn.startswith("coarse decomposition") for wn in d["warnings"])
    # the same units split into two leaves of 4 are within the small width
    a, b = hidden[:4], hidden[4:]
    la = _ann("a", ["-1"], a, ["-1"] + a, [["-1", h] for h in a], parent="root")
    lb = _ann("b", ["-1"], b, ["-1"] + b, [["-1", h] for h in b], parent="root")
    d2 = decomposition_report(s, [la, lb, root])
    assert d2["granularity"] == 1.0 and d2["warnings"] == []
    # split copies count once: 1_a and 1_b are the same unit
    lc = _ann("c", ["-1"], ["1_a", "1_b"], ["-1", "1_a", "1_b"], [["-1", "1_a"], ["-1", "1_b"]], parent="root")
    assert part_shape(lc, set(hidden))["units"] == 1


def test_units_only_as_entries_warning():
    s = wide_net(2)
    root = _ann("root", ["1", "2"], ["0"], ["1", "2", "0"], [["1", "0"], ["2", "0"]])
    d = decomposition_report(s, [root])
    assert d["units_only_as_entries"] == 2 and len(d["warnings"]) == 1
    rep = coverage_report(s, [root])
    assert rep["decomposition"]["units_only_as_entries"] == 2
    assert coverage_report(s, [])["decomposition"]["granularity"] is None


def test_levels_assemble():
    s = wide_net(2)
    hidden = ["1", "2"]
    la = _ann("a", ["-1"], ["1"], ["-1", "1"], [["-1", "1"]], parent="root")
    lb = _ann("b", ["-1"], ["2"], ["-1", "2"], [["-1", "2"]], parent="root")
    root = _ann("root", hidden, ["0"], hidden + ["0"], [[h, "0"] for h in hidden])
    rep = coverage_report(s, [la, lb, root])
    checks = {"function_preservation_max_abs_err": 0.0, "closed_form_max_abs_err": 0.01}
    meanings = {"a": {"grade": "meaningful", "concept": "dose", "probe": {"r2": 0.9, "auc": None, "score": 0.9}},
                "b": {"grade": "meaningful", "concept": "noise", "probe": {"r2": 0.1, "auc": 0.55, "score": 0.1}}}
    lv = explanation_levels(rep, checks, meanings)
    assert lv["faithful"]["passed"] is True
    assert lv["decomposed"]["n_leaves"] == 2 and lv["decomposed"]["passed"] is True
    # the grade is re-derived from the probe: b's typed "meaningful" does not count
    assert lv["understood"]["grades"] == {"a": "meaningful", "b": "closed_form_only"}
    assert lv["understood"]["understood_fraction"] == 0.5 and lv["understood"]["closed_form_only_fraction"] == 0.5
    row = levels_summary_row(lv)
    assert row == {"faithful_max_err": 0.0, "n_leaves": 2, "granularity": 1.0, "understood_fraction": 0.5,
                   "closed_form_only_fraction": 0.5, "understood_fraction_verifier": None,
                   "understood_fraction_mike": None}
    assert lv["understood"]["understood_probe"] == 0.5 and lv["understood"]["verifier_disagreements"] is None
    none = explanation_levels(rep, None, None)
    assert none["faithful"]["passed"] is None and none["understood"]["unnamed_fraction"] == 1.0


def test_meanings_from_operations():
    ops = [{"seq": 1, "type": "annotate", "params": {"name": "a", "entry_nodes": ["-1"], "exit_nodes": ["1"],
                                                      "subgraph_nodes": ["-1", "1"], "subgraph_connections": [["-1", "1"]],
                                                      "meaning": {"grade": "partial", "concept": "x", "probe": {"score": 0.6}}}},
           {"seq": 2, "type": "annotate", "params": {"name": "b", "entry_nodes": ["-1"], "exit_nodes": ["2"],
                                                      "subgraph_nodes": ["-1", "2"], "subgraph_connections": [["-1", "2"]]}},
           {"seq": 3, "type": "annotate", "params": {"name": "root", "entry_nodes": ["1", "2"], "exit_nodes": ["0"],
                                                      "subgraph_nodes": ["1", "2", "0"], "subgraph_connections": [],
                                                      "child_annotation_ids": ["a", "b"]}}]
    m = leaf_meanings_from_operations(ops)
    assert set(m) == {"a", "b"} and m["b"] is None and m["a"]["concept"] == "x"
    u = understood_block(["a", "b"], m)
    assert u["counts"] == {"meaningful": 0, "partial": 1, "closed_form_only": 0, "restatement": 0, "unnamed": 1}


def test_verifier_grades_merge():
    """The blind verifier's leaf grades (keyed by leaf name or annotation id, spelled its way)
    sit beside the probe's; the disagreement count covers leaves graded by both."""
    verdict = {"leaf_meaning": {"a": {"grade": "meaningful", "why": "..."},
                                "ann_2": {"grade": "closed-form-only"},
                                "ann_3": "partial",
                                "no_such_leaf": {"grade": "meaningful"},
                                "d": {"grade": "?"}}}
    vg = verifier_leaf_grades(verdict, ["a", "b", "c", "d"], {"a": "ann_1", "b": "ann_2", "c": "ann_3", "d": "ann_4"})
    assert vg == {"a": "meaningful", "b": "closed_form_only", "c": "partial"}
    meanings = {"a": {"concept": "x", "probe": {"score": 0.9}}, "b": {"concept": "y", "probe": {"score": 0.95}},
                "c": None, "d": {"concept": "z", "probe": {"score": 0.1}}}
    u = understood_block(["a", "b", "c", "d"], meanings, vg)
    assert u["understood_probe"] == 0.5 and u["understood_fraction"] == 0.5
    assert u["understood_verifier"] == round(1 / 3, 4) and u["understood_fraction_verifier"] == u["understood_verifier"]
    assert u["verifier_graded"] == 3 and u["verifier_compared"] == 2      # c is unnamed to the probe
    assert u["verifier_disagreements"] == 1                              # b: probe meaningful, verifier cfo
    assert u["verifier_grades"] == {"a": "meaningful", "b": "closed_form_only", "c": "partial", "d": None}
    # a restatement block counts as its own grade, and the summary row carries the verifier column last
    meanings["b"] = {"concept": "y", "simpler_than_leaf": False, "probe": {"score": 0.95}}
    u2 = understood_block(["a", "b"], meanings, vg)
    assert u2["counts"]["restatement"] == 1 and u2["restatement_fraction"] == 0.5 and u2["verifier_disagreements"] == 1
    assert normalise_grade("Closed form only") == "closed_form_only" and normalise_grade("odd") is None


def test_mike_leaf_verdicts_merge(tmp_path, monkeypatch):
    """Mike's leaf verdicts (the review store) give understood_mike, the measured level, beside the
    probe and verifier screens; None until a leaf has a verdict; over all leaves."""
    from explaneat.review import verdicts as vd
    monkeypatch.setenv("EXPLANEAT_REVIEW_DIR", str(tmp_path))
    meanings = {"a": {"concept": "x", "probe": {"score": 0.9}}, "b": {"concept": "y", "probe": {"score": 0.1}},
                "c": None}
    u = understood_block(["a", "b", "c"], meanings)
    assert u["understood_mike"] is None and u["mike_graded"] == 0 and u["mike_grades"] is None
    assert mike_leaf_grades("heart-s5", ["a", "b", "c"], {"a": "ann_1", "b": "ann_2", "c": "ann_3"}) == {}
    u = understood_block(["a", "b", "c"], meanings, None, {})
    assert u["understood_mike"] is None and u["mike_grades"] == {"a": None, "b": None, "c": None}
    ctx = {"records_dir": "experiments/10seed/explanations_v4"}
    vd.record_leaf("heart-s5", {"id": "ann_2", "name": "b", "hypothesis": "h"}, "understood", context=ctx)
    vd.record_leaf("heart-s5", {"id": "ann_1", "name": "a", "hypothesis": "h"}, "not", context=ctx)
    vd.record_leaf("heart-s5", {"id": "ann_1", "name": "a", "hypothesis": "h"}, "partial", context=ctx)   # supersedes
    mg = mike_leaf_grades("heart-s5", ["a", "b", "c"], {"a": "ann_1", "b": "ann_2", "c": "ann_3"},
                          records_dir="experiments/10seed/explanations_v4")
    assert mg == {"a": "partial", "b": "understood"}
    assert mike_leaf_grades("heart-s5", ["a", "b", "c"], {"a": "ann_1", "b": "ann_2", "c": "ann_3"},
                            records_dir="experiments/10seed/explanations") == {}
    u = understood_block(["a", "b", "c"], meanings, None, mg)
    assert u["understood_mike"] == round(1 / 3, 4) and u["understood_fraction_mike"] == u["understood_mike"]
    assert u["mike_graded"] == 2 and u["mike_counts"] == {"understood": 1, "partial": 1, "not": 0}
    assert u["understood_probe"] == round(1 / 3, 4)          # the screen is untouched
    lv = explanation_levels({"leaf_count": 3, "decomposition": {}}, None, meanings, ["a", "b", "c"], mike_grades=mg)
    assert levels_summary_row(lv)["understood_fraction_mike"] == round(1 / 3, 4)
    assert "understood_mike" in lv["understood"]["test"]
