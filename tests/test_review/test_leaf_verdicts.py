"""Leaf verdicts (understood / partial / not) in the review store: appended to the same
``<label>.verdicts.jsonl`` as the sentence verdicts, kept apart from them by ``target.kind``, read
back by the levels and the sign-off summary. No database."""

import json

import pytest

from explaneat.review import signoff as so
from explaneat.review import verdicts as vd
from explaneat.review.sentences import extract


def ops():
    return [
        {"seq": 3, "type": "annotate", "result": {"annotation_id": "ann_3"},
         "params": {"name": "age_gate", "hypothesis": "Gate on age above 50. It adds 0.4 to the log-odds.",
                    "entry_nodes": ["-1"], "exit_nodes": ["7"], "subgraph_nodes": ["-1", "7"],
                    "subgraph_connections": [["-1", "7"]]}},
        {"seq": 4, "type": "annotate", "result": {"annotation_id": "ann_4"},
         "params": {"name": "linear_terms", "hypothesis": "Additive terms.", "entry_nodes": ["-2"],
                    "exit_nodes": ["8"], "subgraph_nodes": ["-2", "8"], "subgraph_connections": [["-2", "8"]]}},
        {"seq": 5, "type": "annotate", "result": {"annotation_id": "ann_5"},
         "params": {"name": "root", "hypothesis": "Sum of the parts.", "entry_nodes": ["7", "8"], "exit_nodes": ["0"],
                    "subgraph_nodes": ["7", "8", "0"], "subgraph_connections": [],
                    "child_annotation_ids": ["ann_3", "ann_4"]}},
    ]


def test_leaves_of_operations_are_the_childless_annotations():
    leaves = vd.leaves_of(ops())
    assert [lf["name"] for lf in leaves] == ["age_gate", "linear_terms"]
    assert leaves[0]["id"] == "ann_3" and leaves[0]["hypothesis"].startswith("Gate on age")
    # a composition named as a parent (and not by child ids) is not a leaf either
    o = ops()
    o[2]["params"].pop("child_annotation_ids")
    o[0]["params"]["parent_annotation_id"] = "ann_5"
    o[1]["params"]["parent_annotation_id"] = "ann_5"
    assert [lf["name"] for lf in vd.leaves_of(o)] == ["age_gate", "linear_terms"]


def test_find_leaf_by_name_or_id():
    leaves = vd.leaves_of(ops())
    assert vd.find_leaf(leaves, "age_gate")["id"] == "ann_3"
    assert vd.find_leaf(leaves, "ann_4")["name"] == "linear_terms"
    with pytest.raises(KeyError):
        vd.find_leaf(leaves, "nothing")


def test_record_leaf_is_append_only_and_kept_apart_from_sentence_verdicts(tmp_path):
    leaves = vd.leaves_of(ops())
    sents = extract(None, None, ops())
    ctx = {"records_dir": "experiments/10seed/explanations_v4"}
    vd.record("heart-s5", sents[0], "confirmed", context=ctx, directory=tmp_path, now="2026-11-13T10:00:00+13:00")
    l1 = vd.record_leaf("heart-s5", leaves[0], "partial", note="gate ok, threshold unclear", context=ctx,
                        directory=tmp_path, now="2026-11-13T10:01:00+13:00")
    l2 = vd.record_leaf("heart-s5", leaves[0], "understood", note="checked the threshold", context=ctx,
                        directory=tmp_path, now="2026-11-13T10:02:00+13:00")
    lines = vd.load("heart-s5", tmp_path)
    assert [ln["id"] for ln in lines] == ["v-000001", "v-000002", "v-000003"]
    assert l1["target"] == {"kind": "leaf", "annotation_id": "ann_3", "name": "age_gate",
                            "hypothesis_hash": l1["target"]["hypothesis_hash"]}
    assert l2["supersedes"] == "v-000002" and l1["supersedes"] is None
    # the file is three lines, nothing rewritten
    raw = (tmp_path / "heart-s5.verdicts.jsonl").read_text().splitlines()
    assert len(raw) == 3 and json.loads(raw[1])["verdict"] == "partial"
    # sentence readers never see the leaf lines; leaf readers never see the sentences
    assert set(vd.latest(lines)) == {sents[0].sid}
    assert vd.latest_leaf(lines) == {"ann_3": l2}
    assert len(vd.sentence_lines(lines)) == 1 and len(vd.leaf_lines(lines)) == 2
    with pytest.raises(ValueError):
        vd.record_leaf("heart-s5", leaves[1], "maybe", directory=tmp_path)
    # another record set in the same store is refused, as for sentence verdicts
    with pytest.raises(ValueError):
        vd.record_leaf("heart-s5", leaves[1], "not", context={"records_dir": "experiments/10seed/explanations"},
                       directory=tmp_path)


def test_leaf_grades_and_summary(tmp_path):
    leaves = vd.leaves_of(ops())
    ctx = {"records_dir": "experiments/10seed/explanations_v4"}
    assert vd.leaf_grades("heart-s5", leaves, tmp_path) == {}
    assert vd.leaf_summary(leaves, {}) == {"leaves": 2, "graded": 0, "counts": {"understood": 0, "partial": 0, "not": 0},
                                          "understood_mike": None}
    vd.record_leaf("heart-s5", leaves[0], "understood", context=ctx, directory=tmp_path)
    g = vd.leaf_grades("heart-s5", leaves, tmp_path, records_dir="experiments/10seed/explanations_v4")
    assert g == {"age_gate": "understood"}
    assert vd.leaf_grades("heart-s5", leaves, tmp_path, records_dir="experiments/10seed/explanations") == {}
    s = vd.leaf_summary(leaves, g)
    assert s["graded"] == 1 and s["understood_mike"] == 0.5          # over all leaves, not the graded ones
    vd.record_leaf("heart-s5", leaves[1], "not", context=ctx, directory=tmp_path)
    s = vd.leaf_summary(leaves, vd.leaf_grades("heart-s5", leaves, tmp_path))
    assert s == {"leaves": 2, "graded": 2, "counts": {"understood": 1, "partial": 0, "not": 1}, "understood_mike": 0.5}


def test_signoff_reports_leaf_verdicts_without_requiring_them(tmp_path):
    leaves = vd.leaves_of(ops())
    sents = extract(None, None, ops())
    lines = [vd.record_leaf("heart-s5", leaves[0], "understood", directory=tmp_path)]
    rep = {"target_hash": "h", "summary": {}, "findings": []}
    res = so.check("heart-s5", [], sents, lines, rep, "h", leaves=leaves)
    assert res["passed"] and res["counts"] == {"confirmed": 0, "wrong": 0, "unsure": 0}
    assert res["leaf_verdicts"]["graded"] == 1 and res["leaf_verdicts"]["understood_mike"] == 0.5
    assert res["leaf_verdicts"]["grades"] == {"age_gate": "understood", "linear_terms": None}
    strict = so.check("heart-s5", [], sents, lines, rep, "h", leaves=leaves, require_leaves=True)
    assert not strict["passed"] and any("linear_terms" in p for p in strict["problems"])
    assert so.signoff_block(res)["leaf_verdicts"] == {"leaves": 2, "graded": 1,
                                                      "counts": {"understood": 1, "partial": 0, "not": 0},
                                                      "understood_mike": 0.5}
    # without leaves the block is None and the old result shape holds
    assert so.check("heart-s5", [], sents, lines, rep, "h")["leaf_verdicts"] is None
