"""recheck uses the provenance of referenced numbers, and produces the record's recheck_summary."""

from explaneat.verify.recheck import provenance_checks, summary_block

ENTRY = {"id": "q_1", "tool": "query_subgroup", "split": "train", "n": 242, "sampled": False, "op_seq": 5,
         "values": {"subgroup.mean_predicted_risk": 0.1813, "subgroup.n": 35}}


def ops(text, template, ids, extra=()):
    return list(extra) + [{"type": "annotate", "params": {"name": "p", "hypothesis": text,
                                                          "hypothesis_template": template, "provenance": ids}}]


def test_rendered_text_matches_its_template():
    T = "mean risk {{q:q_1.subgroup.mean_predicted_risk|.2f}} for {{q:q_1.subgroup.n}} patients"
    f, st = provenance_checks(ops("mean risk 0.18 for 35 patients", T, ["q_1"]), "g", {"q_1": ENTRY}.get)
    assert f == [] and st == {"templated_texts": 1, "queries": 1}
    f, _ = provenance_checks(ops("mean risk 0.13 for 35 patients", T, ["q_1"]), "g", {"q_1": ENTRY}.get)
    assert [x["severity"] for x in f] == ["mismatch"]          # hand-edited after rendering (E1)


def test_sampled_missing_and_stale_queries():
    T = "{{q:q_1.subgroup.n}} patients"
    sampled = {**ENTRY, "sampled": True, "split": "both"}
    f, _ = provenance_checks(ops("35 patients", T, ["q_1"]), "g", {"q_1": sampled}.get)
    assert any(x["severity"] == "needs_review" and "sample" in x["note"] for x in f)
    f, _ = provenance_checks(ops("35 patients", T, ["q_1"]), "g", {}.get)
    assert any(x["severity"] == "mismatch" for x in f)
    later = [{"type": "split_node"}] * 7                       # a structural change after the query
    f, _ = provenance_checks(ops("35 patients", T, ["q_1"], later), "g", {"q_1": ENTRY}.get)
    assert any("changed the structure" in x["note"] for x in f)


def test_summary_block_gates_completion():
    rep = {"findings": [{"check": "freetext_group_count", "location": "operations.a.hypothesis", "severity": "mismatch",
                         "stated": 27, "recomputed": 26},
                        {"check": "freetext_readout", "location": "x", "severity": "needs_review", "note": "n"}],
           "summary": {"mismatch": 1, "needs_review": 1, "warn": 0, "unchecked_freetext": 4}}
    sb = summary_block(rep)
    assert not sb["complete_ok"] and sb["mismatch"] == 1 and len(sb["needs_review_items"]) == 1
    assert sb["unchecked_freetext_sentences"] == 4
    rep["findings"], rep["summary"]["mismatch"] = rep["findings"][1:], 0
    assert summary_block(rep)["complete_ok"]
