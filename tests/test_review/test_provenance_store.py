"""The review kit's C badge on the real provenance (LogProvenance): stored queries are rerun on the
model and compared; draft slots reproduce through recheck's claims (no database)."""

import copy

import pytest

from explaneat.analysis.provenance import flatten_values, render
from explaneat.review.provenance import LogProvenance, compare_values, default_provenance
from explaneat.review.sentences import extract
from explaneat.review.status import statuses
from explaneat.verify import probe as P
from explaneat.verify.model import VerifyModel
from tests.test_verify.test_verify_tools import GENOME, base_data


@pytest.fixture(scope="module")
def vm():
    return VerifyModel(base_data())


def _entry(qid, tool, arguments, result, query=None):
    return {"id": qid, "tool": tool, "arguments": arguments, "query": query or {"scope": "whole model"},
            "split": "train", "n": 300, "n_split": 300, "sampled": False, "op_seq": 5,
            "values": flatten_values(result)}


@pytest.fixture(scope="module")
def log(vm):
    where = "grp == '1'"
    sub = P.subgroup(vm, where, "train")
    prb = P.probe(vm, sets=["dose=8"])
    return {"q_sub0000001": _entry("q_sub0000001", "query_subgroup", {"where": where, "split": "train"}, sub),
            "q_prb0000001": _entry("q_prb0000001", "query_probe", {"sets": '["dose=8"]', "hold": "typical"}, prb),
            "q_viz0000001": _entry("q_viz0000001", "get_viz_summary", {"annotation_id": "gate"}, {"mean": 0.5})}


def _ops(template, lookup):
    text, used, errs = render(template, lookup)
    assert not errs, errs
    return [{"type": "annotate", "seq": 9, "result": {"annotation_id": "ann_9"},
             "params": {"name": "gate", "hypothesis": text, "hypothesis_template": template, "provenance": used}}]


def _badges(template, log, vm, layer=None, report=None):
    ops = _ops(template, log.get)
    prov = LogProvenance(GENOME, ops, layer, report, lookup=log.get, vm=vm)
    sents = extract(None, layer, ops)
    st = statuses(sents, report, set(), layer, prov)
    return sents, st, prov


def test_compare_values():
    assert compare_values(0.1234, 0.1234) == "match"
    assert compare_values(0.1234, 0.1234 + 1e-8) == "drift"
    assert compare_values(0.1234, 0.1241, ".2f") == "match-rounded"
    assert compare_values(0.1234, 0.18, ".2f") == "mismatch"
    assert compare_values(35, 35) == "match"


def test_sentence_with_reproducing_references_is_c(log, vm):
    t = ("For the {{q:q_sub0000001.subgroup.n|d}} patients in group B the mean predicted risk is "
         "{{q:q_sub0000001.subgroup.mean_predicted_risk|.2f}}.")
    sents, st, prov = _badges(t, log, vm)
    s = sents[0]
    assert s.source == t and "{{" not in s.text
    assert st[s.sid].badge == "C", st[s.sid].reasons
    r = prov.resolve("q:q_sub0000001.subgroup.n|d")
    assert r["reproduce"] == "match" and r["split"] == "train" and r["value"] == r["rerun"]


def test_a_tampered_stored_value_fails(log, vm):
    bad = copy.deepcopy(log)
    bad["q_sub0000001"]["values"]["subgroup.mean_predicted_risk"] += 0.05
    sents, st, _ = _badges("Mean predicted risk {{q:q_sub0000001.subgroup.mean_predicted_risk|.2f}}.", bad, vm)
    assert st[sents[0].sid].badge == "F"
    assert any("does not reproduce" in r for r in st[sents[0].sid].reasons)


def test_a_bare_number_or_an_unrerunnable_tool_is_not_c(log, vm):
    sents, st, _ = _badges("In {{q:q_sub0000001.subgroup.n|d}} patients the risk is 0.31.", log, vm)
    assert st[sents[0].sid].badge == "U"
    sents, st, _ = _badges("The mean exit value is {{q:q_viz0000001.mean|.2f}}.", log, vm)
    assert st[sents[0].sid].badge == "U"
    assert any("unavailable" in r for r in st[sents[0].sid].reasons)


def test_unknown_reference_is_u_and_number_words_need_references(log, vm):
    prov = LogProvenance(GENOME, [], lookup=log.get, vm=vm)
    assert prov.resolve("q:q_missing0001.subgroup.n") is None
    sents, st, _ = _badges("Zero of the {{q:q_sub0000001.subgroup.n|d}} patients in group B are unaffected.",
                           log, vm)
    assert st[sents[0].sid].badge == "C"


def test_direct_wording_on_a_whole_model_query_is_a_scope_mismatch(log, vm):
    sents, st, _ = _badges("The direct effect of this dose is a risk of {{q:q_prb0000001.rows[0].risk|.2f}}.",
                           log, vm)
    assert st[sents[0].sid].badge == "F"
    assert any("scope" in r for r in st[sents[0].sid].reasons)


def test_draft_slots_reproduce_through_recheck_claims(log, vm):
    layer = {"claims": [{"id": "C01", "key": "k1", "verified": True, "risk": {"from": 0.1, "to": 0.4}}],
             "drafted": {"lead": [{"text": "It rises from 10% to 40%.", "claims": ["C01"],
                                   "template": "It rises from {{C01.risk.from|pct}} to {{C01.risk.to|pct}}."}]}}
    ok = {"claims": {"stated": 1, "reproduced": 1}, "findings": []}
    sents = [s for s in extract(None, layer, []) if s.origin == "drafted"]
    prov = LogProvenance(GENOME, [], layer, ok, lookup=log.get, vm=vm)
    assert statuses(sents, ok, set(), layer, prov)[sents[0].sid].badge == "C"
    bad = {"claims": {"stated": 1, "reproduced": 0}, "findings": [
        {"check": "claim", "location": "claims.C01", "severity": "mismatch",
         "diffs": [{"path": "risk.to", "stated": 0.4, "recomputed": 0.3}]}]}
    prov = LogProvenance(GENOME, [], layer, bad, lookup=log.get, vm=vm)
    assert statuses(sents, bad, set(), layer, prov)[sents[0].sid].badge == "F"


def test_default_store_only_for_text_with_references(log):
    assert type(default_provenance(GENOME, [{"params": {"hypothesis": "plain 0.3"}}], None, None)).__name__ \
        == "NoProvenance"
    ops = _ops("n = {{q:q_sub0000001.subgroup.n|d}}", log.get)
    assert type(default_provenance(GENOME, ops, None, None)).__name__ == "LogProvenance"
