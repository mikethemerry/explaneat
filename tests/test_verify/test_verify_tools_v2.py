"""verify tools v2: free-text checks, prose directions and rounding, multi-part / node / edge
ablation, probe --nodes / --claim, subgroup "acts on", linearise, enumerate, the claims cache
and the blindness rule (no DB operations loaded when the target has its own)."""

import copy
import json

import numpy as np
import pytest
from click.testing import CliRunner

from explaneat.verify import freetext as FT, probe as P, recheck as R
from explaneat.verify.model import BaseData, VerifyModel
from tests.test_plain_language import helpers as H
from tests.test_verify.test_verify_tools import GENOME, base_data, record_for, target, write_item


@pytest.fixture(scope="module")
def base():
    return base_data()


@pytest.fixture(scope="module")
def vm(base):
    return VerifyModel(base)


@pytest.fixture(scope="module")
def layer(vm):
    from explaneat.analysis.plain_language import build_layer
    return build_layer(vm.m, {**H.META, "genome_id": GENOME})


def gate_part(vm):
    return next(p for p in vm.parts if p.name == "gate")


def with_hypothesis(vm, text, name="gate"):
    ops = copy.deepcopy(vm.operations)
    op = next(o for o in ops if o["params"]["name"] == name)
    op["params"]["hypothesis"] = text
    return ops


# ----------------------------------------------------------------------------- free text
def statements(vm, text, kind=None):
    r = FT.check_operations_text(vm, with_hypothesis(vm, text))
    out = [s for s in r["statements"] if s["location"] == "operations.gate.hypothesis"]
    return [s for s in out if kind is None or s["kind"] == kind], r


def test_formula_statement_checks_weights_and_bias(vm):
    ok, _ = statements(vm, "Node 5 = ReLU(1.5*dose + 2.0*grp:1 - 1.0) on the data.", "formula")
    assert ok and ok[0]["status"] == "ok" and ok[0]["node"] == "5"
    bad, _ = statements(vm, "Node 5 = ReLU(1.5*dose + 2.0*grp:1 - 2.5).", "formula")
    assert bad[0]["status"] == "mismatch" and bad[0]["rows_off"] > 0
    unk, _ = statements(vm, "5 = relu(1.5*dosage - 1.0)", "formula")
    assert unk[0]["status"] == "unchecked" and "dosage" in unk[0]["reason"]
    z, _ = statements(vm, "5 = relu(1.5 z(dose) + 2.0 z(grp:1) - 1.0)", "formula")   # implicit products
    assert z[0]["status"] == "ok"


def test_readout_active_share_n_and_rate(vm):
    m = vm.m
    act = vm.m._vals_train()["5"] > 0
    share = round(100 * float(act.mean()), 1)
    rate_a, rate_b = 100 * m.y[act].mean(), 100 * m.y[~act].mean()
    text = (f"Node 5 is read out with +1.0. 5 active {share}% of rows. Evidence (train, n=300): "
            f"event rate {rate_a:.0f}% when active vs {rate_b:.0f}% when inactive. It only ever adds risk.")
    st, r = statements(vm, text)
    kinds = {s["kind"]: s["status"] for s in st}
    assert kinds == {"readout": "ok", "active_share": "ok", "n_train": "ok", "outcome_rate": "ok", "direction": "ok"}
    st2, _ = statements(vm, f"Node 5 is read out with -0.4. 5 active {share + 9:.1f}% of rows. (train, n=250). "
                            "It only ever lowers risk.")
    assert {s["kind"]: s["status"] for s in st2} == {"readout": "mismatch", "active_share": "mismatch",
                                                    "n_train": "mismatch", "direction": "mismatch"}
    # a share inside a subgroup is not the node's overall share
    st3, _ = statements(vm, "5 active for 12% of patients with group A.", "active_share")
    assert st3[0]["status"] == "unchecked"


def test_unchecked_sentences_are_listed(vm):
    _, r = statements(vm, "The gate mirrors a clinical rule seen in 37 cohorts. Node 5 is read out with +1.0.")
    texts = [u["text"] for u in r["unchecked"] if u["location"] == "operations.gate.hypothesis"]
    assert any("37 cohorts" in t for t in texts)
    assert not any("read out" in t for t in texts)


def test_recheck_reports_freetext_findings_and_record_text(vm, layer):
    ops = with_hypothesis(vm, "Node 5 = ReLU(1.5*dose + 2.0*grp:1 - 2.5).")
    rec = record_for(vm)
    rec["summary"] = "The model is evaluated on all rows (train, n=301)."
    rep = R.recheck(target(layer, rec, ops), vm=VerifyModel(vm.base, ops), claims=False)
    checks = {(f["check"], f["location"]) for f in rep["findings"] if f["severity"] == "mismatch"}
    assert ("freetext_formula", "operations.gate.hypothesis") in checks
    assert ("freetext_n_train", "record.summary") in checks
    assert "unchecked_sentences" in rep["freetext"]


# ----------------------------------------------------------------------------- prose
def gate_direction_claim(layer):
    gid = next(p["id"] for p in layer["parts"] if p["name"] == "gate")
    return next(c for c in layer["claims"] if c["annotation"]["id"] == gid and c["type"] == "direction"
                and c["statement"].get("direction") in ("raises", "lowers"))


def prose_layer(layer, text, cites):
    L = copy.deepcopy(layer)
    gid = next(p["id"] for p in L["parts"] if p["name"] == "gate")
    L["drafted"] = {"readdown": {gid: {"hypothesis": [{"text": text, "claims": cites}], "evidence": []}}}
    return L


def test_prose_direction_against_the_cited_claim(layer):
    c = gate_direction_claim(layer)
    d = c["statement"]["direction"]
    right = "raises" if d == "raises" else "lowers"
    wrong = "lowers" if d == "raises" else "raises"
    ok, _ = R.check_prose(prose_layer(layer, f"Through it, the dose {right} the predicted risk.", [c["id"]]),
                          layer["claims"])
    assert not [f for f in ok if f["check"] == "prose_direction"]
    bad, st = R.check_prose(prose_layer(layer, f"Through it, the dose {wrong} the predicted risk.", [c["id"]]),
                            layer["claims"])
    hit = [f for f in bad if f["check"] == "prose_direction"]
    assert hit and hit[0]["severity"] == "mismatch" and hit[0]["claims"] == [c["id"]]
    assert st["direction_mismatches"] == 1
    neg, _ = R.check_prose(prose_layer(layer, f"Through it, the dose does not {wrong[:-1]} the risk.", [c["id"]]),
                           layer["claims"])
    assert not [f for f in neg if f["check"] == "prose_direction"]


def test_prose_rounding_uncited_and_wrong_numbers(layer):
    c = gate_direction_claim(layer)
    r = c["risk"]["to"]
    once = round(100 * r)
    twice = once + 1 if 100 * r > once else once - 1      # one unit off: a double-rounding slip
    L = prose_layer(layer, f"Through it, the predicted risk goes to about {twice}%.", [c["id"]])
    out, st = R.check_prose(L, layer["claims"])
    assert [f["check"] for f in out if f["check"].startswith("prose_")] == ["prose_rounding"]
    assert out[0]["severity"] == "warn" and st["rounding_notes"] == 1
    other = next(x for x in layer["claims"] if x["id"] != c["id"] and x.get("n"))
    L2 = prose_layer(layer, f"It was checked on {other['n'] + 0} training patients and 12345 more.", [c["id"]])
    out2, _ = R.check_prose(L2, layer["claims"])
    assert any(f["check"] == "prose_number" and "12345" in f["stated"] for f in out2)


def test_lint_percents_flags_double_rounding_only():
    from explaneat.analysis.plain_language_store import lint_percents
    claims = [{"risk": {"from": 0.0226, "to": 0.0445479}}]
    assert lint_percents("from about 2% to about 4%", claims) == []
    errs = lint_percents("from about 2% to about 5%", claims)
    assert len(errs) == 1 and "rounded twice" in errs[0]
    assert lint_percents("it acts on 26% of customers", claims) == []       # unrelated share: not flagged
    from explaneat.analysis.plain_language_templates import pct
    assert pct(0.0445479) == "4%" and pct(0.0455) == "5%"


def test_lint_percents_ignores_category_shares_of_a_subgroup():
    """heart100-s0 (errata v2, E13): a 19.3% category share in evidence.who is not a risk."""
    from explaneat.analysis.plain_language_store import lint_percents
    claims = [{"evidence": {"who": {"categories": [{"share_in": 0.126761, "share_out": 0.192982}]}}}]
    assert lint_percents("the model predicts a risk of heart disease of about 20%", claims) == []
    claims.append({"risk": {"from": 0.192982, "to": 0.3}})      # the same value as a risk is still checked
    assert lint_percents("about 20%", claims)


# ----------------------------------------------------------------------------- ablate
def test_ablate_node_edge_and_multiple(vm):
    part = P.ablate(vm, "gate")
    node = P.ablate(vm, nodes=["5"])
    edge = P.ablate(vm, edges=["5->0"], to="zero")
    assert node["data"]["mean_risk_change"] == pytest.approx(part["data"]["mean_risk_change"])
    assert edge["data"]["mean_risk_change"] == pytest.approx(part["data"]["mean_risk_change"])
    both = P.ablate(vm, parts=["gate"], edges=["-1_b->0"], to="zero")
    assert len(both["switched_off"]) == 2 and both["data"]["mean_abs_score_change"] > part["data"]["mean_abs_score_change"]
    direct = P.ablate(vm, direct=True, to="zero")
    assert any("direct_edges_of_root" in w for w in direct["switched_off"])
    kept = P.ablate(vm, edges=["-1_b->0"], to="mean")         # mean contribution kept in the bias
    assert kept["switched_off"][0]["replaced_by_mean_contribution"] == pytest.approx(
        0.5 * float(np.mean(vm.m.X[:, 0])), abs=1e-9)
    with pytest.raises(KeyError):
        P.ablate(vm, edges=["5->-2"])
    with pytest.raises(ValueError):
        P.ablate(vm)
    assert "baseline" in part["to_meaning"]


# ----------------------------------------------------------------------------- probe
def test_probe_nodes_and_full_precision(vm):
    res = P.probe(vm, sets=["dose=3.33333"], nodes=["5"])
    row = res["rows"][0]
    z = row["nodes"]["5"]
    assert z["value"] == pytest.approx(max(z["pre_activation"], 0.0))
    assert "3.33333 mg" in res["set"]["dose"]
    res2 = P.probe(vm, nodes=["gate"])
    assert "5" in res2["nodes"]
    gid = gate_part(vm).id
    assert "acts" in row["parts"][gid] and "exit_nonzero" in row["parts"][gid]


def test_probe_claim_reproduces_the_claim_case(vm, layer):
    c = next(c for c in layer["claims"] if c["type"] == "direction" and c["scope"] == "model"
             and (c.get("risk") or {}).get("from") is not None and c["statement"].get("change"))
    res = P.probe(vm, claim=c, parts=False)
    risks = [r["risk"] for r in res["rows"]]
    assert risks[0] == pytest.approx(c["risk"]["from"], abs=1e-4)
    assert risks[-1] == pytest.approx(c["risk"]["to"], abs=1e-4)
    assert res["claim"]["id"] == c["id"]


# ----------------------------------------------------------------------------- subgroup
def test_subgroup_acts_on_and_quoting_hint(vm):
    res = P.subgroup(vm, "grp == '1'")
    gid = gate_part(vm).id
    pp = res["subgroup"]["parts"][gid]
    assert {"n_acts_on", "share_acts_on", "share_exit_nonzero"} <= set(pp)
    with pytest.raises(ValueError, match="quoted"):
        P.subgroup(vm, "grp == 1")
    with pytest.raises(ValueError, match="backticks"):
        P.subgroup(vm, "dose >> ")


# ----------------------------------------------------------------------------- linearise, enumerate
def test_linearise_parts_nodes_core(vm):
    from explaneat.verify.linear import linearise
    r = linearise(vm, "parts", core=True)
    assert r["auc_full"] >= r["auc_all_affine"] - 1e-9 and "gate" in r["units"]
    assert "auc_linear_core_alone" in r
    rn = linearise(vm, "nodes")
    assert rn["switching_units"] == ["5"]


def rule_base():
    m = H.rule_model()
    return BaseData(genome_id=GENOME, phenotype=m.structure, X_train=m.X, y_train=m.y, X_test=m.Xt, y_test=m.yt,
                    feature_names=m.feature_names, raw_train=m.raw_train, raw_test=m.raw_test, mean=m.mean,
                    scale=m.scale, dataset_name="rule", meta={}, db_operations=m.ops, codebook=m.codebook)


def test_enumerate_small_space_with_a_rule():
    from explaneat.verify.enumerate import enumerate_space
    rvm = VerifyModel(rule_base())
    r = enumerate_space(rvm, rule="a1_1 + a2_1 == 1", group_by="a1_1 + a2_1")
    assert r["n_combinations"] == 4 and r["rule"]["agreement"] == 1.0 and r["rule"]["counterexamples"] == []
    bad = enumerate_space(rvm, rule="a1 == '1'")
    assert bad["rule"]["agree"] == 2 and len(bad["rule"]["counterexamples"]) == 2
    assert {g["value"]: g["share_yes"] for g in r["groups"]} == {0: 0.0, 1: 1.0, 2: 0.0}


def test_enumerate_refuses_continuous_inputs(vm):
    from explaneat.verify.enumerate import enumerate_space
    with pytest.raises(ValueError, match="distinct values"):
        enumerate_space(vm)


# ----------------------------------------------------------------------------- cache and blindness
def test_claims_cache(tmp_path, monkeypatch, vm):
    monkeypatch.setenv("EXPLANEAT_HOME", str(tmp_path))
    first = R.recompute_claims(vm)
    assert list((tmp_path / "verify_cache" / "claims").glob("*.json"))
    seen = []
    second = R.recompute_claims(vm, log=seen.append)
    assert second == json.loads(json.dumps(first)) and any("cache" in s for s in seen)
    ops = with_hypothesis(vm, "changed")
    assert R.claims_cache_key(VerifyModel(vm.base, ops)) != R.claims_cache_key(vm)


def test_target_operations_never_load_the_db_ones(monkeypatch, base):
    import explaneat.verify.model as M

    def boom(*a, **k):
        raise AssertionError("the DB operations were loaded")
    b = copy.deepcopy(base)
    b.db_operations = []
    monkeypatch.setattr(M, "load_db_operations", boom)
    vm2 = VerifyModel(b, H.operations("interaction"))
    assert vm2.operations_source == "given"
    with pytest.raises(AssertionError):
        VerifyModel(b)
    assert not hasattr(M.BaseData, "db_operations") or M.BaseData.__dataclass_fields__["db_operations"].default_factory() == []


def test_cli_new_commands(tmp_path, monkeypatch, base, vm):
    import explaneat.verify.model as M
    from explaneat.cli import cli
    monkeypatch.setattr(M, "load_base", lambda gid, cache=True, db=None: base)
    monkeypatch.setenv("EXPLANEAT_HOME", str(tmp_path / "home"))
    d = write_item(tmp_path, {"genome_id": GENOME, "claims": [], "parts": []}, record_for(vm), vm.operations)
    run = CliRunner().invoke
    r = run(cli, ["verify", "ablate", str(d), "--part", "gate", "--edge", "-1_b->0"])
    assert r.exit_code == 0, r.output
    assert "edge -1_b->0" in r.output
    r = run(cli, ["verify", "probe", str(d), "--sweep", "dose=0:10:3", "--nodes", "5"])
    assert r.exit_code == 0 and "node 5" in r.output, r.output
    r = run(cli, ["verify", "probe", str(d), "--sweep", "dose=0:10:3", "--wide"])
    assert r.exit_code == 0 and "risk without" in r.output
    r = run(cli, ["verify", "subgroup", str(d), "--where", "grp == 1"])
    assert r.exit_code != 0 and "quoted" in r.output
    r = run(cli, ["verify", "linearise", str(d), "--core"])
    assert r.exit_code == 0 and "linear core alone" in r.output, r.output
    r = run(cli, ["verify", "enumerate", str(d)])
    assert r.exit_code != 0 and "distinct values" in r.output
    assert "claude" not in r.output.lower()
