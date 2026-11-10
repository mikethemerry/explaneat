"""The ``explaneat verify`` tools on a synthetic explained network (no database).

The synthetic network is the plain-language test network (tests/test_plain_language/helpers.py):
dose (mg), group A/B, noise; a ReLU gate on dose and group feeding the output, plus a direct
dose edge. The tools must reproduce a layer built from the same model, and flag each kind of
stated number that has been changed.
"""

import copy
import json

import numpy as np
import pytest
from click.testing import CliRunner

from explaneat.verify import artefacts, probe as P, recheck as R
from explaneat.verify.model import BaseData, VerifyModel
from tests.test_plain_language import helpers as H

GENOME = "00000000-0000-0000-0000-00000000000a"


def base_data(kind="interaction", n=300) -> BaseData:
    from explaneat.analysis.explanation_tools import StateEvaluator
    raw, raw_t = H.raw_data(n, seed=0), H.raw_data(100, seed=1)
    mean, scale = raw.mean(axis=0), raw.std(axis=0)
    X, Xt = (raw - mean) / scale, (raw_t - mean) / scale
    s = H.structure(kind)
    ev = StateEvaluator(s)
    rng = np.random.RandomState(42)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-ev.logit(X)))).astype(float)
    yt = (rng.uniform(size=len(Xt)) < 1 / (1 + np.exp(-ev.logit(Xt)))).astype(float)
    return BaseData(genome_id=GENOME, phenotype=s, X_train=X, y_train=y, X_test=Xt, y_test=yt,
                    feature_names=list(H.FEATURES), raw_train=raw, raw_test=raw_t, mean=mean, scale=scale,
                    dataset_name="synthetic", meta={}, db_operations=H.operations(kind),
                    codebook=H.CODEBOOK)


@pytest.fixture(scope="module")
def base():
    return base_data()


@pytest.fixture(scope="module")
def vm(base):
    return VerifyModel(base)


@pytest.fixture(scope="module")
def layer(vm):
    from explaneat.analysis.plain_language import build_layer
    L = build_layer(vm.m, {**H.META, "genome_id": GENOME})
    gate = next(p for p in L["parts"] if p["name"] == "gate")
    sub = next(c for c in L["claims"] if c["annotation"]["id"] == gate["id"] and c["type"] == "subgroup")
    r = sub["risk"]
    L["drafted"] = {"titles": {gate["id"]: "Dose gate"}, "readdown": {gate["id"]: {
        "hypothesis": [{"text": "This part raises the predicted risk of the event with the dose.",
                        "claims": [sub["id"]]}],
        "evidence": [{"text": f"For the typical patient it acts on, the predicted risk is about "
                              f"{round(100 * r['from'])}% without it and about {round(100 * r['to'])}% with it.",
                      "claims": [sub["id"]]}]}}}
    return L


def record_for(vm):
    s = R.structure_checks(vm)
    return {"label": "synthetic", "genome_id": GENOME,
            "structural_coverage": s["structural_coverage"], "compositional_coverage": s["compositional_coverage"],
            "full_coverage": s["full_coverage"], "hierarchy_complete": s["hierarchy"]["complete"],
            "function_preservation_max_abs_err": s["function_preservation_max_abs_err"],
            "closed_form_max_abs_err": s["closed_form_max_abs_err"]}


def target(layer, record, ops):
    return artefacts.Target(genome_id=GENOME, source="test", record=copy.deepcopy(record),
                            layer=copy.deepcopy(layer), operations=copy.deepcopy(ops))


def mismatches(rep):
    return [f for f in rep["findings"] if f["severity"] == "mismatch"]


# ----------------------------------------------------------------------------- helpers
def test_numbers_in_reads_percent_money_and_decimals():
    got = R.numbers_in("from about 62% to $50,000 at 0.8 mm, then 1,234 and a1 = 1")
    assert ("62", 62.0, True) in got
    assert any(v == 50000.0 for _, v, _ in got)
    assert any(v == 0.8 for _, v, _ in got) and any(v == 1234.0 for _, v, _ in got)
    assert not any(t == "1" and p for t, _, p in got)


def test_latex_numbers_ignore_variable_names():
    lat = r"y_0 = \operatorname{relu}\bigl( 2.55 x_{oldpeak_a} + 0.937 x_{thal:7_b} - 0.276 \bigr)"
    assert sorted(R.latex_numbers(lat)) == [0.276, 0.937, 2.55]


def test_compare_claim_skips_absent_and_flags_changed():
    c = {"id": "C1", "risk": {"from": 0.2, "to": 0.5}, "shape": {"sensitivity": {"x0.75": "steady"}}}
    assert R.compare_claim(c, copy.deepcopy(c)) == []
    d = copy.deepcopy(c)
    d["risk"]["to"] = 0.6
    diffs = R.compare_claim(d, c)
    assert diffs and diffs[0]["path"] == "risk.to"


# ----------------------------------------------------------------------------- recheck
def test_recheck_reproduces_a_faithful_explanation(vm, layer):
    rec = record_for(vm)
    rep = R.recheck(target(layer, rec, vm.operations), vm=vm)
    assert rep["structure"]["full_coverage"]
    assert rep["claims"]["reproduced"] == rep["claims"]["stated"]
    assert rep["prose"]["unmatched_numbers"] == 0
    assert mismatches(rep) == []


def test_recheck_flags_a_changed_claim_number(vm, layer):
    L = copy.deepcopy(layer)
    c = next(c for c in L["claims"] if c.get("risk") and c["risk"].get("to") is not None)
    c["risk"]["to"] = c["risk"]["to"] + 0.2
    rep = R.recheck(target(L, record_for(vm), vm.operations), vm=vm)
    hit = [f for f in mismatches(rep) if f["location"] == f"claims.{c['id']}"]
    assert hit and any(d["path"] == "risk.to" for d in hit[0]["diffs"])


def test_recheck_flags_a_prose_number_and_a_wrong_citation(vm, layer):
    L = copy.deepcopy(layer)
    pid = next(iter(L["drafted"]["readdown"]))
    ev = L["drafted"]["readdown"][pid]["evidence"][0]
    ev["text"] = ev["text"].replace("without it", "without it (about 97% for most)")
    rep = R.recheck(target(L, record_for(vm), vm.operations), vm=vm)
    assert any(f["check"] == "prose_number" and "97%" in f["stated"] for f in mismatches(rep))
    L2 = copy.deepcopy(layer)
    L2["drafted"]["readdown"][pid]["evidence"][0]["claims"] = ["C999"]
    rep2 = R.recheck(target(L2, record_for(vm), vm.operations), vm=vm)
    assert any(f["check"] == "citation" for f in mismatches(rep2))


def test_recheck_flags_coverage_when_a_node_is_dropped(vm, layer):
    ops = copy.deepcopy(vm.operations)
    gate = next(o for o in ops if o["params"]["name"] == "gate")
    # drop the dose copy that only the gate reads: it is no longer covered by any annotation
    gate["params"]["subgraph_nodes"] = [n for n in gate["params"]["subgraph_nodes"] if n != "-1_a"]
    gate["params"]["entry_nodes"] = [n for n in gate["params"]["entry_nodes"] if n != "-1_a"]
    gate["params"]["subgraph_connections"] = [c for c in gate["params"]["subgraph_connections"] if c[0] != "-1_a"]
    t = target(layer, record_for(vm), ops)
    vm2 = VerifyModel(vm.base, ops)
    rep = R.recheck(t, vm=vm2, claims=False)
    assert rep["structure"]["structural_coverage"] < 1.0
    assert any(f["check"] == "coverage" and f["location"] == "record.structural_coverage" for f in mismatches(rep))


def test_recheck_flags_a_changed_closed_form_coefficient(vm, layer):
    L = copy.deepcopy(layer)
    p = next(p for p in L["parts"] if p["name"] == "gate")
    lat = p["closed_form"]["latex"]
    nums = R.latex_numbers(lat)
    assert nums
    old = f"{nums[0]:g}"
    assert old in lat
    p["closed_form"]["latex"] = lat.replace(old, f"{nums[0] + 0.5:g}", 1)
    rep = R.recheck(target(L, record_for(vm), vm.operations), vm=vm, claims=False)
    assert any(f["check"] == "closed_form" and f["location"].endswith("closed_form.latex") for f in mismatches(rep))


def test_recheck_flags_record_numbers(vm, layer):
    rec = record_for(vm)
    rec["structural_coverage"] = 0.5
    rec["n_splits"] = 3
    rep = R.recheck(target(layer, rec, vm.operations), vm=vm, claims=False)
    locs = {f["location"] for f in mismatches(rep)}
    assert "record.structural_coverage" in locs and "record.n_splits" in locs


# ----------------------------------------------------------------------------- probe, ablate, subgroup
def test_variable_and_category_lookup(vm):
    v = P.variable(vm, "group")
    assert v.base == "grp"
    assert P.category_code(v, "group B") == "1" and P.category_code(v, "0") == "0"
    with pytest.raises(ValueError):
        P.category_code(v, "group C")
    with pytest.raises(KeyError):
        P.variable(vm, "height")


def test_probe_matches_a_direct_forward_pass(vm):
    res = P.probe(vm, sets=["grp=1"], sweeps=["dose=0:10:6"])
    assert [r["inputs"]["dose"] for r in res["rows"]][:2] == ["0 mg", "2 mg"]
    raw = P.hold_row(vm, "typical")
    P.set_raw(raw, P.variable(vm, "grp"), "1")
    for r, d in zip(res["rows"], np.linspace(0, 10, 6)):
        x = raw.copy()
        x[0] = d
        want = float(vm.m.risk(vm.m.logit(((x - vm.m.mean) / vm.m.scale)[None, :]))[0])
        assert r["risk"] == pytest.approx(want)
    gate = next(p for p in vm.parts if p.name == "gate")
    contribs = [r["parts"][gate.id]["contribution"] for r in res["rows"]]
    assert contribs == sorted(contribs)       # the gate grows with the dose
    assert "training range" in res["support"]["dose"]


def test_probe_two_dimensional_grid(vm):
    res = P.probe(vm, sweeps=["dose=2,8", "grp=all"], parts=False)
    assert len(res["rows"]) == 4
    by = {(r["inputs"]["dose"], r["inputs"]["grp"]): r["risk"] for r in res["rows"]}
    assert by[("8 mg", "group B (grp=1)")] > by[("8 mg", "group A (grp=0)")] > by[("2 mg", "group A (grp=0)")]


def test_ablate_switches_the_gate_off(vm):
    a = P.ablate(vm, "gate")
    d = a["data"]
    assert a["part"]["gated"] and d["n"] == vm.m.X.shape[0]
    assert 0 < d["n_affected"] <= d["n"]
    assert d["mean_risk_change"] > 0          # the gate only adds risk
    assert a == P.ablate(vm, "Dose gate", titles={next(p.id for p in vm.parts if p.name == 'gate'): "Dose gate"})


def test_subgroup_counts_and_part_activity(vm):
    res = P.subgroup(vm, "grp == '1' and dose > 5")
    v = P.variable(vm, "grp")
    want = int(np.sum((v.raw_train == "1") & (vm.m.raw_train[:, 0] > 5)))
    assert res["subgroup"]["n"] == want
    assert res["subgroup"]["n"] + res["rest"]["n"] == res["n_total"]
    gate = next(p for p in vm.parts if p.name == "gate")
    assert res["subgroup"]["parts"][gate.id]["share_acts_on"] == pytest.approx(1.0)
    lab = P.subgroup(vm, "grp_label == 'group B'")
    assert lab["subgroup"]["n"] == int(np.sum(v.raw_train == "1"))


# ----------------------------------------------------------------------------- artefacts and the CLI
def write_item(tmp_path, layer, record, ops, item_id="vb-test01"):
    d = tmp_path / item_id
    d.mkdir()
    (d / "item.json").write_text(json.dumps({"item_id": item_id, "genome_id": GENOME}))
    (d / "record.json").write_text(json.dumps(record))
    (d / "plain_language.json").write_text(json.dumps(layer))
    (d / "operations.json").write_text(json.dumps({"genome_id": GENOME, "operations": ops}))
    return d


def test_resolve_item_directory_and_json(tmp_path, vm, layer):
    d = write_item(tmp_path, layer, record_for(vm), vm.operations)
    t = artefacts.resolve(str(d))
    assert t.genome_id == GENOME and t.operations is not None and t.layer and t.record
    assert t.label == "vb-test01"
    t2 = artefacts.resolve(str(d / "record.json"))
    assert t2.operations is not None and t2.layer is not None
    assert artefacts.kind_of(layer) == "layer" and artefacts.kind_of(vm.operations) == "operations"


def test_cli_commands_on_an_item(tmp_path, monkeypatch, base, vm, layer):
    import explaneat.verify.model as M
    from explaneat.cli import cli
    monkeypatch.setattr(M, "load_base", lambda gid, cache=True, db=None: base)
    d = write_item(tmp_path, layer, record_for(vm), vm.operations)
    run = CliRunner().invoke
    r = run(cli, ["verify", "probe", str(d), "--set", "grp=group B", "--sweep", "dose=0:10:3", "--json"])
    assert r.exit_code == 0, r.output
    assert len(json.loads(r.output)["rows"]) == 3
    r = run(cli, ["verify", "probe", str(d), "--sweep", "dose=0:10:3"])
    assert r.exit_code == 0 and "risk" in r.output
    r = run(cli, ["verify", "ablate", str(d), "--part", "gate"])
    assert r.exit_code == 0 and "affected" in r.output
    r = run(cli, ["verify", "subgroup", str(d), "--where", "dose > 5"])
    assert r.exit_code == 0 and "outcome rate" in r.output
    r = run(cli, ["verify", "vars", str(d)])
    assert r.exit_code == 0 and "group B" in r.output
    out = tmp_path / "rc.json"
    r = run(cli, ["verify", "recheck", str(d), "--out", str(out)])
    assert r.exit_code == 0, r.output
    rep = json.loads(out.read_text())
    assert rep["summary"]["mismatch"] == 0 and rep["operations_source"] == "target"
    assert "claude" not in r.output.lower()
