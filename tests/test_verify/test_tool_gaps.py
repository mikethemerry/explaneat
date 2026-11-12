"""Tool gaps reported by the cox_case and line-features explainers (RESULTS.md caveats):
probe sets/sweeps on one-hot inputs whose names contain "=", enumerate with binned continuous
inputs, ablate --direct through identity relays, node ids in evidence payloads."""

import numpy as np
import pytest

from explaneat.analysis import provenance as P
from explaneat.core.genome_network import NetworkConnection, NetworkStructure, NodeType
from explaneat.verify import probe as Pr
from explaneat.verify.model import BaseData, VerifyModel
from tests.test_plain_language import helpers as H
from tests.test_verify.test_verify_tools import GENOME, base_data


# ----------------------------------------------------------------------------- "=" in input names
def eq_base() -> BaseData:
    """scikit-survival's encoder names one-hot columns horTh=yes; the model reads them as binary numbers."""
    from explaneat.analysis.plain_language_codebooks import generic_codebook
    b = base_data()
    names = ["dose", "horTh=no", "horTh=yes", "noise"]
    return BaseData(genome_id=GENOME, phenotype=b.phenotype, X_train=b.X_train, y_train=b.y_train,
                    X_test=b.X_test, y_test=b.y_test, feature_names=names, raw_train=b.raw_train,
                    raw_test=b.raw_test, mean=b.mean, scale=b.scale, dataset_name="eq", meta={},
                    db_operations=H.operations("interaction"), codebook=generic_codebook(names))


def test_probe_sets_and_sweeps_inputs_named_with_an_equals_sign():
    vm = VerifyModel(eq_base())
    assert Pr.parse_assign("horTh=yes=1", vm) == ("horTh=yes", "1")
    assert Pr.parse_assign("dose=3", vm) == ("dose", "3")
    assert Pr.parse_assign("dose=3") == ("dose", "3")            # no model: the first "=" splits
    r = Pr.probe(vm, sets=["horTh=yes=1", "dose=4"])
    assert "horTh=yes" in r["set"] and r["set"]["horTh=yes"].startswith("1")
    sw = Pr.probe(vm, sweeps=["horTh=yes=0,1"])
    assert sw["sweep"] == ["horTh=yes"] and len(sw["rows"]) == 2
    assert sw["rows"][0]["risk"] != sw["rows"][1]["risk"]        # horTh=yes feeds the gate
    with pytest.raises(KeyError):
        Pr.probe(vm, sets=["horTh=maybe=1"])


# ----------------------------------------------------------------------------- enumerate with bins
def test_enumerate_bins_continuous_inputs():
    from explaneat.verify.enumerate import enumerate_space
    vm = VerifyModel(base_data())
    with pytest.raises(ValueError, match="bins=N"):
        enumerate_space(vm)
    r = enumerate_space(vm, bins=3, group_by="grp_1")
    # dose and noise: 3 quantiles each; grp: 2 levels
    assert r["n_combinations"] == 3 * 2 * 3 and set(r["binned"]) == {"dose", "noise"}
    assert len(r["binned"]["dose"]) == 3 and "quantiles" in r["binned_note"]
    assert {g["value"]: g["n"] for g in r["groups"]} == {0: 9, 1: 9}
    assert all(0 <= row["risk"] <= 1 for row in r["table"])


# ----------------------------------------------------------------------------- ablate --direct through a relay
def relay_base() -> BaseData:
    """dose's direct edge passes through identity_0 (add_identity_node on the output)."""
    b = base_data()
    nodes = [H._node("-1_a", NodeType.INPUT, "identity"), H._node("-1_b", NodeType.INPUT, "identity"),
             H._node("-2", NodeType.INPUT, "identity"), H._node("-3", NodeType.INPUT, "identity"),
             H._node("-4", NodeType.INPUT, "identity"),
             H._node("5", NodeType.HIDDEN, "relu", -1.0), H._node("identity_0", NodeType.HIDDEN, "identity", 0.0),
             H._node("0", NodeType.OUTPUT, "sigmoid", -0.5)]
    conns = [NetworkConnection(from_node="-1_a", to_node="5", weight=1.5, enabled=True),
             NetworkConnection(from_node="-3", to_node="5", weight=2.0, enabled=True),
             NetworkConnection(from_node="5", to_node="0", weight=1.0, enabled=True),
             NetworkConnection(from_node="-1_b", to_node="identity_0", weight=0.5, enabled=True),
             NetworkConnection(from_node="identity_0", to_node="0", weight=1.0, enabled=True)]
    s = NetworkStructure(nodes=nodes, connections=conns, input_node_ids=["-1_a", "-1_b", "-2", "-3", "-4"],
                         output_node_ids=["0"])
    return BaseData(genome_id=GENOME, phenotype=s, X_train=b.X_train, y_train=b.y_train, X_test=b.X_test,
                    y_test=b.y_test, feature_names=list(H.FEATURES), raw_train=b.raw_train, raw_test=b.raw_test,
                    mean=b.mean, scale=b.scale, dataset_name="synthetic", meta={}, db_operations=[],
                    codebook=H.CODEBOOK)


def test_ablate_direct_follows_identity_relays():
    vm = VerifyModel(relay_base(), [])
    r = Pr.ablate(vm, direct=True, to="zero")
    assert r["switched_off"][0] == {"direct_edges_of_root": 1}
    assert r["switched_off"][1]["edge"] == "-1_b->identity_0"
    # the same as switching that edge off by name
    e = Pr.ablate(vm, edges=["-1_b->identity_0"], to="zero")
    assert r["data"]["mean_abs_score_change"] == pytest.approx(e["data"]["mean_abs_score_change"])
    assert r["data"]["mean_abs_score_change"] > 0


# ----------------------------------------------------------------------------- node ids are not typed numbers
def test_bare_numbers_accepts_negative_input_node_ids():
    assert P.bare_numbers("entry -6 and -12 feed 5", node_ids=["-6", "-12", "5"]) == []
    assert P.bare_numbers("entry -6 and 7", node_ids=["-6"]) == ["7"]


def test_payload_node_lists_pass_the_strict_check():
    from explaneat.analysis.narrative_check import process_payload
    vm = VerifyModel(base_data())
    payload = {"entry_nodes": ["-1_a", "-3"], "exit_nodes": ["5"], "subgraph_connections": [["-1_a", "5"], ["-3", "5"]],
               "node": "5", "latex": "5 = relu(1.5 x + 2.0 g - 1.0)", "method": "closed form of the gate"}
    out, tmpl, rep = process_payload(None, GENOME, payload, vm.operations, "gate", strict=True, model=vm)
    assert rep["errors"] == [] and out == payload and tmpl is None
    bad = {"entry_nodes": ["-1_a"], "note": "checked on 300 rows"}
    _, _, rep2 = process_payload(None, GENOME, bad, vm.operations, "gate", strict=True, model=vm)
    assert rep2["errors"] and "300" in rep2["errors"][0] and "payload.note" in rep2["errors"][0]
