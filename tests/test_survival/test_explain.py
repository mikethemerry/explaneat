"""Hazard-ratio readings of a survival network (explaneat.survival.explain)."""

import math

import numpy as np
import pytest

from explaneat.core.genome_network import NetworkConnection, NetworkNode, NetworkStructure, NodeType
from explaneat.survival.explain import (
    contribution_summary,
    direct_edge_reading,
    explain_output,
    hinge_reading,
    output_terms,
)


def _structure():
    # g = 0.3 + 0.5 * x1 + 0.8 * relu(2 * x2 - 1)
    nodes = [
        NetworkNode(id="-1", type=NodeType.INPUT),
        NetworkNode(id="-2", type=NodeType.INPUT),
        NetworkNode(id="5", type=NodeType.HIDDEN, bias=-1.0, activation="relu"),
        NetworkNode(id="0", type=NodeType.OUTPUT, bias=0.3, activation="identity"),
    ]
    conns = [
        NetworkConnection(from_node="-1", to_node="0", weight=0.5, enabled=True),
        NetworkConnection(from_node="-2", to_node="5", weight=2.0, enabled=True),
        NetworkConnection(from_node="5", to_node="0", weight=0.8, enabled=True),
    ]
    return NetworkStructure(nodes=nodes, connections=conns, input_node_ids=["-1", "-2"],
                            output_node_ids=["0"], metadata={"output_activation": "identity"})


SCALER = {"mean": [10.0, 50.0], "scale": [2.0, 5.0]}
NAMES = ["age_decade", "sbp"]


def test_output_terms():
    t = output_terms(_structure())
    assert t["activation"] == "identity" and t["bias"] == 0.3
    assert [x["from"] for x in t["terms"]] == ["5", "-1"]  # sorted by |weight|


def test_direct_edge_reading_units():
    r = direct_edge_reading("-1", 0.5, SCALER, NAMES)
    assert r["feature"] == "age_decade"
    assert r["hr_per_sd"] == pytest.approx(math.exp(0.5))
    assert r["log_hr_per_unit"] == pytest.approx(0.25)


def test_hinge_reading_threshold_in_raw_units():
    r = hinge_reading(_structure(), "5", 0.8, SCALER, NAMES)
    assert r["feature"] == "sbp" and r["active_side"] == "above"
    assert r["threshold_sd"] == pytest.approx(0.5)
    assert r["threshold_raw"] == pytest.approx(50.0 + 5.0 * 0.5)
    assert r["log_hr_per_sd_active"] == pytest.approx(1.6)
    assert r["log_hr_per_unit_active"] == pytest.approx(1.6 / 5.0)


def test_explain_output_spreads_match_the_model():
    X = np.random.default_rng(0).normal(size=(500, 2))
    readings = {r["node"]: r for r in explain_output(_structure(), X, SCALER, NAMES)}
    hinge = 0.8 * np.maximum(2 * X[:, 1] - 1, 0)
    assert readings["5"]["spread"]["fraction_nonzero"] == pytest.approx(np.mean(hinge > 0))
    assert readings["-1"]["spread"]["hr_p95_vs_p5"] == pytest.approx(
        math.exp(0.5 * (np.percentile(X[:, 0], 95) - np.percentile(X[:, 0], 5))))


def test_contribution_summary_constant_term():
    s = contribution_summary(np.zeros(10), 3.0)
    assert s["hr_p95_vs_p5"] == 1.0 and s["fraction_nonzero"] == 0.0
