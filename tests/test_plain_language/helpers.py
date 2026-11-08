"""Synthetic explained networks with a known interaction (or none) for the plain-language tests.

Columns: ``dose`` (mg, continuous), ``grp:0`` / ``grp:1`` (one-hot group A / B), ``noise``
(no edges: the model ignores it). The input ``dose`` is split into two copies: ``-1_a`` feeds
the hidden gate, ``-1_b`` goes straight to the output.

* ``interaction``: gate 5 = relu(1.5 dose + 2.0 grp:1 - 1.0) -> output (weight 1); in group B
  the gate is open for almost every dose, in group A only for high doses, so dose raises the
  risk more in group B.
* ``tiny``: the same gate with grp:1 weight 0.01: an interaction exists in the formula but is
  negligible on the data.
* ``additive``: gate 5 = relu(1.5 dose - 1.0) and group B adds 1.0 straight to the output:
  no interaction between dose and group.
"""

import numpy as np

from explaneat.core.genome_network import NetworkConnection, NetworkNode, NetworkStructure, NodeType

FEATURES = ["dose", "grp:0", "grp:1", "noise"]

CODEBOOK = {
    "id": "test-codebook",
    "source": "synthetic test data",
    "outcome": {"positive": "the event", "short": "the event", "risk": "the risk of the event"},
    "patient": "patients",
    "gaps": [],
    "variables": {
        "dose": {"label": "dose", "unit": "mg", "decimals": 1, "kind": "continuous"},
        "grp": {"label": "group", "kind": "category", "levels": {
            "0": {"label": "group A", "group": "patients in group A", "cond": "the patient is in group A"},
            "1": {"label": "group B", "group": "patients in group B", "cond": "the patient is in group B"}}},
        "noise": {"label": "noise", "unit": "units", "decimals": 2, "kind": "continuous"},
    },
}


def _node(i, t, act, bias=0.0):
    return NetworkNode(id=i, type=t, bias=bias, activation=act, response=1.0, aggregation="sum")


def raw_data(n=300, seed=0):
    rng = np.random.RandomState(seed)
    dose = np.round(rng.uniform(0, 10, n), 1)
    g = (rng.uniform(size=n) < 0.5).astype(float)
    noise = rng.normal(size=n)
    return np.column_stack([dose, 1 - g, g, noise])


def structure(kind: str) -> NetworkStructure:
    nodes = [_node("-1_a", NodeType.INPUT, "identity"), _node("-1_b", NodeType.INPUT, "identity"),
             _node("-2", NodeType.INPUT, "identity"), _node("-3", NodeType.INPUT, "identity"),
             _node("-4", NodeType.INPUT, "identity"),
             _node("5", NodeType.HIDDEN, "relu", -1.0), _node("0", NodeType.OUTPUT, "sigmoid", -0.5)]
    w_g = {"interaction": 2.0, "tiny": 0.01, "additive": 0.0}[kind]
    conns = [NetworkConnection(from_node="-1_a", to_node="5", weight=1.5, enabled=True),
             NetworkConnection(from_node="5", to_node="0", weight=1.0, enabled=True),
             NetworkConnection(from_node="-1_b", to_node="0", weight=0.5, enabled=True)]
    if w_g:
        conns.append(NetworkConnection(from_node="-3", to_node="5", weight=w_g, enabled=True))
    else:
        conns.append(NetworkConnection(from_node="-3", to_node="0", weight=1.0, enabled=True))
    return NetworkStructure(nodes=nodes, connections=conns,
                            input_node_ids=["-1_a", "-1_b", "-2", "-3", "-4"], output_node_ids=["0"])


def operations(kind: str):
    gate_entries = ["-1_a", "-3"] if kind != "additive" else ["-1_a"]
    gate = {"seq": 1, "type": "annotate", "result": None, "created_at": "2026-11-08T10:00:00",
            "params": {"name": "gate", "hypothesis": "Node 5 is a ReLU gate on dose; Claude checked it.",
                       "entry_nodes": gate_entries, "exit_nodes": ["5"],
                       "subgraph_nodes": gate_entries + ["5"],
                       "subgraph_connections": [[e, "5"] for e in gate_entries]}}
    root_entries = ["5", "-1_b", "-2", "-4"] + (["-3"] if kind == "additive" else [])
    root = {"seq": 2, "type": "annotate", "result": None, "created_at": "2026-11-08T10:05:00",
            "params": {"name": "root", "hypothesis": "The output adds the gate and dose.",
                       "entry_nodes": root_entries, "exit_nodes": ["0"],
                       "subgraph_nodes": root_entries + ["0"],
                       "subgraph_connections": [[e, "0"] for e in root_entries if e not in ("-2", "-4")],
                       "child_annotation_ids": ["gate"]}}
    return [gate, root]


def model(kind: str = "interaction", n=300):
    from explaneat.analysis.plain_language import PlainModel
    from explaneat.analysis.explanation_tools import StateEvaluator
    raw = raw_data(n, seed=0)
    raw_t = raw_data(100, seed=1)
    mean, scale = raw.mean(axis=0), raw.std(axis=0)
    X, Xt = (raw - mean) / scale, (raw_t - mean) / scale
    s = structure(kind)
    ev = StateEvaluator(s)
    rng = np.random.RandomState(42)
    p = 1 / (1 + np.exp(-ev.logit(X)))
    pt = 1 / (1 + np.exp(-ev.logit(Xt)))
    y = (rng.uniform(size=n) < p).astype(float)
    yt = (rng.uniform(size=len(Xt)) < pt).astype(float)
    return PlainModel(s, operations(kind), X, y, Xt, yt, FEATURES, mean, scale, CODEBOOK,
                      raw_train=raw, raw_test=raw_t)


META = {"label": "synthetic", "genome_id": "00000000-0000-0000-0000-00000000000a",
        "explanation_id": "x", "dataset": {"id": "d", "name": "synthetic"},
        "split": {"id": "s", "scaler_type": "StandardScaler"}}
