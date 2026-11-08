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
    "outcome": {"positive": "the event", "short": "the event", "risk": "the risk of the event",
                "predicted": "the predicted risk of the event"},
    "patient": "patients", "patient_one": "patient",
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
             _node("5", NodeType.HIDDEN, "relu", {"mixed": 1.5}.get(kind, -1.0)), _node("0", NodeType.OUTPUT, "sigmoid", -0.5)]
    w_g = {"interaction": 2.0, "tiny": 0.01, "additive": 0.0, "mixed": 3.0}[kind]
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


# ---------------------------------------------------------------- a known logistic model
LIN_W, LIN_B = 1.2, 0.3          # logit = LIN_B + LIN_W * z(dose) - 0.8 * z(grp:1)


def linear_model(n=400):
    """A plain logistic model (no hidden node): its effects are known in closed form."""
    from explaneat.analysis.plain_language import PlainModel
    from explaneat.analysis.explanation_tools import StateEvaluator
    nodes = [_node("-1", NodeType.INPUT, "identity"), _node("-2", NodeType.INPUT, "identity"),
             _node("-3", NodeType.INPUT, "identity"), _node("-4", NodeType.INPUT, "identity"),
             _node("0", NodeType.OUTPUT, "sigmoid", LIN_B)]
    conns = [NetworkConnection(from_node="-1", to_node="0", weight=LIN_W, enabled=True),
             NetworkConnection(from_node="-3", to_node="0", weight=-0.8, enabled=True)]
    s = NetworkStructure(nodes=nodes, connections=conns, input_node_ids=["-1", "-2", "-3", "-4"],
                         output_node_ids=["0"])
    ops = [{"seq": 1, "type": "annotate", "result": None, "created_at": "2026-11-08T10:00:00",
            "params": {"name": "root", "hypothesis": "Logistic.", "entry_nodes": ["-1", "-2", "-3", "-4"],
                       "exit_nodes": ["0"], "subgraph_nodes": ["-1", "-2", "-3", "-4", "0"],
                       "subgraph_connections": [["-1", "0"], ["-3", "0"]]}}]
    raw, raw_t = raw_data(n, 0), raw_data(150, 1)
    mean, scale = raw.mean(axis=0), raw.std(axis=0)
    X, Xt = (raw - mean) / scale, (raw_t - mean) / scale
    ev = StateEvaluator(s)
    rng = np.random.RandomState(7)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-ev.logit(X)))).astype(float)
    yt = (rng.uniform(size=len(Xt)) < 1 / (1 + np.exp(-ev.logit(Xt)))).astype(float)
    return PlainModel(s, ops, X, y, Xt, yt, FEATURES, mean, scale, CODEBOOK, raw_train=raw, raw_test=raw_t)


# ---------------------------------------------------------------- a count rule (MONK-like)
def rule_model(threshold_logit=8.0):
    """Two binary attributes; the model predicts positive iff exactly one is at its first value."""
    import itertools as it
    from explaneat.analysis.plain_language import PlainModel
    feats = ["a1:1", "a1:2", "a2:1", "a2:2"]
    rows = []
    for _ in range(30):
        for a1, a2 in it.product((1, 2), (1, 2)):
            rows.append([a1 == 1, a1 == 2, a2 == 1, a2 == 2])
    raw = np.array(rows, dtype=float)
    mean, scale = raw.mean(axis=0), raw.std(axis=0)
    X = (raw - mean) / scale
    # h = relu(z(a1:1) + z(a2:1) - 1.5) fires when both are 1 (h = 0.5); logit = k/2 + k*(z1 + z2) - 6k*h:
    # one at its first value: k/2 > 0; none: -1.5k; both: -0.5k
    nodes = [_node("-1", NodeType.INPUT, "identity"), _node("-2", NodeType.INPUT, "identity"),
             _node("-3", NodeType.INPUT, "identity"), _node("-4", NodeType.INPUT, "identity"),
             _node("5", NodeType.HIDDEN, "relu", -1.5), _node("0", NodeType.OUTPUT, "sigmoid", threshold_logit / 2)]
    k = threshold_logit
    conns = [NetworkConnection(from_node="-1", to_node="5", weight=1.0, enabled=True),
             NetworkConnection(from_node="-3", to_node="5", weight=1.0, enabled=True),
             NetworkConnection(from_node="5", to_node="0", weight=-6 * k, enabled=True),
             NetworkConnection(from_node="-1", to_node="0", weight=k, enabled=True),
             NetworkConnection(from_node="-3", to_node="0", weight=k, enabled=True)]
    s = NetworkStructure(nodes=nodes, connections=conns, input_node_ids=["-1", "-2", "-3", "-4"],
                         output_node_ids=["0"])
    ops = [{"seq": 1, "type": "annotate", "result": None, "created_at": "2026-11-08T10:00:00",
            "params": {"name": "root", "hypothesis": "Rule.", "entry_nodes": ["-1", "-2", "-3", "-4"],
                       "exit_nodes": ["0"], "subgraph_nodes": ["-1", "-2", "-3", "-4", "5", "0"],
                       "subgraph_connections": [["-1", "5"], ["-3", "5"], ["5", "0"], ["-1", "0"], ["-3", "0"]]}}]
    cb = {"id": "rule-test", "source": "synthetic", "gaps": [],
          "outcome": {"short": "'yes'", "risk": "the chance of 'yes'", "predicted": "the predicted probability of 'yes'"},
          "patient": "cases", "patient_one": "case",
          "count_rules": [{"id": "first", "name": "number of attributes at their first value",
                           "terms": [["a1", "1"], ["a2", "1"]], "positive_label": "'yes'",
                           "count_phrase": "attributes take their first value"}],
          "variables": {a: {"label": f"attribute {a}", "kind": "category",
                             "levels": {str(v): {"label": f"{a} = {v}", "group": f"cases with {a} = {v}",
                                                  "cond": f"{a} = {v}"} for v in (1, 2)}} for a in ("a1", "a2")}}
    from explaneat.analysis.explanation_tools import StateEvaluator
    y = (StateEvaluator(s).logit(X) > 0).astype(float)
    return PlainModel(s, ops, X, y, X, y, feats, mean, scale, cb, raw_train=raw, raw_test=raw)
