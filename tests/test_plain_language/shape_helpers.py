"""Synthetic models with known response shapes for the shape tests."""

import numpy as np

from explaneat.core.genome_network import NetworkConnection, NetworkStructure, NodeType

from .helpers import _node


def curve_model(hidden, direct=0.0, bias=0.0, n=400):
    """One numeric variable ``dose`` (raw = model input: mean 0, scale 1) whose score is
    ``bias + direct*dose + sum(w * relu(a*dose + b))`` for ``hidden = [(a, b, w), ...]``, plus an
    ignored ``noise`` column. Every patient has the same curve in ``dose``."""
    from explaneat.analysis.explanation_tools import StateEvaluator
    from explaneat.analysis.plain_language import PlainModel
    rng = np.random.RandomState(3)
    raw = np.column_stack([np.round(rng.uniform(0, 10, n), 2), rng.normal(size=n)])
    raw_t = np.column_stack([np.round(rng.uniform(0, 10, 150), 2), rng.normal(size=150)])
    nodes = [_node("-1", NodeType.INPUT, "identity"), _node("-2", NodeType.INPUT, "identity"),
             _node("0", NodeType.OUTPUT, "sigmoid", bias)]
    conns = []
    if direct:
        conns.append(NetworkConnection(from_node="-1", to_node="0", weight=direct, enabled=True))
    for k, (a, b, w) in enumerate(hidden):
        h = str(10 + k)
        nodes.append(_node(h, NodeType.HIDDEN, "relu", b))
        conns += [NetworkConnection(from_node="-1", to_node=h, weight=a, enabled=True),
                  NetworkConnection(from_node=h, to_node="0", weight=w, enabled=True)]
    s = NetworkStructure(nodes=nodes, connections=conns, input_node_ids=["-1", "-2"], output_node_ids=["0"])
    sub = ["-1", "-2", "0"] + [str(10 + k) for k in range(len(hidden))]
    ops = [{"seq": 1, "type": "annotate", "result": None, "created_at": "2026-11-09T10:00:00",
            "params": {"name": "root", "hypothesis": "Curve.", "entry_nodes": ["-1", "-2"], "exit_nodes": ["0"],
                       "subgraph_nodes": sub,
                       "subgraph_connections": [[c.from_node, c.to_node] for c in conns]}}]
    ev = StateEvaluator(s)
    r = np.random.RandomState(9)
    y = (r.uniform(size=n) < 1 / (1 + np.exp(-ev.logit(raw)))).astype(float)
    yt = (r.uniform(size=150) < 1 / (1 + np.exp(-ev.logit(raw_t)))).astype(float)
    cb = {"id": "curve-test", "source": "synthetic", "gaps": [],
          "outcome": {"short": "the event", "risk": "the risk of the event",
                      "predicted": "the predicted risk of the event"},
          "patient": "patients", "patient_one": "patient",
          "variables": {"dose": {"label": "dose", "unit": "mg", "decimals": 1, "kind": "continuous"},
                        "noise": {"label": "noise", "unit": "units", "decimals": 2, "kind": "continuous"}}}
    return PlainModel(s, ops, raw, y, raw_t, yt, ["dose", "noise"], np.zeros(2), np.ones(2), cb,
                      raw_train=raw, raw_test=raw_t)


# hidden units (a, b, w): score contribution w * relu(a*dose + b)
SHAPES = {
    "steady": dict(hidden=[], direct=0.6, bias=-3.0),
    "hinge": dict(hidden=[(1.0, -4.0, 0.9)], bias=-2.0),                       # flat, then steady from 4
    "step": dict(hidden=[(10.0, -50.0, 0.8), (10.0, -54.0, -0.8)], bias=-2.0),  # a jump between 5.0 and 5.4
    "levelling_off": dict(hidden=[(1.0, -4.0, -0.75)], direct=1.0, bias=-4.0),  # slope 1, then 0.25 from 4
    "peak": dict(hidden=[(1.0, -6.0, -2.0)], direct=1.0, bias=-4.0),            # up to 6, then down
}
