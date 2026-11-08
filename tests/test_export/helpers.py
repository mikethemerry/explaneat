"""A toy explained network for the export tests (no database).

    -1 --\\
          5 (relu) --> 0 (sigmoid output)
    -2 --/   \\
    -5 --> 6 (relu, dead end: no path to the output)
    -3       (disconnected input)
    -4 --> 0 with weight 0 (zero-weight edge: -4 is Null)

Annotations: leaf ``gate`` over {-1, -2, 5}, and a composition ``root`` over {5, 0} with
child ``gate``.
"""

import numpy as np

from explaneat.core.genome_network import NetworkConnection, NetworkNode, NetworkStructure, NodeType
from explaneat.core.model_state import ModelStateEngine


def _node(i, t, act, bias=0.0):
    return NetworkNode(id=i, type=t, bias=bias, activation=act, response=1.0, aggregation="sum")


def toy_phenotype():
    nodes = [_node("-1", NodeType.INPUT, "identity"), _node("-2", NodeType.INPUT, "identity"),
             _node("-3", NodeType.INPUT, "identity"), _node("-4", NodeType.INPUT, "identity"),
             _node("-5", NodeType.INPUT, "identity"), _node("5", NodeType.HIDDEN, "relu", 0.1), _node("6", NodeType.HIDDEN, "relu", -0.2),
             _node("0", NodeType.OUTPUT, "sigmoid", 0.05)]
    conns = [NetworkConnection(from_node="-1", to_node="5", weight=1.25, enabled=True),
             NetworkConnection(from_node="-2", to_node="5", weight=-0.5, enabled=True),
             NetworkConnection(from_node="-5", to_node="6", weight=0.7, enabled=True),
             NetworkConnection(from_node="5", to_node="0", weight=2.0, enabled=True),
             NetworkConnection(from_node="-4", to_node="0", weight=0.0, enabled=True)]
    return NetworkStructure(nodes=nodes, connections=conns,
                            input_node_ids=["-1", "-2", "-3", "-4", "-5"], output_node_ids=["0"])


OPS = [
    {"seq": 0, "type": "rename_node", "params": {"node_id": "-1", "display_name": "age"},
     "result": None, "created_at": "2026-11-01T10:00:00"},
    {"seq": 1, "type": "annotate", "created_at": "2026-11-01T10:05:00",
     "params": {"name": "gate", "hypothesis": "Node 5 is a ReLU of age minus half of x2; Claude checked it.",
                "entry_nodes": ["-1", "-2"], "exit_nodes": ["5"], "subgraph_nodes": ["-1", "-2", "5"],
                "subgraph_connections": [["-1", "5"], ["-2", "5"]],
                "evidence": {"records": [
                    {"type": "analytical_formula", "payload": {"latex": "y = relu(1.25 x_1 - 0.5 x_2 + 0.1)"},
                     "narrative": "closed form", "timestamp": "2026-11-01T10:06:00"},
                    {"type": "empirical_viz", "payload": {"split": "train", "frac_active": 0.6},
                     "narrative": "active on 60% of rows", "timestamp": "2026-11-01T10:07:00"},
                    {"type": "ablation_study", "payload": {"delta_auc": -0.1},
                     "narrative": "no split recorded", "timestamp": "2026-11-01T10:08:00"}]}},
     "result": None},
    {"seq": 2, "type": "annotate", "created_at": "2026-11-01T10:10:00",
     "params": {"name": "root", "hypothesis": "The output is sigmoid(2 gate).",
                "entry_nodes": ["5"], "exit_nodes": ["0"], "subgraph_nodes": ["5", "0"],
                "subgraph_connections": [["5", "0"]], "child_annotation_ids": ["gate"],
                "evidence": {"records": [
                    {"type": "analytical_formula", "payload": {"latex": "y = sigmoid(2 g)"},
                     "narrative": "composition", "timestamp": "2026-11-01T10:11:00"}]}},
     "result": None},
]

RECORD = {"label": "toy-s0", "genome_id": "00000000-0000-0000-0000-000000000001",
          "protocol_version": "v3", "start": "2026-11-01T10:00:00", "end": "2026-11-01T10:20:00",
          "structural_coverage": 1.0, "compositional_coverage": 1.0,
          "summary": "Claude explained this toy network.", "resisted": "Nothing.",
          "verified_by_orchestrator": {"full_coverage": True}}

META = {
    "experiment": {"id": "e0", "name": "toy experiment", "random_seed": 0,
                   "training_git_commit": None, "status": "completed"},
    "dataset": {"id": "d0", "name": "toy", "num_classes": 2, "num_features": 5,
                "feature_names": ["age", "x2", "x3", "x4", "x5"]},
    "split": {"id": "s0", "name": None, "train_size": 64, "test_size": 32, "scaler_type": None},
    "genome": {"id": "00000000-0000-0000-0000-000000000001", "neat_key": 7, "generation": 3,
               "fitness": 0.9},
    "explanation": {"id": "x0", "name": None, "description": None, "created_at": "2026-11-01",
                    "updated_at": "2026-11-01", "n_operations": len(OPS)},
}


def toy_engine():
    import copy
    eng = ModelStateEngine(toy_phenotype())
    eng.load_operations({"operations": copy.deepcopy(OPS)})
    return eng


def toy_data(n=64, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, 5))
    y = (X[:, 0] - 0.5 * X[:, 1] > 0).astype(float)
    return X, y


def toy_doc(now="2026-11-08T00:00:00+00:00", record=RECORD):
    import copy
    from explaneat.export.canonical import seal
    from explaneat.export.collect import build_document
    X, y = toy_data()
    Xt, yt = toy_data(32, seed=1)
    doc = build_document(toy_engine(), copy.deepcopy(OPS), copy.deepcopy(META),
                         record=copy.deepcopy(record) if record else None, label="toy-s0",
                         study={"name": "test", "set": None, "position": 1},
                         X_train=X, y_train=y, X_test=Xt, y_test=yt, num_classes=2, now=now)
    return seal(doc)
