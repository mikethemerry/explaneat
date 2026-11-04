"""Every params example in apply_operation's docstring must be accepted by the
engine. The examples are written to apply, in the order listed, to the small
network documented in the docstring (rebuilt here)."""

import json
import re

from explaneat.core.genome_network import (
    NetworkConnection,
    NetworkNode,
    NetworkStructure,
    NodeType,
)
from explaneat.core.model_state import ModelStateEngine
from explaneat.core.operations import validate_operation

# All types validate_operation knows, except "retrain" (written by the
# training pipeline, not by agents through apply_operation).
DOCUMENTED_TYPES = {
    "split_node",
    "consolidate_node",
    "add_identity_node",
    "add_node",
    "remove_node",
    "disable_connection",
    "enable_connection",
    "rename_node",
    "annotate",
    "rename_annotation",
    "prune_connection",
    "prune_node",
}


def _doc_network() -> NetworkStructure:
    nodes = [
        NetworkNode(id="-1", type=NodeType.INPUT),
        NetworkNode(id="-2", type=NodeType.INPUT),
        NetworkNode(id="-3", type=NodeType.INPUT),
        NetworkNode(id="5", type=NodeType.HIDDEN, bias=0.1, activation="relu"),
        NetworkNode(id="6", type=NodeType.HIDDEN, bias=0.2, activation="relu"),
        NetworkNode(id="0", type=NodeType.OUTPUT, bias=0.0, activation="sigmoid"),
    ]
    edges = [("-1", "5"), ("-2", "5"), ("5", "0"), ("5", "6"), ("-3", "6"), ("6", "0"), ("-3", "0")]
    conns = [NetworkConnection(from_node=a, to_node=b, weight=0.5, enabled=True) for a, b in edges]
    return NetworkStructure(
        nodes=nodes, connections=conns, input_node_ids=["-1", "-2", "-3"], output_node_ids=["0"]
    )


def _documented_examples():
    from mcp_server.tools.operations import apply_operation

    doc = apply_operation.__doc__
    examples = []
    for m in re.finditer(r"^\s*- (\w+): (\{.*\})\s*$", doc, flags=re.M):
        examples.append((m.group(1), json.loads(m.group(2))))
    return examples


def test_docstring_documents_every_operation_type():
    types = [t for t, _ in _documented_examples()]
    assert set(types) == DOCUMENTED_TYPES
    assert len(types) == len(set(types))


def test_documented_add_identity_node_uses_engine_param_names():
    params = dict(_documented_examples())["add_identity_node"]
    assert set(params) == {"target_node", "connections", "new_node_id"}


def test_every_documented_example_validates_and_applies_in_order():
    engine = ModelStateEngine(_doc_network())
    for op_type, params in _documented_examples():
        errors = engine.validate_operation(op_type, params)
        assert errors == [], f"{op_type} {params}: {errors}"
        engine.add_operation(op_type, params, validate=True)
    assert len(engine.operations) == len(DOCUMENTED_TYPES)
