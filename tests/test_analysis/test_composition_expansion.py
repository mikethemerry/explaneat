"""Composition annotations evaluate (and render) their children's subgraphs.

A composition's own ``subgraph_nodes`` hold only its junction nodes; the
children's (and grandchildren's) nodes belong to the descendants.  The
function of a composition must nevertheless be the function of the whole
subtree, otherwise child terms vanish from the formula and child-entry
inputs show zero sensitivity.
"""

import numpy as np
import torch

from explaneat.analysis import viz_data as vd
from explaneat.analysis.activation_extractor import ActivationExtractor
from explaneat.analysis.annotation_function import AnnotationFunction
from explaneat.analysis.composition import (
    annotation_dicts_from_operations,
    expand_annotation,
)
from explaneat.core.genome_network import (
    NetworkConnection,
    NetworkNode,
    NetworkStructure,
    NodeType,
)
from explaneat.core.structure_network import StructureNetwork


def _net():
    """-1,-2 -> 3 ; 3,-3 -> 4 ; 4,-4 -> 0."""
    return NetworkStructure(
        nodes=[
            NetworkNode(id="-1", type=NodeType.INPUT),
            NetworkNode(id="-2", type=NodeType.INPUT),
            NetworkNode(id="-3", type=NodeType.INPUT),
            NetworkNode(id="-4", type=NodeType.INPUT),
            NetworkNode(id="3", type=NodeType.HIDDEN, bias=0.1, activation="relu"),
            NetworkNode(id="4", type=NodeType.HIDDEN, bias=-0.2, activation="relu"),
            NetworkNode(id="0", type=NodeType.OUTPUT, bias=0.3, activation="sigmoid"),
        ],
        connections=[
            NetworkConnection(from_node="-1", to_node="3", weight=1.0, enabled=True),
            NetworkConnection(from_node="-2", to_node="3", weight=-1.0, enabled=True),
            NetworkConnection(from_node="3", to_node="4", weight=2.0, enabled=True),
            NetworkConnection(from_node="-3", to_node="4", weight=0.5, enabled=True),
            NetworkConnection(from_node="4", to_node="0", weight=1.5, enabled=True),
            NetworkConnection(from_node="-4", to_node="0", weight=-0.8, enabled=True),
        ],
        input_node_ids=["-1", "-2", "-3", "-4"],
        output_node_ids=["0"],
    )


# Stored the way the 10-seed explanations were built: the root lists its
# child's entries but (before the validator fix) not the grandchild's.
OPS = [
    {"seq": 0, "type": "annotate", "params": {
        "name": "A", "hypothesis": "", "entry_nodes": ["-1", "-2"], "exit_nodes": ["3"],
        "subgraph_nodes": ["-1", "-2", "3"],
        "subgraph_connections": [["-1", "3"], ["-2", "3"]]}},
    {"seq": 1, "type": "annotate", "params": {
        "name": "B", "hypothesis": "", "entry_nodes": ["-1", "-2", "-3"], "exit_nodes": ["4"],
        "subgraph_nodes": ["-3", "4"], "subgraph_connections": [["3", "4"], ["-3", "4"]],
        "child_annotation_ids": ["A"]}},
    {"seq": 2, "type": "annotate", "params": {
        "name": "R", "hypothesis": "", "entry_nodes": ["-3", "-4"], "exit_nodes": ["0"],
        "subgraph_nodes": ["-4", "0"], "subgraph_connections": [["4", "0"], ["-4", "0"]],
        "child_annotation_ids": ["B"]}},
]


def _X(n=64):
    return np.random.default_rng(0).normal(size=(n, 4))


def _full_output(X):
    return StructureNetwork(_net()).forward(torch.tensor(X)).detach().numpy().ravel()


def test_annotation_dicts_from_operations():
    anns = annotation_dicts_from_operations(OPS)
    assert [a["id"] for a in anns] == ["ann_0", "ann_1", "ann_2"]
    assert anns[2]["child_annotation_ids"] == ["B"]


def test_leaf_unchanged():
    anns = annotation_dicts_from_operations(OPS)
    leaf = expand_annotation(anns[0], anns, _net())
    assert leaf["entry_nodes"] == ["-1", "-2"]
    assert leaf["subgraph_nodes"] == ["-1", "-2", "3"]


def test_root_expansion_includes_all_descendants():
    anns = annotation_dicts_from_operations(OPS)
    root = expand_annotation(anns[2], anns, _net())
    # declared entries first, then grandchild entries the root omitted
    assert root["entry_nodes"][:2] == ["-3", "-4"]
    assert set(root["entry_nodes"]) == {"-1", "-2", "-3", "-4"}
    assert root["added_entry_nodes"] == ["-1", "-2"]
    assert set(root["subgraph_nodes"]) == {"-1", "-2", "-3", "-4", "3", "4", "0"}


def test_children_referenced_by_id():
    ops = [dict(op, params=dict(op["params"])) for op in OPS]
    ops[2]["params"]["child_annotation_ids"] = ["ann_1"]
    anns = annotation_dicts_from_operations(ops)
    root = expand_annotation(anns[2], anns, _net())
    assert set(root["subgraph_nodes"]) == {"-1", "-2", "-3", "-4", "3", "4", "0"}


def test_expanded_root_function_equals_model():
    net = _net()
    anns = annotation_dicts_from_operations(OPS)
    root = expand_annotation(anns[2], anns, net)
    X = _X()
    entry, exit_acts = ActivationExtractor.from_structure(net).extract(X, root)
    fn = AnnotationFunction.from_structure(root, net)
    np.testing.assert_allclose(fn(entry).ravel(), exit_acts.ravel(), atol=1e-12)
    np.testing.assert_allclose(exit_acts.ravel(), _full_output(X), atol=1e-12)


def test_sensitivity_nonzero_for_child_entries():
    net = _net()
    anns = annotation_dicts_from_operations(OPS)
    mid = expand_annotation(anns[1], anns, net)  # B over child A
    X = np.abs(_X()) + 0.5  # keep node 3 active so -1 matters
    X[:, 1] = -X[:, 1]
    entry, _ = ActivationExtractor.from_structure(net).extract(X, mid)
    fn = AnnotationFunction.from_structure(mid, net)
    sens = vd.compute_sensitivity(fn, entry, entry_names=mid["entry_nodes"])
    by_name = dict(zip(sens["input_labels"], sens["sensitivities"]["y_0"]))
    assert abs(by_name["-1"]) > 1e-3 and abs(by_name["-2"]) > 1e-3


def test_formula_includes_child_terms():
    net = _net()
    anns = annotation_dicts_from_operations(OPS)
    root = expand_annotation(anns[2], anns, net)
    latex = AnnotationFunction.from_structure(root, net).to_latex(force=True)
    for sym in ("x_{-1}", "x_{-2}", "x_{-3}", "x_{-4}"):
        assert sym in latex, latex

    children = [expand_annotation(anns[1], anns, net)]
    composed = AnnotationFunction.from_structure(root, net, child_annotations=children)
    collapsed = composed.to_latex(expand=False, force=True)
    assert "B" in collapsed and "x_{-4}" in collapsed, collapsed
    X = _X()
    entry, exit_acts = ActivationExtractor.from_structure(net).extract(X, root)
    np.testing.assert_allclose(composed(entry).ravel(), exit_acts.ravel(), atol=1e-12)


def test_collapsed_style_composition_keeps_declared_interface():
    """A composition whose entries are its children's exits stays a
    function of those exits (no upstream expansion)."""
    net = _net()
    ops = OPS[:1] + [{"seq": 1, "type": "annotate", "params": {
        "name": "C", "hypothesis": "", "entry_nodes": ["3", "-3"], "exit_nodes": ["4"],
        "subgraph_nodes": ["-3", "4"], "subgraph_connections": [["3", "4"], ["-3", "4"]],
        "child_annotation_ids": ["A"]}}]
    anns = annotation_dicts_from_operations(ops)
    c = expand_annotation(anns[1], anns, net)
    assert c["entry_nodes"] == ["3", "-3"]
    assert set(c["subgraph_nodes"]) == {"3", "-3", "4"}
