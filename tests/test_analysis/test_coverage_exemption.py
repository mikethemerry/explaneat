"""Disconnected-input exemption from the single-root rule (decision of 7 Nov 2026)."""
from explaneat.analysis.coverage import (
    compute_compositional_coverage_from_hierarchy as comp,
    nodes_without_path_to_outputs,
)


def test_nodes_without_path_to_outputs():
    nodes = {"-1", "-2", "-3", "5", "0"}
    edges = {("-1", "5"), ("5", "0"), ("-2", "0")}
    assert nodes_without_path_to_outputs(nodes, edges, {"0"}) == {"-3"}


ROOT = {"name": "root", "subgraph_nodes": ["-1", "-2", "5", "0"], "exit_nodes": ["0"]}
LEAF = {"name": "leaf", "subgraph_nodes": ["-1", "5"], "exit_nodes": ["5"], "parent_annotation_id": "root"}
UNUSED = {"name": "unused_inputs", "subgraph_nodes": ["-3", "-4"], "exit_nodes": []}


def test_strict_without_disconnected_set():
    r = comp([ROOT, LEAF, UNUSED], {"0"})
    assert r["compositional_coverage"] == 0.5 and r["exempt"] == []


def test_disconnected_inputs_annotation_is_exempt():
    r = comp([ROOT, LEAF, UNUSED], {"0"}, disconnected_nodes={"-3", "-4"})
    assert r["exempt"] == ["unused_inputs"]
    assert r["compositional_coverage"] == 1.0 and r["complete"]


def test_annotation_with_a_connected_node_is_not_exempt():
    partly = {"name": "partly", "subgraph_nodes": ["-3", "-1"], "exit_nodes": []}
    r = comp([ROOT, LEAF, partly], {"0"}, disconnected_nodes={"-3"})
    assert r["exempt"] == [] and r["compositional_coverage"] < 1.0


def _state(conns, inputs, outputs, hidden=()):
    from types import SimpleNamespace as NS
    nodes = list(inputs) + list(hidden) + list(outputs)
    return NS(get_node_ids=lambda: nodes, input_node_ids=list(inputs), output_node_ids=list(outputs),
              connections=[NS(from_node=a, to_node=b, weight=w, enabled=True) for a, b, w in conns])


def test_exact_nulls_count_as_covered():
    """Dead-end hidden unit, zero-weight edge and disconnected input are Null."""
    from explaneat.analysis.coverage_report import coverage_report
    # -1 -> 5 -> 0 is live; -2 -> 6 is a dead end (6 has no path to 0);
    # -3 -> 0 has weight 0; -4 is disconnected.
    st = _state([("-1", "5", 1.0), ("5", "0", 1.0), ("-2", "6", 0.7), ("-3", "0", 0.0)],
                inputs=["-1", "-2", "-3", "-4"], outputs=["0"], hidden=["5", "6"])
    rep = coverage_report(st, [])
    assert set(rep["null"]["nodes"]) == {"-2", "6", "-3", "-4"}
    assert ["-3", "0"] in rep["null"]["edges"] and ["-2", "6"] in rep["null"]["edges"]
    assert rep["structural_coverage"] == round(4 / 6, 4)  # 6 candidates, 4 null
