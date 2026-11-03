"""Compositional coverage (paper Def 11) computed from the annotation hierarchy.

Full compositional coverage (1.0) requires that the hierarchy composes into a
single global root: exactly one root annotation whose (expanded) subgraph
contains every output node, with every annotation descending from it.
"""

import pytest

from explaneat.analysis.coverage import compute_compositional_coverage_from_hierarchy as ccov


def _ann(name, nodes, exits=(), children=()):
    return {
        "name": name,
        "subgraph_nodes": list(nodes),
        "exit_nodes": list(exits),
        "child_annotation_ids": list(children),
    }


OUT = {"0"}


def test_no_annotations_is_zero():
    r = ccov([], OUT)
    assert r["compositional_coverage"] == 0.0


def test_single_leaf_covering_output_is_complete():
    r = ccov([_ann("A", ["-1", "0"], ["0"])], OUT)
    assert r["compositional_coverage"] == 1.0
    assert r["complete"] is True


def test_single_leaf_not_reaching_output():
    r = ccov([_ann("A", ["-1", "3"], ["3"])], OUT)
    assert r["compositional_coverage"] == 0.0
    assert r["complete"] is False


def test_leaves_without_composition_are_incomplete():
    """The old metric was 1.0 here (no compositions -> nothing required)."""
    anns = [_ann("A", ["-1", "3"], ["3"]), _ann("B", ["3", "-2", "0"], ["0"])]
    r = ccov(anns, OUT)
    assert r["compositional_coverage"] == pytest.approx(0.0)
    assert r["roots"] == ["A", "B"]


def test_complete_hierarchy():
    anns = [
        _ann("A", ["-1", "3"], ["3"]),
        _ann("B", ["-2", "4"], ["4"]),
        _ann("C", ["5"], ["5"], children=["A", "B"]),
        _ann("R", ["0"], ["0"], children=["C"]),
    ]
    r = ccov(anns, OUT)
    assert r["compositional_coverage"] == 1.0
    assert r["complete"] is True
    assert r["roots"] == ["R"]


def test_any_composition_no_longer_implies_full_coverage():
    """Old metric: compositions/(annotations-leaves) == 1.0 whenever a
    composition exists.  A composition that does not reach the output and
    leaves another root aside is not complete."""
    anns = [
        _ann("A", ["-1", "3"], ["3"]),
        _ann("B", ["-2", "4"], ["4"]),
        _ann("C", ["5"], ["5"], children=["A", "B"]),  # does not reach 0
        _ann("D", ["-3", "0"], ["0"]),                # separate root
    ]
    r = ccov(anns, OUT)
    assert 0.0 < r["compositional_coverage"] < 1.0
    # 2 of the 3 required parent links (A, B, and one of C/D) exist
    assert r["compositional_coverage"] == pytest.approx(2 / 3)
    assert r["complete"] is False


def test_single_root_not_containing_output():
    anns = [
        _ann("A", ["-1", "3"], ["3"]),
        _ann("C", ["5"], ["5"], children=["A"]),
    ]
    r = ccov(anns, OUT)
    assert r["compositional_coverage"] == pytest.approx(1 / 2)
    assert r["root_contains_outputs"] is False


def test_output_inside_descendant_counts_for_root():
    """Root's expanded subgraph (its descendants) may hold the output node."""
    anns = [
        _ann("A", ["-1", "0"], ["0"]),
        _ann("R", [], ["0"], children=["A"]),
    ]
    r = ccov(anns, OUT)
    assert r["compositional_coverage"] == 1.0


def test_children_referenced_by_id():
    anns = [
        dict(_ann("A", ["-1", "3"], ["3"]), id="ann_0"),
        dict(_ann("R", ["0"], ["0"], children=["ann_0"]), id="ann_1"),
    ]
    r = ccov(anns, OUT)
    assert r["compositional_coverage"] == 1.0


def test_parent_pointer_form():
    """AnnotationData-style input (parent_annotation_id) is accepted too."""
    anns = [
        {"name": "A", "subgraph_nodes": ["-1", "3"], "exit_nodes": ["3"],
         "parent_annotation_id": "R"},
        {"name": "R", "subgraph_nodes": ["0"], "exit_nodes": ["0"],
         "parent_annotation_id": None},
    ]
    r = ccov(anns, OUT)
    assert r["compositional_coverage"] == 1.0
