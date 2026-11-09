"""v5b: one grouping everywhere (the hierarchy), part counts checked, and structure diagrams that
draw every node and edge."""

import pytest

from explaneat.analysis.plain_language import build_layer
from explaneat.analysis.plain_language_store import (ADDITIVE, lint_draft, lint_part_counts, load_layer,
                                                     readdown_order, summary_blocks)
from explaneat.export.render_structure import overview, section

from .helpers import META
from .test_v5 import _draft, _parts, _structure, copy_model


@pytest.fixture(scope="module")
def lay():
    return build_layer(copy_model(), dict(META, label="copy"))


def _coverage(layer):
    g = layer["graph"]
    edges = {(e["from"], e["to"]) for e in g["edges"]}
    tk, meta = overview(layer)
    assert meta["edges"] == edges                              # the overview draws every edge
    real = [d for d in meta["drawn"] if d[2] is not None]       # (the drawn sum's one edge aside)
    assert len(real) == len(g["edges"])
    from explaneat.export.render_structure import grouped_variables, input_key
    groups = grouped_variables(layer)
    keys = {input_key(n, groups) for n in g["nodes"] if n["kind"] == "input"}
    assert meta["inputs"] == keys                             # each input (or input group) drawn once
    hidden = {n["id"] for n in g["nodes"] if n["kind"] != "input"}
    assert hidden <= meta["nodes"]                            # ... and every other node
    owned = []
    for pid in readdown_order(layer):
        owned += sorted(section(layer, pid)[1]["edges"])
    assert sorted(owned) == sorted(edges)                     # each edge in exactly one section
    return tk


def test_every_edge_drawn_once_synthetic(lay):
    tk = _coverage(lay)
    assert "regular polygon sides=6" in tk                    # the output hexagon
    assert "Not used" not in tk and lay["graph"]["not_used"] == ["noise"]
    for n in lay["graph"]["nodes"]:                           # no numeric node ids printed
        if n["kind"] != "input":
            assert f"{{{n['id']}}}" not in tk


def test_every_edge_drawn_once_heart_paper():
    layer = load_layer("heart-paper")
    if layer is None or "graph" not in layer:
        pytest.skip("heart-paper layer without a graph record")
    _coverage(layer)


def test_grouped_inputs_cover_their_edges():
    """churn100-s4: 'state' (51 one-hot levels) is one grouped input; each drawn grouped edge covers
    the underlying edges from that variable's columns into one target, and every edge is still
    drawn (overview) and owned by exactly one section."""
    layer = load_layer("churn100-s4")
    if layer is None or "graph" not in layer:
        pytest.skip("churn100-s4 layer without a graph record")
    from explaneat.export.render_structure import grouped_variables
    groups = grouped_variables(layer)
    assert {g["base"] for g in groups.values()} == {"state"}          # area code (3 levels) is not grouped
    _coverage(layer)
    tk, meta = overview(layer)
    nodes = {n["id"]: n for n in layer["graph"]["nodes"]}
    underlying = [e for e in layer["graph"]["edges"] if nodes[e["from"]].get("column") in groups]
    grouped = [d for d in meta["merged"] if isinstance(d[2], list)]
    assert sum(len(d[2]) for d in grouped) == len(underlying)        # grouped edges cover all of them
    assert len({d[1] for d in grouped}) == len(grouped)              # one drawn edge per target
    assert sum(1 for k in meta["inputs"] if k.startswith("grp::")) == 1
    assert "state (" in tk and "of 51 used)" in tk


def test_grouping_is_the_hierarchy(lay):
    root, part, cp = _parts(lay)
    blocks = summary_blocks(lay)
    assert [b["block"] for b in blocks] == [ADDITIVE, part["id"]]
    assert blocks[0]["parts"] == [root["id"]]                 # parts are never folded into the sum


def test_part_count_lint(lay):
    root, part, cp = _parts(lay)
    comp = _structure(lay, root["id"], "composition")         # root is made of 2 parts (the fold and its copy)
    assert lint_part_counts(lay, root["id"], "It is made of two parts.", [comp]) == []
    assert lint_part_counts(lay, root["id"], "It is made of three parts.", [comp])
    assert lint_part_counts(lay, part["id"], "It is made of two parts.", [])          # a leaf has none
    assert lint_part_counts(lay, root["id"], "Through one part, it rises.", []) == []  # names one, not a count
    d = _draft(lay)
    d["lead"][0]["text"] = "The model adds dose up directly and through three parts."
    assert any("hierarchy has 2 part(s)" in e for e in lint_draft(lay, d))
    d["lead"][0]["text"] = "The model adds dose up directly and through two parts."
    assert lint_draft(lay, d) == []
