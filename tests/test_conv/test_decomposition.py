"""Conv graph decomposition: model state, operation stream, validation, strategy, hierarchy, coverage."""

import numpy as np
import pytest
import torch

from explaneat.analysis.collapse_validator import CollapseAnnotation, CollapseValidator
from explaneat.conv.decomposition import (coverage_summary, decompose, function_preserved, repair_candidates)
from explaneat.conv.genome import ConnGene, ConvGenome, ConvNeatConfig, NodeGene
from explaneat.conv.model_state import ConvModelStateEngine, load_record, make_record
from explaneat.conv.network import NaiveConvNet
from explaneat.conv.operations import (annotation_strategy, collapse_graph, execute_strategy, final_annotation)
from explaneat.core.operations import OperationError

torch.set_num_threads(1)


def _genome(edges, levels, seed=0, n_out=2, hw=(8, 8)):
    """A hand-built genome. ``edges``: (src, dst, k); hidden ids >= 2, outputs 0..n_out-1, input -1."""
    rng = np.random.default_rng(seed)
    cfg = ConvNeatConfig(in_channels=1, image_hw=hw, n_classes=n_out, max_level=2)
    g = ConvGenome(key=0)
    g.nodes[-1] = NodeGene(-1, "input", 0, 0.0)
    for o in range(n_out):
        g.nodes[o] = NodeGene(o, "output", 0, float(rng.normal() * 0.1))
    for n, lv in levels.items():
        g.nodes[n] = NodeGene(n, "hidden", lv, float(rng.normal() * 0.2))
    for i, (a, b, k) in enumerate(edges):
        g.conns[i] = ConnGene(i, a, b, (rng.standard_normal((k, k)) * 0.5).astype(np.float32))
    g.head_w = rng.standard_normal((n_out, n_out)).astype(np.float32)
    g.head_b = rng.standard_normal(n_out).astype(np.float32)
    return g, cfg


# x -> a, x -> b, a -> c, b -> c, c -> e, e -> 0, a -> 1, b -> 1, x -> 1
# a is shared (feeds c inside a structure and the score map 1 outside); c mixes a and b.
EDGES = [(-1, 2, 3), (-1, 3, 5), (2, 4, 3), (3, 4, 1), (4, 5, 3), (5, 0, 3), (2, 1, 1), (3, 1, 3), (-1, 1, 3)]
LEVELS = {2: 0, 3: 1, 4: 1, 5: 2}


def _x(n=16, hw=(8, 8), seed=1):
    return torch.as_tensor(np.random.default_rng(seed).random((n, 1, *hw)), dtype=torch.float64)


def _logits(eng, X):
    return eng.state.forward(X)[1]


def _ref(g, cfg, X):
    with torch.no_grad():
        return NaiveConvNet(g, cfg, dtype=torch.float64)(X)


def test_state_matches_the_genome():
    g, cfg = _genome(EDGES, LEVELS)
    eng = ConvModelStateEngine.from_genome(g, cfg)
    X = _x()
    assert float((_logits(eng, X) - _ref(g, cfg, X)).abs().max()) < 1e-12
    assert set(eng.state.node_ids()) == {"-1", "0", "1", "2", "3", "4", "5"}


def test_split_and_identity_preserve_the_function():
    g, cfg = _genome(EDGES, LEVELS)
    eng = ConvModelStateEngine.from_genome(g, cfg)
    X = _x()
    ref = _ref(g, cfg, X)
    o = eng.add_operation("split_node", {"node_id": "2", "partition": [["4"], ["1"]]})
    assert sorted(o.result["created_nodes"]) == ["2_a", "2_b"]
    assert float((_logits(eng, X) - ref).abs().max()) < 1e-12
    # target-side identity (ExplaNEAT's form): intercept b -> c (c also gets 2_a -> c)
    eng.add_operation("add_identity_node", {"side": "target", "target_node": "4", "connections": [["3", "4"]],
                                            "new_node_id": "i1"})
    assert eng.state.level("i1") == eng.state.level("4")
    assert float((_logits(eng, X) - ref).abs().max()) < 1e-12
    # source-side identity (a port): re-root x -> b and x -> 1 at a copy of x
    eng.add_operation("add_identity_node", {"side": "source", "connections": [["-1", "3"], ["-1", "1"]],
                                            "new_node_id": "p1"})
    assert ("-1", "p1") in eng.state.edges() and ("p1", "3") in eng.state.edges()
    assert float((_logits(eng, X) - ref).abs().max()) < 1e-12
    # consolidate the split back
    eng.add_operation("consolidate_node", {"node_ids": ["2_a", "2_b"]})
    assert "2_ab" in eng.state.node_ids()
    assert float((_logits(eng, X) - ref).abs().max()) < 1e-12


def _annotate(eng, nodes, entries, exits, name="A"):
    st = eng.state
    conns = [[a, b] for a, b in st.edges() if a in nodes and b in nodes]
    return eng.validate_operation("annotate", {"name": name, "hypothesis": "h", "entry_nodes": entries,
                                               "exit_nodes": exits, "subgraph_nodes": nodes,
                                               "subgraph_connections": conns})


def test_validator_rejects_each_precondition_with_a_clear_message():
    g, cfg = _genome(EDGES, LEVELS)
    eng = ConvModelStateEngine.from_genome(g, cfg)
    # P1: intermediate c has an external input from b
    e1 = _annotate(eng, ["2", "4", "5"], ["2"], ["5"])
    assert any(m.startswith("P1 violation") and "4" in m and "3" in m for m in e1), e1
    # P2: entry a sends an edge to the score map 1, outside the selection
    assert any(m.startswith("P2 violation") and "node 2" in m and "1" in m for m in e1), e1
    # P3: exit c has an external input from a
    e3 = _annotate(eng, ["3", "4"], ["3"], ["4"])
    assert any(m.startswith("P3 violation") and "exit node 4" in m and "2" in m for m in e3), e3
    # a clean selection passes once the strategy has run (identity -> splits)
    r = execute_strategy(eng, {"2", "3", "4", "5"}, "clean")
    assert r["annotation"]["exit_nodes"] and eng.annotations[-1].name == "clean"
    with pytest.raises(OperationError):
        eng.add_operation("annotate", {"name": "bad", "hypothesis": "", "entry_nodes": ["1"], "exit_nodes": ["1"],
                                       "subgraph_nodes": ["1", "4"], "subgraph_connections": []})


def test_collapse_cycle_is_reported():
    # x -> a -> w -> c, a -> c, c -> 0, w -> 1: the selection {a, c} is not convex (w re-enters)
    g, cfg = _genome([(-1, 2, 3), (2, 3, 3), (3, 4, 3), (2, 4, 1), (4, 0, 3), (3, 1, 3)], {2: 0, 3: 0, 4: 1})
    eng = ConvModelStateEngine.from_genome(g, cfg)
    st = annotation_strategy(eng.state, {"2", "4"})
    assert [b["node"] for b in st["blocking"]] == ["3"]
    assert "re-enters" in st["blocking"][0]["message"]
    with pytest.raises(OperationError, match="blocking"):
        execute_strategy(eng, {"2", "4"}, "A")
    # the repair merges the blocking node in; then it is collapsible
    groups, log = repair_candidates(eng.state, [{"2", "4"}])
    assert groups == [{"2", "3", "4"}] and log[0]["node"] == "3"
    execute_strategy(eng, groups[0], "A")


def _collapsible(eng, name):
    a = eng.annotation(name)
    nodes = eng.expanded_nodes(name)
    graph = collapse_graph(eng.state)
    ann = CollapseAnnotation.from_sets(name, set(a.entry_nodes), set(a.exit_nodes), nodes,
                                       {e for e in graph.edges if e[0] in nodes and e[1] in nodes})
    return CollapseValidator.validate_collapsible(graph, ann)


def test_strategy_makes_shared_nodes_collapsible():
    g, cfg = _genome(EDGES, LEVELS)
    eng = ConvModelStateEngine.from_genome(g, cfg)
    X = _x()
    ref = _ref(g, cfg, X)
    s = annotation_strategy(eng.state, {"2", "4", "5"})
    srcs = sorted(op["params"]["connections"][0][0] for op in s["identity_ops"])
    assert srcs == ["-1", "3"]  # the image and b feed the selection from outside
    execute_strategy(eng, {"2", "4", "5"}, "A")
    execute_strategy(eng, {"3"}, "B")
    for nm in ("A", "B"):
        assert _collapsible(eng, nm).is_valid
    a = eng.annotation("A")
    assert all(eng.state.meta[e]["identity"] == "source" for e in a.entry_nodes)  # entries are ports
    assert any(n.startswith("2_") for n in a.exit_nodes)  # the shared map was split, its outside copy is an exit
    assert float((_logits(eng, X) - ref).abs().max()) < 1e-12


def test_ui_mode_identity_and_blocking():
    g, cfg = _genome(EDGES, LEVELS)
    eng = ConvModelStateEngine.from_genome(g, cfg)
    # exit c has inputs from a (inside) and b (outside): one identity node replaces the exit
    s = annotation_strategy(eng.state, {"2", "4"}, mode="ui")
    assert len(s["identity_ops"]) == 1 and s["identity_ops"][0]["params"]["replaces_exit"] == "4"
    assert not s["blocking"]
    # intermediate c with an external input: blocking (the UI says expand the selection)
    s2 = annotation_strategy(eng.state, {"2", "4", "5"}, mode="ui")
    assert any(b["node"] == "4" and b["kind"] == "external_input" for b in s2["blocking"])
    X = _x()
    execute_strategy(eng, {"2", "4"}, "U", mode="ui")
    assert _collapsible(eng, "U").is_valid
    assert float((_logits(eng, X) - _ref(g, cfg, X)).abs().max()) < 1e-12


def test_undo():
    g, cfg = _genome(EDGES, LEVELS)
    eng = ConvModelStateEngine.from_genome(g, cfg)
    before = (sorted(eng.state.node_ids()), sorted(eng.state.edges()))
    execute_strategy(eng, {"2", "4", "5"}, "A")
    n = len(eng.operations)
    assert n >= 3 and eng.covered_nodes
    removed = eng.remove_operation(0)
    assert len(removed) == n and not eng.operations
    assert (sorted(eng.state.node_ids()), sorted(eng.state.edges())) == before
    assert not eng.covered_nodes and not eng.annotations
    execute_strategy(eng, {"2", "4", "5"}, "A")
    last = len(eng.operations) - 1
    eng.remove_operation(last)  # undo just the annotation
    assert not eng.annotations and len(eng.operations) == last


def _hierarchy_candidates():
    return {"structures": [["2", "4", "5"], ["3"]], "subparts": [[["2", "4"], ["5"]], [["3"]]],
            "flow": {"node_abs": {}}, "merge_log": [], "inactive": [], "method": "hand-made"}


def test_replay_is_exact_and_the_hierarchy_is_valid(tmp_path):
    g, cfg = _genome(EDGES, LEVELS, seed=3)
    X = np.random.default_rng(2).random((40, 1, 8, 8)).astype(np.float32)
    eng, summ = decompose(g, cfg, X, np.zeros(40, int), X, "synthetic", candidates=_hierarchy_candidates(),
                          hypotheses=False, log=lambda *a: None)
    assert summ["blocking_left"] == [] and summ["function_preserved_max_abs_logit_diff"] < 1e-12
    names = {a.name: a for a in eng.annotations}
    # b alone feeds the other structure, so it comes first (structures are in topological order)
    assert set(names) == {"S1", "S2.1", "S2.2", "S2", "root"}
    assert names["S2.1"].parent_annotation_id == "S2" and names["S2"].parent_annotation_id == "root"
    assert names["S1"].parent_annotation_id == "root"
    for nm in names:
        assert _collapsible(eng, nm).is_valid, nm
    after = summ["boundaries"]["after_structures"]
    assert after["unclean_crossing_edges"] == 0 and after["collapsible_groups"] == 2
    assert summ["coverage"]["compositional"] == 1.0
    # the record replays on its own, with validation, to the same function and annotations
    rec = make_record(eng, g, cfg, "synthetic", summary=summ)
    p = tmp_path / "rec.json"
    import json
    p.write_text(json.dumps(rec))
    eng2, g2, cfg2, _ = load_record(str(p), validate=True)
    Xt = torch.as_tensor(X, dtype=torch.float64)
    assert float((_logits(eng2, Xt) - _logits(eng, Xt)).abs().max()) == 0.0
    assert float((_logits(eng2, Xt) - _ref(g, cfg, Xt)).abs().max()) < 1e-12
    assert [a.name for a in eng2.annotations] == [a.name for a in eng.annotations]
    assert function_preserved(eng2, g2, cfg2, X) < 1e-12


def test_coverage_matches_hand_computation():
    # x -> a -> 0, x -> b -> 0 (one score map); leaves {port(x), a} and {port(x), b}, root on top
    g, cfg = _genome([(-1, 2, 3), (-1, 3, 3), (2, 0, 3), (3, 0, 1)], {2: 0, 3: 1}, n_out=1)
    eng = ConvModelStateEngine.from_genome(g, cfg)
    execute_strategy(eng, {"2"}, "L1")
    execute_strategy(eng, {"3"}, "L2")
    cov = coverage_summary(eng)
    # non-output nodes: x, p1, a, p2, b; leaves cover the two ports only (a and b feed the root's edges)
    assert cov["structural_leaves_only"] == pytest.approx(2 / 5)
    assert cov["compositional"] == 0.0  # two roots, neither holds the score map
    st = eng.state
    ann_edges = {tuple(c) for a in eng.annotations for c in a.subgraph_connections}
    eng.add_operation("annotate", {"name": "root", "hypothesis": "", "entry_nodes": ["-1"], "exit_nodes": ["0"],
                                   "subgraph_nodes": ["-1", "0"],
                                   "subgraph_connections": [[a, b] for a, b in st.edges() if (a, b) not in ann_edges],
                                   "child_annotation_ids": ["L1", "L2"]})
    cov = coverage_summary(eng)
    assert cov["structural_all_annotations"] == 1.0
    assert cov["compositional"] == 1.0 and cov["global_root"] == "root"
    assert final_annotation(eng.state, {"2"})["exit_nodes"] == ["2"]


def test_groups_that_feed_each_other_are_repaired():
    # a -> c, d -> b: {a, b} and {c, d} are each convex, but a -> c and d -> b make the
    # collapsed pair a cycle; the repair moves one map (or merges) until the quotient is a DAG
    g, cfg = _genome([(-1, 2, 3), (-1, 5, 3), (2, 4, 3), (5, 3, 3), (4, 0, 1), (3, 1, 1), (2, 0, 1), (5, 1, 1)],
                     {2: 0, 3: 0, 4: 0, 5: 0})
    eng = ConvModelStateEngine.from_genome(g, cfg)
    groups, log = repair_candidates(eng.state, [{"2", "3"}, {"4", "5"}])
    from explaneat.conv.decomposition import _quotient_cycle
    assert set().union(*groups) == {"2", "3", "4", "5"} and _quotient_cycle(eng.state, groups) is None
    assert log and ("fed each other" in log[0]["fix"] or "cycle" in log[0]["fix"])
