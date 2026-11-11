"""Split-node ids past z (``a``..``z``, then ``Aa``..``Zz``), function preservation of wide splits,
consolidation and re-splitting with the new tokens, and validate_operation never calling a split
valid that the handler would refuse."""

import numpy as np
import pytest
import torch

from explaneat.core import split_ids
from explaneat.core.genome_network import NetworkConnection, NetworkNode, NetworkStructure, NodeType
from explaneat.core.operations import (
    OperationError,
    _get_base_node_id,
    _get_split_suffix,
    apply_consolidate_node,
    apply_split_node,
    validate_operation,
)
from explaneat.core.structure_network import StructureNetwork


def wide_net(n_inputs=3, width=40, seed=0):
    """Dense inputs -> `width` ReLU units -> sigmoid output (every input fans out `width` ways)."""
    r = np.random.default_rng(seed)
    ins = [str(-(i + 1)) for i in range(n_inputs)]
    hid = [str(i + 1) for i in range(width)]
    nodes = [NetworkNode(id=i, type=NodeType.INPUT) for i in ins]
    nodes += [NetworkNode(id=h, type=NodeType.HIDDEN, bias=float(r.normal()), activation="relu",
                          response=1.0, aggregation="sum") for h in hid]
    nodes.append(NetworkNode(id="0", type=NodeType.OUTPUT, bias=0.1, activation="sigmoid", response=1.0,
                             aggregation="sum"))
    conns = [NetworkConnection(i, h, float(r.normal()), True) for i in ins for h in hid]
    conns += [NetworkConnection(h, "0", float(r.normal()), True) for h in hid]
    return NetworkStructure(nodes=nodes, connections=conns, input_node_ids=ins, output_node_ids=["0"])


def forward(structure, X):
    return StructureNetwork(structure).forward(torch.as_tensor(X)).detach().numpy().ravel()


class TestTokens:
    def test_first_26_unchanged(self):
        assert [split_ids.token(i) for i in range(26)] == list("abcdefghijklmnopqrstuvwxyz")

    def test_past_z(self):
        assert split_ids.token(26) == "Aa"
        assert split_ids.token(27) == "Ab"
        assert split_ids.token(52) == "Ba"
        assert split_ids.token(split_ids.MAX_TOKENS - 1) == "Zz"
        with pytest.raises(ValueError):
            split_ids.token(split_ids.MAX_TOKENS)

    def test_round_trip(self):
        for i in range(split_ids.MAX_TOKENS):
            assert split_ids.token_index(split_ids.token(i)) == i

    def test_tokenize_and_join(self):
        assert split_ids.tokens("aAbc") == ["a", "Ab", "c"]
        assert split_ids.join(["Ab", "c", "a"]) == "acAb"
        # legacy consolidation names: sorted letters, unchanged
        assert split_ids.join(list("ca")) == "ac"
        for bad in ("A", "aB", "AB", "a1", ""):
            assert not split_ids.is_suffix(bad)

    def test_base_and_suffix(self):
        assert _get_base_node_id("5_Ab") == "5"
        assert _get_split_suffix("-20_Zz") == "Zz"
        assert _get_base_node_id("5_ac") == "5"
        assert _get_base_node_id("5") == "5"
        assert _get_base_node_id("identity_5") is None
        assert split_ids.strip("-3_aAb") == "-3"


class TestWideSplit:
    def test_legacy_ids_for_26_or_fewer(self):
        net = wide_net(width=26)
        res = apply_split_node(net, "-1", set())
        assert sorted(res["created_nodes"]) == sorted(f"-1_{c}" for c in "abcdefghijklmnopqrstuvwxyz")

    def test_split_past_z_preserves_function(self):
        net = wide_net(width=40)
        X = np.random.default_rng(1).normal(size=(50, 3))
        before = forward(net, X)
        for i in ("-1", "-2", "-3"):
            assert validate_operation(net, "split_node", {"node_id": i}, set(), set()) == []
            res = apply_split_node(net, i, set())
            assert len(res["created_nodes"]) == 40
        ids = {n.id for n in net.nodes}
        assert {"-1_a", "-1_z", "-1_Aa", "-1_An"} <= ids
        assert "-1_Ao" not in ids
        np.testing.assert_allclose(forward(net, X), before, atol=1e-12)

    def test_consolidate_and_resplit_new_tokens(self):
        net = wide_net(width=30)
        X = np.random.default_rng(2).normal(size=(20, 3))
        before = forward(net, X)
        apply_split_node(net, "-1", set())
        res = apply_consolidate_node(net, ["-1_Ab", "-1_c", "-1_z"], set())
        cid = res["created_nodes"][0] if "created_nodes" in res else "-1_czAb"
        assert cid == "-1_czAb"
        assert _get_base_node_id(cid) == "-1"
        np.testing.assert_allclose(forward(net, X), before, atol=1e-12)
        res = apply_split_node(net, cid, set())
        assert sorted(res["created_nodes"]) == sorted(["-1_c", "-1_z", "-1_Ab"])
        np.testing.assert_allclose(forward(net, X), before, atol=1e-12)


class TestValidateNeverLies:
    def test_split_that_cannot_execute_is_invalid(self, monkeypatch):
        monkeypatch.setattr(split_ids, "MAX_TOKENS", 5)
        net = wide_net(width=6)
        with pytest.raises(OperationError):
            apply_split_node(wide_net(width=6), "-1", set())
        errs = validate_operation(net, "split_node", {"node_id": "-1"}, set(), set())
        assert errs and "Too many splits" in errs[0]

    def test_valid_split_still_valid(self):
        assert validate_operation(wide_net(width=6), "split_node", {"node_id": "-1"}, set(), set()) == []


class TestDecompositionWarning:
    def _ann(self, name, nodes, entries, parent=None):
        from explaneat.core.model_state import AnnotationData
        return AnnotationData(name=name, hypothesis="", entry_nodes=entries, exit_nodes=[nodes[-1]],
                              subgraph_nodes=nodes, subgraph_connections=[], parent_annotation_id=parent)

    def test_one_leaf_swallowing_a_wide_layer_is_flagged(self):
        from explaneat.analysis.coverage_report import decomposition_report
        net = wide_net(width=32)
        hid = [str(i + 1) for i in range(32)]
        anns = [self._ann("all", ["-1", "-2", "-3"] + hid, ["-1", "-2", "-3"], parent="root"),
                self._ann("root", ["0"], hid)]
        d = decomposition_report(net, anns)
        assert d["granularity"] == 0.0 and d["largest_part"]["hidden_units"] == 32
        assert any("coarse decomposition" in w for w in d["warnings"])

    def test_fine_decomposition_is_not_flagged(self):
        from explaneat.analysis.coverage_report import decomposition_report
        net = wide_net(width=8)
        anns = [self._ann(f"u{i}", ["-1", str(i)], ["-1"], parent="root") for i in range(1, 9)]
        anns.append(self._ann("root", ["0"], [str(i) for i in range(1, 9)]))
        d = decomposition_report(net, anns)
        assert d["granularity"] == 1.0 and d["warnings"] == []

    def test_units_only_as_entries_are_flagged(self):
        from explaneat.analysis.coverage_report import decomposition_report
        net = wide_net(width=8)
        anns = [self._ann("root", ["0"], [str(i) for i in range(1, 9)])]
        d = decomposition_report(net, anns)
        assert d["units_only_as_entries"] == 8 and d["warnings"]
