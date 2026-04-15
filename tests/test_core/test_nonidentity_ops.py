"""Tests for non-identity operations: prune_node, prune_connection.

Also tests the identity/non-identity classification and ModelStateEngine
properties (has_non_identity_ops, last_identity_seq, get_state_at_seq).
"""
import pytest
from explaneat.core.genome_network import (
    NetworkNode,
    NetworkConnection,
    NetworkStructure,
    NodeType,
)
from explaneat.core.operations import (
    apply_prune_node,
    apply_prune_connection,
    apply_retrain,
    validate_operation,
    is_identity_op,
    IDENTITY_OPS,
    NON_IDENTITY_OPS,
    OperationError,
)
from explaneat.core.model_state import ModelStateEngine


def _simple_network():
    """A -> H1 -> H2 -> O, with B -> H1 and H1 -> O skip connection."""
    nodes = [
        NetworkNode(id="-1", type=NodeType.INPUT),
        NetworkNode(id="-2", type=NodeType.INPUT),
        NetworkNode(id="5", type=NodeType.HIDDEN, bias=0.5, activation="sigmoid"),
        NetworkNode(id="6", type=NodeType.HIDDEN, bias=0.3, activation="relu"),
        NetworkNode(id="0", type=NodeType.OUTPUT, bias=-0.1, activation="sigmoid"),
    ]
    connections = [
        NetworkConnection(from_node="-1", to_node="5", weight=1.0, enabled=True),
        NetworkConnection(from_node="-2", to_node="5", weight=0.5, enabled=True),
        NetworkConnection(from_node="5", to_node="6", weight=0.8, enabled=True),
        NetworkConnection(from_node="5", to_node="0", weight=0.3, enabled=True),  # skip
        NetworkConnection(from_node="6", to_node="0", weight=0.7, enabled=True),
    ]
    return NetworkStructure(
        nodes=nodes, connections=connections,
        input_node_ids=["-1", "-2"], output_node_ids=["0"],
    )


# =============================================================================
# Operation classification tests
# =============================================================================


class TestOperationClassification:
    def test_identity_ops(self):
        for op in ["split_node", "consolidate_node", "remove_node",
                    "add_node", "add_identity_node", "annotate",
                    "rename_node", "rename_annotation"]:
            assert is_identity_op(op), f"{op} should be identity"

    def test_non_identity_ops(self):
        for op in ["disable_connection", "enable_connection",
                    "prune_node", "prune_connection", "retrain"]:
            assert not is_identity_op(op), f"{op} should be non-identity"

    def test_unknown_op_is_not_identity(self):
        assert not is_identity_op("unknown_op_type")

    def test_sets_are_disjoint(self):
        assert IDENTITY_OPS & NON_IDENTITY_OPS == frozenset()


# =============================================================================
# prune_node tests
# =============================================================================


class TestApplyPruneNode:
    def test_prune_hidden_node_bypass(self):
        """Pruning node 6 (1 input, 1 output) bypasses it: 5->6->0 becomes 5->0."""
        model = _simple_network()
        # Node 6: input from 5 (weight 0.8), output to 0 (weight 0.7)
        assert model.get_node_by_id("6") is not None

        result = apply_prune_node(model, "6", set())

        assert model.get_node_by_id("6") is None
        assert result["removed_nodes"] == ["6"]
        assert result["removed_connections"] == [("5", "6"), ("6", "0")]
        assert result["created_connections"] == [("5", "0")]

        # Bypass connection: weight = 0.8 * 0.7 = 0.56
        bypass = [c for c in model.connections if c.from_node == "5" and c.to_node == "0"]
        # There's the original skip (0.3) and the new bypass (0.56)
        assert len(bypass) == 2
        weights = sorted(c.weight for c in bypass)
        assert abs(weights[0] - 0.3) < 1e-9   # original skip
        assert abs(weights[1] - 0.56) < 1e-9   # bypass

    def test_prune_node_multi_connection_raises(self):
        """Pruning node 5 (2 inputs, 2 outputs) raises OperationError."""
        model = _simple_network()
        with pytest.raises(OperationError, match="Prune requires exactly 1 input and 1 output"):
            apply_prune_node(model, "5", set())

    def test_prune_input_node_raises(self):
        model = _simple_network()
        with pytest.raises(OperationError, match="Cannot prune input"):
            apply_prune_node(model, "-1", set())

    def test_prune_output_node_raises(self):
        model = _simple_network()
        with pytest.raises(OperationError, match="Cannot prune output"):
            apply_prune_node(model, "0", set())

    def test_prune_covered_node_raises(self):
        model = _simple_network()
        with pytest.raises(OperationError, match="covered by annotation"):
            apply_prune_node(model, "6", {"6"})

    def test_prune_nonexistent_node_raises(self):
        model = _simple_network()
        with pytest.raises(OperationError, match="not found"):
            apply_prune_node(model, "999", set())

    def test_prune_node_preserves_other_nodes(self):
        """Pruning node 6 doesn't affect other nodes."""
        model = _simple_network()
        apply_prune_node(model, "6", set())
        assert model.get_node_by_id("-1") is not None
        assert model.get_node_by_id("-2") is not None
        assert model.get_node_by_id("5") is not None
        assert model.get_node_by_id("0") is not None


# =============================================================================
# prune_connection tests
# =============================================================================


class TestApplyPruneConnection:
    def test_prune_connection(self):
        """Pruning a connection permanently removes it."""
        model = _simple_network()
        n_before = len(model.connections)

        result = apply_prune_connection(model, "5", "0", set())

        assert len(model.connections) == n_before - 1
        assert result["removed_connections"] == [("5", "0")]
        # Connection should be gone
        assert all(
            not (c.from_node == "5" and c.to_node == "0")
            for c in model.connections
        )

    def test_prune_nonexistent_connection_raises(self):
        model = _simple_network()
        with pytest.raises(OperationError, match="not found"):
            apply_prune_connection(model, "-1", "0", set())

    def test_prune_covered_connection_raises(self):
        model = _simple_network()
        with pytest.raises(OperationError, match="covered by an annotation"):
            apply_prune_connection(model, "5", "0", {("5", "0")})

    def test_prune_leaves_other_connections(self):
        model = _simple_network()
        apply_prune_connection(model, "5", "0", set())
        # Other connections should be intact
        assert any(c.from_node == "-1" and c.to_node == "5" for c in model.connections)
        assert any(c.from_node == "5" and c.to_node == "6" for c in model.connections)
        assert any(c.from_node == "6" and c.to_node == "0" for c in model.connections)


# =============================================================================
# Validation tests
# =============================================================================


class TestPruneValidation:
    def test_validate_prune_node_valid(self):
        """Node 6 has exactly 1 input and 1 output — valid for pruning."""
        model = _simple_network()
        errors = validate_operation(model, "prune_node", {"node_id": "6"}, set(), set())
        assert errors == []

    def test_validate_prune_node_multi_connection(self):
        """Node 5 has 2 inputs and 2 outputs — validation should fail."""
        model = _simple_network()
        errors = validate_operation(model, "prune_node", {"node_id": "5"}, set(), set())
        assert any("Prune requires exactly 1 input and 1 output" in e for e in errors)

    def test_validate_prune_node_missing_id(self):
        model = _simple_network()
        errors = validate_operation(model, "prune_node", {}, set(), set())
        assert any("node_id is required" in e for e in errors)

    def test_validate_prune_node_nonexistent(self):
        model = _simple_network()
        errors = validate_operation(model, "prune_node", {"node_id": "999"}, set(), set())
        assert any("does not exist" in e for e in errors)

    def test_validate_prune_node_covered(self):
        model = _simple_network()
        errors = validate_operation(model, "prune_node", {"node_id": "6"}, {"6"}, set())
        assert any("covered by an annotation" in e for e in errors)

    def test_validate_prune_node_input(self):
        model = _simple_network()
        errors = validate_operation(model, "prune_node", {"node_id": "-1"}, set(), set())
        assert any("Cannot prune input" in e for e in errors)

    def test_validate_prune_connection_valid(self):
        model = _simple_network()
        errors = validate_operation(
            model, "prune_connection",
            {"from_node": "5", "to_node": "0"}, set(), set(),
        )
        assert errors == []

    def test_validate_prune_connection_nonexistent(self):
        model = _simple_network()
        errors = validate_operation(
            model, "prune_connection",
            {"from_node": "-1", "to_node": "0"}, set(), set(),
        )
        assert any("not found" in e for e in errors)

    def test_validate_prune_connection_covered(self):
        model = _simple_network()
        errors = validate_operation(
            model, "prune_connection",
            {"from_node": "5", "to_node": "0"},
            set(), {("5", "0")},
        )
        assert any("covered by an annotation" in e for e in errors)


# =============================================================================
# ModelStateEngine integration tests
# =============================================================================


class TestModelStateEngineNonIdentity:
    def test_has_non_identity_ops_false_initially(self):
        engine = ModelStateEngine(_simple_network())
        assert engine.has_non_identity_ops is False

    def test_has_non_identity_ops_after_identity_op(self):
        engine = ModelStateEngine(_simple_network())
        engine.add_operation("rename_node", {"node_id": "5", "display_name": "hidden1"})
        assert engine.has_non_identity_ops is False

    def test_has_non_identity_ops_after_prune(self):
        engine = ModelStateEngine(_simple_network())
        engine.add_operation("prune_node", {"node_id": "6"})
        assert engine.has_non_identity_ops is True

    def test_has_non_identity_ops_after_prune_connection(self):
        engine = ModelStateEngine(_simple_network())
        engine.add_operation("prune_connection", {"from_node": "5", "to_node": "0"})
        assert engine.has_non_identity_ops is True

    def test_last_identity_seq_no_nonidentity(self):
        engine = ModelStateEngine(_simple_network())
        engine.add_operation("rename_node", {"node_id": "5", "display_name": "hidden1"})
        assert engine.last_identity_seq is None

    def test_last_identity_seq_with_preceding_identity(self):
        engine = ModelStateEngine(_simple_network())
        engine.add_operation("rename_node", {"node_id": "5", "display_name": "h1"})  # seq 0
        engine.add_operation("rename_node", {"node_id": "6", "display_name": "h2"})  # seq 1
        engine.add_operation("prune_connection", {"from_node": "5", "to_node": "0"})  # seq 2
        assert engine.last_identity_seq == 1

    def test_last_identity_seq_first_op_nonidentity(self):
        engine = ModelStateEngine(_simple_network())
        engine.add_operation("prune_connection", {"from_node": "5", "to_node": "0"})
        assert engine.last_identity_seq is None

    def test_prune_node_via_engine(self):
        engine = ModelStateEngine(_simple_network())
        op = engine.add_operation("prune_node", {"node_id": "6"})
        state = engine.current_state
        assert state.get_node_by_id("6") is None
        # Node 5 and others should still exist
        assert state.get_node_by_id("5") is not None

    def test_prune_connection_via_engine(self):
        engine = ModelStateEngine(_simple_network())
        op = engine.add_operation("prune_connection", {"from_node": "5", "to_node": "0"})
        state = engine.current_state
        assert all(
            not (c.from_node == "5" and c.to_node == "0")
            for c in state.connections
        )


# =============================================================================
# get_state_at_seq tests
# =============================================================================


class TestGetStateAtSeq:
    def test_state_at_seq_0(self):
        engine = ModelStateEngine(_simple_network())
        engine.add_operation("rename_node", {"node_id": "5", "display_name": "h1"})  # seq 0
        engine.add_operation("prune_connection", {"from_node": "5", "to_node": "0"})  # seq 1

        state = engine.get_state_at_seq(0)
        # After seq 0 (rename), node 5 should have display name
        node5 = state.get_node_by_id("5")
        assert node5 is not None
        assert node5.display_name == "h1"
        # Skip connection should still exist
        assert any(c.from_node == "5" and c.to_node == "0" for c in state.connections)

    def test_state_at_seq_includes_all_up_to(self):
        engine = ModelStateEngine(_simple_network())
        engine.add_operation("rename_node", {"node_id": "5", "display_name": "h1"})  # seq 0
        engine.add_operation("prune_connection", {"from_node": "5", "to_node": "0"})  # seq 1

        state = engine.get_state_at_seq(1)
        # After seq 1 (prune), skip connection should be gone
        assert not any(c.from_node == "5" and c.to_node == "0" for c in state.connections)

    def test_state_at_seq_doesnt_affect_engine(self):
        engine = ModelStateEngine(_simple_network())
        engine.add_operation("prune_connection", {"from_node": "5", "to_node": "0"})

        # Getting an intermediate state shouldn't affect the engine's current state
        _ = engine.get_state_at_seq(0)
        current = engine.current_state
        # Prune should still be reflected in current state
        assert not any(c.from_node == "5" and c.to_node == "0" for c in current.connections)


# =============================================================================
# apply_retrain tests
# =============================================================================


class TestApplyRetrain:
    def test_retrain_updates_weights(self):
        model = _simple_network()
        result = apply_retrain(
            model,
            weight_updates={("-1", "5"): 2.0, ("5", "6"): -0.5},
            bias_updates={},
        )
        conn = next(c for c in model.connections if c.from_node == "-1" and c.to_node == "5")
        assert conn.weight == 2.0
        conn2 = next(c for c in model.connections if c.from_node == "5" and c.to_node == "6")
        assert conn2.weight == -0.5
        assert result["weights_updated"] == 2

    def test_retrain_updates_biases(self):
        model = _simple_network()
        result = apply_retrain(
            model,
            weight_updates={},
            bias_updates={"5": 1.5, "0": -2.0},
        )
        node5 = model.get_node_by_id("5")
        assert node5.bias == 1.5
        node0 = model.get_node_by_id("0")
        assert node0.bias == -2.0
        assert result["biases_updated"] == 2

    def test_retrain_via_engine(self):
        engine = ModelStateEngine(_simple_network())
        engine.add_operation("retrain", {
            "weight_updates": {"-1,5": 2.0},
            "bias_updates": {"5": 1.5},
        }, validate=False)
        assert engine.has_non_identity_ops is True
