"""Tests for disable_connection and enable_connection operation handlers."""
import pytest
from explaneat.core.genome_network import (
    NetworkNode,
    NetworkConnection,
    NetworkStructure,
    NodeType,
)
from explaneat.core.operations import (
    apply_disable_connection,
    apply_enable_connection,
    validate_operation,
    OperationError,
)


def _simple_network():
    nodes = [
        NetworkNode(id="-1", type=NodeType.INPUT),
        NetworkNode(id="-2", type=NodeType.INPUT),
        NetworkNode(id="5", type=NodeType.HIDDEN, bias=0.5, activation="sigmoid"),
        NetworkNode(id="0", type=NodeType.OUTPUT, bias=-0.1, activation="sigmoid"),
    ]
    connections = [
        NetworkConnection(from_node="-1", to_node="5", weight=1.0, enabled=True),
        NetworkConnection(from_node="-2", to_node="5", weight=0.5, enabled=True),
        NetworkConnection(from_node="5", to_node="0", weight=0.8, enabled=True),
    ]
    return NetworkStructure(
        nodes=nodes, connections=connections,
        input_node_ids=["-1", "-2"], output_node_ids=["0"],
    )


# =============================================================================
# Tests for apply_disable_connection
# =============================================================================

class TestApplyDisableConnection:
    def test_disable_existing_connection(self):
        """Disabling an existing enabled connection sets enabled=False and returns from/to/previous_weight."""
        model = _simple_network()
        result = apply_disable_connection(model, "-1", "5", set())

        # Connection should now be disabled
        conn = next(c for c in model.connections if c.from_node == "-1" and c.to_node == "5")
        assert conn.enabled is False

        # Result should contain from_node, to_node, and previous_weight
        assert result["from_node"] == "-1"
        assert result["to_node"] == "5"
        assert result["previous_weight"] == 1.0

    def test_disable_nonexistent_connection_raises(self):
        """Disabling a connection that does not exist raises OperationError with 'not found'."""
        model = _simple_network()
        with pytest.raises(OperationError, match="not found"):
            apply_disable_connection(model, "-1", "0", set())

    def test_disable_covered_connection_raises(self):
        """Disabling a covered connection raises OperationError with 'covered'."""
        model = _simple_network()
        covered = {("-1", "5")}
        with pytest.raises(OperationError, match="covered"):
            apply_disable_connection(model, "-1", "5", covered)

    def test_disable_already_disabled_raises(self):
        """Disabling a connection that is already disabled raises OperationError with 'already disabled'."""
        model = _simple_network()
        # Disable once — should succeed
        apply_disable_connection(model, "-1", "5", set())
        # Disable again — should raise
        with pytest.raises(OperationError, match="already disabled"):
            apply_disable_connection(model, "-1", "5", set())

    def test_disable_returns_correct_weight_for_other_connection(self):
        """Result previous_weight reflects the actual weight of the disabled connection."""
        model = _simple_network()
        result = apply_disable_connection(model, "-2", "5", set())
        assert result["from_node"] == "-2"
        assert result["to_node"] == "5"
        assert result["previous_weight"] == 0.5

    def test_disable_does_not_remove_connection(self):
        """Disabling a connection sets enabled=False but does not remove it from the list."""
        model = _simple_network()
        original_count = len(model.connections)
        apply_disable_connection(model, "-1", "5", set())
        assert len(model.connections) == original_count

    def test_disable_does_not_affect_other_connections(self):
        """Disabling one connection leaves other connections unchanged."""
        model = _simple_network()
        apply_disable_connection(model, "-1", "5", set())
        # Other connections remain enabled
        for c in model.connections:
            if c.from_node == "-2" and c.to_node == "5":
                assert c.enabled is True
            if c.from_node == "5" and c.to_node == "0":
                assert c.enabled is True


# =============================================================================
# Tests for apply_enable_connection
# =============================================================================

class TestApplyEnableConnection:
    def test_enable_disabled_connection(self):
        """Re-enabling a disabled connection sets enabled=True and returns from/to."""
        model = _simple_network()
        # First disable it
        apply_disable_connection(model, "-1", "5", set())

        result = apply_enable_connection(model, "-1", "5", set())

        # Connection should now be enabled again
        conn = next(c for c in model.connections if c.from_node == "-1" and c.to_node == "5")
        assert conn.enabled is True

        # Result should contain from_node and to_node
        assert result["from_node"] == "-1"
        assert result["to_node"] == "5"

    def test_enable_already_enabled_raises(self):
        """Enabling an already-enabled connection raises OperationError with 'already enabled'."""
        model = _simple_network()
        with pytest.raises(OperationError, match="already enabled"):
            apply_enable_connection(model, "-1", "5", set())

    def test_enable_nonexistent_connection_raises(self):
        """Enabling a non-existent connection raises OperationError with 'not found'."""
        model = _simple_network()
        with pytest.raises(OperationError, match="not found"):
            apply_enable_connection(model, "-1", "0", set())

    def test_enable_covered_connection_raises(self):
        """Enabling a covered connection raises OperationError with 'covered'."""
        model = _simple_network()
        # Disable it first so the state check doesn't trigger before the covered check
        # (covered check should fire regardless of enabled state)
        covered = {("-1", "5")}
        # Even if it were disabled, the covered check fires
        with pytest.raises(OperationError, match="covered"):
            apply_enable_connection(model, "-1", "5", covered)

    def test_enable_does_not_remove_connection(self):
        """Enabling a connection does not change the total count of connections."""
        model = _simple_network()
        apply_disable_connection(model, "-1", "5", set())
        original_count = len(model.connections)
        apply_enable_connection(model, "-1", "5", set())
        assert len(model.connections) == original_count


# =============================================================================
# Tests for validate_operation — disable_connection
# =============================================================================

class TestValidateDisableConnection:
    def test_validate_passes_for_valid_connection(self):
        model = _simple_network()
        errors = validate_operation(
            model, "disable_connection",
            {"from_node": "-1", "to_node": "5"},
            set(), set(),
        )
        assert errors == []

    def test_validate_fails_for_missing_from_node_param(self):
        model = _simple_network()
        errors = validate_operation(
            model, "disable_connection",
            {"to_node": "5"},
            set(), set(),
        )
        assert any("from_node" in e for e in errors)

    def test_validate_fails_for_missing_to_node_param(self):
        model = _simple_network()
        errors = validate_operation(
            model, "disable_connection",
            {"from_node": "-1"},
            set(), set(),
        )
        assert any("to_node" in e for e in errors)

    def test_validate_fails_for_nonexistent_connection(self):
        model = _simple_network()
        errors = validate_operation(
            model, "disable_connection",
            {"from_node": "-1", "to_node": "0"},
            set(), set(),
        )
        assert len(errors) > 0

    def test_validate_fails_for_covered_connection(self):
        model = _simple_network()
        errors = validate_operation(
            model, "disable_connection",
            {"from_node": "-1", "to_node": "5"},
            set(), {("-1", "5")},
        )
        assert len(errors) > 0

    def test_validate_fails_for_already_disabled_connection(self):
        model = _simple_network()
        # Manually disable the connection
        conn = next(c for c in model.connections if c.from_node == "-1" and c.to_node == "5")
        conn.enabled = False
        errors = validate_operation(
            model, "disable_connection",
            {"from_node": "-1", "to_node": "5"},
            set(), set(),
        )
        assert len(errors) > 0


# =============================================================================
# Tests for validate_operation — enable_connection
# =============================================================================

class TestValidateEnableConnection:
    def test_validate_passes_for_disabled_connection(self):
        model = _simple_network()
        # Manually disable the connection
        conn = next(c for c in model.connections if c.from_node == "-1" and c.to_node == "5")
        conn.enabled = False
        errors = validate_operation(
            model, "enable_connection",
            {"from_node": "-1", "to_node": "5"},
            set(), set(),
        )
        assert errors == []

    def test_validate_fails_for_already_enabled_connection(self):
        model = _simple_network()
        errors = validate_operation(
            model, "enable_connection",
            {"from_node": "-1", "to_node": "5"},
            set(), set(),
        )
        assert len(errors) > 0

    def test_validate_fails_for_nonexistent_connection(self):
        model = _simple_network()
        errors = validate_operation(
            model, "enable_connection",
            {"from_node": "-1", "to_node": "0"},
            set(), set(),
        )
        assert len(errors) > 0

    def test_validate_fails_for_covered_connection(self):
        model = _simple_network()
        conn = next(c for c in model.connections if c.from_node == "-1" and c.to_node == "5")
        conn.enabled = False
        errors = validate_operation(
            model, "enable_connection",
            {"from_node": "-1", "to_node": "5"},
            set(), {("-1", "5")},
        )
        assert len(errors) > 0


# =============================================================================
# Tests for ModelStateEngine integration
# =============================================================================

from explaneat.core.model_state import ModelStateEngine


class TestConnectionOpsInEngine:
    def test_disable_via_engine(self):
        net = _simple_network()
        engine = ModelStateEngine(net)
        op = engine.add_operation("disable_connection", {"from_node": "-1", "to_node": "5"})
        assert op.type == "disable_connection"
        conn = [c for c in engine.current_state.connections
                if c.from_node == "-1" and c.to_node == "5"][0]
        assert conn.enabled is False

    def test_undo_disable_re_enables(self):
        net = _simple_network()
        engine = ModelStateEngine(net)
        op = engine.add_operation("disable_connection", {"from_node": "-1", "to_node": "5"})
        engine.remove_operation(op.seq)
        conn = [c for c in engine.current_state.connections
                if c.from_node == "-1" and c.to_node == "5"][0]
        assert conn.enabled is True

    def test_enable_via_engine(self):
        net = _simple_network()
        engine = ModelStateEngine(net)
        engine.add_operation("disable_connection", {"from_node": "-1", "to_node": "5"})
        engine.add_operation("enable_connection", {"from_node": "-1", "to_node": "5"})
        conn = [c for c in engine.current_state.connections
                if c.from_node == "-1" and c.to_node == "5"][0]
        assert conn.enabled is True

    def test_disable_serialization_round_trip(self):
        net = _simple_network()
        engine = ModelStateEngine(net)
        engine.add_operation("disable_connection", {"from_node": "-1", "to_node": "5"})
        data = engine.to_dict()

        engine2 = ModelStateEngine(_simple_network())
        engine2.load_operations(data)
        conn = [c for c in engine2.current_state.connections
                if c.from_node == "-1" and c.to_node == "5"][0]
        assert conn.enabled is False
