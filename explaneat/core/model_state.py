"""
Model State Engine for managing the explained model.

The ModelStateEngine applies operations to the original phenotype to produce
the current model state. Operations are stored as an ordered list and can
be added, removed (undo), and replayed.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Tuple
from copy import deepcopy
import json

from .genome_network import (
    NetworkStructure,
    NetworkNode,
    NetworkConnection,
    NodeType,
)
from .operations import (
    apply_split_node,
    apply_consolidate_node,
    apply_remove_node,
    apply_add_node,
    apply_add_identity_node,
    validate_operation,
    OperationError,
)


@dataclass
class Operation:
    """Represents a single operation in the event stream."""

    seq: int
    type: str
    params: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "seq": self.seq,
            "type": self.type,
            "params": self.params,
            "result": self.result,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Operation":
        """Create from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        return cls(
            seq=data["seq"],
            type=data["type"],
            params=data["params"],
            result=data.get("result"),
            created_at=created_at or datetime.utcnow(),
        )


@dataclass
class AnnotationData:
    """Data for an annotation operation."""

    name: str
    hypothesis: str
    entry_nodes: List[str]
    exit_nodes: List[str]
    subgraph_nodes: List[str]
    subgraph_connections: List[Tuple[str, str]]
    evidence: Optional[Dict[str, Any]] = None


class ModelStateEngine:
    """
    Engine for managing model state through operations.

    The engine maintains:
    - The original phenotype (immutable)
    - A list of operations
    - The current model state (computed by replaying operations)
    - Annotation coverage (which nodes are immutable)
    """

    def __init__(self, phenotype: NetworkStructure):
        """
        Initialize the engine with an original phenotype.

        Args:
            phenotype: The original network structure (will be copied)
        """
        self._original_phenotype = deepcopy(phenotype)
        self._operations: List[Operation] = []
        self._current_state: Optional[NetworkStructure] = None
        self._annotations: List[AnnotationData] = []
        self._covered_nodes: Set[str] = set()
        self._covered_connections: Set[Tuple[str, str]] = set()

        # Invalidate cache
        self._state_valid = False

    @property
    def original_phenotype(self) -> NetworkStructure:
        """Get the original phenotype (read-only copy)."""
        return deepcopy(self._original_phenotype)

    @property
    def operations(self) -> List[Operation]:
        """Get the list of operations (read-only copy)."""
        return list(self._operations)

    @property
    def current_state(self) -> NetworkStructure:
        """Get the current model state after applying all operations."""
        if not self._state_valid:
            self._replay_operations()
        return deepcopy(self._current_state)

    @property
    def covered_nodes(self) -> Set[str]:
        """Get nodes covered by annotations (immutable)."""
        if not self._state_valid:
            self._replay_operations()
        return set(self._covered_nodes)

    @property
    def annotations(self) -> List[AnnotationData]:
        """Get all annotations."""
        if not self._state_valid:
            self._replay_operations()
        return list(self._annotations)

    def _replay_operations(self) -> None:
        """Replay all operations from the original phenotype."""
        self._current_state = deepcopy(self._original_phenotype)
        self._annotations = []
        self._covered_nodes = set()
        self._covered_connections = set()

        for op in self._operations:
            self._apply_operation_internal(op)

        self._current_state.metadata["is_original"] = len(self._operations) == 0
        self._state_valid = True

    def _apply_operation_internal(self, op: Operation) -> None:
        """Apply a single operation to the current state (internal use)."""
        if op.type == "split_node":
            result = apply_split_node(
                self._current_state,
                op.params["node_id"],
                self._covered_nodes,
            )
            op.result = result

        elif op.type == "consolidate_node":
            result = apply_consolidate_node(
                self._current_state,
                op.params["node_ids"],
                self._covered_nodes,
            )
            op.result = result

        elif op.type == "remove_node":
            result = apply_remove_node(
                self._current_state,
                op.params["node_id"],
                self._covered_nodes,
            )
            op.result = result

        elif op.type == "add_node":
            result = apply_add_node(
                self._current_state,
                tuple(op.params["connection"]),
                op.params["new_node_id"],
                self._covered_connections,
                bias=op.params.get("bias", 0.0),
                activation=op.params.get("activation", "identity"),
            )
            op.result = result

        elif op.type == "add_identity_node":
            connections = [tuple(c) for c in op.params["connections"]]
            result = apply_add_identity_node(
                self._current_state,
                op.params["target_node"],
                connections,
                op.params["new_node_id"],
                self._covered_connections,
            )
            op.result = result

        elif op.type == "annotate":
            # Annotations mark nodes as covered (immutable)
            annotation = AnnotationData(
                name=op.params["name"],
                hypothesis=op.params["hypothesis"],
                entry_nodes=op.params["entry_nodes"],
                exit_nodes=op.params["exit_nodes"],
                subgraph_nodes=op.params["subgraph_nodes"],
                subgraph_connections=[tuple(c) for c in op.params["subgraph_connections"]],
                evidence=op.params.get("evidence"),
            )
            self._annotations.append(annotation)
            self._covered_nodes.update(annotation.subgraph_nodes)
            self._covered_connections.update(annotation.subgraph_connections)
            op.result = {"annotation_index": len(self._annotations) - 1}

        else:
            raise OperationError(f"Unknown operation type: {op.type}")

    def add_operation(
        self,
        op_type: str,
        params: Dict[str, Any],
        validate: bool = True,
    ) -> Operation:
        """
        Add a new operation to the event stream.

        Args:
            op_type: Operation type (split_node, add_identity_node, etc.)
            params: Operation parameters
            validate: Whether to validate before applying

        Returns:
            The created operation with result

        Raises:
            OperationError: If validation fails or operation cannot be applied
        """
        # Ensure state is current
        if not self._state_valid:
            self._replay_operations()

        # Validate if requested
        if validate:
            errors = validate_operation(
                self._current_state,
                op_type,
                params,
                self._covered_nodes,
                self._covered_connections,
            )
            if errors:
                raise OperationError(f"Validation failed: {'; '.join(errors)}")

        # Create operation
        op = Operation(
            seq=len(self._operations),
            type=op_type,
            params=params,
        )

        # Apply operation
        self._apply_operation_internal(op)

        # Add to list
        self._operations.append(op)

        # Update metadata
        self._current_state.metadata["is_original"] = False

        return op

    def remove_operation(self, seq: int) -> List[Operation]:
        """
        Remove an operation and all subsequent operations (undo).

        Args:
            seq: Sequence number of operation to remove

        Returns:
            List of removed operations

        Raises:
            OperationError: If sequence number is invalid
        """
        if seq < 0 or seq >= len(self._operations):
            raise OperationError(f"Invalid sequence number: {seq}")

        # Remove operations from seq onwards
        removed = self._operations[seq:]
        self._operations = self._operations[:seq]

        # Invalidate state cache
        self._state_valid = False

        return removed

    def validate_operation(
        self,
        op_type: str,
        params: Dict[str, Any],
    ) -> List[str]:
        """
        Validate an operation without applying it.

        Args:
            op_type: Operation type
            params: Operation parameters

        Returns:
            List of error messages (empty if valid)
        """
        if not self._state_valid:
            self._replay_operations()

        return validate_operation(
            self._current_state,
            op_type,
            params,
            self._covered_nodes,
            self._covered_connections,
        )

    def can_modify_node(self, node_id: str) -> bool:
        """Check if a node can be modified (not covered by annotation)."""
        if not self._state_valid:
            self._replay_operations()
        return node_id not in self._covered_nodes

    def can_modify_connection(self, from_node: str, to_node: str) -> bool:
        """Check if a connection can be modified (not covered by annotation)."""
        if not self._state_valid:
            self._replay_operations()
        return (from_node, to_node) not in self._covered_connections

    def to_dict(self) -> Dict[str, Any]:
        """Serialize operations to dictionary."""
        return {
            "operations": [op.to_dict() for op in self._operations],
        }

    def load_operations(self, data: Dict[str, Any]) -> None:
        """
        Load operations from dictionary.

        Args:
            data: Dictionary with "operations" key containing operation list
        """
        self._operations = [
            Operation.from_dict(op_data)
            for op_data in data.get("operations", [])
        ]
        self._state_valid = False

    @classmethod
    def from_phenotype_and_operations(
        cls,
        phenotype: NetworkStructure,
        operations_data: Optional[Dict[str, Any]] = None,
    ) -> "ModelStateEngine":
        """
        Create engine from phenotype and optional saved operations.

        Args:
            phenotype: The original network structure
            operations_data: Optional dict with operations to load

        Returns:
            Configured ModelStateEngine
        """
        engine = cls(phenotype)
        if operations_data:
            engine.load_operations(operations_data)
        return engine
