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
    apply_rename_node,
    apply_prune_node,
    apply_prune_connection,
    apply_retrain,
    validate_operation,
    is_identity_op,
    OperationError,
)


def _parse_weight_updates(raw: dict) -> Dict[Tuple[str, str], float]:
    """Convert weight_updates from JSON-friendly format to tuple-keyed dict.

    JSON keys may be "from_node,to_node" strings or [from, to] lists.
    """
    result = {}
    for k, v in raw.items():
        if isinstance(k, str) and "," in k:
            parts = k.split(",", 1)
            result[(parts[0], parts[1])] = v
        elif isinstance(k, (list, tuple)) and len(k) == 2:
            result[(k[0], k[1])] = v
        else:
            result[k] = v
    return result


@dataclass
class Operation:
    """Represents a single operation in the event stream."""

    seq: int
    type: str
    params: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = {
            "seq": self.seq,
            "type": self.type,
            "params": self.params,
            "result": self.result,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
        if self.notes is not None:
            d["notes"] = self.notes
        return d

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
            notes=data.get("notes"),
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
    parent_annotation_id: Optional[str] = None  # Set when this is a child of a compositional annotation
    display_name: Optional[str] = None  # Cosmetic label; canonical name stays stable for collapse ops


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

    @property
    def has_non_identity_ops(self) -> bool:
        """Check if any operation in the stream is non-identity (changes function)."""
        return any(not is_identity_op(op.type) for op in self._operations)

    @property
    def last_identity_seq(self) -> Optional[int]:
        """Get the seq of the last identity op before the first non-identity op.

        Returns None if there are no non-identity ops, or if the first op
        is non-identity (no preceding identity ops).
        """
        first_non_identity_seq = None
        for op in self._operations:
            if not is_identity_op(op.type):
                first_non_identity_seq = op.seq
                break

        if first_non_identity_seq is None:
            return None  # No non-identity ops

        if first_non_identity_seq == 0:
            return None  # First op is non-identity, no preceding identity ops

        return first_non_identity_seq - 1

    def get_state_at_seq(self, seq: int) -> NetworkStructure:
        """Replay operations only up to seq and return the intermediate state.

        Args:
            seq: Sequence number (inclusive) — operations 0..seq are applied.

        Returns:
            NetworkStructure at that point in the operation history.
        """
        state = deepcopy(self._original_phenotype)

        # Temporarily build state by replaying up to seq
        temp_annotations: List[AnnotationData] = []
        temp_covered_nodes: Set[str] = set()
        temp_covered_connections: Set[Tuple[str, str]] = set()

        for op in self._operations:
            if op.seq > seq:
                break
            # We need a temporary engine-like context to apply ops
            # Use the internal method with temp state
            self._apply_op_to_state(
                op, state, temp_annotations,
                temp_covered_nodes, temp_covered_connections,
            )

        return state

    def _apply_op_to_state(
        self,
        op: "Operation",
        state: NetworkStructure,
        annotations: List[AnnotationData],
        covered_nodes: Set[str],
        covered_connections: Set[Tuple[str, str]],
    ) -> None:
        """Apply a single operation to an arbitrary state (for get_state_at_seq)."""
        if op.type == "split_node":
            apply_split_node(state, op.params["node_id"], covered_nodes)
        elif op.type == "consolidate_node":
            apply_consolidate_node(state, op.params["node_ids"], covered_nodes)
        elif op.type == "remove_node":
            apply_remove_node(state, op.params["node_id"], covered_nodes)
        elif op.type == "add_node":
            from .operations import apply_add_node as _apply_add_node
            _apply_add_node(
                state, tuple(op.params["connection"]), op.params["new_node_id"],
                covered_connections,
                bias=op.params.get("bias", 0.0),
                activation=op.params.get("activation", "identity"),
            )
        elif op.type == "add_identity_node":
            from .operations import apply_add_identity_node as _apply_identity
            connections = [tuple(c) for c in op.params["connections"]]
            _apply_identity(
                state, op.params["target_node"], connections,
                op.params["new_node_id"], covered_connections,
            )
        elif op.type == "annotate":
            ann = AnnotationData(
                name=op.params["name"],
                hypothesis=op.params.get("hypothesis", ""),
                entry_nodes=op.params["entry_nodes"],
                exit_nodes=op.params["exit_nodes"],
                subgraph_nodes=op.params["subgraph_nodes"],
                subgraph_connections=[tuple(c) for c in op.params.get("subgraph_connections", [])],
                evidence=op.params.get("evidence"),
            )
            annotations.append(ann)
            covered_nodes.update(ann.subgraph_nodes)
            covered_connections.update(ann.subgraph_connections)
        elif op.type == "disable_connection":
            from .operations import apply_disable_connection
            apply_disable_connection(
                state, op.params["from_node"], op.params["to_node"],
                covered_connections,
            )
        elif op.type == "enable_connection":
            from .operations import apply_enable_connection
            apply_enable_connection(
                state, op.params["from_node"], op.params["to_node"],
                covered_connections,
            )
        elif op.type == "rename_node":
            apply_rename_node(
                state, op.params["node_id"],
                op.params.get("display_name"), covered_nodes,
            )
        elif op.type == "rename_annotation":
            ann_id = op.params["annotation_id"]
            display_name = op.params.get("display_name")
            for ann in annotations:
                if ann.name == ann_id:
                    ann.display_name = display_name
                    break
        elif op.type == "prune_node":
            apply_prune_node(state, op.params["node_id"], covered_nodes)
        elif op.type == "prune_connection":
            apply_prune_connection(
                state, op.params["from_node"], op.params["to_node"],
                covered_connections,
            )
        elif op.type == "retrain":
            weight_updates = _parse_weight_updates(op.params.get("weight_updates", {}))
            bias_updates = op.params.get("bias_updates", {})
            apply_retrain(state, weight_updates, bias_updates)

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
                hypothesis=op.params.get("hypothesis", ""),
                entry_nodes=op.params["entry_nodes"],
                exit_nodes=op.params["exit_nodes"],
                subgraph_nodes=op.params["subgraph_nodes"],
                subgraph_connections=[tuple(c) for c in op.params.get("subgraph_connections", [])],
                evidence=op.params.get("evidence"),
            )
            self._annotations.append(annotation)
            self._covered_nodes.update(annotation.subgraph_nodes)
            self._covered_connections.update(annotation.subgraph_connections)

            # For compositional annotations, update child annotations' parent_annotation_id
            child_annotation_ids = op.params.get("child_annotation_ids", [])
            if child_annotation_ids:
                for child_name in child_annotation_ids:
                    # Find child annotation by name and set its parent
                    for ann in self._annotations:
                        if ann.name == child_name:
                            if ann.parent_annotation_id is not None:
                                import logging
                                logger = logging.getLogger(__name__)
                                logger.warning(
                                    f"Child '{child_name}' already has parent '{ann.parent_annotation_id}', "
                                    f"overwriting with '{annotation.name}'"
                                )
                            ann.parent_annotation_id = annotation.name
                            break

            op.result = {"annotation_index": len(self._annotations) - 1}

        elif op.type == "disable_connection":
            from .operations import apply_disable_connection
            result = apply_disable_connection(
                self._current_state, op.params["from_node"], op.params["to_node"],
                self._covered_connections,
            )
            op.result = result

        elif op.type == "enable_connection":
            from .operations import apply_enable_connection
            result = apply_enable_connection(
                self._current_state, op.params["from_node"], op.params["to_node"],
                self._covered_connections,
            )
            op.result = result

        elif op.type == "rename_node":
            result = apply_rename_node(
                self._current_state,
                op.params["node_id"],
                op.params.get("display_name"),
                self._covered_nodes,
            )
            op.result = result

        elif op.type == "rename_annotation":
            annotation_id = op.params["annotation_id"]
            display_name = op.params.get("display_name")
            for ann in self._annotations:
                if ann.name == annotation_id:
                    ann.display_name = display_name
                    break
            else:
                raise OperationError(f"Annotation '{annotation_id}' not found")
            op.result = {"annotation_id": annotation_id, "display_name": display_name}

        elif op.type == "prune_node":
            result = apply_prune_node(
                self._current_state,
                op.params["node_id"],
                self._covered_nodes,
            )
            op.result = result

        elif op.type == "prune_connection":
            result = apply_prune_connection(
                self._current_state,
                op.params["from_node"],
                op.params["to_node"],
                self._covered_connections,
            )
            op.result = result

        elif op.type == "retrain":
            weight_updates = _parse_weight_updates(op.params.get("weight_updates", {}))
            bias_updates = op.params.get("bias_updates", {})
            result = apply_retrain(
                self._current_state,
                weight_updates,
                bias_updates,
            )
            op.result = result

        else:
            raise OperationError(f"Unknown operation type: {op.type}")

    def add_operation(
        self,
        op_type: str,
        params: Dict[str, Any],
        validate: bool = True,
        notes: Optional[str] = None,
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
                self._annotations,
            )
            if errors:
                raise OperationError(f"Validation failed: {'; '.join(errors)}")

        # Create operation
        op = Operation(
            seq=len(self._operations),
            type=op_type,
            params=params,
            notes=notes,
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
            self._annotations,
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
