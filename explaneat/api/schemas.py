"""
Pydantic schemas for API request/response validation.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Literal, Union
from pydantic import BaseModel, Field, ConfigDict


# =============================================================================
# Base schemas
# =============================================================================


class NodeSchema(BaseModel):
    """Schema for a node in the network."""

    id: str
    type: Literal["input", "hidden", "output", "identity"]
    bias: Optional[float] = None
    activation: Optional[str] = None
    response: Optional[float] = None
    aggregation: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class ConnectionSchema(BaseModel):
    """Schema for a connection in the network."""

    from_node: str = Field(alias="from")
    to_node: str = Field(alias="to")
    weight: float
    enabled: bool = True

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class ModelMetadata(BaseModel):
    """Metadata about the model state."""

    input_nodes: List[str]
    output_nodes: List[str]
    is_original: bool = True  # True if no operations applied


class ModelStateResponse(BaseModel):
    """Response schema for model state (phenotype or after operations)."""

    nodes: List[NodeSchema]
    connections: List[ConnectionSchema]
    metadata: ModelMetadata


# =============================================================================
# Genome schemas
# =============================================================================


class GenomeListItem(BaseModel):
    """Schema for genome in list response."""

    id: str
    genome_id: int
    fitness: Optional[float] = None
    num_nodes: Optional[int] = None
    num_connections: Optional[int] = None
    population_id: str
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class GenomeDetail(GenomeListItem):
    """Detailed genome response."""

    genome_data: Dict[str, Any]
    network_depth: Optional[int] = None
    network_width: Optional[int] = None


class GenomeListResponse(BaseModel):
    """Response for list of genomes."""

    genomes: List[GenomeListItem]
    total: int


# =============================================================================
# Operation schemas
# =============================================================================


class SplitNodeParams(BaseModel):
    """Parameters for split_node operation."""

    node_id: str


class ConsolidateNodeParams(BaseModel):
    """Parameters for consolidate_node operation."""

    node_ids: List[str]


class RemoveNodeParams(BaseModel):
    """Parameters for remove_node operation."""

    node_id: str


class AddNodeParams(BaseModel):
    """Parameters for add_node operation."""

    connection: tuple[str, str]
    new_node_id: str
    bias: Optional[float] = 0.0
    activation: Optional[str] = "identity"


class AddIdentityNodeParams(BaseModel):
    """Parameters for add_identity_node operation."""

    target_node: str
    connections: List[tuple[str, str]]
    new_node_id: str


class AnnotateParams(BaseModel):
    """Parameters for annotate operation."""

    name: str
    hypothesis: str
    entry_nodes: List[str]
    exit_nodes: List[str]
    subgraph_nodes: List[str]
    subgraph_connections: List[tuple[str, str]]
    evidence: Optional[Dict[str, Any]] = None


class OperationRequest(BaseModel):
    """Request to add a new operation."""

    type: Literal[
        "split_node",
        "consolidate_node",
        "remove_node",
        "add_node",
        "add_identity_node",
        "annotate",
    ]
    params: Union[
        SplitNodeParams,
        ConsolidateNodeParams,
        RemoveNodeParams,
        AddNodeParams,
        AddIdentityNodeParams,
        AnnotateParams,
    ]


class OperationResult(BaseModel):
    """Result of applying an operation."""

    created_nodes: Optional[List[str]] = None
    removed_nodes: Optional[List[str]] = None
    created_connections: Optional[List[tuple[str, str]]] = None
    removed_connections: Optional[List[tuple[str, str]]] = None
    annotation_id: Optional[str] = None


class OperationResponse(BaseModel):
    """Response for a single operation."""

    seq: int
    type: str
    params: Dict[str, Any]
    result: Optional[OperationResult] = None
    created_at: datetime


class OperationListResponse(BaseModel):
    """Response for list of operations."""

    operations: List[OperationResponse]
    total: int


class OperationValidationResponse(BaseModel):
    """Response for operation validation."""

    valid: bool
    errors: List[str] = []
    warnings: List[str] = []


# =============================================================================
# Explanation schemas
# =============================================================================


class ExplanationResponse(BaseModel):
    """Response for explanation document."""

    id: str
    genome_id: str
    name: Optional[str] = None
    description: Optional[str] = None
    operations: List[OperationResponse] = []
    is_well_formed: bool = False
    structural_coverage: Optional[float] = None
    compositional_coverage: Optional[float] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class ExplanationUpdateRequest(BaseModel):
    """Request to update explanation metadata."""

    name: Optional[str] = None
    description: Optional[str] = None


# =============================================================================
# Analysis schemas
# =============================================================================


class ViolationDetail(BaseModel):
    """Details about a node that violates annotation constraints."""

    node_id: str
    reason: Literal["has_external_input_and_output"]
    external_inputs: List[tuple[str, str]]
    external_outputs: List[tuple[str, str]]
    internal_outputs: Optional[List[tuple[str, str]]] = None


class SplitDetectionRequest(BaseModel):
    """Request for split detection analysis."""

    proposed_coverage: List[str]


class SplitDetectionResponse(BaseModel):
    """Response for split detection analysis."""

    proposed_coverage: List[str]
    violations: List[ViolationDetail]
    suggested_operations: List[OperationRequest] = []
    adjusted_coverage: Optional[List[str]] = None


class NodeClassification(BaseModel):
    """Classification of nodes within proposed coverage."""

    entry: List[str]
    intermediate: List[str]
    exit: List[str]


class ClassifyNodesRequest(BaseModel):
    """Request for node classification."""

    coverage: List[str]


class ClassifyNodesResponse(BaseModel):
    """Response for node classification."""

    coverage: List[str]
    classification: NodeClassification
    valid: bool
    violations: List[ViolationDetail] = []


class CoverageResponse(BaseModel):
    """Response for coverage analysis."""

    structural_coverage: float
    compositional_coverage: float
    covered_nodes: List[str]
    uncovered_nodes: List[str]
    annotations_count: int


# =============================================================================
# Error schemas
# =============================================================================


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str
    error_code: Optional[str] = None
