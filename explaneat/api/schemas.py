"""
Pydantic schemas for API request/response validation.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Literal, Union
from pydantic import BaseModel, Field, ConfigDict


# =============================================================================
# Base schemas
# =============================================================================


class FunctionNodeMetadataSchema(BaseModel):
    """Schema for collapsed annotation function node metadata."""

    annotation_name: str
    annotation_id: str
    hypothesis: Optional[str] = None
    n_inputs: int
    n_outputs: int
    input_names: List[str]
    output_names: List[str]
    formula_latex: Optional[str] = None
    subgraph_nodes: List[str]
    subgraph_connections: List[tuple[str, str]]

    model_config = ConfigDict(from_attributes=True)


class NodeSchema(BaseModel):
    """Schema for a node in the network."""

    id: str
    type: Literal["input", "hidden", "output", "identity", "function"]
    bias: Optional[float] = None
    activation: Optional[str] = None
    response: Optional[float] = None
    aggregation: Optional[str] = None
    function_metadata: Optional[FunctionNodeMetadataSchema] = None
    display_name: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class ConnectionSchema(BaseModel):
    """Schema for a connection in the network."""

    from_node: str = Field(alias="from")
    to_node: str = Field(alias="to")
    weight: float
    enabled: bool = True
    output_index: Optional[int] = None

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class ModelMetadata(BaseModel):
    """Metadata about the model state."""

    input_nodes: List[str]
    output_nodes: List[str]
    is_original: bool = True  # True if no operations applied
    collapsed_annotations: List[str] = []


class ModelStateResponse(BaseModel):
    """Response schema for model state (phenotype or after operations)."""

    nodes: List[NodeSchema]
    connections: List[ConnectionSchema]
    metadata: ModelMetadata


# =============================================================================
# Experiment schemas
# =============================================================================


class ExperimentListItem(BaseModel):
    """Schema for experiment in list response."""

    id: str
    name: str
    status: str
    dataset_name: Optional[str] = None
    dataset_id: Optional[str] = None
    has_split: bool = False
    generations: int
    total_genomes: int
    best_fitness: Optional[float] = None
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class ExperimentListResponse(BaseModel):
    """Response for list of experiments."""

    experiments: List[ExperimentListItem]
    total: int


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
    hypothesis: Optional[str] = None
    entry_nodes: List[str]
    exit_nodes: List[str]
    subgraph_nodes: List[str]
    subgraph_connections: List[tuple[str, str]] = []
    evidence: Optional[Dict[str, Any]] = None
    child_annotation_ids: List[str] = []  # IDs of child annotations for compositional annotations


class RenameNodeParams(BaseModel):
    """Parameters for rename_node operation."""

    node_id: str
    display_name: Optional[str] = None  # None to clear


class RenameAnnotationParams(BaseModel):
    """Parameters for rename_annotation operation."""

    annotation_id: str
    display_name: Optional[str] = None  # None to clear


class DisableConnectionParams(BaseModel):
    """Parameters for disable_connection operation."""

    from_node: str
    to_node: str


class EnableConnectionParams(BaseModel):
    """Parameters for enable_connection operation."""

    from_node: str
    to_node: str


class OperationRequest(BaseModel):
    """Request to add a new operation."""

    type: Literal[
        "split_node",
        "consolidate_node",
        "remove_node",
        "add_node",
        "add_identity_node",
        "annotate",
        "rename_node",
        "rename_annotation",
        "disable_connection",
        "enable_connection",
    ]
    params: Union[
        SplitNodeParams,
        ConsolidateNodeParams,
        RemoveNodeParams,
        AddNodeParams,
        AddIdentityNodeParams,
        AnnotateParams,
        RenameNodeParams,
        RenameAnnotationParams,
        DisableConnectionParams,
        EnableConnectionParams,
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
# Annotation schemas
# =============================================================================


class AnnotationSummary(BaseModel):
    """Summary of an annotation for collapsing/hierarchy."""

    id: str
    name: Optional[str] = None
    display_name: Optional[str] = None
    entry_nodes: List[str]
    exit_nodes: List[str]
    subgraph_nodes: List[str]
    parent_annotation_id: Optional[str] = None
    children_ids: List[str] = []
    is_leaf: bool = True

    model_config = ConfigDict(from_attributes=True)


class AnnotationListResponse(BaseModel):
    """Response for list of annotations."""

    annotations: List[AnnotationSummary]
    total: int


# =============================================================================
# Dataset schemas
# =============================================================================


class DatasetResponse(BaseModel):
    """Response for dataset metadata."""

    id: str
    name: str
    version: Optional[str] = None
    source: Optional[str] = None
    source_url: Optional[str] = None
    description: Optional[str] = None
    num_samples: Optional[int] = None
    num_features: Optional[int] = None
    num_classes: Optional[int] = None
    feature_names: Optional[List[str]] = None
    feature_types: Optional[Dict[str, str]] = None
    feature_descriptions: Optional[Dict[str, str]] = None
    target_name: Optional[str] = None
    target_description: Optional[str] = None
    class_names: Optional[List[str]] = None
    has_data: bool = False

    model_config = ConfigDict(from_attributes=True)


class DatasetListResponse(BaseModel):
    """Response for list of datasets."""

    datasets: List[DatasetResponse]
    total: int


class PMLBDownloadRequest(BaseModel):
    """Request to download and store a PMLB dataset."""

    name: str
    version: Optional[str] = None


class LinkDatasetRequest(BaseModel):
    """Request to link a dataset to an experiment and create a split."""

    dataset_id: str
    test_proportion: float = 0.2
    random_seed: int = 42
    stratify: bool = False


class ExperimentSplitResponse(BaseModel):
    """Response for an experiment's linked split + dataset metadata."""

    split_id: str
    dataset_id: str
    dataset_name: str
    num_samples: Optional[int] = None
    num_features: Optional[int] = None
    num_classes: Optional[int] = None
    split_type: str
    test_size: Optional[float] = None
    random_state: Optional[int] = None
    train_size: Optional[int] = None
    test_size_actual: Optional[int] = None
    feature_names: Optional[List[str]] = None
    feature_types: Optional[Dict[str, str]] = None
    feature_descriptions: Optional[Dict[str, str]] = None
    target_name: Optional[str] = None
    target_description: Optional[str] = None


class SplitCreateRequest(BaseModel):
    """Request to create a dataset split."""

    experiment_id: str
    test_proportion: float = 0.2
    random_seed: int = 42
    stratify: bool = False


class SplitResponse(BaseModel):
    """Response for a dataset split."""

    id: str
    dataset_id: str
    experiment_id: str
    split_type: str
    test_size: Optional[float] = None
    random_state: Optional[int] = None
    train_size: Optional[int] = None
    test_size_actual: Optional[int] = None

    model_config = ConfigDict(from_attributes=True)


class SplitListResponse(BaseModel):
    """Response for list of splits."""

    splits: List[SplitResponse]
    total: int


# =============================================================================
# Evidence & Visualization schemas
# =============================================================================


class VizDataRequest(BaseModel):
    """Request for computing visualization data."""

    annotation_id: str
    dataset_split_id: str
    viz_type: Literal[
        "line", "heatmap", "partial_dependence", "pca_scatter", "sensitivity"
    ]
    params: Optional[Dict[str, Any]] = None
    split: Literal["train", "test", "both"] = "both"
    sample_fraction: float = 0.1
    max_samples: int = 1000


class VizDataResponse(BaseModel):
    """Response for visualization data."""

    viz_type: str
    data: Dict[str, Any]
    dimensionality: List[int]
    suggested_viz_types: List[str]


class ChildFormulaInfo(BaseModel):
    """Formula info for a child annotation within a composed annotation."""

    name: str
    latex: Optional[str] = None
    dimensionality: List[int]


class FormulaResponse(BaseModel):
    """Response for closed-form formula."""

    latex: Optional[str] = None                    # backwards-compat: expanded form
    latex_collapsed: Optional[str] = None          # collapsed form (references child names)
    latex_expanded: Optional[str] = None           # fully expanded form
    tractable: bool = False
    dimensionality: List[int]                      # [n_inputs, n_outputs]
    is_composed: bool = False                      # True if annotation has children
    children: List[ChildFormulaInfo] = []           # child annotation formulas


class SnapshotRequest(BaseModel):
    """Request to save a visualization snapshot as evidence."""

    annotation_id: str
    viz_config: Dict[str, Any]
    svg_data: str  # base64-encoded SVG
    narrative: str = ""
    category: Literal[
        "analytical_methods", "visualizations", "counterfactuals", "other_evidence"
    ] = "visualizations"


class NarrativeUpdateRequest(BaseModel):
    """Request to update narrative text on evidence entry."""

    annotation_id: str
    evidence_index: int
    narrative: str


class EvidenceEntry(BaseModel):
    """A single evidence entry."""

    viz_config: Optional[Dict[str, Any]] = None
    svg_data: Optional[str] = None
    narrative: str = ""
    category: str = "other_evidence"
    timestamp: Optional[str] = None


class EvidenceListResponse(BaseModel):
    """Response for list of evidence entries."""

    annotation_id: str
    entries: List[EvidenceEntry]
    total: int


# =============================================================================
# Input Distribution schemas
# =============================================================================


class InputDistributionRequest(BaseModel):
    """Request for input feature distribution data."""

    dataset_split_id: str
    feature_indices: List[int]  # 1 or 2 indices
    split: Literal["train", "test", "both"] = "both"
    num_bins: int = 30


class InputDistributionResponse(BaseModel):
    """Response for input feature distribution data."""

    viz_type: str  # "histogram" or "scatter2d"
    data: Dict[str, Any]
    feature_names: List[str]


# =============================================================================
# Error schemas
# =============================================================================


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str
    error_code: Optional[str] = None
