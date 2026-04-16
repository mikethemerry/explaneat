"""
Pydantic schemas for API request/response validation.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Literal, Union
from pydantic import BaseModel, Field, ConfigDict, model_validator


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
    has_non_identity_ops: bool = False  # True if model function has been changed


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


class ExperimentDetailResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    status: str
    dataset_id: Optional[str] = None
    dataset_name: Optional[str] = None
    config_template_id: Optional[str] = None
    config_template_name: Optional[str] = None
    resolved_config: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None


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


class PruneNodeParams(BaseModel):
    """Parameters for prune_node operation (non-identity: removes node and all connections)."""

    node_id: str


class PruneConnectionParams(BaseModel):
    """Parameters for prune_connection operation (non-identity: permanently removes connection)."""

    from_node: str
    to_node: str


class RetrainParams(BaseModel):
    """Parameters for retrain operation (non-identity: updates weights/biases from training)."""

    weight_updates: Dict[str, float] = {}  # JSON keys are "from_node,to_node"
    bias_updates: Dict[str, float] = {}
    metadata: Optional[Dict[str, Any]] = None  # training config, loss, etc.


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
        "prune_node",
        "prune_connection",
        "retrain",
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
        PruneNodeParams,
        PruneConnectionParams,
        RetrainParams,
    ]
    notes: Optional[str] = None  # Human-readable justification


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
    notes: Optional[str] = None


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
    task_type: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    source_dataset_id: Optional[str] = None
    encoding_config: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(from_attributes=True)


class DatasetListResponse(BaseModel):
    """Response for list of datasets."""

    datasets: List[DatasetResponse]
    total: int


class DatasetUpdateRequest(BaseModel):
    """Request to partially update dataset metadata."""

    description: Optional[str] = None
    num_classes: Optional[int] = None
    class_names: Optional[List[str]] = None
    feature_descriptions: Optional[Dict[str, str]] = None
    feature_types: Optional[Dict[str, str]] = None
    target_name: Optional[str] = None
    target_description: Optional[str] = None
    task_type: Optional[str] = None


class PrepareDatasetRequest(BaseModel):
    """Request to create a prepared (one-hot encoded) dataset."""
    name: Optional[str] = None
    encoding_config: Optional[Dict[str, Any]] = None
    ordinal_onehot: Optional[List[str]] = None
    ordinal_orders: Optional[Dict[str, List[str]]] = None


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
    validation_size: Optional[int] = None
    feature_names: Optional[List[str]] = None
    feature_types: Optional[Dict[str, str]] = None
    feature_descriptions: Optional[Dict[str, str]] = None
    target_name: Optional[str] = None
    target_description: Optional[str] = None
    class_names: Optional[List[str]] = None


class SplitCreateRequest(BaseModel):
    """Request to create a dataset split."""

    test_proportion: float = 0.2
    random_seed: int = 42
    stratify: bool = False


class SplitResponse(BaseModel):
    """Response for a dataset split."""

    id: str
    dataset_id: str
    name: Optional[str] = None
    split_type: str
    test_size: Optional[float] = None
    random_state: Optional[int] = None
    train_size: Optional[int] = None
    test_size_actual: Optional[int] = None
    validation_size: Optional[int] = None

    model_config = ConfigDict(from_attributes=True)


class SplitListResponse(BaseModel):
    """Response for list of splits."""

    splits: List[SplitResponse]
    total: int


# =============================================================================
# Evidence & Visualization schemas
# =============================================================================


class NodeEvidenceInfoResponse(BaseModel):
    """Response for node evidence info (virtual annotation for a single node)."""

    node_id: str
    entry_nodes: List[str]
    exit_nodes: List[str]
    subgraph_nodes: List[str]
    display_name: str


class VizDataRequest(BaseModel):
    """Request for computing visualization data."""

    annotation_id: Optional[str] = None
    node_id: Optional[str] = None
    dataset_split_id: str
    viz_type: Literal[
        "line", "heatmap", "partial_dependence", "pca_scatter", "sensitivity",
        "ice", "feature_output_scatter", "output_distribution",
    ]
    params: Optional[Dict[str, Any]] = None
    split: Literal["train", "test", "val", "both"] = "both"
    sample_fraction: float = 0.1
    max_samples: int = 1000
    view: Literal["network", "source"] = "network"

    @model_validator(mode="after")
    def check_at_most_one_target(self):
        if self.annotation_id and self.node_id:
            raise ValueError("Provide at most one of annotation_id or node_id, not both")
        return self


class VizDataResponse(BaseModel):
    """Response for visualization data."""

    viz_type: str
    data: Dict[str, Any]
    dimensionality: List[int]
    suggested_viz_types: List[str]
    entry_names: Optional[List[str]] = None
    exit_names: Optional[List[str]] = None
    correctness: Optional[List[bool]] = None
    predicted_class: Optional[List[int]] = None
    true_class: Optional[List[int]] = None
    class_names: Optional[List[str]] = None
    num_classes: Optional[int] = None


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
    split: Literal["train", "test", "val", "both"] = "both"
    num_bins: int = 30


class InputDistributionResponse(BaseModel):
    """Response for input feature distribution data."""

    viz_type: str  # "histogram" or "scatter2d"
    data: Dict[str, Any]
    feature_names: List[str]


# =============================================================================
# SHAP schemas
# =============================================================================


class ShapRequest(BaseModel):
    """Request for SHAP value computation."""

    dataset_split_id: str
    annotation_id: Optional[str] = None  # None = whole model
    node_id: Optional[str] = None  # Single-node evidence
    split: Literal["train", "test", "val", "both"] = "both"
    max_samples: int = 100
    force_recompute: bool = False


class ShapOutputResult(BaseModel):
    """SHAP results for a single output dimension."""

    output_name: str
    mean_abs_shap: List[float]
    base_value: float


class ShapResponse(BaseModel):
    """Response with SHAP values."""

    feature_names: List[str]
    mean_abs_shap: List[float]
    base_value: float
    outputs: Optional[List[ShapOutputResult]] = None


# =============================================================================
# Performance schemas
# =============================================================================


class RetrainStartRequest(BaseModel):
    """Request to start a retraining job."""

    dataset_split_id: str
    split: Literal["train", "test", "val", "both"] = "both"
    n_epochs: int = 50
    learning_rate: float = 0.01
    freeze_annotations: bool = False
    max_samples: int = 10000


class RetrainStartResponse(BaseModel):
    """Response for starting a retraining job."""

    job_id: str


class RetrainStatusResponse(BaseModel):
    """Response for retraining job status."""

    job_id: str
    status: str
    current_epoch: int
    total_epochs: int
    metrics: Dict[str, List[float]] = {}
    error: Optional[str] = None


class RetrainApplyResponse(BaseModel):
    """Response for applying retrain results."""

    operation_seq: int
    final_loss: Optional[float] = None
    final_val_loss: Optional[float] = None
    epochs_completed: int


class PerformanceRequest(BaseModel):
    """Request for model performance computation."""

    dataset_split_id: str
    split: Literal["train", "test", "val", "both"] = "both"
    sample_fraction: float = 1.0
    max_samples: int = 10000
    at_seq: Optional[int] = None  # Compute at specific operation seq (for before/after)


class PerformanceResponse(BaseModel):
    """Response for model performance metrics."""

    mse: float
    rmse: float
    mae: float
    accuracy: Optional[float] = None  # Classification only
    auc_roc: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    log_loss: Optional[float] = None
    brier_score: Optional[float] = None
    balanced_accuracy: Optional[float] = None
    calibration: Optional[Dict[str, List[float]]] = None  # {bin_means, fraction_positives}
    n_samples: int
    at_seq: Optional[int] = None
    has_non_identity_ops: bool = False


# =============================================================================
# Experiment pipeline schemas
# =============================================================================


class ExperimentCreateRequest(BaseModel):
    """Request to create and run a new experiment."""

    name: str
    description: str = ""
    dataset_id: str
    dataset_split_id: str
    n_generations: int = 10
    n_epochs_backprop: int = 5
    fitness_function: Literal["bce", "auc"] = "bce"
    population_size: int = 150
    mutation_rate: float = 0.3
    crossover_rate: float = 0.5
    config_template_id: Optional[str] = None
    config_overrides: Optional[Dict[str, Any]] = None


class ExperimentCreateResponse(BaseModel):
    """Response for starting an experiment."""

    job_id: str


class ExperimentProgressResponse(BaseModel):
    """Response for experiment progress polling."""

    job_id: str
    experiment_id: Optional[str] = None
    status: str
    current_generation: int
    total_generations: int
    best_fitness: Optional[float] = None
    mean_fitness: Optional[float] = None
    pop_size: int = 0
    num_species: int = 0
    error: Optional[str] = None


# =============================================================================
# Error schemas
# =============================================================================


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str
    error_code: Optional[str] = None


# =============================================================================
# Config template schemas
# =============================================================================


class ConfigTemplateResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    config: Dict[str, Any]
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ConfigTemplateListResponse(BaseModel):
    templates: List[ConfigTemplateResponse]
    total: int


class ConfigTemplateCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    config: Dict[str, Any]


class ConfigTemplateUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
