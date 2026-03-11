/**
 * API client for ExplaNEAT REST API
 */

// =============================================================================
// Logging utilities
// =============================================================================

const LOG_PREFIX = "[API]";

function logDebug(message: string, data?: unknown) {
  console.log(`${LOG_PREFIX} ${message}`, data !== undefined ? data : "");
}

function logInfo(message: string, data?: unknown) {
  console.info(`${LOG_PREFIX} ${message}`, data !== undefined ? data : "");
}

function logError(message: string, data?: unknown) {
  console.error(`${LOG_PREFIX} ${message}`, data !== undefined ? data : "");
}

// =============================================================================
// Configuration
// =============================================================================

// Use relative URL when proxying through Vite dev server
// Falls back to localhost:8000 for direct access
const API_BASE = "/api";

// ============================================================================
// Types matching API schemas
// ============================================================================

export type NodeType = "input" | "hidden" | "output" | "identity" | "function";

export type FunctionNodeMetadata = {
  annotation_name: string;
  annotation_id: string;
  hypothesis: string;
  n_inputs: number;
  n_outputs: number;
  input_names: string[];
  output_names: string[];
  formula_latex: string | null;
  subgraph_nodes: string[];
  subgraph_connections: [string, string][];
};

export type ApiNode = {
  id: string;
  type: NodeType;
  bias: number | null;
  activation: string | null;
  response: number | null;
  aggregation: string | null;
  function_metadata?: FunctionNodeMetadata | null;
};

export type ApiConnection = {
  from: string;
  to: string;
  weight: number;
  enabled: boolean;
  output_index?: number | null;
};

export type ModelMetadata = {
  input_nodes: string[];
  output_nodes: string[];
  is_original: boolean;
  collapsed_annotations?: string[];
};

export type ModelState = {
  nodes: ApiNode[];
  connections: ApiConnection[];
  metadata: ModelMetadata;
};

export type GenomeListItem = {
  id: string;
  genome_id: number;
  fitness: number | null;
  num_nodes: number;
  num_connections: number;
  population_id: string;
  created_at: string;
};

export type GenomeListResponse = {
  genomes: GenomeListItem[];
  total: number;
};

export type ExperimentListItem = {
  id: string;
  name: string;
  status: string;
  dataset_name: string | null;
  dataset_id: string | null;
  has_split: boolean;
  generations: number;
  total_genomes: number;
  best_fitness: number | null;
  created_at: string;
};

export type ExperimentListResponse = {
  experiments: ExperimentListItem[];
  total: number;
};

export type BestGenomeResponse = {
  genome_id: string;
  neat_genome_id: number;
  fitness: number;
  num_nodes: number;
  num_connections: number;
  experiment_id: string;
  experiment_name: string;
};

export type OperationResult = {
  created_nodes?: string[];
  removed_nodes?: string[];
  created_connections?: [string, string][];
  removed_connections?: [string, string][];
  annotation_id?: string;
};

export type Operation = {
  seq: number;
  type: string;
  params: Record<string, unknown>;
  result?: OperationResult;
  created_at: string;
};

export type OperationListResponse = {
  operations: Operation[];
  total: number;
};

export type ExplanationResponse = {
  id: string;
  genome_id: string;
  name: string | null;
  description: string | null;
  operations: Operation[];
  is_well_formed: boolean;
  structural_coverage: number | null;
  compositional_coverage: number | null;
  created_at: string;
  updated_at: string;
};

export type ViolationDetail = {
  node_id: string;
  reason: string;
  external_inputs: [string, string][];
  external_outputs: [string, string][];
  internal_outputs?: [string, string][];
};

export type SplitDetectionRequest = {
  proposed_coverage: string[];
};

export type SplitDetectionResponse = {
  proposed_coverage: string[];
  violations: ViolationDetail[];
  suggested_operations: { type: string; params: { node_id: string } }[];
  adjusted_coverage: string[] | null;
};

export type NodeClassification = {
  entry: string[];
  intermediate: string[];
  exit: string[];
};

export type ClassifyNodesRequest = {
  coverage: string[];
};

export type ClassifyNodesResponse = {
  coverage: string[];
  classification: NodeClassification;
  valid: boolean;
  violations: ViolationDetail[];
};

export type CoverageResponse = {
  structural_coverage: number;
  compositional_coverage: number;
  covered_nodes: string[];
  uncovered_nodes: string[];
  annotations_count: number;
};

// Annotation types for collapsing
export type AnnotationSummary = {
  id: string;
  name: string | null;
  entry_nodes: string[];
  exit_nodes: string[];
  subgraph_nodes: string[];
  parent_annotation_id: string | null;
  children_ids: string[];
  is_leaf: boolean;
};

export type AnnotationListResponse = {
  annotations: AnnotationSummary[];
  total: number;
};

// Operation request types
export type SplitNodeParams = { node_id: string };
export type ConsolidateNodeParams = { node_ids: string[] };
export type RemoveNodeParams = { node_id: string };
export type AddNodeParams = {
  connection: [string, string];
  new_node_id: string;
  bias?: number;
  activation?: string;
};
export type AddIdentityNodeParams = {
  target_node: string;
  connections: [string, string][];
  new_node_id: string;
};
export type AnnotateParams = {
  name: string;
  hypothesis?: string;
  entry_nodes: string[];
  exit_nodes: string[];
  subgraph_nodes: string[];
  subgraph_connections?: [string, string][];
};

export type OperationRequest =
  | { type: "split_node"; params: SplitNodeParams }
  | { type: "consolidate_node"; params: ConsolidateNodeParams }
  | { type: "remove_node"; params: RemoveNodeParams }
  | { type: "add_node"; params: AddNodeParams }
  | { type: "add_identity_node"; params: AddIdentityNodeParams }
  | { type: "annotate"; params: AnnotateParams };

// ============================================================================
// API Client
// ============================================================================

class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const method = options?.method || "GET";
  logDebug(`${method} ${url}`, options?.body ? JSON.parse(options.body as string) : undefined);
  const startTime = performance.now();

  const response = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });

  const elapsed = performance.now() - startTime;

  if (!response.ok) {
    const text = await response.text();
    logError(`${method} ${url} failed (${response.status}) in ${elapsed.toFixed(2)}ms`, text);
    throw new ApiError(response.status, text);
  }

  const data = await response.json();
  logInfo(`${method} ${url} completed in ${elapsed.toFixed(2)}ms`, {
    status: response.status,
    dataKeys: Object.keys(data),
  });

  return data;
}

// ============================================================================
// Experiment endpoints
// ============================================================================

export async function listExperiments(
  limit = 50,
  offset = 0,
): Promise<ExperimentListResponse> {
  return fetchJson<ExperimentListResponse>(
    `${API_BASE}/experiments?limit=${limit}&offset=${offset}`,
  );
}

export async function getBestGenome(
  experimentId: string,
): Promise<BestGenomeResponse> {
  return fetchJson<BestGenomeResponse>(
    `${API_BASE}/experiments/${experimentId}/best-genome`,
  );
}

export type LinkDatasetRequest = {
  dataset_id: string;
  test_proportion: number;
  random_seed: number;
  stratify: boolean;
};

export type ExperimentSplitResponse = {
  split_id: string;
  dataset_id: string;
  dataset_name: string;
  num_samples: number | null;
  num_features: number | null;
  num_classes: number | null;
  split_type: string;
  test_size: number | null;
  random_state: number | null;
  train_size: number | null;
  test_size_actual: number | null;
};

export async function getExperimentSplit(
  experimentId: string,
): Promise<ExperimentSplitResponse> {
  return fetchJson<ExperimentSplitResponse>(
    `${API_BASE}/experiments/${experimentId}/split`,
  );
}

export async function linkDatasetToExperiment(
  experimentId: string,
  request: LinkDatasetRequest,
): Promise<ExperimentSplitResponse> {
  return fetchJson<ExperimentSplitResponse>(
    `${API_BASE}/experiments/${experimentId}/dataset`,
    {
      method: "PUT",
      body: JSON.stringify(request),
    },
  );
}

// ============================================================================
// Genome endpoints
// ============================================================================

export async function listGenomes(
  limit = 50,
  offset = 0,
): Promise<GenomeListResponse> {
  return fetchJson<GenomeListResponse>(
    `${API_BASE}/genomes?limit=${limit}&offset=${offset}`,
  );
}

export async function getGenomePhenotype(genomeId: string): Promise<ModelState> {
  return fetchJson<ModelState>(`${API_BASE}/genomes/${genomeId}/phenotype`);
}

export async function getGenomeExplanation(
  genomeId: string,
): Promise<ExplanationResponse> {
  return fetchJson<ExplanationResponse>(
    `${API_BASE}/genomes/${genomeId}/explanation`,
  );
}

// ============================================================================
// Model state endpoints
// ============================================================================

export async function getCurrentModel(
  genomeId: string,
  collapsed?: string[],
): Promise<ModelState> {
  let url = `${API_BASE}/genomes/${genomeId}/model`;
  if (collapsed && collapsed.length > 0) {
    url += `?collapsed=${collapsed.join(",")}`;
  }
  return fetchJson<ModelState>(url);
}

// ============================================================================
// Operations endpoints
// ============================================================================

export async function listOperations(
  genomeId: string,
): Promise<OperationListResponse> {
  return fetchJson<OperationListResponse>(
    `${API_BASE}/genomes/${genomeId}/operations`,
  );
}

export async function addOperation(
  genomeId: string,
  operation: OperationRequest,
): Promise<Operation> {
  return fetchJson<Operation>(`${API_BASE}/genomes/${genomeId}/operations`, {
    method: "POST",
    body: JSON.stringify(operation),
  });
}

export async function removeOperation(
  genomeId: string,
  seq: number,
): Promise<{ status: string; removed_count: number; remaining_operations: number }> {
  return fetchJson(`${API_BASE}/genomes/${genomeId}/operations/${seq}`, {
    method: "DELETE",
  });
}

export async function validateOperation(
  genomeId: string,
  operation: OperationRequest,
): Promise<{ valid: boolean; errors: string[]; warnings: string[] }> {
  return fetchJson(`${API_BASE}/genomes/${genomeId}/operations/validate`, {
    method: "POST",
    body: JSON.stringify(operation),
  });
}

// ============================================================================
// Analysis endpoints
// ============================================================================

export async function detectSplits(
  genomeId: string,
  proposedCoverage: string[],
): Promise<SplitDetectionResponse> {
  return fetchJson<SplitDetectionResponse>(
    `${API_BASE}/genomes/${genomeId}/analyze/split-detection`,
    {
      method: "POST",
      body: JSON.stringify({ proposed_coverage: proposedCoverage }),
    },
  );
}

export async function classifyNodes(
  genomeId: string,
  coverage: string[],
): Promise<ClassifyNodesResponse> {
  return fetchJson<ClassifyNodesResponse>(
    `${API_BASE}/genomes/${genomeId}/analyze/classify-nodes`,
    {
      method: "POST",
      body: JSON.stringify({ coverage }),
    },
  );
}

export async function autoClassify(
  genomeId: string,
  coverage: string[],
): Promise<{
  coverage: string[];
  suggested_entry_nodes: string[];
  suggested_exit_nodes: string[];
  intermediate_nodes: string[];
  valid: boolean;
  violations: ViolationDetail[];
}> {
  return fetchJson(`${API_BASE}/genomes/${genomeId}/analyze/auto-classify`, {
    method: "POST",
    body: JSON.stringify({ coverage }),
  });
}

export async function getCoverageAnalysis(
  genomeId: string,
): Promise<CoverageResponse> {
  return fetchJson<CoverageResponse>(
    `${API_BASE}/genomes/${genomeId}/analyze/coverage`,
  );
}

// ============================================================================
// Annotation endpoints
// ============================================================================

export async function listAnnotations(
  genomeId: string,
): Promise<AnnotationListResponse> {
  return fetchJson<AnnotationListResponse>(
    `${API_BASE}/genomes/${genomeId}/annotations`,
  );
}

// ============================================================================
// Dataset types and endpoints
// ============================================================================

export type DatasetResponse = {
  id: string;
  name: string;
  version: string | null;
  source: string | null;
  source_url: string | null;
  description: string | null;
  num_samples: number | null;
  num_features: number | null;
  num_classes: number | null;
  feature_names: string[] | null;
  target_name: string | null;
  class_names: string[] | null;
  has_data: boolean;
};

export type DatasetListResponse = {
  datasets: DatasetResponse[];
  total: number;
};

export type SplitResponse = {
  id: string;
  dataset_id: string;
  experiment_id: string;
  split_type: string;
  test_size: number | null;
  random_state: number | null;
  train_size: number | null;
  test_size_actual: number | null;
};

export type SplitListResponse = {
  splits: SplitResponse[];
  total: number;
};

export async function listDatasets(): Promise<DatasetListResponse> {
  return fetchJson<DatasetListResponse>(`${API_BASE}/datasets`);
}

export async function downloadPMLBDataset(
  name: string,
  version?: string,
): Promise<DatasetResponse> {
  return fetchJson<DatasetResponse>(`${API_BASE}/datasets/pmlb`, {
    method: "POST",
    body: JSON.stringify({ name, version }),
  });
}

export async function listSplits(
  datasetId: string,
): Promise<SplitListResponse> {
  return fetchJson<SplitListResponse>(
    `${API_BASE}/datasets/${datasetId}/splits`,
  );
}

export async function createSplit(
  datasetId: string,
  experimentId: string,
  testProportion = 0.2,
  randomSeed = 42,
  stratify = false,
): Promise<SplitResponse> {
  return fetchJson<SplitResponse>(
    `${API_BASE}/datasets/${datasetId}/splits`,
    {
      method: "POST",
      body: JSON.stringify({
        experiment_id: experimentId,
        test_proportion: testProportion,
        random_seed: randomSeed,
        stratify,
      }),
    },
  );
}

// ============================================================================
// Evidence & Visualization types and endpoints
// ============================================================================

export type VizDataRequest = {
  annotation_id: string;
  dataset_split_id: string;
  viz_type: "line" | "heatmap" | "partial_dependence" | "pca_scatter" | "sensitivity";
  params?: Record<string, unknown>;
  split?: "train" | "test" | "both";
  sample_fraction?: number;
  max_samples?: number;
};

export type VizDataResponse = {
  viz_type: string;
  data: Record<string, unknown>;
  dimensionality: [number, number];
  suggested_viz_types: string[];
};

export type ChildFormulaInfo = {
  name: string;
  latex: string | null;
  dimensionality: [number, number];
};

export type FormulaResponse = {
  latex: string | null;
  latex_collapsed: string | null;
  latex_expanded: string | null;
  tractable: boolean;
  dimensionality: [number, number];
  is_composed: boolean;
  children: ChildFormulaInfo[];
};

export type EvidenceEntry = {
  viz_config: Record<string, unknown> | null;
  svg_data: string | null;
  narrative: string;
  category: string;
  timestamp: string | null;
};

export type EvidenceListResponse = {
  annotation_id: string;
  entries: EvidenceEntry[];
  total: number;
};

export async function computeVizData(
  genomeId: string,
  request: VizDataRequest,
): Promise<VizDataResponse> {
  return fetchJson<VizDataResponse>(
    `${API_BASE}/genomes/${genomeId}/evidence/viz-data`,
    {
      method: "POST",
      body: JSON.stringify(request),
    },
  );
}

export async function getFormula(
  genomeId: string,
  annotationId: string,
): Promise<FormulaResponse> {
  return fetchJson<FormulaResponse>(
    `${API_BASE}/genomes/${genomeId}/evidence/formula?annotation_id=${annotationId}`,
  );
}

export async function saveSnapshot(
  genomeId: string,
  annotationId: string,
  vizConfig: Record<string, unknown>,
  svgData: string,
  narrative: string,
  category = "visualizations",
): Promise<{ status: string }> {
  return fetchJson(`${API_BASE}/genomes/${genomeId}/evidence/snapshot`, {
    method: "POST",
    body: JSON.stringify({
      annotation_id: annotationId,
      viz_config: vizConfig,
      svg_data: svgData,
      narrative,
      category,
    }),
  });
}

export async function updateNarrative(
  genomeId: string,
  annotationId: string,
  evidenceIndex: number,
  narrative: string,
): Promise<{ status: string }> {
  return fetchJson(`${API_BASE}/genomes/${genomeId}/evidence/narrative`, {
    method: "PUT",
    body: JSON.stringify({
      annotation_id: annotationId,
      evidence_index: evidenceIndex,
      narrative,
    }),
  });
}

export async function listEvidence(
  genomeId: string,
  annotationId: string,
): Promise<EvidenceListResponse> {
  return fetchJson<EvidenceListResponse>(
    `${API_BASE}/genomes/${genomeId}/evidence?annotation_id=${annotationId}`,
  );
}
