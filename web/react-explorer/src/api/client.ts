/**
 * API client for ExplaNEAT REST API
 */

// Use relative URL when proxying through Vite dev server
// Falls back to localhost:8000 for direct access
const API_BASE = "/api";

// ============================================================================
// Types matching API schemas
// ============================================================================

export type NodeType = "input" | "hidden" | "output" | "identity";

export type ApiNode = {
  id: string;
  type: NodeType;
  bias: number | null;
  activation: string | null;
  response: number | null;
  aggregation: string | null;
};

export type ApiConnection = {
  from: string;
  to: string;
  weight: number;
  enabled: boolean;
};

export type ModelMetadata = {
  input_nodes: string[];
  output_nodes: string[];
  is_original: boolean;
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

// Operation request types
export type SplitNodeParams = { node_id: string };
export type ConsolidateNodeParams = { node_ids: string[] };
export type RemoveNodeParams = { node_id: string };
export type AddNodeParams = { connection: [string, string] };
export type AddIdentityNodeParams = {
  target_node: string;
  intercepted_connections: [string, string][];
};
export type AnnotateParams = {
  name: string;
  hypothesis?: string;
  entry_nodes: string[];
  exit_nodes: string[];
  subgraph_nodes: string[];
  subgraph_connections: [string, string][];
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
  const response = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });

  if (!response.ok) {
    const text = await response.text();
    throw new ApiError(response.status, text);
  }

  return response.json();
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

export async function getCurrentModel(genomeId: string): Promise<ModelState> {
  return fetchJson<ModelState>(`${API_BASE}/genomes/${genomeId}/model`);
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
