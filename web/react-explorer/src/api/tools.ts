/**
 * API client for the explanation tools, explanation records and lineage routes.
 */

const API_BASE = "/api";

export class ToolsApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = "ToolsApiError";
  }
}

async function req<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
  });
  if (!r.ok) {
    let detail = await r.text();
    try {
      const j = JSON.parse(detail);
      if (j && typeof j.detail === "string") detail = j.detail;
    } catch {
      /* not JSON */
    }
    throw new ToolsApiError(r.status, detail);
  }
  return r.json();
}

const g = (genomeId: string) => `${API_BASE}/genomes/${genomeId}`;

// ----------------------------------------------------------------- checks

export type Hierarchy = {
  n_annotations: number;
  roots: string[];
  single_root: boolean;
  root_contains_outputs: boolean;
  all_under_root: boolean;
  not_under_root: string[];
  complete: boolean;
};

export type CoverageReport = {
  structural_coverage: number;
  compositional_coverage: number;
  node_coverage: {
    covered: string[];
    uncovered: string[];
    total_candidate: number;
    by_annotation: Record<string, string[]>;
  };
  edge_coverage: { covered: [string, string][]; uncovered: [string, string][]; total: number };
  annotations_count: number;
  leaf_count: number;
  composition_count: number;
  hierarchy?: {
    roots: string[];
    global_root: string | null;
    root_contains_outputs: boolean;
    complete: boolean;
    composed: number;
    required: number;
  };
};

export type LeafCheck = {
  id: string;
  name: string;
  numeric_max_abs_err: number | null;
  closed_form_max_abs_err: number | string | null;
  exit_scale?: number;
  error?: string;
};

export type CheckResult = {
  n_rows: number;
  function_preservation_max_abs_err: number;
  function_preserved: boolean;
  hierarchy: Hierarchy;
  op_counts: Record<string, number>;
  leaves: LeafCheck[];
  closed_form_max_abs_err: number | null;
  numeric_max_abs_err: number | null;
  coverage: CoverageReport;
  full_coverage?: boolean;
  split?: string;
};

export const getCheck = (genomeId: string, closedForm = true) =>
  req<CheckResult>(`${g(genomeId)}/tools/check?closed_form=${closedForm}`);

// ----------------------------------------------------------------- linearise

export type UnitLinearisation = {
  nonlinear: boolean;
  max_abs_residual_fit: number;
  min_r2_fit: number;
  n_entries: number;
  n_exits: number;
  linearised_alone: number;
  linearised_alone_drop: number;
  restore_alone: number;
  restore_gain: number;
};

export type LineariseResult = {
  full: number;
  all_linear: number;
  nonlinear_units: string[];
  units: Record<string, UnitLinearisation>;
  scope: string;
  n_rows: number;
  metric: string;
  split: string;
};

export const linearise = (genomeId: string, annotationNames?: string[]) =>
  req<LineariseResult>(`${g(genomeId)}/tools/linearise`, {
    method: "POST",
    body: JSON.stringify({ annotation_names: annotationNames || null, split: "train" }),
  });

// ----------------------------------------------------------------- unit activity

export type UnitRow = {
  node_id: string;
  display_name: string;
  status: "dead" | "rare" | "always_on" | "hinge" | "identity";
  active_rows: number;
  n_rows: number;
  active_fraction: number;
  mean: number;
  std: number;
  direct_output_weight: number;
  direct_logit_sd: number;
  n_successors: number;
  annotation: string | null;
};

export const getUnitActivity = (genomeId: string) =>
  req<{ n_rows: number; units: UnitRow[] }>(`${g(genomeId)}/tools/unit-activity`);

// ----------------------------------------------------------------- candidates

export type Candidate = {
  kind: "hidden_gate" | "chain";
  title: string;
  entry_nodes: string[];
  exit_nodes: string[];
  subgraph_nodes: string[];
  entry_names: string[];
  splits_needed: string[];
  already_covered: string[];
  clean: boolean;
};

export type CandidatesResult = {
  candidates: Candidate[];
  direct_inputs: { node_id: string; name: string }[];
  unconnected_inputs: { node_id: string; name: string }[];
  output_nodes: string[];
};

export const getCandidates = (genomeId: string) =>
  req<CandidatesResult>(`${g(genomeId)}/tools/candidates`);

// ----------------------------------------------------------------- category offsets

export type CategoryLevel = {
  feature: string;
  column: number;
  rows: number;
  positives: number;
  rate: number;
  model_mean_p: number;
  model_logit_effect: number;
  empirical_logodds_offset: number;
  warranted_offset: number;
  rate_ci: [number, number];
  supported: boolean;
  flag: "" | "large" | "large_unsupported";
};

export type CategoryGroup = {
  base_feature: string;
  n_levels: number;
  n_levels_used: number;
  max_abs_effect: number;
  n_large_unsupported: number;
  levels: CategoryLevel[];
};

export const getCategoryOffsets = (genomeId: string) =>
  req<{ n_rows: number; large_effect: number; groups: CategoryGroup[] }>(
    `${g(genomeId)}/tools/category-offsets`,
  );

// ----------------------------------------------------------------- hypothesis

export const getHypothesis = (genomeId: string, annotationId: string) =>
  req<{ annotation_id: string; name: string; hypothesis: string; evidence_counts: Record<string, number> }>(
    `${g(genomeId)}/tools/annotations/${encodeURIComponent(annotationId)}/hypothesis`,
  );

export const putHypothesis = (genomeId: string, annotationId: string, hypothesis: string) =>
  req<{ status: string }>(`${g(genomeId)}/tools/annotations/${encodeURIComponent(annotationId)}/hypothesis`, {
    method: "PUT",
    body: JSON.stringify({ hypothesis }),
  });

// ----------------------------------------------------------------- records

export type MikeVerification = {
  verified: boolean;
  minutes?: number;
  changes?: string;
  notes?: string;
  recorded_at?: string;
};

export type RecordSummary = {
  label: string;
  genome_id: string;
  full_coverage: boolean | null;
  orchestrator_full_coverage: boolean | null;
  agrees_with_agent: boolean | null;
  n_leaf_annotations: number | null;
  n_composition_annotations: number | null;
  hidden_nodes: number | null;
  closed_form_max_abs_err: number | null;
  function_preservation_max_abs_err: number | null;
  summary: string | null;
  verified: boolean;
  mike_verification: MikeVerification | null;
  in_db?: boolean;
  experiment_id?: string;
  experiment_name?: string;
};

export type ExplanationRecord = Record<string, unknown> & {
  label: string;
  genome_id: string;
  summary?: string;
  resisted?: string;
  tool_bugs?: string;
  mike_verification?: MikeVerification;
};

export const listRecords = () =>
  req<{ records_dir: string; records: RecordSummary[]; n_verified: number }>(`${API_BASE}/records`);

export const getGenomeRecord = (genomeId: string) =>
  req<{ record: ExplanationRecord | null; records_dir: string }>(`${g(genomeId)}/tools/record`);

export const putVerification = (label: string, v: { verified: boolean; minutes?: number | null; changes: string; notes?: string }) =>
  req<{ label: string; mike_verification: MikeVerification }>(
    `${API_BASE}/records/${encodeURIComponent(label)}/verification`,
    { method: "PUT", body: JSON.stringify(v) },
  );

// ----------------------------------------------------------------- import / export

export const exportOperationsUrl = (genomeId: string) => `${g(genomeId)}/tools/operations/export`;

export type ImportPreview = CheckResult & {
  n_operations: number;
  n_existing_operations: number;
  identical_to_existing: boolean;
};

export const importPreview = (genomeId: string, doc: unknown) =>
  req<ImportPreview>(`${g(genomeId)}/tools/operations/import-preview`, {
    method: "POST",
    body: JSON.stringify(Array.isArray(doc) ? { operations: doc } : { document: doc }),
  });

export const importOperations = (genomeId: string, doc: unknown, replace: boolean) =>
  req<{ status: string; n_operations: number }>(`${g(genomeId)}/tools/operations/import`, {
    method: "POST",
    body: JSON.stringify(Array.isArray(doc) ? { operations: doc, replace } : { document: doc, replace }),
  });

// ----------------------------------------------------------------- lineage

export type LineageRow = {
  generation: number;
  genome_id: string;
  genome_key: number;
  fitness: number | null;
  train_auc: number | null;
  n_nodes: number | null;
  n_connections: number | null;
  n_enabled_connections: number | null;
  parent1_id: string | null;
  parent2_id: string | null;
  is_elite_copy: boolean;
  followed: string | null;
  structural_change: boolean;
};

export type PopulationStat = {
  generation: number;
  best_fitness: number | null;
  mean_fitness: number | null;
  stdev_fitness: number | null;
  population_size: number;
};

export type LineageResponse = {
  genome_id: string;
  experiment_id: string;
  experiment_name: string;
  generation: number;
  mode: string;
  available: boolean;
  message: string | null;
  lineage: LineageRow[];
  population: PopulationStat[];
};

export type Jump = {
  generation: number;
  delta: number;
  before_genome_id: string;
  after_genome_id: string;
  structural_change: boolean;
  is_elite_copy: boolean;
  diff_summary: Record<string, number | null>;
  added_nodes: number[];
  added_connections: string[];
  annotations: Record<string, string[]>;
  genes_not_in_final?: string[];
  annotation_error?: string;
};

export const getLineage = (genomeId: string, mode = "fitter_parent") =>
  req<LineageResponse>(`${g(genomeId)}/lineage?mode=${mode}`);

export const getJumps = (genomeId: string, mode = "fitter_parent", method = "threshold", k = 5, minDelta?: number) =>
  req<{ available: boolean; message?: string; jumps: Jump[] }>(
    `${g(genomeId)}/lineage/jumps?mode=${mode}&method=${method}&k=${k}${minDelta != null ? `&min_delta=${minDelta}` : ""}`,
  );

export type AncestorResponse = {
  generation: number;
  ancestor: LineageRow;
  generation_best: { genome_id: string; genome_key: number; fitness: number | null; n_nodes: number; n_connections: number };
  ancestor_is_generation_best: boolean;
  ancestor_rank: number;
  population_size: number;
  fitness_gap: number | null;
};

export const getAncestor = (genomeId: string, generation: number, mode = "fitter_parent") =>
  req<AncestorResponse>(`${g(genomeId)}/ancestor?generation=${generation}&mode=${mode}`);

export type DiffStatus = "common" | "added" | "removed";

export type DiffGraph = {
  genome_a: string;
  genome_b: string;
  fitness_a: number | null;
  fitness_b: number | null;
  nodes: {
    id: string;
    type: "input" | "hidden" | "output" | "identity" | "function";
    bias: number | null;
    activation: string | null;
    response: number | null;
    aggregation: string | null;
    display_name: string | null;
    status: DiffStatus;
    bias_a: number | null;
    bias_b: number | null;
  }[];
  connections: {
    from: string;
    to: string;
    weight: number;
    enabled: boolean;
    status: DiffStatus;
    weight_a: number | null;
    weight_b: number | null;
  }[];
  metadata: { input_nodes: string[]; output_nodes: string[]; is_original: boolean };
};

export const getDiffGraph = (a: string, b: string, labelsFrom?: string) =>
  req<DiffGraph>(`${g(a)}/diff/${b}/graph${labelsFrom ? `?labels_from=${labelsFrom}` : ""}`);
