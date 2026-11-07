/** Client for /api/portal (CSV datasets, seed batches, results, hazard ratios). */

const B = "/api/portal";

async function req<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, { ...init, headers: { "Content-Type": "application/json", ...(init?.headers || {}) } });
  if (!r.ok) {
    let d = await r.text();
    try {
      const j = JSON.parse(d);
      if (j && j.detail) d = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {
      /* not JSON */
    }
    throw new Error(d);
  }
  return r.json();
}

export type Task = "survival" | "classification";

export type ColumnInfo = {
  name: string;
  dtype: string;
  numeric: boolean;
  n_missing: number;
  n_unique: number;
  min?: number;
  max?: number;
  levels?: string[];
  suggest: "" | "time" | "event" | "categorical";
};

export const previewCsv = (path: string) =>
  req<{ file: string; n_rows: number; columns: ColumnInfo[] }>(`${B}/csv/preview`, {
    method: "POST",
    body: JSON.stringify({ path }),
  });

export type ImportCsvRequest = {
  path: string;
  name: string;
  task: Task;
  features?: string[];
  categorical?: string[];
  impute?: "median" | null;
  time_col?: string;
  event_col?: string;
  event_value?: string | number;
  target_col?: string;
  positive_value?: string | number;
  time_unit?: string;
  restricted: boolean;
  test_size: number;
  val_size: number;
  split_seed: number;
  horizons?: number[] | null;
  calibration_horizon?: number | null;
};

export type ImportCsvResult = {
  dataset_id: string;
  split_id: string;
  task: Task;
  n_rows: number;
  n_features: number;
  feature_names: string[];
  n_events_or_positives: number;
  restricted: boolean;
  train: number;
  val: number;
  test: number;
};

export const importCsv = (r: ImportCsvRequest) =>
  req<ImportCsvResult>(`${B}/datasets/csv`, { method: "POST", body: JSON.stringify(r) });

export type PortalDataset = {
  id: string;
  name: string;
  task: Task;
  restricted: boolean;
  n_rows: number | null;
  n_features: number | null;
  time_unit?: string | null;
  splits: { id: string; train: number | null; val: number | null; test: number | null; seed: number | null; has_val: boolean }[];
};

export const listPortalDatasets = () => req<{ datasets: PortalDataset[] }>(`${B}/datasets`);

export type Template = {
  name: string;
  task: Task;
  title: string;
  description: string;
  defaults: { generations: number; pop_size: number; epochs: number; complexity_penalty: number; test_size: number; val_size: number };
  ini_text: string;
};

export const listTemplates = () => req<{ templates: Template[] }>(`${B}/templates`);

export type BatchRequest = {
  task: Task;
  dataset_split_id: string;
  seeds: number[];
  template?: string;
  name_prefix: string;
  generations?: number;
  pop_size?: number;
  epochs?: number;
  complexity_penalty?: number;
  max_concurrent: number;
  config_overrides?: Record<string, string | number>;
  fitness?: "auc" | "bce";
};

export type RunState = {
  run_id: string;
  seed: number | null;
  status: string;
  experiment_id: string | null;
  generation: number | null;
  generations_done: number;
  total_generations: number;
  best_fitness: number | null;
  mean_fitness: number | null;
  best_val_score: number | null;
  selected: { genome_id: string | null; generation: number; rule: string; val_score?: number } | null;
  error: string | null;
};

export type Batch = {
  batch_id: string;
  task: Task;
  template: string;
  name_prefix: string;
  dataset_split_id: string;
  seeds: number[];
  generations: number;
  pop_size: number;
  max_concurrent: number;
  created_at: number;
  runs: RunState[];
  counts: Record<string, number>;
  done: boolean;
};

export const createBatch = (r: BatchRequest) => req<Batch>(`${B}/batches`, { method: "POST", body: JSON.stringify(r) });
export const listBatches = () =>
  req<{ batches: Omit<Batch, "runs">[]; jobs_dir: string }>(`${B}/batches`);
export const getBatch = (id: string) => req<Batch>(`${B}/batches/${id}`);
export const cancelBatch = (id: string, runId?: string) =>
  req<Batch>(`${B}/batches/${id}/cancel${runId ? `?run_id=${runId}` : ""}`, { method: "POST" });
export const resumeBatch = (id: string, runId?: string) =>
  req<Batch>(`${B}/batches/${id}/resume${runId ? `?run_id=${runId}` : ""}`, { method: "POST" });

export type Interval = { n: number; mean: number | null; lo: number | null; hi: number | null; sd: number | null };

export type ResultRow = {
  experiment_id: string;
  experiment_name: string;
  seed: number | null;
  task: Task;
  selected: { genome_id: string; rule: string; generation?: number; train_fitness?: number };
  structure: Record<string, number | null>;
  metrics: Record<string, number | null>;
  primary: string;
  primary_ci: [number | null, number | null];
  comparator_primary: number | null;
  comparator_primary_ci: [number | null, number | null];
  delta_primary: number | null;
  delta_primary_ci: [number | null, number | null];
};

export type Results = {
  task: Task | null;
  primary: string;
  rows: ResultRow[];
  comparators: Record<string, { info: Record<string, unknown>; metrics: Record<string, number | null> }>;
  summary: Record<string, Interval>;
  errors: { experiment_id: string; error: string }[];
  pending?: string[];
  n_boot: number;
};

export const getResults = (id: string, refresh = false) =>
  req<Results>(`${B}/batches/${id}/results${refresh ? "?refresh=true" : ""}`);
export const resultsCsvUrl = (id: string) => `${B}/batches/${id}/results.csv`;
export const reportMdUrl = (id: string) => `${B}/batches/${id}/report.md`;

// ------------------------------------------------------------- survival evidence

export type SurvivalPerf = {
  harrell_c: number | null;
  uno_c: number | null;
  td_auc?: { times: number[]; auc: number[]; mean_auc: number | null };
  ibs?: number | null;
  cox_loss?: number | null;
  null_cox_loss?: number | null;
  censored_fraction?: number | null;
  n_test?: number;
  events_test?: number;
  calibration?: {
    horizon: number;
    groups: { mean_predicted: number; observed_km: number; n?: number }[];
    expected_over_observed: number | null;
    max_abs_group_diff: number | null;
  };
  log_risk?: { hr_p95_vs_p5: number };
  time_unit?: string;
  ph_caveat?: string;
};

export type HazardRatios = {
  genome_id: string;
  time_unit: string | null;
  log_risk_spread: Record<string, number>;
  terms: {
    node: string;
    feature?: string;
    hr_per_sd?: number;
    hr_per_unit?: number;
    reading?: string;
    spread?: Record<string, number>;
  }[];
  annotation_terms: { annotation: string; exit: string; output_weight: number; spread: Record<string, number> }[];
  note: string;
};

export const getHazardRatios = (genomeId: string) => req<HazardRatios>(`${B}/genomes/${genomeId}/hazard-ratios`);
