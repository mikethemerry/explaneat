/**
 * Client for /api/features: "features of interest" pinned while exploring a model.
 * Pins are files under $EXPLANEAT_FEATURES_DIR (default ~/.explaneat/features), never DB rows.
 */
const API_BASE = "/api/features";

export type PinStatus = "open" | "working" | "explained" | "dropped";
export const PIN_STATUSES: PinStatus[] = ["open", "working", "explained", "dropped"];

export type PinScope = {
  kind: "model" | "annotation" | "node" | "input";
  annotation_id?: string | null;
  annotation_name?: string | null;
  node_id?: string | null;
};

export type PinInput = { node_id?: string | null; raw?: string | null; label?: string | null; index?: number | null };

/** One visualisation exactly as it was on screen (reopened by the evidence panel). */
export type PinView = {
  type: string;
  params: Record<string, unknown>;
  split_choice?: "train" | "test" | "val" | "both" | null;
  sample_fraction?: number | null;
  view?: "network" | "source" | null;
  dataset_split_id?: string | null;
  input_nodes?: string[] | null;
  output_node?: string | null;
  description?: string | null;
};

export type FeaturePin = {
  id: string;
  version: number;
  n_versions?: number;
  created_at: string;
  updated_at: string;
  edit?: string;
  genome_id: string;
  experiment_id?: string | null;
  experiment_name?: string | null;
  record_label?: string | null;
  scope: PinScope;
  inputs: PinInput[];
  viz?: PinView | null;
  also_views: PinView[];
  note: string;
  tags: string[];
  status: PinStatus;
  dossier_path?: string | null;
  dossier_exists?: boolean;
  dossier_command?: string | null;
  history?: FeaturePin[];
};

export type PinDraft = Omit<FeaturePin, "id" | "version" | "created_at" | "updated_at">;

async function req<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, { ...init, headers: { "Content-Type": "application/json", ...(init?.headers || {}) } });
  if (!r.ok) {
    let detail = await r.text();
    try {
      const j = JSON.parse(detail);
      if (j && typeof j.detail === "string") detail = j.detail;
    } catch {
      /* not JSON */
    }
    throw new Error(detail || `HTTP ${r.status}`);
  }
  return r.json();
}

export type PinFilter = { genome_id?: string; label?: string; tag?: string; status?: string; q?: string };

export function listPins(f: PinFilter = {}): Promise<{ features_dir: string; statuses: string[]; pins: FeaturePin[] }> {
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(f)) if (v) qs.set(k, v);
  const s = qs.toString();
  return req(`${API_BASE}${s ? `?${s}` : ""}`);
}

export function getPin(id: string): Promise<FeaturePin> {
  return req(`${API_BASE}/${encodeURIComponent(id)}`);
}

export function createPin(d: Partial<PinDraft> & { genome_id: string }): Promise<FeaturePin> {
  return req(API_BASE, { method: "POST", body: JSON.stringify(d) });
}

export function updatePin(id: string, changes: Partial<PinDraft> & { edit?: string }): Promise<FeaturePin> {
  return req(`${API_BASE}/${encodeURIComponent(id)}`, { method: "PATCH", body: JSON.stringify(changes) });
}

export type DossierInfo = { path: string | null; resolved: string | null; exists: boolean; files: string[]; command: string | null };

export function getDossier(id: string): Promise<DossierInfo> {
  return req(`${API_BASE}/${encodeURIComponent(id)}/dossier`);
}

export const dossierFileUrl = (id: string, file: string) =>
  `${API_BASE}/${encodeURIComponent(id)}/dossier/${file.split("/").map(encodeURIComponent).join("/")}`;

/** Suggested dossier folder, matching `explaneat dossier`: experiments/feature_dossiers/<label>_<input>/ */
export function suggestedDossierPath(label: string | null | undefined, input: string | null | undefined): string {
  if (!label || !input) return "";
  return `experiments/feature_dossiers/${label}_${input.replace(/[^A-Za-z0-9._-]+/g, "_")}/`;
}

export const VIEW_LABELS: Record<string, string> = {
  line: "Line plot",
  heatmap: "Heatmap",
  partial_dependence: "Partial dependence",
  pca_scatter: "PCA scatter",
  sensitivity: "Sensitivity",
  ice: "ICE plot",
  feature_output_scatter: "Feature vs output",
  output_distribution: "Output distribution",
  shap: "SHAP values",
  activation_profile: "Activation profile",
  regime_map: "Regime map",
  edge_influence: "Edge influence",
  pairs_plot: "Pairs plot",
  input_distribution: "Input distribution",
};

export function scopeText(s: PinScope | undefined | null): string {
  if (!s || s.kind === "model") return "whole model";
  if (s.kind === "annotation") return `annotation ${s.annotation_name || s.annotation_id}`;
  if (s.kind === "node") return `node ${s.node_id}`;
  return `input ${s.node_id ?? ""}`.trim();
}

export function inputText(i: PinInput): string {
  const raw = i.raw || i.node_id || "?";
  return i.label && i.label !== raw ? `${raw} · ${i.label}` : raw;
}
