/**
 * API client for export packs (explaneat/api/routes/exports.py): the same deterministic
 * explaneat.export code as the CLI, run as a background job.
 */

export type ExportFormat = "json" | "tex" | "tikz" | "pdf" | "docx";
export const ALL_FORMATS: ExportFormat[] = ["json", "tex", "tikz", "pdf", "docx"];

export type Capability = { available: boolean; reason?: string | null; figures?: boolean; figures_reason?: string | null };
export type Capabilities = Record<ExportFormat, Capability>;

export type ExportFile = { name: string; size: number; url: string };
export type ExportManifest = {
  export_id: string;
  path: string;
  kind: "genome" | "experiment";
  label?: string;
  content_hash?: string;
  formats: Record<string, string> | Record<string, Record<string, string>>;
  failures?: { genome_id: string; error: string }[];
  n_genomes?: number;
  exported?: number;
  files: ExportFile[];
};
export type ExportJob = {
  job_id: string;
  kind: string;
  status: "running" | "done" | "failed";
  progress: { done: number; total: number } | null;
  log: string[];
  manifest: ExportManifest | null;
  error: string | null;
};

async function req<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, { ...init, headers: { "Content-Type": "application/json", ...(init?.headers || {}) } });
  if (!r.ok) {
    let detail = await r.text();
    try {
      const j = JSON.parse(detail);
      if (j && j.detail) detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {
      /* not JSON */
    }
    throw new Error(detail);
  }
  return r.json();
}

export const getExportCapabilities = () => req<Capabilities>("/api/exports/capabilities");

export const startExport = (
  target: { kind: "genome" | "experiment"; id: string },
  formats: ExportFormat[],
  reviewMarks: boolean,
) =>
  req<{ job_id: string; status: string }>(
    `/api/${target.kind === "genome" ? "genomes" : "experiments"}/${target.id}/export`,
    { method: "POST", body: JSON.stringify({ formats, review_marks: reviewMarks }) },
  );

export const getExportJob = (jobId: string) => req<ExportJob>(`/api/exports/jobs/${jobId}`);
