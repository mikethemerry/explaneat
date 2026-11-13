/**
 * API client for the review page (/review/<label>): the signoff-v1 queue, verdicts and sign-off.
 * Routes: explaneat/api/routes/review.py.
 */

const API_BASE = "/api/review";

export class ReviewApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = "ReviewApiError";
  }
}

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
    throw new ReviewApiError(r.status, detail);
  }
  return r.json();
}

export type SentenceVerdictKind = "confirmed" | "wrong" | "unsure";
export type LeafVerdictKind = "understood" | "partial" | "not";

export type VerdictState = {
  id: string;
  verdict: SentenceVerdictKind;
  note: string;
  at: string;
  disclosed: boolean;
  current: boolean;
  by: string | null;
};

export type VerifierFinding = {
  id: string | null;
  grade: string | null;
  level: string | null;
  location: string | null;
  statement: string | null;
  evidence: string | null;
  check: string | null;
  source: string;
  stale: boolean;
  why_stale: string;
};

export type QueueItem = {
  sid: string;
  ref: string;
  location: string;
  part: string | null;
  origin: string;
  tier: string;
  badge: string;
  score: number;
  section: "gate" | "tier A" | "sample B/C" | "sample A'";
  why: string[];
  gates: string[];
  text: string;
  paragraph_id: string;
  verifier: VerifierFinding[];
  verdict: VerdictState | null;
  change: string | null;
};

export type Paragraph = {
  id: string;
  title: string;
  origin: string;
  part: string | null;
  sentences: { sid: string; ref: string; text: string }[];
};

export type LeafRow = {
  id: string;
  name: string;
  hypothesis: string;
  probe: string | null;
  verdict: { id: string; verdict: LeafVerdictKind; note: string; at: string; current: boolean } | null;
};

export type Change = {
  sid: string;
  state: "changed" | "removed";
  ref: string | null;
  old: string | null;
  new: string | null;
  word_diff?: string;
  numbers?: [string, string][];
  verdict: string;
  verdict_at: string;
  why: string[];
};

export type SignoffResult = {
  label: string;
  protocol: string;
  passed: boolean;
  checked_at: string;
  content_hash: string;
  tiers: Record<string, number>;
  missing: { sid: string; ref: string; tier: string; why: string; stale: boolean }[];
  problems: string[];
  sample: Record<string, { items: number; drawn: string[]; extended: boolean; failed?: boolean }>;
  sample_n: number;
  sample_wrong: number;
  error_bound_95: number;
  counts: Record<SentenceVerdictKind, number>;
  wrong_fixed: number;
  unsure_disclosed: number;
  minutes: number;
  leaf_verdicts: {
    leaves: number;
    graded: number;
    counts: Record<LeafVerdictKind, number>;
    understood_mike: number | null;
    grades: Record<string, string | null>;
  } | null;
};

export type ReviewData = {
  label: string;
  genome_id: string | null;
  protocol: string;
  target_hash: string;
  records_dir: string;
  layers_dir: string;
  review_dir: string;
  store: string;
  experiment: { in_db: boolean; experiment_id?: string; experiment_name?: string };
  record: {
    summary: string | null;
    hidden_nodes: number | null;
    n_leaf_annotations: number | null;
    n_composition_annotations: number | null;
    closed_form_max_abs_err: number | null;
    mike_verification: { verified: boolean; minutes?: number; protocol?: string; recorded_at?: string } | null;
  };
  badges: Record<string, number>;
  tiers: Record<string, number>;
  recheck: Record<string, unknown> | null;
  queue: QueueItem[];
  document: Paragraph[];
  leaves: LeafRow[];
  changes: Change[];
  verdicts: { n: number; n_leaf: number; first_at: string | null; last_at: string | null; seconds: number };
  signoff: SignoffResult;
  unattached_verifier: VerifierFinding[];
};

export type ReviewProgress = {
  label: string;
  n_verdicts: number;
  n_sentences: number;
  n_leaf_verdicts: number;
  first_at: string | null;
  last_at: string | null;
  signoff_passed: boolean | null;
};

export const getReviewProgress = () =>
  req<{ review_dir: string; records_dir: string; protocol: string; labels: Record<string, ReviewProgress> }>(API_BASE);

export const getReview = (label: string) => req<ReviewData>(`${API_BASE}/${encodeURIComponent(label)}`);

export const postVerdict = (
  label: string,
  body: { sid: string; verdict: SentenceVerdictKind; note: string; disclose: boolean; seconds?: number | null },
) =>
  req<{ label: string; verdict: VerdictState; store: string }>(`${API_BASE}/${encodeURIComponent(label)}/verdict`, {
    method: "POST",
    body: JSON.stringify(body),
  });

export const postLeafVerdict = (label: string, body: { leaf: string; verdict: LeafVerdictKind; note: string; seconds?: number | null }) =>
  req<{ label: string; leaf: string; line: { id: string; verdict: LeafVerdictKind; note: string; at: string } }>(
    `${API_BASE}/${encodeURIComponent(label)}/leaf`,
    { method: "POST", body: JSON.stringify(body) },
  );

export const postSignoff = (
  label: string,
  body: { write: boolean; minutes?: number | null; changes?: string; notes?: string; require_leaves?: boolean },
) =>
  req<{ label: string; result: SignoffResult; signoff_file: string; record_written: boolean; record: string | null }>(
    `${API_BASE}/${encodeURIComponent(label)}/signoff`,
    { method: "POST", body: JSON.stringify(body) },
  );

export const postRebuildCsv = (label: string) =>
  req<{ ran: boolean; command: string; detail?: string; returncode?: number; stdout?: string; stderr?: string }>(
    `${API_BASE}/${encodeURIComponent(label)}/rebuild-csv`,
    { method: "POST", body: "{}" },
  );
