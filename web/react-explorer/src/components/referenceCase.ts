/**
 * The reference case: one full input row that the "others held fixed" plots (line, heatmap,
 * partial dependence, feature vs output, pairs at median) hold the inputs not on their axes at.
 *
 * Off (the default), the plots hold each input at its median, as before. On, every plot of the
 * genome (whole model and every part) holds them at the same case: the server forward-passes the
 * case to each part's entry nodes. One case per genome, shared by all evidence slots, kept in the
 * URL as `?case=<spec JSON>` so a view can be reopened. Pins and snapshots store it in their
 * params as `reference: {x, spec}`.
 */
import { useSyncExternalStore } from "react";
import type { CaseSpec } from "../api/client";

export type CaseState = {
  genomeId: string | null;
  enabled: boolean;
  spec: CaseSpec;
  /** the resolved case, raw units per dataset column (null until resolved) */
  x: number[] | null;
};

/** Plots that hold the inputs not on their axes at one point (the others ignore the case). */
export const USES_REFERENCE = new Set(["line", "heatmap", "partial_dependence", "feature_output_scatter", "pairs_plot"]);

const INITIAL: CaseState = { genomeId: null, enabled: false, spec: { base: "median" }, x: null };
let state: CaseState = INITIAL;
const listeners = new Set<() => void>();

function emit() {
  for (const l of listeners) l();
}

export function getCase(): CaseState {
  return state;
}

export function setCase(patch: Partial<CaseState>) {
  state = { ...state, ...patch };
  writeUrl();
  emit();
}

/** Start (or keep) the case for a genome: a different genome starts from the URL or off. */
export function bindGenome(genomeId: string) {
  if (state.genomeId === genomeId) return;
  const fromUrl = readUrl();
  state = fromUrl ? { genomeId, enabled: true, spec: fromUrl, x: null } : { ...INITIAL, genomeId };
  emit();
}

export function useReferenceCase(): CaseState {
  return useSyncExternalStore(
    (l) => {
      listeners.add(l);
      return () => listeners.delete(l);
    },
    getCase,
    getCase,
  );
}

/** `params.reference` for a viz request, or undefined (case off, not resolved, or unused). */
export function referenceParam(c: CaseState, vizType: string, pairFix?: string): { x: number[]; spec: CaseSpec } | undefined {
  if (!c.enabled || !c.x || !USES_REFERENCE.has(vizType)) return undefined;
  if (vizType === "pairs_plot" && pairFix === "average") return undefined;
  return { x: c.x, spec: c.spec };
}

// ── spec edits (pure) ─────────────────────────────────────────────────────

export function withOverride(spec: CaseSpec, raw: string, value: number | null): CaseSpec {
  const overrides = { ...(spec.overrides ?? {}) };
  if (value == null || !Number.isFinite(value)) delete overrides[raw];
  else overrides[raw] = value;
  return { ...spec, overrides };
}

export function withGroup(spec: CaseSpec, base: string, level: string | null | undefined): CaseSpec {
  const groups = { ...(spec.groups ?? {}) };
  if (level === undefined) delete groups[base];
  else groups[base] = level;
  // a group choice replaces overrides of its own columns
  const overrides = Object.fromEntries(
    Object.entries(spec.overrides ?? {}).filter(([k]) => !k.startsWith(`${base}:`)),
  );
  return { ...spec, groups, overrides };
}

/** A new base clears the edits (they were relative to the old base). */
export function withBase(base: NonNullable<CaseSpec["base"]>, extra: Partial<CaseSpec> = {}): CaseSpec {
  return { base, ...extra };
}

export function describeSpec(spec: CaseSpec): string {
  const n = Object.keys(spec.overrides ?? {}).length + Object.keys(spec.groups ?? {}).length;
  const base =
    spec.base === "row" ? `row #${spec.row}`
    : spec.base === "quantile" ? `${Math.round((spec.q ?? 0.5) * 100)}th percentile case`
    : spec.base === "mean" ? "mean case"
    : spec.base === "per_column_median" ? "per-column median"
    : "median case";
  return n ? `${base}, ${n} edit${n > 1 ? "s" : ""}` : base;
}

// ── URL (?case=) ──────────────────────────────────────────────────────────

export function encodeSpec(spec: CaseSpec): string {
  const s: CaseSpec = { base: spec.base ?? "median" };
  if (spec.q != null && spec.base === "quantile") s.q = spec.q;
  if (spec.row != null && spec.base === "row") s.row = spec.row;
  if (spec.overrides && Object.keys(spec.overrides).length) s.overrides = spec.overrides;
  if (spec.groups && Object.keys(spec.groups).length) s.groups = spec.groups;
  return JSON.stringify(s);
}

export function decodeSpec(text: string | null): CaseSpec | null {
  if (!text) return null;
  try {
    const v = JSON.parse(text);
    if (!v || typeof v !== "object" || Array.isArray(v)) return null;
    const bases = ["median", "per_column_median", "mean", "quantile", "row"];
    if (v.base != null && !bases.includes(v.base)) return null;
    return v as CaseSpec;
  } catch {
    return null;
  }
}

/** The routed query string (the hash's in the static appendix). */
function searchParams(): URLSearchParams {
  const h = window.location.hash;
  if (h.startsWith("#/")) {
    const q = h.indexOf("?");
    return new URLSearchParams(q < 0 ? "" : h.slice(q));
  }
  return new URLSearchParams(window.location.search);
}

function readUrl(): CaseSpec | null {
  if (typeof window === "undefined") return null;
  return decodeSpec(searchParams().get("case"));
}

function writeUrl() {
  if (typeof window === "undefined" || window.location.hash.startsWith("#/")) return;
  const url = new URL(window.location.href);
  if (state.enabled) url.searchParams.set("case", encodeSpec(state.spec));
  else url.searchParams.delete("case");
  if (url.href !== window.location.href) window.history.replaceState(window.history.state, "", url.href);
}

/** The `case` query value to keep when the explorer rewrites its URL (same genome), else null. */
export function caseForUrl(genomeId: string): string | null {
  return state.genomeId === genomeId && state.enabled ? encodeSpec(state.spec) : null;
}
