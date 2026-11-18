/**
 * Readable names of the model inputs ("feature key"): every place that names an input shows
 * both the raw column / node name (e.g. `marital-status:2`) and its decoded meaning from the
 * dataset codebook (e.g. "married (civilian spouse)"). Stored names are never changed.
 */
import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { getFeatureLabels, type FeatureLabel, type FeatureLabelsResponse } from "./api/client";

export type FeatureLabels = {
  data: FeatureLabelsResponse | null;
  /** Entry for a node id (input, split input `-31_a`, identity `identity_-31`), else null. */
  forNode: (nodeId: string | null | undefined) => FeatureLabel | null;
  /** Entry for a name: raw column name, the camelCase display name, or a node id. */
  forName: (name: string | null | undefined) => FeatureLabel | null;
};

const EMPTY: FeatureLabels = { data: null, forNode: () => null, forName: () => null };

const FeatureLabelsContext = createContext<FeatureLabels>(EMPTY);

/** The display name experiments/10seed/rename_inputs.py gives a feature ("total day charge" -> "totalDayCharge"). */
export function camelDisplay(feature: string): string {
  const i = feature.indexOf(":");
  const base = i >= 0 ? feature.slice(0, i) : feature;
  const sep = i >= 0 ? ":" : "";
  const value = i >= 0 ? feature.slice(i + 1) : "";
  const words = base.split(/\s+/).filter(Boolean);
  const b = words.length
    ? words[0] + words.slice(1).map((w) => w.slice(0, 1).toUpperCase() + w.slice(1)).join("")
    : base;
  return b + sep + value.replace(/ /g, "_");
}

/** The input node id behind a (split / identity) node id, or null if it is not an input. */
export function baseInputId(nodeId: string): string | null {
  let id = nodeId;
  if (id.startsWith("identity_")) id = id.slice("identity_".length);
  const head = id.split("_", 1)[0];
  return /^-\d+$/.test(head) ? head : null;
}

export function buildFeatureLabels(data: FeatureLabelsResponse | null): FeatureLabels {
  if (!data) return EMPTY;
  const byNode = new Map<string, FeatureLabel>();
  const byName = new Map<string, FeatureLabel>();
  for (const f of data.features) {
    byNode.set(f.node_id, f);
    byName.set(f.raw, f);
    byName.set(camelDisplay(f.raw), f);
  }
  const forNode = (nodeId: string | null | undefined) => {
    if (!nodeId) return null;
    const b = baseInputId(nodeId);
    return b ? byNode.get(b) ?? null : null;
  };
  const forName = (name: string | null | undefined) => {
    if (!name) return null;
    const hit = byName.get(name) ?? forNode(name);
    if (hit) return hit;
    // split copies are named <base>_<token> (tokens a..z, Aa..Zz, joined when consolidated)
    const m = name.match(/^(.*)_([A-Za-z]+)$/);
    return m ? byName.get(m[1]) ?? null : null;
  };
  return { data, forNode, forName };
}

/** Fetch once per genome. */
export function useFeatureLabelsLoader(genomeId: string | null | undefined): FeatureLabels {
  const [data, setData] = useState<FeatureLabelsResponse | null>(null);
  useEffect(() => {
    setData(null);
    if (!genomeId) return;
    let live = true;
    getFeatureLabels(genomeId)
      .then((d) => live && setData(d))
      .catch((e) => console.warn("[featureLabels] could not load", e));
    return () => {
      live = false;
    };
  }, [genomeId]);
  return useMemo(() => buildFeatureLabels(data), [data]);
}

export function FeatureLabelsProvider({ value, children }: { value: FeatureLabels; children: ReactNode }) {
  return <FeatureLabelsContext.Provider value={value}>{children}</FeatureLabelsContext.Provider>;
}

export function useFeatureLabels(): FeatureLabels {
  return useContext(FeatureLabelsContext);
}

/** `name · readable label` when the label adds something, else `name`. */
export function withLabel(name: string, entry: FeatureLabel | null | undefined): string {
  if (!entry || !entry.decoded || !entry.label || entry.label === name) return name;
  return `${name} · ${entry.label}`;
}

/** Multi-line description for a hover title: raw name, readable label, what 1 means. */
export function describe(entry: FeatureLabel): string {
  const lines = [`Feature: ${entry.raw}`];
  if (entry.decoded) lines.push(`Meaning: ${entry.full_label}`);
  if (entry.one_means) lines.push(`1 means: ${entry.one_means}`);
  if (entry.unit) lines.push(`Unit: ${entry.unit}`);
  return lines.join("\n");
}

/** A node id with its input feature when it is one: `-31 (marital-status:2 · married (civilian spouse))`. */
export function nodeWithFeature(id: string, fl: FeatureLabels): string {
  const f = fl.forNode(id);
  return f ? `${id} (${withLabel(f.raw, f)})` : id;
}

export function nodeListText(ids: string[], fl: FeatureLabels): string {
  return ids.map((id) => nodeWithFeature(id, fl)).join(", ");
}
