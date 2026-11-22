/**
 * Colour by part: which annotation each node of the graph belongs to, as a colour.
 * Pure functions (no React, no fetch), so the static build needs nothing more, and
 * the logic is unit-tested from Python (tests/test_ui/test_annotation_colours_ts.py).
 *
 * Owner of a node: the deepest (leaf-most) annotation whose own part holds it as an
 * exit or intermediate node, i.e. `exit_nodes ∪ (subgraph_nodes \ entry_nodes)`. A
 * composition owns the nodes none of its children owns (its combining node). Ties at
 * the same depth go to the first annotation in tree order.
 * Entry-only nodes (owned by no annotation, but an entry of one or more) keep their
 * type fill and get a ring per annotation they enter (the leaf-most ones: a
 * composition is dropped when one of its descendants lists the same entry).
 * Nodes in no annotation at all are neutral grey.
 * A collapsed annotation's function node (`fn_<name>`) takes that annotation's colour.
 * An edge u -> v is drawn in the colour of v's owner A when u lies in A's part
 * (A's nodes and its descendants'); other edges are muted.
 * Colours follow the annotation tree order (AnnotationListPanel's), so the same model
 * always gets the same colours.
 */
import type { AnnotationSummary, ApiConnection, ModelState } from "../api/client";
import { annotationLabel, descendantsOf, focusNodeIds } from "./annotationFocus";

/**
 * Okabe–Ito (black left out: it is no fill), then Paul Tol's "muted" set (its grey
 * left out: grey means "no part"). Past 16 parts the colours repeat.
 */
export const PART_PALETTE: string[] = [
  "#E69F00", // orange
  "#56B4E9", // sky blue
  "#009E73", // bluish green
  "#F0E442", // yellow
  "#0072B2", // blue
  "#D55E00", // vermillion
  "#CC79A7", // reddish purple
  "#332288", // indigo
  "#DDCC77", // sand
  "#117733", // green
  "#88CCEE", // cyan
  "#882255", // wine
  "#44AA99", // teal
  "#999933", // olive
  "#AA4499", // purple
  "#CC6677", // rose
];

/** Fill and border of a node in no annotation. */
export const NO_PART = { fill: "#E5E7EB", border: "#9CA3AF", text: "#374151" };

export type TreeEntry = { annotation: AnnotationSummary; depth: number; isLeaf: boolean };

export type PartColour = {
  id: string;
  name: string | null;
  label: string;
  colour: string;
  /** A darker shade for borders. */
  border: string;
  /** Black or white, whichever reads on `colour`. */
  text: string;
  depth: number;
  isLeaf: boolean;
};

export type NodePart = {
  /** Annotation id owning the node, or null (entry-only or no part). */
  owner: string | null;
  /** Annotations the node is an entry of (leaf-most, tree order). */
  enters: string[];
};

export type PartColouring = {
  /** Legend rows in tree order. */
  legend: PartColour[];
  byId: Record<string, PartColour>;
  nodes: Record<string, NodePart>;
  /** Edge key `from->to` -> annotation id whose colour the edge takes. */
  edges: Record<string, string>;
};

// ── colour helpers ────────────────────────────────────────────────────────

function rgb(hex: string): [number, number, number] {
  const h = hex.replace("#", "");
  return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)) as [number, number, number];
}

/** WCAG relative luminance of a #rrggbb colour. */
export function luminance(hex: string): number {
  const [r, g, b] = rgb(hex).map((v) => {
    const c = v / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/** "#000000" or "#ffffff", whichever has the higher contrast on `hex`. */
export function textOn(hex: string): string {
  const l = luminance(hex);
  return (l + 0.05) / 0.05 >= 1.05 / (l + 0.05) ? "#000000" : "#ffffff";
}

/** `hex` darkened by `f` (0..1). */
export function darken(hex: string, f = 0.3): string {
  return "#" + rgb(hex).map((v) => Math.round(v * (1 - f)).toString(16).padStart(2, "0")).join("");
}

export function paletteColour(i: number): string {
  return PART_PALETTE[i % PART_PALETTE.length];
}

// ── tree ──────────────────────────────────────────────────────────────────

/**
 * The annotation tree in display order: roots in list order, each followed by its
 * children (by parent_annotation_id or children_ids) in list order. Cycle-safe;
 * anything unreachable is appended at depth 0.
 */
export function annotationTree(annotations: AnnotationSummary[]): TreeEntry[] {
  const ids = new Set(annotations.map((a) => a.id));
  const kids = new Map<string, AnnotationSummary[]>();
  const hasParent = new Set<string>();
  const add = (parent: string, child: AnnotationSummary) => {
    if (!ids.has(parent) || parent === child.id) return;
    const list = kids.get(parent) ?? [];
    if (!list.includes(child)) list.push(child);
    kids.set(parent, list);
    hasParent.add(child.id);
  };
  const byId = new Map(annotations.map((a) => [a.id, a]));
  for (const a of annotations) {
    if (a.parent_annotation_id) add(a.parent_annotation_id, a);
    for (const c of a.children_ids ?? []) {
      const child = byId.get(c);
      if (child) add(a.id, child);
    }
  }
  // children in list order
  const pos = new Map(annotations.map((a, i) => [a.id, i]));
  for (const list of kids.values()) list.sort((x, y) => pos.get(x.id)! - pos.get(y.id)!);

  const out: TreeEntry[] = [];
  const seen = new Set<string>();
  const walk = (a: AnnotationSummary, depth: number) => {
    if (seen.has(a.id)) return;
    seen.add(a.id);
    const children = kids.get(a.id) ?? [];
    out.push({ annotation: a, depth, isLeaf: children.length === 0 });
    for (const c of children) walk(c, depth + 1);
  };
  for (const a of annotations) if (!hasParent.has(a.id)) walk(a, 0);
  for (const a of annotations) walk(a, 0);
  return out;
}

/** The nodes an annotation owns before deeper annotations take theirs. */
export function ownedNodes(a: AnnotationSummary): string[] {
  const entries = new Set(a.entry_nodes);
  const out = new Set<string>(a.exit_nodes);
  for (const n of a.subgraph_nodes) if (!entries.has(n)) out.add(n);
  return [...out];
}

// ── colouring ─────────────────────────────────────────────────────────────

export function computePartColouring(model: ModelState, annotations: AnnotationSummary[]): PartColouring {
  const tree = annotationTree(annotations);
  const legend: PartColour[] = tree.map((t, i) => {
    const colour = paletteColour(i);
    return {
      id: t.annotation.id,
      name: t.annotation.name,
      label: annotationLabel(t.annotation),
      colour,
      border: darken(colour),
      text: textOn(colour),
      depth: t.depth,
      isLeaf: t.isLeaf,
    };
  });
  const byId: Record<string, PartColour> = Object.fromEntries(legend.map((p) => [p.id, p]));
  const depthOf = new Map(tree.map((t) => [t.annotation.id, t.depth]));
  const byName = new Map<string, string>();
  for (const t of tree) if (t.annotation.name) byName.set(t.annotation.name, t.annotation.id);

  // owner: deepest annotation claiming the node (first in tree order on a tie)
  const owner = new Map<string, string>();
  for (const t of tree) {
    for (const n of ownedNodes(t.annotation)) {
      const cur = owner.get(n);
      if (cur === undefined || depthOf.get(cur)! < t.depth) owner.set(n, t.annotation.id);
    }
  }
  // collapsed function nodes take their annotation's colour
  for (const node of model.nodes) {
    const name = node.function_metadata?.annotation_name ?? (node.id.startsWith("fn_") ? node.id.slice(3) : null);
    if (node.type === "function" || node.id.startsWith("fn_")) {
      const id = name ? byName.get(name) : undefined;
      if (id) owner.set(node.id, id);
    }
  }

  // entries: leaf-most annotations entered
  const enterLists = new Map<string, string[]>();
  for (const t of tree) {
    for (const n of t.annotation.entry_nodes) {
      const list = enterLists.get(n) ?? [];
      if (!list.includes(t.annotation.id)) list.push(t.annotation.id);
      enterLists.set(n, list);
    }
  }
  const descIds = new Map<string, Set<string>>();
  const descOf = (id: string) => {
    let s = descIds.get(id);
    if (!s) {
      const a = annotations.find((x) => x.id === id)!;
      s = new Set(descendantsOf(a, annotations).map((d) => d.id));
      descIds.set(id, s);
    }
    return s;
  };

  const nodes: Record<string, NodePart> = {};
  for (const node of model.nodes) {
    const list = enterLists.get(node.id) ?? [];
    const enters = list.filter((id) => !list.some((other) => other !== id && descOf(id).has(other)));
    nodes[node.id] = { owner: owner.get(node.id) ?? null, enters };
  }

  // edges into an owned node from inside that owner's part
  const partSets = new Map<string, Set<string>>();
  const partOf = (id: string) => {
    let s = partSets.get(id);
    if (!s) partSets.set(id, (s = emphasisNodes(id, annotations)));
    return s;
  };
  const edges: Record<string, string> = {};
  for (const c of model.connections) {
    const a = nodes[c.to]?.owner;
    if (!a) continue;
    if (nodes[c.from]?.owner === a || partOf(a).has(c.from)) edges[edgeKey(c)] = a;
  }
  return { legend, byId, nodes, edges };
}

export function edgeKey(c: Pick<ApiConnection, "from" | "to">): string {
  return `${c.from}->${c.to}`;
}

/** Nodes drawn at full strength while part `id` is emphasised (its part, with descendants and function nodes). */
export function emphasisNodes(id: string, annotations: AnnotationSummary[]): Set<string> {
  const a = annotations.find((x) => x.id === id) ?? annotations.find((x) => x.name === id);
  if (!a) return new Set();
  const s = focusNodeIds(a, annotations);
  for (const d of [a, ...descendantsOf(a, annotations)]) if (d.name) s.add(`fn_${d.name}`);
  return s;
}
