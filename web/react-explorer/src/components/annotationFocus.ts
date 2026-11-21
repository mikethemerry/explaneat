/**
 * Focus on one annotation: the part of the network that belongs to it, and how that
 * part connects to the rest. Pure functions (no React, no fetch), so the static
 * build needs nothing the precomputed data does not already answer, and the
 * computation is unit-tested from Python (tests/test_ui/test_annotation_focus_ts.py).
 *
 * Focus set of annotation A: A's entry, intermediate (subgraph) and exit nodes,
 * plus those of every descendant (a composition is the union of its children's
 * parts and its own). Its connections are the model's connections with both ends
 * in the set (the annotation summary carries no edge list; this is the induced
 * subgraph, as the collapse transform uses). Edges crossing the boundary are not
 * drawn; each node instead carries the count of enabled edges coming in from
 * outside (`extIn`) and going out of the part (`extOut`).
 */
import type { AnnotationSummary, ApiConnection, ModelState } from "../api/client";

export type FocusMarker = { extIn: number; extOut: number };

export type FocusGraph = {
  annotation: AnnotationSummary;
  /** The model restricted to the focus set (what the graph draws). */
  model: ModelState;
  /** The focused annotation's own entry and exit nodes (present in the model). */
  entries: string[];
  exits: string[];
  /** Boundary edge counts per focus node (only nodes with a nonzero count). */
  markers: Record<string, FocusMarker>;
  /** Focus-set node ids the current model does not contain. */
  missing: string[];
  /** Totals for the legend line. */
  summary: { inNodes: number; inEdges: number; outNodes: number; outEdges: number };
};

/** An annotation by id or name (URL parameters accept either). */
export function findAnnotation(annotations: AnnotationSummary[], idOrName: string | null | undefined): AnnotationSummary | null {
  if (!idOrName) return null;
  return annotations.find((a) => a.id === idOrName) ?? annotations.find((a) => a.name === idOrName) ?? null;
}

/** Every descendant of `ann` (children, grandchildren, ...), cycle-safe. */
export function descendantsOf(ann: AnnotationSummary, annotations: AnnotationSummary[]): AnnotationSummary[] {
  const byId = new Map(annotations.map((a) => [a.id, a]));
  const kids = new Map<string, AnnotationSummary[]>();
  const add = (parent: string, child: AnnotationSummary) => {
    const list = kids.get(parent) ?? [];
    if (!list.includes(child)) list.push(child);
    kids.set(parent, list);
  };
  for (const a of annotations) {
    if (a.parent_annotation_id) add(a.parent_annotation_id, a);
    for (const c of a.children_ids ?? []) {
      const child = byId.get(c);
      if (child) add(a.id, child);
    }
  }
  const out: AnnotationSummary[] = [];
  const seen = new Set<string>([ann.id]);
  const stack = [...(kids.get(ann.id) ?? [])];
  while (stack.length) {
    const a = stack.pop()!;
    if (seen.has(a.id)) continue;
    seen.add(a.id);
    out.push(a);
    stack.push(...(kids.get(a.id) ?? []));
  }
  return out;
}

function ownNodes(a: AnnotationSummary): string[] {
  return [...a.entry_nodes, ...a.subgraph_nodes, ...a.exit_nodes];
}

/** The focus set: the annotation's nodes and its descendants' nodes. */
export function focusNodeIds(ann: AnnotationSummary, annotations: AnnotationSummary[]): Set<string> {
  const ids = new Set<string>(ownNodes(ann));
  for (const d of descendantsOf(ann, annotations)) ownNodes(d).forEach((n) => ids.add(n));
  return ids;
}

/**
 * Names of annotations to show expanded while focused: every annotation whose
 * nodes (with its descendants') meet the focus set. This covers the focused
 * annotation itself, its descendants and its ancestors (a collapsed parent would
 * hide the part inside its function node), and any other collapsed part that
 * would swallow one of the focus nodes.
 */
export function namesToExpand(ann: AnnotationSummary, annotations: AnnotationSummary[]): string[] {
  const focus = focusNodeIds(ann, annotations);
  const out: string[] = [];
  for (const a of annotations) {
    if (!a.name) continue;
    const nodes = focusNodeIds(a, annotations);
    for (const n of nodes) {
      if (focus.has(n)) {
        out.push(a.name);
        break;
      }
    }
  }
  return out.sort();
}

/** The collapsed set to request while focused (the user's set minus `namesToExpand`). */
export function focusCollapsed(collapsed: Iterable<string>, expand: string[]): Set<string> {
  const drop = new Set(expand);
  return new Set([...collapsed].filter((n) => !drop.has(n)));
}

/** The focused view of `model`: nodes in the focus set, induced connections, boundary markers. */
export function computeFocusGraph(model: ModelState, ann: AnnotationSummary, annotations: AnnotationSummary[]): FocusGraph {
  const wanted = focusNodeIds(ann, annotations);
  const present = new Set(model.nodes.map((n) => n.id));
  // While a collapsed model is still being replaced, keep the part's function node(s) visible
  for (const a of [ann, ...descendantsOf(ann, annotations)]) {
    if (a.name && present.has(`fn_${a.name}`)) wanted.add(`fn_${a.name}`);
  }
  const inFocus = new Set([...wanted].filter((n) => present.has(n)));
  const missing = [...wanted].filter((n) => !present.has(n) && !n.startsWith("fn_")).sort();

  const connections: ApiConnection[] = model.connections.filter((c) => inFocus.has(c.from) && inFocus.has(c.to));
  const markers: Record<string, FocusMarker> = {};
  const bump = (id: string, key: keyof FocusMarker) => {
    const m = markers[id] ?? (markers[id] = { extIn: 0, extOut: 0 });
    m[key] += 1;
  };
  for (const c of model.connections) {
    if (!c.enabled) continue;
    const f = inFocus.has(c.from);
    const t = inFocus.has(c.to);
    if (!f && t) bump(c.to, "extIn");
    else if (f && !t) bump(c.from, "extOut");
  }
  const vals = Object.values(markers);
  const summary = {
    inNodes: vals.filter((m) => m.extIn > 0).length,
    inEdges: vals.reduce((s, m) => s + m.extIn, 0),
    outNodes: vals.filter((m) => m.extOut > 0).length,
    outEdges: vals.reduce((s, m) => s + m.extOut, 0),
  };
  return {
    annotation: ann,
    model: {
      nodes: model.nodes.filter((n) => inFocus.has(n.id)),
      connections,
      metadata: {
        ...model.metadata,
        input_nodes: model.metadata.input_nodes.filter((n) => inFocus.has(n)),
        output_nodes: model.metadata.output_nodes.filter((n) => inFocus.has(n)),
      },
    },
    entries: ann.entry_nodes.filter((n) => inFocus.has(n)),
    exits: ann.exit_nodes.filter((n) => inFocus.has(n)),
    markers,
    missing,
    summary,
  };
}

/** The name shown for an annotation. */
export function annotationLabel(a: AnnotationSummary): string {
  return a.display_name || a.name || `Annotation ${a.id.slice(0, 8)}`;
}
