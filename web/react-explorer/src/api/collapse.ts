/**
 * Client-side collapse of annotations into function nodes: a TypeScript port of
 * explaneat/core/collapse_transform.py (collapse_structure / _collapse_one), used by
 * the static appendix, which has no server to call GET /model?collapsed=...
 *
 * Only what the explorer draws is built: the function node's annotation name,
 * hypothesis, inputs/outputs and the formula (precomputed per annotation by the
 * exporter); the forward-pass metadata the Python version carries is left out.
 */
import type { ApiConnection, ApiNode, ModelState } from "./client";

export type CollapseAnnotation = {
  name: string;
  id: string;
  hypothesis: string;
  entry_nodes: string[];
  exit_nodes: string[];
  subgraph_nodes: string[];
  /** The parent annotation's name (null for a root). */
  parent: string | null;
  /** Index of the annotate operation in the operation stream. */
  op_index: number;
  formula_latex: string | null;
};

type Ann = Omit<CollapseAnnotation, "op_index" | "id"> & { id?: string };

function descendantNames(name: string, children: Map<string, Ann[]>): Set<string> {
  const out = new Set<string>();
  for (const c of children.get(name) ?? []) {
    out.add(c.name);
    for (const d of descendantNames(c.name, children)) out.add(d);
  }
  return out;
}

function effectiveSubgraph(ann: Ann, byName: Map<string, Ann>, children: Map<string, Ann[]>): Set<string> {
  const eff = new Set<string>([...ann.subgraph_nodes, ...ann.entry_nodes, ...ann.exit_nodes]);
  for (const d of descendantNames(ann.name, children)) {
    const a = byName.get(d);
    if (a) a.subgraph_nodes.forEach((n) => eff.add(n));
  }
  return eff;
}

function effectiveEntriesExits(sub: Set<string>, m: ModelState): [Set<string>, Set<string>] {
  const entries = new Set<string>();
  const exits = new Set<string>();
  for (const c of m.connections) {
    if (!c.enabled) continue;
    const fi = sub.has(c.from);
    const ti = sub.has(c.to);
    if (!fi && ti) entries.add(c.to);
    else if (fi && !ti) exits.add(c.from);
  }
  const inputs = new Set(m.metadata.input_nodes);
  for (const nid of sub) {
    if (!inputs.has(nid)) continue;
    const internalPred = m.connections.some((c) => c.enabled && c.to === nid && sub.has(c.from));
    if (!internalPred) entries.add(nid);
  }
  return [entries, exits];
}

const pyStrCmp = (a: string, b: string) => (a < b ? -1 : a > b ? 1 : 0);

function collapseOne(m: ModelState, ann: Ann, override?: Set<string>): ModelState {
  let entrySet = new Set(ann.entry_nodes);
  const exitSet = new Set(ann.exit_nodes);
  const outputIds = new Set(m.metadata.output_nodes);
  let sub = new Set<string>(override ?? ann.subgraph_nodes);
  entrySet.forEach((n) => sub.add(n));
  exitSet.forEach((n) => sub.add(n));
  sub = new Set([...sub].filter((n) => !outputIds.has(n)));

  const nonSubFeeds = new Set(m.connections.filter((c) => c.enabled && !sub.has(c.from)).map((c) => c.to));
  const intraSources = new Set(m.connections.filter((c) => c.enabled && sub.has(c.to)).map((c) => c.from));
  const passthrough = new Set(
    [...entrySet].filter((n) => exitSet.has(n) && sub.has(n) && !nonSubFeeds.has(n) && !intraSources.has(n)),
  );
  entrySet = new Set([...entrySet].filter((n) => !passthrough.has(n)));
  const internal = new Set([...sub].filter((n) => !entrySet.has(n)));
  const fnId = `fn_${ann.name}`;

  const feedsOutside = new Set<string>();
  for (const c of m.connections) {
    if (c.enabled && internal.has(c.from) && !internal.has(c.to)) feedsOutside.add(c.from);
  }
  const orderedExits = ann.exit_nodes.filter((e) => feedsOutside.has(e));
  for (const n of [...feedsOutside].sort(pyStrCmp)) if (!orderedExits.includes(n)) orderedExits.push(n);

  const label = new Map(m.nodes.map((n) => [n.id, n.display_name || n.id]));
  const fnNode: ApiNode = {
    id: fnId,
    type: "function",
    bias: null,
    activation: null,
    response: null,
    aggregation: null,
    display_name: null,
    function_metadata: {
      annotation_name: ann.name,
      annotation_id: ann.name,
      hypothesis: ann.hypothesis,
      n_inputs: ann.entry_nodes.length,
      n_outputs: orderedExits.length,
      input_names: ann.entry_nodes.map((n) => label.get(n) ?? n),
      output_names: orderedExits.map((n) => label.get(n) ?? n),
      formula_latex: ann.formula_latex,
      subgraph_nodes: [...sub],
      subgraph_connections: m.connections
        .filter((c) => c.enabled && sub.has(c.from) && sub.has(c.to))
        .map((c) => [c.from, c.to] as [string, string]),
    },
  };
  const nodes = m.nodes.filter((n) => !internal.has(n.id));
  nodes.push(fnNode);

  const exitIndex = new Map(orderedExits.map((n, i) => [n, i]));
  const conns: ApiConnection[] = [];
  const seenEntry = new Set<string>();
  const seenTarget = new Set<string>();
  for (const c of m.connections) {
    const fromInt = internal.has(c.from);
    const toInt = internal.has(c.to);
    if (fromInt) {
      if (toInt) continue;
      const idx = exitIndex.get(c.from) ?? null;
      const key = `${c.to}\u0000${idx}`;
      if (!seenTarget.has(key)) {
        seenTarget.add(key);
        conns.push({ from: fnId, to: c.to, weight: c.weight, enabled: c.enabled, output_index: idx });
      }
    } else if (toInt) {
      if (entrySet.has(c.from) && !seenEntry.has(c.from)) {
        seenEntry.add(c.from);
        conns.push({ from: c.from, to: fnId, weight: c.weight, enabled: c.enabled, output_index: null });
      }
    } else {
      conns.push(c);
    }
  }
  const valid = new Set(nodes.map((n) => n.id));
  return {
    nodes,
    connections: conns.filter((c) => valid.has(c.from) && valid.has(c.to)),
    metadata: { ...m.metadata },
  };
}

/** The model with the named annotations collapsed into function nodes (children before parents). */
export function collapseModel(model: ModelState, annotations: Ann[], collapsed: string[]): ModelState {
  const result0: ModelState = {
    nodes: model.nodes,
    connections: model.connections,
    metadata: { ...model.metadata, collapsed_annotations: collapsed },
  };
  if (!collapsed.length) return result0;
  const byName = new Map(annotations.map((a) => [a.name, a]));
  const toCollapse = [...new Set(collapsed)].filter((n) => byName.has(n)).map((n) => byName.get(n)!);
  if (!toCollapse.length) return result0;
  const children = new Map<string, Ann[]>();
  for (const a of annotations) {
    if (a.parent && byName.has(a.parent)) {
      if (!children.has(a.parent)) children.set(a.parent, []);
      children.get(a.parent)!.push(a);
    }
  }
  const size = new Map(toCollapse.map((a) => [a.name, effectiveSubgraph(a, byName, children).size]));
  toCollapse.sort((a, b) => size.get(a.name)! - size.get(b.name)!);

  let result = result0;
  for (const ann of toCollapse) {
    const desc = descendantNames(ann.name, children);
    if (desc.size) {
      const original = effectiveSubgraph(ann, byName, children);
      const current = new Set(result.nodes.map((n) => n.id));
      const curSub = new Set<string>([...original].filter((n) => current.has(n)));
      for (const d of desc) if (current.has(`fn_${d}`)) curSub.add(`fn_${d}`);
      const [entries, exits] = effectiveEntriesExits(curSub, result);
      const corrected: Ann = {
        ...ann,
        entry_nodes: [...entries].sort(pyStrCmp),
        exit_nodes: [...exits].sort(pyStrCmp),
        subgraph_nodes: [...curSub],
      };
      result = collapseOne(result, corrected, curSub);
    } else {
      result = collapseOne(result, ann);
    }
  }
  return result;
}
