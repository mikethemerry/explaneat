/**
 * Lineage of the loaded genome: fitness along its ancestor chain against the
 * population best/mean per generation, jumps (structural vs weight-only steps),
 * and a diff view of any ancestor against the descendant, its parent, or the
 * best genome of its generation.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import * as Plot from "@observablehq/plot";
import type { ModelState } from "../api/client";
import {
  getAncestor,
  getDiffGraph,
  getJumps,
  getLineage,
  type AncestorResponse,
  type DiffGraph,
  type Jump,
  type LineageResponse,
  type LineageRow,
} from "../api/tools";
import type { HighlightKind } from "./NetworkViewer";

export type LineageView = {
  title: string;
  model: ModelState;
  nodeHighlight: Record<string, HighlightKind>;
  edgeHighlight: Record<string, HighlightKind>;
  olderGenomeId: string;
  newerGenomeId: string;
};

type CompareWith = "descendant" | "parent" | "generation_best";

const fmt = (v: number | null | undefined, d = 4) => (v == null ? "—" : v.toFixed(d));

export function diffToView(d: DiffGraph, title: string): LineageView {
  const nodeHighlight: Record<string, HighlightKind> = {};
  const edgeHighlight: Record<string, HighlightKind> = {};
  for (const n of d.nodes) {
    if (n.status !== "common") nodeHighlight[n.id] = n.status;
    else if (n.bias_a != null && n.bias_b != null && n.bias_a !== n.bias_b && n.type !== "input") nodeHighlight[n.id] = "changed";
  }
  for (const c of d.connections) {
    const k = `${c.from}->${c.to}`;
    if (c.status !== "common") edgeHighlight[k] = c.status;
    else if (c.weight_a != null && c.weight_b != null && Math.abs(c.weight_a - c.weight_b) > 1e-12) edgeHighlight[k] = "changed";
  }
  const model: ModelState = {
    nodes: d.nodes.map((n) => ({
      id: n.id,
      type: n.type,
      bias: n.bias,
      activation: n.activation,
      response: n.response,
      aggregation: n.aggregation,
      display_name: n.display_name,
    })),
    connections: d.connections.map((c) => ({ from: c.from, to: c.to, weight: c.weight, enabled: true })),
    metadata: { ...d.metadata, is_original: true },
  };
  return { title, model, nodeHighlight, edgeHighlight, olderGenomeId: d.genome_a, newerGenomeId: d.genome_b };
}

function LineageChart({
  data,
  jumps,
  selectedGen,
  onPick,
}: {
  data: LineageResponse;
  jumps: Jump[];
  selectedGen: number | null;
  onPick: (gen: number) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!ref.current) return;
    const lin = data.lineage;
    const pop = data.population.filter((p) => p.best_fitness != null);
    const jumpGens = new Set(jumps.map((j) => j.generation));
    const points = lin
      .filter((r) => r.fitness != null)
      .map((r) => ({
        generation: r.generation,
        fitness: r.fitness as number,
        kind: r.generation === 0 ? "start" : r.structural_change ? "structural change" : r.is_elite_copy ? "elite copy" : "weights only",
        jump: jumpGens.has(r.generation),
      }));
    const width = Math.max(480, ref.current.clientWidth || 700);
    const plot = Plot.plot({
      width,
      height: 230,
      marginLeft: 50,
      x: { label: "Generation", grid: true },
      y: { label: "Fitness", grid: true },
      color: {
        domain: ["structural change", "weights only", "elite copy", "start"],
        range: ["#ea580c", "#2563eb", "#93c5fd", "#6b7280"],
        legend: true,
      },
      marks: [
        Plot.line(pop, { x: "generation", y: "best_fitness", stroke: "#9ca3af", strokeDasharray: "4 3" }),
        Plot.line(pop.filter((p) => p.mean_fitness != null), { x: "generation", y: "mean_fitness", stroke: "#d1d5db" }),
        Plot.line(points, { x: "generation", y: "fitness", stroke: "#1e3a8a", strokeWidth: 1.5 }),
        Plot.ruleX(points.filter((p) => p.jump), { x: "generation", stroke: "#dc2626", strokeOpacity: 0.35, strokeWidth: 6 }),
        Plot.dot(points, { x: "generation", y: "fitness", fill: "kind", r: 3.5, tip: true, title: (p: { generation: number; fitness: number; kind: string }) => `gen ${p.generation}\nfitness ${p.fitness.toFixed(4)}\n${p.kind}` }),
        ...(selectedGen != null ? [Plot.ruleX([selectedGen], { stroke: "#111827", strokeWidth: 1.5 })] : []),
        Plot.text(["population best (dashed), population mean (light), lineage (dark); red bands = jumps"], {
          frameAnchor: "top-left", dy: -2, fontSize: 10, fill: "#6b7280",
        }),
      ],
    });
    const x = plot.scale("x");
    plot.addEventListener("click", (e: Event) => {
      const ev = e as MouseEvent;
      const svg = (plot.tagName.toLowerCase() === "svg" ? plot : plot.querySelector("svg")) as SVGSVGElement | null;
      if (!svg || !x || !("invert" in x) || typeof x.invert !== "function") return;
      const rect = svg.getBoundingClientRect();
      const gen = Math.round(x.invert(ev.clientX - rect.left) as number);
      const gens = lin.map((r) => r.generation);
      if (gens.length) onPick(Math.min(Math.max(gen, gens[0]), gens[gens.length - 1]));
    });
    (plot as HTMLElement).style.cursor = "crosshair";
    ref.current.replaceChildren(plot);
  }, [data, jumps, selectedGen, onPick]);
  return <div ref={ref} className="lineage-chart" />;
}

export function LineagePanel({
  genomeId,
  onView,
  onOpenGenome,
}: {
  genomeId: string;
  onView: (v: LineageView | null) => void;
  onOpenGenome: (genomeId: string) => void;
}) {
  const [mode, setMode] = useState("fitter_parent");
  const [data, setData] = useState<LineageResponse | null>(null);
  const [jumps, setJumps] = useState<Jump[]>([]);
  const [jumpMethod, setJumpMethod] = useState("threshold");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [gen, setGen] = useState<number | null>(null);
  const [compare, setCompare] = useState<CompareWith>("descendant");
  const [ancestor, setAncestor] = useState<AncestorResponse | null>(null);
  const [viewMsg, setViewMsg] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    Promise.all([getLineage(genomeId, mode), getJumps(genomeId, mode, jumpMethod, 5)])
      .then(([l, j]) => {
        setData(l);
        setJumps(j.jumps || []);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
  }, [genomeId, mode, jumpMethod]);

  const byGen = useMemo(() => {
    const m = new Map<number, LineageRow>();
    for (const r of data?.lineage || []) m.set(r.generation, r);
    return m;
  }, [data]);

  // Load the diff view for the chosen generation
  useEffect(() => {
    if (gen == null || !data?.available) return;
    const row = byGen.get(gen);
    if (!row) return;
    let cancelled = false;
    setViewMsg("Loading diff…");
    (async () => {
      try {
        const anc = await getAncestor(genomeId, gen, mode);
        if (cancelled) return;
        setAncestor(anc);
        let a = row.genome_id;
        let b = genomeId;
        let title = `Ancestor at gen ${gen} → descendant (gen ${data.generation})`;
        if (compare === "parent") {
          const prev = byGen.get(gen - 1);
          if (!prev) {
            setViewMsg("Generation 0 has no parent.");
            return;
          }
          a = prev.genome_id;
          b = row.genome_id;
          title = `Step gen ${gen - 1} → ${gen} along the lineage`;
        } else if (compare === "generation_best") {
          b = anc.generation_best.genome_id;
          title = `Ancestor at gen ${gen} vs that generation's best`;
        }
        if (a === b) {
          setViewMsg("The two genomes are the same (the ancestor is the generation best).");
          onView(null);
          return;
        }
        const d = await getDiffGraph(a, b, genomeId);
        if (cancelled) return;
        onView(diffToView(d, `${title}: fitness ${fmt(d.fitness_a)} → ${fmt(d.fitness_b)}`));
        setViewMsg(null);
      } catch (e) {
        setViewMsg(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gen, compare, data, mode]);

  if (loading) return <p className="hint">Reconstructing the lineage…</p>;
  if (error) return <p className="error-text">{error}</p>;
  if (!data) return null;
  if (!data.available) {
    return (
      <div className="lineage-missing">
        <p><strong>Lineage not recorded for this run.</strong></p>
        <p className="hint">{data.message}</p>
        <p className="hint">
          Runs started after 2026-11-07 record parents automatically. For older runs, run the replay/backfill
          (<code>experiments/lineage/replay_lineage.py</code>, <code>backfill_parents.py</code>), then reopen this tab.
          Population best/mean per generation are still available: {data.population.length} generations.
        </p>
      </div>
    );
  }
  const row = gen != null ? byGen.get(gen) : undefined;
  const nStructural = data.lineage.filter((r) => r.structural_change).length;
  return (
    <div className="tools-grid lineage-grid">
      <div style={{ gridColumn: "span 2" }}>
        <div className="form-row">
          <label>
            Follow{" "}
            <select value={mode} onChange={(e) => setMode(e.target.value)}>
              <option value="fitter_parent">fitter parent (structural line)</option>
              <option value="parent1">parent 1</option>
            </select>
          </label>
          <label>
            Jumps{" "}
            <select value={jumpMethod} onChange={(e) => setJumpMethod(e.target.value)}>
              <option value="threshold">≥ 95th pct of rises</option>
              <option value="top_k">top 5 rises</option>
            </select>
          </label>
          <span className="hint">
            {data.lineage.length} generations, {nStructural} structural steps. Click the chart to pick a generation.
          </span>
        </div>
        <LineageChart data={data} jumps={jumps} selectedGen={gen} onPick={setGen} />
      </div>
      <div>
        <h5>Ancestor</h5>
        <div className="form-row">
          <label>
            Generation{" "}
            <input
              type="number"
              min={0}
              max={data.generation}
              value={gen ?? ""}
              onChange={(e) => setGen(e.target.value === "" ? null : Number(e.target.value))}
              style={{ width: 70 }}
            />
          </label>
          <label>
            vs{" "}
            <select value={compare} onChange={(e) => setCompare(e.target.value as CompareWith)}>
              <option value="descendant">the descendant (this genome)</option>
              <option value="parent">its parent (this step)</option>
              <option value="generation_best">best of that generation</option>
            </select>
          </label>
        </div>
        {row && (
          <p className="hint">
            Gen {row.generation}: key {row.genome_key}, fitness {fmt(row.fitness)}, {row.n_nodes} nodes /{" "}
            {row.n_enabled_connections ?? row.n_connections} enabled conns
            {row.is_elite_copy ? ", elite copy" : row.followed ? `, via ${row.followed}` : ""}.
            {ancestor && ` Rank ${ancestor.ancestor_rank}/${ancestor.population_size} in its generation; gap to best ${fmt(ancestor.fitness_gap)}.`}
          </p>
        )}
        {viewMsg && <p className="hint">{viewMsg}</p>}
        {row && (
          <button className="op-btn secondary" onClick={() => onOpenGenome(row.genome_id)}>
            Open ancestor in the explorer
          </button>
        )}
        {gen != null && (
          <button className="op-btn secondary" onClick={() => { setGen(null); onView(null); }}>
            Back to the explained model
          </button>
        )}
      </div>
      <div style={{ gridColumn: "span 3" }}>
        <h5>Jumps ({jumps.length})</h5>
        {jumps.length === 0 ? (
          <p className="hint">No jumps by this rule.</p>
        ) : (
          <table className="tools-table clickable">
            <thead>
              <tr><th>Gen</th><th>Δ fitness</th><th>Step</th><th>Added genes</th><th>In final explanation's annotations</th></tr>
            </thead>
            <tbody>
              {jumps.map((j) => (
                <tr key={j.generation} onClick={() => { setCompare("parent"); setGen(j.generation); }} title="Show this step's diff">
                  <td>{j.generation}</td>
                  <td>+{j.delta.toFixed(4)}</td>
                  <td>{j.structural_change ? "structural" : j.is_elite_copy ? "elite (retrained)" : "weights only"}</td>
                  <td>
                    {j.added_nodes.length ? `nodes ${j.added_nodes.join(", ")}; ` : ""}
                    {j.added_connections.length ? `conns ${j.added_connections.join(", ")}` : ""}
                    {!j.added_nodes.length && !j.added_connections.length ? "—" : ""}
                  </td>
                  <td>
                    {Object.keys(j.annotations || {}).length
                      ? Object.entries(j.annotations).map(([a, genes]) => `${a} (${genes.join(", ")})`).join("; ")
                      : j.annotation_error
                        ? <span className="hint">{j.annotation_error}</span>
                        : j.added_nodes.length || j.added_connections.length
                          ? <span className="hint">none (genes not in the final explanation)</span>
                          : "—"}
                    {j.genes_not_in_final && j.genes_not_in_final.length > 0 && Object.keys(j.annotations || {}).length > 0 && (
                      <span className="hint"> · not in final: {j.genes_not_in_final.join(", ")}</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <p className="hint">
          A jump's added genes are mapped to the annotations of this genome's explanation that contain them — e.g. whether
          a jump is where the minimal realisation of a named signal first appeared.
        </p>
      </div>
    </div>
  );
}
