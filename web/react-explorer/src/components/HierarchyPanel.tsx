/**
 * The hierarchy navigator of the review page: the annotation tree (root, compositions, leaves)
 * and, for the selected node, the hypothesis, the closed form (FormulaDisplay), entries and
 * exits, the probe screen grade, the leaf verdict buttons, every stored evidence record, a viz
 * computed on demand against the live model (VizCanvas), the plain-language paragraphs and the
 * verifier findings about it. At the root: performance on the train and test rows, coverage and
 * the three levels. Keys (when the tab is active and nothing is being typed): up = parent,
 * down = first child, left / right = siblings.
 *
 * Read only, except the leaf verdict buttons, which append to the same store as the queue.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import katex from "katex";
import {
  computePerformance,
  computeVizData,
  getExperimentSplit,
  type PerformanceResponse,
  type VizDataRequest,
  type VizDataResponse,
} from "../api/client";
import { getReviewNode, getReviewTree, type LeafVerdictKind, type NodeCard, type Paragraph, type ReviewTree, type TreeNode } from "../api/review";
import { FormulaDisplay } from "./FormulaDisplay";
import { VizCanvas } from "./VizCanvas";

const PROBE_WORD: Record<string, string> = {
  meaningful: "meaningful",
  partial: "partial",
  closed_form_only: "closed-form only",
  restatement: "restatement",
  unnamed: "not probed",
};
const VIZ_CHOICES = ["line", "heatmap", "partial_dependence", "sensitivity"];
const NEEDS_SINGLE_DIM = new Set(["line", "partial_dependence"]);

function errMsg(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

function isEditable(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  if (!el || !el.tagName) return false;
  return el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.tagName === "SELECT" || el.isContentEditable;
}

function Latex({ tex }: { tex: string }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!ref.current) return;
    try {
      katex.render(tex, ref.current, { displayMode: true, throwOnError: false });
    } catch {
      ref.current.textContent = tex;
    }
  }, [tex]);
  return <div ref={ref} className="hier-latex" />;
}

function num(v: unknown, digits = 3): string {
  return typeof v === "number" ? (Math.abs(v) < 1e-3 && v !== 0 ? v.toExponential(2) : v.toFixed(digits)) : v == null ? "—" : String(v);
}

export type HierarchyPanelProps = {
  label: string;
  selected: string | null;
  onSelect: (id: string | null) => void;
  active: boolean;
  document: Paragraph[];
  filtered: string | null;
  onFilterQueue: (id: string | null) => void;
  onLeafVerdict: (leafId: string, verdict: LeafVerdictKind, note: string) => Promise<void>;
  leafVerdicts: Record<string, { verdict: LeafVerdictKind; note: string; at: string } | null>;
  busy: boolean;
};

export function HierarchyPanel(props: HierarchyPanelProps) {
  const { label, selected, onSelect, active } = props;
  const [tree, setTree] = useState<ReviewTree | null>(null);
  const [treeErr, setTreeErr] = useState<string | null>(null);
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  const [card, setCard] = useState<NodeCard | null>(null);
  const [cardErr, setCardErr] = useState<string | null>(null);
  const [cardLoading, setCardLoading] = useState(false);
  const [splitId, setSplitId] = useState<string | null>(null);
  const [splitErr, setSplitErr] = useState<string | null>(null);
  const [viz, setViz] = useState<VizDataResponse | null>(null);
  const [vizType, setVizType] = useState("line");
  const [inputDim, setInputDim] = useState(0);
  const [vizErr, setVizErr] = useState<string | null>(null);
  const [vizLoading, setVizLoading] = useState(false);
  const [perf, setPerf] = useState<{ train: PerformanceResponse | null; test: PerformanceResponse | null; error: string | null } | null>(null);
  const [leafNote, setLeafNote] = useState("");

  const byId = useMemo(() => {
    const m = new Map<string, TreeNode>();
    tree?.nodes.forEach((n) => m.set(n.id, n));
    return m;
  }, [tree]);

  useEffect(() => {
    getReviewTree(label)
      .then((t) => {
        setTree(t);
        if (!selected && t.roots.length) onSelect(t.roots[0]);
        if (t.experiment.in_db && t.experiment.experiment_id) {
          getExperimentSplit(t.experiment.experiment_id)
            .then((s) => setSplitId(s.split_id))
            .catch((e) => setSplitErr(errMsg(e)));
        }
      })
      .catch((e) => setTreeErr(errMsg(e)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [label]);

  useEffect(() => {
    if (!selected) {
      setCard(null);
      return;
    }
    let cancelled = false;
    setCardLoading(true);
    setCardErr(null);
    setViz(null);
    setVizErr(null);
    setLeafNote("");
    getReviewNode(label, selected)
      .then((c) => {
        if (cancelled) return;
        setCard(c);
        const nIn = c.dimensionality[0];
        setVizType(nIn === 1 ? "line" : nIn === 2 ? "heatmap" : "partial_dependence");
        setInputDim(0);
      })
      .catch((e) => !cancelled && setCardErr(errMsg(e)))
      .finally(() => !cancelled && setCardLoading(false));
    return () => {
      cancelled = true;
    };
  }, [label, selected]);

  const node = selected ? byId.get(selected) || null : null;

  const runViz = useCallback(
    async (c: NodeCard, vt: string, dim: number) => {
      if (!splitId || !c.genome_id) return;
      setVizLoading(true);
      setVizErr(null);
      try {
        const params: Record<string, unknown> = {};
        if (NEEDS_SINGLE_DIM.has(vt)) {
          params.input_dim = dim;
          if (vt === "partial_dependence") params.vary_dims = [dim];
        } else if (vt === "heatmap") {
          params.input_dims = [0, Math.min(1, c.dimensionality[0] - 1)];
        }
        params.output_dim = 0;
        const r = await computeVizData(c.genome_id, {
          annotation_id: c.node.id,
          dataset_split_id: splitId,
          viz_type: vt as VizDataRequest["viz_type"],
          params,
          split: "train",
          sample_fraction: 1.0,
          max_samples: 2000,
        });
        setViz(r);
      } catch (e) {
        setVizErr(errMsg(e));
      } finally {
        setVizLoading(false);
      }
    },
    [splitId],
  );

  // on-demand viz for small leaves, as soon as the card and the split are known
  useEffect(() => {
    if (card && card.compute_viz && splitId && !viz && !vizLoading && !vizErr) runViz(card, vizType, inputDim);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [card, splitId]);

  // root: whole-model performance on train and test
  useEffect(() => {
    if (!node || node.kind !== "root" || !splitId || !tree?.genome_id) return;
    let cancelled = false;
    const gid = tree.genome_id;
    Promise.all([
      computePerformance(gid, { dataset_split_id: splitId, split: "train" }),
      computePerformance(gid, { dataset_split_id: splitId, split: "test" }).catch(() => null),
    ])
      .then(([tr, te]) => !cancelled && setPerf({ train: tr, test: te, error: null }))
      .catch((e) => !cancelled && setPerf({ train: null, test: null, error: errMsg(e) }));
    return () => {
      cancelled = true;
    };
  }, [node, splitId, tree]);

  // keys: up parent, down first child, left/right siblings
  useEffect(() => {
    if (!active) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey || isEditable(e.target) || !node) return;
      if (e.key === "ArrowUp" && node.parent) onSelect(node.parent);
      else if (e.key === "ArrowDown" && node.children.length) onSelect(node.children[0]);
      else if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        const sibs = node.parent ? byId.get(node.parent)?.children || [] : tree?.roots || [];
        const i = sibs.indexOf(node.id);
        if (i < 0 || sibs.length < 2) return;
        onSelect(sibs[(i + (e.key === "ArrowRight" ? 1 : sibs.length - 1)) % sibs.length]);
      } else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [active, node, byId, tree, onSelect]);

  if (treeErr) return <p className="error-text">{treeErr}</p>;
  if (!tree) return <p className="hint">Loading the hierarchy…</p>;

  const renderNode = (id: string, depth: number): React.ReactNode => {
    const n = byId.get(id);
    if (!n) return null;
    const isCollapsed = collapsed.has(id);
    const v = props.leafVerdicts[id] ?? n.verdict;
    return (
      <div key={id} className="hier-node">
        <div
          className={"hier-row" + (selected === id ? " selected" : "") + (props.filtered === id ? " filtered" : "")}
          style={{ paddingLeft: 8 + depth * 14 }}
          onClick={() => onSelect(id)}
        >
          {n.children.length > 0 ? (
            <button
              className="hier-toggle"
              onClick={(e) => {
                e.stopPropagation();
                setCollapsed((s) => {
                  const next = new Set(s);
                  if (next.has(id)) next.delete(id);
                  else next.add(id);
                  return next;
                });
              }}
              title={isCollapsed ? "expand" : "collapse"}
            >
              {isCollapsed ? "▸" : "▾"}
            </button>
          ) : (
            <span className="hier-toggle" />
          )}
          <span className={`kind-chip kind-${n.kind}`}>{n.kind === "composition" ? "comp" : n.kind}</span>
          <span className="hier-name">{n.name || n.id}</span>
          {n.kind === "leaf" && n.probe && <span className="hint hier-probe">{PROBE_WORD[n.probe] || n.probe}</span>}
          {v && <span className={`verdict-pill ${v.verdict}`}>{v.verdict}</span>}
          <span className="hint hier-count">{n.sids.length} s.</span>
        </div>
        {!isCollapsed && n.children.map((c) => renderNode(c, depth + 1))}
      </div>
    );
  };

  const sibs = node ? (node.parent ? byId.get(node.parent)?.children || [] : tree.roots) : [];
  const sibIndex = node ? sibs.indexOf(node.id) : -1;
  const explorerUrl =
    tree.genome_id && tree.experiment.in_db && node
      ? `/genome/${tree.genome_id}?experiment=${encodeURIComponent(tree.experiment.experiment_id || "")}&name=${encodeURIComponent(tree.experiment.experiment_name || label)}&annotation=${encodeURIComponent(node.id)}`
      : null;
  const paragraphs = card ? props.document.filter((p) => card.paragraphs.includes(p.id)) : [];
  const leafV = node ? props.leafVerdicts[node.id] ?? node.verdict : null;

  return (
    <div className="hier-panel">
      <div className="hier-tree">
        {tree.roots.map((r) => renderNode(r, 0))}
        {tree.roots.length === 0 && <p className="hint">No annotations.</p>}
        <p className="hint" style={{ margin: "0.4rem 0 0" }}>
          Keys: <kbd>↑</kbd> parent, <kbd>↓</kbd> first child, <kbd>←</kbd>/<kbd>→</kbd> siblings.
        </p>
      </div>

      {node && (
        <div className="hier-card">
          <div className="hier-nav">
            <button className="op-btn secondary" disabled={!node.parent} onClick={() => node.parent && onSelect(node.parent)} title="parent (↑)">↑ up</button>
            <button className="op-btn secondary" disabled={!node.children.length} onClick={() => onSelect(node.children[0])} title="first child (↓)">↓ down</button>
            <button className="op-btn secondary" disabled={sibs.length < 2} onClick={() => onSelect(sibs[(sibIndex + sibs.length - 1) % sibs.length])} title="previous sibling (←)">←</button>
            <button className="op-btn secondary" disabled={sibs.length < 2} onClick={() => onSelect(sibs[(sibIndex + 1) % sibs.length])} title="next sibling (→)">→</button>
            <button
              className={"op-btn " + (props.filtered === node.id ? "primary" : "secondary")}
              onClick={() => props.onFilterQueue(props.filtered === node.id ? null : node.id)}
              title="Show only this part's sentences in the queue"
            >
              {props.filtered === node.id ? "Queue: all parts" : `Queue: this part (${node.sids.length})`}
            </button>
            {explorerUrl && (
              <a className="op-btn secondary" href={explorerUrl} target="_blank" rel="noreferrer">Show in explorer</a>
            )}
          </div>
          <h4>
            <span className={`kind-chip kind-${node.kind}`}>{node.kind}</span> {node.name || node.id} <span className="hint">{node.id}</span>
          </h4>
          <p className="hier-hyp">{node.hypothesis || <em className="hint">no hypothesis</em>}</p>

          {node.kind === "leaf" && (
            <div className="review-actions" style={{ marginBottom: "0.5rem" }}>
              {(["understood", "partial", "not"] as LeafVerdictKind[]).map((k) => (
                <button
                  key={k}
                  className={`verdict-btn leaf ${k}` + (leafV?.verdict === k ? " chosen" : "")}
                  disabled={props.busy}
                  onClick={() => props.onLeafVerdict(node.id, k, leafNote).then(() => setLeafNote(""))}
                >
                  {k}
                </button>
              ))}
              <input className="text-input review-note" placeholder="Note" value={leafNote} onChange={(e) => setLeafNote(e.target.value)} />
              {leafV?.note && <span className="hint">— {leafV.note}</span>}
            </div>
          )}

          {cardLoading && <p className="hint">Loading the node…</p>}
          {cardErr && <p className="error-text">{cardErr}</p>}
          {card && (
            <>
              <div className="hier-io">
                <div><strong>Entries ({card.dimensionality[0]}):</strong> {card.entry_names.join(", ")}</div>
                <div><strong>Exits ({card.dimensionality[1]}):</strong> {card.exit_names.join(", ")}</div>
                {card.meaning && (
                  <div>
                    <strong>Probe screen:</strong> {PROBE_WORD[card.meaning.grade || ""] || card.meaning.grade}
                    {card.meaning.concept ? ` (best concept ${card.meaning.concept}` : ""}
                    {card.meaning.probe?.r2 != null ? `, r² ${num(card.meaning.probe.r2, 2)}` : ""}
                    {card.meaning.probe?.auc != null ? `, AUC ${num(card.meaning.probe.auc, 2)}` : ""}
                    {card.meaning.concept ? ")" : ""}
                  </div>
                )}
              </div>

              {tree.genome_id && (
                <div className="hier-section">
                  <h5>Closed form (live model)</h5>
                  <FormulaDisplay genomeId={tree.genome_id} annotationId={node.id} />
                </div>
              )}

              {node.kind === "root" && (
                <RootEvidence tree={tree} perf={perf} splitId={splitId} splitErr={splitErr} />
              )}

              <div className="hier-section">
                <h5>
                  Visualisation{" "}
                  <span className="hint">
                    {card.compute_viz ? "computed on demand against the live model" : "on request (train rows)"}
                  </span>
                </h5>
                {!splitId ? (
                  <p className="hint">{splitErr ? `No dataset split: ${splitErr}` : "Waiting for the experiment's dataset split…"}</p>
                ) : (
                  <div className="review-actions">
                    <select className="text-input" value={vizType} onChange={(e) => setVizType(e.target.value)}>
                      {VIZ_CHOICES.filter((v) => card.suggested_viz_types.includes(v)).map((v) => (
                        <option key={v} value={v}>{v.replace("_", " ")}</option>
                      ))}
                    </select>
                    {NEEDS_SINGLE_DIM.has(vizType) && (
                      <select className="text-input" value={inputDim} onChange={(e) => setInputDim(Number(e.target.value))}>
                        {card.entry_names.map((n, i) => (
                          <option key={i} value={i}>{n}</option>
                        ))}
                      </select>
                    )}
                    <button className="op-btn primary" disabled={vizLoading} onClick={() => runViz(card, vizType, inputDim)}>
                      {vizLoading ? "Computing…" : "Compute"}
                    </button>
                  </div>
                )}
                {vizErr && <p className="error-text">{vizErr}</p>}
                {viz && (
                  <div className="hier-viz">
                    <VizCanvas vizType={viz.viz_type} data={viz.data} />
                  </div>
                )}
              </div>

              <div className="hier-section">
                <h5>Stored evidence ({card.evidence.length})</h5>
                {card.evidence.length === 0 && <p className="hint">None stored on this annotation.</p>}
                {card.evidence.map((ev) => (
                  <div key={`${ev.type}-${ev.index}`} className="hier-evidence">
                    <div>
                      <span className="badge-chip">{ev.type}</span>{" "}
                      {ev.tool && <span className="hint">{ev.tool}</span>}
                      {ev.split && <span className="hint"> · {ev.split}{ev.n != null && ev.n !== "None" ? `, n = ${ev.n}` : ""}</span>}
                      {ev.provenance.length > 0 && <span className="hint"> · {ev.provenance.map((p) => <code key={p}>{p}</code>)}</span>}
                      {ev.history > 0 && <span className="hint"> · amended {ev.history}×</span>}
                    </div>
                    {ev.narrative && <p className="hier-narrative">{ev.narrative}</p>}
                    {ev.query && (
                      <p className="hint hier-query">
                        {Object.entries(ev.query).map(([k, v]) => `${k}: ${String(v)}`).join(" · ")}
                      </p>
                    )}
                    {ev.latex && <Latex tex={ev.latex_collapsed || ev.latex} />}
                    {ev.rows.length > 0 && (
                      <table className="hier-table">
                        <tbody>
                          {ev.rows.map(([k, v], i) => (
                            <tr key={i}><td>{k}</td><td>{v}</td></tr>
                          ))}
                        </tbody>
                      </table>
                    )}
                    {ev.svg_data && (
                      <img className="hier-svg" alt={`${ev.type} snapshot`} src={`data:image/svg+xml;base64,${ev.svg_data}`} />
                    )}
                    {ev.viz_config && !ev.svg_data && (
                      <p className="hint">stored viz: {JSON.stringify(ev.viz_config)}</p>
                    )}
                  </div>
                ))}
              </div>

              {paragraphs.length > 0 && (
                <div className="hier-section">
                  <h5>Plain-language document</h5>
                  {paragraphs.map((p) => (
                    <div key={p.id} className="review-paragraph" style={{ marginBottom: 6 }}>
                      <div className="review-paragraph-title">{p.title}</div>
                      <p>{p.sentences.map((s) => s.text).join(" ")}</p>
                    </div>
                  ))}
                </div>
              )}

              <div className="hier-section">
                <h5>Verifier findings ({card.findings.length})</h5>
                {card.findings.length === 0 && <p className="hint">None attached to this part's sentences.</p>}
                {card.findings.map((f, i) => (
                  <p key={i} className={"review-verifier" + (f.stale ? " stale" : "")}>
                    <strong>{f.grade}</strong> {f.level} ({f.source}{f.stale ? ", stale" : ""}): {f.evidence}
                  </p>
                ))}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

function RootEvidence({ tree, perf, splitId, splitErr }: {
  tree: ReviewTree;
  perf: { train: PerformanceResponse | null; test: PerformanceResponse | null; error: string | null } | null;
  splitId: string | null;
  splitErr: string | null;
}) {
  const cov = tree.coverage;
  const lv = tree.levels?.levels || null;
  const row = (name: string, p: PerformanceResponse | null) => {
    if (!p) return null;
    if (p.survival) {
      const s = p.survival as unknown as Record<string, unknown>;
      return (
        <tr key={name}><td>{name}</td><td>{p.n_samples}</td><td colSpan={3}>survival: C {num(s["harrell_c"] ?? s["c_index"], 3)}</td></tr>
      );
    }
    return (
      <tr key={name}>
        <td>{name}</td><td>{p.n_samples}</td><td>{num(p.auc_roc)}</td><td>{num(p.accuracy)}</td><td>{num(p.log_loss)}</td>
      </tr>
    );
  };
  return (
    <div className="hier-section">
      <h5>Whole model</h5>
      {!splitId && <p className="hint">{splitErr ? `No dataset split: ${splitErr}` : "Waiting for the dataset split…"}</p>}
      {perf?.error && <p className="error-text">{perf.error}</p>}
      {perf && (perf.train || perf.test) && (
        <table className="hier-table">
          <thead><tr><th>rows</th><th>n</th><th>AUC</th><th>accuracy</th><th>log loss</th></tr></thead>
          <tbody>{row("train", perf.train)}{row("test", perf.test)}</tbody>
        </table>
      )}
      <table className="hier-table">
        <tbody>
          <tr><td>structural / compositional coverage</td><td>{num(cov.structural_coverage, 3)} / {num(cov.compositional_coverage, 3)}{cov.full_coverage ? " (full)" : ""}</td></tr>
          <tr><td>hierarchy complete</td><td>{cov.hierarchy_complete == null ? "—" : cov.hierarchy_complete ? "yes" : "no"}</td></tr>
          <tr><td>function preservation / closed-form max |err|</td><td>{num(cov.function_preservation_max_abs_err)} / {num(cov.closed_form_max_abs_err)}</td></tr>
          <tr><td>hidden units, leaves, compositions</td><td>{String(cov.hidden_nodes ?? "—")}, {String(cov.n_leaf_annotations ?? "—")}, {String(cov.n_composition_annotations ?? "—")}</td></tr>
        </tbody>
      </table>
      {lv ? (
        <table className="hier-table">
          <thead><tr><th>level</th><th>result</th></tr></thead>
          <tbody>
            <tr><td>faithful</td><td>{lv.faithful?.passed ? "passed" : "not passed"} (closed form {num(lv.faithful?.closed_form_max_abs_err)})</td></tr>
            <tr><td>decomposed</td><td>{lv.decomposed?.passed ? "passed" : "not passed"} (granularity {num(lv.decomposed?.granularity, 2)}, {String(lv.decomposed?.n_leaves)} leaves)</td></tr>
            <tr><td>understood (probe)</td><td>{num(lv.understood?.understood_fraction, 2)} of leaves meaningful{lv.understood?.understood_mike != null ? `; your leaf verdicts ${num(lv.understood?.understood_mike, 2)}` : ""}</td></tr>
          </tbody>
        </table>
      ) : (
        <p className="hint">No levels sidecar for this label (experiments/explanation_levels/levels).</p>
      )}
    </div>
  );
}
