/**
 * Bottom drawer of the genome explorer with the explanation tools the 10-seed
 * agents ran as scripts: review & verify a Claude explanation record, automatic
 * checks + coverage checklist, unit activity, candidate subgraphs, restore gain,
 * one-hot category offsets, and lineage.
 */
import { Fragment, useCallback, useEffect, useMemo, useState } from "react";
import type { AnnotationSummary, ModelState } from "../api/client";
import {
  getCandidates,
  getCategoryOffsets,
  getCheck,
  getUnitActivity,
  linearise,
  type CandidatesResult,
  type CategoryGroup,
  type CheckResult,
  type LineariseResult,
  type UnitRow,
} from "../api/tools";
import type { HighlightKind } from "./NetworkViewer";
import { RecordPanel } from "./RecordPanel";
import { LineagePanel, type LineageView } from "./LineagePanel";

export type ToolsTab = "review" | "checks" | "units" | "candidates" | "restore" | "categories" | "lineage";

const TABS: { key: ToolsTab; label: string; title: string }[] = [
  { key: "review", label: "Review & verify", title: "Claude explanation record, Mike's verification, import/export" },
  { key: "checks", label: "Checks & coverage", title: "Function preservation, closed-form error, coverage and single-root checklist" },
  { key: "units", label: "Unit activity", title: "Dead / always-on (affine) / rare / hinge hidden units on train" },
  { key: "candidates", label: "Candidates", title: "Candidate leaf subgraphs from the topology" },
  { key: "restore", label: "Restore gain", title: "Linearise annotations: how much each nonlinearity matters (train AUC)" },
  { key: "categories", label: "Categories", title: "One-hot levels grouped by base feature: model effect vs data" },
  { key: "lineage", label: "Lineage", title: "Ancestor chain, jumps and diffs" },
];

type Props = {
  genomeId: string;
  model: ModelState;
  annotations: AnnotationSummary[];
  refreshKey: number;
  onSelectNodes: (ids: string[]) => void;
  onHighlight: (nodes: Record<string, HighlightKind> | null, edges?: Record<string, HighlightKind> | null) => void;
  onLineageView: (view: LineageView | null) => void;
  onOpenGenome: (genomeId: string) => void;
  onOperationChange: () => void;
};

export const fmt = (v: number | null | undefined, digits = 3) =>
  v == null || Number.isNaN(v) ? "—" : Math.abs(v) < 1e-3 && v !== 0 ? v.toExponential(1) : v.toFixed(digits);

function useLoader<T>(fn: () => Promise<T>, deps: unknown[], enabled: boolean) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const run = useCallback(() => {
    setLoading(true);
    setError(null);
    fn()
      .then(setData)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  useEffect(() => {
    if (enabled) run();
  }, [enabled, run]);
  return { data, error, loading, reload: run };
}

const Status = ({ ok, label }: { ok: boolean; label: string }) => (
  <span className={`check-pill ${ok ? "ok" : "bad"}`}>{ok ? "✓" : "✗"} {label}</span>
);

export function ToolsDrawer(props: Props) {
  const [open, setOpen] = useState(true);
  const [tab, setTab] = useState<ToolsTab>("review");
  const { genomeId, model, annotations, refreshKey, onSelectNodes, onHighlight } = props;

  // Clear highlights when switching tabs
  useEffect(() => {
    onHighlight(null);
    if (tab !== "lineage") props.onLineageView(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tab]);

  return (
    <div className={`tools-drawer ${open ? "open" : "closed"}`}>
      <div className="tools-tabs">
        <button className="tools-toggle" onClick={() => setOpen(!open)} title={open ? "Hide tools" : "Show tools"}>
          {open ? "▼" : "▲"} Tools
        </button>
        {TABS.map((t) => (
          <button
            key={t.key}
            className={`tools-tab ${tab === t.key && open ? "active" : ""}`}
            title={t.title}
            onClick={() => {
              setTab(t.key);
              setOpen(true);
            }}
          >
            {t.label}
          </button>
        ))}
      </div>
      {open && (
        <div className="tools-body">
          {tab === "review" && (
            <RecordPanel genomeId={genomeId} onOperationChange={props.onOperationChange} />
          )}
          {tab === "checks" && (
            <ChecksTab genomeId={genomeId} refreshKey={refreshKey} onSelectNodes={onSelectNodes} onHighlight={onHighlight} />
          )}
          {tab === "units" && <UnitsTab genomeId={genomeId} refreshKey={refreshKey} onSelectNodes={onSelectNodes} />}
          {tab === "candidates" && (
            <CandidatesTab genomeId={genomeId} refreshKey={refreshKey} onSelectNodes={onSelectNodes} onHighlight={onHighlight} />
          )}
          {tab === "restore" && <RestoreTab genomeId={genomeId} refreshKey={refreshKey} annotations={annotations} />}
          {tab === "categories" && <CategoriesTab genomeId={genomeId} refreshKey={refreshKey} model={model} onSelectNodes={onSelectNodes} />}
          {tab === "lineage" && (
            <LineagePanel genomeId={genomeId} onView={props.onLineageView} onOpenGenome={props.onOpenGenome} />
          )}
        </div>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ checks

function ChecksTab({
  genomeId,
  refreshKey,
  onSelectNodes,
  onHighlight,
}: {
  genomeId: string;
  refreshKey: number;
  onSelectNodes: (ids: string[]) => void;
  onHighlight: Props["onHighlight"];
}) {
  const { data, error, loading, reload } = useLoader<CheckResult>(() => getCheck(genomeId), [genomeId, refreshKey], true);
  const [showUncovered, setShowUncovered] = useState(true);

  useEffect(() => {
    if (!data || !showUncovered) {
      onHighlight(null);
      return;
    }
    const nodes: Record<string, HighlightKind> = {};
    for (const n of data.coverage.node_coverage.uncovered) nodes[n] = "uncovered";
    const edges: Record<string, HighlightKind> = {};
    for (const [a, b] of data.coverage.edge_coverage.uncovered) edges[`${a}->${b}`] = "uncovered";
    onHighlight(nodes, edges);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, showUncovered]);

  if (loading && !data) return <p className="hint">Running checks on the training split…</p>;
  if (error) return <p className="error-text">{error}</p>;
  if (!data) return null;
  const cov = data.coverage;
  const h = data.hierarchy;
  const uncovered = cov.node_coverage.uncovered;
  return (
    <div className="tools-grid">
      <div>
        <h5>Checklist (train, {data.n_rows} rows)</h5>
        <ul className="checklist">
          <li><Status ok={data.function_preserved} label={`Function preserved (max |Δ| ${fmt(data.function_preservation_max_abs_err)})`} /></li>
          <li><Status ok={cov.structural_coverage === 1} label={`Structural coverage ${fmt(cov.structural_coverage, 3)}`} /></li>
          <li><Status ok={cov.compositional_coverage === 1} label={`Compositional coverage ${fmt(cov.compositional_coverage, 3)}`} /></li>
          <li><Status ok={h.single_root} label={`Single root (${h.roots.length ? h.roots.join(", ") : "no annotations"})`} /></li>
          <li><Status ok={h.root_contains_outputs} label="Root contains the output" /></li>
          <li><Status ok={h.all_under_root} label={h.not_under_root.length ? `Not under root: ${h.not_under_root.join(", ")}` : "Every annotation under the root"} /></li>
          <li><Status ok={!!data.full_coverage} label="Full coverage (all of the above)" /></li>
        </ul>
        <button className="op-btn secondary" onClick={reload} disabled={loading}>{loading ? "Running…" : "Re-run checks"}</button>
      </div>
      <div>
        <h5>
          Uncovered nodes ({uncovered.length}){" "}
          <label className="hint" style={{ fontWeight: 400 }}>
            <input type="checkbox" checked={showUncovered} onChange={(e) => setShowUncovered(e.target.checked)} /> highlight in graph
          </label>
        </h5>
        {uncovered.length === 0 ? (
          <p className="hint">Every candidate node is covered.</p>
        ) : (
          <>
            <div className="chip-list">
              {uncovered.map((n) => (
                <button key={n} className="chip" onClick={() => onSelectNodes([n])}>{n}</button>
              ))}
            </div>
            <button className="op-btn secondary" onClick={() => onSelectNodes(uncovered)}>Select all uncovered</button>
            <p className="hint">
              Inputs with no edges count too: list them as zero-effect entries of the root annotation, not as a separate
              annotation (that breaks the single root).
            </p>
          </>
        )}
        {cov.edge_coverage.uncovered.length > 0 && (
          <p className="hint">{cov.edge_coverage.uncovered.length} uncovered edge(s), shown dashed orange.</p>
        )}
      </div>
      <div>
        <h5>Closed form vs subgraph (per leaf)</h5>
        <table className="tools-table">
          <thead>
            <tr><th>Leaf</th><th title="AnnotationFunction vs forward pass">numeric err</th><th title="3 s.f. displayed formula vs forward pass">displayed err</th><th>exit scale</th></tr>
          </thead>
          <tbody>
            {data.leaves.map((l) => {
              const cf = l.closed_form_max_abs_err;
              const bad = typeof cf === "number" && l.exit_scale != null && cf > 0.01 * Math.max(1, l.exit_scale);
              return (
                <tr key={l.id} className={bad ? "warn-row" : ""}>
                  <td>{l.name}</td>
                  <td>{fmt(l.numeric_max_abs_err)}</td>
                  <td>{typeof cf === "string" ? cf : fmt(cf)}</td>
                  <td>{fmt(l.exit_scale ?? null, 2)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <p className="hint">
          The displayed formula rounds to 3 significant figures, so its error grows with the exit's magnitude.
          Operations: {Object.entries(data.op_counts).map(([k, v]) => `${k} ${v}`).join(", ") || "none"}.
        </p>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ units

const STATUS_HELP: Record<string, string> = {
  dead: "never active on train: output identically 0",
  rare: "active on ≤ 5 train rows",
  always_on: "active on every train row: affine on the data",
  hinge: "switches on some rows and off on others",
  identity: "identity node (linear)",
};

function UnitsTab({ genomeId, refreshKey, onSelectNodes }: { genomeId: string; refreshKey: number; onSelectNodes: (ids: string[]) => void }) {
  const { data, error, loading } = useLoader(() => getUnitActivity(genomeId), [genomeId, refreshKey], true);
  if (loading && !data) return <p className="hint">Computing unit activity…</p>;
  if (error) return <p className="error-text">{error}</p>;
  if (!data) return null;
  const counts = data.units.reduce<Record<string, number>>((acc, u) => ({ ...acc, [u.status]: (acc[u.status] || 0) + 1 }), {});
  return (
    <div>
      <p className="hint">
        {data.units.length} hidden units on {data.n_rows} train rows:{" "}
        {Object.entries(counts).map(([k, v]) => `${v} ${k.replace("_", " ")}`).join(", ")}. Click a row to select the node.
      </p>
      <table className="tools-table clickable">
        <thead>
          <tr><th>Node</th><th>Status</th><th>Active rows</th><th>Mean</th><th>SD</th><th title="Sum of weights to the output">w→out</th><th title="SD of the unit's direct logit contribution">logit SD</th><th>Annotation</th></tr>
        </thead>
        <tbody>
          {data.units.map((u: UnitRow) => (
            <tr key={u.node_id} onClick={() => onSelectNodes([u.node_id])}>
              <td>{u.display_name !== u.node_id ? `${u.display_name} (${u.node_id})` : u.node_id}</td>
              <td><span className={`status-tag ${u.status}`} title={STATUS_HELP[u.status]}>{u.status.replace("_", " ")}</span></td>
              <td>{u.active_rows} ({(100 * u.active_fraction).toFixed(0)}%)</td>
              <td>{fmt(u.mean, 2)}</td>
              <td>{fmt(u.std, 2)}</td>
              <td>{u.direct_output_weight ? fmt(u.direct_output_weight, 2) : ""}</td>
              <td>{u.direct_output_weight ? fmt(u.direct_logit_sd, 3) : ""}</td>
              <td>{u.annotation || <span className="hint">—</span>}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ------------------------------------------------------------------ candidates

function CandidatesTab({
  genomeId,
  refreshKey,
  onSelectNodes,
  onHighlight,
}: {
  genomeId: string;
  refreshKey: number;
  onSelectNodes: (ids: string[]) => void;
  onHighlight: Props["onHighlight"];
}) {
  const { data, error, loading } = useLoader<CandidatesResult>(() => getCandidates(genomeId), [genomeId, refreshKey], true);
  if (loading && !data) return <p className="hint">Reading the topology…</p>;
  if (error) return <p className="error-text">{error}</p>;
  if (!data) return null;
  const focus = (ids: string[]) => {
    onHighlight(Object.fromEntries(ids.map((i) => [i, "focus" as HighlightKind])));
    onSelectNodes(ids);
  };
  return (
    <div className="tools-grid">
      <div style={{ gridColumn: "span 2" }}>
        <h5>Candidate leaf subgraphs ({data.candidates.length})</h5>
        {data.candidates.length === 0 ? (
          <p className="hint">Every hidden node is already in an annotation.</p>
        ) : (
          <table className="tools-table clickable">
            <thead><tr><th>Candidate</th><th>Entries</th><th>Splits needed</th><th>Already covered</th></tr></thead>
            <tbody>
              {data.candidates.map((c, i) => (
                <tr key={i} onClick={() => focus(c.subgraph_nodes)} title="Select these nodes; then use the Operations panel's annotation strategy">
                  <td>{c.clean ? "● " : "○ "}{c.title}</td>
                  <td>{c.entry_names.join(", ")}</td>
                  <td>{c.splits_needed.join(", ") || "—"}</td>
                  <td>{c.already_covered.join(", ") || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <p className="hint">
          ● clean (no splits needed). Clicking selects the subgraph: the Operations panel then proposes the splits /
          identity nodes and the annotation.
        </p>
      </div>
      <div>
        <h5>Direct input → output edges ({data.direct_inputs.length})</h5>
        <div className="chip-list">
          {data.direct_inputs.map((d) => (
            <span key={d.node_id} className="chip static">{d.name}</span>
          ))}
        </div>
        {data.direct_inputs.length > 0 && (
          <button className="op-btn secondary" onClick={() => focus([...data.direct_inputs.map((d) => d.node_id), ...data.output_nodes])}>
            Select as the linear-part candidate
          </button>
        )}
        <h5>Unconnected inputs ({data.unconnected_inputs.length})</h5>
        <div className="chip-list">
          {data.unconnected_inputs.map((d) => (
            <span key={d.node_id} className="chip static">{d.name}</span>
          ))}
        </div>
        <p className="hint">List these as zero-effect entries of the root annotation.</p>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ restore gain

function RestoreTab({ genomeId, refreshKey, annotations }: { genomeId: string; refreshKey: number; annotations: AnnotationSummary[] }) {
  const [selected, setSelected] = useState<string[]>([]);
  const [run, setRun] = useState(0);
  const { data, error, loading } = useLoader<LineariseResult>(
    () => linearise(genomeId, selected.length ? selected : undefined),
    [genomeId, refreshKey, run],
    annotations.length > 0,
  );
  const names = useMemo(() => annotations.map((a) => a.name || a.id), [annotations]);
  if (annotations.length === 0) return <p className="hint">Annotate first: restore gain is computed per annotation.</p>;
  return (
    <div className="tools-grid">
      <div style={{ gridColumn: "span 2" }}>
        {loading && <p className="hint">Linearising…</p>}
        {error && <p className="error-text">{error}</p>}
        {data && (
          <>
            <p>
              Train AUC: full <strong>{fmt(data.full, 4)}</strong>, all nonlinear units linearised{" "}
              <strong>{fmt(data.all_linear, 4)}</strong> (Δ {fmt(data.full - data.all_linear, 4)}). Units: {data.scope}.
            </p>
            <table className="tools-table">
              <thead>
                <tr>
                  <th>Unit</th><th>Nonlinear?</th><th title="min R² of the affine fit of the exits on the entries">fit R²</th>
                  <th title="AUC with only this unit linearised">AUC lin. alone</th>
                  <th title="full AUC − AUC with this unit linearised">drop</th>
                  <th title="AUC with every other nonlinear unit linearised − all-linear AUC">restore gain</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(data.units)
                  .sort((a, b) => b[1].restore_gain - a[1].restore_gain)
                  .map(([name, u]) => (
                    <tr key={name}>
                      <td>{name}</td>
                      <td>{u.nonlinear ? "yes" : "no (affine on train)"}</td>
                      <td>{fmt(u.min_r2_fit, 4)}</td>
                      <td>{fmt(u.linearised_alone, 4)}</td>
                      <td>{fmt(u.linearised_alone_drop, 4)}</td>
                      <td><strong>{fmt(u.restore_gain, 4)}</strong></td>
                    </tr>
                  ))}
              </tbody>
            </table>
            <p className="hint">
              Linearising replaces a unit's exits by their least-squares affine fit on its entries (fitted on train). A
              composition is linearised over its whole subtree. Restore gain = how much AUC the unit's nonlinearity adds
              on its own (churn study A).
            </p>
          </>
        )}
      </div>
      <div>
        <h5>Units</h5>
        <p className="hint">Default: the root's children (else all leaves). Pick to override:</p>
        <div className="chip-list">
          {names.map((n) => (
            <button
              key={n}
              className={`chip ${selected.includes(n) ? "on" : ""}`}
              onClick={() => setSelected((s) => (s.includes(n) ? s.filter((x) => x !== n) : [...s, n]))}
            >
              {n}
            </button>
          ))}
        </div>
        <button className="op-btn primary" onClick={() => setRun((r) => r + 1)} disabled={loading}>Compute</button>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ categories

function CategoriesTab({
  genomeId,
  refreshKey,
  model,
  onSelectNodes,
}: {
  genomeId: string;
  refreshKey: number;
  model: ModelState;
  onSelectNodes: (ids: string[]) => void;
}) {
  const { data, error, loading } = useLoader(() => getCategoryOffsets(genomeId), [genomeId, refreshKey], true);
  const [openGroup, setOpenGroup] = useState<string | null>(null);
  const [onlyFlagged, setOnlyFlagged] = useState(false);
  if (loading && !data) return <p className="hint">Computing per-level effects…</p>;
  if (error) return <p className="error-text">{error}</p>;
  if (!data) return null;
  if (data.groups.length === 0) return <p className="hint">No one-hot (base:level) inputs reach the output.</p>;
  const nodesForColumn = (col: number) => {
    const base = `-${col + 1}`;
    return model.nodes.filter((n) => n.id === base || n.id.startsWith(`${base}_`)).map((n) => n.id);
  };
  const groups: CategoryGroup[] = onlyFlagged ? data.groups.filter((g) => g.n_large_unsupported > 0) : data.groups;
  return (
    <div>
      <p className="hint">
        One-hot inputs grouped by base feature. Effect = mean change of the logit on the level's rows when the level is
        switched off (z-scored "0" value). Flag ⚠ = |effect| ≥ {data.large_effect} logit and the model's mean prediction on
        those rows lies outside the 95% Wilson interval of their observed rate (a rare-category switch the data does not
        support; in churn only 39% of these held on test). Train split, {data.n_rows} rows.{" "}
        <label><input type="checkbox" checked={onlyFlagged} onChange={(e) => setOnlyFlagged(e.target.checked)} /> only groups with ⚠</label>
      </p>
      <table className="tools-table clickable">
        <thead>
          <tr><th>Feature</th><th>Levels used</th><th>max |effect|</th><th>⚠</th></tr>
        </thead>
        <tbody>
          {groups.map((g) => (
            <Fragment key={g.base_feature}>
              <tr onClick={() => setOpenGroup(openGroup === g.base_feature ? null : g.base_feature)}>
                <td>{openGroup === g.base_feature ? "▼" : "▶"} {g.base_feature}:*</td>
                <td>{g.n_levels_used} / {g.n_levels}</td>
                <td>{fmt(g.max_abs_effect, 2)}</td>
                <td>{g.n_large_unsupported || ""}</td>
              </tr>
              {openGroup === g.base_feature && (
                <tr>
                  <td colSpan={4}>
                    <table className="tools-table inner clickable">
                      <thead>
                        <tr><th>Level</th><th>Rows</th><th>Rate</th><th>Model p</th><th>Model effect</th><th>Empirical offset</th><th>Warranted</th><th>Rate 95% CI</th><th></th></tr>
                      </thead>
                      <tbody>
                        {g.levels.map((l) => (
                          <tr key={l.feature} className={l.flag === "large_unsupported" ? "warn-row" : ""} onClick={() => onSelectNodes(nodesForColumn(l.column))}>
                            <td>{l.feature}</td>
                            <td>{l.rows} ({l.positives}+)</td>
                            <td>{fmt(l.rate, 3)}</td>
                            <td>{fmt(l.model_mean_p, 3)}</td>
                            <td>{fmt(l.model_logit_effect, 2)}</td>
                            <td>{fmt(l.empirical_logodds_offset, 2)}</td>
                            <td>{fmt(l.warranted_offset, 2)}</td>
                            <td>[{fmt(l.rate_ci[0], 2)}, {fmt(l.rate_ci[1], 2)}]</td>
                            <td>{l.flag === "large_unsupported" ? "⚠" : l.flag === "large" ? "large" : ""}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </td>
                </tr>
              )}
            </Fragment>
          ))}
        </tbody>
      </table>
    </div>
  );
}
