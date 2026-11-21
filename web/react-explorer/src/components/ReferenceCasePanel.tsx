/**
 * Case explorer: the reference case the "others held fixed" plots hold the other inputs at.
 * A one-line strip (on/off, what the case is, its prediction, how typical it is) that opens into
 * an editor: base (median case, per-column median, mean, a percentile, a dataset row), a value
 * per input in the data's units, a choice per one-hot group, the nearest training rows and a row
 * browser. State: components/referenceCase.ts (one case per genome, in the URL).
 */
import { useEffect, useMemo, useRef, useState } from "react";
import {
  getCaseRows,
  getReferenceCase,
  type CaseColumn,
  type CaseOutcome,
  type CaseRowsResponse,
  type CaseSpec,
  type ReferenceCaseResponse,
} from "../api/client";
import { STATIC_MODE } from "../api/staticData";
import {
  bindGenome,
  describeSpec,
  setCase,
  useReferenceCase,
  withBase,
  withGroup,
  withOverride,
} from "./referenceCase";

type Props = {
  genomeId: string;
  splitId: string | null;
  annotationId?: string;
  nodeId?: string;
};

const fmt = (v: number) =>
  !Number.isFinite(v) ? String(v) : Math.abs(v) >= 1000 || Number.isInteger(v) ? String(Math.round(v * 1000) / 1000) : v.toPrecision(3);
function ordinal(n: number): string {
  const t = n % 100;
  if (t >= 11 && t <= 13) return `${n}th`;
  return `${n}${["th", "st", "nd", "rd"][n % 10] ?? "th"}`;
}
const pct = (p: number) => `${ordinal(Math.round(p * 100))} pct`;

function outcomeText(o: CaseOutcome): string {
  if (o == null) return "–";
  if (typeof o === "object") return `${o.event ? "event" : "censored"} at ${fmt(o.time)}`;
  return String(o);
}

export function ReferenceCasePanel({ genomeId, splitId, annotationId, nodeId }: Props) {
  useEffect(() => bindGenome(genomeId), [genomeId]);
  const c = useReferenceCase();
  const [open, setOpen] = useState(false);
  const [resp, setResp] = useState<ReferenceCaseResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [filter, setFilter] = useState("");
  const seq = useRef(0);

  // what is shown: the case when on, else what the plots use (each column's median)
  const shownSpec: CaseSpec = c.enabled ? c.spec : { base: "per_column_median" };
  const specKey = JSON.stringify(shownSpec);

  useEffect(() => {
    if (STATIC_MODE || !splitId || (!open && !c.enabled)) return;
    const mine = ++seq.current;
    setLoading(true);
    const t = window.setTimeout(async () => {
      try {
        const r = await getReferenceCase(genomeId, {
          dataset_split_id: splitId, annotation_id: annotationId, node_id: nodeId, spec: shownSpec,
        });
        if (mine !== seq.current) return;
        setResp(r);
        setError(null);
        if (c.enabled) setCase({ x: r.x });
      } catch (e) {
        if (mine === seq.current) setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (mine === seq.current) setLoading(false);
      }
    }, 250);
    return () => window.clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [genomeId, splitId, annotationId, nodeId, specKey, open, c.enabled]);

  if (STATIC_MODE) {
    return (
      <div className="case-strip case-static">
        <span className="case-title">Reference case</span>
        <span className="dim-hint">Plots hold the other inputs at each one's median. Choosing a case needs the live explorer.</span>
      </div>
    );
  }

  const updateSpec = (spec: CaseSpec) => setCase({ spec, x: null });
  const toggle = (on: boolean) => {
    setCase({ enabled: on, x: null });
    if (on) setOpen(true);
  };

  const out0 = resp?.prediction.outputs[0];
  const near = resp?.nearest[0];
  const summary = !resp
    ? (loading ? "computing…" : c.enabled ? describeSpec(c.spec) : "each input at its median")
    : [
        c.enabled ? describeSpec(c.spec) : "each input at its median",
        out0 ? `predicted ${fmt(out0.value)} (${pct(out0.percentile)})` : null,
        resp.typicality.verdict,
        near ? `nearest row #${near.row}, outcome ${outcomeText(near.outcome)}` : null,
      ].filter(Boolean).join(" · ");

  return (
    <div className={`case-strip${c.enabled ? " on" : ""}`}>
      <div className="case-head">
        <label className="case-toggle" title="Off: each input not on a plot's axes is held at its median (the default). On: all plots hold them at the case below.">
          <input type="checkbox" checked={c.enabled} onChange={(e) => toggle(e.target.checked)} />
          <span className="case-title">Hold others at a reference case</span>
        </label>
        <button className="op-btn case-open" onClick={() => setOpen(!open)}>{open ? "Hide case" : "Show case"}</button>
      </div>
      <div className="case-summary">
        {summary}
        {resp && resp.warnings.length > 0 && (
          <span className="case-warn-badge" title={resp.warnings.join("\n")}> · ⚠ {resp.warnings.length}</span>
        )}
      </div>
      {error && <div className="error-message">{error}</div>}
      {open && resp && (
        <CaseEditor
          genomeId={genomeId}
          splitId={splitId}
          resp={resp}
          enabled={c.enabled}
          spec={c.spec}
          onSpec={(s) => { if (!c.enabled) setCase({ enabled: true }); updateSpec(s); }}
          filter={filter}
          onFilter={setFilter}
        />
      )}
    </div>
  );
}

function CaseEditor({
  genomeId, splitId, resp, enabled, spec, onSpec, filter, onFilter,
}: {
  genomeId: string;
  splitId: string | null;
  resp: ReferenceCaseResponse;
  enabled: boolean;
  spec: CaseSpec;
  onSpec: (s: CaseSpec) => void;
  filter: string;
  onFilter: (f: string) => void;
}) {
  const base = enabled ? spec.base ?? "median" : "per_column_median";
  const [rowText, setRowText] = useState(spec.row != null ? String(spec.row) : "");
  const [browse, setBrowse] = useState(false);
  const grouped = useMemo(() => new Set(resp.groups.flatMap((g) => g.columns)), [resp.groups]);
  const f = filter.trim().toLowerCase();
  const match = (col: CaseColumn) => !f || col.raw.toLowerCase().includes(f) || col.full_label.toLowerCase().includes(f);
  const edited = (raw: string) => spec.overrides?.[raw] != null;

  return (
    <div className="case-editor">
      <div className="case-base">
        <label className="dim-label">Case</label>
        <select
          className="dim-select"
          value={base}
          onChange={(e) => {
            const b = e.target.value as NonNullable<CaseSpec["base"]>;
            if (b === "row") onSpec(withBase("row", { row: resp.nearest[0]?.row ?? 0 }));
            else if (b === "quantile") onSpec(withBase("quantile", { q: 0.75 }));
            else onSpec(withBase(b));
          }}
        >
          <option value="median">Median case (medians; most common value of binary inputs and of each one-hot group)</option>
          <option value="per_column_median">Per-column median (what plots use when this is off)</option>
          <option value="mean">Mean case</option>
          <option value="quantile">Percentile case</option>
          <option value="row">A dataset row</option>
        </select>
        {base === "quantile" && (
          <input
            className="case-num" type="number" min={0} max={100} step={5}
            value={Math.round((spec.q ?? 0.5) * 100)}
            onChange={(e) => onSpec({ ...spec, q: Math.min(1, Math.max(0, Number(e.target.value) / 100)) })}
            title="Percentile of every continuous input"
          />
        )}
        {base === "row" && (
          <>
            <input
              className="case-num" type="number" min={0} value={rowText}
              onChange={(e) => setRowText(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && rowText !== "") onSpec(withBase("row", { row: Number(rowText) })); }}
              onBlur={() => { if (rowText !== "" && Number(rowText) !== spec.row) onSpec(withBase("row", { row: Number(rowText) })); }}
              title="Dataset row index (Enter to use)"
            />
            {resp.row && (
              <span className="dim-hint">
                outcome {outcomeText(resp.row.outcome)}{resp.row.in_training ? "" : " (not a training row)"}
              </span>
            )}
          </>
        )}
        <button className="op-btn" onClick={() => setBrowse(!browse)}>{browse ? "Hide rows" : "Browse rows"}</button>
        {enabled && (spec.overrides || spec.groups) && (
          <button className="op-btn" onClick={() => onSpec({ ...spec, overrides: {}, groups: {} })}>Clear edits</button>
        )}
      </div>

      {browse && splitId && (
        <RowBrowser genomeId={genomeId} splitId={splitId} onPick={(r) => { setRowText(String(r)); onSpec(withBase("row", { row: r })); }} />
      )}

      <div className="case-facts">
        {resp.prediction.outputs.map((o) => (
          <div key={o.name}>Prediction {o.name}: <b>{fmt(o.value)}</b> ({pct(o.percentile)} of the training rows' predictions)</div>
        ))}
        <div>
          Typicality: <b>{resp.typicality.verdict}</b>
          <span className="dim-hint"> (distance to the nearest training row {fmt(resp.typicality.nn_distance)}, {pct(resp.typicality.percentile)} of rows' own nearest-neighbour distances; standardised over {resp.n_rows} training rows)</span>
        </div>
        {resp.warnings.map((w) => <div key={w} className="case-warning">⚠ {w}</div>)}
      </div>

      {resp.entry && (
        <details className="case-entry">
          <summary className="dim-label">
            This part's {resp.entry.names.length} entries at the case (what its plots hold, network units)
          </summary>
          <div className="case-entry-list">
            {resp.entry.names.map((n, i) => (
              <span key={resp.entry!.nodes[i]} className="case-chip">{n} = {fmt(resp.entry!.values[i])}</span>
            ))}
          </div>
        </details>
      )}

      <input className="case-filter" placeholder="Filter inputs…" value={filter} onChange={(e) => onFilter(e.target.value)} />
      <table className="case-table">
        <thead>
          <tr><th>Input</th><th>Value</th><th title="Share of training rows below this value">Pct</th><th>Median</th></tr>
        </thead>
        <tbody>
          {resp.groups.filter((g) => !f || g.base.toLowerCase().includes(f) || g.levels.some((l) => l.toLowerCase().includes(f))).map((g) => {
            const cols = g.columns.map((k) => resp.columns[k]);
            const active = g.active.length === 1 ? g.active[0] : g.active.length === 0 ? "" : "__several";
            return (
              <tr key={g.base} className={spec.groups?.[g.base] !== undefined && enabled ? "edited" : ""}>
                <td>{g.base} <span className="dim-hint">one-hot</span></td>
                <td>
                  <select className="dim-select" value={active} onChange={(e) => onSpec(withGroup(spec, g.base, e.target.value || null))}>
                    <option value="">(no level on)</option>
                    {active === "__several" && <option value="__several" disabled>several levels on</option>}
                    {cols.map((col) => <option key={col.raw} value={col.raw}>{col.raw}{col.label !== col.raw ? ` · ${col.label}` : ""}</option>)}
                  </select>
                </td>
                <td colSpan={2} className={g.support === 0 ? "case-warning" : "dim-hint"}>
                  {g.support} training row{g.support === 1 ? "" : "s"} like this
                </td>
              </tr>
            );
          })}
          {resp.columns.filter((col) => !grouped.has(col.column) && match(col)).map((col) => (
            <tr key={col.raw} className={edited(col.raw) && enabled ? "edited" : ""}>
              <td title={col.one_means ? `1 means: ${col.one_means}` : col.full_label}>
                {col.raw}{col.label !== col.raw ? <span className="dim-hint"> · {col.label}</span> : null}
                {col.unit ? <span className="dim-hint"> ({col.unit})</span> : null}
              </td>
              <td>
                {col.kind === "binary" ? (
                  <select
                    className="dim-select"
                    value={col.value === col.max ? "max" : col.value === col.min ? "min" : "other"}
                    onChange={(e) => onSpec(withOverride(spec, col.raw, e.target.value === "max" ? col.max : col.min))}
                  >
                    <option value="min">{fmt(col.min)}</option>
                    <option value="max">{fmt(col.max)}{col.one_means ? ` · ${col.one_means}` : ""}</option>
                    {col.value !== col.max && col.value !== col.min && <option value="other" disabled>{fmt(col.value)}</option>}
                  </select>
                ) : (
                  <ValueInput col={col} onSet={(v) => onSpec(withOverride(spec, col.raw, v))} />
                )}
                {edited(col.raw) && enabled && (
                  <button className="case-reset" title="Back to the base value" onClick={() => onSpec(withOverride(spec, col.raw, null))}>↺</button>
                )}
              </td>
              <td>{Math.round(col.percentile * 100)}</td>
              <td className="dim-hint">{fmt(col.median)}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <div className="case-nearest">
        <div className="dim-label">Nearest training rows</div>
        <table className="case-table">
          <thead><tr><th>Row</th><th>Distance</th><th>Outcome</th><th>Prediction</th><th>Differs most in</th><th /></tr></thead>
          <tbody>
            {resp.nearest.map((n) => (
              <tr key={n.row}>
                <td>#{n.row}</td>
                <td>{fmt(n.distance)}</td>
                <td>{outcomeText(n.outcome)}</td>
                <td>{n.prediction.map(fmt).join(", ")}</td>
                <td className="dim-hint">{n.differs.join(", ") || "–"}</td>
                <td><button className="op-btn" onClick={() => { setRowText(String(n.row)); onSpec(withBase("row", { row: n.row })); }}>Use</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ValueInput({ col, onSet }: { col: CaseColumn; onSet: (v: number | null) => void }) {
  const [text, setText] = useState(fmt(col.value));
  useEffect(() => setText(fmt(col.value)), [col.value]);
  const commit = () => {
    const v = Number(text);
    if (text.trim() !== "" && Number.isFinite(v) && fmt(v) !== fmt(col.value)) onSet(v);
  };
  return (
    <span className="case-value">
      <input
        className="case-num" type="number" value={text} step="any"
        onChange={(e) => setText(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => { if (e.key === "Enter") commit(); }}
        title={`range ${fmt(col.min)} … ${fmt(col.max)}`}
      />
      {(["median", "mean", "q25", "q75"] as const).map((k) => (
        <button key={k} className="case-quick" title={`${k} = ${fmt(col[k])}`} onClick={() => onSet(col[k])}>
          {k === "median" ? "med" : k}
        </button>
      ))}
    </span>
  );
}

function RowBrowser({ genomeId, splitId, onPick }: { genomeId: string; splitId: string; onPick: (row: number) => void }) {
  const [split, setSplit] = useState<"train" | "test">("train");
  const [outcome, setOutcome] = useState<string>("");
  const [order, setOrder] = useState<"index" | "prediction">("index");
  const [offset, setOffset] = useState(0);
  const [data, setData] = useState<CaseRowsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const limit = 10;
  useEffect(() => {
    let live = true;
    getCaseRows(genomeId, {
      dataset_split_id: splitId, split, offset, limit, order,
      outcome: outcome === "" ? null : Number(outcome),
    }).then((d) => { if (live) { setData(d); setError(null); } })
      .catch((e) => { if (live) setError(e instanceof Error ? e.message : String(e)); });
    return () => { live = false; };
  }, [genomeId, splitId, split, outcome, order, offset]);
  return (
    <div className="case-rows">
      <div className="case-base">
        <select className="dim-select" value={split} onChange={(e) => { setSplit(e.target.value as "train" | "test"); setOffset(0); }}>
          <option value="train">training rows</option>
          <option value="test">test rows</option>
        </select>
        <input className="case-num" placeholder="outcome" value={outcome} onChange={(e) => { setOutcome(e.target.value); setOffset(0); }} title="Only rows with this outcome (survival: 1 = event)" />
        <select className="dim-select" value={order} onChange={(e) => { setOrder(e.target.value as "index" | "prediction"); setOffset(0); }}>
          <option value="index">by row</option>
          <option value="prediction">by prediction</option>
        </select>
        {data && (
          <>
            <button className="op-btn" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - limit))}>‹</button>
            <span className="dim-hint">{data.total ? `${offset + 1}–${Math.min(offset + limit, data.total)} of ${data.total}` : "none"}</span>
            <button className="op-btn" disabled={offset + limit >= data.total} onClick={() => setOffset(offset + limit)}>›</button>
            <button className="op-btn" disabled={!data.total} onClick={() => setOffset(Math.floor(Math.random() * Math.max(1, data.total - limit + 1)))}>Random</button>
          </>
        )}
      </div>
      {error && <div className="error-message">{error}</div>}
      {data && (
        <table className="case-table">
          <tbody>
            {data.rows.map((r) => (
              <tr key={r.row}>
                <td>#{r.row}</td><td>{outcomeText(r.outcome)}</td><td>{r.prediction.map(fmt).join(", ")}</td>
                <td><button className="op-btn" onClick={() => onPick(r.row)}>Use</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
