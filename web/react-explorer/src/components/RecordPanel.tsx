/**
 * Review a Claude-produced explanation record against the live checks, record
 * Mike's verification (`mike_verification` in the record JSON), and export /
 * import the explanation's operations.
 */
import { useEffect, useRef, useState } from "react";
import {
  exportOperationsUrl,
  getCheck,
  getGenomeRecord,
  importOperations,
  importPreview,
  putVerification,
  type CheckResult,
  type ExplanationRecord,
  type ImportPreview,
} from "../api/tools";

const fmt = (v: unknown, d = 3) =>
  typeof v === "number" ? (Math.abs(v) < 1e-3 && v !== 0 ? v.toExponential(1) : v.toFixed(d)) : v == null ? "—" : String(v);

function Agreement({ label, recorded, live, tol }: { label: string; recorded: unknown; live: unknown; tol?: number }) {
  let ok: boolean | null = null;
  if (typeof recorded === "number" && typeof live === "number") {
    ok = Math.abs(recorded - live) <= (tol ?? 1e-6) * Math.max(1, Math.abs(recorded));
  } else if (typeof recorded === "boolean" && typeof live === "boolean") {
    ok = recorded === live;
  }
  return (
    <tr className={ok === false ? "warn-row" : ""}>
      <td>{label}</td>
      <td>{fmt(recorded)}</td>
      <td>{fmt(live)}</td>
      <td>{ok == null ? "" : ok ? "✓" : "✗ differs"}</td>
    </tr>
  );
}

export function RecordPanel({ genomeId, onOperationChange }: { genomeId: string; onOperationChange: () => void }) {
  const [record, setRecord] = useState<ExplanationRecord | null>(null);
  const [recordsDir, setRecordsDir] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [check, setCheck] = useState<CheckResult | null>(null);
  const [checking, setChecking] = useState(false);

  // verification form
  const [verified, setVerified] = useState(false);
  const [minutes, setMinutes] = useState<string>("");
  const [changes, setChanges] = useState("");
  const [notes, setNotes] = useState("");
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState<string | null>(null);
  const startedAt = useRef<number>(Date.now());
  const [elapsed, setElapsed] = useState(0);

  // import
  const [importDoc, setImportDoc] = useState<unknown>(null);
  const [preview, setPreview] = useState<ImportPreview | null>(null);
  const [importMsg, setImportMsg] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    startedAt.current = Date.now();
    getGenomeRecord(genomeId)
      .then((r) => {
        setRecord(r.record);
        setRecordsDir(r.records_dir);
        const mv = r.record?.mike_verification;
        setVerified(!!mv?.verified);
        setMinutes(mv?.minutes != null ? String(mv.minutes) : "");
        setChanges(mv?.changes || "");
        setNotes(mv?.notes || "");
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
  }, [genomeId]);

  useEffect(() => {
    const t = setInterval(() => setElapsed(Math.round((Date.now() - startedAt.current) / 60000)), 30000);
    return () => clearInterval(t);
  }, []);

  const runChecks = () => {
    setChecking(true);
    getCheck(genomeId)
      .then(setCheck)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setChecking(false));
  };

  const save = async () => {
    if (!record) return;
    setSaving(true);
    setSaved(null);
    try {
      const m = minutes.trim() === "" ? null : Number(minutes);
      const r = await putVerification(record.label, { verified, minutes: m, changes, notes });
      setRecord({ ...record, mike_verification: r.mike_verification });
      setSaved(`Saved to ${record.label}.json at ${new Date().toLocaleTimeString()}. Rebuild the CSVs with experiments/10seed/build_explanations_csv.py.`);
    } catch (e) {
      setSaved(`Error: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setSaving(false);
    }
  };

  const onFile = async (f: File | undefined) => {
    setPreview(null);
    setImportMsg(null);
    if (!f) return;
    try {
      const doc = JSON.parse(await f.text());
      setImportDoc(doc);
      setImportMsg("Replaying in memory…");
      const p = await importPreview(genomeId, doc);
      setPreview(p);
      setImportMsg(null);
    } catch (e) {
      setImportMsg(`Cannot import: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  const doImport = async () => {
    if (!preview) return;
    const replace = preview.n_existing_operations > 0;
    if (replace && !confirm(`Replace the genome's ${preview.n_existing_operations} existing operations with the ${preview.n_operations} imported ones?`)) return;
    try {
      const r = await importOperations(genomeId, importDoc, replace);
      setImportMsg(`Imported ${r.n_operations} operations.`);
      setPreview(null);
      onOperationChange();
    } catch (e) {
      setImportMsg(`Import failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  if (loading) return <p className="hint">Looking for an explanation record…</p>;
  const orch = (record?.verified_by_orchestrator || {}) as Record<string, unknown>;
  const hier = (orch.hierarchy || {}) as Record<string, unknown>;

  return (
    <div className="tools-grid">
      <div style={{ gridColumn: "span 2" }}>
        {error && <p className="error-text">{error}</p>}
        {!record ? (
          <p className="hint">
            No Claude explanation record for this genome in <code>{recordsDir}</code>. You can still import an
            explanation file (right) or explain it here.
          </p>
        ) : (
          <>
            <h5>
              Record <code>{record.label}</code>{" "}
              {record.mike_verification?.verified ? <span className="check-pill ok">✓ verified by Mike</span> : <span className="check-pill bad">not verified</span>}
            </h5>
            {record.summary && <p><strong>Summary.</strong> {record.summary}</p>}
            {record.resisted && <p><strong>What resisted.</strong> {record.resisted}</p>}
            {record.tool_bugs && <p className="hint"><strong>Tool notes.</strong> {record.tool_bugs}</p>}
            <h5>Recorded vs live checks</h5>
            <table className="tools-table">
              <thead><tr><th></th><th>Record (orchestrator)</th><th>Live</th><th></th></tr></thead>
              <tbody>
                <Agreement label="Structural coverage" recorded={orch.structural_coverage ?? record.structural_coverage} live={check?.coverage.structural_coverage} tol={1e-4} />
                <Agreement label="Compositional coverage" recorded={orch.compositional_coverage ?? record.compositional_coverage} live={check?.coverage.compositional_coverage} tol={1e-4} />
                <Agreement label="Hierarchy complete" recorded={hier.complete ?? record.hierarchy_complete} live={check?.hierarchy.complete} />
                <Agreement
                  label="Function preserved (max |Δ| < 1e-9)"
                  recorded={typeof (orch.function_preservation_max_abs_err ?? record.function_preservation_max_abs_err) === "number"
                    ? Number(orch.function_preservation_max_abs_err ?? record.function_preservation_max_abs_err) < 1e-9 : undefined}
                  live={check ? check.function_preservation_max_abs_err < 1e-9 : undefined}
                />
                <Agreement label="Closed-form max error" recorded={orch.closed_form_max_abs_err ?? record.closed_form_max_abs_err} live={check?.closed_form_max_abs_err} tol={1e-6} />
                <Agreement label="Full coverage" recorded={orch.full_coverage ?? record.full_coverage} live={check?.full_coverage} />
              </tbody>
            </table>
            <button className="op-btn secondary" onClick={runChecks} disabled={checking}>
              {checking ? "Running checks…" : check ? "Re-run live checks" : "Run live checks"}
            </button>
            <p className="hint">
              Review the annotations (left list; click one for its formula, hypothesis and evidence), then record your
              verdict below.
            </p>
          </>
        )}
      </div>
      <div>
        {record && (
          <>
            <h5>Mike's verification</h5>
            <label className="form-row">
              <input type="checkbox" checked={verified} onChange={(e) => setVerified(e.target.checked)} /> Verified by Mike
            </label>
            <label className="form-row">
              Minutes{" "}
              <input type="number" min={0} value={minutes} onChange={(e) => setMinutes(e.target.value)} style={{ width: 70 }} />
              <button className="op-btn secondary" type="button" onClick={() => setMinutes(String(Math.max(1, elapsed)))} title="Minutes since this record was opened">
                use timer ({elapsed} min)
              </button>
            </label>
            <label className="form-col">
              Changes made
              <textarea rows={2} value={changes} onChange={(e) => setChanges(e.target.value)} placeholder="e.g. renamed ca_relay; none" />
            </label>
            <label className="form-col">
              Notes
              <textarea rows={2} value={notes} onChange={(e) => setNotes(e.target.value)} />
            </label>
            <button className="op-btn primary" onClick={save} disabled={saving}>{saving ? "Saving…" : "Save verification"}</button>
            {saved && <p className="hint">{saved}</p>}
          </>
        )}
        <h5>Export / import</h5>
        <a className="op-btn secondary" href={exportOperationsUrl(genomeId)} download>Export explanation (JSON)</a>
        <label className="form-col" style={{ marginTop: 6 }}>
          Import explanation (exported JSON or an operations list)
          <input type="file" accept="application/json,.json" onChange={(e) => onFile(e.target.files?.[0])} />
        </label>
        {importMsg && <p className="hint">{importMsg}</p>}
        {preview && (
          <div className="import-preview">
            <p>
              {preview.n_operations} operations replay cleanly.{" "}
              {preview.identical_to_existing ? "Identical to the current explanation." : `The genome currently has ${preview.n_existing_operations}.`}
            </p>
            <p>
              Function preserved: {preview.function_preserved ? "yes" : "NO"} (max |Δ| {fmt(preview.function_preservation_max_abs_err)}),
              coverage {fmt(preview.coverage.structural_coverage)} / {fmt(preview.coverage.compositional_coverage)},
              hierarchy {preview.hierarchy.complete ? "complete" : "incomplete"}.
            </p>
            {!preview.identical_to_existing && (
              <button className="op-btn primary" onClick={doImport}>
                {preview.n_existing_operations ? "Replace current explanation" : "Save as this genome's explanation"}
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
