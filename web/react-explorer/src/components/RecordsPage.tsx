/**
 * The Claude-produced explanation records of every study (the records database,
 * else experiments/<study>/explanations*), grouped by study, with their
 * verification status: Mike's review queue.
 */
import { useEffect, useMemo, useState } from "react";
import { listRecords, type RecordSummary } from "../api/tools";
import { getReviewProgress, type ReviewProgress } from "../api/review";
import { ExportModal } from "./ExportPanel";
import { useAppConfig } from "../api/appConfig";

export function RecordsPage({
  onOpen,
  onVerify,
}: {
  onOpen: (genomeId: string, experimentId: string, experimentName: string) => void;
  onVerify: (label: string) => void;
}) {
  const [records, setRecords] = useState<RecordSummary[]>([]);
  const [progress, setProgress] = useState<Record<string, ReviewProgress>>({});
  const [dirs, setDirs] = useState<string[]>([]);
  const [database, setDatabase] = useState(false);
  const [study, setStudy] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState("");
  const [onlyUnverified, setOnlyUnverified] = useState(false);
  const [exporting, setExporting] = useState<{ id: string; name: string } | null>(null);
  const cfg = useAppConfig();

  useEffect(() => {
    listRecords()
      .then((r) => {
        setRecords(r.records);
        setDirs(r.records_dirs?.length ? r.records_dirs : [r.records_dir]);
        setDatabase(!!r.database);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
    getReviewProgress()
      .then((p) => setProgress(p.labels))
      .catch(() => setProgress({}));
  }, []);

  const studies = useMemo(() => {
    const seen: string[] = [];
    records.forEach((r) => r.study && !seen.includes(r.study) && seen.push(r.study));
    return seen;
  }, [records]);
  const shown = useMemo(
    () =>
      records.filter(
        (r) =>
          (!onlyUnverified || !r.verified) &&
          (!study || r.study === study) &&
          (!filter || r.label.toLowerCase().includes(filter.toLowerCase())),
      ),
    [records, filter, onlyUnverified, study],
  );
  // grouped by study, in the order the studies first appear (the search order)
  const groups = useMemo(() => {
    const out: { study: string; rows: RecordSummary[] }[] = [];
    shown.forEach((r) => {
      const k = r.study || "";
      let g = out.find((x) => x.study === k);
      if (!g) out.push((g = { study: k, rows: [] }));
      g.rows.push(r);
    });
    return out;
  }, [shown]);
  const nVerified = records.filter((r) => r.verified).length;
  const minutes = records.reduce((a, r) => a + (r.mike_verification?.minutes || 0), 0);

  return (
    <div className="experiment-list-container records-page">
      <header className="experiment-list-header">
        <h2>{cfg.static ? "Explanation records" : "Review explanations"}</h2>
        {cfg.read_only ? (
          <p className="hint">
            {records.length} explanation records. <strong>{nVerified}</strong> verified by Mike
            {minutes ? ` (${minutes} min in total)` : ""}. <em>Explore</em> opens the network with its explanation.
          </p>
        ) : <p className="hint">
          {records.length} explanation records in {studies.length} {studies.length === 1 ? "study" : "studies"}
          {database ? " (records database, then files)" : ""}
          <span title={dirs.join("\n")}> from <code>{dirs[0]}</code>{dirs.length > 1 ? ` and ${dirs.length - 1} more folders` : ""}</span>.{" "}
          <strong>{nVerified}</strong> verified
          {minutes ? `, ${minutes} min recorded` : ""}. <strong>Verify</strong> opens the sentence-by-sentence review (signoff-v1:
          gates, tier A, the random sample, leaf verdicts, sign-off); <em>Explore</em> opens the genome in the explorer.
        </p>}
        <div style={{ display: "flex", gap: "0.75rem", alignItems: "center" }}>
          <input
            type="search"
            placeholder="Filter (e.g. heart, churn100)"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            style={{ padding: "6px 10px", border: "1px solid #d1d5db", borderRadius: 4, minWidth: 240 }}
          />
          {studies.length > 1 && (
            <select value={study} onChange={(e) => setStudy(e.target.value)} aria-label="Study">
              <option value="">All studies</option>
              {studies.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          )}
          <label className="hint">
            <input type="checkbox" checked={onlyUnverified} onChange={(e) => setOnlyUnverified(e.target.checked)} /> only unverified
          </label>
        </div>
      </header>
      {loading && <p className="hint">Loading records…</p>}
      {error && <p className="error-text">{error}</p>}
      <table className="experiment-table">
        <thead>
          <tr>
            <th>Record</th><th>Full coverage</th><th>Agrees</th><th>Leaves / comps</th><th>Hidden</th><th>Closed-form err</th><th>Verdicts</th><th>Verified</th><th>Summary</th><th></th>
          </tr>
        </thead>
        <tbody>
          {groups.map((grp) => [
            grp.study ? (
              <tr key={`study:${grp.study}`} className="study-row">
                <td colSpan={10}>
                  <strong>{grp.study}</strong>{" "}
                  <span className="hint">{grp.rows.length} records{grp.rows[0]?.records_dir ? ` · ${grp.rows[0].records_dir}` : ""}</span>
                </td>
              </tr>
            ) : null,
            ...grp.rows.map((r) => (
            <tr key={r.label}>
              <td><strong>{r.label}</strong>{r.storage === "db" && <span className="hint" title={r.source}> db</span>}</td>
              <td>{r.orchestrator_full_coverage == null ? "—" : r.orchestrator_full_coverage ? "yes" : "no"}</td>
              <td>{r.agrees_with_agent == null ? "—" : r.agrees_with_agent ? "✓" : "✗"}</td>
              <td>{r.n_leaf_annotations ?? "—"} / {r.n_composition_annotations ?? "—"}</td>
              <td>{r.hidden_nodes ?? "—"}</td>
              <td>{r.closed_form_max_abs_err != null ? r.closed_form_max_abs_err.toPrecision(2) : "—"}</td>
              <td>
                {progress[r.label] ? (
                  <span className="hint" title={progress[r.label].last_at || ""}>
                    {progress[r.label].n_sentences} sent., {progress[r.label].n_leaf_verdicts} leaf
                    {progress[r.label].signoff_passed === true ? " · sign-off ok" : progress[r.label].signoff_passed === false ? " · sign-off open" : ""}
                  </span>
                ) : (
                  <span className="hint">—</span>
                )}
              </td>
              <td>
                {r.verified ? (
                  <span className="check-pill ok" title={r.mike_verification?.changes || ""}>
                    ✓ {r.mike_verification?.minutes != null ? `${r.mike_verification.minutes} min` : ""}
                  </span>
                ) : (
                  <span className="hint">no</span>
                )}
              </td>
              <td className="summary-cell" title={r.summary || ""}>{r.summary}</td>
              <td>
                {!cfg.read_only && <><button className="explore-btn" title="Sentence-by-sentence review and sign-off" onClick={() => onVerify(r.label)}>
                  Verify
                </button>{" "}</>}
                <button
                  className="op-btn"
                  disabled={!r.in_db || !r.experiment_id}
                  title={r.in_db ? "Open in the explorer" : "Genome not in this database"}
                  onClick={() => r.experiment_id && onOpen(r.genome_id, r.experiment_id, r.experiment_name || r.label)}
                >
                  Explore
                </button>{" "}
                {!cfg.read_only && <button
                  className="op-btn"
                  disabled={!r.in_db}
                  title={r.in_db ? "Export pack (JSON, LaTeX, TikZ, PDF, DOCX)" : "Genome not in this database"}
                  onClick={() => setExporting({ id: r.genome_id, name: r.label })}
                >
                  Export
                </button>}
              </td>
            </tr>
            )),
          ])}
        </tbody>
      </table>
      {exporting && (
        <ExportModal target={{ kind: "genome", id: exporting.id, name: exporting.name }} onClose={() => setExporting(null)} />
      )}
    </div>
  );
}
