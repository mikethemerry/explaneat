/**
 * The Claude-produced explanation records (experiments/10seed/explanations/*.json)
 * with their verification status: Mike's review queue.
 */
import { useEffect, useMemo, useState } from "react";
import { listRecords, type RecordSummary } from "../api/tools";
import { getReviewProgress, type ReviewProgress } from "../api/review";
import { ExportModal } from "./ExportPanel";

export function RecordsPage({
  onOpen,
  onVerify,
}: {
  onOpen: (genomeId: string, experimentId: string, experimentName: string) => void;
  onVerify: (label: string) => void;
}) {
  const [records, setRecords] = useState<RecordSummary[]>([]);
  const [progress, setProgress] = useState<Record<string, ReviewProgress>>({});
  const [dir, setDir] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState("");
  const [onlyUnverified, setOnlyUnverified] = useState(false);
  const [exporting, setExporting] = useState<{ id: string; name: string } | null>(null);

  useEffect(() => {
    listRecords()
      .then((r) => {
        setRecords(r.records);
        setDir(r.records_dir);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
    getReviewProgress()
      .then((p) => setProgress(p.labels))
      .catch(() => setProgress({}));
  }, []);

  const shown = useMemo(
    () =>
      records.filter(
        (r) => (!onlyUnverified || !r.verified) && (!filter || r.label.toLowerCase().includes(filter.toLowerCase())),
      ),
    [records, filter, onlyUnverified],
  );
  const nVerified = records.filter((r) => r.verified).length;
  const minutes = records.reduce((a, r) => a + (r.mike_verification?.minutes || 0), 0);

  return (
    <div className="experiment-list-container records-page">
      <header className="experiment-list-header">
        <h2>Review explanations</h2>
        <p className="hint">
          {records.length} explanation records in <code>{dir}</code>. <strong>{nVerified}</strong> verified
          {minutes ? `, ${minutes} min recorded` : ""}. <strong>Verify</strong> opens the sentence-by-sentence review (signoff-v1:
          gates, tier A, the random sample, leaf verdicts, sign-off); <em>Explore</em> opens the genome in the explorer.
        </p>
        <div style={{ display: "flex", gap: "0.75rem", alignItems: "center" }}>
          <input
            type="search"
            placeholder="Filter (e.g. heart, churn100)"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            style={{ padding: "6px 10px", border: "1px solid #d1d5db", borderRadius: 4, minWidth: 240 }}
          />
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
          {shown.map((r) => (
            <tr key={r.label}>
              <td><strong>{r.label}</strong></td>
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
                <button className="explore-btn" title="Sentence-by-sentence review and sign-off" onClick={() => onVerify(r.label)}>
                  Verify
                </button>{" "}
                <button
                  className="op-btn"
                  disabled={!r.in_db || !r.experiment_id}
                  title={r.in_db ? "Open in the explorer" : "Genome not in this database"}
                  onClick={() => r.experiment_id && onOpen(r.genome_id, r.experiment_id, r.experiment_name || r.label)}
                >
                  Explore
                </button>{" "}
                <button
                  className="op-btn"
                  disabled={!r.in_db}
                  title={r.in_db ? "Export pack (JSON, LaTeX, TikZ, PDF, DOCX)" : "Genome not in this database"}
                  onClick={() => setExporting({ id: r.genome_id, name: r.label })}
                >
                  Export
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {exporting && (
        <ExportModal target={{ kind: "genome", id: exporting.id, name: exporting.name }} onClose={() => setExporting(null)} />
      )}
    </div>
  );
}
