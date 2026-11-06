/**
 * The Claude-produced explanation records (experiments/10seed/explanations/*.json)
 * with their verification status: Mike's review queue.
 */
import { useEffect, useMemo, useState } from "react";
import { listRecords, type RecordSummary } from "../api/tools";

export function RecordsPage({ onOpen }: { onOpen: (genomeId: string, experimentId: string, experimentName: string) => void }) {
  const [records, setRecords] = useState<RecordSummary[]>([]);
  const [dir, setDir] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState("");
  const [onlyUnverified, setOnlyUnverified] = useState(false);

  useEffect(() => {
    listRecords()
      .then((r) => {
        setRecords(r.records);
        setDir(r.records_dir);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
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
          {records.length} Claude-produced explanation records in <code>{dir}</code>. <strong>{nVerified}</strong> verified
          {minutes ? `, ${minutes} min recorded` : ""}. Open one to review it: the explorer's <em>Review &amp; verify</em> tab
          compares the record with live checks and saves your verification into the record.
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
            <th>Record</th><th>Full coverage</th><th>Agrees</th><th>Leaves / comps</th><th>Hidden</th><th>Closed-form err</th><th>Verified</th><th>Summary</th><th></th>
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
                <button
                  className="explore-btn"
                  disabled={!r.in_db || !r.experiment_id}
                  title={r.in_db ? "Open in the explorer" : "Genome not in this database"}
                  onClick={() => r.experiment_id && onOpen(r.genome_id, r.experiment_id, r.experiment_name || r.label)}
                >
                  Review
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
