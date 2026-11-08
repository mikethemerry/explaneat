/**
 * Export an explained genome (or every explained genome of an experiment) as an export
 * pack: JSON (canonical record), LaTeX appendix, TikZ figures, PDF, DOCX. Runs the same
 * explaneat.export code as `explaneat export`; formats this machine cannot produce are
 * shown as unavailable with what to install. Files are written under
 * $EXPLANEAT_HOME/exports and stay on this machine.
 */
import { useEffect, useRef, useState } from "react";
import {
  ALL_FORMATS,
  getExportCapabilities,
  getExportJob,
  startExport,
  type Capabilities,
  type ExportFormat,
  type ExportJob,
} from "../api/exports";

const LABELS: Record<ExportFormat, string> = {
  json: "JSON (canonical record)",
  tex: "LaTeX appendix",
  tikz: "TikZ figures",
  pdf: "PDF (standalone)",
  docx: "Word (DOCX)",
};

type Target = { kind: "genome" | "experiment"; id: string; name?: string };

export function ExportPanel({ target }: { target: Target }) {
  const [caps, setCaps] = useState<Capabilities | null>(null);
  const [formats, setFormats] = useState<ExportFormat[]>(["json", "tex", "tikz", "pdf", "docx"]);
  const [reviewMarks, setReviewMarks] = useState(true);
  const [job, setJob] = useState<ExportJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<number | null>(null);

  useEffect(() => {
    getExportCapabilities()
      .then((c) => {
        setCaps(c);
        setFormats((f) => f.filter((x) => c[x]?.available));
      })
      .catch((e) => setError(String(e instanceof Error ? e.message : e)));
    return () => {
      if (timer.current) window.clearInterval(timer.current);
    };
  }, []);

  const toggle = (f: ExportFormat) =>
    setFormats((cur) => (cur.includes(f) ? cur.filter((x) => x !== f) : [...cur, f]));

  const run = async () => {
    setError(null);
    setJob(null);
    try {
      const { job_id } = await startExport(target, formats, reviewMarks);
      const poll = async () => {
        try {
          const j = await getExportJob(job_id);
          setJob(j);
          if (j.status !== "running" && timer.current) {
            window.clearInterval(timer.current);
            timer.current = null;
          }
        } catch (e) {
          setError(String(e instanceof Error ? e.message : e));
        }
      };
      await poll();
      timer.current = window.setInterval(poll, 1000);
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
    }
  };

  const running = job?.status === "running";
  const m = job?.manifest;
  return (
    <div className="export-panel">
      <div className="export-options">
        {ALL_FORMATS.map((f) => {
          const c = caps?.[f];
          const unavailable = c && !c.available;
          return (
            <label key={f} className={unavailable ? "hint" : ""} title={unavailable ? c?.reason || "" : ""}>
              <input
                type="checkbox"
                checked={formats.includes(f) || f === "json"}
                disabled={f === "json" || !!unavailable || running}
                onChange={() => toggle(f)}
              />{" "}
              {LABELS[f]}
              {unavailable && <span className="hint"> ({c?.reason})</span>}
            </label>
          );
        })}
        <label title="Wrap each appendix in the thesis's drafted environment (review markers)">
          <input type="checkbox" checked={reviewMarks} disabled={running} onChange={(e) => setReviewMarks(e.target.checked)} />{" "}
          Review marks (thesis drafted environment)
        </label>
      </div>
      <button className="explore-btn" onClick={run} disabled={running || !caps}>
        {running ? "Exporting..." : target.kind === "genome" ? "Export" : "Export all explained genomes"}
      </button>
      {error && <div className="error-message">{error}</div>}
      {job && (
        <div className="export-status">
          <div>
            Status: <strong>{job.status}</strong>
            {job.progress && ` (${job.progress.done} / ${job.progress.total} genomes)`}
            {running && job.log.length > 0 && <span className="hint"> {job.log[job.log.length - 1]}</span>}
          </div>
          {job.error && <div className="error-message">{job.error}</div>}
          {m && (
            <>
              <div className="hint">Written to {m.path}</div>
              {m.kind === "genome" && (
                <ul className="export-formats">
                  {Object.entries(m.formats as Record<string, string>).map(([f, s]) => (
                    <li key={f}>
                      {f}: {s}
                    </li>
                  ))}
                </ul>
              )}
              {m.failures && m.failures.length > 0 && (
                <div className="error-message">
                  {m.failures.length} genome(s) failed: {m.failures.map((x) => x.genome_id).join(", ")}
                </div>
              )}
              <ul className="export-files">
                {m.files.map((f) => (
                  <li key={f.name}>
                    <a href={f.url} download>
                      {f.name}
                    </a>{" "}
                    <span className="hint">({Math.max(1, Math.round(f.size / 1024))} KB)</span>
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export function ExportModal({ target, onClose }: { target: Target; onClose: () => void }) {
  return (
    <div className="wizard-overlay" onClick={onClose}>
      <div className="wizard-content export-modal" onClick={(e) => e.stopPropagation()}>
        <div className="dataset-setup-header">
          <h3>Export {target.kind === "genome" ? "explanation" : "experiment"}</h3>
          <span className="hint">{target.name || target.id}</span>
          <button className="op-btn" onClick={onClose}>
            Close
          </button>
        </div>
        <ExportPanel target={target} />
      </div>
    </div>
  );
}
