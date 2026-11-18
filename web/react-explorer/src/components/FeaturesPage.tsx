/**
 * /features: every feature of interest pinned while exploring (file store, see
 * explaneat/analysis/feature_pins.py). Filter and search, edit status / note / tags / dossier
 * path (each edit appends a version), open the genome with the evidence panel restored to the
 * saved view, and copy the command that builds a feature dossier.
 */
import { useReadOnly } from "../api/appConfig";
import { Fragment, useCallback, useEffect, useMemo, useState } from "react";
import {
  dossierFileUrl,
  getDossier,
  getPin,
  inputText,
  listPins,
  scopeText,
  suggestedDossierPath,
  updatePin,
  PIN_STATUSES,
  VIEW_LABELS,
  type DossierInfo,
  type FeaturePin,
  type PinStatus,
} from "../api/features";

type Props = {
  onOpen: (genomeId: string, experimentId: string, experimentName: string, pinId: string) => void;
};

function fmtDate(iso: string): string {
  const d = new Date(iso);
  return isNaN(d.getTime()) ? iso : d.toLocaleString(undefined, { year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

function CopyButton({ text }: { text: string }) {
  const [done, setDone] = useState(false);
  return (
    <button
      className="op-btn"
      style={{ fontSize: "11px", padding: "2px 6px" }}
      onClick={() => {
        navigator.clipboard?.writeText(text).then(() => {
          setDone(true);
          window.setTimeout(() => setDone(false), 1500);
        });
      }}
      title={text}
    >
      {done ? "Copied" : "Copy"}
    </button>
  );
}

export function FeaturesPage({ onOpen }: Props) {
  const readOnly = useReadOnly();
  const [pins, setPins] = useState<FeaturePin[]>([]);
  const [dir, setDir] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [q, setQ] = useState("");
  const [status, setStatus] = useState("");
  const [tag, setTag] = useState("");
  const [label, setLabel] = useState("");
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState({ note: "", tags: "", dossier_path: "" });
  const [expanded, setExpanded] = useState<string | null>(null);
  const [history, setHistory] = useState<Record<string, FeaturePin[]>>({});
  const [dossiers, setDossiers] = useState<Record<string, DossierInfo>>({});

  const load = useCallback(() => {
    listPins()
      .then((r) => {
        setPins(r.pins);
        setDir(r.features_dir);
        setError(null);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
  }, []);
  useEffect(() => load(), [load]);

  const allTags = useMemo(() => Array.from(new Set(pins.flatMap((p) => p.tags))).sort(), [pins]);
  const allLabels = useMemo(
    () => Array.from(new Set(pins.map((p) => p.record_label).filter((x): x is string => !!x))).sort(),
    [pins],
  );
  const shown = useMemo(() => {
    const ql = q.trim().toLowerCase();
    return pins.filter((p) => {
      if (status && p.status !== status) return false;
      if (tag && !p.tags.includes(tag)) return false;
      if (label && p.record_label !== label) return false;
      if (!ql) return true;
      const hay = [p.note, p.record_label, p.experiment_name, p.genome_id, p.id, scopeText(p.scope), p.viz?.description,
        ...p.tags, ...p.inputs.map(inputText)].join(" ").toLowerCase();
      return hay.includes(ql);
    });
  }, [pins, q, status, tag, label]);

  const refreshHistory = (id: string) => {
    if (expanded !== id) return;
    getPin(id).then((full) => setHistory((h) => ({ ...h, [id]: full.history ?? [] }))).catch(() => {});
  };
  const replace = (pin: FeaturePin) => setPins((ps) => ps.map((p) => (p.id === pin.id ? pin : p)));

  const setPinStatus = async (p: FeaturePin, s: PinStatus) => {
    try {
      replace(await updatePin(p.id, { status: s }));
      refreshHistory(p.id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const startEdit = (p: FeaturePin) => {
    setEditing(p.id);
    setDraft({ note: p.note, tags: p.tags.join(", "), dossier_path: p.dossier_path ?? "" });
  };
  const saveEdit = async (p: FeaturePin) => {
    try {
      replace(
        await updatePin(p.id, {
          note: draft.note,
          tags: draft.tags.split(/[,\s]+/).map((t) => t.trim()).filter(Boolean),
          dossier_path: draft.dossier_path.trim() || null,
        }),
      );
      setEditing(null);
      refreshHistory(p.id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const toggleDetails = async (p: FeaturePin) => {
    if (expanded === p.id) {
      setExpanded(null);
      return;
    }
    setExpanded(p.id);
    try {
      const [full, dos] = await Promise.all([getPin(p.id), p.dossier_path ? getDossier(p.id) : Promise.resolve(null)]);
      setHistory((h) => ({ ...h, [p.id]: full.history ?? [] }));
      if (dos) setDossiers((d) => ({ ...d, [p.id]: dos }));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const counts = useMemo(() => {
    const c: Record<string, number> = {};
    for (const p of pins) c[p.status] = (c[p.status] || 0) + 1;
    return c;
  }, [pins]);

  return (
    <div className="experiment-list-container features-page">
      <header className="experiment-list-header">
        <h2>Features of interest</h2>
        <p className="hint">
          {pins.length} pinned feature{pins.length === 1 ? "" : "s"}
          {pins.length > 0 && ` (${PIN_STATUSES.filter((s) => counts[s]).map((s) => `${counts[s]} ${s}`).join(", ")})`} in{" "}
          <code>{dir}</code>. Pin a view from the evidence panel ("Pin as feature of interest") or an input from its
          inspector. Open one to reopen the genome with the same view; every edit keeps the earlier versions.
        </p>
        <div className="features-filters">
          <input type="search" placeholder="Search notes, inputs, labels, tags" value={q} onChange={(e) => setQ(e.target.value)} />
          <select value={status} onChange={(e) => setStatus(e.target.value)} aria-label="Status filter">
            <option value="">any status</option>
            {PIN_STATUSES.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
          <select value={label} onChange={(e) => setLabel(e.target.value)} aria-label="Model filter">
            <option value="">any model</option>
            {allLabels.map((l) => <option key={l} value={l}>{l}</option>)}
          </select>
          <select value={tag} onChange={(e) => setTag(e.target.value)} aria-label="Tag filter">
            <option value="">any tag</option>
            {allTags.map((t) => <option key={t} value={t}>{t}</option>)}
          </select>
        </div>
      </header>
      {loading && <p className="hint">Loading pins…</p>}
      {error && <div className="error-message">{error}</div>}
      {!loading && pins.length === 0 && <p className="hint">No features pinned yet.</p>}
      {shown.length > 0 && (
        <table className="features-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Input</th>
              <th>Scope / view</th>
              <th>Note</th>
              <th>Tags</th>
              <th>Status</th>
              <th>Date</th>
              <th>Dossier</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {shown.map((p) => {
              const cmd = p.dossier_command;
              const isEditing = editing === p.id;
              return (
                <Fragment key={p.id}>
                  <tr data-testid="feature-row" className={`pin-row-${p.status}`}>
                    <td>
                      <strong>{p.record_label || p.experiment_name || "—"}</strong>
                      <div className="hint" title={p.genome_id}><code>{p.genome_id.slice(0, 8)}</code></div>
                    </td>
                    <td>{p.inputs.length ? p.inputs.map((i, k) => <div key={k}>{inputText(i)}</div>) : "—"}</td>
                    <td>
                      {scopeText(p.scope)}
                      {p.viz && <div className="hint">{p.viz.description || VIEW_LABELS[p.viz.type] || p.viz.type}</div>}
                      {p.also_views.length > 0 && <div className="hint">+ {p.also_views.map((v) => VIEW_LABELS[v.type] || v.type).join(", ")}</div>}
                    </td>
                    <td className="features-note">
                      {isEditing ? (
                        <textarea rows={4} value={draft.note} onChange={(e) => setDraft({ ...draft, note: e.target.value })} />
                      ) : (
                        p.note || <span className="hint">—</span>
                      )}
                    </td>
                    <td>
                      {isEditing ? (
                        <input type="text" value={draft.tags} onChange={(e) => setDraft({ ...draft, tags: e.target.value })} />
                      ) : (
                        p.tags.map((t) => (
                          <button key={t} className="tag-chip" onClick={() => setTag(t)} title="Filter by this tag">{t}</button>
                        ))
                      )}
                    </td>
                    <td>
                      <select
                        className={`pin-status-select pin-status-${p.status}`}
                        value={p.status}
                        onChange={(e) => setPinStatus(p, e.target.value as PinStatus)}
                        aria-label="Status"
                        disabled={readOnly}
                      >
                        {PIN_STATUSES.map((s) => <option key={s} value={s}>{s}</option>)}
                      </select>
                    </td>
                    <td className="hint" title={`updated ${fmtDate(p.updated_at)}`}>
                      {fmtDate(p.created_at)}
                      {p.version > 1 && <div>v{p.version}</div>}
                    </td>
                    <td className="features-dossier">
                      {isEditing ? (
                        <input
                          type="text"
                          value={draft.dossier_path}
                          placeholder={suggestedDossierPath(p.record_label, p.inputs[0]?.raw)}
                          onChange={(e) => setDraft({ ...draft, dossier_path: e.target.value })}
                        />
                      ) : p.dossier_path ? (
                        <>
                          <button className="link-btn" onClick={() => toggleDetails(p)} title={p.dossier_path}>
                            {p.dossier_exists ? "dossier" : "dossier (not yet built)"}
                          </button>
                          <div className="hint"><code>{p.dossier_path}</code></div>
                        </>
                      ) : (
                        <span className="hint">—</span>
                      )}
                      {cmd && !isEditing && !p.dossier_exists && (
                        <div className="dossier-cmd">
                          <code>{cmd}</code> <CopyButton text={cmd} />
                        </div>
                      )}
                    </td>
                    <td className="features-actions">
                      <button
                        className="op-btn primary"
                        onClick={() => onOpen(p.genome_id, p.experiment_id ?? "", p.experiment_name ?? "", p.id)}
                        disabled={!p.experiment_id}
                        title={p.experiment_id ? "Open the genome with this view" : "No experiment recorded on this pin"}
                      >
                        Open
                      </button>
                      {isEditing ? (
                        <>
                          <button className="op-btn" onClick={() => saveEdit(p)}>Save</button>
                          <button className="op-btn" onClick={() => setEditing(null)}>Cancel</button>
                        </>
                      ) : readOnly ? null : (
                        <button className="op-btn" onClick={() => startEdit(p)}>Edit</button>
                      )}
                      <button className="link-btn" onClick={() => toggleDetails(p)}>{expanded === p.id ? "less" : "history"}</button>
                    </td>
                  </tr>
                  {expanded === p.id && (
                    <tr className="features-details">
                      <td colSpan={9}>
                        <div className="features-details-grid">
                          <div>
                            <strong>Versions</strong> <span className="hint">({p.id})</span>
                            <ol className="pin-history">
                              {(history[p.id] ?? []).map((v) => (
                                <li key={v.version}>
                                  v{v.version} · {fmtDate(v.updated_at)} · {v.edit || "—"} · <em>{v.status}</em>
                                  {v.note && <div className="hint">{v.note}</div>}
                                </li>
                              ))}
                            </ol>
                          </div>
                          {p.viz && (
                            <div>
                              <strong>Saved view</strong>
                              <pre className="pin-json">{JSON.stringify({ viz: p.viz, also_views: p.also_views }, null, 2)}</pre>
                            </div>
                          )}
                          {dossiers[p.id] && (
                            <div>
                              <strong>Dossier</strong> <span className="hint">{dossiers[p.id].resolved}</span>
                              {dossiers[p.id].exists ? (
                                <ul>
                                  {dossiers[p.id].files.map((f) => (
                                    <li key={f}><a href={dossierFileUrl(p.id, f)} target="_blank" rel="noreferrer">{f}</a></li>
                                  ))}
                                </ul>
                              ) : (
                                <p className="hint">Not generated yet. Run: <code>{dossiers[p.id].command}</code></p>
                              )}
                            </div>
                          )}
                        </div>
                      </td>
                    </tr>
                  )}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      )}
      {!loading && pins.length > 0 && shown.length === 0 && <p className="hint">No pins match the filters.</p>}
    </div>
  );
}
