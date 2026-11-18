/**
 * "Pin as feature of interest": a small form pre-filled with what is on screen (genome, scope,
 * input(s), the visualisation and its exact parameters). Saves a pin to /api/features.
 */
import { useState } from "react";
import {
  createPin,
  inputText,
  scopeText,
  suggestedDossierPath,
  VIEW_LABELS,
  PIN_STATUSES,
  type FeaturePin,
  type PinDraft,
  type PinStatus,
} from "../api/features";

export type PinModalDraft = Omit<PinDraft, "note" | "tags" | "status" | "dossier_path"> & {
  note?: string;
  tags?: string[];
};

export function PinFeatureModal({
  draft,
  otherViewText,
  onClose,
  onSaved,
}: {
  draft: PinModalDraft;
  /** When set, offers to keep the other evidence slot's view on the pin too. */
  otherViewText?: string | null;
  onClose: () => void;
  onSaved: (pin: FeaturePin) => void;
}) {
  const [note, setNote] = useState(draft.note ?? "");
  const [tags, setTags] = useState((draft.tags ?? []).join(", "));
  const [status, setStatus] = useState<PinStatus>("open");
  const [keepOther, setKeepOther] = useState(true);
  const [dossier, setDossier] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const firstInput = draft.inputs[0];
  const suggestion = suggestedDossierPath(draft.record_label, firstInput?.raw ?? firstInput?.node_id);

  const save = async () => {
    setSaving(true);
    setError(null);
    try {
      const pin = await createPin({
        ...draft,
        also_views: keepOther ? draft.also_views : [],
        note,
        tags: tags.split(/[,\s]+/).map((t) => t.trim()).filter(Boolean),
        status,
        dossier_path: dossier.trim() || null,
      });
      onSaved(pin);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setSaving(false);
    }
  };

  return (
    <div className="pin-modal-backdrop" onClick={onClose}>
      <div className="pin-modal" onClick={(e) => e.stopPropagation()} role="dialog" aria-label="Pin as feature of interest">
        <div className="pin-modal-head">
          <h3>Pin as feature of interest</h3>
          <button className="pin-x" onClick={onClose} aria-label="Close">x</button>
        </div>
        <table className="pin-context">
          <tbody>
            <tr><td>Model</td><td>{draft.record_label || draft.experiment_name || "—"} <code>{draft.genome_id.slice(0, 8)}</code></td></tr>
            <tr><td>Scope</td><td>{scopeText(draft.scope)}</td></tr>
            <tr><td>Input{draft.inputs.length === 1 ? "" : "s"}</td><td>{draft.inputs.length ? draft.inputs.map(inputText).join(", ") : "—"}</td></tr>
            {draft.viz && (
              <tr><td>View</td><td>{draft.viz.description || VIEW_LABELS[draft.viz.type] || draft.viz.type}</td></tr>
            )}
          </tbody>
        </table>
        {otherViewText && draft.also_views.length > 0 && (
          <label className="pin-check">
            <input type="checkbox" checked={keepOther} onChange={(e) => setKeepOther(e.target.checked)} /> also keep the other
            panel's view: {otherViewText}
          </label>
        )}
        <label className="pin-field">
          Note
          <textarea
            autoFocus
            rows={4}
            value={note}
            onChange={(e) => setNote(e.target.value)}
            placeholder="What is interesting here, and what to work out"
          />
        </label>
        <label className="pin-field">
          Tags <span className="hint">(comma separated)</span>
          <input type="text" value={tags} onChange={(e) => setTags(e.target.value)} />
        </label>
        <div className="pin-row">
          <label className="pin-field">
            Status
            <select value={status} onChange={(e) => setStatus(e.target.value as PinStatus)}>
              {PIN_STATUSES.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          </label>
          <label className="pin-field" style={{ flex: 1 }}>
            Dossier path <span className="hint">(optional)</span>
            <input type="text" value={dossier} onChange={(e) => setDossier(e.target.value)} placeholder={suggestion || "experiments/feature_dossiers/..."} />
          </label>
        </div>
        {error && <div className="error-message">{error}</div>}
        <div className="pin-actions">
          <button className="op-btn" onClick={onClose} disabled={saving}>Cancel</button>
          <button className="op-btn primary" onClick={save} disabled={saving}>{saving ? "Saving..." : "Save pin"}</button>
        </div>
      </div>
    </div>
  );
}
