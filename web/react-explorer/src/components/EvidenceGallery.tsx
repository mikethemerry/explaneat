import { useEffect, useState, useCallback } from "react";
import {
  listEvidence,
  updateNarrative,
  type EvidenceEntry,
} from "../api/client";

type EvidenceGalleryProps = {
  genomeId: string;
  annotationId: string;
};

export function EvidenceGallery({ genomeId, annotationId }: EvidenceGalleryProps) {
  const [entries, setEntries] = useState<EvidenceEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [editText, setEditText] = useState("");

  const loadEvidence = useCallback(() => {
    setLoading(true);
    listEvidence(genomeId, annotationId)
      .then((data) => {
        setEntries(data.entries);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Failed to load evidence:", err);
        setLoading(false);
      });
  }, [genomeId, annotationId]);

  useEffect(() => {
    loadEvidence();
  }, [loadEvidence]);

  const handleStartEdit = useCallback((index: number, currentText: string) => {
    setEditingIndex(index);
    setEditText(currentText);
  }, []);

  const handleSaveNarrative = useCallback(
    async (index: number) => {
      try {
        await updateNarrative(genomeId, annotationId, index, editText);
        setEditingIndex(null);
        loadEvidence();
      } catch (err) {
        console.error("Failed to update narrative:", err);
      }
    },
    [genomeId, annotationId, editText, loadEvidence],
  );

  if (loading) {
    return <div className="evidence-gallery loading">Loading evidence...</div>;
  }

  if (entries.length === 0) {
    return (
      <div className="evidence-gallery empty">
        No evidence snapshots yet. Use the snapshot button to capture visualizations.
      </div>
    );
  }

  return (
    <div className="evidence-gallery">
      <h5 className="gallery-title">Evidence ({entries.length})</h5>
      <div className="gallery-grid">
        {entries.map((entry, i) => (
          <div key={i} className="gallery-item">
            {entry.svg_data && (
              <div
                className="gallery-thumbnail"
                dangerouslySetInnerHTML={{
                  __html: atob(entry.svg_data),
                }}
              />
            )}
            <div className="gallery-info">
              <span className="gallery-category">{entry.category}</span>
              {entry.timestamp && (
                <span className="gallery-time">
                  {new Date(entry.timestamp).toLocaleDateString()}
                </span>
              )}
            </div>
            {editingIndex === i ? (
              <div className="gallery-edit">
                <textarea
                  className="gallery-textarea"
                  value={editText}
                  onChange={(e) => setEditText(e.target.value)}
                  rows={3}
                />
                <div className="gallery-edit-actions">
                  <button
                    className="op-btn primary"
                    onClick={() => handleSaveNarrative(i)}
                  >
                    Save
                  </button>
                  <button
                    className="op-btn secondary"
                    onClick={() => setEditingIndex(null)}
                  >
                    Cancel
                  </button>
                </div>
              </div>
            ) : (
              <div
                className="gallery-narrative"
                onClick={() => handleStartEdit(i, entry.narrative)}
              >
                {entry.narrative || "Click to add narrative..."}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
