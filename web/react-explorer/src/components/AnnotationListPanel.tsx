/**
 * AnnotationListPanel - displays annotations with collapse/expand toggles.
 *
 * Features:
 * - Tree view showing annotation hierarchy
 * - Collapse/expand toggle per annotation
 * - Shows name, entry/exit count, leaf/composition badge
 * - Click to select annotation
 * - Double-click annotation name to rename (sets display_name)
 */

import { useCallback, useState } from "react";
import { addOperation, type AnnotationSummary } from "../api/client";

// =============================================================================
// Logging utilities
// =============================================================================

const LOG_PREFIX = "[AnnotationListPanel]";

function logDebug(message: string, data?: unknown) {
  console.log(`${LOG_PREFIX} ${message}`, data !== undefined ? data : "");
}

// =============================================================================
// Types
// =============================================================================

type AnnotationListPanelProps = {
  annotations: AnnotationSummary[];
  collapsedAnnotations: Set<string>;
  onToggleCollapse: (annotationId: string) => void;
  selectedAnnotationId: string | null;
  onSelectAnnotation: (annotationId: string | null) => void;
  genomeId: string;
  onOperationChange: () => void;
};

// =============================================================================
// Helper components
// =============================================================================

type AnnotationTreeItemProps = {
  annotation: AnnotationSummary;
  annotations: AnnotationSummary[];
  collapsedAnnotations: Set<string>;
  onToggleCollapse: (annotationId: string) => void;
  selectedAnnotationId: string | null;
  onSelectAnnotation: (annotationId: string | null) => void;
  genomeId: string;
  onOperationChange: () => void;
  depth: number;
};

function AnnotationTreeItem({
  annotation,
  annotations,
  collapsedAnnotations,
  onToggleCollapse,
  selectedAnnotationId,
  onSelectAnnotation,
  genomeId,
  onOperationChange,
  depth,
}: AnnotationTreeItemProps) {
  const isCollapsed = annotation.name ? collapsedAnnotations.has(annotation.name) : false;
  const isSelected = selectedAnnotationId === annotation.id;
  const children = annotations.filter(a => a.parent_annotation_id === annotation.id);
  const hasChildren = children.length > 0;

  const [editing, setEditing] = useState(false);
  const [editValue, setEditValue] = useState("");

  const handleToggle = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    logDebug("Toggle collapse", { annotationId: annotation.id, currentState: isCollapsed });
    onToggleCollapse(annotation.id);
  }, [annotation.id, isCollapsed, onToggleCollapse]);

  const handleSelect = useCallback(() => {
    logDebug("Select annotation", { annotationId: annotation.id });
    onSelectAnnotation(isSelected ? null : annotation.id);
  }, [annotation.id, isSelected, onSelectAnnotation]);

  const handleDoubleClick = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    setEditValue(annotation.display_name || annotation.name || "");
    setEditing(true);
  }, [annotation.display_name, annotation.name]);

  const submitRename = useCallback(async (value: string) => {
    setEditing(false);
    const trimmed = value.trim();
    const currentDisplay = annotation.display_name || annotation.name || "";
    if (trimmed === currentDisplay) return;

    // If user cleared the field back to the canonical name (or empty), set null to clear display_name
    const newDisplayName = (!trimmed || trimmed === annotation.name) ? null : trimmed;

    if (!annotation.name) return;
    try {
      await addOperation(genomeId, {
        type: "rename_annotation",
        params: { annotation_id: annotation.name, display_name: newDisplayName },
      });
      onOperationChange();
    } catch (err) {
      logDebug("Rename annotation failed", err);
    }
  }, [annotation.display_name, annotation.name, genomeId, onOperationChange]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      submitRename(editValue);
    } else if (e.key === "Escape") {
      setEditing(false);
    }
  }, [editValue, submitRename]);

  const handleBlur = useCallback(() => {
    submitRename(editValue);
  }, [editValue, submitRename]);

  const displayName = annotation.display_name || annotation.name || `Annotation ${annotation.id.slice(0, 8)}`;

  return (
    <div className="annotation-tree-item" style={{ marginLeft: depth * 16 }}>
      <div
        className={`annotation-item ${isSelected ? "selected" : ""} ${isCollapsed ? "collapsed" : ""}`}
        onClick={handleSelect}
      >
        <button
          className="collapse-toggle"
          onClick={handleToggle}
          title={isCollapsed ? "Expand annotation" : "Collapse annotation"}
        >
          {isCollapsed ? "+" : "-"}
        </button>
        {editing ? (
          <input
            className="annotation-rename-input"
            value={editValue}
            onChange={e => setEditValue(e.target.value)}
            onKeyDown={handleKeyDown}
            onBlur={handleBlur}
            autoFocus
            onClick={e => e.stopPropagation()}
          />
        ) : (
          <span
            className="annotation-name"
            onDoubleClick={handleDoubleClick}
            title="Double-click to rename"
          >
            {displayName}
          </span>
        )}
        <span className="annotation-meta">
          <span className="node-count" title="Entry nodes">
            {annotation.entry_nodes.length}
          </span>
          <span className="arrow">-&gt;</span>
          <span className="node-count" title="Exit nodes">
            {annotation.exit_nodes.length}
          </span>
        </span>
        {hasChildren ? (
          <span className="badge composition" title="Has child annotations">comp</span>
        ) : (
          <span className="badge leaf" title="Leaf annotation">leaf</span>
        )}
      </div>
      {/* Render children if parent is not collapsed */}
      {!isCollapsed && children.map(child => (
        <AnnotationTreeItem
          key={child.id}
          annotation={child}
          annotations={annotations}
          collapsedAnnotations={collapsedAnnotations}
          onToggleCollapse={onToggleCollapse}
          selectedAnnotationId={selectedAnnotationId}
          onSelectAnnotation={onSelectAnnotation}
          genomeId={genomeId}
          onOperationChange={onOperationChange}
          depth={depth + 1}
        />
      ))}
    </div>
  );
}

// =============================================================================
// Main component
// =============================================================================

export function AnnotationListPanel({
  annotations,
  collapsedAnnotations,
  onToggleCollapse,
  selectedAnnotationId,
  onSelectAnnotation,
  genomeId,
  onOperationChange,
}: AnnotationListPanelProps) {
  const [panelCollapsed, setPanelCollapsed] = useState(false);

  // Get root-level annotations (no parent)
  const rootAnnotations = annotations.filter(a => !a.parent_annotation_id);
  const collapsedCount = collapsedAnnotations.size;

  return (
    <section className="panel-section annotation-list-panel">
      <div
        className="panel-header"
        onClick={() => setPanelCollapsed(!panelCollapsed)}
        style={{ cursor: "pointer", display: "flex", alignItems: "center", gap: "6px" }}
      >
        <span className="collapse-toggle">{panelCollapsed ? "\u25b6" : "\u25bc"}</span>
        <h4 style={{ margin: 0 }}>Annotations ({annotations.length})</h4>
      </div>
      {!panelCollapsed && (
        <>
          {annotations.length === 0 ? (
            <p className="hint">No annotations yet. Select nodes and create an annotation.</p>
          ) : (
            <>
              {collapsedCount > 0 && (
                <p className="hint">{collapsedCount} annotation(s) collapsed</p>
              )}
              <div className="annotation-tree">
                {rootAnnotations.map(annotation => (
                  <AnnotationTreeItem
                    key={annotation.id}
                    annotation={annotation}
                    annotations={annotations}
                    collapsedAnnotations={collapsedAnnotations}
                    onToggleCollapse={onToggleCollapse}
                    selectedAnnotationId={selectedAnnotationId}
                    onSelectAnnotation={onSelectAnnotation}
                    genomeId={genomeId}
                    onOperationChange={onOperationChange}
                    depth={0}
                  />
                ))}
              </div>
            </>
          )}
        </>
      )}
    </section>
  );
}
