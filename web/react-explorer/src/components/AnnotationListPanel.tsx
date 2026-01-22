/**
 * AnnotationListPanel - displays annotations with collapse/expand toggles.
 *
 * Features:
 * - Tree view showing annotation hierarchy
 * - Collapse/expand toggle per annotation
 * - Shows name, entry/exit count, leaf/composition badge
 * - Click to select annotation
 */

import { useCallback } from "react";
import type { AnnotationSummary } from "../api/client";
import type { CollapsedState } from "../hooks/useCollapsedView";

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
  collapsedState: CollapsedState;
  onToggleCollapse: (annotationId: string) => void;
  selectedAnnotationId: string | null;
  onSelectAnnotation: (annotationId: string | null) => void;
};

// =============================================================================
// Helper components
// =============================================================================

type AnnotationTreeItemProps = {
  annotation: AnnotationSummary;
  annotations: AnnotationSummary[];
  collapsedState: CollapsedState;
  onToggleCollapse: (annotationId: string) => void;
  selectedAnnotationId: string | null;
  onSelectAnnotation: (annotationId: string | null) => void;
  depth: number;
};

function AnnotationTreeItem({
  annotation,
  annotations,
  collapsedState,
  onToggleCollapse,
  selectedAnnotationId,
  onSelectAnnotation,
  depth,
}: AnnotationTreeItemProps) {
  const isCollapsed = collapsedState.get(annotation.id) ?? false;
  const isSelected = selectedAnnotationId === annotation.id;
  const children = annotations.filter(a => a.parent_annotation_id === annotation.id);
  const hasChildren = children.length > 0;

  const handleToggle = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    logDebug("Toggle collapse", { annotationId: annotation.id, currentState: isCollapsed });
    onToggleCollapse(annotation.id);
  }, [annotation.id, isCollapsed, onToggleCollapse]);

  const handleSelect = useCallback(() => {
    logDebug("Select annotation", { annotationId: annotation.id });
    onSelectAnnotation(isSelected ? null : annotation.id);
  }, [annotation.id, isSelected, onSelectAnnotation]);

  const displayName = annotation.name || `Annotation ${annotation.id.slice(0, 8)}`;

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
        <span className="annotation-name">{displayName}</span>
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
          collapsedState={collapsedState}
          onToggleCollapse={onToggleCollapse}
          selectedAnnotationId={selectedAnnotationId}
          onSelectAnnotation={onSelectAnnotation}
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
  collapsedState,
  onToggleCollapse,
  selectedAnnotationId,
  onSelectAnnotation,
}: AnnotationListPanelProps) {
  // Get root-level annotations (no parent)
  const rootAnnotations = annotations.filter(a => !a.parent_annotation_id);

  if (annotations.length === 0) {
    return (
      <section className="panel-section annotation-list-panel">
        <h4>Annotations (0)</h4>
        <p className="hint">No annotations yet. Select nodes and create an annotation.</p>
      </section>
    );
  }

  const collapsedCount = Array.from(collapsedState.values()).filter(v => v).length;

  return (
    <section className="panel-section annotation-list-panel">
      <h4>Annotations ({annotations.length})</h4>
      {collapsedCount > 0 && (
        <p className="hint">{collapsedCount} annotation(s) collapsed</p>
      )}
      <div className="annotation-tree">
        {rootAnnotations.map(annotation => (
          <AnnotationTreeItem
            key={annotation.id}
            annotation={annotation}
            annotations={annotations}
            collapsedState={collapsedState}
            onToggleCollapse={onToggleCollapse}
            selectedAnnotationId={selectedAnnotationId}
            onSelectAnnotation={onSelectAnnotation}
            depth={0}
          />
        ))}
      </div>
    </section>
  );
}
