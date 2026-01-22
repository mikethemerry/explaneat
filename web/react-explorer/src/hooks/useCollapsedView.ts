/**
 * Hook for computing collapsed view of the model when annotations are collapsed.
 *
 * When an annotation is collapsed:
 * 1. ALL subgraph nodes (entry, exit, intermediate) are hidden
 * 2. A new annotation node is created with ID `A_{name}` or `A_{id[:8]}`
 * 3. Connections INTO subgraph route to the annotation node
 * 4. Connections FROM subgraph route from the annotation node
 * 5. Internal connections are hidden
 * 6. Duplicate connections are deduplicated
 */

import { useMemo } from "react";
import type { ModelState, ApiNode, ApiConnection, AnnotationSummary } from "../api/client";

// =============================================================================
// Types
// =============================================================================

export type CollapsedState = Map<string, boolean>; // annotation_id -> is_collapsed

export type CollapsedModelState = ModelState & {
  // Additional metadata about collapsed view
  collapsedAnnotations: AnnotationSummary[];
  annotationNodes: string[]; // IDs of synthetic annotation nodes
};

// =============================================================================
// Helper functions
// =============================================================================

const LOG_PREFIX = "[useCollapsedView]";

function logDebug(message: string, data?: unknown) {
  console.log(`${LOG_PREFIX} ${message}`, data !== undefined ? data : "");
}

/**
 * Get the annotation node ID for a given annotation.
 * Uses the name if available, otherwise uses shortened ID.
 */
export function getAnnotationNodeId(annotation: AnnotationSummary): string {
  const baseName = annotation.name || annotation.id.slice(0, 8);
  return `A_${baseName}`;
}

/**
 * Get all annotations that should be collapsed, including children of collapsed parents.
 * When a parent is collapsed, all descendants are implicitly collapsed.
 */
function getEffectivelyCollapsed(
  annotations: AnnotationSummary[],
  collapsedState: CollapsedState
): Set<string> {
  const effectivelyCollapsed = new Set<string>();
  const annotationMap = new Map(annotations.map(a => [a.id, a]));

  // Build children lookup
  const childrenOf = new Map<string, string[]>();
  for (const ann of annotations) {
    if (!childrenOf.has(ann.id)) {
      childrenOf.set(ann.id, []);
    }
    if (ann.parent_annotation_id) {
      const siblings = childrenOf.get(ann.parent_annotation_id) || [];
      siblings.push(ann.id);
      childrenOf.set(ann.parent_annotation_id, siblings);
    }
  }

  // Process in order: parents first
  // An annotation is effectively collapsed if:
  // 1. It is explicitly collapsed, OR
  // 2. Any of its ancestors is collapsed
  function isAncestorCollapsed(annId: string): boolean {
    const ann = annotationMap.get(annId);
    if (!ann) return false;
    if (!ann.parent_annotation_id) return false;
    if (collapsedState.get(ann.parent_annotation_id)) return true;
    return isAncestorCollapsed(ann.parent_annotation_id);
  }

  for (const ann of annotations) {
    if (collapsedState.get(ann.id) || isAncestorCollapsed(ann.id)) {
      effectivelyCollapsed.add(ann.id);
    }
  }

  return effectivelyCollapsed;
}

/**
 * Reroute connections for collapsed annotations.
 */
function rerouteConnections(
  connections: ApiConnection[],
  subgraphSet: Set<string>,
  annotationNodeId: string
): ApiConnection[] {
  const result: ApiConnection[] = [];
  const seen = new Set<string>();

  for (const conn of connections) {
    const fromIn = subgraphSet.has(conn.from);
    const toIn = subgraphSet.has(conn.to);

    // Internal connection - skip
    if (fromIn && toIn) continue;

    let newConn = conn;
    if (!fromIn && toIn) {
      // External -> Subgraph: route to annotation node
      newConn = { ...conn, to: annotationNodeId };
    } else if (fromIn && !toIn) {
      // Subgraph -> External: route from annotation node
      newConn = { ...conn, from: annotationNodeId };
    }

    // Deduplicate
    const key = `${newConn.from}->${newConn.to}`;
    if (!seen.has(key)) {
      seen.add(key);
      result.push(newConn);
    }
  }

  return result;
}

// =============================================================================
// Hook
// =============================================================================

/**
 * Compute a collapsed view of the model based on which annotations are collapsed.
 *
 * @param model - The full model state
 * @param annotations - List of annotations for the genome
 * @param collapsedState - Map of annotation IDs to their collapsed state
 * @returns The model with collapsed annotations rendered as single nodes
 */
export function useCollapsedView(
  model: ModelState | null,
  annotations: AnnotationSummary[],
  collapsedState: CollapsedState
): CollapsedModelState | null {
  return useMemo(() => {
    if (!model) return null;

    // No annotations or nothing collapsed - return original model
    const hasCollapsed = Array.from(collapsedState.values()).some(v => v);
    if (annotations.length === 0 || !hasCollapsed) {
      return {
        ...model,
        collapsedAnnotations: [],
        annotationNodes: [],
      };
    }

    logDebug("Computing collapsed view", {
      annotations: annotations.length,
      collapsed: Array.from(collapsedState.entries()).filter(([, v]) => v).map(([k]) => k),
    });

    // Get effectively collapsed annotations (including children of collapsed parents)
    const effectivelyCollapsed = getEffectivelyCollapsed(annotations, collapsedState);

    // Get annotations to display (only top-level collapsed ones)
    // When a parent is collapsed, we show the parent node, not the children
    const displayedCollapsedAnnotations = annotations.filter(ann => {
      if (!effectivelyCollapsed.has(ann.id)) return false;
      // Only display if parent is not also collapsed
      if (ann.parent_annotation_id && effectivelyCollapsed.has(ann.parent_annotation_id)) {
        return false;
      }
      return true;
    });

    logDebug("Displayed collapsed annotations", displayedCollapsedAnnotations.map(a => a.name || a.id));

    // Collect all hidden nodes (from all effectively collapsed annotations)
    const hiddenNodes = new Set<string>();
    for (const ann of annotations) {
      if (effectivelyCollapsed.has(ann.id)) {
        for (const nodeId of ann.subgraph_nodes) {
          hiddenNodes.add(nodeId);
        }
      }
    }

    // Build annotation nodes
    const annotationNodes: ApiNode[] = displayedCollapsedAnnotations.map(ann => ({
      id: getAnnotationNodeId(ann),
      type: "annotation" as any, // Special type for annotation nodes
      bias: null,
      activation: null,
      response: null,
      aggregation: null,
    }));

    // Filter out hidden nodes
    const filteredNodes = model.nodes.filter(n => !hiddenNodes.has(n.id));

    // Reroute connections for each displayed collapsed annotation
    let connections = model.connections;
    for (const ann of displayedCollapsedAnnotations) {
      const subgraphSet = new Set(ann.subgraph_nodes);
      const annotationNodeId = getAnnotationNodeId(ann);
      connections = rerouteConnections(connections, subgraphSet, annotationNodeId);
    }

    // Filter connections that reference hidden nodes (except annotation node connections)
    const annotationNodeIds = new Set(annotationNodes.map(n => n.id));
    connections = connections.filter(conn => {
      const fromHidden = hiddenNodes.has(conn.from) && !annotationNodeIds.has(conn.from);
      const toHidden = hiddenNodes.has(conn.to) && !annotationNodeIds.has(conn.to);
      return !fromHidden && !toHidden;
    });

    logDebug("Collapsed view result", {
      originalNodes: model.nodes.length,
      filteredNodes: filteredNodes.length,
      annotationNodes: annotationNodes.length,
      originalConnections: model.connections.length,
      resultConnections: connections.length,
    });

    return {
      nodes: [...filteredNodes, ...annotationNodes],
      connections,
      metadata: model.metadata,
      collapsedAnnotations: displayedCollapsedAnnotations,
      annotationNodes: annotationNodes.map(n => n.id),
    };
  }, [model, annotations, collapsedState]);
}
