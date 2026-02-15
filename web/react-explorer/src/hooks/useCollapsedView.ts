/**
 * Hook for computing collapsed view of the model when annotations are collapsed.
 *
 * When an annotation is collapsed:
 * 1. INTERNAL nodes (intermediate + exit) are hidden; entry nodes stay visible
 * 2. A new annotation node is created with ID `A_{name}` or `A_{id[:8]}`
 * 3. Entry nodes that connected to internal nodes get edges to the annotation node
 * 4. The annotation node gets edges to where exit nodes connected externally
 * 5. Entry-to-external edges are preserved as-is (the key bug fix)
 * 6. Internal connections are hidden
 * 7. Duplicate connections are deduplicated
 *
 * When a PARENT annotation is collapsed, all descendant nodes (including child
 * entry nodes) are hidden — only the parent's own entry nodes stay visible.
 *
 * See docs/annotation_collapsing_model.md for the mathematical formalization.
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

  // Build parent lookup by inverting children_ids (the source of truth).
  // Do NOT use parent_annotation_id — it's a computed back-pointer that can
  // be incorrect when multiple annotations share the same name.
  const parentOf = new Map<string, string>();
  for (const ann of annotations) {
    for (const childId of ann.children_ids) {
      parentOf.set(childId, ann.id);
    }
  }

  // An annotation is effectively collapsed if:
  // 1. It is explicitly collapsed, OR
  // 2. Any of its ancestors is collapsed
  function isAncestorCollapsed(annId: string): boolean {
    const parentId = parentOf.get(annId);
    if (!parentId) return false;
    if (collapsedState.get(parentId)) return true;
    return isAncestorCollapsed(parentId);
  }

  for (const ann of annotations) {
    if (collapsedState.get(ann.id) || isAncestorCollapsed(ann.id)) {
      effectivelyCollapsed.add(ann.id);
    }
  }

  return effectivelyCollapsed;
}

/**
 * Get all descendant annotation IDs for a given annotation.
 */
function getDescendantAnnotationIds(
  annotationId: string,
  annotations: AnnotationSummary[]
): Set<string> {
  const descendants = new Set<string>();

  // Build children lookup from children_ids (source of truth)
  const annotationMap = new Map(annotations.map(a => [a.id, a]));

  function collectDescendants(id: string) {
    const ann = annotationMap.get(id);
    if (!ann) return;
    for (const childId of ann.children_ids) {
      descendants.add(childId);
      collectDescendants(childId);
    }
  }

  collectDescendants(annotationId);
  return descendants;
}

/**
 * Get all subgraph nodes for an annotation including all descendant annotations.
 * For compositional annotations, this includes nodes from child annotations.
 */
function getFullSubgraphNodes(
  annotation: AnnotationSummary,
  annotations: AnnotationSummary[]
): Set<string> {
  const allNodes = new Set(annotation.subgraph_nodes);

  // Get all descendant annotations
  const descendantIds = getDescendantAnnotationIds(annotation.id, annotations);
  const annotationMap = new Map(annotations.map(a => [a.id, a]));

  // Add nodes from all descendants
  for (const descId of descendantIds) {
    const descAnn = annotationMap.get(descId);
    if (descAnn) {
      for (const nodeId of descAnn.subgraph_nodes) {
        allNodes.add(nodeId);
      }
    }
  }

  return allNodes;
}

/**
 * Get the internal nodes for an annotation (V_internal = V_A \ V_entry).
 * These are the nodes that get hidden during collapse.
 * For compositional annotations, includes internal nodes from all descendants.
 */
function getInternalNodes(
  annotation: AnnotationSummary,
  annotations: AnnotationSummary[]
): Set<string> {
  const allSubgraphNodes = getFullSubgraphNodes(annotation, annotations);
  const entryNodes = new Set(annotation.entry_nodes);

  // Internal = all subgraph nodes minus this annotation's entry nodes
  const internalNodes = new Set<string>();
  for (const nodeId of allSubgraphNodes) {
    if (!entryNodes.has(nodeId)) {
      internalNodes.add(nodeId);
    }
  }
  return internalNodes;
}

/**
 * Reroute connections for a collapsed annotation.
 *
 * The collapse operation:
 * - Entry nodes stay visible (they are the annotation's interface)
 * - Internal nodes (intermediate + exit) are replaced by annotation node a_A
 * - entry -> internal becomes entry -> a_A
 * - internal -> external becomes a_A -> external (from exit nodes)
 * - entry -> external is PRESERVED as-is
 * - internal -> internal is REMOVED
 */
function rerouteConnections(
  connections: ApiConnection[],
  internalNodes: Set<string>,
  entryNodes: Set<string>,
  annotationNodeId: string
): ApiConnection[] {
  const result: ApiConnection[] = [];
  const seen = new Set<string>();

  for (const conn of connections) {
    const fromInternal = internalNodes.has(conn.from);
    const toInternal = internalNodes.has(conn.to);
    const fromEntry = entryNodes.has(conn.from);

    // Both endpoints are internal — skip (hidden internal edge)
    if (fromInternal && toInternal) continue;

    // Entry -> internal: reroute to entry -> a_A
    if (fromEntry && toInternal) {
      const newConn = { ...conn, to: annotationNodeId };
      const key = `${newConn.from}->${newConn.to}`;
      if (!seen.has(key)) {
        seen.add(key);
        result.push(newConn);
      }
      continue;
    }

    // Internal -> external (exit node outputs): reroute to a_A -> external
    if (fromInternal && !toInternal) {
      const newConn = { ...conn, from: annotationNodeId };
      const key = `${newConn.from}->${newConn.to}`;
      if (!seen.has(key)) {
        seen.add(key);
        result.push(newConn);
      }
      continue;
    }

    // External -> internal: reroute to external -> a_A
    // (This shouldn't happen if preconditions hold, but handle gracefully)
    if (!fromInternal && !fromEntry && toInternal) {
      const newConn = { ...conn, to: annotationNodeId };
      const key = `${newConn.from}->${newConn.to}`;
      if (!seen.has(key)) {
        seen.add(key);
        result.push(newConn);
      }
      continue;
    }

    // All other edges: preserve as-is (includes entry -> external)
    const key = `${conn.from}->${conn.to}`;
    if (!seen.has(key)) {
      seen.add(key);
      result.push(conn);
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

    // Build canonical parent lookup from children_ids (source of truth)
    const parentOf = new Map<string, string>();
    for (const ann of annotations) {
      for (const childId of ann.children_ids) {
        parentOf.set(childId, ann.id);
      }
    }

    // Get annotations to display (only top-level collapsed ones)
    // When a parent is collapsed, we show the parent node, not the children
    const displayedCollapsedAnnotations = annotations.filter(ann => {
      if (!effectivelyCollapsed.has(ann.id)) return false;
      // Only display if parent is not also collapsed
      const canonicalParent = parentOf.get(ann.id);
      if (canonicalParent && effectivelyCollapsed.has(canonicalParent)) {
        return false;
      }
      return true;
    });

    logDebug("Displayed collapsed annotations", displayedCollapsedAnnotations.map(a => a.name || a.id));

    // Collect hidden nodes: only INTERNAL nodes (V_A \ V_entry) are hidden
    // For displayed collapsed annotations, use their own entry nodes
    // For child annotations of a collapsed parent, ALL nodes are hidden
    // (child entries are intermediate nodes of the parent)
    const hiddenNodes = new Set<string>();
    const preservedEntryNodes = new Set<string>();

    for (const ann of displayedCollapsedAnnotations) {
      // Get internal nodes for this top-level collapsed annotation
      const internal = getInternalNodes(ann, annotations);
      for (const nodeId of internal) {
        hiddenNodes.add(nodeId);
      }
      // Mark this annotation's entry nodes as preserved
      for (const nodeId of ann.entry_nodes) {
        preservedEntryNodes.add(nodeId);
      }
    }

    // For effectively collapsed annotations that are NOT displayed
    // (i.e., children of a collapsed parent), hide ALL their nodes
    // unless they are entry nodes of a displayed parent
    for (const ann of annotations) {
      if (effectivelyCollapsed.has(ann.id)) {
        const isDisplayed = displayedCollapsedAnnotations.some(d => d.id === ann.id);
        if (!isDisplayed) {
          for (const nodeId of ann.subgraph_nodes) {
            if (!preservedEntryNodes.has(nodeId)) {
              hiddenNodes.add(nodeId);
            }
          }
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
    // Uses internal nodes (hidden) and entry nodes (preserved) separately
    let connections = model.connections;
    for (const ann of displayedCollapsedAnnotations) {
      const internal = getInternalNodes(ann, annotations);
      const entrySet = new Set(ann.entry_nodes);
      const annotationNodeId = getAnnotationNodeId(ann);
      connections = rerouteConnections(connections, internal, entrySet, annotationNodeId);
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

    // Compute visible input/output nodes for correct depth assignment
    const visibleNodeIds = new Set([...filteredNodes, ...annotationNodes].map(n => n.id));
    const visibleInputNodes = model.metadata.input_nodes.filter(id => visibleNodeIds.has(id));
    const visibleOutputNodes = model.metadata.output_nodes.filter(id => visibleNodeIds.has(id));

    return {
      nodes: [...filteredNodes, ...annotationNodes],
      connections,
      metadata: {
        ...model.metadata,
        input_nodes: visibleInputNodes,
        output_nodes: visibleOutputNodes,
      },
      collapsedAnnotations: displayedCollapsedAnnotations,
      annotationNodes: annotationNodes.map(n => n.id),
    };
  }, [model, annotations, collapsedState]);
}
