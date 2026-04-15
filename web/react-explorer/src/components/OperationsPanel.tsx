import { useState, useCallback, useMemo, useEffect } from "react";
import {
  addOperation,
  removeOperation,
  getExperimentSplit,
  type Operation,
  type OperationRequest,
  type ModelState,
  type ViolationDetail,
  type AnnotationSummary,
} from "../api/client";

// =============================================================================
// Logging utilities
// =============================================================================

const LOG_PREFIX = "[OperationsPanel]";

function logDebug(message: string, data?: unknown) {
  console.log(`${LOG_PREFIX} ${message}`, data !== undefined ? data : "");
}

function logInfo(message: string, data?: unknown) {
  console.info(`${LOG_PREFIX} ${message}`, data !== undefined ? data : "");
}

function logWarn(message: string, data?: unknown) {
  console.warn(`${LOG_PREFIX} ${message}`, data !== undefined ? data : "");
}

function logError(message: string, data?: unknown) {
  console.error(`${LOG_PREFIX} ${message}`, data !== undefined ? data : "");
}

// =============================================================================
// Types
// =============================================================================

type OperationsPanelProps = {
  genomeId: string;
  experimentId: string;
  operations: Operation[];
  selectedNodes: Set<string>;
  model: ModelState;
  annotations: AnnotationSummary[];
  selectedAnnotationId?: string | null;
  onOperationChange: () => void;
};

/**
 * Analysis of a node selection for annotation creation.
 */
type SelectionAnalysis = {
  // Original selection
  selectedNodes: string[];

  // Computed subgraph (may include discovered intermediate nodes)
  entryNodes: string[];
  exitNodes: string[];
  intermediateNodes: string[];
  subgraphConnections: [string, string][];

  // Nodes discovered by graph traversal (not in original selection)
  discoveredNodes: string[];

  // External connections analysis
  externalInputs: [string, string][]; // connections INTO subgraph
  externalOutputs: [string, string][]; // connections OUT OF subgraph

  // Annotation strategy with ordered steps
  strategy: AnnotationStrategy;

  // Is the selection valid for annotation as-is?
  isValid: boolean;
};

/**
 * A single step in the annotation strategy.
 */
type StrategyStep =
  | {
      type: "expand_selection";
      nodeId: string;
      reason: string;
      externalInputs: [string, string][];
    }
  | {
      type: "add_identity_node";
      params: {
        target_node: string;
        connections: [string, string][];
        new_node_id: string;
      };
      description: string;
    }
  | {
      type: "split_node";
      params: {
        node_id: string;
      };
      description: string;
    }
  | {
      type: "annotate";
      params: {
        entry_nodes: string[];
        exit_nodes: string[];
        subgraph_nodes: string[];
      };
      description: string;
    };

/**
 * Strategy for creating an annotation, with steps in order.
 *
 * IDENTITY NODE LOGIC:
 * - Only applies when there's exactly ONE exit node
 * - If the exit node has inputs from OUTSIDE the subgraph (external inputs),
 *   it means the subgraph only partially covers the exit's inputs
 * - Strategy: Add identity node intercepting connections from subgraph to exit
 * - The identity node becomes the NEW exit; original exit is REMOVED from annotation
 *
 * SPLIT NODE LOGIC:
 * - Entry or intermediate nodes with external outputs need splitting
 * - Creates two versions: original (entry) and split (exit)
 *
 * BLOCKING ISSUES:
 * - Intermediate/exit nodes with external inputs require user to expand selection
 */
type AnnotationStrategy = {
  // Ordered list of steps to execute
  steps: StrategyStep[];

  // Blocking steps (expand_selection) - these prevent execution
  blockingSteps: StrategyStep[];

  // Executable steps (identity, split, annotate) - these run automatically
  executableSteps: StrategyStep[];

  // Is the strategy executable? (no blocking issues)
  canExecute: boolean;

  // Final annotation params after all fixes
  finalEntryNodes: string[];
  finalExitNodes: string[];
  finalSubgraphNodes: string[];
};

type WizardState =
  | { step: "idle" }
  | { step: "analyzing" }
  | { step: "review"; analysis: SelectionAnalysis; annotationName: string }
  | { step: "applying"; currentFix: number; totalFixes: number }
  | { step: "creating" }
  | { step: "error"; message: string };

// =============================================================================
// Selection Analysis Types
// =============================================================================

/**
 * Result of analyzing the current node selection.
 */
type SelectionContext =
  | { type: "empty" }
  | { type: "single"; nodeId: string; canSplit: boolean; canPrune: boolean; enabledInputs: number; enabledOutputs: number }
  | { type: "connectedPair"; fromNode: string; toNode: string; weight: number; actualConnections?: { from: string; to: string; weight: number; enabled: boolean }[] }
  | { type: "joinable"; nodeIds: string[]; baseNodeId: string }
  | { type: "subgraph"; analysis: SelectionAnalysis }
  | { type: "invalid"; reason: string };

// =============================================================================
// Node Type Detection (including split nodes)
// =============================================================================

/**
 * Get the base node ID for a potentially split node.
 * E.g., "-28_b" -> "-28", "5_split" -> "5", "5_a" -> "5"
 */
function getBaseNodeId(nodeId: string): string {
  // Match patterns like "nodeId_a", "nodeId_b", "nodeId_split"
  const splitMatch = nodeId.match(/^(.+)_([a-z]|split)$/);
  if (splitMatch) {
    return splitMatch[1];
  }
  return nodeId;
}

/**
 * Check if a node is an input-type node (including splits of input nodes).
 */
function isInputTypeNode(nodeId: string, inputNodeIds: Set<string>): boolean {
  if (inputNodeIds.has(nodeId)) return true;
  // Check if this is a split of an input node
  const baseId = getBaseNodeId(nodeId);
  return baseId !== nodeId && inputNodeIds.has(baseId);
}

/**
 * Check if a node is an output-type node (including splits of output nodes).
 */
function isOutputTypeNode(nodeId: string, outputNodeIds: Set<string>): boolean {
  if (outputNodeIds.has(nodeId)) return true;
  // Check if this is a split of an output node
  const baseId = getBaseNodeId(nodeId);
  return baseId !== nodeId && outputNodeIds.has(baseId);
}

// =============================================================================
// Annotation Node Detection
// =============================================================================

/**
 * Check if a node ID is an annotation node (starts with "A_").
 */
function isAnnotationNode(nodeId: string): boolean {
  return nodeId.startsWith("A_");
}

/**
 * Find the annotation summary for a given annotation node ID.
 */
function findAnnotationForNode(
  nodeId: string,
  annotations: AnnotationSummary[]
): AnnotationSummary | null {
  if (!isAnnotationNode(nodeId)) return null;

  // Extract the name/id from the node ID (remove "A_" prefix)
  const nameOrId = nodeId.slice(2);

  // Try to match by name first
  let annotation = annotations.find(a => a.name === nameOrId);
  if (annotation) return annotation;

  // Try to match by ID prefix
  annotation = annotations.find(a => a.id.startsWith(nameOrId));
  return annotation || null;
}

/**
 * Get all annotation nodes from a selection and their corresponding annotations.
 */
function getSelectedAnnotations(
  selectedNodes: Set<string>,
  annotations: AnnotationSummary[]
): AnnotationSummary[] {
  const result: AnnotationSummary[] = [];
  for (const nodeId of selectedNodes) {
    const annotation = findAnnotationForNode(nodeId, annotations);
    if (annotation) {
      result.push(annotation);
    }
  }
  return result;
}

// =============================================================================
// Subgraph Analysis Functions
// =============================================================================

/**
 * Build adjacency lists from model connections.
 */
function buildAdjacencyLists(model: ModelState): {
  forward: Map<string, string[]>;
  backward: Map<string, string[]>;
} {
  const forward = new Map<string, string[]>();
  const backward = new Map<string, string[]>();

  for (const node of model.nodes) {
    forward.set(node.id, []);
    backward.set(node.id, []);
  }

  for (const conn of model.connections) {
    if (!conn.enabled) continue;
    forward.get(conn.from)?.push(conn.to);
    backward.get(conn.to)?.push(conn.from);
  }

  return { forward, backward };
}

/**
 * Build annotation-aware adjacency lists.
 *
 * For annotation nodes (like A_A1678), we create virtual connections:
 * - Annotation node's outputs = connections FROM the annotation's exit nodes
 * - Annotation node's inputs = connections TO the annotation's entry nodes
 *
 * This allows selection analysis to work with annotation nodes as first-class nodes.
 */
function buildAnnotationAwareAdjacencyLists(
  model: ModelState,
  annotations: AnnotationSummary[],
  selectedNodes: Set<string>
): {
  forward: Map<string, string[]>;
  backward: Map<string, string[]>;
  annotationNodeIds: Set<string>;
} {
  const forward = new Map<string, string[]>();
  const backward = new Map<string, string[]>();
  const annotationNodeIds = new Set<string>();

  // Initialize all regular nodes
  for (const node of model.nodes) {
    forward.set(node.id, []);
    backward.set(node.id, []);
  }

  // Find which annotation nodes are in the selection
  const selectedAnnotationMap = new Map<string, AnnotationSummary>();
  for (const nodeId of selectedNodes) {
    if (isAnnotationNode(nodeId)) {
      const ann = findAnnotationForNode(nodeId, annotations);
      if (ann) {
        selectedAnnotationMap.set(nodeId, ann);
        annotationNodeIds.add(nodeId);
        // Initialize annotation node in adjacency lists
        forward.set(nodeId, []);
        backward.set(nodeId, []);
      }
    }
  }

  // Build set of all nodes that are "inside" an annotation (for exclusion)
  const nodesInsideAnnotations = new Set<string>();
  for (const ann of selectedAnnotationMap.values()) {
    for (const nodeId of ann.subgraph_nodes) {
      nodesInsideAnnotations.add(nodeId);
    }
  }

  // Process regular connections, but remap connections involving annotation internals
  for (const conn of model.connections) {
    if (!conn.enabled) continue;

    const fromInsideAnn = nodesInsideAnnotations.has(conn.from);
    const toInsideAnn = nodesInsideAnnotations.has(conn.to);

    // Find if from/to are exit/entry nodes of selected annotations
    let remappedFrom = conn.from;
    let remappedTo = conn.to;

    for (const [annNodeId, ann] of selectedAnnotationMap) {
      // If 'from' is an exit node of this annotation, remap to annotation node
      if (ann.exit_nodes.includes(conn.from)) {
        remappedFrom = annNodeId;
      }
      // If 'to' is an entry node of this annotation, remap to annotation node
      if (ann.entry_nodes.includes(conn.to)) {
        remappedTo = annNodeId;
      }
    }

    // Skip purely internal connections (both inside same annotation)
    if (fromInsideAnn && toInsideAnn) {
      // Check if they're in the same annotation
      let sameAnnotation = false;
      for (const ann of selectedAnnotationMap.values()) {
        if (ann.subgraph_nodes.includes(conn.from) && ann.subgraph_nodes.includes(conn.to)) {
          sameAnnotation = true;
          break;
        }
      }
      if (sameAnnotation) continue;
    }

    // Skip if from is inside annotation but not an exit (internal node)
    if (fromInsideAnn && remappedFrom === conn.from) continue;
    // Skip if to is inside annotation but not an entry (internal node)
    if (toInsideAnn && remappedTo === conn.to) continue;

    // Add the (possibly remapped) connection
    if (!forward.has(remappedFrom)) forward.set(remappedFrom, []);
    if (!backward.has(remappedTo)) backward.set(remappedTo, []);

    const fwdList = forward.get(remappedFrom)!;
    if (!fwdList.includes(remappedTo)) {
      fwdList.push(remappedTo);
    }
    const bwdList = backward.get(remappedTo)!;
    if (!bwdList.includes(remappedFrom)) {
      bwdList.push(remappedFrom);
    }
  }

  logDebug("Built annotation-aware adjacency lists", {
    annotationNodes: Array.from(annotationNodeIds),
    nodesInsideAnnotations: nodesInsideAnnotations.size,
  });

  return { forward, backward, annotationNodeIds };
}

/**
 * Check if selected nodes are joinable (split variants of the same node).
 * Returns the base node ID if joinable, null otherwise.
 */
function detectJoinableNodes(selectedNodes: string[]): string | null {
  if (selectedNodes.length < 2) return null;

  // Pattern 1: X_a, X_b, X_c... or X_split
  // Pattern 2: X_entry, X_exit
  const splitPatterns = [
    /^(.+)_([a-z])$/,           // X_a, X_b
    /^(.+)_split$/,             // X_split
    /^(.+)_(entry|exit)$/,      // X_entry, X_exit
  ];

  const baseIds = new Set<string>();

  for (const nodeId of selectedNodes) {
    let matched = false;
    for (const pattern of splitPatterns) {
      const match = nodeId.match(pattern);
      if (match) {
        baseIds.add(match[1]);
        matched = true;
        break;
      }
    }
    if (!matched) {
      // This node doesn't match any split pattern
      return null;
    }
  }

  // All nodes must derive from the same base
  if (baseIds.size === 1) {
    return Array.from(baseIds)[0];
  }

  return null;
}

/**
 * Check if a subgraph has valid paths from entries to exits.
 */
function hasValidPaths(
  entryNodes: string[],
  exitNodes: string[],
  forward: Map<string, string[]>,
  subgraphNodes: Set<string>
): boolean {
  if (entryNodes.length === 0 || exitNodes.length === 0) {
    return false;
  }

  const exitSet = new Set(exitNodes);

  // BFS from each entry to see if we can reach any exit within the subgraph
  for (const entry of entryNodes) {
    const visited = new Set<string>();
    const queue = [entry];

    while (queue.length > 0) {
      const node = queue.shift()!;
      if (visited.has(node)) continue;
      visited.add(node);

      if (exitSet.has(node) && node !== entry) {
        return true; // Found a path from entry to exit
      }

      for (const next of forward.get(node) || []) {
        if (subgraphNodes.has(next) && !visited.has(next)) {
          queue.push(next);
        }
      }
    }
  }

  // Special case: if a node is both entry and exit (single-node subgraph)
  for (const entry of entryNodes) {
    if (exitSet.has(entry)) {
      return true;
    }
  }

  return false;
}

/**
 * Analyze the current selection and determine what operations are available.
 */
function analyzeSelectionContext(
  selectedNodes: Set<string>,
  model: ModelState,
  annotations: AnnotationSummary[]
): SelectionContext {
  const selectedArray = Array.from(selectedNodes);

  if (selectedArray.length === 0) {
    return { type: "empty" };
  }

  if (selectedArray.length === 1) {
    const nodeId = selectedArray[0];

    // Check if it's an annotation node
    if (isAnnotationNode(nodeId)) {
      // For single annotation node, analyze as subgraph (it might connect to other things)
      const analysis = analyzeSelection(selectedNodes, model, annotations);
      return { type: "subgraph", analysis };
    }

    // Can split if it's a hidden or input node (not output)
    const node = model.nodes.find((n) => n.id === nodeId);
    const canSplit = node?.type === "hidden" || node?.type === "input";
    // Can prune if it's a hidden node with exactly 1 enabled input and 1 enabled output
    const enabledInputs = model.connections.filter(c => c.enabled && c.to === nodeId).length;
    const enabledOutputs = model.connections.filter(c => c.enabled && c.from === nodeId).length;
    const canPrune = node?.type === "hidden" && enabledInputs === 1 && enabledOutputs === 1;
    return { type: "single", nodeId, canSplit, canPrune, enabledInputs, enabledOutputs };
  }

  // Check if exactly 2 nodes with a direct connection between them
  // Also handles annotation nodes by checking their entry/exit nodes
  if (selectedArray.length === 2) {
    const [nodeA, nodeB] = selectedArray;

    // Helper to get the actual source nodes for a node (handles annotation nodes)
    const getSourceNodes = (nodeId: string): string[] => {
      if (isAnnotationNode(nodeId)) {
        const ann = findAnnotationForNode(nodeId, annotations);
        return ann ? ann.exit_nodes : [];
      }
      return [nodeId];
    };

    // Helper to get the actual target nodes for a node (handles annotation nodes)
    const getTargetNodes = (nodeId: string): string[] => {
      if (isAnnotationNode(nodeId)) {
        const ann = findAnnotationForNode(nodeId, annotations);
        return ann ? ann.entry_nodes : [];
      }
      return [nodeId];
    };

    // Get actual nodes to check for connections
    const sourcesA = getSourceNodes(nodeA);
    const targetsA = getTargetNodes(nodeA);
    const sourcesB = getSourceNodes(nodeB);
    const targetsB = getTargetNodes(nodeB);

    // Check for connection A -> B (A's exits to B's entries) — include disabled
    const connsAB: { from: string; to: string; weight: number; enabled: boolean }[] = [];
    for (const srcA of sourcesA) {
      for (const tgtB of targetsB) {
        const conn = model.connections.find(
          c => c.from === srcA && c.to === tgtB
        );
        if (conn) {
          connsAB.push({ from: srcA, to: tgtB, weight: conn.weight, enabled: conn.enabled });
        }
      }
    }

    // Check for connection B -> A (B's exits to A's entries) — include disabled
    const connsBA: { from: string; to: string; weight: number; enabled: boolean }[] = [];
    for (const srcB of sourcesB) {
      for (const tgtA of targetsA) {
        const conn = model.connections.find(
          c => c.from === srcB && c.to === tgtA
        );
        if (conn) {
          connsBA.push({ from: srcB, to: tgtA, weight: conn.weight, enabled: conn.enabled });
        }
      }
    }

    // If there's exactly one direction of connection, offer identity node insertion
    // For annotation nodes, we use the display ID but store the actual connection
    if (connsAB.length > 0 && connsBA.length === 0) {
      // Use first connection for weight display, but we'll show all in UI
      return {
        type: "connectedPair",
        fromNode: nodeA,
        toNode: nodeB,
        weight: connsAB[0].weight,
        // Store actual connections for the operation
        actualConnections: connsAB,
      };
    }
    if (connsBA.length > 0 && connsAB.length === 0) {
      return {
        type: "connectedPair",
        fromNode: nodeB,
        toNode: nodeA,
        weight: connsBA[0].weight,
        actualConnections: connsBA,
      };
    }
    // If both directions exist, fall through to subgraph analysis
  }

  // Check if nodes are joinable (split variants)
  const baseNodeId = detectJoinableNodes(selectedArray);
  if (baseNodeId) {
    return { type: "joinable", nodeIds: selectedArray, baseNodeId };
  }

  // Check if nodes form a valid subgraph (annotation-aware)
  const analysis = analyzeSelection(selectedNodes, model, annotations);
  const { forward } = buildAnnotationAwareAdjacencyLists(model, annotations, selectedNodes);
  const allSubgraphNodes = new Set([
    ...analysis.selectedNodes,
    ...analysis.discoveredNodes,
  ]);

  const hasValidSubgraph = hasValidPaths(
    analysis.entryNodes,
    analysis.exitNodes,
    forward,
    allSubgraphNodes
  );

  if (hasValidSubgraph) {
    return { type: "subgraph", analysis };
  }

  // No valid subgraph - check why
  if (analysis.entryNodes.length === 0) {
    return { type: "invalid", reason: "No entry nodes found" };
  }
  if (analysis.exitNodes.length === 0) {
    return { type: "invalid", reason: "No exit nodes found" };
  }

  return { type: "invalid", reason: "No valid paths between selected nodes" };
}

/**
 * Compute the containing subgraph from a node selection.
 *
 * Given a set of selected nodes, this function:
 * 1. Identifies entry nodes (inputs or nodes with external inputs)
 * 2. Identifies exit nodes (nodes with external outputs)
 * 3. Finds intermediate nodes on paths between entries and exits
 * 4. Returns the complete subgraph structure
 *
 * Supports annotation nodes - when an annotation node (A_xxx) is selected,
 * it's treated as a unit with virtual connections based on its entry/exit nodes.
 */
function computeContainingSubgraph(
  selectedNodes: Set<string>,
  model: ModelState,
  annotations: AnnotationSummary[]
): {
  entryNodes: string[];
  exitNodes: string[];
  intermediateNodes: string[];
  discoveredNodes: string[]; // Nodes found by traversal that weren't in selection
  allSubgraphNodes: string[];
  subgraphConnections: [string, string][];
} {
  logInfo("Computing containing subgraph from selection", {
    selectedCount: selectedNodes.size,
    selected: Array.from(selectedNodes),
  });

  // Use annotation-aware adjacency lists
  const { forward, backward, annotationNodeIds } = buildAnnotationAwareAdjacencyLists(
    model,
    annotations,
    selectedNodes
  );
  const inputNodeIds = new Set(model.metadata.input_nodes);
  const outputNodeIds = new Set(model.metadata.output_nodes);

  // Step 1: Identify entry and exit candidates from selection
  // Entry nodes: ALL inputs are external (from outside subgraph) OR they are input nodes
  // Exit nodes: ALL outputs are external (to outside subgraph)
  const entryNodes: string[] = [];
  const exitNodes: string[] = [];

  for (const nodeId of selectedNodes) {
    const incoming = backward.get(nodeId) || [];
    const outgoing = forward.get(nodeId) || [];

    // Check if this is an input-type node (including splits of inputs)
    // Annotation nodes are never input-type, they have their own entry semantics
    const isInputType = !annotationNodeIds.has(nodeId) && isInputTypeNode(nodeId, inputNodeIds);
    const isOutputType = !annotationNodeIds.has(nodeId) && isOutputTypeNode(nodeId, outputNodeIds);

    const incomingFromInside = incoming.filter((from) => selectedNodes.has(from));
    const incomingFromOutside = incoming.filter((from) => !selectedNodes.has(from));
    const outgoingToInside = outgoing.filter((to) => selectedNodes.has(to));
    const outgoingToOutside = outgoing.filter((to) => !selectedNodes.has(to));

    // Entry: input node OR has no internal inputs (all inputs from outside or none)
    // For annotation nodes: entry if no selected nodes connect to it
    const isEntry = isInputType || (incomingFromInside.length === 0);

    // Exit: has outputs going outside (no internal outputs) OR is an output node with no internal outputs
    // For annotation nodes: exit if it connects to nodes outside selection
    const isExit = (outgoingToOutside.length > 0 || isOutputType) && outgoingToInside.length === 0;

    if (isEntry) {
      entryNodes.push(nodeId);
    }
    if (isExit) {
      exitNodes.push(nodeId);
    }

    logDebug(`Node ${nodeId} classification`, {
      isInputType,
      incomingFromInside: incomingFromInside.length,
      incomingFromOutside: incomingFromOutside.length,
      outgoingToInside: outgoingToInside.length,
      outgoingToOutside: outgoingToOutside.length,
      isEntry,
      isExit,
    });
  }

  logDebug("Initial classification from selection", {
    entries: entryNodes,
    exits: exitNodes,
  });

  // Step 1.5: Check for direct connections between entries and exits
  // If direct connections exist, don't auto-discover intermediate nodes on longer paths
  const entrySet = new Set(entryNodes);
  const exitSet = new Set(exitNodes);
  let hasDirectConnection = false;

  for (const conn of model.connections) {
    if (!conn.enabled) continue;
    // Check if this connection goes directly from an entry to an exit
    // (or to the same node if it's both entry and exit)
    if (entrySet.has(conn.from) && exitSet.has(conn.to)) {
      hasDirectConnection = true;
      logDebug("Found direct connection from entry to exit", {
        from: conn.from,
        to: conn.to,
      });
      break;
    }
  }

  // Also check annotation-aware adjacency for direct connections
  if (!hasDirectConnection) {
    for (const entry of entryNodes) {
      const outgoing = forward.get(entry) || [];
      for (const target of outgoing) {
        if (exitSet.has(target)) {
          hasDirectConnection = true;
          logDebug("Found direct connection (annotation-aware) from entry to exit", {
            from: entry,
            to: target,
          });
          break;
        }
      }
      if (hasDirectConnection) break;
    }
  }

  // If direct connections exist, skip path discovery - only use selected nodes
  if (hasDirectConnection) {
    logInfo("Direct connection exists - skipping intermediate node discovery", {
      entryNodes,
      exitNodes,
      selectedCount: selectedNodes.size,
    });

    // No intermediate nodes discovered, just use the selection
    const allSubgraphNodes = Array.from(selectedNodes);
    const subgraphNodeSet = new Set(allSubgraphNodes);
    const subgraphConnections: [string, string][] = [];

    for (const conn of model.connections) {
      if (!conn.enabled) continue;
      if (subgraphNodeSet.has(conn.from) && subgraphNodeSet.has(conn.to)) {
        subgraphConnections.push([conn.from, conn.to]);
      }
    }

    // Compute intermediate nodes (selected nodes that are neither entry nor exit)
    const intermediateNodes = allSubgraphNodes.filter(
      n => !entryNodes.includes(n) && !exitNodes.includes(n)
    );

    return {
      entryNodes,
      exitNodes,
      intermediateNodes,
      discoveredNodes: [], // No discovery when direct connection exists
      allSubgraphNodes,
      subgraphConnections,
    };
  }

  // Step 2: Find all nodes reachable from entries (forward BFS)
  const reachableFromEntries = new Set<string>();
  let queue = [...entryNodes];
  while (queue.length > 0) {
    const node = queue.shift()!;
    if (reachableFromEntries.has(node)) continue;
    reachableFromEntries.add(node);
    for (const next of forward.get(node) || []) {
      if (!reachableFromEntries.has(next)) {
        queue.push(next);
      }
    }
  }

  // Step 3: Find all nodes that can reach exits (backward BFS)
  const canReachExits = new Set<string>();
  queue = [...exitNodes];
  while (queue.length > 0) {
    const node = queue.shift()!;
    if (canReachExits.has(node)) continue;
    canReachExits.add(node);
    for (const prev of backward.get(node) || []) {
      if (!canReachExits.has(prev)) {
        queue.push(prev);
      }
    }
  }

  // Step 4: Nodes on paths = intersection of reachable from entries AND can reach exits
  const nodesOnPaths = new Set<string>();
  for (const node of reachableFromEntries) {
    if (canReachExits.has(node)) {
      nodesOnPaths.add(node);
    }
  }

  logDebug("Path analysis", {
    reachableFromEntries: reachableFromEntries.size,
    canReachExits: canReachExits.size,
    nodesOnPaths: nodesOnPaths.size,
  });

  // Step 5: Find discovered intermediate nodes (on paths but not in selection)
  const discoveredNodes: string[] = [];
  const intermediateNodes: string[] = [];

  for (const node of nodesOnPaths) {
    if (!selectedNodes.has(node)) {
      discoveredNodes.push(node);
      intermediateNodes.push(node);
      logDebug(`Discovered intermediate node: ${node}`);
    } else if (!entryNodes.includes(node) && !exitNodes.includes(node)) {
      intermediateNodes.push(node);
    }
  }

  // The complete subgraph is: selected nodes + discovered intermediate nodes
  const allSubgraphNodes = Array.from(new Set([...selectedNodes, ...discoveredNodes]));

  // Step 6: Compute subgraph connections (connections within the subgraph)
  const subgraphNodeSet = new Set(allSubgraphNodes);
  const subgraphConnections: [string, string][] = [];

  for (const conn of model.connections) {
    if (!conn.enabled) continue;
    if (subgraphNodeSet.has(conn.from) && subgraphNodeSet.has(conn.to)) {
      subgraphConnections.push([conn.from, conn.to]);
    }
  }

  logInfo("Containing subgraph computed", {
    entryNodes,
    exitNodes,
    intermediateNodes,
    discoveredNodes,
    totalNodes: allSubgraphNodes.length,
    connections: subgraphConnections.length,
  });

  return {
    entryNodes,
    exitNodes,
    intermediateNodes,
    discoveredNodes,
    allSubgraphNodes,
    subgraphConnections,
  };
}

/**
 * Analyze a node selection to determine the subgraph structure and annotation strategy.
 */
function analyzeSelection(
  selectedNodes: Set<string>,
  model: ModelState,
  annotations: AnnotationSummary[]
): SelectionAnalysis {
  logInfo("Analyzing selection for annotation", {
    selectedCount: selectedNodes.size,
    selected: Array.from(selectedNodes),
  });

  // First, compute the containing subgraph (finds intermediate nodes)
  const containingSubgraph = computeContainingSubgraph(selectedNodes, model, annotations);

  // Use the complete subgraph for analysis
  const allSubgraphNodes = new Set(containingSubgraph.allSubgraphNodes);

  // Build annotation-aware adjacency for connection analysis
  const { forward, backward, annotationNodeIds } = buildAnnotationAwareAdjacencyLists(
    model,
    annotations,
    allSubgraphNodes
  );

  // Build set of nodes that are "inside" the subgraph (including annotation internals)
  // A node is inside if:
  // 1. It's directly in allSubgraphNodes, OR
  // 2. It's inside a selected annotation's subgraph (entry, exit, or intermediate)
  const nodesInsideSubgraph = new Set(allSubgraphNodes);
  for (const nodeId of allSubgraphNodes) {
    if (isAnnotationNode(nodeId)) {
      const ann = findAnnotationForNode(nodeId, annotations);
      if (ann) {
        for (const subNode of ann.subgraph_nodes) {
          nodesInsideSubgraph.add(subNode);
        }
      }
    }
  }

  logDebug("Expanded subgraph for connection analysis", {
    originalSize: allSubgraphNodes.size,
    expandedSize: nodesInsideSubgraph.size,
    annotationNodes: Array.from(annotationNodeIds),
  });

  // Categorize connections relative to the complete subgraph
  // Use annotation-aware adjacency lists
  const subgraphConnections: [string, string][] = [];
  const externalInputConnections: [string, string][] = [];
  const externalOutputConnections: [string, string][] = [];

  for (const nodeId of allSubgraphNodes) {
    const outgoing = forward.get(nodeId) || [];
    const incoming = backward.get(nodeId) || [];

    for (const to of outgoing) {
      // Check if target is internal (in subgraph OR inside a selected annotation)
      if (nodesInsideSubgraph.has(to)) {
        // Internal connection
        const connKey = `${nodeId}->${to}`;
        if (!subgraphConnections.some(([f, t]) => `${f}->${t}` === connKey)) {
          subgraphConnections.push([nodeId, to]);
        }
      } else {
        // External output
        externalOutputConnections.push([nodeId, to]);
      }
    }

    for (const from of incoming) {
      // Check if source is external (not in subgraph AND not inside a selected annotation)
      if (!nodesInsideSubgraph.has(from)) {
        // External input
        externalInputConnections.push([from, nodeId]);
      }
    }
  }

  logDebug("Connection analysis (full subgraph)", {
    internal: subgraphConnections.length,
    externalInputs: externalInputConnections.length,
    externalOutputs: externalOutputConnections.length,
    annotationNodes: Array.from(annotationNodeIds),
  });

  // Classify nodes based on the complete subgraph
  const inputNodeIds = new Set(model.metadata.input_nodes);
  const outputNodeIds = new Set(model.metadata.output_nodes);

  // Build sets for faster lookups
  const nodesWithExternalInput = new Set(externalInputConnections.map(([, to]) => to));
  const nodesWithExternalOutput = new Set(externalOutputConnections.map(([from]) => from));
  const nodesWithInternalInput = new Set(subgraphConnections.map(([, to]) => to));
  const nodesWithInternalOutput = new Set(subgraphConnections.map(([from]) => from));

  const entryNodes: string[] = [];
  const exitNodes: string[] = [];
  const intermediateNodes: string[] = [];

  for (const nodeId of allSubgraphNodes) {
    const isAnnNode = annotationNodeIds.has(nodeId);

    // Use helper functions to handle split nodes (e.g., -28_b is input type if -28 is input)
    // Annotation nodes don't have input/output type - they have their own entry/exit semantics
    const isInputType = !isAnnNode && isInputTypeNode(nodeId, inputNodeIds);
    const isOutputType = !isAnnNode && isOutputTypeNode(nodeId, outputNodeIds);
    const hasInternalInput = nodesWithInternalInput.has(nodeId);
    const hasInternalOutput = nodesWithInternalOutput.has(nodeId);
    const hasExternalInput = nodesWithExternalInput.has(nodeId);
    const hasExternalOutput = nodesWithExternalOutput.has(nodeId);

    // Entry conditions:
    // - Regular node: input type OR has no internal inputs
    // - Annotation node: has no internal inputs (nothing in selection connects to it)
    const isEntry = isInputType || !hasInternalInput;

    // Exit conditions:
    // - Regular node: has external outputs OR is output type, AND has no internal outputs
    // - Annotation node: has external outputs AND has no internal outputs
    const isExit = (hasExternalOutput || isOutputType) && !hasInternalOutput;

    logDebug(`Classifying node ${nodeId} in subgraph`, {
      isAnnotationNode: isAnnNode,
      isInputType,
      isOutputType,
      hasInternalInput,
      hasInternalOutput,
      hasExternalInput,
      hasExternalOutput,
      isEntry,
      isExit,
    });

    if (isEntry && isExit) {
      entryNodes.push(nodeId);
      exitNodes.push(nodeId);
    } else if (isEntry) {
      entryNodes.push(nodeId);
    } else if (isExit) {
      exitNodes.push(nodeId);
    } else {
      intermediateNodes.push(nodeId);
    }
  }

  logDebug("Node classification (full subgraph)", {
    entry: entryNodes,
    exit: exitNodes,
    intermediate: intermediateNodes,
    discovered: containingSubgraph.discoveredNodes,
  });

  // Build annotation strategy
  const strategy = buildAnnotationStrategy(
    entryNodes,
    exitNodes,
    intermediateNodes,
    externalInputConnections,
    externalOutputConnections,
    allSubgraphNodes,
    model,
    annotations,
    containingSubgraph.discoveredNodes
  );

  const isValid = strategy.canExecute && strategy.executableSteps.length === 1;

  const analysis: SelectionAnalysis = {
    selectedNodes: Array.from(selectedNodes),
    entryNodes,
    exitNodes,
    intermediateNodes,
    subgraphConnections,
    discoveredNodes: containingSubgraph.discoveredNodes,
    externalInputs: externalInputConnections,
    externalOutputs: externalOutputConnections,
    strategy,
    isValid,
  };

  logInfo("Selection analysis complete", {
    isValid,
    canExecute: strategy.canExecute,
    totalSteps: strategy.steps.length,
    blockingSteps: strategy.blockingSteps.length,
    executableSteps: strategy.executableSteps.length,
  });

  return analysis;
}

/**
 * Build the annotation strategy with ordered steps.
 *
 * Order of evaluation:
 * 1. Check if identity node can resolve exit node's external inputs (single exit case)
 * 2. Determine blocking expand_selection steps (excluding nodes resolved by identity)
 * 3. Determine split nodes for entry/intermediate with external outputs
 * 4. Compute final annotation parameters
 */
function buildAnnotationStrategy(
  entryNodes: string[],
  exitNodes: string[],
  intermediateNodes: string[],
  externalInputs: [string, string][],
  externalOutputs: [string, string][],
  allSubgraphNodes: Set<string>,
  model: ModelState,
  annotations: AnnotationSummary[],
  discoveredNodes: string[]
): AnnotationStrategy {
  const entrySet = new Set(entryNodes);
  const exitSet = new Set(exitNodes);
  const intermediateSet = new Set(intermediateNodes);

  const steps: StrategyStep[] = [];
  const blockingSteps: StrategyStep[] = [];
  const executableSteps: StrategyStep[] = [];

  // Track nodes that will be created/modified
  let exitNodeToReplace: string | null = null;
  const identityNodes: { targetNode: string; newNodeId: string; connections: [string, string][] }[] = [];
  const splitNodes: string[] = [];

  // =========================================================================
  // 1. FIRST: Check if identity node can resolve exit node's external inputs
  // =========================================================================
  // Only if there's exactly one exit node with external inputs
  if (exitNodes.length === 1) {
    const exitNode = exitNodes[0];
    const exitHasExternalInputs = externalInputs.some(([, to]) => to === exitNode);

    if (exitHasExternalInputs) {
      // Find all connections from subgraph to exit node
      const connectionsToExit: [string, string][] = [];
      for (const conn of model.connections) {
        if (!conn.enabled) continue;
        if (conn.to === exitNode && allSubgraphNodes.has(conn.from)) {
          connectionsToExit.push([conn.from, conn.to]);
        }
      }

      if (connectionsToExit.length > 0) {
        let identityCounter = 1;
        const existingNodeIds = new Set(model.nodes.map((n) => n.id));
        while (existingNodeIds.has(`identity_${identityCounter}`)) {
          identityCounter++;
        }
        const newNodeId = `identity_${identityCounter}`;

        identityNodes.push({ targetNode: exitNode, newNodeId, connections: connectionsToExit });
        exitNodeToReplace = exitNode;

        const step: StrategyStep = {
          type: "add_identity_node",
          params: {
            target_node: exitNode,
            connections: connectionsToExit,
            new_node_id: newNodeId,
          },
          description: `Add ${newNodeId} intercepting ${connectionsToExit.length} connection(s) to ${exitNode}`,
        };
        steps.push(step);
        executableSteps.push(step);
      }
    }
  }

  // =========================================================================
  // 2. BLOCKING: Intermediate/exit nodes with external inputs (need expansion)
  //    SKIP exit nodes that will be replaced by identity node
  // =========================================================================
  const nodesNeedingExpansion = new Map<string, [string, string][]>();

  for (const [from, to] of externalInputs) {
    // External input is OK for entry nodes (that's their role)
    // But intermediate or exit nodes should NOT have external inputs
    // EXCEPTION: Skip exit node if it will be replaced by identity node
    if (to === exitNodeToReplace) {
      continue; // Identity node resolves this
    }
    if (intermediateSet.has(to) || (exitSet.has(to) && !entrySet.has(to))) {
      if (!nodesNeedingExpansion.has(to)) {
        nodesNeedingExpansion.set(to, []);
      }
      nodesNeedingExpansion.get(to)!.push([from, to]);
    }
  }

  for (const [nodeId, extInputs] of nodesNeedingExpansion) {
    const step: StrategyStep = {
      type: "expand_selection",
      nodeId,
      reason: `Node has external inputs from: ${extInputs.map(([from]) => from).join(", ")}`,
      externalInputs: extInputs,
    };
    steps.push(step);
    blockingSteps.push(step);
  }

  // =========================================================================
  // 3. AUTO: Handle entry/intermediate nodes with external outputs
  //    - Regular nodes: split them
  //    - Annotation nodes: add identity nodes on internal connections
  // =========================================================================
  const nodesToSplitMap = new Map<string, [string, string][]>();
  const annotationNodesToIntercept = new Map<string, [string, string][]>();

  for (const [from, to] of externalOutputs) {
    // External output is OK for exit nodes (that's their role)
    // But entry or intermediate nodes should NOT have external outputs
    if ((entrySet.has(from) || intermediateSet.has(from)) && !exitSet.has(from)) {
      if (isAnnotationNode(from)) {
        // Annotation nodes can't be split - track for identity node interception
        if (!annotationNodesToIntercept.has(from)) {
          annotationNodesToIntercept.set(from, []);
        }
        annotationNodesToIntercept.get(from)!.push([from, to]);
      } else {
        // Regular nodes can be split
        if (!nodesToSplitMap.has(from)) {
          nodesToSplitMap.set(from, []);
        }
        nodesToSplitMap.get(from)!.push([from, to]);
      }
    }
  }

  // For annotation nodes with external outputs, add identity nodes on INTERNAL connections
  // This intercepts the annotation's output going INTO the subgraph
  for (const [annNodeId, extOutputs] of annotationNodesToIntercept) {
    // Find internal connections from this annotation node to subgraph nodes
    const internalConnections: [string, string][] = [];
    for (const conn of model.connections) {
      if (!conn.enabled) continue;
      // We need to find connections from annotation's exits to subgraph nodes
      // Since annNodeId is synthetic, look for connections from real exit nodes
      const ann = annotations.find(a => `A_${a.name || a.id.slice(0, 8)}` === annNodeId);
      if (ann) {
        for (const exitNode of ann.exit_nodes) {
          if (conn.from === exitNode && allSubgraphNodes.has(conn.to) && conn.to !== annNodeId) {
            internalConnections.push([conn.from, conn.to]);
          }
        }
      }
    }

    if (internalConnections.length > 0) {
      // Group by target node and add identity nodes
      const targetNodes = [...new Set(internalConnections.map(([, to]) => to))];
      for (const targetNode of targetNodes) {
        const connectionsToTarget = internalConnections.filter(([, to]) => to === targetNode);

        let identityCounter = 1;
        const existingNodeIds = new Set(model.nodes.map((n) => n.id));
        while (existingNodeIds.has(`identity_${identityCounter}`)) {
          identityCounter++;
        }
        // Also check already planned identity nodes
        for (const planned of identityNodes) {
          if (planned.newNodeId === `identity_${identityCounter}`) {
            identityCounter++;
          }
        }
        const newNodeId = `identity_${identityCounter}`;

        identityNodes.push({ targetNode, newNodeId, connections: connectionsToTarget });

        const step: StrategyStep = {
          type: "add_identity_node",
          params: {
            target_node: targetNode,
            connections: connectionsToTarget,
            new_node_id: newNodeId,
          },
          description: `Add ${newNodeId} intercepting connection(s) from ${annNodeId} to ${targetNode}`,
        };
        steps.push(step);
        executableSteps.push(step);
      }
    }
  }

  // Regular nodes with external outputs: need to be split
  // For ENTRY nodes, this is BLOCKING - user must split first to choose which variant goes in the annotation
  // For INTERMEDIATE nodes, this is also BLOCKING - same reason
  for (const [nodeId, extOutputs] of nodesToSplitMap) {
    const isEntryNode = entrySet.has(nodeId);
    const targets = extOutputs.map(([, to]) => to).join(", ");

    const step: StrategyStep = {
      type: "split_node",
      params: { node_id: nodeId },
      description: `Split ${nodeId} (has external output to ${targets})`,
      reason: `${nodeId} outputs to ${targets} outside the annotation`,
    };
    steps.push(step);

    // Entry/intermediate nodes with external outputs are BLOCKING
    // User must split first, then select the appropriate variant
    blockingSteps.push(step);

    // Note: we don't add to splitNodes[] here because the split hasn't happened yet
    // The user needs to manually split and re-select
  }

  // =========================================================================
  // 4. Compute final annotation parameters
  // =========================================================================
  let finalSubgraphNodes = Array.from(allSubgraphNodes);

  // Remove exit node if replaced by identity
  if (exitNodeToReplace) {
    finalSubgraphNodes = finalSubgraphNodes.filter((n) => n !== exitNodeToReplace);
  }

  // Remove annotation nodes that are being replaced by identity node interceptions
  // (annotation nodes with external outputs whose internal connections are being intercepted)
  const annotationNodesToRemove = new Set(annotationNodesToIntercept.keys());
  finalSubgraphNodes = finalSubgraphNodes.filter((n) => !annotationNodesToRemove.has(n));

  // Add new nodes from operations
  finalSubgraphNodes = [
    ...finalSubgraphNodes,
    ...identityNodes.map((n) => n.newNodeId),
    ...splitNodes.map((n) => `${n}_split`),
  ];

  // Compute final entry nodes
  // Remove annotation nodes being replaced, add identity nodes that intercept their connections
  let finalEntryNodes: string[] = [];

  // Track which identity nodes were created for annotation interception
  const identityNodesForAnnotations = new Set<string>();
  for (const [annNodeId] of annotationNodesToIntercept) {
    // Find identity nodes created for this annotation's internal connections
    const ann = annotations.find(a => `A_${a.name || a.id.slice(0, 8)}` === annNodeId);
    if (ann) {
      for (const identity of identityNodes) {
        // Check if any of the intercepted connections come from this annotation's exit nodes
        const fromAnnExits = identity.connections.some(([from]) => ann.exit_nodes.includes(from));
        if (fromAnnExits) {
          identityNodesForAnnotations.add(identity.newNodeId);
        }
      }
    }
  }

  for (const entryNode of entryNodes) {
    if (annotationNodesToRemove.has(entryNode)) {
      // This annotation entry is being replaced - identity nodes become entries
      // Add all identity nodes that intercept connections from this annotation
      const ann = annotations.find(a => `A_${a.name || a.id.slice(0, 8)}` === entryNode);
      if (ann) {
        for (const identity of identityNodes) {
          const fromAnnExits = identity.connections.some(([from]) => ann.exit_nodes.includes(from));
          if (fromAnnExits) {
            finalEntryNodes.push(identity.newNodeId);
          }
        }
      }
    } else {
      finalEntryNodes.push(entryNode);
    }
  }
  // Deduplicate entries
  finalEntryNodes = [...new Set(finalEntryNodes)];

  // Compute final exit nodes
  let finalExitNodes: string[] = [];

  if (exitNodeToReplace) {
    // Replace exit with identity
    for (const exitNode of exitNodes) {
      if (exitNode === exitNodeToReplace) {
        const identity = identityNodes.find((n) => n.targetNode === exitNode);
        if (identity) {
          finalExitNodes.push(identity.newNodeId);
        }
      } else {
        finalExitNodes.push(exitNode);
      }
    }
  } else {
    finalExitNodes = [...exitNodes];
  }

  // Note: Split nodes will be handled after the split operation runs
  // The actual split variant names (_a, _b) are determined by the backend
  // We don't add them here since they don't exist yet

  // =========================================================================
  // 5. Filter out nodes that shouldn't be in the backend annotation request
  // =========================================================================
  // Two types of nodes to filter:
  // a) Synthetic annotation nodes (A_xxx) - frontend-only, don't exist in backend
  // b) Nodes already covered by existing annotations - would cause "already covered" error

  // Build set of all nodes covered by existing annotations
  const coveredByExistingAnnotations = new Set<string>();
  for (const ann of annotations) {
    for (const nodeId of ann.subgraph_nodes) {
      coveredByExistingAnnotations.add(nodeId);
    }
  }

  logDebug("Filtering nodes for backend", {
    syntheticAnnotationNodes: finalSubgraphNodes.filter(n => isAnnotationNode(n)),
    coveredByExisting: finalSubgraphNodes.filter(n => coveredByExistingAnnotations.has(n)),
  });

  // Subgraph nodes must exclude both synthetic annotation nodes AND covered nodes
  // (backend rejects covered nodes in subgraph_nodes)
  const filterSubgraphForBackend = (nodes: string[]) =>
    nodes.filter(n => !isAnnotationNode(n) && !coveredByExistingAnnotations.has(n));

  // Entry/exit nodes only need to exclude synthetic annotation nodes.
  // They CAN reference covered nodes (e.g., nodes inside child annotations)
  // because the backend checks entry/exit as subsets of internal_nodes
  // (which includes child annotation nodes).
  const filterBoundaryForBackend = (nodes: string[]) =>
    nodes.filter(n => !isAnnotationNode(n));

  const backendSubgraphNodes = filterSubgraphForBackend(finalSubgraphNodes);
  const backendEntryNodes = filterBoundaryForBackend(finalEntryNodes);
  const backendExitNodes = filterBoundaryForBackend(finalExitNodes);

  // Warn if filtering removed all nodes (no valid glue nodes)
  if (backendSubgraphNodes.length === 0) {
    logWarn("All subgraph nodes filtered out - no uncovered glue nodes available", {
      originalCount: finalSubgraphNodes.length,
      filteredAsSynthetic: finalSubgraphNodes.filter(n => isAnnotationNode(n)).length,
      filteredAsCovered: finalSubgraphNodes.filter(n => coveredByExistingAnnotations.has(n)).length,
    });
  }

  // =========================================================================
  // 6. Add final annotate step
  // =========================================================================
  const annotateStep: StrategyStep = {
    type: "annotate",
    params: {
      entry_nodes: backendEntryNodes,
      exit_nodes: backendExitNodes,
      subgraph_nodes: backendSubgraphNodes,
    },
    description: `Create annotation with ${backendSubgraphNodes.length} nodes`,
  };
  steps.push(annotateStep);
  executableSteps.push(annotateStep);

  // Can execute if there are no blocking issues (nodes needing expansion)
  const canExecute = blockingSteps.length === 0 &&
    (exitNodes.length > 0 || identityNodes.length > 0);

  logDebug("Annotation strategy built", {
    canExecute,
    totalSteps: steps.length,
    blockingSteps: blockingSteps.length,
    executableSteps: executableSteps.length,
  });

  return {
    steps,
    blockingSteps,
    executableSteps,
    canExecute,
    finalEntryNodes,
    finalExitNodes,
    finalSubgraphNodes,
  };
}

// =============================================================================
// Component
// =============================================================================

export function OperationsPanel({
  genomeId,
  experimentId,
  operations,
  selectedNodes,
  model,
  annotations,
  selectedAnnotationId,
  onOperationChange,
}: OperationsPanelProps) {
  const [collapsed, setCollapsed] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [annotationName, setAnnotationName] = useState("");
  const [renameInput, setRenameInput] = useState("");
  const [annotationRenameInput, setAnnotationRenameInput] = useState("");
  const [operationNotes, setOperationNotes] = useState("");
  const [featureNames, setFeatureNames] = useState<string[] | null>(null);
  const [featureTypes, setFeatureTypes] = useState<Record<string, string> | null>(null);
  const [featureDescriptions, setFeatureDescriptions] = useState<Record<string, string> | null>(null);

  // Wizard state (for applying fixes)
  const [wizard, setWizard] = useState<WizardState>({ step: "idle" });

  // Fetch feature names from experiment split (for rename suggestions)
  useEffect(() => {
    if (!experimentId) return;
    getExperimentSplit(experimentId)
      .then((split) => {
        setFeatureNames(split.feature_names ?? null);
        setFeatureTypes(split.feature_types ?? null);
        setFeatureDescriptions(split.feature_descriptions ?? null);
      })
      .catch(() => {
        setFeatureNames(null);
        setFeatureTypes(null);
        setFeatureDescriptions(null);
      });
  }, [experimentId]);

  const selectedArray = useMemo(() => Array.from(selectedNodes), [selectedNodes]);

  // Detect if any selected nodes are annotation nodes
  const selectedAnnotationNodes = useMemo(() => {
    return getSelectedAnnotations(selectedNodes, annotations);
  }, [selectedNodes, annotations]);

  // Compute selection context dynamically
  const selectionContext = useMemo(() => {
    const context = analyzeSelectionContext(selectedNodes, model, annotations);
    logDebug("Selection context computed", {
      type: context.type,
      selectedCount: selectedNodes.size,
      selectedAnnotationNodes: selectedAnnotationNodes.length,
    });
    return context;
  }, [selectedNodes, model, annotations, selectedAnnotationNodes]);

  logDebug("Render", {
    selectedCount: selectedNodes.size,
    operationsCount: operations.length,
    wizardStep: wizard.step,
    contextType: selectionContext.type,
  });

  // ==========================================================================
  // Basic Operations
  // ==========================================================================

  const handleAddOperation = useCallback(
    async (operation: OperationRequest): Promise<boolean> => {
      logInfo("Adding operation", operation);
      try {
        setLoading(true);
        setError(null);
        const notes = operationNotes.trim() || undefined;
        const result = await addOperation(genomeId, operation, notes);
        logInfo("Operation added successfully", result);
        setOperationNotes(""); // Clear after successful operation
        onOperationChange();
        return true;
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : "Failed to add operation";
        logError("Failed to add operation", { error: err, operation });
        setError(errorMsg);
        return false;
      } finally {
        setLoading(false);
      }
    },
    [genomeId, operationNotes, onOperationChange]
  );

  const handleRemoveOperation = useCallback(
    async (seq: number) => {
      // Find how many operations will be removed (this one and all subsequent)
      const opsToRemove = operations.filter(op => op.seq >= seq);
      const count = opsToRemove.length;

      const message = count === 1
        ? `Remove operation #${seq}?`
        : `Remove operation #${seq} and ${count - 1} subsequent operation(s)?`;

      if (!window.confirm(message)) {
        logDebug("Operation removal cancelled by user", { seq });
        return;
      }

      logInfo("Removing operation", { seq, count });
      try {
        setLoading(true);
        setError(null);
        await removeOperation(genomeId, seq);
        logInfo("Operation removed successfully");
        onOperationChange();
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : "Failed to remove operation";
        logError("Failed to remove operation", { error: err, seq });
        setError(errorMsg);
      } finally {
        setLoading(false);
      }
    },
    [genomeId, operations, onOperationChange]
  );

  const handleSplitNode = useCallback(
    (nodeId: string) => {
      logInfo("Splitting node", { nodeId });
      handleAddOperation({
        type: "split_node",
        params: { node_id: nodeId },
      });
    },
    [handleAddOperation]
  );

  const handleConsolidateNodes = useCallback(
    (nodeIds: string[]) => {
      logInfo("Consolidating nodes", { nodeIds });
      handleAddOperation({
        type: "consolidate_node",
        params: { node_ids: nodeIds },
      });
    },
    [handleAddOperation]
  );

  const handleAddIdentityOnConnection = useCallback(
    async (
      fromNode: string,
      toNode: string,
      actualConnections?: { from: string; to: string; weight: number }[]
    ) => {
      // Use actual connections if provided (for annotation nodes), otherwise use the simple pair
      const connections: [string, string][] = actualConnections
        ? actualConnections.map(c => [c.from, c.to])
        : [[fromNode, toNode]];

      // Determine the target node (the "to" side of the connections)
      const targetNode = actualConnections ? actualConnections[0].to : toNode;

      logInfo("Adding identity node on connection", {
        fromNode,
        toNode,
        actualConnections,
        targetNode,
        connections,
      });

      // Generate a new identity node ID
      let identityCounter = 1;
      const existingNodeIds = new Set(model.nodes.map((n) => n.id));
      while (existingNodeIds.has(`identity_${identityCounter}`)) {
        identityCounter++;
      }
      const newNodeId = `identity_${identityCounter}`;

      const success = await handleAddOperation({
        type: "add_identity_node",
        params: {
          target_node: targetNode,
          connections,
          new_node_id: newNodeId,
        },
      });

    },
    [model, handleAddOperation]
  );

  /**
   * Execute the annotation strategy by processing each step in order.
   * Steps are: expand_selection (blocking), add_identity_node, split_node, annotate
   *
   * When creating compositional annotations (selection includes annotation nodes),
   * the child annotations have their parent_annotation_id set after creation.
   */
  const executeAnnotationStrategy = useCallback(
    async (analysis: SelectionAnalysis, name: string, childAnnotations: AnnotationSummary[]) => {
      const { strategy } = analysis;
      const executableSteps = strategy.executableSteps;
      const totalSteps = executableSteps.length;

      setWizard({ step: "applying", currentFix: 0, totalFixes: totalSteps });

      try {
        for (let i = 0; i < executableSteps.length; i++) {
          const step = executableSteps[i];
          const stepNum = i + 1;

          logInfo(`Executing step ${stepNum}/${totalSteps}: ${step.type}`, step);
          setWizard({ step: "applying", currentFix: stepNum, totalFixes: totalSteps });

          if (step.type === "add_identity_node") {
            const success = await handleAddOperation({
              type: "add_identity_node",
              params: {
                target_node: step.params.target_node,
                connections: step.params.connections,
                new_node_id: step.params.new_node_id,
              },
            });
            if (!success) {
              throw new Error(`Failed to add identity node ${step.params.new_node_id}`);
            }
          } else if (step.type === "split_node") {
            const success = await handleAddOperation({
              type: "split_node",
              params: { node_id: step.params.node_id },
            });
            if (!success) {
              throw new Error(`Failed to split node ${step.params.node_id}`);
            }
          } else if (step.type === "annotate") {
            setWizard({ step: "creating" });

            // For compositional annotations, do NOT include child annotation nodes
            // The parent annotation only contains "glue" nodes connecting child annotations
            // Child annotation nodes stay in their own annotations (no overlap allowed by backend)
            let finalSubgraphNodes = [...step.params.subgraph_nodes];
            let finalEntryNodes = [...step.params.entry_nodes];
            let finalExitNodes = [...step.params.exit_nodes];

            if (childAnnotations.length > 0) {
              logInfo("Creating compositional annotation with children", {
                childCount: childAnnotations.length,
                childIds: childAnnotations.map(a => a.id),
              });

              // Collect all nodes that belong to child annotations
              const childAnnotationNodes = new Set<string>();
              for (const child of childAnnotations) {
                for (const nodeId of child.subgraph_nodes) {
                  childAnnotationNodes.add(nodeId);
                }
              }

              // Filter child annotation nodes from subgraph_nodes only.
              // Entry/exit nodes CAN reference nodes inside child annotations —
              // they define the composition's interface to the outside world.
              finalSubgraphNodes = finalSubgraphNodes.filter(n => !childAnnotationNodes.has(n));

              // If all nodes were from child annotations, we have a problem
              // The parent needs at least its own entry/exit nodes
              if (finalSubgraphNodes.length === 0) {
                logWarn("Compositional annotation has no glue nodes - all nodes belong to children");
                // Use identity nodes as the glue if they were created
                // These are the connections between annotations
              }

              logInfo("Filtered child annotation nodes from parent", {
                childAnnotationNodes: childAnnotationNodes.size,
                remainingSubgraph: finalSubgraphNodes.length,
                remainingEntry: finalEntryNodes.length,
                remainingExit: finalExitNodes.length,
              });
            }

            logInfo("Creating annotation", {
              name,
              entryNodes: finalEntryNodes,
              exitNodes: finalExitNodes,
              subgraphNodes: finalSubgraphNodes,
              isCompositional: childAnnotations.length > 0,
            });

            const success = await handleAddOperation({
              type: "annotate",
              params: {
                name,
                entry_nodes: finalEntryNodes,
                exit_nodes: finalExitNodes,
                subgraph_nodes: finalSubgraphNodes,
                subgraph_connections: [], // Let backend compute
                // For compositional annotations, pass child annotation IDs
                // Backend uses this to consider child nodes as "internal" during validation
                child_annotation_ids: childAnnotations.map(a => a.name),
              },
            });

            if (!success) {
              throw new Error("Failed to create annotation");
            }
          }
        }

        logInfo("Annotation strategy executed successfully");
        setAnnotationName("");
        setWizard({ step: "idle" });
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : "Failed to execute strategy";
        logError("Failed to execute annotation strategy", err);
        setWizard({ step: "error", message: errorMsg });
      }
    },
    [handleAddOperation]
  );

  // ==========================================================================
  // Smart Annotation Wizard
  // ==========================================================================

  /**
   * Start the smart annotation wizard.
   * Uses the current selectionContext which is already computed.
   */
  const handleStartAnnotationWizard = useCallback(() => {
    if (selectionContext.type !== "subgraph") {
      setError("Invalid selection for annotation");
      return;
    }
    if (!annotationName.trim()) {
      setError("Enter annotation name");
      return;
    }

    const { analysis } = selectionContext;

    if (!analysis.strategy.canExecute) {
      setError("Cannot create annotation: expand selection to include required nodes");
      return;
    }

    logInfo("Starting annotation wizard", {
      name: annotationName,
      selectedNodes: analysis.selectedNodes,
      canExecute: analysis.strategy.canExecute,
      isCompositional: selectedAnnotationNodes.length > 0,
    });

    // If no operations needed, create directly
    if (analysis.isValid) {
      logInfo("Selection is valid, creating annotation directly");
      executeAnnotationStrategy(analysis, annotationName, selectedAnnotationNodes);
    } else {
      // Show review step with strategy
      logInfo("Strategy needed, showing review step");
      setWizard({ step: "review", analysis, annotationName });
    }
  }, [selectionContext, annotationName, selectedAnnotationNodes, executeAnnotationStrategy]);

  /**
   * Apply strategy and create annotation (from review step).
   */
  const handleApplyStrategyAndCreate = useCallback(async () => {
    if (wizard.step !== "review") return;
    const { analysis, annotationName: name } = wizard;
    executeAnnotationStrategy(analysis, name, selectedAnnotationNodes);
  }, [wizard, selectedAnnotationNodes, executeAnnotationStrategy]);

  /**
   * Execute a single step from the strategy (e.g., just add identity node).
   */
  const handleExecuteSingleStep = useCallback(
    async (step: StrategyStep) => {
      logInfo("Executing single step", { type: step.type, step });

      try {
        setLoading(true);
        setError(null);

        if (step.type === "add_identity_node") {
          const success = await handleAddOperation({
            type: "add_identity_node",
            params: {
              target_node: step.params.target_node,
              connections: step.params.connections,
              new_node_id: step.params.new_node_id,
            },
          });
          if (!success) {
            throw new Error(`Failed to add identity node ${step.params.new_node_id}`);
          }
        } else if (step.type === "split_node") {
          const success = await handleAddOperation({
            type: "split_node",
            params: { node_id: step.params.node_id },
          });
          if (!success) {
            throw new Error(`Failed to split node ${step.params.node_id}`);
          }
        } else {
          throw new Error(`Cannot execute step type: ${step.type}`);
        }

        logInfo("Single step executed successfully");
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : "Failed to execute step";
        logError("Failed to execute single step", err);
        setError(errorMsg);
      } finally {
        setLoading(false);
      }
    },
    [handleAddOperation]
  );

  /**
   * Cancel the wizard.
   */
  const handleCancelWizard = useCallback(() => {
    logDebug("Cancelling annotation wizard");
    setWizard({ step: "idle" });
  }, []);

  // ==========================================================================
  // Render
  // ==========================================================================

  return (
    <aside className="operations-panel">
      <div
        className="panel-header"
        onClick={() => setCollapsed(!collapsed)}
        style={{ cursor: "pointer", display: "flex", alignItems: "center", gap: "6px" }}
      >
        <span className="collapse-toggle">{collapsed ? "\u25b6" : "\u25bc"}</span>
        <h3 style={{ margin: 0 }}>Operations</h3>
      </div>

      {collapsed ? null : <>
      {error && <div className="error-message">{error}</div>}

      {/* Annotation Rename Section */}
      {selectedAnnotationId && (() => {
        const ann = annotations.find(a => a.id === selectedAnnotationId);
        if (!ann) return null;
        const currentDisplayName = ann.display_name;
        const annRenameValid = annotationRenameInput.length > 0 && !annotationRenameInput.includes(" ");
        return (
          <section className="panel-section">
            <h4>Rename Annotation: {ann.display_name || ann.name || ann.id.slice(0, 8)}</h4>
            <div style={{ display: "flex", gap: "4px", alignItems: "center" }}>
              <input
                type="text"
                value={annotationRenameInput}
                onChange={(e) => setAnnotationRenameInput(e.target.value)}
                placeholder={currentDisplayName || ann.name || "Display name"}
                style={{
                  flex: 1,
                  padding: "4px 8px",
                  borderRadius: "4px",
                  border: "1px solid #555",
                  background: "#2a2a2a",
                  color: "#fff",
                  fontSize: "12px",
                }}
                onKeyDown={async (e) => {
                  if (e.key === "Enter" && annRenameValid) {
                    const ok = await handleAddOperation({
                      type: "rename_annotation",
                      params: { annotation_id: ann.name || ann.id, display_name: annotationRenameInput },
                    });
                    if (ok) { setAnnotationRenameInput(""); }
                  }
                }}
              />
              <button
                className="op-btn primary"
                disabled={loading || !annRenameValid}
                onClick={async () => {
                  const ok = await handleAddOperation({
                    type: "rename_annotation",
                    params: { annotation_id: ann.name || ann.id, display_name: annotationRenameInput },
                  });
                  if (ok) { setAnnotationRenameInput(""); }
                }}
                style={{ whiteSpace: "nowrap" }}
              >
                Rename
              </button>
            </div>
            {currentDisplayName && (
              <button
                className="op-btn"
                style={{ marginTop: "4px", fontSize: "11px" }}
                disabled={loading}
                onClick={async () => {
                  const ok = await handleAddOperation({
                    type: "rename_annotation",
                    params: { annotation_id: ann.name || ann.id, display_name: null },
                  });
                  if (ok) { setAnnotationRenameInput(""); }
                }}
              >
                Clear name ({currentDisplayName})
              </button>
            )}
            {annotationRenameInput.includes(" ") && (
              <p className="hint" style={{ color: "#f87171" }}>Name cannot contain spaces</p>
            )}
          </section>
        );
      })()}

      {/* Wizard Modal/Overlay */}
      {wizard.step !== "idle" && (
        <div className="wizard-overlay">
          <div className="wizard-content">
            {wizard.step === "analyzing" && (
              <div className="wizard-step">
                <h4>Analyzing Selection...</h4>
                <p className="hint">Checking subgraph structure and requirements</p>
              </div>
            )}

            {wizard.step === "review" && (
              <div className="wizard-step">
                <h4>Annotation: "{wizard.annotationName}"</h4>

                <div className="wizard-summary">
                  <p><strong>Entry:</strong> {wizard.analysis.entryNodes.join(", ") || "none"}</p>
                  <p><strong>Intermediate:</strong> {wizard.analysis.intermediateNodes.join(", ") || "none"}</p>
                  {wizard.analysis.discoveredNodes.length > 0 && (
                    <p className="discovered-nodes">
                      <strong>Discovered:</strong> {wizard.analysis.discoveredNodes.join(", ")}
                    </p>
                  )}
                  <p><strong>Exit:</strong> {wizard.analysis.exitNodes.join(", ") || "none"}</p>
                </div>

                {/* Strategy display - ordered steps */}
                <div className="strategy-section">
                  <h5>Annotation Strategy ({wizard.analysis.strategy.steps.length} steps)</h5>
                  <ol className="strategy-steps">
                    {wizard.analysis.strategy.steps.map((step, i) => (
                      <li
                        key={i}
                        className={`strategy-step ${
                          step.type === "expand_selection" ? "blocking" :
                          step.type === "annotate" ? "final" : "auto"
                        }`}
                      >
                        {step.type === "expand_selection" && (
                          <>
                            <span className="step-label">Expand selection:</span>
                            <span className="step-detail">{step.reason}</span>
                          </>
                        )}
                        {step.type === "add_identity_node" && (
                          <div className="step-with-action">
                            <div className="step-content">
                              <span className="step-label">Add identity node:</span>
                              <span className="step-detail">
                                {step.params.new_node_id} intercepts:
                              </span>
                              <ul className="connection-list">
                                {step.params.connections.map(([from, to], idx) => (
                                  <li key={idx} className="connection-item">
                                    {from} → {to}
                                  </li>
                                ))}
                              </ul>
                            </div>
                            <button
                              className="step-action-btn"
                              onClick={() => handleExecuteSingleStep(step)}
                              disabled={loading}
                              title="Execute just this step"
                            >
                              Do it
                            </button>
                          </div>
                        )}
                        {step.type === "split_node" && (
                          <div className="step-with-action">
                            <div className="step-content">
                              <span className="step-label">Split node:</span>
                              <span className="step-detail">{step.params.node_id}</span>
                            </div>
                            <button
                              className="step-action-btn"
                              onClick={() => handleExecuteSingleStep(step)}
                              disabled={loading}
                              title="Execute just this step"
                            >
                              Do it
                            </button>
                          </div>
                        )}
                        {step.type === "annotate" && (
                          <>
                            <span className="step-label">Create annotation:</span>
                            <span className="step-detail">
                              {step.params.subgraph_nodes.length} nodes ({step.params.entry_nodes.length} entry, {step.params.exit_nodes.length} exit)
                            </span>
                          </>
                        )}
                      </li>
                    ))}
                  </ol>
                </div>

                <div className="wizard-actions">
                  <button
                    className="op-btn primary"
                    onClick={handleApplyStrategyAndCreate}
                  >
                    Execute Strategy
                  </button>
                  <button className="op-btn secondary" onClick={handleCancelWizard}>
                    Cancel
                  </button>
                </div>
              </div>
            )}

            {wizard.step === "applying" && (
              <div className="wizard-step">
                <h4>Applying Changes...</h4>
                <p className="hint">
                  Step {wizard.currentFix} of {wizard.totalFixes}
                </p>
                <div className="progress-bar">
                  <div
                    className="progress-fill"
                    style={{ width: `${(wizard.currentFix / wizard.totalFixes) * 100}%` }}
                  />
                </div>
              </div>
            )}

            {wizard.step === "creating" && (
              <div className="wizard-step">
                <h4>Creating Annotation...</h4>
                <p className="hint">Finalizing the annotation</p>
              </div>
            )}

            {wizard.step === "error" && (
              <div className="wizard-step">
                <h4>Error</h4>
                <p className="error-message">{wizard.message}</p>
                <div className="wizard-actions">
                  <button className="op-btn secondary" onClick={handleCancelWizard}>
                    Close
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      <section className="panel-section">
        <h4>Selected Nodes ({selectedNodes.size})</h4>
        {selectedArray.length > 0 ? (
          <div className="selected-nodes">
            {selectedArray.slice(0, 10).join(", ")}
            {selectedArray.length > 10 && ` +${selectedArray.length - 10} more`}
          </div>
        ) : (
          <p className="hint">Click nodes in the graph to select</p>
        )}
      </section>

      {/* Context-aware operations based on selection */}
      {selectionContext.type === "single" && (() => {
        const nodeId = selectionContext.nodeId;
        const nodeObj = model.nodes.find((n) => n.id === nodeId);
        const currentDisplayName = nodeObj?.display_name ?? null;
        // Suggest feature name for input nodes based on position
        const inputIndex = model.metadata.input_nodes.indexOf(nodeId);
        const baseSuggestion =
          inputIndex >= 0 && featureNames && inputIndex < featureNames.length
            ? featureNames[inputIndex]
            : null;
        // For split input nodes (e.g. "-17_a"), look up the base node
        let suggestion = baseSuggestion;
        if (!suggestion && nodeId.includes("_")) {
          const baseId = nodeId.replace(/_[a-z]+$/, "");
          const baseIndex = model.metadata.input_nodes.indexOf(baseId);
          if (baseIndex >= 0 && featureNames && baseIndex < featureNames.length) {
            suggestion = featureNames[baseIndex];
          }
        }
        const renameValid = renameInput.length > 0 && !renameInput.includes(" ");

        // Look up feature metadata for input nodes
        const featureName = baseSuggestion ?? suggestion;
        const featureType = featureName && featureTypes ? featureTypes[featureName] ?? null : null;
        const featureDesc = featureName && featureDescriptions ? featureDescriptions[featureName] ?? null : null;
        const isInputNode = inputIndex >= 0 || (suggestion && nodeId.includes("_"));

        return (
          <section className="panel-section">
            {/* Feature info for input nodes */}
            {isInputNode && featureName && (
              <div className="feature-info-block">
                <h4>Feature Info</h4>
                <div style={{ fontSize: "12px", marginBottom: "8px" }}>
                  <div><strong>{featureName}</strong></div>
                  {featureType && <div style={{ color: "#9ca3af", fontStyle: "italic" }}>{featureType}</div>}
                  {featureDesc && <div style={{ color: "#9ca3af", marginTop: "2px" }}>{featureDesc}</div>}
                </div>
              </div>
            )}
            <h4>Node Operations</h4>
            <div className="button-group">
              <button
                className="op-btn"
                onClick={() => handleSplitNode(nodeId)}
                disabled={loading || !selectionContext.canSplit}
                title={selectionContext.canSplit ? "Split this node" : "Output nodes cannot be split"}
              >
                Split Node
              </button>
              <button
                className="op-btn"
                style={{ background: "#7f1d1d" }}
                onClick={async () => {
                  await handleAddOperation({
                    type: "prune_node",
                    params: { node_id: nodeId },
                  });
                }}
                disabled={loading || !selectionContext.canPrune}
                title={selectionContext.canPrune
                  ? "Bypass this node: connect its input directly to its output (non-identity)"
                  : `Requires exactly 1 input and 1 output (has ${selectionContext.enabledInputs} in, ${selectionContext.enabledOutputs} out)`}
              >
                Prune Node
              </button>
            </div>
            {!selectionContext.canSplit && (
              <p className="hint">Output nodes cannot be split</p>
            )}

            {/* Rename Node */}
            <h4 style={{ marginTop: "12px" }}>Rename Node</h4>
            <div style={{ display: "flex", gap: "4px", alignItems: "center" }}>
              <input
                type="text"
                value={renameInput}
                onChange={(e) => setRenameInput(e.target.value)}
                placeholder={currentDisplayName || suggestion || "camelCase"}
                style={{
                  flex: 1,
                  padding: "4px 8px",
                  borderRadius: "4px",
                  border: "1px solid #555",
                  background: "#2a2a2a",
                  color: "#fff",
                  fontSize: "12px",
                }}
                onKeyDown={async (e) => {
                  if (e.key === "Enter" && renameValid) {
                    const ok = await handleAddOperation({
                      type: "rename_node",
                      params: { node_id: nodeId, display_name: renameInput },
                    });
                    if (ok) { setRenameInput(""); }
                  }
                }}
              />
              <button
                className="op-btn primary"
                disabled={loading || !renameValid}
                onClick={async () => {
                  const ok = await handleAddOperation({
                    type: "rename_node",
                    params: { node_id: nodeId, display_name: renameInput },
                  });
                  if (ok) { setRenameInput(""); }
                }}
                style={{ whiteSpace: "nowrap" }}
              >
                Rename
              </button>
            </div>
            {suggestion && !currentDisplayName && (
              <button
                className="op-btn"
                style={{ marginTop: "4px", fontSize: "11px" }}
                disabled={loading}
                onClick={async () => {
                  const ok = await handleAddOperation({
                    type: "rename_node",
                    params: { node_id: nodeId, display_name: suggestion },
                  });
                  if (ok) { setRenameInput(""); }
                }}
              >
                Use: {suggestion}
              </button>
            )}
            {currentDisplayName && (
              <button
                className="op-btn"
                style={{ marginTop: "4px", fontSize: "11px" }}
                disabled={loading}
                onClick={async () => {
                  const ok = await handleAddOperation({
                    type: "rename_node",
                    params: { node_id: nodeId, display_name: null },
                  });
                  if (ok) { setRenameInput(""); }
                }}
              >
                Clear name ({currentDisplayName})
              </button>
            )}
            {renameInput.includes(" ") && (
              <p className="hint" style={{ color: "#f87171" }}>Name cannot contain spaces</p>
            )}
          </section>
        );
      })()}

      {selectionContext.type === "connectedPair" && (() => {
        const conns = selectionContext.actualConnections ?? [
          { from: selectionContext.fromNode, to: selectionContext.toNode, weight: selectionContext.weight, enabled: true }
        ];
        const allEnabled = conns.every(c => c.enabled);
        const allDisabled = conns.every(c => !c.enabled);

        return (
          <section className="panel-section">
            <h4>Connection Operations</h4>
            <div className="connection-info">
              <span className="connection-display">
                {selectionContext.fromNode} → {selectionContext.toNode}
              </span>
              <ul className="actual-connections">
                {conns.map((conn, idx) => (
                  <li key={idx}>
                    {conn.from} → {conn.to}{" "}
                    <span className="conn-weight">({conn.weight.toFixed(3)})</span>
                    {!conn.enabled && <span style={{ color: "#f87171", marginLeft: "4px" }}>(disabled)</span>}
                  </li>
                ))}
              </ul>
            </div>
            <div className="button-group">
              <button
                className="op-btn primary"
                onClick={() => handleAddIdentityOnConnection(
                  selectionContext.fromNode,
                  selectionContext.toNode,
                  selectionContext.actualConnections
                )}
                disabled={loading || allDisabled}
                title={allDisabled ? "Cannot add identity on disabled connection" : "Insert an identity node on this connection"}
              >
                Add Identity Node
              </button>
              {allEnabled && (
                <button
                  className="op-btn"
                  style={{ background: "#92400e" }}
                  onClick={async () => {
                    for (const conn of conns) {
                      await handleAddOperation({
                        type: "disable_connection",
                        params: { from_node: conn.from, to_node: conn.to },
                      });
                    }
                  }}
                  disabled={loading}
                  title="Disable this connection (non-identity, reversible)"
                >
                  Disable Connection
                </button>
              )}
              {allDisabled && (
                <button
                  className="op-btn"
                  style={{ background: "#065f46" }}
                  onClick={async () => {
                    for (const conn of conns) {
                      await handleAddOperation({
                        type: "enable_connection",
                        params: { from_node: conn.from, to_node: conn.to },
                      });
                    }
                  }}
                  disabled={loading}
                  title="Re-enable this connection"
                >
                  Enable Connection
                </button>
              )}
              <button
                className="op-btn"
                style={{ background: "#7f1d1d" }}
                onClick={async () => {
                  for (const conn of conns) {
                    if (confirm(`Prune connection ${conn.from} → ${conn.to}? This permanently removes it.`)) {
                      await handleAddOperation({
                        type: "prune_connection",
                        params: { from_node: conn.from, to_node: conn.to },
                      });
                    }
                  }
                }}
                disabled={loading}
                title="Permanently remove this connection (non-identity)"
              >
                Prune Connection
              </button>
            </div>
          </section>
        );
      })()}

      {selectionContext.type === "joinable" && (
        <section className="panel-section">
          <h4>Join Nodes</h4>
          <p className="hint">
            These nodes appear to be split variants of <strong>{selectionContext.baseNodeId}</strong>
          </p>
          <div className="button-group">
            <button
              className="op-btn primary"
              onClick={() => handleConsolidateNodes(selectionContext.nodeIds)}
              disabled={loading}
            >
              Join into {selectionContext.baseNodeId}
            </button>
          </div>
        </section>
      )}

      {selectionContext.type === "subgraph" && (
        <section className="panel-section">
          <h4>Subgraph Analysis</h4>
          <div className="subgraph-summary">
            <div className="subgraph-row">
              <span className="subgraph-label">Entry:</span>
              <span className="subgraph-nodes entry">
                {selectionContext.analysis.entryNodes.join(", ") || "none"}
              </span>
            </div>
            <div className="subgraph-row">
              <span className="subgraph-label">Intermediate:</span>
              <span className="subgraph-nodes intermediate">
                {selectionContext.analysis.intermediateNodes.length > 0
                  ? selectionContext.analysis.intermediateNodes.join(", ")
                  : "none"}
              </span>
            </div>
            {selectionContext.analysis.discoveredNodes.length > 0 && (
              <div className="subgraph-row discovered">
                <span className="subgraph-label">Discovered:</span>
                <span className="subgraph-nodes">
                  {selectionContext.analysis.discoveredNodes.join(", ")}
                </span>
              </div>
            )}
            <div className="subgraph-row">
              <span className="subgraph-label">Exit:</span>
              <span className="subgraph-nodes exit">
                {selectionContext.analysis.exitNodes.join(", ") || "none"}
              </span>
            </div>
          </div>

          {/* Strategy display - ordered steps */}
          {selectionContext.analysis.strategy.steps.length > 1 && (
            <div className="strategy-section">
              <h5>Annotation Strategy ({selectionContext.analysis.strategy.steps.length} steps)</h5>
              <ol className="strategy-steps">
                {selectionContext.analysis.strategy.steps.map((step, i) => {
                  const isBlocking = selectionContext.analysis.strategy.blockingSteps.includes(step);
                  return (
                  <li
                    key={i}
                    className={`strategy-step ${
                      isBlocking ? "blocking" :
                      step.type === "annotate" ? "final" : "auto"
                    }`}
                  >
                    {step.type === "expand_selection" && (
                      <>
                        <span className="step-label">Expand selection:</span>
                        <span className="step-detail">{step.reason}</span>
                      </>
                    )}
                    {step.type === "add_identity_node" && (
                      <div className="step-with-action">
                        <div className="step-content">
                          <span className="step-label">Add identity node:</span>
                          <span className="step-detail">
                            {step.params.new_node_id} intercepts:
                          </span>
                          <ul className="connection-list">
                            {step.params.connections.map(([from, to], idx) => (
                              <li key={idx} className="connection-item">
                                {from} → {to}
                              </li>
                            ))}
                          </ul>
                          <span className="step-note">
                            ({step.params.target_node} removed from annotation)
                          </span>
                        </div>
                        <button
                          className="step-action-btn"
                          onClick={() => handleExecuteSingleStep(step)}
                          disabled={loading}
                          title="Execute just this step"
                        >
                          Do it
                        </button>
                      </div>
                    )}
                    {step.type === "split_node" && (
                      <div className="step-with-action">
                        <div className="step-content">
                          <span className="step-label">Split node:</span>
                          <span className="step-detail">{step.params.node_id}</span>
                          {step.reason && (
                            <span className="step-reason">{step.reason}</span>
                          )}
                        </div>
                        <button
                          className="step-action-btn"
                          onClick={() => handleExecuteSingleStep(step)}
                          disabled={loading}
                          title="Execute just this step"
                        >
                          Do it
                        </button>
                      </div>
                    )}
                    {step.type === "annotate" && (
                      <>
                        <span className="step-label">Create annotation:</span>
                        <span className="step-detail">
                          {step.params.subgraph_nodes.length} nodes ({step.params.entry_nodes.length} entry, {step.params.exit_nodes.length} exit)
                        </span>
                      </>
                    )}
                  </li>
                  );
                })}
              </ol>
            </div>
          )}

          <div className="annotation-create">
            <input
              type="text"
              className="text-input"
              placeholder="Annotation name"
              value={annotationName}
              onChange={(e) => setAnnotationName(e.target.value)}
              disabled={wizard.step !== "idle"}
            />
            <button
              className="op-btn primary"
              onClick={handleStartAnnotationWizard}
              disabled={
                loading ||
                !annotationName.trim() ||
                wizard.step !== "idle" ||
                !selectionContext.analysis.strategy.canExecute
              }
              title={
                !selectionContext.analysis.strategy.canExecute
                  ? "Expand selection to fix blocking issues first"
                  : undefined
              }
            >
              {selectionContext.analysis.isValid
                ? "Create Annotation"
                : "Execute Strategy"}
            </button>
            {!selectionContext.analysis.strategy.canExecute && (
              <p className="hint warning-text">
                Expand selection to include nodes with external inputs before creating annotation
              </p>
            )}
          </div>
        </section>
      )}

      {selectionContext.type === "invalid" && selectedArray.length > 1 && (
        <section className="panel-section">
          <h4>Selection Analysis</h4>
          <p className="hint warning-text">{selectionContext.reason}</p>
          <p className="hint">
            Select nodes that form a connected subgraph with valid paths from entries to exits.
          </p>
        </section>
      )}

      {selectionContext.type === "empty" && (
        <section className="panel-section">
          <h4>Getting Started</h4>
          <p className="hint">
            Select nodes in the graph to see available operations.
            Click to select, Shift+click to add to selection.
          </p>
        </section>
      )}

      <section className="panel-section">
        <h4>Operation Notes</h4>
        <textarea
          value={operationNotes}
          onChange={(e) => setOperationNotes(e.target.value)}
          placeholder="Optional: Why are you making this change?"
          rows={2}
          style={{
            width: "100%",
            fontSize: "12px",
            padding: "6px",
            borderRadius: "4px",
            border: "1px solid #ccc",
            resize: "vertical",
            boxSizing: "border-box",
          }}
        />
        <p className="hint" style={{ fontSize: "11px", marginTop: "4px" }}>
          This note will be attached to the next operation you perform.
        </p>
      </section>

      {model.metadata.has_non_identity_ops && (
        <section className="panel-section" style={{ background: "#451a03", borderLeft: "3px solid #f59e0b", padding: "8px 12px" }}>
          <div style={{ fontSize: "12px", color: "#fbbf24" }}>
            Model has been modified (non-identity operations applied). The computed function has changed.
          </div>
        </section>
      )}

      <section className="panel-section">
        <h4>Operation History ({operations.length})</h4>
        {operations.length === 0 ? (
          <p className="hint">No operations yet</p>
        ) : (
          <ul className="operation-list">
            {operations.map((op) => {
              const isNonIdentity = ["prune_node", "prune_connection", "disable_connection", "enable_connection", "retrain"].includes(op.type);
              return (
              <li key={op.seq} className="operation-item" style={isNonIdentity ? { borderLeft: "2px solid #f59e0b" } : undefined}>
                <span className="op-seq">#{op.seq}</span>
                <span className="op-type" style={isNonIdentity ? { color: "#fbbf24" } : undefined}>{op.type}</span>
                <span className="op-params">{formatOperationParams(op)}</span>
                <button
                  className="op-undo"
                  onClick={() => handleRemoveOperation(op.seq)}
                  disabled={loading}
                  title="Undo this and all subsequent operations"
                >
                  &times;
                </button>
                {op.notes && (
                  <div className="op-notes" style={{ fontSize: "11px", color: "#666", fontStyle: "italic", marginLeft: "24px" }}>
                    {op.notes}
                  </div>
                )}
              </li>
              );
            })}
          </ul>
        )}
      </section>
      </>}
    </aside>
  );
}

function formatOperationParams(op: Operation): string {
  const params = op.params;
  switch (op.type) {
    case "split_node":
      return `node: ${params.node_id}`;
    case "consolidate_node":
      return `nodes: ${(params.node_ids as string[]).join(", ")}`;
    case "remove_node":
      return `node: ${params.node_id}`;
    case "add_node":
      return `connection: ${(params.connection as [string, string]).join("->")}`;
    case "add_identity_node":
      return `target: ${params.target_node}, id: ${params.new_node_id}`;
    case "annotate":
      return `"${params.name}"`;
    case "disable_connection":
      return `${params.from_node} -> ${params.to_node}`;
    case "enable_connection":
      return `${params.from_node} -> ${params.to_node}`;
    case "rename_node":
      return `${params.node_id} → ${params.display_name || "(clear)"}`;
    case "rename_annotation":
      return `${params.annotation_id} → ${params.display_name || "(clear)"}`;
    case "prune_node":
      return `node: ${params.node_id}`;
    case "prune_connection":
      return `${params.from_node} -> ${params.to_node}`;
    case "retrain":
      return `weights: ${Object.keys(params.weight_updates ?? {}).length}, biases: ${Object.keys(params.bias_updates ?? {}).length}`;
    default:
      return JSON.stringify(params).slice(0, 30);
  }
}
