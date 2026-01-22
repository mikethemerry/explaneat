import { useState, useCallback, useMemo } from "react";
import {
  addOperation,
  removeOperation,
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
  operations: Operation[];
  selectedNodes: Set<string>;
  model: ModelState;
  annotations: AnnotationSummary[];
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
  | { type: "single"; nodeId: string; canSplit: boolean }
  | { type: "joinable"; nodeIds: string[]; baseNodeId: string }
  | { type: "subgraph"; analysis: SelectionAnalysis }
  | { type: "invalid"; reason: string };

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
  model: ModelState
): SelectionContext {
  const selectedArray = Array.from(selectedNodes);

  if (selectedArray.length === 0) {
    return { type: "empty" };
  }

  if (selectedArray.length === 1) {
    const nodeId = selectedArray[0];
    // Can split if it's a hidden or input node (not output)
    const node = model.nodes.find((n) => n.id === nodeId);
    const canSplit = node?.type === "hidden" || node?.type === "input";
    return { type: "single", nodeId, canSplit };
  }

  // Check if nodes are joinable (split variants)
  const baseNodeId = detectJoinableNodes(selectedArray);
  if (baseNodeId) {
    return { type: "joinable", nodeIds: selectedArray, baseNodeId };
  }

  // Check if nodes form a valid subgraph
  const analysis = analyzeSelection(selectedNodes, model);
  const { forward } = buildAdjacencyLists(model);
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
 */
function computeContainingSubgraph(
  selectedNodes: Set<string>,
  model: ModelState
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

  const { forward, backward } = buildAdjacencyLists(model);
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
    const isInputType = inputNodeIds.has(nodeId);

    const incomingFromInside = incoming.filter((from) => selectedNodes.has(from));
    const incomingFromOutside = incoming.filter((from) => !selectedNodes.has(from));
    const outgoingToInside = outgoing.filter((to) => selectedNodes.has(to));
    const outgoingToOutside = outgoing.filter((to) => !selectedNodes.has(to));

    // Entry: input node (no inputs) OR ALL incoming edges are from outside the subgraph
    const isEntry = isInputType || (incoming.length > 0 && incomingFromInside.length === 0);

    // Exit: ALL outgoing edges go to outside the subgraph (no internal outputs)
    const isExit = outgoing.length > 0 && outgoingToInside.length === 0;

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
  model: ModelState
): SelectionAnalysis {
  logInfo("Analyzing selection for annotation", {
    selectedCount: selectedNodes.size,
    selected: Array.from(selectedNodes),
  });

  // First, compute the containing subgraph (finds intermediate nodes)
  const containingSubgraph = computeContainingSubgraph(selectedNodes, model);

  // Use the complete subgraph for analysis
  const allSubgraphNodes = new Set(containingSubgraph.allSubgraphNodes);

  // Categorize connections relative to the complete subgraph
  const subgraphConnections: [string, string][] = [];
  const externalInputConnections: [string, string][] = [];
  const externalOutputConnections: [string, string][] = [];

  for (const conn of model.connections) {
    if (!conn.enabled) continue;

    const fromInSubgraph = allSubgraphNodes.has(conn.from);
    const toInSubgraph = allSubgraphNodes.has(conn.to);

    if (fromInSubgraph && toInSubgraph) {
      subgraphConnections.push([conn.from, conn.to]);
    } else if (!fromInSubgraph && toInSubgraph) {
      externalInputConnections.push([conn.from, conn.to]);
    } else if (fromInSubgraph && !toInSubgraph) {
      externalOutputConnections.push([conn.from, conn.to]);
    }
  }

  logDebug("Connection analysis (full subgraph)", {
    internal: subgraphConnections.length,
    externalInputs: externalInputConnections.length,
    externalOutputs: externalOutputConnections.length,
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
    const isInputType = inputNodeIds.has(nodeId);
    const isOutputType = outputNodeIds.has(nodeId);
    const hasInternalInput = nodesWithInternalInput.has(nodeId);
    const hasInternalOutput = nodesWithInternalOutput.has(nodeId);
    const hasExternalInput = nodesWithExternalInput.has(nodeId);
    const hasExternalOutput = nodesWithExternalOutput.has(nodeId);

    // Entry: input node OR has no internal inputs (all inputs come from outside)
    const isEntry = isInputType || (hasExternalInput && !hasInternalInput);

    // Exit: has no internal outputs (all outputs go outside)
    // Network output nodes are also exits if they don't have internal outputs
    const isExit = (hasExternalOutput || isOutputType) && !hasInternalOutput;

    logDebug(`Classifying node ${nodeId} in subgraph`, {
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
  // 3. AUTO: Split nodes (entry/intermediate with external outputs)
  // =========================================================================
  const nodesToSplitMap = new Map<string, [string, string][]>();

  for (const [from, to] of externalOutputs) {
    // External output is OK for exit nodes (that's their role)
    // But entry or intermediate nodes should NOT have external outputs
    if ((entrySet.has(from) || intermediateSet.has(from)) && !exitSet.has(from)) {
      if (!nodesToSplitMap.has(from)) {
        nodesToSplitMap.set(from, []);
      }
      nodesToSplitMap.get(from)!.push([from, to]);
    }
  }

  for (const [nodeId, extOutputs] of nodesToSplitMap) {
    splitNodes.push(nodeId);

    const step: StrategyStep = {
      type: "split_node",
      params: { node_id: nodeId },
      description: `Split ${nodeId} (has ${extOutputs.length} external output(s))`,
    };
    steps.push(step);
    executableSteps.push(step);
  }

  // =========================================================================
  // 4. Compute final annotation parameters
  // =========================================================================
  let finalSubgraphNodes = Array.from(allSubgraphNodes);

  // Remove exit node if replaced by identity
  if (exitNodeToReplace) {
    finalSubgraphNodes = finalSubgraphNodes.filter((n) => n !== exitNodeToReplace);
  }

  // Add new nodes from operations
  finalSubgraphNodes = [
    ...finalSubgraphNodes,
    ...identityNodes.map((n) => n.newNodeId),
    ...splitNodes.map((n) => `${n}_split`),
  ];

  // Compute final entry nodes (unchanged)
  const finalEntryNodes = [...entryNodes];

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

  // Add split nodes as exits
  for (const nodeId of splitNodes) {
    finalExitNodes.push(`${nodeId}_split`);
  }

  // =========================================================================
  // 5. Add final annotate step
  // =========================================================================
  const annotateStep: StrategyStep = {
    type: "annotate",
    params: {
      entry_nodes: finalEntryNodes,
      exit_nodes: finalExitNodes,
      subgraph_nodes: finalSubgraphNodes,
    },
    description: `Create annotation with ${finalSubgraphNodes.length} nodes`,
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
  operations,
  selectedNodes,
  model,
  annotations,
  onOperationChange,
}: OperationsPanelProps) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [annotationName, setAnnotationName] = useState("");

  // Wizard state (for applying fixes)
  const [wizard, setWizard] = useState<WizardState>({ step: "idle" });

  const selectedArray = useMemo(() => Array.from(selectedNodes), [selectedNodes]);

  // Detect if any selected nodes are annotation nodes
  const selectedAnnotationNodes = useMemo(() => {
    return getSelectedAnnotations(selectedNodes, annotations);
  }, [selectedNodes, annotations]);

  // Compute selection context dynamically
  const selectionContext = useMemo(() => {
    const context = analyzeSelectionContext(selectedNodes, model);
    logDebug("Selection context computed", {
      type: context.type,
      selectedCount: selectedNodes.size,
      selectedAnnotationNodes: selectedAnnotationNodes.length,
    });
    return context;
  }, [selectedNodes, model, selectedAnnotationNodes]);

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
        const result = await addOperation(genomeId, operation);
        logInfo("Operation added successfully", result);
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
    [genomeId]
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
      }).then((success) => {
        if (success) onOperationChange();
      });
    },
    [handleAddOperation, onOperationChange]
  );

  const handleConsolidateNodes = useCallback(
    (nodeIds: string[]) => {
      logInfo("Consolidating nodes", { nodeIds });
      handleAddOperation({
        type: "consolidate_node",
        params: { node_ids: nodeIds },
      }).then((success) => {
        if (success) onOperationChange();
      });
    },
    [handleAddOperation, onOperationChange]
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

            // Expand subgraph to include all nodes from child annotations
            let expandedSubgraphNodes = [...step.params.subgraph_nodes];
            let expandedEntryNodes = [...step.params.entry_nodes];
            let expandedExitNodes = [...step.params.exit_nodes];

            if (childAnnotations.length > 0) {
              logInfo("Creating compositional annotation with children", {
                childCount: childAnnotations.length,
                childIds: childAnnotations.map(a => a.id),
              });

              // Include all nodes from child annotations in the subgraph
              for (const child of childAnnotations) {
                for (const nodeId of child.subgraph_nodes) {
                  if (!expandedSubgraphNodes.includes(nodeId)) {
                    expandedSubgraphNodes.push(nodeId);
                  }
                }
                // Child entry nodes become part of parent entry (if not already internal)
                for (const nodeId of child.entry_nodes) {
                  if (!expandedEntryNodes.includes(nodeId)) {
                    expandedEntryNodes.push(nodeId);
                  }
                }
                // Child exit nodes become part of parent exit (if not already internal)
                for (const nodeId of child.exit_nodes) {
                  if (!expandedExitNodes.includes(nodeId)) {
                    expandedExitNodes.push(nodeId);
                  }
                }
              }
            }

            logInfo("Creating annotation", {
              name,
              entryNodes: expandedEntryNodes,
              exitNodes: expandedExitNodes,
              subgraphNodes: expandedSubgraphNodes,
              isCompositional: childAnnotations.length > 0,
            });

            const success = await handleAddOperation({
              type: "annotate",
              params: {
                name,
                entry_nodes: expandedEntryNodes,
                exit_nodes: expandedExitNodes,
                subgraph_nodes: expandedSubgraphNodes,
                subgraph_connections: [], // Let backend compute
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
        onOperationChange();
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : "Failed to execute strategy";
        logError("Failed to execute annotation strategy", err);
        setWizard({ step: "error", message: errorMsg });
      }
    },
    [handleAddOperation, onOperationChange]
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
      <h3>Operations</h3>

      {error && <div className="error-message">{error}</div>}

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
                          <>
                            <span className="step-label">Add identity node:</span>
                            <span className="step-detail">
                              {step.params.new_node_id} intercepts {step.params.connections.length} connection(s) to {step.params.target_node}
                            </span>
                          </>
                        )}
                        {step.type === "split_node" && (
                          <>
                            <span className="step-label">Split node:</span>
                            <span className="step-detail">{step.params.node_id}</span>
                          </>
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
      {selectionContext.type === "single" && (
        <section className="panel-section">
          <h4>Node Operations</h4>
          <div className="button-group">
            <button
              className="op-btn"
              onClick={() => handleSplitNode(selectionContext.nodeId)}
              disabled={loading || !selectionContext.canSplit}
              title={selectionContext.canSplit ? "Split this node" : "Output nodes cannot be split"}
            >
              Split Node
            </button>
          </div>
          {!selectionContext.canSplit && (
            <p className="hint">Output nodes cannot be split</p>
          )}
        </section>
      )}

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
                {selectionContext.analysis.strategy.steps.map((step, i) => (
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
                      <>
                        <span className="step-label">Add identity node:</span>
                        <span className="step-detail">
                          {step.params.new_node_id} intercepts {step.params.connections.length} connection(s) to {step.params.target_node}
                        </span>
                        <span className="step-note">
                          ({step.params.target_node} removed from annotation)
                        </span>
                      </>
                    )}
                    {step.type === "split_node" && (
                      <>
                        <span className="step-label">Split node:</span>
                        <span className="step-detail">{step.params.node_id}</span>
                      </>
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
        <h4>Operation History ({operations.length})</h4>
        {operations.length === 0 ? (
          <p className="hint">No operations yet</p>
        ) : (
          <ul className="operation-list">
            {operations.map((op) => (
              <li key={op.seq} className="operation-item">
                <span className="op-seq">#{op.seq}</span>
                <span className="op-type">{op.type}</span>
                <span className="op-params">{formatOperationParams(op)}</span>
                <button
                  className="op-undo"
                  onClick={() => handleRemoveOperation(op.seq)}
                  disabled={loading}
                  title="Undo this and all subsequent operations"
                >
                  &times;
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>
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
    default:
      return JSON.stringify(params).slice(0, 30);
  }
}
