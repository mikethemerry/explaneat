import { useState, useCallback, useMemo } from "react";
import {
  addOperation,
  removeOperation,
  detectSplits,
  autoClassify,
  type Operation,
  type OperationRequest,
  type SplitDetectionResponse,
  type ModelState,
  type ViolationDetail,
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
  onOperationChange: () => void;
};

/**
 * Analysis of a node selection for annotation creation.
 */
type SelectionAnalysis = {
  // Original selection
  selectedNodes: string[];

  // Computed subgraph
  entryNodes: string[];
  exitNodes: string[];
  intermediateNodes: string[];
  subgraphConnections: [string, string][];

  // External connections (issues to fix)
  externalOutputs: {
    connections: [string, string][];
    targetNodes: string[];
  };

  // Nodes that need splitting (have both external in and out)
  splitsNeeded: ViolationDetail[];

  // Is the selection valid for annotation as-is?
  isValid: boolean;

  // Proposed fixes
  fixes: AnnotationFix[];
};

type AnnotationFix = {
  type: "split_node" | "add_identity_node";
  description: string;
  params: Record<string, unknown>;
  // After this fix, these nodes will be added to the subgraph
  newNodes?: string[];
};

type WizardState =
  | { step: "idle" }
  | { step: "analyzing" }
  | { step: "review"; analysis: SelectionAnalysis; annotationName: string }
  | { step: "applying"; currentFix: number; totalFixes: number }
  | { step: "creating" }
  | { step: "error"; message: string };

// =============================================================================
// Subgraph Analysis Functions
// =============================================================================

/**
 * Analyze a node selection to determine the subgraph structure and any issues.
 */
function analyzeSelection(
  selectedNodes: Set<string>,
  model: ModelState
): SelectionAnalysis {
  logInfo("Analyzing selection for annotation", {
    selectedCount: selectedNodes.size,
    selected: Array.from(selectedNodes),
  });

  const selected = Array.from(selectedNodes);

  // Build connection maps
  const incomingConnections = new Map<string, [string, string][]>();
  const outgoingConnections = new Map<string, [string, string][]>();

  for (const nodeId of selected) {
    incomingConnections.set(nodeId, []);
    outgoingConnections.set(nodeId, []);
  }

  // Categorize all connections
  const subgraphConnections: [string, string][] = [];
  const externalInputConnections: [string, string][] = [];
  const externalOutputConnections: [string, string][] = [];

  for (const conn of model.connections) {
    if (!conn.enabled) continue;

    const fromInSelection = selectedNodes.has(conn.from);
    const toInSelection = selectedNodes.has(conn.to);

    if (fromInSelection && toInSelection) {
      // Internal connection
      subgraphConnections.push([conn.from, conn.to]);
      outgoingConnections.get(conn.from)?.push([conn.from, conn.to]);
      incomingConnections.get(conn.to)?.push([conn.from, conn.to]);
    } else if (!fromInSelection && toInSelection) {
      // External input
      externalInputConnections.push([conn.from, conn.to]);
    } else if (fromInSelection && !toInSelection) {
      // External output
      externalOutputConnections.push([conn.from, conn.to]);
    }
  }

  logDebug("Connection analysis", {
    internal: subgraphConnections.length,
    externalInputs: externalInputConnections.length,
    externalOutputs: externalOutputConnections.length,
  });

  // Classify nodes
  const entryNodes: string[] = [];
  const exitNodes: string[] = [];
  const intermediateNodes: string[] = [];

  // Track which nodes have external connections
  const nodesWithExternalInput = new Set(externalInputConnections.map(([, to]) => to));
  const nodesWithExternalOutput = new Set(externalOutputConnections.map(([from]) => from));

  // Also consider input/output node types
  const inputNodeIds = new Set(model.metadata.input_nodes);
  const outputNodeIds = new Set(model.metadata.output_nodes);

  for (const nodeId of selected) {
    const hasExternalInput = nodesWithExternalInput.has(nodeId) || inputNodeIds.has(nodeId);
    const hasExternalOutput = nodesWithExternalOutput.has(nodeId) || outputNodeIds.has(nodeId);

    if (hasExternalInput && !hasExternalOutput) {
      entryNodes.push(nodeId);
    } else if (hasExternalOutput && !hasExternalInput) {
      exitNodes.push(nodeId);
    } else if (hasExternalInput && hasExternalOutput) {
      // This node has both - it needs splitting or special handling
      // For now, classify as entry (the split will create an exit version)
      entryNodes.push(nodeId);
    } else {
      intermediateNodes.push(nodeId);
    }
  }

  logDebug("Node classification", {
    entry: entryNodes,
    exit: exitNodes,
    intermediate: intermediateNodes,
  });

  // Detect violations (nodes needing splits)
  const splitsNeeded: ViolationDetail[] = [];
  for (const nodeId of selected) {
    const hasExternalInput = nodesWithExternalInput.has(nodeId);
    const hasExternalOutput = nodesWithExternalOutput.has(nodeId);

    if (hasExternalInput && hasExternalOutput) {
      splitsNeeded.push({
        node_id: nodeId,
        reason: "has_external_input_and_output",
        external_inputs: externalInputConnections.filter(([, to]) => to === nodeId),
        external_outputs: externalOutputConnections.filter(([from]) => from === nodeId),
      });
    }
  }

  // Determine external output targets
  const externalOutputTargets = new Set(externalOutputConnections.map(([, to]) => to));

  // Build fixes list
  const fixes: AnnotationFix[] = [];

  // First, add splits for nodes that need them
  for (const violation of splitsNeeded) {
    fixes.push({
      type: "split_node",
      description: `Split node ${violation.node_id} (has both external inputs and outputs)`,
      params: { node_id: violation.node_id },
      newNodes: [`${violation.node_id}_split`], // The new exit version
    });
  }

  // Then, if there are external outputs (and no exit nodes), add identity node
  if (externalOutputConnections.length > 0 && exitNodes.length === 0) {
    // Group by target node
    for (const targetNode of externalOutputTargets) {
      const connectionsToTarget = externalOutputConnections.filter(
        ([, to]) => to === targetNode
      );
      const newNodeId = generateIdentityNodeId(model.nodes.map((n) => n.id), fixes);

      fixes.push({
        type: "add_identity_node",
        description: `Add identity node to intercept ${connectionsToTarget.length} connection(s) to ${targetNode}`,
        params: {
          target_node: targetNode,
          connections: connectionsToTarget,
          new_node_id: newNodeId,
        },
        newNodes: [newNodeId],
      });
    }
  }

  const isValid = splitsNeeded.length === 0 &&
    (exitNodes.length > 0 || externalOutputConnections.length === 0);

  const analysis: SelectionAnalysis = {
    selectedNodes: selected,
    entryNodes,
    exitNodes,
    intermediateNodes,
    subgraphConnections,
    externalOutputs: {
      connections: externalOutputConnections,
      targetNodes: Array.from(externalOutputTargets),
    },
    splitsNeeded,
    isValid,
    fixes,
  };

  logInfo("Selection analysis complete", {
    isValid,
    fixesNeeded: fixes.length,
    fixes: fixes.map((f) => f.description),
  });

  return analysis;
}

/**
 * Generate a unique identity node ID.
 */
function generateIdentityNodeId(existingNodes: string[], pendingFixes: AnnotationFix[]): string {
  const existingIds = new Set(existingNodes);

  // Also account for nodes that will be created by pending fixes
  for (const fix of pendingFixes) {
    if (fix.newNodes) {
      for (const newNode of fix.newNodes) {
        existingIds.add(newNode);
      }
    }
  }

  let counter = 1;
  while (existingIds.has(`identity_${counter}`)) {
    counter++;
  }
  return `identity_${counter}`;
}

/**
 * Compute the final subgraph nodes after all fixes are applied.
 */
function computeFinalSubgraph(analysis: SelectionAnalysis): string[] {
  const finalNodes = new Set(analysis.selectedNodes);

  // Add all new nodes that will be created by fixes
  for (const fix of analysis.fixes) {
    if (fix.newNodes) {
      for (const newNode of fix.newNodes) {
        finalNodes.add(newNode);
      }
    }
  }

  return Array.from(finalNodes);
}

// =============================================================================
// Component
// =============================================================================

export function OperationsPanel({
  genomeId,
  operations,
  selectedNodes,
  model,
  onOperationChange,
}: OperationsPanelProps) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [splitAnalysis, setSplitAnalysis] = useState<SplitDetectionResponse | null>(null);
  const [annotationName, setAnnotationName] = useState("");

  // Wizard state
  const [wizard, setWizard] = useState<WizardState>({ step: "idle" });

  const selectedArray = useMemo(() => Array.from(selectedNodes), [selectedNodes]);

  logDebug("Render", {
    selectedCount: selectedNodes.size,
    operationsCount: operations.length,
    wizardStep: wizard.step,
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
      logInfo("Removing operation", { seq });
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
    [genomeId, onOperationChange]
  );

  const handleSplitNode = useCallback(() => {
    if (selectedArray.length !== 1) {
      setError("Select exactly one node to split");
      return;
    }
    handleAddOperation({
      type: "split_node",
      params: { node_id: selectedArray[0] },
    }).then((success) => {
      if (success) onOperationChange();
    });
  }, [selectedArray, handleAddOperation, onOperationChange]);

  const handleAnalyzeSplits = useCallback(async () => {
    if (selectedArray.length === 0) {
      setError("Select nodes to analyze");
      return;
    }
    try {
      setLoading(true);
      setError(null);
      const result = await detectSplits(genomeId, selectedArray);
      logInfo("Split analysis result", result);
      setSplitAnalysis(result);
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Failed to analyze splits";
      logError("Failed to analyze splits", err);
      setError(errorMsg);
    } finally {
      setLoading(false);
    }
  }, [genomeId, selectedArray]);

  const handleApplySuggestedSplits = useCallback(async () => {
    if (!splitAnalysis?.suggested_operations.length) return;

    try {
      setLoading(true);
      setError(null);
      for (const op of splitAnalysis.suggested_operations) {
        await addOperation(genomeId, {
          type: op.type as "split_node",
          params: op.params,
        });
      }
      setSplitAnalysis(null);
      onOperationChange();
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Failed to apply splits";
      logError("Failed to apply splits", err);
      setError(errorMsg);
    } finally {
      setLoading(false);
    }
  }, [genomeId, splitAnalysis, onOperationChange]);

  // ==========================================================================
  // Smart Annotation Wizard
  // ==========================================================================

  /**
   * Start the smart annotation wizard.
   */
  const handleStartAnnotationWizard = useCallback(() => {
    if (selectedArray.length === 0) {
      setError("Select nodes for annotation");
      return;
    }
    if (!annotationName.trim()) {
      setError("Enter annotation name");
      return;
    }

    logInfo("Starting annotation wizard", {
      name: annotationName,
      selectedNodes: selectedArray,
    });

    setWizard({ step: "analyzing" });

    // Analyze the selection
    const analysis = analyzeSelection(selectedNodes, model);

    if (analysis.isValid && analysis.fixes.length === 0) {
      // No fixes needed, create annotation directly
      logInfo("Selection is valid, creating annotation directly");
      createAnnotationDirect(analysis, annotationName);
    } else {
      // Show review step with proposed fixes
      logInfo("Fixes needed, showing review step", { fixes: analysis.fixes });
      setWizard({ step: "review", analysis, annotationName });
    }
  }, [selectedArray, annotationName, selectedNodes, model]);

  /**
   * Create annotation directly (when no fixes needed).
   */
  const createAnnotationDirect = useCallback(
    async (analysis: SelectionAnalysis, name: string) => {
      setWizard({ step: "creating" });

      try {
        const success = await handleAddOperation({
          type: "annotate",
          params: {
            name,
            entry_nodes: analysis.entryNodes,
            exit_nodes: analysis.exitNodes,
            subgraph_nodes: analysis.selectedNodes,
            subgraph_connections: analysis.subgraphConnections,
          },
        });

        if (success) {
          logInfo("Annotation created successfully");
          setAnnotationName("");
          setWizard({ step: "idle" });
          onOperationChange();
        } else {
          setWizard({ step: "error", message: "Failed to create annotation" });
        }
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : "Failed to create annotation";
        logError("Failed to create annotation", err);
        setWizard({ step: "error", message: errorMsg });
      }
    },
    [handleAddOperation, onOperationChange]
  );

  /**
   * Apply fixes and create annotation.
   */
  const handleApplyFixesAndCreate = useCallback(async () => {
    if (wizard.step !== "review") return;

    const { analysis, annotationName: name } = wizard;
    const fixes = analysis.fixes;

    logInfo("Applying fixes and creating annotation", {
      fixCount: fixes.length,
      annotationName: name,
    });

    setWizard({ step: "applying", currentFix: 0, totalFixes: fixes.length });

    try {
      // Apply each fix
      for (let i = 0; i < fixes.length; i++) {
        const fix = fixes[i];
        logInfo(`Applying fix ${i + 1}/${fixes.length}`, fix);

        setWizard({ step: "applying", currentFix: i + 1, totalFixes: fixes.length });

        let operation: OperationRequest;
        if (fix.type === "split_node") {
          operation = {
            type: "split_node",
            params: { node_id: fix.params.node_id as string },
          };
        } else if (fix.type === "add_identity_node") {
          operation = {
            type: "add_identity_node",
            params: {
              target_node: fix.params.target_node as string,
              connections: fix.params.connections as [string, string][],
              new_node_id: fix.params.new_node_id as string,
            },
          };
        } else {
          throw new Error(`Unknown fix type: ${fix.type}`);
        }

        const success = await handleAddOperation(operation);
        if (!success) {
          throw new Error(`Failed to apply fix: ${fix.description}`);
        }
      }

      // Now create the annotation with the adjusted subgraph
      setWizard({ step: "creating" });

      // Compute final subgraph including new nodes
      const finalSubgraph = computeFinalSubgraph(analysis);

      // Re-classify nodes for the final annotation
      // For simplicity, entry nodes are original entries, exit nodes are new identity nodes
      const finalEntryNodes = analysis.entryNodes;
      const finalExitNodes = [...analysis.exitNodes];

      // Add new identity nodes as exit nodes
      for (const fix of fixes) {
        if (fix.type === "add_identity_node" && fix.newNodes) {
          finalExitNodes.push(...fix.newNodes);
        }
      }

      // Add split nodes to appropriate categories
      for (const fix of fixes) {
        if (fix.type === "split_node" && fix.newNodes) {
          // Split creates an exit version - add to exit nodes
          finalExitNodes.push(...fix.newNodes);
        }
      }

      logInfo("Creating annotation with adjusted subgraph", {
        name,
        entryNodes: finalEntryNodes,
        exitNodes: finalExitNodes,
        subgraphNodes: finalSubgraph,
      });

      const success = await handleAddOperation({
        type: "annotate",
        params: {
          name,
          entry_nodes: finalEntryNodes,
          exit_nodes: finalExitNodes,
          subgraph_nodes: finalSubgraph,
          subgraph_connections: [], // Let backend compute
        },
      });

      if (success) {
        logInfo("Annotation created successfully with fixes applied");
        setAnnotationName("");
        setWizard({ step: "idle" });
        onOperationChange();
      } else {
        setWizard({ step: "error", message: "Failed to create annotation after applying fixes" });
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Failed to apply fixes";
      logError("Failed in annotation wizard", err);
      setWizard({ step: "error", message: errorMsg });
    }
  }, [wizard, handleAddOperation, onOperationChange]);

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
                  <p>
                    <strong>Selected nodes:</strong> {wizard.analysis.selectedNodes.length}
                  </p>
                  <p>
                    <strong>Entry nodes:</strong> {wizard.analysis.entryNodes.join(", ") || "none"}
                  </p>
                  <p>
                    <strong>Exit nodes:</strong> {wizard.analysis.exitNodes.join(", ") || "none"}
                  </p>
                  <p>
                    <strong>Intermediate:</strong> {wizard.analysis.intermediateNodes.length}
                  </p>
                </div>

                {wizard.analysis.fixes.length > 0 ? (
                  <>
                    <div className="wizard-fixes">
                      <h5>Required Changes ({wizard.analysis.fixes.length})</h5>
                      <ul>
                        {wizard.analysis.fixes.map((fix, i) => (
                          <li key={i} className="fix-item">
                            <span className="fix-type">{fix.type}</span>
                            <span className="fix-desc">{fix.description}</span>
                          </li>
                        ))}
                      </ul>
                    </div>

                    <div className="wizard-final">
                      <p className="hint">
                        After these changes, the annotation will include:{" "}
                        <strong>{computeFinalSubgraph(wizard.analysis).length} nodes</strong>
                      </p>
                    </div>

                    <div className="wizard-actions">
                      <button
                        className="op-btn primary"
                        onClick={handleApplyFixesAndCreate}
                      >
                        Apply Changes & Create
                      </button>
                      <button className="op-btn secondary" onClick={handleCancelWizard}>
                        Cancel
                      </button>
                    </div>
                  </>
                ) : (
                  <>
                    <p className="success">Selection is valid for annotation!</p>
                    <div className="wizard-actions">
                      <button
                        className="op-btn primary"
                        onClick={() => createAnnotationDirect(wizard.analysis, wizard.annotationName)}
                      >
                        Create Annotation
                      </button>
                      <button className="op-btn secondary" onClick={handleCancelWizard}>
                        Cancel
                      </button>
                    </div>
                  </>
                )}
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

      <section className="panel-section">
        <h4>Node Operations</h4>
        <div className="button-group">
          <button
            className="op-btn"
            onClick={handleSplitNode}
            disabled={loading || selectedArray.length !== 1}
            title="Split selected node"
          >
            Split Node
          </button>
          <button
            className="op-btn"
            onClick={handleAnalyzeSplits}
            disabled={loading || selectedArray.length === 0}
            title="Check if selected nodes need splitting"
          >
            Analyze Splits
          </button>
        </div>

        {splitAnalysis && (
          <div className="split-analysis">
            <h5>Split Analysis</h5>
            {splitAnalysis.violations.length === 0 ? (
              <p className="success">No splits required!</p>
            ) : (
              <>
                <p className="warning">
                  {splitAnalysis.violations.length} node(s) need splitting:
                </p>
                <ul>
                  {splitAnalysis.violations.map((v) => (
                    <li key={v.node_id}>
                      <strong>{v.node_id}</strong>: {v.reason}
                    </li>
                  ))}
                </ul>
                <button
                  className="op-btn primary"
                  onClick={handleApplySuggestedSplits}
                  disabled={loading}
                >
                  Apply Suggested Splits
                </button>
              </>
            )}
            <button
              className="op-btn secondary"
              onClick={() => setSplitAnalysis(null)}
            >
              Dismiss
            </button>
          </div>
        )}
      </section>

      <section className="panel-section">
        <h4>Create Annotation</h4>
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
          disabled={loading || selectedArray.length === 0 || !annotationName.trim() || wizard.step !== "idle"}
        >
          Create Annotation
        </button>
        <p className="hint">
          Smart wizard: automatically detects and fixes issues (splits, identity nodes)
        </p>
      </section>

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
