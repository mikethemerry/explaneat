import { useState, useCallback } from "react";
import {
  addOperation,
  removeOperation,
  detectSplits,
  autoClassify,
  type Operation,
  type OperationRequest,
  type SplitDetectionResponse,
  type ModelState,
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

type ExternalConnectionInfo = {
  hasExternalOutputs: boolean;
  externalOutputConnections: [string, string][]; // [from, to] where from is in selection, to is outside
  targetNodes: string[]; // unique target nodes outside selection
};

// =============================================================================
// Helper functions
// =============================================================================

/**
 * Analyze the selected nodes to find external output connections.
 * These are connections from selected nodes to nodes outside the selection.
 */
function analyzeExternalConnections(
  selectedNodes: Set<string>,
  model: ModelState
): ExternalConnectionInfo {
  const externalOutputConnections: [string, string][] = [];
  const targetNodes = new Set<string>();

  for (const conn of model.connections) {
    if (!conn.enabled) continue;

    // Connection from inside selection to outside selection
    if (selectedNodes.has(conn.from) && !selectedNodes.has(conn.to)) {
      externalOutputConnections.push([conn.from, conn.to]);
      targetNodes.add(conn.to);
    }
  }

  const result: ExternalConnectionInfo = {
    hasExternalOutputs: externalOutputConnections.length > 0,
    externalOutputConnections,
    targetNodes: Array.from(targetNodes),
  };

  logDebug("Analyzed external connections", result);
  return result;
}

/**
 * Generate a unique node ID for a new identity node.
 */
function generateIdentityNodeId(existingNodes: string[]): string {
  const existingIds = new Set(existingNodes);
  let counter = 1;
  while (existingIds.has(`identity_${counter}`)) {
    counter++;
  }
  return `identity_${counter}`;
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
  const [showIdentityHelper, setShowIdentityHelper] = useState(false);

  const selectedArray = Array.from(selectedNodes);

  // Analyze external connections for the current selection
  const externalInfo = analyzeExternalConnections(selectedNodes, model);

  logDebug("Render", {
    selectedCount: selectedNodes.size,
    operationsCount: operations.length,
    hasExternalOutputs: externalInfo.hasExternalOutputs,
  });

  const handleAddOperation = useCallback(
    async (operation: OperationRequest) => {
      logInfo("Adding operation", operation);
      try {
        setLoading(true);
        setError(null);
        const result = await addOperation(genomeId, operation);
        logInfo("Operation added successfully", result);
        onOperationChange();
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : "Failed to add operation";
        logError("Failed to add operation", { error: err, operation });
        setError(errorMsg);
      } finally {
        setLoading(false);
      }
    },
    [genomeId, onOperationChange]
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
    logInfo("Splitting node", { nodeId: selectedArray[0] });
    handleAddOperation({
      type: "split_node",
      params: { node_id: selectedArray[0] },
    });
  }, [selectedArray, handleAddOperation]);

  const handleAnalyzeSplits = useCallback(async () => {
    if (selectedArray.length === 0) {
      setError("Select nodes to analyze");
      return;
    }
    logInfo("Analyzing splits for nodes", selectedArray);
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

    logInfo("Applying suggested splits", splitAnalysis.suggested_operations);
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

  /**
   * Add an identity node to intercept external output connections.
   * This allows annotating a partial coverage (e.g., some inputs to an output).
   */
  const handleAddIdentityForAnnotation = useCallback(async () => {
    if (!externalInfo.hasExternalOutputs) {
      setError("No external outputs to intercept");
      return;
    }

    // Pick the first target node (usually there's only one output being targeted)
    const targetNode = externalInfo.targetNodes[0];
    const connectionsToIntercept = externalInfo.externalOutputConnections.filter(
      ([, to]) => to === targetNode
    );

    const newNodeId = generateIdentityNodeId(model.nodes.map((n) => n.id));

    logInfo("Adding identity node to intercept connections", {
      targetNode,
      connections: connectionsToIntercept,
      newNodeId,
    });

    try {
      setLoading(true);
      setError(null);

      await addOperation(genomeId, {
        type: "add_identity_node",
        params: {
          target_node: targetNode,
          connections: connectionsToIntercept,
          new_node_id: newNodeId,
        },
      });

      logInfo("Identity node added successfully", { newNodeId });
      setShowIdentityHelper(false);
      onOperationChange();
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Failed to add identity node";
      logError("Failed to add identity node", err);
      setError(errorMsg);
    } finally {
      setLoading(false);
    }
  }, [genomeId, model, externalInfo, onOperationChange]);

  const handleCreateAnnotation = useCallback(async () => {
    if (selectedArray.length === 0) {
      setError("Select nodes for annotation");
      return;
    }
    if (!annotationName.trim()) {
      setError("Enter annotation name");
      return;
    }

    logInfo("Creating annotation", {
      name: annotationName,
      selectedNodes: selectedArray,
    });

    try {
      setLoading(true);
      setError(null);

      // Auto-classify to get entry/exit nodes
      const classification = await autoClassify(genomeId, selectedArray);
      logDebug("Auto-classification result", classification);

      if (!classification.valid) {
        logWarn("Invalid coverage", classification.violations);
        setError(
          `Invalid coverage: ${classification.violations.map((v) => v.reason).join(", ")}`
        );
        return;
      }

      // Create the annotation
      await handleAddOperation({
        type: "annotate",
        params: {
          name: annotationName,
          entry_nodes: classification.suggested_entry_nodes,
          exit_nodes: classification.suggested_exit_nodes,
          subgraph_nodes: selectedArray,
          subgraph_connections: [],
        },
      });

      logInfo("Annotation created successfully", { name: annotationName });
      setAnnotationName("");
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Failed to create annotation";
      logError("Failed to create annotation", err);
      setError(errorMsg);
    } finally {
      setLoading(false);
    }
  }, [genomeId, selectedArray, annotationName, handleAddOperation]);

  return (
    <aside className="operations-panel">
      <h3>Operations</h3>

      {error && <div className="error-message">{error}</div>}

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

        {/* Show external output warning/helper */}
        {selectedArray.length > 0 && externalInfo.hasExternalOutputs && (
          <div className="external-outputs-info">
            <p className="hint">
              <strong>External outputs:</strong>{" "}
              {externalInfo.externalOutputConnections.length} connection(s) to{" "}
              {externalInfo.targetNodes.join(", ")}
            </p>
            {!showIdentityHelper ? (
              <button
                className="op-btn secondary"
                onClick={() => setShowIdentityHelper(true)}
              >
                Add Identity Node
              </button>
            ) : (
              <div className="identity-helper">
                <p className="hint">
                  Add an identity node to intercept{" "}
                  {externalInfo.externalOutputConnections.length} connection(s) going to{" "}
                  <strong>{externalInfo.targetNodes[0]}</strong>?
                </p>
                <p className="hint">
                  This will let you annotate the selected nodes as a concept feeding
                  into the output.
                </p>
                <div className="button-group">
                  <button
                    className="op-btn primary"
                    onClick={handleAddIdentityForAnnotation}
                    disabled={loading}
                  >
                    Add Identity Node
                  </button>
                  <button
                    className="op-btn secondary"
                    onClick={() => setShowIdentityHelper(false)}
                  >
                    Cancel
                  </button>
                </div>
              </div>
            )}
          </div>
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
        />
        <button
          className="op-btn primary"
          onClick={handleCreateAnnotation}
          disabled={loading || selectedArray.length === 0 || !annotationName.trim()}
        >
          Create Annotation
        </button>
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
