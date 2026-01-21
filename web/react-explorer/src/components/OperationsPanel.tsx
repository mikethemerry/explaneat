import { useState, useCallback } from "react";
import {
  addOperation,
  removeOperation,
  detectSplits,
  autoClassify,
  type Operation,
  type OperationRequest,
  type SplitDetectionResponse,
} from "../api/client";

type OperationsPanelProps = {
  genomeId: string;
  operations: Operation[];
  selectedNodes: Set<string>;
  onOperationChange: () => void;
};

export function OperationsPanel({
  genomeId,
  operations,
  selectedNodes,
  onOperationChange,
}: OperationsPanelProps) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [splitAnalysis, setSplitAnalysis] = useState<SplitDetectionResponse | null>(null);
  const [annotationName, setAnnotationName] = useState("");

  const selectedArray = Array.from(selectedNodes);

  const handleAddOperation = useCallback(
    async (operation: OperationRequest) => {
      try {
        setLoading(true);
        setError(null);
        await addOperation(genomeId, operation);
        onOperationChange();
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to add operation");
      } finally {
        setLoading(false);
      }
    },
    [genomeId, onOperationChange],
  );

  const handleRemoveOperation = useCallback(
    async (seq: number) => {
      try {
        setLoading(true);
        setError(null);
        await removeOperation(genomeId, seq);
        onOperationChange();
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to remove operation");
      } finally {
        setLoading(false);
      }
    },
    [genomeId, onOperationChange],
  );

  const handleSplitNode = useCallback(() => {
    if (selectedArray.length !== 1) {
      setError("Select exactly one node to split");
      return;
    }
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
    try {
      setLoading(true);
      setError(null);
      const result = await detectSplits(genomeId, selectedArray);
      setSplitAnalysis(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to analyze splits");
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
      setError(err instanceof Error ? err.message : "Failed to apply splits");
    } finally {
      setLoading(false);
    }
  }, [genomeId, splitAnalysis, onOperationChange]);

  const handleCreateAnnotation = useCallback(async () => {
    if (selectedArray.length === 0) {
      setError("Select nodes for annotation");
      return;
    }
    if (!annotationName.trim()) {
      setError("Enter annotation name");
      return;
    }

    try {
      setLoading(true);
      setError(null);

      // Auto-classify to get entry/exit nodes
      const classification = await autoClassify(genomeId, selectedArray);

      if (!classification.valid) {
        setError(
          `Invalid coverage: ${classification.violations.map((v) => v.reason).join(", ")}`,
        );
        return;
      }

      // Build connections within the subgraph
      // For now, we'll let the backend compute these
      await handleAddOperation({
        type: "annotate",
        params: {
          name: annotationName,
          entry_nodes: classification.suggested_entry_nodes,
          exit_nodes: classification.suggested_exit_nodes,
          subgraph_nodes: selectedArray,
          subgraph_connections: [], // Backend will compute
        },
      });

      setAnnotationName("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create annotation");
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
                <span className="op-params">
                  {formatOperationParams(op)}
                </span>
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
      return `target: ${params.target_node}`;
    case "annotate":
      return `"${params.name}"`;
    default:
      return JSON.stringify(params).slice(0, 30);
  }
}
