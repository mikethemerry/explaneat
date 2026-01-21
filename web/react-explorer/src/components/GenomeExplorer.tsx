import { useCallback, useEffect, useState } from "react";
import {
  getCurrentModel,
  listOperations,
  type ModelState,
  type Operation,
} from "../api/client";
import { NetworkViewer } from "./NetworkViewer";
import { OperationsPanel } from "./OperationsPanel";

// =============================================================================
// Logging utilities
// =============================================================================

const LOG_PREFIX = "[GenomeExplorer]";

function logDebug(message: string, data?: unknown) {
  console.log(`${LOG_PREFIX} ${message}`, data !== undefined ? data : "");
}

function logInfo(message: string, data?: unknown) {
  console.info(`${LOG_PREFIX} ${message}`, data !== undefined ? data : "");
}

function logError(message: string, data?: unknown) {
  console.error(`${LOG_PREFIX} ${message}`, data !== undefined ? data : "");
}

// =============================================================================
// Component
// =============================================================================

type GenomeExplorerProps = {
  genomeId: string;
  experimentName: string;
  onBack: () => void;
};

export function GenomeExplorer({ genomeId, experimentName, onBack }: GenomeExplorerProps) {
  const [model, setModel] = useState<ModelState | null>(null);
  const [operations, setOperations] = useState<Operation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedNodes, setSelectedNodes] = useState<Set<string>>(new Set());

  logDebug("Render", { genomeId, experimentName, loading, hasModel: !!model, operationsCount: operations.length });

  const loadData = useCallback(async () => {
    logInfo("Loading data for genome", { genomeId });
    const startTime = performance.now();

    try {
      setLoading(true);
      setError(null);

      logDebug("Fetching model and operations...");
      const [modelData, opsData] = await Promise.all([
        getCurrentModel(genomeId),
        listOperations(genomeId),
      ]);

      const elapsed = performance.now() - startTime;
      logInfo(`Data loaded in ${elapsed.toFixed(2)}ms`, {
        nodeCount: modelData.nodes.length,
        connectionCount: modelData.connections.length,
        operationsCount: opsData.operations.length,
        isOriginal: modelData.metadata.is_original,
      });

      logDebug("Model metadata", modelData.metadata);
      logDebug("Nodes", modelData.nodes);
      logDebug("Connections", modelData.connections);

      setModel(modelData);
      setOperations(opsData.operations);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Failed to load genome";
      logError("Failed to load data", { error: err, message: errorMessage });
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  }, [genomeId]);

  useEffect(() => {
    logDebug("Effect triggered: loading data");
    loadData();
  }, [loadData]);

  const handleOperationChange = useCallback(() => {
    logInfo("Operation changed, reloading data");
    loadData();
  }, [loadData]);

  const handleNodeSelect = useCallback((nodeIds: string[]) => {
    logDebug("Node selection changed", { nodeIds, count: nodeIds.length });
    setSelectedNodes(new Set(nodeIds));
  }, []);

  if (loading) {
    return (
      <div className="explorer-container">
        <div className="loading">Loading genome...</div>
      </div>
    );
  }

  if (error || !model) {
    return (
      <div className="explorer-container">
        <div className="error">
          <h3>Error loading genome</h3>
          <p>{error}</p>
          <button className="back-btn" onClick={onBack}>
            Back to list
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="explorer-container">
      <header className="explorer-header">
        <button className="back-btn" onClick={onBack}>
          &larr; Back
        </button>
        <h2>{experimentName}</h2>
        <span className="genome-id">Best Genome: {genomeId.slice(0, 8)}...</span>
      </header>

      <div className="explorer-content">
        <OperationsPanel
          genomeId={genomeId}
          operations={operations}
          selectedNodes={selectedNodes}
          model={model}
          onOperationChange={handleOperationChange}
        />
        <NetworkViewer
          model={model}
          selectedNodes={selectedNodes}
          onNodeSelect={handleNodeSelect}
        />
      </div>
    </div>
  );
}
