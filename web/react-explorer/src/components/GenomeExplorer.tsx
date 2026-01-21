import { useCallback, useEffect, useState } from "react";
import {
  getCurrentModel,
  getGenomeExplanation,
  listOperations,
  type ModelState,
  type Operation,
} from "../api/client";
import { NetworkViewer } from "./NetworkViewer";
import { OperationsPanel } from "./OperationsPanel";

type GenomeExplorerProps = {
  genomeId: string;
  onBack: () => void;
};

export function GenomeExplorer({ genomeId, onBack }: GenomeExplorerProps) {
  const [model, setModel] = useState<ModelState | null>(null);
  const [operations, setOperations] = useState<Operation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedNodes, setSelectedNodes] = useState<Set<string>>(new Set());

  const loadData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const [modelData, opsData] = await Promise.all([
        getCurrentModel(genomeId),
        listOperations(genomeId),
      ]);

      setModel(modelData);
      setOperations(opsData.operations);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load genome");
    } finally {
      setLoading(false);
    }
  }, [genomeId]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const handleOperationChange = useCallback(() => {
    // Reload data after operation is added/removed
    loadData();
  }, [loadData]);

  const handleNodeSelect = useCallback((nodeIds: string[]) => {
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
        <h2>Genome Explorer</h2>
        <span className="genome-id">ID: {genomeId.slice(0, 8)}...</span>
      </header>

      <div className="explorer-content">
        <OperationsPanel
          genomeId={genomeId}
          operations={operations}
          selectedNodes={selectedNodes}
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
