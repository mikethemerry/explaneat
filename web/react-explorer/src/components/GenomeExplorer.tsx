import { useCallback, useEffect, useState } from "react";
import {
  getCurrentModel,
  listOperations,
  listAnnotations,
  type ModelState,
  type Operation,
  type AnnotationSummary,
} from "../api/client";
import { NetworkViewer } from "./NetworkViewer";
import { OperationsPanel } from "./OperationsPanel";
import { AnnotationListPanel } from "./AnnotationListPanel";
import { useCollapsedView, type CollapsedState } from "../hooks/useCollapsedView";

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

  // Annotation state
  const [annotations, setAnnotations] = useState<AnnotationSummary[]>([]);
  const [collapsedState, setCollapsedState] = useState<CollapsedState>(new Map());
  const [selectedAnnotationId, setSelectedAnnotationId] = useState<string | null>(null);

  // Compute collapsed view of the model
  const collapsedModel = useCollapsedView(model, annotations, collapsedState);

  logDebug("Render", { genomeId, experimentName, loading, hasModel: !!model, operationsCount: operations.length, annotationsCount: annotations.length });

  const loadData = useCallback(async () => {
    logInfo("Loading data for genome", { genomeId });
    const startTime = performance.now();

    try {
      setLoading(true);
      setError(null);

      logDebug("Fetching model, operations, and annotations...");
      const [modelData, opsData, annotationsData] = await Promise.all([
        getCurrentModel(genomeId),
        listOperations(genomeId),
        listAnnotations(genomeId),
      ]);

      const elapsed = performance.now() - startTime;
      logInfo(`Data loaded in ${elapsed.toFixed(2)}ms`, {
        nodeCount: modelData.nodes.length,
        connectionCount: modelData.connections.length,
        operationsCount: opsData.operations.length,
        annotationsCount: annotationsData.annotations.length,
        isOriginal: modelData.metadata.is_original,
      });

      logDebug("Model metadata", modelData.metadata);
      logDebug("Nodes", modelData.nodes);
      logDebug("Connections", modelData.connections);
      logDebug("Annotations", annotationsData.annotations);

      setModel(modelData);
      setOperations(opsData.operations);
      setAnnotations(annotationsData.annotations);

      // Reconcile selection: map old node IDs to new ones after operations
      const newNodeIds = new Set(modelData.nodes.map(n => n.id));
      setSelectedNodes(prev => {
        const reconciled = new Set<string>();
        for (const nodeId of prev) {
          if (newNodeIds.has(nodeId)) {
            // Node still exists, keep it selected
            reconciled.add(nodeId);
          } else {
            // Node was renamed/split - look for split variants
            const variants = Array.from(newNodeIds).filter(id =>
              id.startsWith(`${nodeId}_`)
            );
            if (variants.length > 0) {
              logDebug(`Reconciling selection: ${nodeId} -> ${variants.join(", ")}`);
              variants.forEach(v => reconciled.add(v));
            } else {
              logDebug(`Reconciling selection: ${nodeId} no longer exists, removing from selection`);
            }
          }
        }
        if (reconciled.size !== prev.size || ![...prev].every(id => reconciled.has(id))) {
          logInfo("Selection reconciled after model reload", {
            before: Array.from(prev),
            after: Array.from(reconciled),
          });
        }
        return reconciled;
      });
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
    logInfo("Operation changed, reloading data and clearing selection");
    setSelectedNodes(new Set());
    loadData();
  }, [loadData]);

  const handleNodeSelect = useCallback((nodeIds: string[]) => {
    logDebug("Node selection changed", { nodeIds, count: nodeIds.length });
    setSelectedNodes(new Set(nodeIds));
  }, []);

  const handleToggleCollapse = useCallback((annotationId: string) => {
    logDebug("Toggle annotation collapse", { annotationId });
    setCollapsedState(prev => {
      const newState = new Map(prev);
      const current = newState.get(annotationId) ?? false;
      newState.set(annotationId, !current);
      return newState;
    });
  }, []);

  const handleSelectAnnotation = useCallback((annotationId: string | null) => {
    logDebug("Select annotation", { annotationId });
    setSelectedAnnotationId(annotationId);
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
        <div className="left-panels">
          <OperationsPanel
            genomeId={genomeId}
            operations={operations}
            selectedNodes={selectedNodes}
            model={model}
            annotations={annotations}
            onOperationChange={handleOperationChange}
          />
          <AnnotationListPanel
            annotations={annotations}
            collapsedState={collapsedState}
            onToggleCollapse={handleToggleCollapse}
            selectedAnnotationId={selectedAnnotationId}
            onSelectAnnotation={handleSelectAnnotation}
          />
        </div>
        <NetworkViewer
          model={collapsedModel || model}
          selectedNodes={selectedNodes}
          onNodeSelect={handleNodeSelect}
          annotationNodeIds={collapsedModel?.annotationNodes || []}
        />
      </div>
    </div>
  );
}
