import { useCallback, useEffect, useMemo, useState } from "react";
import {
  getCurrentModel,
  listOperations,
  listAnnotations,
  addOperation,
  type ModelState,
  type Operation,
  type AnnotationSummary,
} from "../api/client";
import { NetworkViewer } from "./NetworkViewer";
import { OperationsPanel } from "./OperationsPanel";
import { DatasetInfoPanel } from "./DatasetInfoPanel";
import { AnnotationListPanel } from "./AnnotationListPanel";
import { EvidencePanel } from "./EvidencePanel";
import { InputDistributionPanel } from "./InputDistributionPanel";

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
  experimentId: string;
  experimentName: string;
  onBack: () => void;
};

export function GenomeExplorer({ genomeId, experimentId, experimentName, onBack }: GenomeExplorerProps) {
  const [model, setModel] = useState<ModelState | null>(null);
  const [operations, setOperations] = useState<Operation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedNodes, setSelectedNodes] = useState<Set<string>>(new Set());

  // Annotation state
  const [annotations, setAnnotations] = useState<AnnotationSummary[]>([]);
  const [collapsedAnnotations, setCollapsedAnnotations] = useState<Set<string>>(new Set());
  const [selectedAnnotationId, setSelectedAnnotationId] = useState<string | null>(null);

  // Derive annotation node IDs from model (FUNCTION nodes returned by server)
  const annotationNodeIds = useMemo(
    () => model?.nodes.filter(n => n.type === "function").map(n => n.id) || [],
    [model]
  );

  // Derive selected input nodes for distribution panel
  const selectedInputNodes = useMemo(() => {
    if (!model) return [];
    const inputIds = new Set(model.metadata.input_nodes);
    return Array.from(selectedNodes).filter((nodeId) => {
      if (inputIds.has(nodeId)) return true;
      // Check split variants: "-20_a" -> base "-20"
      const baseId = nodeId.replace(/_[a-z]$/, "");
      return inputIds.has(baseId);
    });
  }, [model, selectedNodes]);

  // Detect if an output node is selected (for whole-model evidence)
  const selectedOutputNode = useMemo(() => {
    if (!model || selectedNodes.size !== 1) return false;
    const nodeId = Array.from(selectedNodes)[0];
    return model.metadata.output_nodes.includes(nodeId);
  }, [model, selectedNodes]);

  // Synthetic "whole model" annotation for evidence panel
  const wholeModelAnnotation = useMemo((): AnnotationSummary | null => {
    if (!model || !selectedOutputNode) return null;
    return {
      id: "__whole_model__",
      name: "Whole Model",
      display_name: "Whole Model",
      entry_nodes: model.metadata.input_nodes,
      exit_nodes: model.metadata.output_nodes,
      subgraph_nodes: model.nodes.map(n => n.id),
      parent_annotation_id: null,
      children_ids: [],
      is_leaf: true,
    };
  }, [model, selectedOutputNode]);

  logDebug("Render", { genomeId, experimentName, loading, hasModel: !!model, operationsCount: operations.length, annotationsCount: annotations.length });

  // Fetch model with current collapsed annotations
  const fetchModel = useCallback(async (collapsed: Set<string>) => {
    const collapsedArray = collapsed.size > 0 ? Array.from(collapsed) : undefined;
    logDebug("Fetching model", { collapsed: collapsedArray });
    const modelData = await getCurrentModel(genomeId, collapsedArray);
    setModel(modelData);

    // Reconcile selection: map old node IDs to new ones after operations
    const newNodeIds = new Set(modelData.nodes.map(n => n.id));
    setSelectedNodes(prev => {
      const reconciled = new Set<string>();
      for (const nodeId of prev) {
        if (newNodeIds.has(nodeId)) {
          reconciled.add(nodeId);
        } else {
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

    return modelData;
  }, [genomeId]);

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

      // Default all annotations to collapsed (by name, for server-side collapse)
      setCollapsedAnnotations(prev => {
        const next = new Set(prev);
        for (const ann of annotationsData.annotations) {
          if (ann.name && !next.has(ann.name)) {
            next.add(ann.name); // collapsed by default
          }
        }
        return next;
      });

      // Reconcile selection: map old node IDs to new ones after operations
      const newNodeIds = new Set(modelData.nodes.map(n => n.id));
      setSelectedNodes(prev => {
        const reconciled = new Set<string>();
        for (const nodeId of prev) {
          if (newNodeIds.has(nodeId)) {
            reconciled.add(nodeId);
          } else {
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

  // Re-fetch model when collapsed annotations change (after initial load)
  useEffect(() => {
    if (!loading && model) {
      logDebug("Collapsed annotations changed, re-fetching model", { collapsed: Array.from(collapsedAnnotations) });
      fetchModel(collapsedAnnotations);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [collapsedAnnotations, fetchModel]);

  const handleOperationChange = useCallback(() => {
    logInfo("Operation changed, reloading data and clearing selection");
    setSelectedNodes(new Set());
    loadData();
  }, [loadData]);

  const handleNodeSelect = useCallback((nodeIds: string[]) => {
    logDebug("Node selection changed", { nodeIds, count: nodeIds.length });
    setSelectedNodes(new Set(nodeIds));

    // Auto-select annotation when a single function node is clicked
    if (nodeIds.length === 1 && model) {
      const node = model.nodes.find(n => n.id === nodeIds[0]);
      if (node?.type === "function" && node.function_metadata?.annotation_id) {
        logInfo("Auto-selecting annotation from function node", {
          nodeId: node.id,
          annotationId: node.function_metadata.annotation_id,
        });
        setSelectedAnnotationId(node.function_metadata.annotation_id);
        return;
      }
    }
  }, [model]);

  const handleConnectionToggle = useCallback(async (fromNode: string, toNode: string, currentlyEnabled: boolean) => {
    try {
      const opType = currentlyEnabled ? "disable_connection" : "enable_connection";
      logInfo("Toggling connection", { fromNode, toNode, currentlyEnabled, opType });
      await addOperation(genomeId, {
        type: opType,
        params: { from_node: fromNode, to_node: toNode },
      });
      handleOperationChange();
    } catch (err) {
      logError("Failed to toggle connection", err);
    }
  }, [genomeId, handleOperationChange]);

  const handleToggleCollapse = useCallback((annotationId: string) => {
    // Find annotation name from ID (server uses names for collapse)
    const annotation = annotations.find(a => a.id === annotationId);
    if (!annotation?.name) {
      logDebug("Cannot toggle collapse: annotation has no name", { annotationId });
      return;
    }

    logDebug("Toggle annotation collapse", { annotationId, name: annotation.name });
    setCollapsedAnnotations(prev => {
      const next = new Set(prev);
      if (next.has(annotation.name!)) {
        next.delete(annotation.name!);
      } else {
        next.add(annotation.name!);
      }
      return next;
    });
  }, [annotations]);

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
          <DatasetInfoPanel
            experimentId={experimentId}
            genomeId={genomeId}
            model={model}
            onOperationChange={handleOperationChange}
          />
          <OperationsPanel
            genomeId={genomeId}
            experimentId={experimentId}
            operations={operations}
            selectedNodes={selectedNodes}
            model={model}
            annotations={annotations}
            onOperationChange={handleOperationChange}
          />
          <AnnotationListPanel
            annotations={annotations}
            collapsedAnnotations={collapsedAnnotations}
            onToggleCollapse={handleToggleCollapse}
            selectedAnnotationId={selectedAnnotationId}
            onSelectAnnotation={handleSelectAnnotation}
            genomeId={genomeId}
            onOperationChange={handleOperationChange}
          />
        </div>
        <NetworkViewer
          model={model}
          selectedNodes={selectedNodes}
          onNodeSelect={handleNodeSelect}
          annotationNodeIds={annotationNodeIds}
          onConnectionToggle={handleConnectionToggle}
        />
        {selectedAnnotationId ? (
          <div className="right-panel">
            <EvidencePanel
              genomeId={genomeId}
              experimentId={experimentId}
              annotation={
                annotations.find((a) => a.id === selectedAnnotationId)!
              }
            />
          </div>
        ) : wholeModelAnnotation ? (
          <div className="right-panel">
            <EvidencePanel
              genomeId={genomeId}
              experimentId={experimentId}
              annotation={wholeModelAnnotation}
              isWholeModel
            />
          </div>
        ) : selectedInputNodes.length >= 1 && selectedInputNodes.length <= 2 ? (
          <div className="right-panel">
            <InputDistributionPanel
              genomeId={genomeId}
              experimentId={experimentId}
              selectedInputNodes={selectedInputNodes}
              model={model}
            />
          </div>
        ) : null}
      </div>
    </div>
  );
}
