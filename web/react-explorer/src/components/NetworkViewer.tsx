import { useCallback, useMemo, useEffect } from "react";
import {
  ReactFlow,
  Node,
  Edge,
  useNodesState,
  useEdgesState,
  Controls,
  MiniMap,
  Background,
  BackgroundVariant,
  MarkerType,
  type NodeChange,
  type EdgeChange,
  type OnSelectionChangeParams,
} from "@xyflow/react";
import Dagre from "@dagrejs/dagre";
import "@xyflow/react/dist/style.css";

import type { ModelState } from "../api/client";

// =============================================================================
// Logging utilities
// =============================================================================

const LOG_PREFIX = "[NetworkViewer]";

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
// Constants
// =============================================================================

const NODE_COLORS: Record<string, { background: string; border: string }> = {
  input: { background: "#4CAF50", border: "#388E3C" },
  output: { background: "#2196F3", border: "#1976D2" },
  hidden: { background: "#9E9E9E", border: "#757575" },
  identity: { background: "#FF9800", border: "#F57C00" },
};

const NODE_WIDTH = 120;
const NODE_HEIGHT = 40;

// =============================================================================
// Types
// =============================================================================

type NetworkViewerProps = {
  model: ModelState;
  selectedNodes: Set<string>;
  onNodeSelect: (nodeIds: string[]) => void;
};

// =============================================================================
// Dagre layout helper
// =============================================================================

function getLayoutedElements(
  nodes: Node[],
  edges: Edge[],
  direction: "TB" | "LR" = "LR"
): { nodes: Node[]; edges: Edge[] } {
  logDebug("Starting dagre layout", { nodeCount: nodes.length, edgeCount: edges.length, direction });

  const dagreGraph = new Dagre.graphlib.Graph().setDefaultEdgeLabel(() => ({}));
  const isHorizontal = direction === "LR";

  dagreGraph.setGraph({
    rankdir: direction,
    nodesep: 50,
    ranksep: 100,
    marginx: 20,
    marginy: 20,
  });

  // Add nodes to dagre
  nodes.forEach((node) => {
    dagreGraph.setNode(node.id, { width: NODE_WIDTH, height: NODE_HEIGHT });
  });

  // Add edges to dagre
  edges.forEach((edge) => {
    dagreGraph.setEdge(edge.source, edge.target);
  });

  // Run the layout algorithm
  try {
    Dagre.layout(dagreGraph);
    logDebug("Dagre layout completed successfully");
  } catch (error) {
    logError("Dagre layout failed", error);
    throw error;
  }

  // Map positions back to nodes
  const layoutedNodes = nodes.map((node) => {
    const nodeWithPosition = dagreGraph.node(node.id);
    if (!nodeWithPosition) {
      logWarn(`Node position not found for ${node.id}, using default`);
      return node;
    }

    const position = {
      x: nodeWithPosition.x - NODE_WIDTH / 2,
      y: nodeWithPosition.y - NODE_HEIGHT / 2,
    };

    logDebug(`Node ${node.id} positioned at`, position);

    return {
      ...node,
      position,
      targetPosition: isHorizontal ? "left" : "top",
      sourcePosition: isHorizontal ? "right" : "bottom",
    } as Node;
  });

  logInfo("Layout complete", {
    nodes: layoutedNodes.length,
    edges: edges.length,
  });

  return { nodes: layoutedNodes, edges };
}

// =============================================================================
// Helper functions
// =============================================================================

function buildNodeTooltip(node: ModelState["nodes"][number]): string {
  const lines = [`Node ${node.id}`, `Type: ${node.type}`];
  if (node.bias !== null && node.bias !== undefined) {
    lines.push(`Bias: ${node.bias.toFixed(3)}`);
  }
  if (node.activation) {
    lines.push(`Activation: ${node.activation}`);
  }
  if (node.response !== null && node.response !== undefined) {
    lines.push(`Response: ${node.response.toFixed(3)}`);
  }
  if (node.aggregation) {
    lines.push(`Aggregation: ${node.aggregation}`);
  }
  return lines.join("\n");
}

function convertModelToFlow(model: ModelState): { nodes: Node[]; edges: Edge[] } {
  logInfo("Converting model to ReactFlow format", {
    nodeCount: model.nodes.length,
    connectionCount: model.connections.length,
    inputNodes: model.metadata.input_nodes,
    outputNodes: model.metadata.output_nodes,
  });

  // Build nodes
  const nodes: Node[] = model.nodes.map((node, index) => {
    const colors = NODE_COLORS[node.type] || NODE_COLORS.hidden;
    const tooltip = buildNodeTooltip(node);

    logDebug(`Creating node ${node.id}`, {
      type: node.type,
      bias: node.bias,
      activation: node.activation,
    });

    return {
      id: node.id,
      type: "default",
      position: { x: index * 150, y: 0 }, // Will be overwritten by dagre
      data: {
        label: node.id,
        tooltip,
        nodeType: node.type,
      },
      style: {
        background: colors.background,
        border: `2px solid ${colors.border}`,
        borderRadius: "8px",
        color: "#fff",
        fontSize: "12px",
        fontWeight: "bold",
        padding: "8px 12px",
        minWidth: `${NODE_WIDTH}px`,
        textAlign: "center" as const,
      },
    };
  });

  // Build edges (only enabled connections)
  const enabledConnections = model.connections.filter((conn) => conn.enabled);
  logDebug(`Building edges from ${enabledConnections.length} enabled connections`);

  const edges: Edge[] = enabledConnections.map((conn) => {
    const isPositive = conn.weight >= 0;
    const strokeWidth = Math.min(1 + Math.abs(conn.weight) * 2, 6);

    logDebug(`Creating edge ${conn.from} -> ${conn.to}`, {
      weight: conn.weight,
      isPositive,
      strokeWidth,
    });

    return {
      id: `${conn.from}->${conn.to}`,
      source: conn.from,
      target: conn.to,
      type: "smoothstep",
      animated: false,
      style: {
        stroke: isPositive ? "#4CAF50" : "#F44336",
        strokeWidth,
      },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: isPositive ? "#4CAF50" : "#F44336",
        width: 15,
        height: 15,
      },
      label: conn.weight.toFixed(2),
      labelStyle: {
        fontSize: "10px",
        fontWeight: "bold",
        fill: "#333",
      },
      labelBgStyle: {
        fill: "#fff",
        fillOpacity: 0.8,
      },
    };
  });

  // Validate edges - check for missing source/target nodes
  const nodeIds = new Set(nodes.map((n) => n.id));
  edges.forEach((edge) => {
    if (!nodeIds.has(edge.source)) {
      logWarn(`Edge source node not found: ${edge.source}`);
    }
    if (!nodeIds.has(edge.target)) {
      logWarn(`Edge target node not found: ${edge.target}`);
    }
  });

  return { nodes, edges };
}

// =============================================================================
// Component
// =============================================================================

export function NetworkViewer({
  model,
  selectedNodes,
  onNodeSelect,
}: NetworkViewerProps) {
  // Convert model to ReactFlow format with layout
  const { initialNodes, initialEdges } = useMemo(() => {
    logInfo("Building network from model");
    const startTime = performance.now();

    const { nodes: flowNodes, edges: flowEdges } = convertModelToFlow(model);
    const { nodes: layoutedNodes, edges: layoutedEdges } = getLayoutedElements(
      flowNodes,
      flowEdges,
      "LR"
    );

    const elapsed = performance.now() - startTime;
    logInfo(`Network built in ${elapsed.toFixed(2)}ms`, {
      nodes: layoutedNodes.length,
      edges: layoutedEdges.length,
    });

    return { initialNodes: layoutedNodes, initialEdges: layoutedEdges };
  }, [model]);

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  // Update nodes when model changes
  useEffect(() => {
    logDebug("Model changed, updating nodes and edges");
    setNodes(initialNodes);
    setEdges(initialEdges);
  }, [initialNodes, initialEdges, setNodes, setEdges]);

  // Update node selection styling
  useEffect(() => {
    logDebug("Selection changed", { selectedCount: selectedNodes.size, selectedIds: Array.from(selectedNodes) });

    setNodes((nds) =>
      nds.map((node) => {
        const isSelected = selectedNodes.has(node.id);
        const colors = NODE_COLORS[node.data.nodeType as string] || NODE_COLORS.hidden;

        return {
          ...node,
          selected: isSelected,
          style: {
            ...node.style,
            border: isSelected
              ? `3px solid #FFD700`
              : `2px solid ${colors.border}`,
            boxShadow: isSelected ? "0 0 10px rgba(255, 215, 0, 0.5)" : "none",
          },
        };
      })
    );
  }, [selectedNodes, setNodes]);

  // Handle selection changes from ReactFlow
  const onSelectionChange = useCallback(
    (params: OnSelectionChangeParams) => {
      const selectedNodeIds = params.nodes.map((n) => n.id);
      logDebug("ReactFlow selection changed", { selectedNodeIds });
      onNodeSelect(selectedNodeIds);
    },
    [onNodeSelect]
  );

  // Handle node changes with logging
  const handleNodesChange = useCallback(
    (changes: NodeChange[]) => {
      logDebug("Node changes", changes);
      onNodesChange(changes);
    },
    [onNodesChange]
  );

  // Handle edge changes with logging
  const handleEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      logDebug("Edge changes", changes);
      onEdgesChange(changes);
    },
    [onEdgesChange]
  );

  // Handle node click
  const onNodeClick = useCallback(
    (_event: React.MouseEvent, node: Node) => {
      logInfo("Node clicked", { nodeId: node.id, nodeData: node.data });
    },
    []
  );

  // Handle edge click
  const onEdgeClick = useCallback(
    (_event: React.MouseEvent, edge: Edge) => {
      logInfo("Edge clicked", { edgeId: edge.id, source: edge.source, target: edge.target });
    },
    []
  );

  // MiniMap node color function
  const minimapNodeColor = useCallback((node: Node) => {
    const nodeType = node.data?.nodeType as string;
    return NODE_COLORS[nodeType]?.background || NODE_COLORS.hidden.background;
  }, []);

  logDebug("Rendering NetworkViewer", {
    nodeCount: nodes.length,
    edgeCount: edges.length,
    selectedCount: selectedNodes.size,
  });

  return (
    <div className="network-viewer">
      <div className="network-toolbar">
        <span>
          <strong>Nodes:</strong> {model.nodes.length}
        </span>
        <span>
          <strong>Connections:</strong>{" "}
          {model.connections.filter((c) => c.enabled).length}
        </span>
        <span>
          <strong>Selected:</strong> {selectedNodes.size}
        </span>
        {model.metadata.is_original ? (
          <span className="badge original">Original</span>
        ) : (
          <span className="badge modified">Modified</span>
        )}
      </div>
      <div className="network-container">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={handleNodesChange}
          onEdgesChange={handleEdgesChange}
          onSelectionChange={onSelectionChange}
          onNodeClick={onNodeClick}
          onEdgeClick={onEdgeClick}
          fitView
          fitViewOptions={{ padding: 0.2 }}
          minZoom={0.1}
          maxZoom={4}
          defaultEdgeOptions={{
            type: "smoothstep",
          }}
          selectionOnDrag={true}
          selectNodesOnDrag={true}
          panOnDrag={[1, 2]} // Middle and right mouse button for panning
          selectionMode="partial"
          colorMode="light"
        >
          <Controls showZoom showFitView showInteractive />
          <MiniMap
            nodeColor={minimapNodeColor}
            nodeStrokeWidth={3}
            zoomable
            pannable
          />
          <Background variant={BackgroundVariant.Dots} gap={20} size={1} />
        </ReactFlow>
      </div>
      <div className="network-legend">
        <span className="legend-item">
          <span
            className="legend-color"
            style={{ backgroundColor: NODE_COLORS.input.background }}
          />
          Input
        </span>
        <span className="legend-item">
          <span
            className="legend-color"
            style={{ backgroundColor: NODE_COLORS.hidden.background }}
          />
          Hidden
        </span>
        <span className="legend-item">
          <span
            className="legend-color"
            style={{ backgroundColor: NODE_COLORS.output.background }}
          />
          Output
        </span>
        <span className="legend-item">
          <span
            className="legend-color"
            style={{ backgroundColor: NODE_COLORS.identity.background }}
          />
          Identity
        </span>
        <span className="legend-item">
          <span
            className="legend-color"
            style={{ backgroundColor: "#4CAF50" }}
          />
          + Weight
        </span>
        <span className="legend-item">
          <span
            className="legend-color"
            style={{ backgroundColor: "#F44336" }}
          />
          - Weight
        </span>
      </div>
    </div>
  );
}
