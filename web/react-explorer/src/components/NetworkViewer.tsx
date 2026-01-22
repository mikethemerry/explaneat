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
import "@xyflow/react/dist/style.css";

import type { ModelState, ApiConnection } from "../api/client";

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
  annotation: { background: "#7C3AED", border: "#5B21B6" }, // Purple for annotation nodes
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
  annotationNodeIds?: string[]; // IDs of synthetic annotation nodes
};

// =============================================================================
// Layer-based layout (max depth from inputs)
// =============================================================================

// Layout configuration
const LAYOUT_CONFIG = {
  layerSpacing: 250,    // Horizontal spacing between layers (columns)
  nodeSpacing: 80,      // Vertical spacing between nodes in the same layer
  marginX: 80,          // Left margin
  marginY: 80,          // Top margin
};

/**
 * Compute the maximum depth from any input node for each node in the graph.
 *
 * - Input nodes have depth 0
 * - Each other node's depth = max(depth of predecessors) + 1
 * - This ensures nodes are placed based on longest path from inputs
 *
 * Example: if a→b→c and a→c, then:
 *   - a: depth 0 (input)
 *   - b: depth 1
 *   - c: depth 2 (max of path through b)
 */
function computeNodeDepths(
  nodeIds: string[],
  inputNodeIds: Set<string>,
  connections: ApiConnection[]
): Map<string, number> {
  logDebug("Computing node depths from inputs", {
    totalNodes: nodeIds.length,
    inputNodes: inputNodeIds.size,
    connections: connections.length,
  });

  // Build adjacency list (forward edges: from → [to, to, ...])
  const forwardEdges = new Map<string, string[]>();
  const incomingEdges = new Map<string, string[]>();

  for (const nodeId of nodeIds) {
    forwardEdges.set(nodeId, []);
    incomingEdges.set(nodeId, []);
  }

  for (const conn of connections) {
    if (!conn.enabled) continue;
    forwardEdges.get(conn.from)?.push(conn.to);
    incomingEdges.get(conn.to)?.push(conn.from);
  }

  // Initialize depths: inputs are 0, others are -1 (unvisited)
  const depths = new Map<string, number>();
  for (const nodeId of nodeIds) {
    depths.set(nodeId, inputNodeIds.has(nodeId) ? 0 : -1);
  }

  // Use BFS/dynamic programming to compute max depths
  // We need to process nodes in topological order, but since we want MAX depth,
  // we need to ensure all predecessors are processed before a node
  // Use iterative relaxation until no changes
  let changed = true;
  let iterations = 0;
  const maxIterations = nodeIds.length + 1; // Safety limit

  while (changed && iterations < maxIterations) {
    changed = false;
    iterations++;

    for (const nodeId of nodeIds) {
      if (inputNodeIds.has(nodeId)) continue; // Inputs stay at 0

      const predecessors = incomingEdges.get(nodeId) || [];
      if (predecessors.length === 0) {
        // No incoming edges - might be disconnected, place at depth 0
        if (depths.get(nodeId) !== 0) {
          depths.set(nodeId, 0);
          changed = true;
        }
        continue;
      }

      // Find max depth among predecessors that have been visited
      let maxPredDepth = -1;
      let allPredecessorsVisited = true;

      for (const pred of predecessors) {
        const predDepth = depths.get(pred);
        if (predDepth === undefined || predDepth === -1) {
          allPredecessorsVisited = false;
        } else {
          maxPredDepth = Math.max(maxPredDepth, predDepth);
        }
      }

      if (maxPredDepth >= 0) {
        const newDepth = maxPredDepth + 1;
        const currentDepth = depths.get(nodeId) || -1;
        if (newDepth > currentDepth) {
          depths.set(nodeId, newDepth);
          changed = true;
        }
      }
    }
  }

  logDebug(`Depth computation completed in ${iterations} iterations`);

  // Handle any remaining unvisited nodes (disconnected from inputs)
  for (const nodeId of nodeIds) {
    if (depths.get(nodeId) === -1) {
      logWarn(`Node ${nodeId} not reachable from inputs, placing at depth 0`);
      depths.set(nodeId, 0);
    }
  }

  // Log depth distribution
  const depthCounts = new Map<number, number>();
  for (const [nodeId, depth] of depths) {
    depthCounts.set(depth, (depthCounts.get(depth) || 0) + 1);
    logDebug(`Node ${nodeId} -> depth ${depth}`);
  }

  const maxDepth = Math.max(...depths.values());
  logInfo("Depth distribution", {
    maxDepth,
    layerCounts: Object.fromEntries(depthCounts),
  });

  return depths;
}

/**
 * Apply layer-based layout to nodes based on their computed depths.
 * Nodes are arranged in columns (layers) with vertical centering within each layer.
 */
function applyLayerLayout(
  nodes: Node[],
  edges: Edge[],
  depths: Map<string, number>
): { nodes: Node[]; edges: Edge[] } {
  logDebug("Applying layer-based layout");

  // Group nodes by depth (layer)
  const layers = new Map<number, Node[]>();
  const maxDepth = Math.max(...depths.values());

  for (const node of nodes) {
    const depth = depths.get(node.id) ?? 0;
    if (!layers.has(depth)) {
      layers.set(depth, []);
    }
    layers.get(depth)!.push(node);
  }

  // Sort nodes within each layer for consistent ordering
  // Sort by node type first (input, hidden, output), then by id
  const typeOrder: Record<string, number> = { input: 0, hidden: 1, identity: 2, output: 3 };
  for (const [depth, layerNodes] of layers) {
    layerNodes.sort((a, b) => {
      const typeA = typeOrder[a.data.nodeType as string] ?? 1;
      const typeB = typeOrder[b.data.nodeType as string] ?? 1;
      if (typeA !== typeB) return typeA - typeB;
      return String(a.id).localeCompare(String(b.id), undefined, { numeric: true });
    });
  }

  // Calculate positions
  const layoutedNodes = nodes.map((node) => {
    const depth = depths.get(node.id) ?? 0;
    const layerNodes = layers.get(depth) || [];
    const indexInLayer = layerNodes.findIndex((n) => n.id === node.id);
    const layerSize = layerNodes.length;

    // X position based on depth (layer column)
    const x = LAYOUT_CONFIG.marginX + depth * LAYOUT_CONFIG.layerSpacing;

    // Y position: center the layer vertically
    const totalLayerHeight = (layerSize - 1) * LAYOUT_CONFIG.nodeSpacing;
    const layerStartY = LAYOUT_CONFIG.marginY + (maxDepth > 0 ? 100 : 0); // Add some top padding
    const y = layerStartY + indexInLayer * LAYOUT_CONFIG.nodeSpacing - totalLayerHeight / 2 + 200; // Center around y=200

    const position = { x, y };

    logDebug(`Node ${node.id}: layer=${depth}, index=${indexInLayer}/${layerSize}, pos=(${x}, ${y.toFixed(0)})`);

    return {
      ...node,
      position,
      sourcePosition: "right",
      targetPosition: "left",
    } as Node;
  });

  logInfo("Layer layout complete", {
    totalNodes: layoutedNodes.length,
    totalLayers: layers.size,
    maxDepth,
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

function convertModelToFlow(model: ModelState, annotationNodeIds: Set<string>): { nodes: Node[]; edges: Edge[] } {
  logInfo("Converting model to ReactFlow format", {
    nodeCount: model.nodes.length,
    connectionCount: model.connections.length,
    inputNodes: model.metadata.input_nodes,
    outputNodes: model.metadata.output_nodes,
    annotationNodes: annotationNodeIds.size,
  });

  // Build nodes
  const nodes: Node[] = model.nodes.map((node, index) => {
    const isAnnotationNode = annotationNodeIds.has(node.id);
    const effectiveType = isAnnotationNode ? "annotation" : node.type;
    const colors = NODE_COLORS[effectiveType] || NODE_COLORS.hidden;
    const tooltip = isAnnotationNode
      ? `Annotation: ${node.id.replace(/^A_/, "")}`
      : buildNodeTooltip(node);

    logDebug(`Creating node ${node.id}`, {
      type: node.type,
      isAnnotationNode,
      bias: node.bias,
      activation: node.activation,
    });

    // Special styling for annotation nodes
    const style = isAnnotationNode
      ? {
          background: `linear-gradient(135deg, ${colors.background}, #6D28D9)`,
          border: `2px dashed ${colors.border}`,
          borderRadius: "12px",
          color: "#fff",
          fontSize: "12px",
          fontWeight: "bold",
          padding: "10px 16px",
          minWidth: `${NODE_WIDTH + 20}px`,
          textAlign: "center" as const,
          boxShadow: "0 2px 8px rgba(124, 58, 237, 0.3)",
        }
      : {
          background: colors.background,
          border: `2px solid ${colors.border}`,
          borderRadius: "8px",
          color: "#fff",
          fontSize: "12px",
          fontWeight: "bold",
          padding: "8px 12px",
          minWidth: `${NODE_WIDTH}px`,
          textAlign: "center" as const,
        };

    return {
      id: node.id,
      type: "default",
      position: { x: index * 150, y: 0 }, // Will be overwritten by layout
      data: {
        label: isAnnotationNode ? node.id.replace(/^A_/, "") : node.id,
        tooltip,
        nodeType: effectiveType,
        isAnnotationNode,
      },
      style,
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
      type: "default",  // Bezier curves
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
  annotationNodeIds = [],
}: NetworkViewerProps) {
  // Convert model to ReactFlow format with layer-based layout
  const { initialNodes, initialEdges } = useMemo(() => {
    logInfo("Building network from model");
    const startTime = performance.now();

    // Convert model to ReactFlow nodes/edges
    const annotationNodeSet = new Set(annotationNodeIds);
    const { nodes: flowNodes, edges: flowEdges } = convertModelToFlow(model, annotationNodeSet);

    // Get input node IDs from metadata
    const inputNodeIds = new Set(model.metadata.input_nodes);
    logDebug("Input nodes for layout", Array.from(inputNodeIds));

    // Get all node IDs
    const nodeIds = model.nodes.map((n) => n.id);

    // Compute depth from inputs for each node
    const depths = computeNodeDepths(
      nodeIds,
      inputNodeIds,
      model.connections.filter((c) => c.enabled)
    );

    // Apply layer-based layout
    const { nodes: layoutedNodes, edges: layoutedEdges } = applyLayerLayout(
      flowNodes,
      flowEdges,
      depths
    );

    const elapsed = performance.now() - startTime;
    logInfo(`Network built in ${elapsed.toFixed(2)}ms`, {
      nodes: layoutedNodes.length,
      edges: layoutedEdges.length,
      maxDepth: Math.max(...depths.values()),
    });

    return { initialNodes: layoutedNodes, initialEdges: layoutedEdges };
  }, [model, annotationNodeIds]);

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
            type: "default",  // Bezier curves
          }}
          selectionOnDrag={true}
          selectNodesOnDrag={true}
          panOnDrag={[1, 2]} // Middle and right mouse button for panning
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
            style={{ backgroundColor: NODE_COLORS.annotation.background }}
          />
          Annotation
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
